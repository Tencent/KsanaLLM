/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/forwarding_context.h"

#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

void ForwardingBuffers::Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config, bool use_mtp,
                             BufferManager* buffer_mgr) {
  auto env = Singleton<Environment>::GetInstance();
  const size_t tensor_para_size = model_config.tensor_para_size;
  const size_t head_num = model_config.head_num;
  const size_t size_per_head = model_config.size_per_head;
  const size_t hidden_units = size_per_head * head_num;
  const size_t head_num_per_tp = head_num / env->GetAttentionTensorParallel();
  const size_t num_kv_heads_per_tp = model_config.num_key_value_heads / env->GetAttentionTensorParallel();
  const size_t max_token_num = model_config.max_step_token_num;  // max seq len for a step (batch)
  const size_t max_batch_size = model_config.max_batch_size;
  size_t vocab_size_pad = DivRoundUp(model_config.vocab_size, tensor_para_size) * tensor_para_size;

  BatchSchedulerConfig batch_scheduler_config;
  env->GetBatchSchedulerConfig(batch_scheduler_config);
  const size_t max_logits_tokens = max_batch_size * batch_scheduler_config.max_decode_tokens_per_req;

  size_t inter_size_per_tp = model_config.inter_size / tensor_para_size;
  if (model_config.has_shared_experts) {
    size_t shared = model_config.moe_config.shared_expert_inter_size;
    if (!model_config.enable_full_shared_expert) {
      // When enable_full_shared_expert is enabled, each GPU stores the complete shared experts without tensor
      // parallelism sharding across devices.
      shared /= tensor_para_size;
    }
    inter_size_per_tp = std::max(inter_size_per_tp, shared);
  }
  KLLM_LOG_DEBUG << fmt::format("inter_size_per_tp = {}", inter_size_per_tp);
  // inter_size_per_tp * 2 is used for the output of the fused gate_proj and up_proj in mlp
  size_t max_dim = std::max(std::max((head_num_per_tp + 2 * num_kv_heads_per_tp) * size_per_head, hidden_units),
                            inter_size_per_tp * 2);

  if (model_config.use_mla) {
    size_t qk_nope_head_dim = model_config.mla_config.qk_nope_head_dim;
    size_t qk_rope_head_dim = model_config.mla_config.qk_rope_head_dim;
    size_t v_head_dim = model_config.mla_config.v_head_dim;
    size_t kv_lora_rank = model_config.mla_config.kv_lora_rank;
    KLLM_LOG_DEBUG << fmt::format(
        "head_num_per_tp = {}, qk_nope_head_dim = {}, qk_rope_head_dim = {}, v_head_dim = {}, kv_lora_rank = {}",
        head_num_per_tp, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank);

    max_dim = std::max(std::max(max_dim, head_num_per_tp * v_head_dim), head_num_per_tp * qk_nope_head_dim);

    // For buffer reuse of MlaFlashAtten, see MlaAttenVarlen for details.
    size_t mla_flash_attn_size =
        std::max((qk_nope_head_dim * 3 + qk_rope_head_dim * 2), (v_head_dim + qk_nope_head_dim + qk_rope_head_dim * 3));
    max_dim = std::max(max_dim, head_num_per_tp * mla_flash_attn_size);

    // For buffer reuse of MlaPageAtten, see MlaPagedAttention for details.
    size_t mla_page_attn_size = kv_lora_rank * (head_num_per_tp * 2 + 1) + qk_rope_head_dim * (head_num_per_tp + 1);
    mla_page_attn_size = std::max(mla_page_attn_size, head_num_per_tp * mla_flash_attn_size);
    vocab_size_pad = std::max(vocab_size_pad, mla_page_attn_size);
  }

  KLLM_LOG_DEBUG << "max_batch_size=" << max_batch_size << ", vocab_size_pad=" << vocab_size_pad
                 << ", max_token_num=" << max_token_num << ", max_dim=" << max_dim << ", hidden_units=" << hidden_units
                 << ", inter_size_per_tp=" << inter_size_per_tp;
  const size_t hidden_buffer_size = std::max(max_logits_tokens * vocab_size_pad, max_token_num * max_dim);
  const size_t residual_buffer_size = max_token_num * hidden_units;
  // `shared_buffer_` is shared by `gated_buffer_`, `reduce_buffer_` and `paged_buffer_`.
  const size_t shared_buffer_size = max_token_num * std::max(inter_size_per_tp, hidden_units * 2);

  const DataType weight_type = model_config.weight_data_type;
  // TODO(karlluo): all create tensor used dynamic memory pool
  hidden_buffer_0 = buffer_mgr->CreateBufferTensor("hidden_buffer_0", {hidden_buffer_size}, weight_type);
  hidden_buffer_1 = buffer_mgr->CreateBufferTensor("hidden_buffer_1", {hidden_buffer_size}, weight_type);
  shared_buffer = buffer_mgr->CreateBufferTensor("shared_buffer", {shared_buffer_size}, weight_type);

  dp_input_buffer = buffer_mgr->CreateBufferTensor("dp_input_buffer", {hidden_buffer_size}, weight_type);

  const size_t max_seq_len = model_config.max_token_num;  // max seq len for one request
  // TODO(robertyuan): This buffer is too large
  // TODO(jinxcwu): Move all env to environment
  // 双重判断，避免非mla模型但设置了环境变量的情况
  if (IsAbsorbWeightsEnabled() && model_config.use_mla) {
    kv_cache_buffer = buffer_mgr->CreateBufferTensor("kv_cache_buffer", {0}, TYPE_FP32);
  } else {
    kv_cache_buffer = buffer_mgr->CreateBufferTensor(
        "kv_cache_buffer", {max_batch_size, (max_seq_len + 511) / 512, head_num_per_tp, size_per_head + 2}, TYPE_FP32);
  }

  if (use_mtp) {
    // mtp_hidden_buffer_tensors will used across main forward and nextn forward
    TensorBuffer* mtp_hidden_buffer =
        buffer_mgr->CreateBufferTensor("mtp_hidden_buffer_tensors", {residual_buffer_size}, weight_type);
    mtp_hidden_buffer_tensors = mtp_hidden_buffer->GetTensors();
  }
}

void ModelBuffers::Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config, bool use_mtp,
                        BufferManager* buffer_mgr) {
  buffers_ = std::make_unique<ForwardingBuffers>();
  buffers_->Init(context, rank, model_config, use_mtp, buffer_mgr);

  int head_num = model_config.head_num;
  int size_per_head = model_config.size_per_head;
  int hidden_units = size_per_head * head_num;
  size_t max_token_num = model_config.max_step_token_num;
  const size_t residual_buffer_size = max_token_num * hidden_units;
  const DataType weight_type = model_config.weight_data_type;
  // For distributed mode, the device buffer is used directly.
  if (context->IsStandalone()) {
    TensorBuffer* local_residual_buffer =
        buffer_mgr->CreateBufferTensor("local_residual_buffer_", {residual_buffer_size}, weight_type);
    local_residual_buffer_tensors_ = local_residual_buffer->GetTensors();
  }

  int rotary_embedding = model_config.rotary_embedding;
  int max_position_embeddings = model_config.max_position_embeddings;
  float scale_factor = model_config.rope_scaling_factor_config.factor;

  std::unordered_set<std::string> possible_rope_types = {"su", "longrope", "llama3"};
  if (possible_rope_types.find(model_config.rope_scaling_factor_config.type) == possible_rope_types.end() &&
      !model_config.rope_scaling_factor_config.has_alpha) {
    if (model_config.rope_scaling_factor_config.type == "yarn") {
      max_position_embeddings = model_config.rope_scaling_factor_config.original_max_position_embeddings;
      if (model_config.rope_scaling_factor_config.use_deepseek_yarn) {
        rotary_embedding = model_config.mla_config.qk_rope_head_dim;
      }
    }
    TensorBuffer* cos_sin_cache_buffer = buffer_mgr->CreateBufferTensor(
        "cos_sin_cache_tensor_",
        {static_cast<size_t>(rotary_embedding),
         static_cast<size_t>(max_position_embeddings) * static_cast<size_t>(scale_factor)},
        weight_type);
    cos_sin_cache_tensor_ = cos_sin_cache_buffer->GetTensors()[0];
  } else {
    TensorBuffer* cos_sin_cache_buffer = buffer_mgr->CreateBufferTensor(
        "cos_sin_cache_tensor_", {static_cast<size_t>(rotary_embedding), static_cast<size_t>(max_position_embeddings)},
        weight_type);
    cos_sin_cache_tensor_ = cos_sin_cache_buffer->GetTensors()[0];
  }
}

template <typename T>
void ForwardingContext<T>::Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config,
                                const PipelineConfig& pipeline_config, ForwardingBuffers* buffers,
                                BufferManager* buffer_mgr) {
  pipeline_config_ = pipeline_config;
  context_ = context;
  rank_ = rank;
  attn_data_parallel_size_ = model_config.attn_data_para_size;
  buffers_ = buffers;

  vocab_size_ = model_config.vocab_size;
  vocab_size_pad_ = DivRoundUp(model_config.vocab_size, model_config.tensor_para_size) * model_config.tensor_para_size;
  const DataType weight_type = model_config.weight_data_type;

  int head_num = model_config.head_num;
  int size_per_head = model_config.size_per_head;
  int hidden_units = size_per_head * head_num;
  int tensor_para_size = model_config.tensor_para_size;

  size_t max_token_num = model_config.max_step_token_num;
  size_t max_batch_size = model_config.max_batch_size;
  KLLM_LOG_DEBUG << fmt::format("Max Batch Size = {}, Max Seq Len = {}, Max Token Num = {}",
                                model_config.max_batch_size, model_config.max_token_num, max_token_num);

  // TODO(karlluo): we needn't tensor's shape to transfer attribute
  TensorBuffer* forward_shape_buffer = buffer_mgr->CreateBufferTensor("forward_shape", {1}, TYPE_INVALID);
  attn_ctx_.forward_shape = forward_shape_buffer->GetTensors()[0];

  model_input_ = std::make_shared<ModelInput>(model_config, rank_, context_);

  attn_ctx_.flag_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_BOOL, {1}, rank_);

  BatchSchedulerConfig batch_scheduler_config;
  Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
  const size_t max_logits_tokens = max_batch_size * batch_scheduler_config.max_decode_tokens_per_req;
  model_output_ = std::make_shared<ModelOutput>(max_logits_tokens, vocab_size_pad_, rank_, context_,
                                                max_token_num * hidden_units, weight_type);

  // Model communicator is only required when tp size is greater than 1.
  if (tensor_para_size > 1) {
    // Currently, custom all reduce is only enabled when `tp == 2`, so the
    // `buffers_.hidden_buffer_0_` will not be used.

    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, buffers_->hidden_buffer_0);
    CREATE_BUFFER_SCOPE(reduce_buffer_tensors, buffers_->shared_buffer);
    model_communicator_ = std::make_shared<ModelCommunicator<T>>(
        /* buffer */ &(hidden_buffer_tensors_0[0]),
        /* input */ &(reduce_buffer_tensors[0]), rank_, context_);
  } else {
    model_communicator_ = nullptr;
  }
}

template <typename T>
void ForwardingContext<T>::UpdateBeforeForward(std::vector<ForwardRequest>& forward_reqs, RunMode run_mode) {
  ProfileEvent::PushEvent("UpdateBeforeForward", rank_);
  model_input_->ParseFromRequests(forward_reqs, run_mode);


  // create forward shape tensor
  attn_ctx_.forward_shape.shape = {
      model_input_->multi_token_request_num,          // request num use flash_attention kernel
      model_input_->multi_token_request_max_tokens,   // max request tokens of request that use flash_attention kernel
      model_input_->flash_input.kv_cache_block_num,   // total kv cache block num that use flash_attention kernel
      model_input_->single_token_request_num,         // request num use page_attention kernel
      model_input_->single_token_request_max_tokens,  // max request tokens of request that use page_attention kernel
      model_input_->page_single_input.kv_cache_block_num +
          model_input_->page_dual_input.kv_cache_block_num,  // total kv cache block num that use page_attention kernel
      model_input_->dp_max_forwarding_tokens,                // used for blocked_prefill
      model_input_->total_prefix_len,
      model_input_->dp_multi_token_request_num,
      model_input_->dp_multi_token_request_max_tokens,
      model_input_->dp_single_token_request_num,
      model_input_->dp_single_token_request_max_tokens,
      model_input_->dp_total_prefix_len};
#ifdef ENABLE_ACL
  attn_ctx_.forward_shape.shape = {
      std::max(model_input_->multi_token_request_num, model_input_->single_token_request_num),
      std::max(model_input_->multi_token_request_max_tokens, model_input_->single_token_request_max_tokens),
      model_input_->page_single_input.kv_cache_block_num + model_input_->page_dual_input.kv_cache_block_num +
          model_input_->flash_input.kv_cache_block_num};
#endif
  // Pass the `use_cache` flag to `flag_tensor_`.
  ((Tensor)attn_ctx_.flag_tensor).GetPtr<bool>()[0] = model_input_->use_cache;
  ProfileEvent::PopEvent();
}

template <typename T>
void ForwardingContext<T>::UpdateAfterForward(std::vector<ForwardRequest>& forward_reqs) {
  // Cast to float & Copy to logits buffer
  attn_ctx_.forward_shape.shape = {forward_reqs[0].logits_offset * vocab_size_ * sizeof(float), vocab_size_,
                                   vocab_size_pad_};
}

template class ForwardingContext<float>;
template class ForwardingContext<float16>;
#ifdef ENABLE_BFLOAT16
template class ForwardingContext<bfloat16>;
#endif

}  // namespace ksana_llm
