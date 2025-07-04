/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/attention/multihead_latent_attention.h"

#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

template <typename T>
MultiHeadLatentAttention<T>::MultiHeadLatentAttention(int layer_idx, bool is_neox,
                                                      LayerCreationContext<T>& creation_context,
                                                      ModelCreationConfig& model_creation_config,
                                                      MlaBuffers& mla_buffers)
    : layer_idx_(layer_idx), mla_buffers_(mla_buffers) {
  auto& attn_config = model_creation_config.attn_config;
  rank_ = creation_context.rank;

  absorb_type_ = GetAbsorbWeightsType();

  if (absorb_type_ == AbsorbWeightsType::kAbsorbDisabled || absorb_type_ == AbsorbWeightsType::kAbsorbTypeBMM) {
    o_proj_k_dim_ = head_num_per_tp_ * attn_config.model_config.mla_config.v_head_dim;
  } else {
    KLLM_THROW("Unsupported absorb type");
    return;
  }

  use_q_lora_ = (attn_config.model_config.mla_config.q_lora_rank != 0);

  // attn_config.idx is the offset in kv_cache list.
  // e.g., master has normal layer 0-30 and nextn layer 61, the offset of layer_idx_61 is 31
  // e.g., master has normal layer 31-60, the offset of layer_idx_31 is 0
  attn_config.idx = layer_idx - creation_context.pipeline_config.lower_layer_idx;
  if (layer_idx >= model_creation_config.attn_config.model_config.num_layer) {
    attn_config.idx = creation_context.pipeline_config.upper_layer_idx -
                      creation_context.pipeline_config.lower_layer_idx + layer_idx -
                      model_creation_config.attn_config.model_config.num_layer + 1;
  }

  flash_mla_attention_layers_ =
      std::make_shared<FlashMlaAttention<T>>(layer_idx, is_neox, creation_context, attn_config);
  paged_mla_attention_layers_ =
      std::make_shared<PagedMlaAttention<T>>(layer_idx, is_neox, absorb_type_, creation_context, attn_config);

  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
  kv_a_layernorms_ =
      std::make_shared<Layernorm<T>>(layer_prefix + ".self_attn.kv_a_layernorm.weight",
                                     model_creation_config.layernorm_config.layernorm_eps, creation_context);
  q_a_layernorms_ =
      std::make_shared<Layernorm<T>>(layer_prefix + ".self_attn.q_a_layernorm.weight",
                                     model_creation_config.layernorm_config.layernorm_eps, creation_context);

  GroupQuantBackend linear_group_quant_backend = model_creation_config.attn_config.model_config.quant_config.backend;
  attn_q_a_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.q_a_proj.weight", creation_context,
                                                linear_group_quant_backend);
  attn_kv_a_lora_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.kv_a_lora_proj.weight",
                                                      creation_context, linear_group_quant_backend);
  attn_kv_a_ropes_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.kv_a_rope_proj.weight", creation_context,
                                                 linear_group_quant_backend);
  attn_q_b_lora_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.q_b_nope_proj.weight", creation_context,
                                                     linear_group_quant_backend);
  attn_q_b_rope_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.q_b_rope_proj.weight", creation_context,
                                                     linear_group_quant_backend);
  attn_v_head_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.v_head_proj.weight", creation_context,
                                                   linear_group_quant_backend);
  attn_w_q_uks_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.w_q_uk.weight", creation_context,
                                              linear_group_quant_backend);

  if (absorb_type_ == AbsorbWeightsType::kAbsorbTypeBMM) {
    attn_o_proj_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.o_proj.weight", creation_context,
                                               linear_group_quant_backend);
    attn_w_uk_t_bmm_ = std::make_shared<Bmm<T>>(layer_prefix + ".self_attn.w_uk_t.weight", creation_context,
                                                linear_group_quant_backend);
  } else if (absorb_type_ == AbsorbWeightsType::kAbsorbDisabled) {
    attn_o_proj_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.o_proj.weight", creation_context,
                                               linear_group_quant_backend);
  } else {
    KLLM_THROW(fmt::format("Unsupported absorb type {}", absorb_type_));
    return;
  }

#ifdef ENABLE_VLLM_FLASH_ATTN_2
  set_torch_stream_layers_ = std::make_shared<SetTorchStreamLayer<T>>();
  set_torch_stream_layers_->Init({}, creation_context.context, creation_context.rank);
#endif
}

template <typename T>
Status MultiHeadLatentAttention<T>::CreateBuffers(BufferManager* buffer_mgr, const AttentionCreationConfig& attn_config,
                                                  MlaBuffers& mla_buffers) {
  const DataType weight_type = attn_config.model_config.weight_data_type;
  const size_t max_token_num = attn_config.model_config.max_step_token_num;
  const size_t head_num = attn_config.model_config.head_num;
  const size_t max_decode_tokens = attn_config.model_config.max_batch_size * attn_config.max_decode_tokens_per_req;
  const uint32_t qk_rope_head_dim = attn_config.model_config.mla_config.qk_rope_head_dim;
  const uint32_t v_head_dim = attn_config.model_config.mla_config.v_head_dim;

  head_num_per_tp_ = head_num / Singleton<Environment>::GetInstance()->GetAttentionTensorParallel();
  qk_nope_head_dim_ = attn_config.model_config.mla_config.qk_nope_head_dim;
  kv_lora_rank_ = attn_config.model_config.mla_config.kv_lora_rank;
  // 权重吸收后要考虑decode的q维度有变化需要对比。
  const size_t q_buffer_size =
      max_token_num * head_num_per_tp_ * qk_nope_head_dim_ + max_decode_tokens * head_num_per_tp_ * kv_lora_rank_;
  mla_buffers.q_buffer = buffer_mgr->CreateBufferTensor("mla_buffers.q_buffer", {q_buffer_size}, weight_type);

  const size_t q_rope_buffer_size = max_token_num * head_num_per_tp_ * qk_rope_head_dim;
  mla_buffers.q_rope_buffer =
      buffer_mgr->CreateBufferTensor("mla_buffers.q_rope_buffer", {q_rope_buffer_size}, weight_type);

  const size_t kv_buffer_size = max_token_num * kv_lora_rank_;
  mla_buffers.kv_buffer = buffer_mgr->CreateBufferTensor("mla_buffers.kv_buffer", {kv_buffer_size}, weight_type);

  const size_t k_rope_buffer_size = max_token_num * qk_rope_head_dim;
  mla_buffers.k_rope_buffer =
      buffer_mgr->CreateBufferTensor("mla_buffers.k_rope_buffer", {k_rope_buffer_size}, weight_type);

  // TODO(jinxcwu) prefix部分与上面的buffer可能还有潜在的共享可能性
  size_t prefix_k_buffer_size = max_token_num * head_num_per_tp_ * (qk_nope_head_dim_ + qk_rope_head_dim);
  size_t prefix_v_buffer_size = prefix_k_buffer_size;
  size_t prefix_o_buffer_size = 0;
  size_t prefix_kv_buffer_size = max_token_num * head_num_per_tp_ * kv_lora_rank_;
  size_t prefix_k_up_buffer_size = max_token_num * head_num_per_tp_ * qk_nope_head_dim_;
  size_t prefix_v_up_buffer_size = max_token_num * head_num_per_tp_ * v_head_dim;

  // 不启用prefix cache时不创建prefix buffer节约显存
  if (!attn_config.model_config.enable_prefix_caching) {
    KLLM_LOG_INFO << "Not using prefix cache, so set all prefix buffer to 0 for saving vram";
    prefix_k_buffer_size = 0;
    prefix_v_buffer_size = 0;
    prefix_o_buffer_size = 0;
    prefix_kv_buffer_size = 0;
    prefix_k_up_buffer_size = 0;
    prefix_v_up_buffer_size = 0;
  }

  // 非共享buffer
  mla_buffers.prefix_o_buffer =
      buffer_mgr->CreateBufferTensor("mla_buffers.prefix_o_buffer", {prefix_o_buffer_size}, weight_type);
  mla_buffers.prefix_k_up_buffer =
      buffer_mgr->CreateBufferTensor("mla_buffers.prefix_k_up_buffer", {prefix_k_up_buffer_size}, weight_type);
  mla_buffers.prefix_v_up_buffer =
      buffer_mgr->CreateBufferTensor("mla_buffers.prefix_v_up_buffer", {prefix_v_up_buffer_size}, weight_type);

  // 共享buffer
  const size_t shared_prefix_k_and_prefix_v_with_prefix_kv_buffer_size =
      std::max(prefix_k_buffer_size + prefix_v_buffer_size, prefix_kv_buffer_size);
  KLLM_LOG_INFO << fmt::format("Sharing prefix_k_buffer[{}]+prefix_v_buffer[{}] with prefix_kv_buffer[{}]",
                               prefix_k_buffer_size, prefix_v_buffer_size, prefix_kv_buffer_size);
  mla_buffers.shared_prefix_k_v_kv_buffer =
      buffer_mgr->CreateBufferTensor("mla_buffers.shared_prefix_k_v_kv_buffer",
                                     {shared_prefix_k_and_prefix_v_with_prefix_kv_buffer_size}, weight_type);

  mla_buffers.prefix_k_buffer_size = prefix_k_buffer_size;
  mla_buffers.prefix_v_buffer_size = prefix_v_buffer_size;

  return Status();
}

template <typename T>
Status MultiHeadLatentAttention<T>::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                            std::vector<Tensor>& reduce_buffer_tensors,
                                            std::vector<Tensor>& paged_buffer_tensors,
                                            ForwardingContext<T>& forwarding_context) {
  reduce_buffer_tensors[0].shape = hidden_buffer_tensors_0[0].shape;

  const int rank = forwarding_context.GetCurrentRank();
  const int attn_dp_atp_size = Singleton<Environment>::GetInstance()->GetAttentionTensorParallel();
  const int attn_dp_group_id = rank / attn_dp_atp_size;
  const int attn_dp_rank_id = rank % attn_dp_atp_size;

  // if no request in this dp group, skip.
  if (forwarding_context.GetModelInput()->dp_multi_token_request_num +
          forwarding_context.GetModelInput()->dp_single_token_request_num ==
      0) {
    Singleton<Environment>::GetInstance()->SetDataParaGroupStatus(attn_dp_group_id, false);
    return Status();
  }

  Singleton<Environment>::GetInstance()->SetDataParaGroupStatus(attn_dp_group_id, true);

  const Tensor& input = hidden_buffer_tensors_0[0];
  const size_t seq_len = input.shape[0];
  const size_t hidden_units = input.shape[1];
  PROFILE_EVENT_SCOPE(CommonAttention_seq_len_,
                      fmt::format("CommonAttention_seq_len_{}_hidden_units_{}", seq_len, hidden_units), rank);
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  std::vector<Tensor> empty_tensors;
  set_torch_stream_layers_->Forward(empty_tensors, empty_tensors);
#endif

  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
  CREATE_BUFFER_SCOPE(kv_buffer_tensors, mla_buffers_.kv_buffer);
  CREATE_BUFFER_SCOPE(k_rope_buffer_tensors, mla_buffers_.k_rope_buffer);
  CREATE_BUFFER_SCOPE(q_buffer_tensors, mla_buffers_.q_buffer);
  CREATE_BUFFER_SCOPE(q_rope_buffer_tensors, mla_buffers_.q_rope_buffer);

  // kv_a_lora proj MatMul
  {
    PROFILE_EVENT_SCOPE(attn_kv_a_lora_proj, "attn_kv_a_lora_proj", rank);
    STATUS_CHECK_RETURN(attn_kv_a_lora_projs_->Forward(input, hidden_buffer_tensors_1));
  }

  {
    PROFILE_EVENT_SCOPE(kv_a_layernorm, "kv_a_layernorm", rank);
    kv_a_layernorms_->Forward(hidden_buffer_tensors_1, kv_buffer_tensors);
  }

  // kv_a_rope_lora proj MatMul
  {
    PROFILE_EVENT_SCOPE(kv_a_rope_proj, "kv_a_rope_proj", rank);
    STATUS_CHECK_RETURN(attn_kv_a_ropes_->Forward(input, k_rope_buffer_tensors));
  }
  Tensor prefill_hidden_buffer_1 = input;
  Tensor decode_hidden_buffer_1 = input;
  Tensor q_b_rope_input = input;

  // 降维度，q_lora_rank存在
  if (use_q_lora_) {
    // q_a proj MatMul
    {
      PROFILE_EVENT_SCOPE(q_a_proj, "q_a_proj", rank);
      STATUS_CHECK_RETURN(attn_q_a_projs_->Forward(input, q_buffer_tensors));
    }
    {
      PROFILE_EVENT_SCOPE(q_a_layernorm, "q_a_layernorm", rank);
      q_a_layernorms_->Forward(q_buffer_tensors, hidden_buffer_tensors_1);
    }
    prefill_hidden_buffer_1 = hidden_buffer_tensors_1[0];
    decode_hidden_buffer_1 = hidden_buffer_tensors_1[0];
    q_b_rope_input = hidden_buffer_tensors_1[0];
  }

  // q_b_rope proj MatMul
  {
    PROFILE_EVENT_SCOPE(q_b_rope_proj_weight, "q_b_rope_proj_weight", rank);
    STATUS_CHECK_RETURN(attn_q_b_rope_projs_->Forward(q_b_rope_input, q_rope_buffer_tensors));
  }

  const size_t decode_tokens = forwarding_context.GetModelInput()->page_single_input.total_dp_input_ids_len +
                               forwarding_context.GetModelInput()->page_dual_input.total_dp_input_ids_len;
  const size_t context_tokens = forwarding_context.GetModelInput()->flash_input.total_dp_input_ids_len;
  prefill_hidden_buffer_1.shape[0] = context_tokens;
  decode_hidden_buffer_1.shape[0] = decode_tokens;

  Tensor decode_hidden_buffer_1_tmp(decode_hidden_buffer_1.location, decode_hidden_buffer_1.dtype,
                                    std::vector<size_t>(decode_hidden_buffer_1.shape), decode_hidden_buffer_1.device_id,
                                    decode_hidden_buffer_1.GetPtr<void>() + prefill_hidden_buffer_1.GetTotalBytes());
  // 已经统一输入到prefill_hidden_buffer_1和decode_hidden_buffer_1

  /*
    对于mla，由于两个阶段的升维度操作不一致，需要提前进行拆分，
    将q_buffer拆分成 prefill_q_buffer 和 decode_q_buffer。
  */
  std::vector<Tensor> prefill_q_buffer_tensors{1}, decode_q_buffer_tensors{1};
  prefill_q_buffer_tensors[0] = q_buffer_tensors[0];
  prefill_q_buffer_tensors[0].shape = {context_tokens, head_num_per_tp_ * qk_nope_head_dim_};
  decode_q_buffer_tensors[0] = q_buffer_tensors[0];

  Tensor decode_q_buffer_tmp(decode_q_buffer_tensors[0].location, decode_q_buffer_tensors[0].dtype,
                             std::vector<size_t>(decode_q_buffer_tensors[0].shape),
                             decode_q_buffer_tensors[0].device_id,
                             decode_q_buffer_tensors[0].GetPtr<void>() + prefill_q_buffer_tensors[0].GetTotalBytes());
  std::vector<Tensor> decode_q_buffer_tensors_tmp = {decode_q_buffer_tmp};

  // For prefill
  if (context_tokens) {
    // q_b_lora proj MatMul
    PROFILE_EVENT_SCOPE(context_tokens, "context_tokens q_b_nope_proj", rank);
    STATUS_CHECK_RETURN(attn_q_b_lora_projs_->Forward(prefill_hidden_buffer_1, prefill_q_buffer_tensors));
  }

  // For decode
  if (decode_tokens) {
    if (absorb_type_ == AbsorbWeightsType::kAbsorbTypeBMM) {
      {
        PROFILE_EVENT_SCOPE(q_b_nope_proj, "decode_tokens q_b_nope_proj", rank);
        STATUS_CHECK_RETURN(attn_q_b_lora_projs_->Forward(decode_hidden_buffer_1_tmp, decode_q_buffer_tensors_tmp));
      }
      // transpose and w_uk_t bmm
      {
        PROFILE_EVENT_SCOPE(attn_w_uk_t_bmm, "decode_tokens attn_w_uk_t_bmm", rank);
        int decode_tokens_num = decode_q_buffer_tensors_tmp[0].shape[0];
        decode_q_buffer_tensors_tmp[0].shape = {decode_tokens_num, head_num_per_tp_, qk_nope_head_dim_};
        STATUS_CHECK_RETURN(attn_w_uk_t_bmm_->Forward(
            {decode_q_buffer_tensors_tmp[0], hidden_buffer_tensors_1[0], hidden_buffer_tensors_0[0]},
            decode_q_buffer_tensors_tmp));
      }

      hidden_buffer_tensors_0[0].shape[0] = seq_len;
      hidden_buffer_tensors_1[0].shape[0] = seq_len;
    } else {
      PROFILE_EVENT_SCOPE(decode_tokens, "decode_tokens q_b_nope_proj", rank);
      STATUS_CHECK_RETURN(attn_q_b_lora_projs_->Forward(decode_hidden_buffer_1_tmp, decode_q_buffer_tensors_tmp));
    }
  }

  // TODO(robertyuan): swap with reduce_buffer_tensors needs optimize.
  if (forwarding_context.GetModelCommunicator()) {
    std::swap(hidden_buffer_tensors_1, reduce_buffer_tensors);
  }

  if (context_tokens) {
    FlashAttentionForward(hidden_buffer_tensors_0, hidden_buffer_tensors_1, reduce_buffer_tensors,
                          hidden_buffer_tensors_0, prefill_q_buffer_tensors[0], q_rope_buffer_tensors[0],
                          kv_buffer_tensors[0], k_rope_buffer_tensors[0], forwarding_context);
  }

  if (decode_tokens != 0) {
    PagedAttentionForward(hidden_buffer_tensors_0, hidden_buffer_tensors_1, paged_buffer_tensors,
                          decode_q_buffer_tensors_tmp[0], q_rope_buffer_tensors[0], kv_buffer_tensors[0],
                          k_rope_buffer_tensors[0], forwarding_context);
  }

  {
    PROFILE_EVENT_SCOPE(o_prpj, "o_prpj", rank);
    Tensor o_input(hidden_buffer_tensors_0[0].location, hidden_buffer_tensors_0[0].dtype,
                   {context_tokens + decode_tokens, o_proj_k_dim_}, hidden_buffer_tensors_0[0].device_id,
                   hidden_buffer_tensors_0[0].GetPtr<void>());
    attn_o_proj_->Forward({o_input}, hidden_buffer_tensors_1);
  }

  if (forwarding_context.GetModelCommunicator()) {
    std::swap(hidden_buffer_tensors_1, reduce_buffer_tensors);
  }
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);

#ifdef ENABLE_VLLM_FLASH_ATTN_2
  set_torch_stream_layers_->Clear();
#endif

  return Status();
}

template <typename T>
Status MultiHeadLatentAttention<T>::DataParallelForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                        std::vector<Tensor>& reduce_buffer_tensors,
                                                        std::vector<Tensor>& extra_buffer_tensors,
                                                        ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);

  const size_t org_token_size = hidden_buffer_tensors_0[0].shape[0];
  const size_t token_hidden_stat_bytes = hidden_buffer_tensors_0[0].GetTotalBytes() / org_token_size;

  // Make corrent shape for no-request group, otherwise the all reduce will hang.
  reduce_buffer_tensors[0].shape = hidden_buffer_tensors_0[0].shape;

  // The tensors vector maybe swapped, so save it first.
  std::vector<Tensor> context_hidden_buffer_tensors_0 = hidden_buffer_tensors_0;
  std::vector<Tensor> context_hidden_buffer_tensors_1 = hidden_buffer_tensors_1;
  std::vector<Tensor> context_reduce_buffer_tensors = reduce_buffer_tensors;

  const size_t dp_group_id = forwarding_context.GetModelInput()->attn_dp_group_id_;
  const std::vector<int>& dp_token_offset = forwarding_context.GetModelInput()->attn_dp_group_offsets_;
  const int prefill_beg = dp_token_offset[dp_group_id * 4];
  const int prefill_end = dp_token_offset[dp_group_id * 4 + 1];
  const int decode_beg = dp_token_offset[dp_group_id * 4 + 2];
  const int decode_end = dp_token_offset[dp_group_id * 4 + 3];

  if (forwarding_context.GetModelInput()->dp_multi_token_request_num > 0) {
    const size_t tensor_offset = prefill_beg * token_hidden_stat_bytes;

    context_hidden_buffer_tensors_0[0].shape[0] = prefill_end - prefill_beg;
    context_hidden_buffer_tensors_0[0].offset = tensor_offset;

    context_reduce_buffer_tensors[0].shape = context_hidden_buffer_tensors_0[0].shape;
    context_reduce_buffer_tensors[0].offset = tensor_offset;

    context_hidden_buffer_tensors_1[0].shape = context_hidden_buffer_tensors_0[0].shape;
    context_hidden_buffer_tensors_1[0].offset = tensor_offset;

    ContextForward(context_hidden_buffer_tensors_0, context_hidden_buffer_tensors_1, context_reduce_buffer_tensors,
                   extra_buffer_tensors, forwarding_context);

    context_hidden_buffer_tensors_0[0].offset = 0;
    context_hidden_buffer_tensors_0[0].shape[0] = org_token_size;

    context_reduce_buffer_tensors[0].offset = 0;
    context_reduce_buffer_tensors[0].shape[0] = org_token_size;

    context_hidden_buffer_tensors_1[0].offset = 0;
    context_hidden_buffer_tensors_1[0].shape[0] = org_token_size;
  }

  if (forwarding_context.GetModelInput()->dp_single_token_request_num > 0) {
    // Fake a context part, so that the decode part could be work fine.
    const size_t tensor_offset = (decode_beg - (prefill_end - prefill_beg)) * token_hidden_stat_bytes;

    // The shape should include faked part.
    hidden_buffer_tensors_0[0].shape[0] = (prefill_end - prefill_beg) + (decode_end - decode_beg);
    hidden_buffer_tensors_0[0].offset = tensor_offset;

    reduce_buffer_tensors[0].shape = hidden_buffer_tensors_0[0].shape;
    reduce_buffer_tensors[0].offset = tensor_offset;

    hidden_buffer_tensors_1[0].shape = hidden_buffer_tensors_0[0].shape;
    hidden_buffer_tensors_1[0].offset = tensor_offset;

    DecodeForward(hidden_buffer_tensors_0, hidden_buffer_tensors_1, reduce_buffer_tensors, extra_buffer_tensors,
                  forwarding_context);
  } else {
    // If no decode token, apply context stage's change.
    if (forwarding_context.GetModelInput()->dp_multi_token_request_num > 0) {
      hidden_buffer_tensors_0.swap(context_hidden_buffer_tensors_0);
      hidden_buffer_tensors_1.swap(context_hidden_buffer_tensors_1);
      reduce_buffer_tensors.swap(context_reduce_buffer_tensors);
    } else {
      // No-request group, set to zero.
      MemsetAsync(hidden_buffer_tensors_0[0].GetPtr<void>(), 0, hidden_buffer_tensors_0[0].GetTotalBytes(),
                  forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
    }
  }

  // Always reset tensor offset, even if no decode invoked.
  hidden_buffer_tensors_0[0].offset = 0;
  hidden_buffer_tensors_0[0].shape[0] = org_token_size;

  reduce_buffer_tensors[0].offset = 0;
  reduce_buffer_tensors[0].shape[0] = org_token_size;

  hidden_buffer_tensors_1[0].offset = 0;
  hidden_buffer_tensors_1[0].shape[0] = org_token_size;

  // The reduce_buffer_tensors maybe write exceed current group's hidden state range,
  // Clear head and tailing memory.
  if (prefill_beg > 0) {
    MemsetAsync(reduce_buffer_tensors[0].GetPtr<void>(), 0, prefill_beg * token_hidden_stat_bytes,
                forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
  }

  if (decode_beg > prefill_end) {
    MemsetAsync(reduce_buffer_tensors[0].GetPtr<void>() + (prefill_end * token_hidden_stat_bytes), 0,
                (decode_beg - prefill_end) * token_hidden_stat_bytes,
                forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
  }

  if (decode_end < org_token_size) {
    MemsetAsync(reduce_buffer_tensors[0].GetPtr<void>() + (decode_end * token_hidden_stat_bytes), 0,
                (org_token_size - decode_end) * token_hidden_stat_bytes,
                forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
  }

  return Status();
}

template <typename T>
Status MultiHeadLatentAttention<T>::ContextForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                   std::vector<Tensor>& hidden_buffer_tensors_1,
                                                   std::vector<Tensor>& reduce_buffer_tensors,
                                                   std::vector<Tensor>& prefill_buffer_tensors,
                                                   ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(kv_buffer_tensors, mla_buffers_.kv_buffer);
  CREATE_BUFFER_SCOPE(k_rope_buffer_tensors, mla_buffers_.k_rope_buffer);
  CREATE_BUFFER_SCOPE(q_buffer_tensors, mla_buffers_.q_buffer);
  CREATE_BUFFER_SCOPE(q_rope_buffer_tensors, mla_buffers_.q_rope_buffer);

  const int rank = forwarding_context.GetCurrentRank();
  const Tensor& input = hidden_buffer_tensors_0[0];
  const size_t seq_len = input.shape[0];
  const size_t hidden_units = input.shape[1];
  PROFILE_EVENT_SCOPE(CommonAttention_seq_len_,
                      fmt::format("CommonAttention_seq_len_{}_hidden_units_{}", seq_len, hidden_units), rank);
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  std::vector<Tensor> empty_tensors;
  set_torch_stream_layers_->Forward(empty_tensors, empty_tensors);
#endif

  // kv_a_lora proj MatMul
  {
    PROFILE_EVENT_SCOPE(attn_kv_a_lora_proj, "attn_kv_a_lora_proj", rank);
    STATUS_CHECK_RETURN(attn_kv_a_lora_projs_->Forward(input, hidden_buffer_tensors_1));
  }

  {
    PROFILE_EVENT_SCOPE(kv_a_layernorm, "kv_a_layernorm", rank);
    kv_a_layernorms_->Forward(hidden_buffer_tensors_1, kv_buffer_tensors);
  }

  // kv_a_rope_lora proj MatMul
  {
    PROFILE_EVENT_SCOPE(kv_a_rope_proj, "kv_a_rope_proj", rank);
    STATUS_CHECK_RETURN(attn_kv_a_ropes_->Forward(input, k_rope_buffer_tensors));
  }
  Tensor prefill_hidden_buffer_1 = input;
  Tensor q_b_rope_input = input;

  // 降维度，q_lora_rank存在
  if (use_q_lora_) {
    // q_a proj MatMul
    {
      PROFILE_EVENT_SCOPE(q_a_proj, "q_a_proj", rank);
      STATUS_CHECK_RETURN(attn_q_a_projs_->Forward(input, q_buffer_tensors));
    }
    {
      PROFILE_EVENT_SCOPE(q_a_layernorm, "q_a_layernorm", rank);
      q_a_layernorms_->Forward(q_buffer_tensors, hidden_buffer_tensors_1);
    }
    prefill_hidden_buffer_1 = hidden_buffer_tensors_1[0];
    q_b_rope_input = hidden_buffer_tensors_1[0];
  }

  // q_b_rope proj MatMul
  {
    PROFILE_EVENT_SCOPE(q_b_rope_proj_weight, "q_b_rope_proj_weight", rank);
    STATUS_CHECK_RETURN(attn_q_b_rope_projs_->Forward(q_b_rope_input, q_rope_buffer_tensors));
  }

  const size_t context_tokens = forwarding_context.GetModelInput()->flash_input.total_dp_input_ids_len;
  KLLM_LOG_DEBUG << "context token " << context_tokens;

  prefill_hidden_buffer_1.shape[0] = context_tokens;

  /*
    对于mla，由于两个阶段的升维度操作不一致，需要提前进行拆分，
    将q_buffer拆分成 prefill_q_buffer 和 decode_q_buffer。
  */
  std::vector<Tensor> prefill_q_buffer_tensors{1};
  prefill_q_buffer_tensors[0] = q_buffer_tensors[0];
  prefill_q_buffer_tensors[0].shape = {context_tokens, head_num_per_tp_ * qk_nope_head_dim_};

  // For prefill
  if (context_tokens) {
    // q_b_lora proj MatMul
    {
      PROFILE_EVENT_SCOPE(context_tokens, "context_tokens q_b_nope_proj", rank);
      STATUS_CHECK_RETURN(attn_q_b_lora_projs_->Forward(prefill_hidden_buffer_1, prefill_q_buffer_tensors));
    }
    if (forwarding_context.GetModelCommunicator()) {
      std::swap(hidden_buffer_tensors_1, reduce_buffer_tensors);
    }
    FlashAttentionForward(hidden_buffer_tensors_0, hidden_buffer_tensors_1, reduce_buffer_tensors,
                          prefill_buffer_tensors, prefill_q_buffer_tensors[0], q_rope_buffer_tensors[0],
                          kv_buffer_tensors[0], k_rope_buffer_tensors[0], forwarding_context);

    {
      PROFILE_EVENT_SCOPE(o_prpj, "o_prpj", rank);
      Tensor o_input(prefill_buffer_tensors[0].location, prefill_buffer_tensors[0].dtype,
                     {context_tokens, o_proj_k_dim_}, prefill_buffer_tensors[0].device_id,
                     prefill_buffer_tensors[0].GetPtr<void>());
      Tensor o_output(hidden_buffer_tensors_1[0].location, hidden_buffer_tensors_1[0].dtype,
                      {context_tokens, hidden_buffer_tensors_1[0].shape[1]}, hidden_buffer_tensors_1[0].device_id,
                      hidden_buffer_tensors_1[0].GetPtr<void>());
      std::vector<Tensor> o_outputs = {o_output};
      attn_o_proj_->Forward({o_input}, o_outputs);
    }

    if (forwarding_context.GetModelCommunicator()) {
      std::swap(hidden_buffer_tensors_1, reduce_buffer_tensors);
    }
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  }

#ifdef ENABLE_VLLM_FLASH_ATTN_2
  set_torch_stream_layers_->Clear();
#endif

  return Status();
}

template <typename T>
Status MultiHeadLatentAttention<T>::DecodeForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                  std::vector<Tensor>& hidden_buffer_tensors_1,
                                                  std::vector<Tensor>& reduce_buffer_tensors,
                                                  std::vector<Tensor>& paged_buffer_tensors,
                                                  ForwardingContext<T>& forwarding_context) {
  const int rank = forwarding_context.GetCurrentRank();
  const Tensor& input = hidden_buffer_tensors_0[0];
  const size_t seq_len = input.shape[0];
  const size_t hidden_units = input.shape[1];
  PROFILE_EVENT_SCOPE(CommonAttention_seq_len_,
                      fmt::format("CommonAttention_seq_len_{}_hidden_units_{}", seq_len, hidden_units), rank);
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  std::vector<Tensor> empty_tensors;
  set_torch_stream_layers_->Forward(empty_tensors, empty_tensors);
#endif

  CREATE_BUFFER_SCOPE(kv_buffer_tensors, mla_buffers_.kv_buffer);
  CREATE_BUFFER_SCOPE(k_rope_buffer_tensors, mla_buffers_.k_rope_buffer);
  CREATE_BUFFER_SCOPE(q_buffer_tensors, mla_buffers_.q_buffer);
  CREATE_BUFFER_SCOPE(q_rope_buffer_tensors, mla_buffers_.q_rope_buffer);

  // kv_a_lora proj MatMul
  {
    PROFILE_EVENT_SCOPE(attn_kv_a_lora_proj, "attn_kv_a_lora_proj", rank);
    STATUS_CHECK_RETURN(attn_kv_a_lora_projs_->Forward(input, hidden_buffer_tensors_1));
  }
  {
    PROFILE_EVENT_SCOPE(kv_a_layernorm, "kv_a_layernorm", rank);
    kv_a_layernorms_->Forward(hidden_buffer_tensors_1, kv_buffer_tensors);
  }
  // kv_a_rope_lora proj MatMul
  {
    PROFILE_EVENT_SCOPE(kv_a_rope_proj, "kv_a_rope_proj", rank);
    STATUS_CHECK_RETURN(attn_kv_a_ropes_->Forward(input, k_rope_buffer_tensors));
  }
  Tensor prefill_hidden_buffer_1 = input;
  Tensor decode_hidden_buffer_1 = input;
  Tensor q_b_rope_input = input;

  // 降维度，q_lora_rank存在
  if (use_q_lora_) {
    // q_a proj MatMul
    {
      PROFILE_EVENT_SCOPE(q_a_proj, "q_a_proj", rank);
      STATUS_CHECK_RETURN(attn_q_a_projs_->Forward(input, q_buffer_tensors));
    }
    {
      PROFILE_EVENT_SCOPE(q_a_layernorm, "q_a_layernorm", rank);
      q_a_layernorms_->Forward(q_buffer_tensors, hidden_buffer_tensors_1);
    }
    prefill_hidden_buffer_1 = hidden_buffer_tensors_1[0];
    decode_hidden_buffer_1 = hidden_buffer_tensors_1[0];
    q_b_rope_input = hidden_buffer_tensors_1[0];
  }

  // q_b_rope proj MatMul{}
  {
    PROFILE_EVENT_SCOPE(q_b_rope_proj_weight, "q_b_rope_proj_weight", rank);
    STATUS_CHECK_RETURN(attn_q_b_rope_projs_->Forward(q_b_rope_input, q_rope_buffer_tensors));
  }

  const size_t decode_tokens = forwarding_context.GetModelInput()->page_single_input.total_dp_input_ids_len +
                               forwarding_context.GetModelInput()->page_dual_input.total_dp_input_ids_len;
  const size_t context_tokens = forwarding_context.GetModelInput()->flash_input.total_dp_input_ids_len;
  KLLM_LOG_DEBUG << "decode_tokens " << decode_tokens << ", context token " << context_tokens;

  prefill_hidden_buffer_1.shape[0] = context_tokens;
  decode_hidden_buffer_1.shape[0] = decode_tokens;

  Tensor decode_hidden_buffer_1_tmp(decode_hidden_buffer_1.location, decode_hidden_buffer_1.dtype,
                                    std::vector<size_t>(decode_hidden_buffer_1.shape), decode_hidden_buffer_1.device_id,
                                    decode_hidden_buffer_1.GetPtr<void>() + prefill_hidden_buffer_1.GetTotalBytes());

  std::vector<Tensor> prefill_q_buffer_tensors{1}, decode_q_buffer_tensors{1};
  prefill_q_buffer_tensors[0] = q_buffer_tensors[0];
  prefill_q_buffer_tensors[0].shape = {context_tokens, head_num_per_tp_ * qk_nope_head_dim_};
  decode_q_buffer_tensors[0] = q_buffer_tensors[0];

  Tensor decode_q_buffer_tmp(decode_q_buffer_tensors[0].location, decode_q_buffer_tensors[0].dtype,
                             std::vector<size_t>(decode_q_buffer_tensors[0].shape),
                             decode_q_buffer_tensors[0].device_id,
                             decode_q_buffer_tensors[0].GetPtr<void>() + prefill_q_buffer_tensors[0].GetTotalBytes());
  std::vector<Tensor> decode_q_buffer_tensors_tmp = {decode_q_buffer_tmp};

  // For decode
  if (decode_tokens) {
    if (absorb_type_ == AbsorbWeightsType::kAbsorbTypeBMM) {
      {
        PROFILE_EVENT_SCOPE(decode_tokens, "decode_tokens q_b_nope_proj", rank);
        STATUS_CHECK_RETURN(attn_q_b_lora_projs_->Forward(decode_hidden_buffer_1_tmp, decode_q_buffer_tensors_tmp));
      }

      // transpose and w_uk_t bmm
      {
        PROFILE_EVENT_SCOPE(decode_tokens, "decode_tokens attn_w_uk_t_bmm", rank);
        int decode_tokens_num = decode_q_buffer_tensors_tmp[0].shape[0];
        decode_q_buffer_tensors_tmp[0].shape = {decode_tokens_num, head_num_per_tp_, qk_nope_head_dim_};
        STATUS_CHECK_RETURN(attn_w_uk_t_bmm_->Forward(
            {decode_q_buffer_tensors_tmp[0], hidden_buffer_tensors_1[0], hidden_buffer_tensors_0[0]},
            decode_q_buffer_tensors_tmp));
      }

      hidden_buffer_tensors_0[0].shape[0] = seq_len;
      hidden_buffer_tensors_1[0].shape[0] = seq_len;
    } else {
      PROFILE_EVENT_SCOPE(decode_tokens, "decode_tokens q_b_nope_proj", rank);
      STATUS_CHECK_RETURN(attn_q_b_lora_projs_->Forward(decode_hidden_buffer_1_tmp, decode_q_buffer_tensors_tmp));
    }

    if (forwarding_context.GetModelCommunicator()) {
      std::swap(hidden_buffer_tensors_1, reduce_buffer_tensors);
    }

    PagedAttentionForward(hidden_buffer_tensors_0, hidden_buffer_tensors_1, paged_buffer_tensors,
                          decode_q_buffer_tensors_tmp[0], q_rope_buffer_tensors[0], kv_buffer_tensors[0],
                          k_rope_buffer_tensors[0], forwarding_context);
    {
      PROFILE_EVENT_SCOPE(o_prpj, "o_prpj", rank);
      Tensor o_input(hidden_buffer_tensors_0[0].location, hidden_buffer_tensors_0[0].dtype,
                     {decode_tokens, o_proj_k_dim_}, hidden_buffer_tensors_0[0].device_id,
                     hidden_buffer_tensors_0[0].GetPtr<void>() +
                         context_tokens * o_proj_k_dim_ * hidden_buffer_tensors_0[0].GetDTypeSize());
      Tensor o_output(
          hidden_buffer_tensors_1[0].location, hidden_buffer_tensors_1[0].dtype,
          {decode_tokens, hidden_buffer_tensors_1[0].shape[1]}, hidden_buffer_tensors_1[0].device_id,
          hidden_buffer_tensors_1[0].GetPtr<void>() +
              context_tokens * (hidden_buffer_tensors_1[0].GetTotalBytes() / hidden_buffer_tensors_1[0].shape[0]));
      std::vector<Tensor> o_outputs = {o_output};
      attn_o_proj_->Forward({o_input}, o_outputs);
    }
    if (forwarding_context.GetModelCommunicator()) {
      std::swap(hidden_buffer_tensors_1, reduce_buffer_tensors);
    }
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  }

#ifdef ENABLE_VLLM_FLASH_ATTN_2
  set_torch_stream_layers_->Clear();
#endif

  return Status();
}

template <typename T>
Status MultiHeadLatentAttention<T>::FlashAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                          std::vector<Tensor>& hidden_buffer_tensors_1,
                                                          std::vector<Tensor>& reduce_buffer_tensors,
                                                          std::vector<Tensor>& output_tensors,
                                                          Tensor& prefill_q_buffer_tensor, Tensor& q_rope_buffer_tensor,
                                                          Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor,
                                                          ForwardingContext<T>& forwarding_context) {
  PROFILE_EVENT_SCOPE(FlashAttentionForward, "FlashAttentionForward", forwarding_context.GetCurrentRank());
  {
    CREATE_BUFFER_SCOPE(prefix_o_buffer_tensors, mla_buffers_.prefix_o_buffer);
    CREATE_BUFFER_SCOPE(prefix_k_up_buffer_tensors, mla_buffers_.prefix_k_up_buffer);
    CREATE_BUFFER_SCOPE(prefix_v_up_buffer_tensors, mla_buffers_.prefix_v_up_buffer);

    CREATE_BUFFER_SCOPE(prefix_kv_buffer_tensors, mla_buffers_.shared_prefix_k_v_kv_buffer);
    Tensor prefix_k_buffer(prefix_kv_buffer_tensors[0].location, prefix_kv_buffer_tensors[0].dtype,
                           {mla_buffers_.prefix_k_buffer_size}, prefix_kv_buffer_tensors[0].device_id,
                           prefix_kv_buffer_tensors[0].GetPtr<void>());
    Tensor prefix_v_buffer(prefix_kv_buffer_tensors[0].location, prefix_kv_buffer_tensors[0].dtype,
                           {mla_buffers_.prefix_v_buffer_size}, prefix_kv_buffer_tensors[0].device_id,
                           prefix_kv_buffer_tensors[0].GetPtr<void>() +
                               mla_buffers_.prefix_k_buffer_size * prefix_kv_buffer_tensors[0].GetDTypeSize());

    STATUS_CHECK_RETURN(flash_mla_attention_layers_->Forward(
        hidden_buffer_tensors_0, forwarding_context.GetModelInput(), hidden_buffer_tensors_1,
        forwarding_context.GetAttentionForwardContext(), prefill_q_buffer_tensor, q_rope_buffer_tensor,
        kv_buffer_tensor, k_rope_buffer_tensor, prefix_k_buffer, prefix_v_buffer, prefix_o_buffer_tensors[0],
        prefix_kv_buffer_tensors[0], prefix_k_up_buffer_tensors[0], prefix_v_up_buffer_tensors[0], output_tensors));
  }
  return Status();
}

template <typename T>
Status MultiHeadLatentAttention<T>::PagedAttentionForward(std::vector<Tensor>& output_tensor,
                                                          std::vector<Tensor>& hidden_buffer_tensors_1,
                                                          std::vector<Tensor>& workspace_buffer,
                                                          Tensor& decode_q_buffer_tensor, Tensor& q_rope_buffer_tensor,
                                                          Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor,
                                                          ForwardingContext<T>& forwarding_context) {
  PROFILE_EVENT_SCOPE(PagedAttentionForward, "PagedAttentionForward", forwarding_context.GetCurrentRank());
  {
    CREATE_BUFFER_SCOPE(kv_cache_buffer_tensors, forwarding_context.GetForwardingBuffers()->kv_cache_buffer);

    // Process seq1 and seq2 separately
    if (!forwarding_context.GetModelInput()->page_single_input.dp_reqs.empty()) {
      Tensor decode_one_seq(decode_q_buffer_tensor.location, decode_q_buffer_tensor.dtype,
                            std::vector<size_t>(decode_q_buffer_tensor.shape), decode_q_buffer_tensor.device_id,
                            decode_q_buffer_tensor.GetPtr<void>() +
                                decode_q_buffer_tensor.GetTotalBytes() / decode_q_buffer_tensor.shape[0] *
                                    forwarding_context.GetModelInput()->page_dual_input.total_dp_input_ids_len);

      STATUS_CHECK_RETURN(paged_mla_attention_layers_->Forward(
          output_tensor, forwarding_context.GetModelInput()->page_single_input, hidden_buffer_tensors_1,
          kv_cache_buffer_tensors[0], forwarding_context.GetAttentionForwardContext(), workspace_buffer[0],
          decode_one_seq, q_rope_buffer_tensor, kv_buffer_tensor, k_rope_buffer_tensor));
    }

    if (!forwarding_context.GetModelInput()->page_dual_input.dp_reqs.empty()) {
      STATUS_CHECK_RETURN(paged_mla_attention_layers_->Forward(
          output_tensor, forwarding_context.GetModelInput()->page_dual_input, hidden_buffer_tensors_1,
          kv_cache_buffer_tensors[0], forwarding_context.GetAttentionForwardContext(), workspace_buffer[0],
          decode_q_buffer_tensor, q_rope_buffer_tensor, kv_buffer_tensor, k_rope_buffer_tensor));
    }
  }
  return Status();
}

template class MultiHeadLatentAttention<float>;
template class MultiHeadLatentAttention<float16>;
#ifdef ENABLE_BFLOAT16
template class MultiHeadLatentAttention<bfloat16>;
#endif

}  // namespace ksana_llm
