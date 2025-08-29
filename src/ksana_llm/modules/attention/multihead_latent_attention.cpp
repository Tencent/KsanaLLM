/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/attention/multihead_latent_attention.h"

#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

MultiHeadLatentAttention::MultiHeadLatentAttention(int layer_idx, bool is_neox, LayerCreationContext& creation_context,
                                                   ModelCreationConfig& model_creation_config, MlaBuffers& mla_buffers)
    : layer_idx_(layer_idx),
      tensor_parallel_size_(creation_context.runtime_config.parallel_basic_config.tensor_parallel_size),
      mla_buffers_(mla_buffers) {
  auto& attn_config = model_creation_config.attn_config;
  absorb_type_ = GetAbsorbWeightsType();
  if (creation_context.runtime_config.enable_o_proj_out_of_dp) {
    // TODO(rockcao): support `o_proj_out_of_dp_` again
    o_proj_out_of_dp_ = true;
    KLLM_LOG_DEBUG << "Enable o_proj_out_of_dp";
  }
  if (absorb_type_ == AbsorbWeightsType::kAbsorbDisabled || absorb_type_ == AbsorbWeightsType::kAbsorbTypeBMM) {
    o_proj_k_dim_ = head_num_per_atp_ * attn_config.model_config.mla_config.v_head_dim;
  } else {
    KLLM_THROW("Unsupported absorb type");
    return;
  }

  use_q_lora_ = (attn_config.model_config.mla_config.q_lora_rank != 0);

  // attn_config.idx is the offset in kv_cache list.
  // e.g., master has normal layer 0-30 and nextn layer 61, the offset of layer_idx_61 is 31
  // e.g., master has normal layer 31-60, the offset of layer_idx_31 is 0
  attn_config.idx = layer_idx - creation_context.pipeline_config.lower_layer_idx;
  if (layer_idx >= static_cast<int>(model_creation_config.attn_config.model_config.num_layer)) {
    attn_config.idx = creation_context.pipeline_config.upper_layer_idx -
                      creation_context.pipeline_config.lower_layer_idx + layer_idx -
                      model_creation_config.attn_config.model_config.num_layer + 1;
  }

  flash_mla_attention_layers_ = std::make_shared<FlashMlaAttention>(layer_idx, is_neox, creation_context, attn_config);
  paged_mla_attention_layers_ =
      std::make_shared<PagedMlaAttention>(layer_idx, is_neox, absorb_type_, creation_context, attn_config);

  const std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
  kv_a_layernorms_ =
      std::make_shared<Layernorm>(layer_prefix + ".self_attn.kv_a_layernorm.weight",
                                  model_creation_config.layernorm_config.layernorm_eps, creation_context);
  q_a_layernorms_ = std::make_shared<Layernorm>(layer_prefix + ".self_attn.q_a_layernorm.weight",
                                                model_creation_config.layernorm_config.layernorm_eps, creation_context);

  const auto& linear_group_quant_backend = model_creation_config.attn_config.model_config.quant_config.backend;

  // TODO(huicongyao, jinxcwu): suppport INT4 model to keep use_fused_lora_a_ always true
  const std::string fused_lora_a_projs_weight_name = layer_prefix + ".self_attn.fused_lora_a_proj.weight";
  if (creation_context.base_weight->GetModelWeights(fused_lora_a_projs_weight_name).GetElementNumber() > 0) {
    use_fused_lora_a_ = true;
    attn_fused_lora_a_projs_ =
        std::make_shared<Linear>(fused_lora_a_projs_weight_name, creation_context, linear_group_quant_backend);
  } else {
    use_fused_lora_a_ = false;
    attn_q_a_projs_ = std::make_shared<Linear>(layer_prefix + ".self_attn.q_a_proj.weight", creation_context,
                                               linear_group_quant_backend);
    attn_kv_a_lora_projs_ = std::make_shared<Linear>(layer_prefix + ".self_attn.kv_a_lora_proj.weight",
                                                     creation_context, linear_group_quant_backend);
    attn_kv_a_ropes_ = std::make_shared<Linear>(layer_prefix + ".self_attn.kv_a_rope_proj.weight", creation_context,
                                                linear_group_quant_backend);
  }

  std::string q_b_nope_rope_proj_weight_name = layer_prefix + ".self_attn.q_b_nope_rope_proj.weight";
  attn_q_b_projs_ =
      std::make_shared<Linear>(q_b_nope_rope_proj_weight_name, creation_context, linear_group_quant_backend);

  split_ = std::make_shared<Split>(creation_context);

  if (absorb_type_ == AbsorbWeightsType::kAbsorbTypeBMM) {
    attn_o_proj_ = std::make_shared<Linear>(layer_prefix + ".self_attn.o_proj.weight", creation_context,
                                            linear_group_quant_backend);
    attn_w_uk_t_bmm_ =
        std::make_shared<Bmm>(layer_prefix + ".self_attn.w_uk_t.weight", creation_context, linear_group_quant_backend);
  } else if (absorb_type_ == AbsorbWeightsType::kAbsorbDisabled) {
    attn_o_proj_ = std::make_shared<Linear>(layer_prefix + ".self_attn.o_proj.weight", creation_context,
                                            linear_group_quant_backend);
  } else {
    KLLM_THROW(fmt::format("Unsupported absorb type {}", absorb_type_));
    return;
  }

  if (o_proj_out_of_dp_) {
    mem_adjuster_ = std::make_shared<MemAdjuster>(creation_context);
  }

#ifdef ENABLE_CUDA
  set_torch_stream_layers_ = std::make_shared<SetTorchStreamLayer>();
  set_torch_stream_layers_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
#endif
}

Status MultiHeadLatentAttention::CreateBuffers(BufferManager* buffer_mgr, const AttentionCreationConfig& attn_config,
                                               const RuntimeConfig& runtime_config, MlaBuffers& mla_buffers) {
  const DataType weight_type = attn_config.model_config.weight_data_type;
  const size_t max_token_num = runtime_config.max_step_token_num;
  const size_t head_num = attn_config.model_config.head_num;
  const size_t max_decode_tokens = runtime_config.max_batch_size * attn_config.max_decode_tokens_per_req;

  head_num_per_atp_ = head_num / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  qk_nope_head_dim_ = attn_config.model_config.mla_config.qk_nope_head_dim;
  qk_rope_head_dim_ = attn_config.model_config.mla_config.qk_rope_head_dim;
  kv_lora_rank_ = attn_config.model_config.mla_config.kv_lora_rank;
  q_lora_rank_ = attn_config.model_config.mla_config.q_lora_rank;

  // 权重吸收后要考虑decode的q维度有变化需要对比。
  const size_t q_buffer_size = std::max(
      max_decode_tokens * head_num_per_atp_ * std::max(qk_nope_head_dim_, kv_lora_rank_), max_token_num * q_lora_rank_);
  mla_buffers.q_buffer = buffer_mgr->CreateBufferTensor("mla_buffers.q_buffer", {q_buffer_size}, weight_type);

  const size_t kv_lora_or_q_nope_rope_buffer_size =
      max_token_num * std::max(head_num_per_atp_ * (qk_nope_head_dim_ + qk_rope_head_dim_), kv_lora_rank_);
  mla_buffers.kv_lora_or_q_nope_rope_buffer = buffer_mgr->CreateBufferTensor(
      "mla_buffers.kv_lora_or_q_nope_rope_buffer", {kv_lora_or_q_nope_rope_buffer_size}, weight_type);

  const size_t decode_q_rope_buffer_size = max_decode_tokens * head_num_per_atp_ * qk_rope_head_dim_;
  mla_buffers.decode_q_rope_buffer =
      buffer_mgr->CreateBufferTensor("mla_buffers.decode_q_rope_buffer", {decode_q_rope_buffer_size}, weight_type);

  const size_t kv_buffer_size = max_token_num * kv_lora_rank_;
  mla_buffers.kv_buffer = buffer_mgr->CreateBufferTensor("mla_buffers.kv_buffer", {kv_buffer_size}, weight_type);

  const size_t k_rope_buffer_size = max_token_num * qk_rope_head_dim_;
  mla_buffers.k_rope_buffer =
      buffer_mgr->CreateBufferTensor("mla_buffers.k_rope_buffer", {k_rope_buffer_size}, weight_type);

  const size_t prefix_kv_buffer_size =
      runtime_config.enable_prefix_caching ? max_token_num * (kv_lora_rank_ + qk_rope_head_dim_) : 0;
  mla_buffers.shared_prefix_kv_buffer =
      buffer_mgr->CreateBufferTensor("mla_buffers.shared_prefix_kv_buffer", {prefix_kv_buffer_size}, weight_type);
  return Status();
}

Status MultiHeadLatentAttention::AcquireBuffers(ForwardingContext& forwarding_context) {
  // TODO(yancyliu): Get tensors from q_buffer, kv_lora_or_q_nope_rope_buffer, kv_buffer, k_rope_buffer in mla_buffers_.
  // Reset its shape from batch_size and token_num, and then allocate tensor memory.
  return Status();
}

Status MultiHeadLatentAttention::ReleaseBuffers() {
  // TODO(yancyliu): Get tensor from q_buffer, kv_lora_or_q_nope_rope_buffer, kv_buffer, k_rope_buffer
  // then release tenosr memory.
  return Status();
}

Status MultiHeadLatentAttention::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                         std::vector<Tensor>& reduce_buffer_tensors,
                                         std::shared_ptr<TpCommunicator> tp_comm, bool is_multi_token_forward,
                                         ForwardingContext& forwarding_context) {
  const int rank = forwarding_context.GetCurrentRank();

  const size_t total_tokens = hidden_buffer_tensors_0[0].shape[0];
  const size_t hidden_units = hidden_buffer_tensors_0[0].shape[1];
  const size_t hidden_units_bytes = hidden_units * hidden_buffer_tensors_0[0].GetDTypeSize();

  // `dp_group_id` is responsible for the tokens in `[dp_token_offset, dp_token_offset + dp_context_tokens +
  // dp_decode_tokens)`
  // When disable attention dp, `dp_token_offset = 0, dp_context_tokens + dp_decode_tokens = total_tokens`
  const size_t dp_group_id = forwarding_context.GetModelInput()->attn_dp_group_id_;
  const int dp_token_offset = forwarding_context.GetModelInput()->attn_dp_group_offsets_[dp_group_id];
  const size_t dp_context_tokens = forwarding_context.GetModelInput()->flash_input.total_dp_input_ids_len;
  const size_t dp_decode_tokens = forwarding_context.GetModelInput()->page_single_input.total_dp_input_ids_len +
                                  forwarding_context.GetModelInput()->page_dual_input.total_dp_input_ids_len;
  const size_t dp_total_tokens = dp_context_tokens + dp_decode_tokens;
  KLLM_LOG_DEBUG << fmt::format(
      "rank: {}, dp_group_id: {}, dp_token_offset: {}, dp_context_tokens: {}, dp_decode_tokens: {}", rank, dp_group_id,
      dp_token_offset, dp_context_tokens, dp_decode_tokens);

  PROFILE_EVENT_SCOPE(CommonAttention_seq_len_,
                      fmt::format("CommonAttention_seq_len_{}_hidden_units_{}", dp_total_tokens, hidden_units), rank);

#ifdef ENABLE_CUDA
  std::vector<Tensor> empty_tensors;
  set_torch_stream_layers_->Forward(empty_tensors, empty_tensors);
#endif

  // Only take tokens assigned to the current dp group
  const Tensor& dp_hidden_input =
      hidden_buffer_tensors_0[0].GetView({dp_total_tokens, hidden_units}, dp_token_offset * hidden_units);
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
  CREATE_BUFFER_SCOPE(kv_buffer_tensors, mla_buffers_.kv_buffer);
  CREATE_BUFFER_SCOPE(k_rope_buffer_tensors, mla_buffers_.k_rope_buffer);
  CREATE_BUFFER_SCOPE(q_buffer_tensors, mla_buffers_.q_buffer);
  CREATE_BUFFER_SCOPE(decode_q_rope_tensors, mla_buffers_.decode_q_rope_buffer);
  CREATE_BUFFER_SCOPE(q_nope_rope_buffer_tensors, mla_buffers_.kv_lora_or_q_nope_rope_buffer);
  if (dp_total_tokens > 0) {
    std::vector<Tensor> kv_lora_buffer_tensors = {q_nope_rope_buffer_tensors};
    if (use_fused_lora_a_) {
      PROFILE_EVENT_SCOPE(attn_fused_lora_a_projs, "attn_fused_lora_a_proj", rank);
      // weight_shape = (q_lora_rank + kv_lora_rank + qk_rope_head_dim, hidden_units)
      STATUS_CHECK_RETURN(attn_fused_lora_a_projs_->Forward(dp_hidden_input, hidden_buffer_tensors_1));

      // split to q_buffer_tensors, kv_lora_buffer_tensors, k_rope_buffer_tensors
      q_buffer_tensors[0].shape = {dp_total_tokens, q_lora_rank_};
      kv_lora_buffer_tensors[0].shape = {dp_total_tokens, kv_lora_rank_};
      k_rope_buffer_tensors[0].shape = {dp_total_tokens, qk_rope_head_dim_};
      std::vector<Tensor> split_output_tensors = {q_buffer_tensors[0], kv_lora_buffer_tensors[0],
                                                  k_rope_buffer_tensors[0]};
      split_->Forward(hidden_buffer_tensors_1[0], split_output_tensors);
    } else {
      {  // kv_a_lora proj MatMul
        PROFILE_EVENT_SCOPE(attn_kv_a_lora_proj, "attn_kv_a_lora_proj", rank);
        STATUS_CHECK_RETURN(attn_kv_a_lora_projs_->Forward(
            dp_hidden_input, kv_lora_buffer_tensors));  // weight_shape = (kv_lora_rank, hidden_units)
      }
      {  // kv_a_rope_lora proj MatMul
        PROFILE_EVENT_SCOPE(kv_a_rope_proj, "kv_a_rope_proj", rank);
        STATUS_CHECK_RETURN(attn_kv_a_ropes_->Forward(
            dp_hidden_input, k_rope_buffer_tensors));  // weight_shape = (qk_rope_head_dim, hidden_units)
      }
    }
    {
      PROFILE_EVENT_SCOPE(kv_a_layernorm, "kv_a_layernorm", rank);
      kv_a_layernorms_->Forward(kv_lora_buffer_tensors, kv_buffer_tensors);
    }

    Tensor q_b_input = dp_hidden_input;
    // tensor0 is used for input, tensor1 is free for output
    Tensor q_b_output = hidden_buffer_tensors_1[0];
    // 降维度，q_lora_rank存在
    if (use_q_lora_) {
      if (!use_fused_lora_a_) {
        // q_a proj MatMul
        PROFILE_EVENT_SCOPE(q_a_proj, "q_a_proj", rank);
        // weight_shape = (q_lora_rank, hidden_units)
        STATUS_CHECK_RETURN(attn_q_a_projs_->Forward(dp_hidden_input, q_buffer_tensors));
      }
      {
        PROFILE_EVENT_SCOPE(q_a_layernorm, "q_a_layernorm", rank);
        q_a_layernorms_->Forward(q_buffer_tensors, hidden_buffer_tensors_1);
      }
      q_b_input = hidden_buffer_tensors_1[0];
      // tensor1 is used for input, tensor0 is free for output
      q_b_output = hidden_buffer_tensors_0[0];
    }

    // `hidden_buffer_tensors_0/1` is able to hold this space,
    // reshape to avoid memory checker error in the following `GetView`
    q_b_output.shape = {total_tokens, head_num_per_atp_ * (qk_nope_head_dim_ + qk_rope_head_dim_)};
    Tensor q_b_nope_rope_output_tmp = q_b_output.GetView({dp_total_tokens, q_b_output.shape[1]});
    std::vector<Tensor> q_b_nope_rope_output_tmps = {q_b_nope_rope_output_tmp};
    // prefill and decode

    PROFILE_EVENT_SCOPE(q_b_nope_rope_proj_weight, "q_b_nope_rope_proj", rank);
    STATUS_CHECK_RETURN(attn_q_b_projs_->Forward(q_b_input, q_nope_rope_buffer_tensors));
    if (dp_decode_tokens > 0) {
      Tensor decode_q_nope_rope = q_nope_rope_buffer_tensors[0].GetView(
          {dp_decode_tokens * head_num_per_atp_, (qk_nope_head_dim_ + qk_rope_head_dim_)},
          dp_context_tokens * head_num_per_atp_ * (qk_nope_head_dim_ + qk_rope_head_dim_));

      q_buffer_tensors[0].shape = {dp_decode_tokens * head_num_per_atp_, qk_nope_head_dim_};
      decode_q_rope_tensors[0].shape = {dp_decode_tokens * head_num_per_atp_, qk_rope_head_dim_};
      std::vector<Tensor> split_output_tensors = {q_buffer_tensors[0], decode_q_rope_tensors[0]};
      split_->Forward(decode_q_nope_rope, split_output_tensors);
      q_buffer_tensors[0].shape = {dp_decode_tokens, head_num_per_atp_ * qk_nope_head_dim_};
      decode_q_rope_tensors[0].shape = {dp_decode_tokens, head_num_per_atp_ * qk_rope_head_dim_};
    }
  }

  Tensor decode_q_nope = q_buffer_tensors[0].GetView({dp_decode_tokens, head_num_per_atp_ * qk_nope_head_dim_});
  if (dp_decode_tokens > 0 && absorb_type_ == AbsorbWeightsType::kAbsorbTypeBMM) {
    PROFILE_EVENT_SCOPE(attn_w_uk_t_bmm, "decode_tokens attn_w_uk_t_bmm", rank);
    decode_q_nope.shape = {dp_decode_tokens, head_num_per_atp_, qk_nope_head_dim_};
    std::vector<Tensor> decode_q_absorb_wuk = {decode_q_nope};
    // 融合Wuk到Qnope, 最后一维从qk_nope_dim变为kv_lora_rank
    STATUS_CHECK_RETURN(attn_w_uk_t_bmm_->Forward(
        {decode_q_nope, hidden_buffer_tensors_1[0], hidden_buffer_tensors_0[0]}, decode_q_absorb_wuk));
  }

  // TODO(robertyuan): swap with reduce_buffer_tensors needs optimize.
  // Swap required: AllReduce only accepts reduce_buffer_tensors as input
  if (forwarding_context.GetModelCommunicator()) {
    if (o_proj_out_of_dp_) {
      std::swap(hidden_buffer_tensors_0, reduce_buffer_tensors);
    } else {
      std::swap(hidden_buffer_tensors_1, reduce_buffer_tensors);
    }
  }

  const size_t attn_output_offset = o_proj_out_of_dp_ ? dp_token_offset * o_proj_k_dim_ : 0;
  std::vector<Tensor> attn_output_tensor = {
      hidden_buffer_tensors_0[0].GetView(hidden_buffer_tensors_0[0].shape, attn_output_offset)};

  if (dp_context_tokens > 0) {
    Tensor prefill_q_nope_rope = q_nope_rope_buffer_tensors[0].GetView(
        {dp_context_tokens, head_num_per_atp_ * (qk_rope_head_dim_ + qk_nope_head_dim_)});
    FlashAttentionForward(hidden_buffer_tensors_0, hidden_buffer_tensors_1, attn_output_tensor, prefill_q_nope_rope,
                          kv_buffer_tensors[0], k_rope_buffer_tensors[0], forwarding_context);
  }

  if (dp_decode_tokens > 0) {
    PagedAttentionForward(attn_output_tensor, hidden_buffer_tensors_1, reduce_buffer_tensors, decode_q_nope,
                          decode_q_rope_tensors[0], kv_buffer_tensors[0], k_rope_buffer_tensors[0], forwarding_context);
  }

  if (o_proj_out_of_dp_) {
    const size_t o_proj_k_dim_per_tp = o_proj_k_dim_ / tensor_parallel_size_;
    PROFILE_EVENT_SCOPE(o_proj, fmt::format("o_proj_ouf_of_dp_m_{}_n_{}", total_tokens, o_proj_k_dim_per_tp), rank);
    hidden_buffer_tensors_0[0].shape = {total_tokens, o_proj_k_dim_};
    if (dp_token_offset > 0) {
      // `[0, dp_token_offset)`
      MemsetAsync(hidden_buffer_tensors_0[0].GetPtr<void>(), 0,
                  dp_token_offset * o_proj_k_dim_ * hidden_buffer_tensors_0[0].GetDTypeSize(),
                  forwarding_context.GetContext()->GetComputeStreams()[rank]);
    }
    if (dp_token_offset + dp_total_tokens < total_tokens) {
      // `[dp_token_offset + dp_total_tokens, total_tokens)`
      MemsetAsync(hidden_buffer_tensors_0[0].GetPtr<void>() +
                      (dp_token_offset + dp_total_tokens) * o_proj_k_dim_ * hidden_buffer_tensors_0[0].GetDTypeSize(),
                  0,
                  (total_tokens - dp_token_offset - dp_total_tokens) * o_proj_k_dim_ *
                      hidden_buffer_tensors_0[0].GetDTypeSize(),
                  forwarding_context.GetContext()->GetComputeStreams()[rank]);
    }
    tp_comm->AllReduce(hidden_buffer_tensors_0, hidden_buffer_tensors_1, is_multi_token_forward, forwarding_context);

    mem_adjuster_->ExtractSubMatrix(hidden_buffer_tensors_1[0], reduce_buffer_tensors[0],
                                    o_proj_k_dim_per_tp * dp_group_id, o_proj_k_dim_per_tp);

    std::vector<Tensor> o_outputs = {hidden_buffer_tensors_0[0].GetView({dp_total_tokens, hidden_units})};
    attn_o_proj_->Forward(reduce_buffer_tensors, o_outputs);
  } else if (dp_total_tokens > 0) {
    PROFILE_EVENT_SCOPE(o_proj, fmt::format("o_proj_m_{}_n_{}", dp_total_tokens, o_proj_k_dim_), rank);
    // `hidden_buffer_tensors_0` is able to hold this space,
    // reshape to avoid memory checker error in the following `GetView`
    hidden_buffer_tensors_0[0].shape = {total_tokens, o_proj_k_dim_};
    Tensor o_input = hidden_buffer_tensors_0[0].GetView({dp_total_tokens, o_proj_k_dim_});
    // `hidden_buffer_tensors_1` is able to hold this space,
    // reshape to avoid memory checker error in the following `GetView`
    hidden_buffer_tensors_1[0].shape = {total_tokens, hidden_units};
    // Only output tokens assigned to the current dp group
    std::vector<Tensor> o_outputs = {
        hidden_buffer_tensors_1[0].GetView({dp_total_tokens, hidden_units}, dp_token_offset * hidden_units)};
    attn_o_proj_->Forward({o_input}, o_outputs);
  }

  // swap back
  if (forwarding_context.GetModelCommunicator()) {
    if (o_proj_out_of_dp_) {
      std::swap(hidden_buffer_tensors_0, reduce_buffer_tensors);
    } else {
      std::swap(hidden_buffer_tensors_1, reduce_buffer_tensors);
    }
  }
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);

  if (forwarding_context.GetModelCommunicator()) {
    if (!o_proj_out_of_dp_) {
      // The output is now in the `reduce_buffer_tensors[0]` for the following allreduce
      // We should set the output of tokens not assigned to the current dp group to zero
      if (dp_token_offset > 0) {
        // `[0, dp_token_offset)`
        MemsetAsync(reduce_buffer_tensors[0].GetPtr<void>(), 0, dp_token_offset * hidden_units_bytes,
                    forwarding_context.GetContext()->GetComputeStreams()[rank]);
      }
      if (dp_token_offset + dp_total_tokens < total_tokens) {
        // `[dp_token_offset + dp_total_tokens, total_tokens)`
        MemsetAsync(reduce_buffer_tensors[0].GetPtr<void>() + (dp_token_offset + dp_total_tokens) * hidden_units_bytes,
                    0, (total_tokens - dp_token_offset - dp_total_tokens) * hidden_units_bytes,
                    forwarding_context.GetContext()->GetComputeStreams()[rank]);
      }
    }
    // correctly set the output shape
    reduce_buffer_tensors[0].shape = {total_tokens, hidden_units};
  } else {
    // The output is now in the `hidden_buffer_tensors_0[0]`
    // We should correctly set the output shape
    hidden_buffer_tensors_0[0].shape = {total_tokens, hidden_units};
  }

#ifdef ENABLE_CUDA
  set_torch_stream_layers_->Clear();
#endif

  return Status();
}

Status MultiHeadLatentAttention::FlashAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                       std::vector<Tensor>& workspace_buffer,
                                                       std::vector<Tensor>& output_tensors, Tensor& q_nope_rope_tensor,
                                                       Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor,
                                                       ForwardingContext& forwarding_context) {
  PROFILE_EVENT_SCOPE(FlashAttentionForward, "FlashAttentionForward", forwarding_context.GetCurrentRank());
  {
    CREATE_BUFFER_SCOPE(prefix_kv_buffer_tensors, mla_buffers_.shared_prefix_kv_buffer);
    STATUS_CHECK_RETURN(flash_mla_attention_layers_->Forward(
        hidden_buffer_tensors_0, forwarding_context.GetModelInput(), workspace_buffer,
        forwarding_context.GetAttentionForwardContext(), q_nope_rope_tensor, kv_buffer_tensor, k_rope_buffer_tensor,
        prefix_kv_buffer_tensors[0], output_tensors));
  }
  return Status();
}

Status MultiHeadLatentAttention::PagedAttentionForward(std::vector<Tensor>& output_tensor,
                                                       std::vector<Tensor>& hidden_buffer_tensors_1,
                                                       std::vector<Tensor>& workspace_buffer,
                                                       Tensor& decode_q_buffer_tensor, Tensor& q_rope_buffer_tensor,
                                                       Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor,
                                                       ForwardingContext& forwarding_context) {
  PROFILE_EVENT_SCOPE(PagedAttentionForward, "PagedAttentionForward", forwarding_context.GetCurrentRank());
  {
    CREATE_BUFFER_SCOPE(kv_cache_buffer_tensors, forwarding_context.GetForwardingBuffers()->kv_cache_buffer);

    // Process seq1 and seq2 separately
    if (!forwarding_context.GetModelInput()->page_single_input.dp_reqs.empty()) {
      const size_t skip_tokens = forwarding_context.GetModelInput()->page_dual_input.total_dp_input_ids_len;
      Tensor decode_nope = decode_q_buffer_tensor.GetView(
          {decode_q_buffer_tensor.shape[0] - skip_tokens, decode_q_buffer_tensor.shape[1]},
          skip_tokens * decode_q_buffer_tensor.GetElementNumber() / decode_q_buffer_tensor.shape[0]);
      Tensor decode_rope = q_rope_buffer_tensor.GetView(
          {q_rope_buffer_tensor.shape[0] - skip_tokens, q_rope_buffer_tensor.shape[1]},
          skip_tokens * q_rope_buffer_tensor.GetElementNumber() / q_rope_buffer_tensor.shape[0]);
      STATUS_CHECK_RETURN(paged_mla_attention_layers_->Forward(
          output_tensor, forwarding_context.GetModelInput()->page_single_input, hidden_buffer_tensors_1,
          kv_cache_buffer_tensors[0], forwarding_context.GetAttentionForwardContext(), workspace_buffer[0], decode_nope,
          decode_rope, kv_buffer_tensor, k_rope_buffer_tensor));
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

}  // namespace ksana_llm
