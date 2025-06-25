/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/internlmxcomposer2/internlmxcomposer2_model.h"

namespace ksana_llm {

template <typename T>
InternlmxComposer2DecoderLayer<T>::InternlmxComposer2DecoderLayer(int layer_idx,
                                                                  LayerCreationContext<T>& creation_context,
                                                                  ModelCreationConfig& model_creation_config,
                                                                  TensorBuffer* plora_a_buffer_,
                                                                  TensorBuffer* plora_b_buffer_)
    : layer_idx_(layer_idx), plora_a_buffer_(plora_a_buffer_), plora_b_buffer_(plora_b_buffer_) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  input_layernorms_ = std::make_shared<Layernorm<T>>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);
  post_attention_layernorms_ =
      std::make_shared<Layernorm<T>>(layer_prefix + ".post_attention_layernorm.weight",
                                     model_creation_config.layernorm_config.layernorm_eps, creation_context);

  // GEMM related
  adds_ = std::make_shared<Add<T>>(creation_context);
  gate_proj_plora_a_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.gate_proj.Plora_A.weight", creation_context,
                                                   model_creation_config.attn_config.model_config.quant_config.backend);
  gate_proj_plora_b_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.gate_proj.Plora_B.weight", creation_context,
                                                   model_creation_config.attn_config.model_config.quant_config.backend);
  up_proj_plora_a_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.up_proj.Plora_A.weight", creation_context,
                                                 model_creation_config.attn_config.model_config.quant_config.backend);
  up_proj_plora_b_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.up_proj.Plora_B.weight", creation_context,
                                                 model_creation_config.attn_config.model_config.quant_config.backend);
  down_proj_plora_a_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.down_proj.Plora_A.weight", creation_context,
                                                   model_creation_config.attn_config.model_config.quant_config.backend);
  down_proj_plora_b_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.down_proj.Plora_B.weight", creation_context,
                                                   model_creation_config.attn_config.model_config.quant_config.backend);
  qkv_lora_a_proj_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.qkv_proj.Plora_A.weight", creation_context,
                                                 model_creation_config.attn_config.model_config.quant_config.backend);
  qkv_lora_b_proj_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.qkv_rproj.Plora_B.weight", creation_context,
                                                 model_creation_config.attn_config.model_config.quant_config.backend);
  o_lora_a_proj_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.o_proj.Plora_A.weight", creation_context,
                                               model_creation_config.attn_config.model_config.quant_config.backend);
  o_lora_b_proj_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.o_proj.Plora_B.weight", creation_context,
                                               model_creation_config.attn_config.model_config.quant_config.backend);

  // attention related
  bool is_neox = true;
  attn_qkv_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.query_key_value.weight", creation_context,
                                                model_creation_config.attn_config.model_config.quant_config.backend);
  attn_o_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.o_proj.weight", creation_context,
                                              model_creation_config.attn_config.model_config.quant_config.backend);
  model_creation_config.attn_config.idx = layer_idx - creation_context.pipeline_config.lower_layer_idx;
  flash_attentions_ = std::make_shared<FlashAttention<T>>(is_neox, creation_context, model_creation_config.attn_config);
  paged_attentions_ = std::make_shared<PagedAttention<T>>(is_neox, creation_context, model_creation_config.attn_config);

  // ffn related
  mlp_gate_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.gate_proj.weight", creation_context,
                                                model_creation_config.attn_config.model_config.quant_config.backend);
  mlp_up_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.up_proj.weight", creation_context,
                                              model_creation_config.attn_config.model_config.quant_config.backend);
  mlp_down_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.down_proj.weight", creation_context,
                                                model_creation_config.attn_config.model_config.quant_config.backend);
  tp_comm_ = std::make_shared<TpCommunicator<T>>();
  silu_muls_ = std::make_shared<SiluMul<T>>(creation_context);

#ifdef ENABLE_CUDA
  mask_muls_ = std::make_shared<Mul<T>>(creation_context);
#endif
}

template <typename T>
Status InternlmxComposer2DecoderLayer<T>::FlashAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                                std::vector<Tensor>& hidden_buffer_tensors_1,
                                                                std::vector<Tensor>& reduce_buffer_tensors,
                                                                ForwardingContext<T>& forwarding_context) {
  return flash_attentions_->Forward(
      hidden_buffer_tensors_0, forwarding_context.GetModelInput(), hidden_buffer_tensors_1, reduce_buffer_tensors,
      forwarding_context.GetAttentionForwardContext(), query_layernorm_weight_, key_layernorm_weight_);
}

template <typename T>
Status InternlmxComposer2DecoderLayer<T>::PagedAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                                std::vector<Tensor>& hidden_buffer_tensors_1,
                                                                std::vector<Tensor>& reduce_buffer_tensors,
                                                                ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(kv_cache_buffer_tensors, forwarding_context.GetForwardingBuffers()->kv_cache_buffer);
  return paged_attentions_->Forward(hidden_buffer_tensors_0, forwarding_context.GetModelInput(),
                                    hidden_buffer_tensors_1, reduce_buffer_tensors, kv_cache_buffer_tensors[0],
                                    forwarding_context.GetAttentionForwardContext(), query_layernorm_weight_,
                                    key_layernorm_weight_);
}

template <typename T>
Status InternlmxComposer2DecoderLayer<T>::ForwardMha(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                     std::vector<Tensor>& reduce_buffer_tensors,
                                                     std::vector<Tensor>& hidden_buffer_tensors_1,
                                                     const bool is_multi_token_forward,
                                                     ForwardingContext<T>& forwarding_context) {
  // Attn proj MatMul
  STATUS_CHECK_RETURN(attn_qkv_projs_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1));
#ifdef ENABLE_CUDA
  if (forwarding_context.GetModelInput()->multi_token_request_num && forwarding_context.GetModelInput()->is_mask) {
    CREATE_BUFFER_SCOPE(plora_b_buffer_tensor, plora_b_buffer_);
    CREATE_BUFFER_SCOPE(plora_a_buffer_tensor, plora_a_buffer_);
    STATUS_CHECK_RETURN(mask_muls_->Forward(hidden_buffer_tensors_0[0], forwarding_context.GetModelInput()->im_mask,
                                            plora_a_buffer_tensor));
    STATUS_CHECK_RETURN(qkv_lora_a_proj_->Forward(plora_b_buffer_tensor[0], plora_a_buffer_tensor));
    STATUS_CHECK_RETURN(qkv_lora_b_proj_->Forward(plora_a_buffer_tensor[0], plora_b_buffer_tensor));
    STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_1[0], plora_b_buffer_tensor[0], hidden_buffer_tensors_1));
  }
#endif
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  // MMHA Flash/Paged Attention
  if (!forwarding_context.GetModelInput()->is_cudagraph_capture_request && layer_idx_ == 0) {
    // only need sync in the first layer
    StreamWaitEvent(forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()],
                    forwarding_context.GetModelInput()->kvcache_offset_event);
    StreamWaitEvent(forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()],
                    forwarding_context.GetModelInput()->rotary_embedding_event);
  }
  if (forwarding_context.GetModelInput()->multi_token_request_num) {
    FlashAttentionForward(hidden_buffer_tensors_0, hidden_buffer_tensors_1, reduce_buffer_tensors, forwarding_context);
    if (forwarding_context.GetModelInput()->single_token_request_num) {
      std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
    }
  }

  if (forwarding_context.GetModelInput()->single_token_request_num) {
    PagedAttentionForward(hidden_buffer_tensors_0, hidden_buffer_tensors_1, reduce_buffer_tensors, forwarding_context);
  }
  if (forwarding_context.GetModelCommunicator()) {
    STATUS_CHECK_RETURN(attn_o_projs_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors));
  } else {
    STATUS_CHECK_RETURN(attn_o_projs_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1));
    // Plora
#ifdef ENABLE_CUDA
    if (forwarding_context.GetModelInput()->multi_token_request_num && forwarding_context.GetModelInput()->is_mask) {
      CREATE_BUFFER_SCOPE(plora_b_buffer_tensor, plora_b_buffer_);
      CREATE_BUFFER_SCOPE(plora_a_buffer_tensor, plora_a_buffer_);
      STATUS_CHECK_RETURN(mask_muls_->Forward(hidden_buffer_tensors_0[0], forwarding_context.GetModelInput()->im_mask,
                                              plora_b_buffer_tensor));
      STATUS_CHECK_RETURN(o_lora_a_proj_->Forward(plora_b_buffer_tensor[0], plora_a_buffer_tensor));
      STATUS_CHECK_RETURN(o_lora_b_proj_->Forward(plora_a_buffer_tensor[0], plora_b_buffer_tensor));
      STATUS_CHECK_RETURN(
          adds_->Forward(hidden_buffer_tensors_1[0], plora_b_buffer_tensor[0], hidden_buffer_tensors_1));
    }
#endif
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  }
  return Status();
}

template <typename T>
Status InternlmxComposer2DecoderLayer<T>::ForwardMlp(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                     std::vector<Tensor>& hidden_buffer_tensors_1,
                                                     std::vector<Tensor>& reduce_buffer_tensors,
                                                     const bool is_multi_token_forward,
                                                     ForwardingContext<T>& forwarding_context) {
  // Mlp gate_proj MatMul
  STATUS_CHECK_RETURN(mlp_gate_projs_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1));
  // Mlp up_proj MatMul 由于 gate_proj 与 up_proj 为并行关系,因此此处使用额外空间存储 matmul 结果
  STATUS_CHECK_RETURN(mlp_up_projs_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors));
#ifdef ENABLE_CUDA
  // Plora
  if (forwarding_context.GetModelInput()->multi_token_request_num && forwarding_context.GetModelInput()->is_mask) {
    CREATE_BUFFER_SCOPE(plora_b_buffer_tensor, plora_b_buffer_);
    CREATE_BUFFER_SCOPE(plora_a_buffer_tensor, plora_a_buffer_);
    STATUS_CHECK_RETURN(mask_muls_->Forward(hidden_buffer_tensors_0[0], forwarding_context.GetModelInput()->im_mask,
                                            plora_b_buffer_tensor));
    STATUS_CHECK_RETURN(gate_proj_plora_a_->Forward(plora_b_buffer_tensor[0], plora_a_buffer_tensor));
    STATUS_CHECK_RETURN(gate_proj_plora_b_->Forward(plora_a_buffer_tensor[0], plora_b_buffer_tensor));
    STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_1[0], plora_b_buffer_tensor[0], hidden_buffer_tensors_1));
    STATUS_CHECK_RETURN(mask_muls_->Forward(hidden_buffer_tensors_0[0], forwarding_context.GetModelInput()->im_mask,
                                            plora_b_buffer_tensor));
    STATUS_CHECK_RETURN(up_proj_plora_a_->Forward(plora_b_buffer_tensor[0], plora_a_buffer_tensor));
    STATUS_CHECK_RETURN(up_proj_plora_b_->Forward(plora_a_buffer_tensor[0], plora_b_buffer_tensor));
    STATUS_CHECK_RETURN(adds_->Forward(reduce_buffer_tensors[0], plora_b_buffer_tensor[0], reduce_buffer_tensors));
  }
#endif
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  STATUS_CHECK_RETURN(
      silu_muls_->Forward(hidden_buffer_tensors_0[0], reduce_buffer_tensors[0], hidden_buffer_tensors_0));
  // Mlp down_proj MatMul
  if (forwarding_context.GetModelCommunicator()) {
    STATUS_CHECK_RETURN(mlp_down_projs_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors));
  } else {
    STATUS_CHECK_RETURN(mlp_down_projs_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1));
    // Plora
#ifdef ENABLE_CUDA
    if (forwarding_context.GetModelInput()->multi_token_request_num && forwarding_context.GetModelInput()->is_mask) {
      CREATE_BUFFER_SCOPE(plora_b_buffer_tensor, plora_b_buffer_);
      CREATE_BUFFER_SCOPE(plora_a_buffer_tensor, plora_a_buffer_);
      STATUS_CHECK_RETURN(mask_muls_->Forward(hidden_buffer_tensors_0[0], forwarding_context.GetModelInput()->im_mask,
                                              plora_b_buffer_tensor));
      STATUS_CHECK_RETURN(down_proj_plora_a_->Forward(plora_b_buffer_tensor[0], plora_a_buffer_tensor));
      STATUS_CHECK_RETURN(down_proj_plora_b_->Forward(plora_a_buffer_tensor[0], plora_b_buffer_tensor));
      STATUS_CHECK_RETURN(
          adds_->Forward(hidden_buffer_tensors_1[0], plora_b_buffer_tensor[0], hidden_buffer_tensors_1));
    }
#endif
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  }

  return Status();
}

template <typename T>
Status InternlmxComposer2DecoderLayer<T>::Forward(std::vector<Tensor>& residual_buffer,
                                                  const bool is_multi_token_forward,
                                                  ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);
  // Pre attn layernorm
  // Pre layernorm uses layernorm input for residual connection.
  input_layernorms_->Forward(residual_buffer, hidden_buffer_tensors_0);
  // MultiHeadAttention
  STATUS_CHECK_RETURN(ForwardMha(hidden_buffer_tensors_0, reduce_buffer_tensors, hidden_buffer_tensors_1,
                                 is_multi_token_forward, forwarding_context));
  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);
  // Attn residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));
  // Pre mlp layernorm
  // Pre layernorm uses layernorm input for residual connection.
  post_attention_layernorms_->Forward(residual_buffer, hidden_buffer_tensors_0);
  // Common mlp
  STATUS_CHECK_RETURN(ForwardMlp(hidden_buffer_tensors_0, hidden_buffer_tensors_1, reduce_buffer_tensors,
                                 is_multi_token_forward, forwarding_context));
  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);
  // Mlp residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));
  return Status();
}

template <typename T>
Status InternlmxComposer2<T>::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

template <typename T>
Status InternlmxComposer2<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                           ModelCreationConfig& model_creation_config) {
  auto& model_config = model_creation_config.attn_config.model_config;
  DataType weight_type = model_config.weight_data_type;
  size_t vocab_size_pad =
      DivRoundUp(model_config.vocab_size, model_config.tensor_para_size) * model_config.tensor_para_size;
  int head_num = model_config.head_num;
  int size_per_head = model_config.size_per_head;
  int hidden_units = size_per_head * head_num;
  int tensor_para_size = model_config.tensor_para_size;
  int head_num_per_tp = head_num / tensor_para_size;
  int num_kv_heads_per_tp = model_config.num_key_value_heads / tensor_para_size;
  size_t max_token_num = model_config.max_step_token_num;
  size_t max_batch_size = model_config.max_batch_size;
  int inter_size_per_tp = model_config.inter_size / tensor_para_size;
  int max_dim =
      std::max(std::max((head_num_per_tp + 2 * num_kv_heads_per_tp) * size_per_head, hidden_units), inter_size_per_tp);
  size_t hidden_buffer_size = std::max(max_batch_size * vocab_size_pad, max_token_num * max_dim);
  plora_a_buffer_ =
      creation_context.buffer_mgr_->CreateBufferTensor("plora_a_buffer", {hidden_buffer_size}, weight_type);
  plora_b_buffer_ =
      creation_context.buffer_mgr_->CreateBufferTensor("plora_b_buffer", {hidden_buffer_size}, weight_type);
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] = std::make_shared<InternlmxComposer2DecoderLayer<T>>(
        layer_idx, creation_context, model_creation_config, plora_a_buffer_, plora_b_buffer_);
  }
  return Status();
}

template <typename T>
Status InternlmxComposer2<T>::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

template class InternlmxComposer2<float>;
template class InternlmxComposer2<float16>;
#ifdef ENABLE_BFLOAT16
template class InternlmxComposer2<bfloat16>;
#endif

template <typename T>
InternlmxComposer2Model<T>::InternlmxComposer2Model(const ModelConfig& model_config, const int rank,
                                                    std::shared_ptr<Context> context,
                                                    std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, rank, context) {
  ModelRunConfig model_run_config;
  internlmx_composer2_.GetModelRunConfig(model_run_config, model_config);
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status InternlmxComposer2Model<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                                ModelCreationConfig& model_creation_config) {
  return internlmx_composer2_.CreateLayers(creation_context, model_creation_config);
}

template <typename T>
Status InternlmxComposer2Model<T>::LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  STATUS_CHECK_RETURN(internlmx_composer2_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);

  return Status();
}

template class InternlmxComposer2Model<float>;
template class InternlmxComposer2Model<float16>;
#ifdef ENABLE_BFLOAT16
template class InternlmxComposer2Model<bfloat16>;
#endif

}  // namespace ksana_llm
