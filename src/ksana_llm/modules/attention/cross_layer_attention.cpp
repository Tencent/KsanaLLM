/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/attention/cross_layer_attention.h"

namespace ksana_llm {

template <typename T>
CrossLayerAttention<T>::CrossLayerAttention(int layer_idx, int cla_share_factor, ClaBuffers& cla_buffers,
                                            LayerCreationContext<T>& creation_context,
                                            ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx), cla_share_factor_(cla_share_factor), cla_buffers_(cla_buffers) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  // Attention related blocks
  if (cla_share_factor_ != 0 && (layer_idx % cla_share_factor_ != 0)) {
    attn_qkv_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.q_proj.weight", creation_context,
                                                  model_creation_config.attn_config.model_config.quant_config.backend);
  } else {
    // Cla only do q_proj for odd layers when cla_share_factor=2
    attn_qkv_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.query_key_value.weight", creation_context,
                                                  model_creation_config.attn_config.model_config.quant_config.backend);
  }

  bool is_neox = true;
  bool use_qk_norm = true;
  attentions_ =
      std::make_shared<CommonAttention<T>>(layer_idx, is_neox, use_qk_norm, creation_context, model_creation_config);

#ifdef ENABLE_VLLM_FLASH_ATTN_2
  set_torch_stream_layer_ = std::make_shared<SetTorchStreamLayer<T>>();
  set_torch_stream_layer_->Init({}, creation_context.context, creation_context.rank);
#endif

  // Init variable for QKVClaBufferCopy
  if (cla_share_factor_) {
    auto& model_config = model_creation_config.attn_config.model_config;
    int size_per_head = model_config.size_per_head;
    size_t tensor_para_size = model_creation_config.runtime_config.parallel_basic_config.tensor_parallel_size;
    int num_kv_heads_per_tp = model_config.num_key_value_heads / tensor_para_size;
    int head_num_per_tp = model_creation_config.attn_config.head_num_per_tp;
    qkv_pitch_ = (head_num_per_tp + num_kv_heads_per_tp * 2) * size_per_head * sizeof(T);
    q_pitch_ = head_num_per_tp * size_per_head * sizeof(T);
    kv_pitch_ = num_kv_heads_per_tp * size_per_head * sizeof(T);
  }
}

template <typename T>
Status CrossLayerAttention<T>::QKVClaBufferCopy(std::vector<Tensor>& hidden_buffer_tensors_0,
                                                std::vector<Tensor>& hidden_buffer_tensors_1,
                                                ForwardingContext<T>& forwarding_context) {
  if (cla_share_factor_ == 0) {
    return Status();
  }
  size_t total_tokens = hidden_buffer_tensors_0[0].shape[0];
  Tensor& cla_k_tensor = cla_buffers_.cla_k_buffer_[0];
  Tensor& cla_v_tensor = cla_buffers_.cla_v_buffer_[0];
  Tensor& hidden_tensor_0 = hidden_buffer_tensors_0[0];
  Tensor& hidden_tensor_1 = hidden_buffer_tensors_1[0];
  if (layer_idx_ % cla_share_factor_ == 0) {
    // qkv -> cla_k, cla_v
    Memcpy2DAsync(cla_k_tensor.GetPtr<void>(), kv_pitch_, hidden_tensor_0.GetPtr<void>() + q_pitch_, qkv_pitch_,
                  kv_pitch_, total_tokens, MEMCPY_DEVICE_TO_DEVICE,
                  forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
    Memcpy2DAsync(cla_v_tensor.GetPtr<void>(), kv_pitch_, hidden_tensor_0.GetPtr<void>() + q_pitch_ + kv_pitch_,
                  qkv_pitch_, kv_pitch_, total_tokens, MEMCPY_DEVICE_TO_DEVICE,
                  forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
  } else {
    // q, cla_k, cla_v -> qkv
    Memcpy2DAsync(hidden_tensor_1.GetPtr<void>(), qkv_pitch_, hidden_tensor_0.GetPtr<void>(), q_pitch_, q_pitch_,
                  total_tokens, MEMCPY_DEVICE_TO_DEVICE,
                  forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
    Memcpy2DAsync(hidden_tensor_1.GetPtr<void>() + q_pitch_, qkv_pitch_, cla_k_tensor.GetPtr<void>(), kv_pitch_,
                  kv_pitch_, total_tokens, MEMCPY_DEVICE_TO_DEVICE,
                  forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
    Memcpy2DAsync(hidden_tensor_1.GetPtr<void>() + q_pitch_ + kv_pitch_, qkv_pitch_, cla_v_tensor.GetPtr<void>(),
                  kv_pitch_, kv_pitch_, total_tokens, MEMCPY_DEVICE_TO_DEVICE,
                  forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
    hidden_tensor_1.shape = {total_tokens, qkv_pitch_ / sizeof(T)};
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  }
  return Status();
}

template <typename T>
Status CrossLayerAttention<T>::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                       std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                       ForwardingContext<T>& forwarding_context) {
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  std::vector<Tensor> empty_tensors;
  set_torch_stream_layer_->Forward(empty_tensors, empty_tensors);
#endif
  {
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
    attn_qkv_projs_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1);
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);

    // For cla
    QKVClaBufferCopy(hidden_buffer_tensors_0, hidden_buffer_tensors_1, forwarding_context);
  }
  attentions_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context);

#ifdef ENABLE_VLLM_FLASH_ATTN_2
  set_torch_stream_layer_->Clear();
#endif
  return Status();
}

template <typename T>
Status CrossLayerAttention<T>::CreateBuffers(BufferManager* buffer_mgr, const RuntimeConfig& runtime_config,
                                             const AttentionCreationConfig& attn_config, ClaBuffers& cla_buffers) {
  auto& model_config = attn_config.model_config;
  DataType weight_type = model_config.weight_data_type;

  size_t max_token_num = runtime_config.max_step_token_num;
  int size_per_head = model_config.size_per_head;
  size_t tensor_para_size = runtime_config.parallel_basic_config.tensor_parallel_size;
  int num_kv_heads_per_tp = model_config.num_key_value_heads / tensor_para_size;
  int cla_buffer_size = max_token_num * num_kv_heads_per_tp * size_per_head;

  // Init buffer to pass kv to next layer, buffers are used after creation
  TensorBuffer* cla_k_buffer =
      buffer_mgr->CreateBufferTensor("cla_k_buffer_", {static_cast<size_t>(cla_buffer_size)}, weight_type);
  cla_buffers.cla_k_buffer_ = cla_k_buffer->GetTensors();

  TensorBuffer* cla_v_buffer =
      buffer_mgr->CreateBufferTensor("cla_v_buffer_", {static_cast<size_t>(cla_buffer_size)}, weight_type);
  cla_buffers.cla_v_buffer_ = cla_v_buffer->GetTensors();

  return Status();
}

template class CrossLayerAttention<float>;
template class CrossLayerAttention<float16>;
template class CrossLayerAttention<bfloat16>;

}  // namespace ksana_llm
