/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/layer_creation_context.h"

#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
void LayerCreationContext<T>::Init(std::shared_ptr<BaseWeight> base_weight_,
                                   std::shared_ptr<Tensor>& shared_matmul_workspace_buffer_,
                                   std::shared_ptr<Context> context_, int rank_, PipelineConfig& pipeline_config_,
                                   ModelConfig& model_config_, BufferManager* buffer_mgr) {
  base_weight = base_weight_;
  matmul_layer_factory =
      std::make_shared<MatMulLayerFactory<T>>(shared_matmul_workspace_buffer_, model_config_, rank_, context_);

  context = context_;
  rank = rank_;
  weight_type = model_config_.weight_data_type;
  input_type = model_config_.weight_data_type;
  output_type = model_config_.weight_data_type;
  pipeline_config = pipeline_config_;
  buffer_mgr_ = buffer_mgr;
}

template class LayerCreationContext<float>;
template class LayerCreationContext<float16>;
#ifdef ENABLE_BFLOAT16
template class LayerCreationContext<bfloat16>;
#endif

void ModelCreationConfig::Init(const ModelConfig& model_config_, Tensor cos_sin_cache_tensor_,
                               PositionEncoding position_encoding, bool reuse_prefix_caching, int layer_num_on_node_,
                               const int* mrope_section_ptr) {
  auto env = Singleton<Environment>::GetInstance();
  const int size_per_head = model_config_.size_per_head;
  const int head_num_per_tp = model_config_.head_num / env->GetAttentionTensorParallel();
  const int num_kv_heads_per_tp = model_config_.num_key_value_heads / env->GetAttentionTensorParallel();
  BatchSchedulerConfig batch_scheduler_config;
  env->GetBatchSchedulerConfig(batch_scheduler_config);

  layernorm_config.layernorm_eps = model_config_.layernorm_eps;

  attn_config.layer_num_on_node = layer_num_on_node_;
  attn_config.model_config = model_config_;
  attn_config.max_position_embeddings = model_config_.max_position_embeddings;
  attn_config.head_num_per_tp = head_num_per_tp;
  attn_config.num_kv_heads_per_tp = num_kv_heads_per_tp;
  attn_config.size_per_head = size_per_head;
  attn_config.stride_size = (head_num_per_tp + num_kv_heads_per_tp * 2) * size_per_head;
  attn_config.tensor_para_size = model_config_.tensor_para_size;
  attn_config.data_para_size = model_config_.attn_data_para_size;
  attn_config.data_type = model_config_.weight_data_type;
  attn_config.rotary_embedding = model_config_.rotary_embedding;
  attn_config.rope_theta = model_config_.rope_theta;
  attn_config.position_encoding = position_encoding;
  attn_config.cos_sin_cache_ptr = std::any(cos_sin_cache_tensor_.GetPtr<void>());
  attn_config.max_batch_size = model_config_.max_batch_size;
  attn_config.max_decode_tokens_per_req = batch_scheduler_config.max_decode_tokens_per_req;
  attn_config.use_qk_norm = model_config_.use_qk_norm;
  attn_config.mrope_section_ptr = mrope_section_ptr;
  attn_config.reuse_prefix_caching = reuse_prefix_caching;
  attn_config.model_config.enable_qk_pre_norm_before_rotary_pos = model_config_.enable_qk_pre_norm_before_rotary_pos;
  attn_config.model_config.enable_add_qkv_bias = model_config_.enable_add_qkv_bias;
}

}  // namespace ksana_llm
