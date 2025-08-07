/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/attention_layer.h"  // Only for PositionEncoding, try to remove later
#include "ksana_llm/layers/layer_workspace_manager.h"
#include "ksana_llm/layers/matmul_layer_factory.h"
#include "ksana_llm/layers/moe_layer_factory.h"
#include "ksana_llm/models/base/model_input.h"

namespace ksana_llm {

struct LayerCreationContext {
  std::shared_ptr<BaseWeight> base_weight;
  std::shared_ptr<MatMulLayerFactory> matmul_layer_factory;
  std::shared_ptr<MoeLayerFactory> moe_layer_factory;
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr;

  std::shared_ptr<Context> context;
  ModelConfig model_config;
  RuntimeConfig runtime_config;

  BufferManager* buffer_mgr_;

  int rank;
  PipelineConfig pipeline_config;

  DataType weight_type;
  DataType input_type;
  DataType output_type;

  void Init(std::shared_ptr<BaseWeight> base_weight_, std::shared_ptr<Context> context_, int rank_,
            PipelineConfig& pipeline_config_, ModelConfig& model_config, const RuntimeConfig& runtime_config,
            BufferManager* buffer_mgr);
};

// TODO(robertyuan): clear useless params
struct AttentionCreationConfig {
  int layer_num_on_node;  // TODO(robertyuan): this param is strange, try to remove later
  ModelConfig model_config;
  int max_position_embeddings;
  int head_num_per_tp;
  int num_kv_heads_per_tp;
  int size_per_head;
  int stride_size;
  bool use_qk_norm;
  size_t tensor_para_size;
  size_t data_para_size;
  DataType kv_cache_dtype;  // kv cache data type
  int rotary_embedding;
  float rope_theta;
  PositionEncoding position_encoding;
  std::any cos_sin_cache_ptr;
  size_t max_batch_size;
  size_t max_decode_tokens_per_req;

  // Caution: idx is an offset of total layer num in current node, not same as layer_id in distributed mode.
  int idx;  // TODO(robertyuan): this param is strange, try to remove later

  // Only for flash attention. need to remove later
  const int* mrope_section_ptr;
  bool reuse_prefix_caching;
};

struct LayernormCreationConfig {
  float layernorm_eps;
  std::string activation_function;
};

// Temp, remove later
struct ModelCreationConfig {
  AttentionCreationConfig attn_config;
  LayernormCreationConfig layernorm_config;

  void Init(const ModelConfig& model_config_, const RuntimeConfig& runtime_config, Tensor cos_sin_cache_tensor_,
            PositionEncoding position_encoding, bool reuse_prefix_caching, int layer_num_on_node_,
            const int* mrope_section_ptr);
};

// Temp, remove later
struct AttentionForwardContext {
  Tensor forward_shape;
  Tensor flag_tensor;
};

}  // namespace ksana_llm
