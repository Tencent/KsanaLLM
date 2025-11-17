/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/layers/set_torch_stream_layer.h"
#endif
#include "ksana_llm/layers/add_layer.h"
#include "ksana_llm/layers/assemble_tokens_hidden_layer.h"
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/cast_layer.h"
#include "ksana_llm/layers/emb_lookup_layer.h"
#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/layers/greedy_sampler_layer.h"
#include "ksana_llm/layers/input_refit_layer.h"
#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/layers/matmul_layer_factory.h"
#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/layers/silu_mul_layer.h"
#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/models/base/model_communicator.h"
#include "ksana_llm/models/base/model_input.h"
#include "ksana_llm/models/base/model_output.h"
#include "ksana_llm/models/common_moe/moe_config.h"
#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/models/llama/llama_weight.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/utils.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/runtime/cuda_graph_runner.h"
#endif

#include "ksana_llm/models/base/forwarding_context.h"

#include "ksana_llm/modules/basic/layernorm.h"
#include "ksana_llm/modules/basic/linear.h"

namespace ksana_llm {

// The layernorm position type.
enum class LayerNormPosition { PRE_NORM = 0, POST_NORM = 1 };

// Describe the model architecture.
struct ModelRunConfig {
  // The model position embedding.
  PositionEncoding position_encoding = PositionEncoding::ROPE;

  // Use pre-norm or post-norm.
  LayerNormPosition layernorm_position = LayerNormPosition::PRE_NORM;

  // If use rotary_embedding_pos for embedding lookup
  bool emb_lookup_use_rotary_embedding_pos = false;

  // Whether word embedding uses emb_scale.
  bool use_emb_scale = false;
  // The word embedding scale factor.
  float emb_scale = 1.f;
  // Scaling the hidden states of residual connections.
  float scale_depth = 1.f;
};

// A common implement
class CommonModel : public BaseModel {
 public:
  CommonModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
              std::shared_ptr<Context> context);
  ~CommonModel() override{};

  // Initialize the run config.
  void InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight);

  float* GetLogitsPtr(size_t multi_batch_id) override;

  int* GetOutputTokensPtr(size_t multi_batch_id) override;

  // refer
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L942
  Status Forward(size_t multi_batch_id, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                 std::vector<ForwardRequest*>& forward_reqs, bool epilogue,
                 const RunMode run_mode = RunMode::kMain) override;

  Status AllocResources(size_t multi_batch_id);
  Status FreeResources(size_t multi_batch_id);

  // Update response. Stop inference when the return value is true.
  bool UpdateResponse(std::vector<ForwardRequest*>& forward_reqs, Tensor& output, const std::string& stage);

 private:
  virtual Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) = 0;

 private:
  // Execute the embedding lookup.
  Status LookupEmbedding(ForwardingContext& forwarding_context, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         std::vector<ForwardRequest*>& forward_reqs, const RunMode run_mode = RunMode::kMain);

  // Execute the forward of specific layers.
  virtual Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) = 0;

  // Execute the lm head, and generate the logits.
  virtual Status LmHead(ForwardingContext& forwarding_context, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                        std::vector<ForwardRequest*>& forward_reqs, RunMode run_mode);

  // Get a reference for hidden buffer.
  std::vector<Tensor>& GetHiddenUnitBufferRef(ForwardingContext& forwarding_context);

 protected:
  // Get hidden state from previous pipeline block
  std::vector<Tensor>& GetHiddenUnitBuffer(ForwardingContext& forwarding_context, bool do_recv);

  // Set hidden state, it will be send  to next pipeline block
  void SetHiddenUnitBuffer(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context);

  ForwardingContext* GetForwardingContext(size_t multi_batch_id);

 public:
  using BaseModel::context_;
  using BaseModel::rank_;

  // Whether auto prefix caching is enabled.
  bool prefix_caching_enabled_;

  // Check if speculative decoding is enabled
  bool speculative_decoding_enabled_;

  // The model config.
  const ModelConfig model_config_;

  const RuntimeConfig runtime_config_;

  // The pipeline_config for distributed mode.
  PipelineConfig pipeline_config_;

  // The expert parallel config for multi nodes.
  ExpertParallelConfig expert_parallel_config_;
  // The model run config.
  ModelRunConfig model_run_config_;

  std::shared_ptr<BaseLayer> emb_lookup_layer_;
  std::shared_ptr<BaseLayer> cpu_emb_lookup_layer_;

  std::shared_ptr<BaseLayer> assemble_tokens_hidden_layer_;
  std::shared_ptr<BaseLayer> cast_layer_;
  std::shared_ptr<BaseLayer> input_refit_layer_;

#ifdef ENABLE_CUDA
  std::shared_ptr<BaseLayer> set_torch_stream_layer_;
#endif

  std::shared_ptr<Linear> lm_head_;
  std::shared_ptr<Layernorm> lm_head_prenorm_{nullptr};

  // The layer for fast greedy sampler
  std::unique_ptr<GreedySamplerLayer> greedy_sampler_layer_;

  // The layer number of the model on current node.
  int layer_num_on_node_;

  // TODO(robertyuan): layer_creation_context_ should be deleted after layer creation.
  // However, matmul_layer_factory will delete the buffer during destroying.
  // Fix this after CommonModel is deleted.
  LayerCreationContext layer_creation_context_;

  ModelBuffers model_buffers_;
  // Buffer of forwarding contexts for parallel batch processing
  std::vector<std::unique_ptr<ForwardingContext>> forwarding_context_buffer_;
  size_t forwarding_context_buffer_size_;

  // Be a replacement of residual_buffer_, for distributed mode only.
  std::vector<Tensor> distributed_device_buffer_;
  std::vector<Tensor> distributed_device_buffer_prefill_;

  Tensor cpu_input_tokens_tensor_;
  Tensor cpu_tokens_emb_tensor_;

  // Only used for QWenVL
  Tensor mrotary_section_tensor_;

 protected:
  bool IsPrefixCachingComputationReuse();

  Status EmbedTokensUseCpu(Tensor& embedding_weight, std::vector<ForwardRequest*>& forward_reqs,
                           ForwardingContext& forwarding_context);

  virtual Status EmbedTokensUseGpu(Tensor& embedding_weight, ForwardingContext& forwarding_context);
};

}  // namespace ksana_llm
