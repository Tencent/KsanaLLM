/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <mutex>
#include <unordered_map>

#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/layers/add_layer.h"
#include "ksana_llm/layers/assemble_tokens_hidden_layer.h"
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/cast_layer.h"
#include "ksana_llm/layers/emb_lookup_layer.h"
#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/layers/input_refit_layer.h"
#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/layers/matmul_layer_factory.h"
#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/layers/silu_mul_layer.h"
#include "ksana_llm/runtime/infer_stage.h"
#ifdef ENABLE_VLLM_FLASH_ATTN_2
#  include "ksana_llm/layers/set_torch_stream_layer.h"
#endif
#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/models/base/model_communicator.h"
#include "ksana_llm/models/base/model_input.h"
#include "ksana_llm/models/base/model_output.h"
#include "ksana_llm/models/common_moe/moe_config.h"
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

  // The word embedding scale factor.
  float emb_scale = 1.f;

  // Use pre-norm or post-norm.
  LayerNormPosition layernorm_position = LayerNormPosition::PRE_NORM;

  // If prepare return hidden states
  bool return_hidden_states = false;

  // If use rotary_embedding_pos for embedding lookup
  bool emb_lookup_use_rotary_embedding_pos = false;
};

// A common implement of transformer based model.
template <typename T>
class CommonModel : public BaseModel {
 public:
  CommonModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
              std::shared_ptr<Context> context);
  ~CommonModel() override;

  // Initialize the run config.
  void InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight);

  float* GetLogitsPtr(size_t multi_batch_id) override;

  // refer
  // github huggingface/transformers main/src/transformers/models/llama/modeling_llama.py#L942
  Status Forward(size_t multi_batch_id, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                 std::vector<ForwardRequest>& forward_reqs, bool epilogue,
                 const RunMode run_mode = RunMode::kMain) override;

  Status AllocResources(size_t multi_batch_id);
  Status FreeResources(size_t multi_batch_id);

  // Update response. Stop inference when the return value is true.
  bool UpdateResponse(std::vector<ForwardRequest>& forward_reqs, Tensor& output, const std::string& stage);

 private:
  virtual Status CreateLayers(LayerCreationContext<T>& creation_context,
                              ModelCreationConfig& model_creation_config) = 0;

 private:
  // Execute the embedding lookup.
  Status LookupEmbedding(ForwardingContext<T>& forwarding_context, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode = RunMode::kMain);

  // Execute the forward of specific layers.
  virtual Status LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode = RunMode::kMain) = 0;

  // Execute the lm head, and generate the logits.
  Status LmHead(ForwardingContext<T>& forwarding_context, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                std::vector<ForwardRequest>& forward_reqs, RunMode run_mode);

  // Get a reference for hidden buffer.
  std::vector<Tensor>& GetHiddenUnitBufferRef(ForwardingContext<T>& forwarding_context);

 protected:
  // Get hidden state from previous pipeline block
  std::vector<Tensor>& GetHiddenUnitBuffer(ForwardingContext<T>& forwarding_context, bool do_recv);

  // Set hidden state, it will be send  to next pipeline block
  void SetHiddenUnitBuffer(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context);

  ForwardingContext<T>* GetForwardingContext(size_t multi_batch_id);

 public:
  using BaseModel::context_;
  using BaseModel::rank_;

  // Whether auto prefix caching is enabled.
  bool prefix_caching_enabled_;

  // Check if speculative decoding is enabled
  bool speculative_decoding_enabled_;

  // The model config.
  ModelConfig model_config_;

  RuntimeConfig runtime_config_;

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
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  std::shared_ptr<BaseLayer> set_torch_stream_layer_;
#endif

  std::shared_ptr<Linear<T>> lm_head_;
  std::shared_ptr<Layernorm<T>> lm_head_prenorm_{nullptr};

  // The layer number of the model on current node.
  int layer_num_on_node_;

  // TODO(robertyuan): layer_creation_context_ should be deleted after layer creation.
  // However, matmul_layer_factory will delete the buffer during destroying.
  // Fix this after CommonModel is deleted.
  LayerCreationContext<T> layer_creation_context_;

  ModelBuffers model_buffers_;
  // Buffer of forwarding contexts for parallel batch processing
  std::vector<std::unique_ptr<ForwardingContext<T>>> forwarding_context_buffer_;
  // Map from multi_batch_id to index in the forwarding_context_buffer_
  std::unordered_map<size_t, size_t> schedule_to_context_map_;
  // Mutex to protect access to the buffer and map
  std::mutex forwarding_context_mutex_;

  // Be a replacement of residual_buffer_, for distributed mode only.
  std::vector<Tensor> distributed_device_buffer_;
  std::vector<Tensor> distributed_device_buffer_prefill_;

  Tensor cpu_input_tokens_tensor_;
  Tensor cpu_tokens_emb_tensor_;

  std::shared_ptr<Tensor> shared_matmul_workspace_buffer_ = nullptr;

  // Only used for QWenVL
  Tensor mrotary_section_tensor_;

 protected:
  bool IsPrefixCachingComputationReuse();

  Status EmbedTokensUseCpu(Tensor& embedding_weight, std::vector<ForwardRequest>& forward_reqs,
                           ForwardingContext<T>& forwarding_context);

  virtual Status EmbedTokensUseGpu(Tensor& embedding_weight, ForwardingContext<T>& forwarding_context);
};

}  // namespace ksana_llm
