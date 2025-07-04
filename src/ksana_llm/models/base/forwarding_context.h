/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "ksana_llm/models/base/layer_creation_context.h"
#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/utils/tensor.h"

#include "ksana_llm/models/base/model_communicator.h"
#include "ksana_llm/models/base/model_input.h"
#include "ksana_llm/models/base/model_output.h"
#include "ksana_llm/profiler/sched_event_tracer.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/runtime/cuda_graph_runner.h"
#endif

namespace ksana_llm {

struct ForwardingBuffers {
  //  NOTE(karlluo): The following 4 buffers(3 different kinds) are used as temporary buffers during the whole model
  //  inference:
  //  1. intermedia buffer: `hidden_buffer_0` and `hidden_buffer_1` serve as the input and output for each layer.
  //     We assume that the input of each layer is taken from `hidden_buffer_0`, the output is
  //     put into `hidden_buffer_1`, and then swapped with `hidden_buffer_0`. This convention
  //     makes each layer independent and pluggable.
  //  2. operators' extra buffer: `shared_buffer` is shared to store the output of the up layer for gated activation
  //     (`gated_buffer_`), as the fixed input buffer for custom reduce sum (`reduce_buffer_`),
  //     and as the extra workspace for paged attention (`paged_buffer_`).
  //  3. kv cache buffer: `kv_cache_buffer` stores the key-value pairs for all attention layers
  //     across different sequence positions. This enables efficient autoregressive generation
  //     by avoiding recomputation of previously calculated key-value pairs during inference.
  //     The buffer supports both paged attention and continuous memory layouts for optimal
  //     memory utilization and access patterns.
  TensorBuffer* hidden_buffer_0;
  TensorBuffer* hidden_buffer_1;
  TensorBuffer* shared_buffer;
  TensorBuffer* kv_cache_buffer;

  TensorBuffer* dp_input_buffer;

  // This buffer is used among multiple forward calls
  std::vector<Tensor> mtp_hidden_buffer_tensors;

  void Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config, bool use_mtp,
            BufferManager* buffer_mgr);

  void CalculateBuffersShape(size_t batch_size, size_t token_num);

  // Model config
  ModelConfig model_config;

  // Is use multi-token prediction
  bool use_mtp{false};

  // Map to record each buffers shape.
  std::unordered_map<std::string, std::vector<size_t>> buffers_shape_map;
};

struct ModelBuffers {
  std::unique_ptr<ForwardingBuffers> buffers_;

  std::vector<Tensor> local_residual_buffer_tensors_{1};
  Tensor cos_sin_cache_tensor_;

  void Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config, bool use_mtp,
            BufferManager* buffer_mgr);
};

template <typename T>
class ForwardingContext {
 public:
  ~ForwardingContext() {}
  void Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config,
            const PipelineConfig& pipeline_config, ForwardingBuffers* buffers, BufferManager* buffer_mgr,
            size_t multi_batch_id);

  void UpdateBeforeForward(std::vector<ForwardRequest>& forward_reqs, RunMode run_mode);

  void UpdateAfterForward(std::vector<ForwardRequest>& forward_reqs);

 public:
  ForwardingBuffers* GetForwardingBuffers() { return buffers_; }

  AttentionForwardContext& GetAttentionForwardContext() { return attn_ctx_; }

  inline const size_t GetAttentionDataParallelSize() { return attn_data_parallel_size_; }

  std::shared_ptr<ModelCommunicator<T>>& GetModelCommunicator() { return model_communicator_; }

  std::shared_ptr<ModelOutput>& GetModelOutput() { return model_output_; }

  std::shared_ptr<ModelInput>& GetModelInput() { return model_input_; }

  inline const size_t GetMultiBatchId() const { return multi_batch_id_; }

  inline void SetMultiBatchId(size_t multi_batch_id) { multi_batch_id_ = multi_batch_id; }

  inline BatchRequestSchedInfo& GetBatchRequestSchedInfo() { return batch_event_info_; }

  inline const bool IsForwardingLayers() { return is_forwarding_layers_; }

  inline void SetIsForwardingLayers(bool is_forwarding_layers) { is_forwarding_layers_ = is_forwarding_layers; }

  inline const std::shared_ptr<Context>& GetContext() { return context_; }

  inline void SetContext(std::shared_ptr<Context> context) { context_ = context; }

  inline int GetCurrentRank() { return rank_; }

  inline void SetCurrentRank(int rank) { rank_ = rank; }

  inline const PipelineConfig& GetPipelineConfig() { return pipeline_config_; }

 private:
  // Rank of current inference device
  int rank_;

  // Current inference context
  std::shared_ptr<Context> context_;

  // Current inference task related Multi-batch id
  size_t multi_batch_id_ = DEFAULT_MULTI_BATCH_ID;

  // The model input information.
  std::shared_ptr<ModelInput> model_input_;

  // The model output.
  std::shared_ptr<ModelOutput> model_output_;

  // Used for tracing sched events.
  BatchRequestSchedInfo batch_event_info_;

  // The model communicator.
  std::shared_ptr<ModelCommunicator<T>> model_communicator_;

  // Pipeline parallel configuration
  PipelineConfig pipeline_config_;

  // Attention data parallel size
  size_t attn_data_parallel_size_ = 1;

  // Extra attributes for attention
  AttentionForwardContext attn_ctx_;

  // Buffers for temporary buffers during the whole model inference
  ForwardingBuffers* buffers_;

  // The original vocab size of the model
  size_t vocab_size_;

  // Vocab size aligned and padded with tensor_para_size
  size_t vocab_size_pad_;

  // mark state for sched event recording
  bool is_forwarding_layers_ = false;
};
}  // namespace ksana_llm
