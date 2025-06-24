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
  /**
   * The following 4 buffers are used as temporary buffers during the whole model inference:
   * 1. `hidden_buffer_0_` and `hidden_buffer_1_` serve as the input and output for each layer.
   *    We assume that the input of each layer is taken from `hidden_buffer_0_`, the output is
   *    put into `hidden_buffer_1_`, and then swapped with `hidden_buffer_0_`. This convention
   *    makes each layer independent and pluggable.
   * 3. `shared_buffer_` is shared to store the output of the up layer for gated activation
   *    (`gated_buffer_`), as the fixed input buffer for custom reduce sum (`reduce_buffer_`),
   *    and as the extra workspace for paged attention (`paged_buffer_`).
   */
  TensorBuffer* hidden_buffer_0;
  TensorBuffer* hidden_buffer_1;
  TensorBuffer* shared_buffer;
  TensorBuffer* kv_cache_buffer;

  // Use for dp only.
  TensorBuffer* dp_input_buffer;

  std::vector<Tensor> mtp_hidden_buffer_tensors;  // This buffer is used among multiple forward calls

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

// Too many obj, try to clear later
template <typename T>
class ForwardingContext {
 public:
  ~ForwardingContext() {}
  void Init(std::shared_ptr<Context> context, int rank, const ModelConfig& model_config,
            const PipelineConfig& pipeline_config, ForwardingBuffers* buffers, BufferManager* buffer_mgr,
            size_t pp_batch_idx);

  void UpdateBeforeForward(std::vector<ForwardRequest>& forward_reqs, RunMode run_mode);

  void UpdateAfterForward(std::vector<ForwardRequest>& forward_reqs);

 public:
  size_t schedule_id = DEFAULT_SCHEDULE_ID;
  ForwardingBuffers* buffers_;

  /*
   * These variables must be set through functions, do not set them
   * TODO(robertyuan): change to Get functions
   */
  AttentionForwardContext attn_ctx_;

  std::shared_ptr<Context> context_;
  int rank_;
  PipelineConfig pipeline_config_;

  // The model input information.
  std::shared_ptr<ModelInput> model_input_;

  // The model output.
  std::shared_ptr<ModelOutput> model_output_;

  // The model communicator.
  std::shared_ptr<ModelCommunicator<T>> model_communicator_;

  // mark state for sched event recording
  bool is_forwarding_layers = false;

  // Used for tracing sched events.
  BatchRequestSchedInfo batch_event_info;

  // fwd context related pp batch idx.
  size_t pp_batch_idx = 0;

  // Attention data parallel size
  size_t attn_data_parallel_size_ = 1;

 private:
  // The original vocab size of the model
  size_t vocab_size_;

  // Vocab size aligned and padded with tensor_para_size
  size_t vocab_size_pad_;
};
}  // namespace ksana_llm
