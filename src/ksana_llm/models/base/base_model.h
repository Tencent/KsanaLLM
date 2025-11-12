/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/base/buffer_manager.h"
#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/runtime/cuda_graph_runner.h"
#endif

namespace ksana_llm {

enum class RunMode {
  kMain,   // default
  kNextN,  // run MTP next n
};

class BaseModel {
 public:
  // Disable a default constructor
  BaseModel();

  virtual ~BaseModel();

  // Forward model.
  virtual Status Forward(size_t multi_batch_id, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         std::vector<ForwardRequest>& forward_reqs, bool epilogue,
                         RunMode run_mode = RunMode::kMain) = 0;

  // Manage resources for different multi_batch_id because Forward is invoked multiple times to serve a multi_batch_id
  virtual Status AllocResources(size_t multi_batch_id) = 0;
  virtual Status FreeResources(size_t multi_batch_id) = 0;

  // The output logits pointer on device, used by sampler to avoid memory copy.
  virtual float* GetLogitsPtr(size_t multi_batch_id) = 0;

  // The output tokens pointer on host when all requests use greedy sampler.
  virtual int* GetOutputTokensPtr(size_t multi_batch_id) = 0;

  // Implement this method if cuda graph is used.
  virtual Status WarmUpCudaGraph() { return Status(); }

  BufferManager* GetBufferManager() { return &buffer_mgr_; }

 protected:
  std::shared_ptr<Context> context_{nullptr};

  int rank_{0};

  BufferManager buffer_mgr_;

  // Whether cuda graph is enabled.
  bool enable_cuda_graph_ = true;

#ifdef ENABLE_CUDA
  // The cuda graph runner.
  CudaGraphRunner cuda_graph_runner_;
#endif

 protected:
  // Log total buffer tensors memory used
  const size_t GetBufferTensorsMemoryUsed() { return buffer_mgr_.GetBufferTensorsMemoryUsed(); }
};

}  // namespace ksana_llm
