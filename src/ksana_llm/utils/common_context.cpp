/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/common_context.h"

#include <iostream>
#include <stdexcept>

#include "fmt/core.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <int T>
ContextT<T>::ContextT(const size_t tensor_parallel_size, const size_t attn_data_parallel_size,
                      const size_t max_multi_batch_size)
    : tensor_parallel_size_(tensor_parallel_size), attn_data_parallel_size_(attn_data_parallel_size) {
  int device_num;
  GetDeviceCount(&device_num);
  KLLM_CHECK_WITH_INFO(
      device_num >= static_cast<int>(tensor_parallel_size_),
      fmt::format("{} tensor_parallel_size should not bigger than devices num: {}", tensor_parallel_size_, device_num));

  memory_manage_streams_.reserve(tensor_parallel_size_);

  max_multi_batch_size_ = max_multi_batch_size;
  KLLM_CHECK_WITH_INFO(max_multi_batch_size > 0, "max_multi_batch_size should be bigger than 0");

  comm_nodes_streams_.reserve(tensor_parallel_size_);
  compute_streams_.reserve(tensor_parallel_size_);
  h2d_streams_.reserve(tensor_parallel_size_);
  d2h_streams_.reserve(tensor_parallel_size_);
  d2d_streams_.reserve(tensor_parallel_size_);
  comm_streams_.reserve(tensor_parallel_size_);
  for (size_t worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    InitStreams(worker_id);
  }

  // Initialize pipeline configure.
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config_);
  is_chief_ = pipeline_config_.world_size == 1 || pipeline_config_.node_rank == 0;
  is_standalone_ = pipeline_config_.world_size == 1;

  // Initialize expert parallel configure.  Make sure ep share is_chief_ with pp
  // is ok later.
  Singleton<Environment>::GetInstance()->GetExpertParallelConfig(expert_parallel_config_);
  is_expert_chief_ = expert_parallel_config_.expert_world_size == 1 || expert_parallel_config_.expert_node_rank == 0;
  is_expert_standalone_ = expert_parallel_config_.expert_world_size == 1;

  // Initialize the device extension.
  InitializeExtension();
}

template <int T>
ContextT<T>::~ContextT() {
  DestroyExtension();

  for (size_t worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    memory_manage_streams_[worker_id].Destroy();
    compute_streams_[worker_id].Destroy();
    comm_nodes_streams_[worker_id].Destroy();
    h2d_streams_[worker_id].Destroy();
    d2h_streams_[worker_id].Destroy();
    d2d_streams_[worker_id].Destroy();
    comm_streams_[worker_id].Destroy();
  }

  memory_manage_streams_.clear();
  compute_streams_.clear();
  h2d_streams_.clear();
  d2h_streams_.clear();
  d2d_streams_.clear();
  comm_streams_.clear();
  comm_nodes_streams_.clear();
}

template <int T>
void ContextT<T>::InitStreams(const int worker_id) {
  memory_manage_streams_.emplace_back(worker_id);
  compute_streams_.emplace_back(worker_id);
  comm_nodes_streams_.emplace_back(worker_id);
  h2d_streams_.emplace_back(worker_id);
  d2h_streams_.emplace_back(worker_id);
  d2d_streams_.emplace_back(worker_id);
  comm_streams_.emplace_back(worker_id);
}

template <int T>
bool ContextT<T>::IsStandalone() const {
  return is_standalone_;
}

template <int T>
bool ContextT<T>::IsChief() const {
  return is_chief_;
}

template <int T>
bool ContextT<T>::IsExpertParallelStandalone() const {
  return is_expert_standalone_;
}

// Master node of expert parallel clusters.
template <int T>
bool ContextT<T>::IsExpertParallelChief() const {
  return is_expert_chief_;
}

template class ContextT<ACTIVE_DEVICE_TYPE>;

}  // namespace ksana_llm
