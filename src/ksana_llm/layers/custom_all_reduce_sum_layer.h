/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/utils/barrier.h"

namespace ksana_llm {

class CustomAllReduceSumLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

  void Clear();

  void ResetInputBuffer(void* input);
  void ResetSignalBuffer(void* signal, size_t signal_sz);

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  void** signals_;
  void* buffer_;
  void* rank_data_;
  size_t rank_data_sz_;
  void** data_handles_;
  void** input_handles_;
  void* reduce_op_{nullptr};
  bool is_init_ = false;
  bool is_full_nvlink_ = true;
  bool need_sync_;
  uint32_t root_rank_{0};
  uint32_t world_size_{1};

  // NOTE(karlluo): For attention data parallelism, we do all reduce as group allreduce: just do allreduce with between
  // some gpus. The root rank is the first rank of the attention data parallel group. For example, if the rank is 0, 1,
  // 2, 3, and the attention data parallel size is 2, the root rank is 0. If the rank is 4, 5, 6, 7, and the attention
  // data parallel size is 2, the root rank is 4. The root rank is used to determine the group of ranks that will
  // perform the all-reduce operation. The root rank is the first rank of the attention data parallel group.
  bool is_group_custom_all_reduce_{false};
};

}  // namespace ksana_llm
