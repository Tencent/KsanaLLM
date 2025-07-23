/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>
#ifdef ENABLE_CUDA
#  include "ksana_llm/layers/custom_all_reduce_sum_layer.h"
#  include "ksana_llm/layers/nccl_all_gather_layer.h"
#  include "ksana_llm/layers/nccl_all_reduce_sum_layer.h"
#elif defined(ENABLE_ACL)
#  include "ksana_llm/layers/hccl_all_gather_layer.h"
#  include "ksana_llm/layers/hccl_all_reduce_sum_layer.h"
#elif defined(ENABLE_TOPS)
#  include "ksana_llm/layers/eccl_all_gather_layer.h"
#  include "ksana_llm/layers/eccl_all_reduce_sum_layer.h"
#endif

namespace ksana_llm {

// The collective communicator library.
template <typename T>
class ModelCommunicator {
 public:
  ModelCommunicator(Tensor* buffer, Tensor* input, int rank, const RuntimeConfig& runtime_config,
                    std::shared_ptr<Context> context);
  ~ModelCommunicator();

  // The all-gather reduce.
  Status AllGather(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  // The reduce-sum reduce.
  Status ReduceSum(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                   bool is_multi_token_forward, bool use_custom);

 private:
#ifdef ENABLE_CUDA
  // The default all reduce layer.
  std::shared_ptr<NcclAllReduceSumLayer<T>> nccl_all_reduce_sum_layer_;

  // The all gather layer
  std::shared_ptr<NcclAllGatherLayer<T>> nccl_all_gather_layer_;

  // The custom all reduce layer.
  std::shared_ptr<CustomAllReduceSumLayer<T>> tp_custom_all_reduce_sum_layer_;

#elif defined(ENABLE_ACL)
  // The default all reduce layer.
  std::shared_ptr<HcclAllReduceSumLayer<T>> hccl_all_reduce_sum_layer_;

  // The all gather layer
  std::shared_ptr<HcclAllGatherLayer<T>> hccl_all_gather_layer_;
#endif

 private:
  int rank_;
  std::shared_ptr<Context> context_;

  // For custom all reduce layer.
  Tensor tp_signal_tensor_;
  Tensor tp_custom_all_reduce_rank_tensor_;

  // Use for custom all reduce layer.
  Tensor* buffer_;
  Tensor* input_;

  // Whether the communication is finished.
  Event comm_finish_event_;

  uint32_t tp_size_ = 1;

  bool is_full_nvlink_ = false;

  bool select_all_reduce_by_size_ = false;

  static constexpr int kAllReduceThreshold = 8 * 1024 * 1024;  // 8MB

 private:
  bool CheckIfUseCustomReduceSum(const std::vector<Tensor>& input_tensors, bool use_custom);

#ifdef ENABLE_CUDA
  void InitTensorParaCustomAllReduceSumLayer(Tensor* input, const RuntimeConfig& runtime_config);
#endif
};

}  // namespace ksana_llm
