/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) {
    context_ = context;
    rank_ = rank;
    inter_data_type_ = runtime_config.inter_data_type;

    tp_size_ = context_->GetTensorParallelSize();
    dp_size_ = runtime_config.parallel_basic_config.attn_data_parallel_size;
    attn_dp_atp_size_ = runtime_config.parallel_basic_config.attn_tensor_parallel_size;
    if (attn_dp_atp_size_ == 0) {
      attn_dp_atp_size_ = 1;
    }
    attn_dp_group_id_ = rank_ / attn_dp_atp_size_;
    attn_dp_rank_id_ = rank_ % attn_dp_atp_size_;

    return Status();
  }

  virtual size_t GetWorkSpaceSize() { return 0; }

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) = 0;

  virtual Status SetWorkSpaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) {
    workspace_buffer_ = workspace_buffer;
    return Status();
  }

  virtual Status Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) { return Status(); }

  virtual void Clear() {}

 protected:
  DataType inter_data_type_;
  int rank_;
  std::shared_ptr<Context> context_;
  std::shared_ptr<Tensor> workspace_buffer_;

  // For Attention data parallel.
  int tp_size_;
  int dp_size_;
  int attn_dp_atp_size_;
  int attn_dp_group_id_;
  int attn_dp_rank_id_;
};

// Dispatch function by float16/bfloat16/float
#define DISPATCH_BY_3_DTYPE(dtype, func, ...)                                                 \
  switch (dtype) {                                                                            \
    case DataType::TYPE_FP16:                                                                 \
      return func<float16>(__VA_ARGS__);                                                      \
    case DataType::TYPE_BF16:                                                                 \
      return func<bfloat16>(__VA_ARGS__);                                                     \
    case DataType::TYPE_FP32:                                                                 \
      return func<float>(__VA_ARGS__);                                                        \
    default:                                                                                  \
      KLLM_THROW(fmt::format("{}: Unsupported Dtype type: {}.", __PRETTY_FUNCTION__, dtype)); \
  }

// Dispatch function by float16/bfloat16
#define DISPATCH_BY_2_DTYPE(dtype, func, ...)                                                 \
  switch (dtype) {                                                                            \
    case DataType::TYPE_FP16:                                                                 \
      return func<float16>(__VA_ARGS__);                                                      \
    case DataType::TYPE_BF16:                                                                 \
      return func<bfloat16>(__VA_ARGS__);                                                     \
    default:                                                                                  \
      KLLM_THROW(fmt::format("{}: Unsupported Dtype type: {}.", __PRETTY_FUNCTION__, dtype)); \
  }

}  // namespace ksana_llm
