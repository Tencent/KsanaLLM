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

// Init template
// TODO(robertyuan): move to BaseLayer after all layers modified
#define LAYER_InitT(inter_data_type, parameters, runtime_config, context, rank)             \
  switch (inter_data_type) {                                                                \
    case DataType::TYPE_FP16:                                                               \
      return InitT<float16>(parameters, runtime_config, context, rank);                     \
    case DataType::TYPE_BF16:                                                               \
      return InitT<bfloat16>(parameters, runtime_config, context, rank);                    \
    case DataType::TYPE_FP32:                                                               \
      return InitT<float>(parameters, runtime_config, context, rank);                       \
    default:                                                                                \
      KLLM_THROW(fmt::format("Preprocess: Unsupported Tensor type: {}.", inter_data_type)); \
  }

// Preprocess template
#define LAYER_PreprocessT(inter_data_type, model_config, runtime_config)                    \
  switch (inter_data_type) {                                                                \
    case DataType::TYPE_FP16:                                                               \
      return PreprocessT<float16>(model_config, runtime_config);                            \
    case DataType::TYPE_BF16:                                                               \
      return PreprocessT<bfloat16>(model_config, runtime_config);                           \
    case DataType::TYPE_FP32:                                                               \
      return PreprocessT<float>(model_config, runtime_config);                              \
    default:                                                                                \
      KLLM_THROW(fmt::format("Preprocess: Unsupported Tensor type: {}.", inter_data_type)); \
  }

// GetWorkSpaceSize template
#define LAYER_GetWorkSpaceSizeT(inter_data_type)                                                  \
  switch (inter_data_type) {                                                                      \
    case DataType::TYPE_FP16:                                                                     \
      return GetWorkSpaceSizeT<float16>();                                                        \
    case DataType::TYPE_BF16:                                                                     \
      return GetWorkSpaceSizeT<bfloat16>();                                                       \
    case DataType::TYPE_FP32:                                                                     \
      return GetWorkSpaceSizeT<float>();                                                          \
    default:                                                                                      \
      KLLM_THROW(fmt::format("GetWorkSpaceSize: Unsupported Tensor type: {}.", inter_data_type)); \
  }

// Forward template
#define LAYER_ForwardT(inter_data_type, input_tensors, output_tensors)                   \
  switch (inter_data_type) {                                                             \
    case DataType::TYPE_FP16:                                                            \
      return ForwardT<float16>(input_tensors, output_tensors);                           \
    case DataType::TYPE_BF16:                                                            \
      return ForwardT<bfloat16>(input_tensors, output_tensors);                          \
    case DataType::TYPE_FP32:                                                            \
      return ForwardT<float>(input_tensors, output_tensors);                             \
    default:                                                                             \
      KLLM_THROW(fmt::format("Forward: Unsupported Tensor type: {}.", inter_data_type)); \
  }

// Init template
#define LAYER_InitT_WO_float(inter_data_type, parameters, runtime_config, context, rank)    \
  switch (inter_data_type) {                                                                \
    case DataType::TYPE_FP16:                                                               \
      return InitT<float16>(parameters, runtime_config, context, rank);                     \
    case DataType::TYPE_BF16:                                                               \
      return InitT<bfloat16>(parameters, runtime_config, context, rank);                    \
    default:                                                                                \
      KLLM_THROW(fmt::format("Preprocess: Unsupported Tensor type: {}.", inter_data_type)); \
  }

// Preprocess template
#define LAYER_PreprocessT_WO_float(inter_data_type, model_config, runtime_config)           \
  switch (inter_data_type) {                                                                \
    case DataType::TYPE_FP16:                                                               \
      return PreprocessT<float16>(model_config, runtime_config);                            \
    case DataType::TYPE_BF16:                                                               \
      return PreprocessT<bfloat16>(model_config, runtime_config);                           \
    default:                                                                                \
      KLLM_THROW(fmt::format("Preprocess: Unsupported Tensor type: {}.", inter_data_type)); \
  }

// GetWorkSpaceSize template
#define LAYER_GetWorkSpaceSizeT_WO_float(inter_data_type)                                         \
  switch (inter_data_type) {                                                                      \
    case DataType::TYPE_FP16:                                                                     \
      return GetWorkSpaceSizeT<float16>();                                                        \
    case DataType::TYPE_BF16:                                                                     \
      return GetWorkSpaceSizeT<bfloat16>();                                                       \
    default:                                                                                      \
      KLLM_THROW(fmt::format("GetWorkSpaceSize: Unsupported Tensor type: {}.", inter_data_type)); \
  }

// Forward template
#define LAYER_ForwardT_WO_float(inter_data_type, input_tensors, output_tensors)          \
  switch (inter_data_type) {                                                             \
    case DataType::TYPE_FP16:                                                            \
      return ForwardT<float16>(input_tensors, output_tensors);                           \
    case DataType::TYPE_BF16:                                                            \
      return ForwardT<bfloat16>(input_tensors, output_tensors);                          \
    default:                                                                             \
      KLLM_THROW(fmt::format("Forward: Unsupported Tensor type: {}.", inter_data_type)); \
  }

}  // namespace ksana_llm
