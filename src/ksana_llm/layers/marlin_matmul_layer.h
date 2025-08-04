/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/base_weight.h"

namespace ksana_llm {

/**
 * @brief MarlinMatMulLayer
 *
 * @note 适用范围和限制：
 * - 仅支持NVIDIA GPU且SM架构 >= 80
 * - 矩阵计算的输出维度N最小为64
 * - 已支持GPTQ和AWQ的Int4量化
 * - Int8量化暂未开放
 * - 支持desc操作
 * - 目前只支持half的激活类型
 */
class MarlinMatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkSpaceSize() override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  template <typename T>
  size_t GetWorkSpaceSizeT();

  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  DataType weight_data_type_;

  bool is_awq_;
  bool is_gptq_desc_;
  // Whether the K-dimension of the weights is complete. If the weights split in the K dimension, is_k_full_ = false
  bool is_k_full_;

  size_t max_m_, max_n_, max_k_;
  size_t groupsize_;

  size_t marlin_workspace_size_;
  size_t marlin_input_tmp_size_;
  size_t marlin_output_tmp_size_;
  size_t marlin_workspace_offset_;
  size_t marlin_input_tmp_offset_;
  size_t marlin_output_tmp_offset_;
};

}  // namespace ksana_llm
