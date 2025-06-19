/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/base_weight.h"

namespace ksana_llm {

/**
 * @brief CutlassMatMulLayer
 *
 * @note 适用范围和限制：
 * - 仅支持NVIDIA GPU且SM架构 >= 80
 * - 矩阵计算的输出维度N最小为64
 * - 已支持GPTQ和AWQ的Int4量化
 * - Int8量化暂未开放
 * - 不支持desc操作
 * - group size只支持64和128
 * - 在M小于5时，支持速度更快的cuda gemv计算
 * - 支持half和bfloat16的激活类型
 */
template <typename T, DataType WT>
class CutlassMatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkSpaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config_) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  bool is_awq_;

  size_t max_m_, max_n_, max_k_;
  size_t groupsize_;

  bool cutlass_use_gemv_cuda_core_;
  std::vector<size_t> cutlass_config_map_;
};

}  // namespace ksana_llm
