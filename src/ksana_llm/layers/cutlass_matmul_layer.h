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

class CutlassMatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  template <typename T>
  Status InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
               std::shared_ptr<Context> context, int rank);

  template <typename T>
  size_t GetWorkspaceSizeT();

  template <typename T>
  Status PreprocessT(const ModelConfig& model_config_, const RuntimeConfig& runtime_config);

  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  DataType weight_data_type_;

  bool is_awq_;

  size_t max_m_, max_n_, max_k_;
  size_t groupsize_;

  bool cutlass_use_gemv_cuda_core_;
  std::vector<size_t> cutlass_config_map_;
};

}  // namespace ksana_llm
