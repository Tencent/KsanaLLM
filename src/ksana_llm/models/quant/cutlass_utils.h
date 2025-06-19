/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/nn/functional/normalization.h>

#include "ksana_llm/utils/context.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#endif

namespace ksana_llm {

class CutlassUtils {
 public:
  CutlassUtils(std::shared_ptr<Context> context, int rank, int bits) : context_(context), rank_(rank), bits_(bits) {}

#ifdef ENABLE_CUDA
  torch::Tensor CutlassGetReverseOrder(const torch::Tensor& iweights);

  torch::Tensor CutlassUnpackQWeight(const torch::Tensor& qtensor);

  torch::Tensor CutlassUnpackAWQ(const torch::Tensor& qweight);

  torch::Tensor CutlassUnpackGPTQ(const torch::Tensor& w_packed);

  torch::Tensor CutlassPackInt8ToPackedInt4(torch::Tensor weight);

  torch::Tensor CutlassPreprocessWeightsForMixedGemmWarpper(torch::Tensor row_major_quantized_weight,
                                                            llm_kernels::nvidia::QuantType quant_type);
#endif

 private:
  std::shared_ptr<Context> context_{nullptr};
  int rank_;
  int bits_;
};

}  // namespace ksana_llm
