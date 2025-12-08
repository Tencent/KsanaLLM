/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from [TensorRT-LLM Project]
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc4/cpp/tensorrt_llm/thop/finegrained_mixed_dtype_gemm_thop.h
 *
 */

#pragma once

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_extensions/gemm_configs.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_extensions/weight_only_quant_op.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

#include <fmt/format.h>
#include <torch/torch.h>

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/utils.h"
#include "csrc/utils/nvidia/assert.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels::nvidia::tensorrt_llm::dev {

namespace internal {
namespace kernels = llm_kernels::nvidia::tensorrt_llm::dev::kernels;
}  // namespace internal

torch::Tensor preprocess_weights_for_mixed_gemm(torch::Tensor tensor, torch::ScalarType quant_mode,
                                                torch::ScalarType act_dtype, int sm = -1,
                                                bool do_weight_interleave = true);

class FinegrainedMixedDtypeGemmRunner {
 public:
  explicit FinegrainedMixedDtypeGemmRunner(ScalarType activationDtype, ScalarType outputDtype, int64_t quant_mode = 0);

  void runGemm(cudaStream_t stream, Tensor& C_tensor, void* workspace_ptr, Tensor const& A, Tensor const& B_packed,
               Tensor const& scales, int64_t group_size_long, int64_t configIdx = -1,
               std::optional<Tensor> bias = std::nullopt, std::optional<Tensor> zeros = std::nullopt,
               float alpha = 1.0f) const;

  size_t getWorkspaceSize(const std::vector<size_t>& A_shape, const std::vector<size_t>& B_packed_shape) const;

  int64_t getNumConfigs() const;

 private:
  std::shared_ptr<internal::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface> mGemmRunner;
  std::vector<llm_kernels::nvidia::tensorrt_llm::dev::cutlass_extensions::CutlassGemmConfig> mConfigs;
  ScalarType mActivationDtype;
  ScalarType mOutputDtype;
};

}  // namespace llm_kernels::nvidia::tensorrt_llm::dev