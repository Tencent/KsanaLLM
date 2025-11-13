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
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc4/cpp/tensorrt_llm/thop/finegrained_mixed_dtype_gemm_thop.cpp
 *
 */

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/finegrained_mixed_dtype_gemm_thop.h"

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/common/cudaUtils.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_extensions/gemm_configs.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_extensions/weight_only_quant_op.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/cutlass_heuristic.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace llm_kernels::nvidia::tensorrt_llm::dev {

FinegrainedMixedDtypeGemmRunner::FinegrainedMixedDtypeGemmRunner(ScalarType activationDtype, ScalarType outputDtype,
                                                                 int64_t quant_mode)
    : mActivationDtype(activationDtype), mOutputDtype(outputDtype) {
  if (quant_mode == 0) {
    if (activationDtype == ScalarType::Float16) {
      KLLM_KERNEL_CHECK_WITH_INFO(outputDtype == activationDtype,
                                  fmt::format("Activation dtype {} needs to match Output stype", activationDtype));
      mGemmRunner = std::make_shared<internal::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
          half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, half, half, half>>();
    } else if (activationDtype == ScalarType::BFloat16) {
      KLLM_KERNEL_CHECK_WITH_INFO(outputDtype == activationDtype,
                                  fmt::format("Activation dtype {} needs to match Output stype", activationDtype));
      mGemmRunner = std::make_shared<internal::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
          __nv_bfloat16, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, __nv_bfloat16,
          __nv_bfloat16, __nv_bfloat16>>();
    }

    else if (activationDtype == ScalarType::Float8_e4m3fn) {
      if (outputDtype == ScalarType::BFloat16) {
        mGemmRunner = std::make_shared<internal::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
            __nv_fp8_e4m3, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, half, __nv_bfloat16,
            __nv_bfloat16>>();
      } else if (outputDtype == ScalarType::Float16) {
        mGemmRunner = std::make_shared<internal::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
            __nv_fp8_e4m3, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, half, half, half>>();
      } else {
        KLLM_KERNEL_CHECK_WITH_INFO(
            false, fmt::format("Unsupported output dtype {} for Float8_e4m3fn activation", outputDtype));
      }
    } else {
      KLLM_KERNEL_CHECK_WITH_INFO(false, fmt::format("Unsupported activation dtype {}", activationDtype));
    }
  }

  else if (quant_mode == 1) {
    if (activationDtype == ScalarType::Float16) {
      KLLM_KERNEL_CHECK_WITH_INFO(outputDtype == activationDtype,
                                  fmt::format("Activation dtype {} needs to match Output stype", activationDtype));
      mGemmRunner = std::make_shared<internal::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
          half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, half, half, half>>();
    } else if (activationDtype == ScalarType::BFloat16) {
      KLLM_KERNEL_CHECK_WITH_INFO(outputDtype == activationDtype,
                                  fmt::format("Activation dtype {} needs to match Output stype", activationDtype));
      mGemmRunner = std::make_shared<internal::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
          __nv_bfloat16, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, __nv_bfloat16,
          __nv_bfloat16, __nv_bfloat16>>();
    } else if (activationDtype == ScalarType::Float8_e4m3fn) {
      if (outputDtype == ScalarType::BFloat16) {
        mGemmRunner = std::make_shared<internal::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
            __nv_fp8_e4m3, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, half,
            __nv_bfloat16, __nv_bfloat16>>();
      } else if (outputDtype == ScalarType::Float16) {
        mGemmRunner = std::make_shared<internal::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
            __nv_fp8_e4m3, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, half, half,
            half>>();
      } else {
        KLLM_KERNEL_CHECK_WITH_INFO(
            false, fmt::format("Unsupported output dtype {} for Float8_e4m3fn activation", outputDtype));
      }
    }
  } else {
    KLLM_KERNEL_CHECK_WITH_INFO(
        false, fmt::format("Unsupported quant mode for FinegrainedMixedDtypeGemmRunner: {}", quant_mode));
  }

  KLLM_KERNEL_CHECK_WITH_INFO(
      mGemmRunner,
      fmt::format("Failed to create finegrained Mixed Dtype GEMM runner for activation type {}", activationDtype));
  mConfigs = mGemmRunner->getConfigs();  // Get configs via the interface
  KLLM_KERNEL_CHECK_WITH_INFO(
      !mConfigs.empty(),
      fmt::format("Failed to get CUTLASS configs for finegrainedMixedDtype GEMM with activation type {}",
                  activationDtype));
}

size_t FinegrainedMixedDtypeGemmRunner::getWorkspaceSize(const std::vector<size_t>& A_shape,
                                                         const std::vector<size_t>& B_packed_shape) const {
  int M = 0, K_act = 0;
  // Logic to determine M and K_act from A_tensor dimensions
  if (A_shape.size() == 2) {
    M = A_shape[0];
    K_act = A_shape[1];
  } else {  // A_shape.size() >= 3
    M = A_shape[0];
    for (int i = 1; i < A_shape.size() - 1; ++i) M *= A_shape[i];
    K_act = A_shape[A_shape.size() - 1];
  }

  // Assuming B_packed is [K_weights, N_packed_int4_pairs] or similar
  // K_weights should match K_act. N_orig is 2 * N_packed_int4_pairs
  int K_weights = B_packed_shape[0];
  int N_packed_int4 = B_packed_shape[1];  // This is number of uint8_t elements, each holding two int4
  int N_orig = N_packed_int4 * 2;         // N_orig is the original N dimension

  int K = K_act;

  return mGemmRunner->getWorkspaceSize(M, N_orig, K);
}

void FinegrainedMixedDtypeGemmRunner::runGemm(cudaStream_t stream, Tensor& C_tensor, Tensor& workspace_tensor,
                                              Tensor const& A, Tensor const& B_packed, Tensor const& scales,
                                              int64_t group_size_long, int64_t configIdx, std::optional<Tensor> bias,
                                              std::optional<Tensor> zeros, double alpha) const {
  KLLM_KERNEL_CHECK_WITH_INFO(A.dtype == mActivationDtype,
                              fmt::format("Activation tensor A's dtype {} does not match runner's expected dtype {}",
                                          A.dtype, mActivationDtype));
  KLLM_KERNEL_CHECK_WITH_INFO(B_packed.dtype == ScalarType::QUInt4x2 || B_packed.dtype == ScalarType::Int8 ||
                                  B_packed.dtype == ScalarType::UInt8,
                              "B_packed must be quint4x2, int8, or uint8 (view of quantized data)");

  void const* zeros_ptr = nullptr;
  if (zeros.has_value()) {
    KLLM_KERNEL_CHECK_WITH_INFO(
        zeros.value().dtype == ScalarType::Float16 || zeros.value().dtype == ScalarType::BFloat16,
        "Zeros must be FP16 or BF16");
    zeros_ptr = zeros.value().data;
  }

  void const* bias_ptr = nullptr;
  if (bias.has_value()) {
    KLLM_KERNEL_CHECK_WITH_INFO(bias.value().dtype == ScalarType::Float16 || bias.value().dtype == ScalarType::BFloat16,
                                "Bias must be FP16 or BF16");
    bias_ptr = bias.value().data;
  }

  int M = 0, K_act = 0;
  // Logic to determine M and K_act from A_tensor dimensions
  if (A.shape.size() == 2) {
    M = A.shape[0];
    K_act = A.shape[1];
  } else {  // A.shape.size() >= 3
    M = A.shape[0];
    for (int i = 1; i < A.shape.size() - 1; ++i) M *= A.shape[i];
    K_act = A.shape[A.shape.size() - 1];
  }

  // Assuming B_packed is [K_weights, N_packed_int4_pairs] or similar
  // K_weights should match K_act. N_orig is 2 * N_packed_int4_pairs
  int K_weights = B_packed.shape[0];
  int N_packed_int4 = B_packed.shape[1];  // This is number of uint8_t elements, each holding two int4
  int N_orig = N_packed_int4 * 2;         // N_orig is the original N dimension

  KLLM_KERNEL_CHECK_WITH_INFO(
      K_act == K_weights,
      fmt::format("K dimension mismatch: A.shape[-1]={} vs B_packed.shape[0]={}", K_act, K_weights));
  int K = K_act;
  int group_size = static_cast<int>(group_size_long);

  std::vector<size_t> output_shape_vec;
  if (A.shape.size() == 2) {
    output_shape_vec = {static_cast<size_t>(M), static_cast<size_t>(N_orig)};
  } else {
    output_shape_vec.reserve(A.shape.size());
    for (int i = 0; i < A.shape.size() - 1; ++i) output_shape_vec.push_back(A.shape[i]);
    output_shape_vec.push_back(N_orig);
  }

  ScalarType output_dtype;
  if (mOutputDtype == ScalarType::Float16) {
    output_dtype = ScalarType::Float16;
  } else if (mOutputDtype == ScalarType::BFloat16) {
    output_dtype = ScalarType::BFloat16;
  } else {
    KLLM_KERNEL_CHECK_WITH_INFO(false, "Unsupported output dtype");
  }

  C_tensor.shape = output_shape_vec;
  C_tensor.dtype = output_dtype;

  void const* A_ptr = A.data;

  void const* B_ptr = B_packed.data;
  void const* scales_ptr = scales.data;
  void* C_ptr = C_tensor.data;

  llm_kernels::nvidia::tensorrt_llm::dev::cutlass_extensions::CutlassGemmConfig gemm_config_to_use;
  if (configIdx >= 0 && configIdx < getNumConfigs()) {
    gemm_config_to_use = mConfigs.at(configIdx);
  } else {
    gemm_config_to_use = mConfigs.at(0);
  }

  size_t workspace_bytes = mGemmRunner->getWorkspaceSize(M, N_orig, K);
  char* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace_ptr = reinterpret_cast<char*>(workspace_tensor.data);
  }

  mGemmRunner->gemm(A_ptr, B_ptr, scales_ptr, zeros_ptr, bias_ptr, static_cast<float>(alpha), C_ptr, M, N_orig, K,
                    group_size, gemm_config_to_use, workspace_ptr, workspace_bytes, stream);
}

int64_t FinegrainedMixedDtypeGemmRunner::getNumConfigs() const {
  KLLM_KERNEL_CHECK_WITH_INFO(mGemmRunner, "FinegrainedMixedDtypeGemmRunner not initialized properly.");
  return static_cast<int64_t>(mConfigs.size());
}

}  // namespace llm_kernels::nvidia::tensorrt_llm::dev