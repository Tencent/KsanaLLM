/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/gemm/finegrained_mixed_dtype_gemm/finegrained_mixed_dtype_gemm_wrapper.h"

#include <numeric>

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/utils.h"
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
#  include "csrc/kernels/nvidia/others/tensorrt-llm/dev/common/cudaFp8Utils.h"
#  include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/finegrained_mixed_dtype_gemm_thop.h"
#endif
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;
using namespace llm_kernels::nvidia::tensorrt_llm::dev;

namespace llm_kernels {
namespace nvidia {

FinegrainedMixedDtypeGemmWrapper::FinegrainedMixedDtypeGemmWrapper(ScalarType activationDtype, ScalarType outputDtype,
                                                                   QuantMode quant_mode, int64_t group_size)
    : group_size_(group_size) {
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  int64_t quant_mode_value = static_cast<int64_t>(quant_mode);
  finegrained_mixed_dtype_gemm_runner_ =
      std::make_unique<FinegrainedMixedDtypeGemmRunner>(activationDtype, outputDtype, quant_mode_value);
#else
  KLLM_KERNEL_THROW("ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0, skipping Finegrained Mixed Dtype GEMM kernel.");
#endif
}

FinegrainedMixedDtypeGemmWrapper::~FinegrainedMixedDtypeGemmWrapper() = default;

torch::Tensor FinegrainedMixedDtypeGemmWrapper::PreprocessWeightsForMixedGemm(torch::Tensor tensor,
                                                                              torch::ScalarType quant_mode,
                                                                              torch::ScalarType act_dtype, int sm,
                                                                              bool do_weight_interleave) {
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  return llm_kernels::nvidia::tensorrt_llm::dev::preprocess_weights_for_mixed_gemm(tensor, quant_mode, act_dtype, sm,
                                                                                   do_weight_interleave);
#else
  KLLM_KERNEL_THROW("ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0, skipping Finegrained Mixed Dtype GEMM kernel.");
#endif
}

int64_t FinegrainedMixedDtypeGemmWrapper::GetNumConfigs() {
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  return finegrained_mixed_dtype_gemm_runner_->getNumConfigs();
#else
  KLLM_KERNEL_THROW("ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0, skipping Finegrained Mixed Dtype GEMM kernel.");
#endif
}

size_t FinegrainedMixedDtypeGemmWrapper::GetWorkspaceSize(const size_t& M, const size_t& N, const size_t& K) {
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  const std::vector<size_t>& A_shape = {M, K};
  const std::vector<size_t>& B_packed_shape = {K, N / 2};  // each holding two int4
  return finegrained_mixed_dtype_gemm_runner_->getWorkspaceSize(A_shape, B_packed_shape);
#else
  KLLM_KERNEL_THROW("ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0, skipping Finegrained Mixed Dtype GEMM kernel.");
#endif
}

int64_t FinegrainedMixedDtypeGemmWrapper::Profile(cudaStream_t stream, Tensor& C_tensor, void* workspace_ptr,
                                                  Tensor const& A, Tensor const& B_packed, Tensor const& scales,
                                                  std::optional<Tensor> bias, std::optional<Tensor> zeros, float alpha,
                                                  size_t warmup, size_t iters) {
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  int64_t num_configs = GetNumConfigs();
  std::vector<float> config_times(num_configs, std::numeric_limits<float>::max());
  for (int64_t config_idx = 0; config_idx < num_configs; config_idx++) {
    try {
      auto once = [&]() {
        Forward(stream, C_tensor, workspace_ptr, A, B_packed, scales, config_idx, bias, zeros, alpha);
      };
      config_times[config_idx] = MeasureCudaExecutionTime(once, stream, warmup, iters);
    } catch (const std::exception& e) {
      config_times[config_idx] = std::numeric_limits<float>::max();
    }
  }
  auto min_it = std::min_element(config_times.begin(), config_times.end());
  return std::distance(config_times.begin(), min_it);
#else
  KLLM_KERNEL_THROW("ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0, skipping Finegrained Mixed Dtype GEMM kernel.");
#endif
}

void FinegrainedMixedDtypeGemmWrapper::Forward(cudaStream_t stream, Tensor& C_tensor, void* workspace_ptr,
                                               Tensor const& A, Tensor const& B_packed, Tensor const& scales,
                                               int64_t configIdx, std::optional<Tensor> bias,
                                               std::optional<Tensor> zeros, float alpha) {
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  finegrained_mixed_dtype_gemm_runner_->runGemm(stream, C_tensor, workspace_ptr, A, B_packed, scales, group_size_,
                                                configIdx, bias, zeros, alpha);
#else
  KLLM_KERNEL_THROW("ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0, skipping Finegrained Mixed Dtype GEMM kernel.");
#endif
}

void FinegrainedMixedDtypeGemmWrapper::E4M3StaticQuantize(Tensor& quantized_input, const Tensor& input,
                                                          const Tensor& scales, common::QuantizeMode quantize_mode,
                                                          cudaStream_t stream) {
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  KLLM_KERNEL_CHECK(scales.dtype == ScalarType::Float);
  __nv_fp8_e4m3* quantized_input_ptr = static_cast<__nv_fp8_e4m3*>(quantized_input.data);
  const float* scales_ptr = static_cast<const float*>(scales.data);
  const int64_t numel = std::accumulate(input.shape.begin(), input.shape.end(), 1LL, std::multiplies<int64_t>());
  const int64_t lda = input.shape.back();
  if (input.dtype == ScalarType::Float) {
    common::invokeQuantizeMatrix(quantized_input_ptr, scales_ptr, static_cast<const float*>(input.data), numel, lda,
                                 quantize_mode, stream);
  } else if (input.dtype == ScalarType::Float16) {
    common::invokeQuantizeMatrix(quantized_input_ptr, scales_ptr, static_cast<const half*>(input.data), numel, lda,
                                 quantize_mode, stream);
  } else if (input.dtype == ScalarType::BFloat16) {
    common::invokeQuantizeMatrix(quantized_input_ptr, scales_ptr, static_cast<const __nv_bfloat16*>(input.data), numel,
                                 lda, quantize_mode, stream);
  } else {
    KLLM_KERNEL_CHECK_WITH_INFO(false, "Invalid datatype. input must be BF16/FP16/FP32");
  }
#else
  KLLM_KERNEL_THROW("ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0, skipping Finegrained Mixed Dtype GEMM kernel.");
#endif
}

void FinegrainedMixedDtypeGemmWrapper::StaticQuantizeE4M3PerTensor(Tensor& quantized_input, const Tensor& input,
                                                                   const Tensor& scales, cudaStream_t stream) {
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  E4M3StaticQuantize(quantized_input, input, scales, common::QuantizeMode::PER_TENSOR, stream);
#else
  KLLM_KERNEL_THROW("ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0, skipping Finegrained Mixed Dtype GEMM kernel.");
#endif
}

void FinegrainedMixedDtypeGemmWrapper::StaticQuantizeE4M3PerToken(Tensor& quantized_input, const Tensor& input,
                                                                  const Tensor& scales, cudaStream_t stream) {
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  E4M3StaticQuantize(quantized_input, input, scales, common::QuantizeMode::PER_TOKEN, stream);
#else
  KLLM_KERNEL_THROW("ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0, skipping Finegrained Mixed Dtype GEMM kernel.");
#endif
}

void FinegrainedMixedDtypeGemmWrapper::StaticQuantizeE4M3PerChannel(Tensor& quantized_input, const Tensor& input,
                                                                    const Tensor& scales, cudaStream_t stream) {
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  E4M3StaticQuantize(quantized_input, input, scales, common::QuantizeMode::PER_CHANNEL, stream);
#else
  KLLM_KERNEL_THROW("ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM=0, skipping Finegrained Mixed Dtype GEMM kernel.");
#endif
}

}  // namespace nvidia
}  // namespace llm_kernels