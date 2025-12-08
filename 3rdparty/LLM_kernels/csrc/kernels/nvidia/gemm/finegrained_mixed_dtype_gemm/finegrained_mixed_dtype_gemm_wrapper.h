/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <optional>

#include <cuda_runtime.h>
#include <torch/torch.h>

namespace llm_kernels {
namespace nvidia {

namespace tensorrt_llm {
namespace dev {
enum ScalarType : int;
class FinegrainedMixedDtypeGemmRunner;
struct Tensor;
namespace common {
enum QuantizeMode : int;
}  // namespace common
}  // namespace dev
}  // namespace tensorrt_llm

using namespace llm_kernels::nvidia::tensorrt_llm::dev;

enum class QuantMode : int64_t { NO_ZERO = 0, WITH_ZERO = 1 };

class FinegrainedMixedDtypeGemmWrapper {
 public:
  FinegrainedMixedDtypeGemmWrapper(ScalarType activationDtype, ScalarType outputDtype, QuantMode quant_mode,
                                   int64_t group_size);

  ~FinegrainedMixedDtypeGemmWrapper();

  torch::Tensor PreprocessWeightsForMixedGemm(torch::Tensor tensor, torch::ScalarType quant_mode,
                                              torch::ScalarType act_dtype, int sm, bool do_weight_interleave);

  int64_t GetNumConfigs();

  size_t GetWorkspaceSize(const size_t& M, const size_t& N, const size_t& K);

  int64_t Profile(cudaStream_t stream, Tensor& C_tensor, void* workspace_ptr, Tensor const& A, Tensor const& B_packed,
                  Tensor const& scales, std::optional<Tensor> bias, std::optional<Tensor> zeros, float alpha,
                  size_t warmup, size_t iters);

  void Forward(cudaStream_t stream, Tensor& C_tensor, void* workspace_ptr, Tensor const& A, Tensor const& B_packed,
               Tensor const& scales, int64_t configIdx, std::optional<Tensor> bias, std::optional<Tensor> zeros,
               float alpha);

  void E4M3StaticQuantize(Tensor& quantized_input, const Tensor& input, const Tensor& scales,
                          tensorrt_llm::dev::common::QuantizeMode quantize_mode, cudaStream_t stream);

  void StaticQuantizeE4M3PerTensor(Tensor& quantized_input, const Tensor& input, const Tensor& scales,
                                   cudaStream_t stream);

  void StaticQuantizeE4M3PerToken(Tensor& quantized_input, const Tensor& input, const Tensor& scales,
                                  cudaStream_t stream);

  void StaticQuantizeE4M3PerChannel(Tensor& quantized_input, const Tensor& input, const Tensor& scales,
                                    cudaStream_t stream);

 private:
#if ENABLE_FINEGRAINED_MIXED_DTYPE_GEMM
  std::unique_ptr<FinegrainedMixedDtypeGemmRunner> finegrained_mixed_dtype_gemm_runner_;
#endif
  int64_t group_size_{128};
};

}  // namespace nvidia
}  // namespace llm_kernels