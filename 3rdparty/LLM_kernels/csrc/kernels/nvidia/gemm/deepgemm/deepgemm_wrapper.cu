/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/gemm/deepgemm/deepgemm_wrapper.h"

#include <fmt/format.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>

#if ENABLE_DEEPGEMM
#  include "csrc/kernels/nvidia/others/tensorrt-llm/dev/deep_gemm/fp8_gemm.cuh"
#endif
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

DeepGEMMWrapper::DeepGEMMWrapper(int id) {
#if ENABLE_DEEPGEMM
  num_device_sms_ = GetSMCount();
  id_ = id;
  KLLM_KERNEL_CHECK(deep_gemm::jit::getThreadCompiler(id_).isValid());
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPGEMM=0, skipping DeepGemm kernel.");
#endif
}

void DeepGEMMWrapper::BuildGemmKernel(int m, int n, int k) {
#if ENABLE_DEEPGEMM
  constexpr uint32_t block_k = 128;
  constexpr uint32_t num_problems = 1;
  auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size] =
      deep_gemm::jit::get_best_gemm_config(m, n, k, num_problems, num_device_sms_);
  deep_gemm::jit::getThreadCompiler(id_).build(n, k, best_block_m, best_block_n, block_k, num_problems, best_num_stages,
                                               best_num_tma_multicast, deep_gemm::GemmType::Normal, false, id_);
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPGEMM=0, skipping DeepGemm kernel.");
#endif
}

void DeepGEMMWrapper::Gemm(void* x_fp8, void* x_scales, void* y_fp8, void* y_scales, void* out, int m, int n, int k,
                           cudaStream_t stream) {
#if ENABLE_DEEPGEMM
  constexpr uint32_t block_k = 128;
  constexpr uint32_t num_problems = 1;

  auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size] =
      deep_gemm::jit::get_best_gemm_config(m, n, k, num_problems, num_device_sms_);

  auto runtime = deep_gemm::jit::getThreadCompiler(id_).build(n, k, best_block_m, best_block_n, block_k, num_problems,
                                                              best_num_stages, best_num_tma_multicast,
                                                              deep_gemm::GemmType::Normal, false, id_);
  auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());

  deep_gemm::runGemm(kernel, x_fp8, k, y_fp8, k, out, n, static_cast<float*>(x_scales), static_cast<float*>(y_scales),
                     m, n, k, best_block_m, best_block_n, block_k, num_problems, best_num_tma_multicast,
                     deep_gemm::GemmType::Normal, static_cast<int*>(nullptr), stream, num_device_sms_,
                     static_cast<uint32_t>(best_smem_size));
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPGEMM=0, skipping DeepGemm kernel.");
#endif
}

void DeepGEMMWrapper::BuildGemmSwapABKernel(int m, int n, int k) {
#if ENABLE_DEEPGEMM
  constexpr uint32_t block_k = 128;
  constexpr uint32_t num_problems = 1;
  auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size] =
      deep_gemm::jit::get_best_gemm_config(n, m, k, num_problems, num_device_sms_, false, true);
  deep_gemm::jit::getThreadCompiler(id_).build(n, k, best_block_m, best_block_n, block_k, num_problems, best_num_stages,
                                               best_num_tma_multicast, deep_gemm::GemmType::Normal, true, id_);
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPGEMM=0, skipping DeepGemm kernel.");
#endif
}

void DeepGEMMWrapper::GemmSwapAB(void* x_fp8, void* x_scales, void* y_fp8, void* y_scales, void* out, int m, int n,
                                 int k, cudaStream_t stream) {
#if ENABLE_DEEPGEMM
  constexpr uint32_t block_k = 128;
  constexpr uint32_t num_problems = 1;

  auto [best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size] =
      deep_gemm::jit::get_best_gemm_config(n, m, k, num_problems, num_device_sms_, false, true);

  auto runtime = deep_gemm::jit::getThreadCompiler(id_).build(n, k, best_block_m, best_block_n, block_k, num_problems,
                                                              best_num_stages, best_num_tma_multicast,
                                                              deep_gemm::GemmType::Normal, true, id_);
  auto kernel = reinterpret_cast<cudaKernel_t>(runtime->getKernel());

  deep_gemm::runGemmSwapAB(kernel, y_fp8, k, x_fp8, k, out, n, static_cast<float*>(y_scales),
                           static_cast<float*>(x_scales), n, m, k, best_block_m, best_block_n, block_k, num_problems,
                           best_num_tma_multicast, deep_gemm::GemmType::Normal, static_cast<int*>(nullptr), stream,
                           num_device_sms_, static_cast<uint32_t>(best_smem_size));
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPGEMM=0, skipping DeepGemm kernel.");
#endif
}

}  // namespace nvidia
}  // namespace llm_kernels