/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */
#include "ksana_llm/utils/nvidia/deepseek_deepgemm_bridge.h"

#if ENABLE_DEEPSEEK_DEEPGEMM

#  include <cuda_runtime.h>
#  include <cstdlib>
#  include <mutex>
#  include <optional>

#  include "jit/compiler.hpp"
#  include "jit/device_runtime.hpp"
#  include "jit/kernel_runtime.hpp"

#  include "jit_kernels/impls/smxx_clean_logits.hpp"
#  include "jit_kernels/impls/smxx_fp8_mqa_logits.hpp"
#  include "jit_kernels/impls/smxx_fp8_paged_mqa_logits.hpp"

namespace {

void InitializeRuntime() {
  const char* env_path_str = std::getenv("DEEPGEMM_INCLUDE_DIR");
  if (env_path_str == nullptr || *env_path_str == '\0') {
    deep_gemm::Compiler::prepare_init(DEEPGEMM_LIBRARY_ROOT_PATH, DEEPGEMM_CUDA_HOME);
  } else {
    std::filesystem::path deepseek_deepgemm_path(env_path_str);
    deepseek_deepgemm_path /= "deepseek_deep_gemm";
    deep_gemm::Compiler::prepare_init(deepseek_deepgemm_path, DEEPGEMM_CUDA_HOME);
  }
  deep_gemm::KernelRuntime::prepare_init(DEEPGEMM_CUDA_HOME);
}

// 按设备初始化，确保当前 GPU 的 compiler/runtime 实例被创建
void InitializeRuntimePerDevice(int num_sms) { deep_gemm::device_runtime->set_num_sms(num_sms); }

void InitializeRuntimeWithDevice(int num_sms) {
  // 全局初始化只执行一次
  static std::once_flag common_flag;
  std::call_once(common_flag, InitializeRuntime);

  // 设备初始化每次都执行，支持多设备场景
  InitializeRuntimePerDevice(num_sms);
}

}  // namespace

namespace deepseek {

// 初始化 DeepGEMM 运行时环境
// 全局初始化只执行一次，设备初始化可以在不同设备上多次调用
void InitializeDeepGemm(int num_sms) { InitializeRuntimeWithDevice(num_sms); }

void ShutdownDeepGemmRuntime() { deep_gemm::shutdown_device_runtime(); }

void SmxxFp8MqaLogits(const torch::Tensor& q, const torch::Tensor& kv, const torch::Tensor& kv_scales,
                      const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                      const torch::Tensor& cu_seq_len_k_end, const torch::Tensor& logits, int seq_len, int seq_len_kv,
                      int stride_kv, int num_heads, int head_dim, int seq_len_alignment, bool enable_clean) {
  deep_gemm::smxx_fp8_mqa_logits(q, kv, kv_scales, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, seq_len,
                                 seq_len_kv, stride_kv, num_heads, head_dim, seq_len_alignment);

  if (enable_clean) {
    deep_gemm::smxx_clean_logits(logits, std::make_optional(cu_seq_len_k_start), cu_seq_len_k_end, 1, seq_len,
                                 seq_len_kv, static_cast<uint64_t>(stride_kv));
  }
}

void SmxxFp8PagedMqaLogits(const torch::Tensor& q, const torch::Tensor& kv_cache, const torch::Tensor& kv_cache_scales,
                           const torch::Tensor& weights, const torch::Tensor& context_lens, const torch::Tensor& logits,
                           const torch::Tensor& block_table, const torch::Tensor& schedule_meta, int batch_size,
                           int next_n, int num_heads, int head_dim, int num_kv_blocks, int block_kv,
                           int kv_cache_stride_bytes, int logits_stride, int block_table_stride, int num_sms,
                           int num_math_warp_groups, bool enable_clean, int clean_max_context_len) {
  deep_gemm::smxx_fp8_paged_mqa_logits(q, kv_cache, kv_cache_scales, weights, context_lens, logits, block_table,
                                       schedule_meta, batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv,
                                       kv_cache_stride_bytes, logits_stride, block_table_stride, num_sms,
                                       num_math_warp_groups);

  if (enable_clean) {
    deep_gemm::smxx_clean_logits(logits, std::nullopt, context_lens, next_n, batch_size * next_n, clean_max_context_len,
                                 static_cast<uint64_t>(logits_stride));
  }
}

void SmxxPagedMqaLogitsMetadata(const torch::Tensor& context_lens, const torch::Tensor& schedule_metadata,
                                int batch_size, int block_kv, int num_sms) {
  deep_gemm::smxx_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, block_kv, num_sms);
}

}  // namespace deepseek

#else  // ENABLE_DEEPSEEK_DEEPGEMM

namespace deepseek {

void InitializeDeepGemmOnce() {}
void SetDeepGemmNumSms(int) {}
void ShutdownDeepGemmRuntime() {}
void SmxxFp8MqaLogits(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                      const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int, int, int, int, int, int,
                      bool) {}
void SmxxFp8PagedMqaLogits(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                           const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int,
                           int, int, int, int, int, int, int, int, int, int, bool, int) {}
void SmxxPagedMqaLogitsMetadata(const torch::Tensor&, const torch::Tensor&, int, int, int) {}

}  // namespace deepseek

#endif  // ENABLE_DEEPSEEK_DEEPGEMM
