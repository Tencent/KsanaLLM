/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_FP8
#  include "ksana_llm/layers/deepgemm_matmul_layer.h"
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#  include "ksana_llm/profiler/timer.h"

namespace ksana_llm {

Status DeepGemmMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  STATUS_CHECK_FAILURE(BaseLayer::Init(parameters, runtime_config, context, rank));
  int parameter_index = 0;
  max_m_ = std::any_cast<size_t>(parameters[parameter_index++]);
  n_ = std::any_cast<size_t>(parameters[parameter_index++]);
  k_ = std::any_cast<size_t>(parameters[parameter_index++]);
  block_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);

  static std::mutex g_mtx;
  std::lock_guard<std::mutex> guard(g_mtx);

  const auto start_time = ProfileTimer::GetCurrentTime();

  deepgemm_wrapper_ = std::make_shared<llm_kernels::nvidia::DeepGEMMWrapper>(rank);

  for (int m = 1; m <= max_m_; m++) {
    int aligned_m = RoundUp(m, 4);
    // 生成普通的gemm算子
    deepgemm_wrapper_->BuildGemmKernel(aligned_m, n_, k_);
    // 生成swapAB的gemm算子
    deepgemm_wrapper_->BuildGemmSwapABKernel(aligned_m, n_, k_);
  }

  KLLM_LOG_INFO << fmt::format("Rank[{}] DeepGemmMatMulLayer Init cost time: {} s", rank_,
                               ProfileTimer::GetCurrentTime() - start_time);
  return Status();
}

size_t DeepGemmMatMulLayer::GetWorkSpaceSize() {
  size_t input_size = max_m_ * k_ * GetTypeSize(TYPE_FP8_E4M3);
  size_t scale_size = max_m_ * DivRoundUp(k_, block_size_) * GetTypeSize(TYPE_FP32);
  size_t workspace_size = input_size + scale_size;
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for DeepGemmMatMulLayer", rank_, workspace_size);
  return workspace_size;
}

Status DeepGemmMatMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status DeepGemmMatMulLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int m = input_tensors[0].shape[0];
  int k = input_tensors[0].shape[1];
  int n = input_tensors[1].shape[0];

  const size_t align_m = 4;
  int aligned_m = RoundUp(m, 4);
  T* a = static_cast<T*>(input_tensors[0].GetPtr<void>());
  void* a_q = workspace_buffer_->GetPtr<void>();
  void* a_s = a_q + GetTypeSize(TYPE_FP8_E4M3) * aligned_m * k;

  InvokePerTokenGroupQuantFp8E4m3<T>(a, a_q, a_s, aligned_m, k, true, context_->GetComputeStreams()[rank_].Get(),
                                     block_size_);

  void* b = input_tensors[1].GetPtr<void>();
  void* b_s = input_tensors[1].weight_scales->GetPtr<void>();

  void* out = output_tensors[0].GetPtr<void>();

  // TODO(jinxcwu) 要做自适应阈值
  if (aligned_m <= 64) {
    deepgemm_wrapper_->GemmSwapAB(a_q, a_s, b, b_s, out, aligned_m, n, k, context_->GetComputeStreams()[rank_].Get());
  } else {
    deepgemm_wrapper_->Gemm(a_q, a_s, b, b_s, out, aligned_m, n, k, context_->GetComputeStreams()[rank_].Get());
  }

  output_tensors[0].shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}

}  // namespace ksana_llm
#endif
