/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/lm_head_matmul_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status LmHeadMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                               std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;
  return Status();
}

Status LmHeadMatMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status LmHeadMatMulLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  const int m = static_cast<int>(input_tensors[0].shape[0]);
  const int n = static_cast<int>(input_tensors[1].shape[1]);
  const int k = static_cast<int>(input_tensors[0].shape[1]);
  const void* a_ptr = reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>());
  const void* b_ptr = reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>());
  void* c_ptr = output_tensors[0].GetPtr<void>();

  // Use strided batched GEMM for decode case (m=1, n>>>k) for better performance
  if (m == 1) {
    const int ldb = n;
    constexpr int64_t stride_b = 0LL;
    const int lda = k;
    const int64_t stride_a = static_cast<int64_t>(k);
    const int ldc = n;
    const int64_t stride_c = static_cast<int64_t>(n);
    constexpr int batch_count = 1;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    InvokeStridedBatchedMatMul<T>(context_->ext->GetCublasHandles()[rank_], context_->ext->GetCublasLtHandles()[rank_],
                                  CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, a_ptr, lda, stride_a, b_ptr, ldb, stride_b, c_ptr,
                                  ldc, stride_c, batch_count, alpha, beta);
  } else {
    InvokeMatMul<T>(context_->ext->GetCublasHandles()[rank_], context_->ext->GetCublasLtHandles()[rank_], m, n, k,
                    a_ptr, b_ptr, c_ptr, context_->GetComputeStreams()[rank_].Get(), nullptr, nullptr, 0);
  }

  output_tensors[0].shape = {input_tensors[0].shape[0], input_tensors[1].shape[1]};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm

