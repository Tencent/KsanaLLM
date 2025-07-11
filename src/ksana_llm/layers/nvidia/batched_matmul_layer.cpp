/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/batched_matmul_layer.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status BatchedMatMulLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                   int rank) {
  context_ = context;
  rank_ = rank;
  return Status();
}

template <typename T>
Status BatchedMatMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_CHECK_WITH_INFO(input_tensors.size() == 2, "shoud have two input tensors.");
  KLLM_CHECK_WITH_INFO(input_tensors[0].shape.size() == 3, "input tensors shape size should be 3.");
  KLLM_CHECK_WITH_INFO(input_tensors[1].shape.size() == 3, "input tensors shape size should be 3.");
  KLLM_CHECK_WITH_INFO(input_tensors[0].shape[0] == input_tensors[1].shape[0], "input batch size should be equal.");
  KLLM_CHECK_WITH_INFO(input_tensors[0].shape[2] == input_tensors[1].shape[1],
                       "input and output k value should be equal.");

  size_t bs = input_tensors[0].shape[0];
  size_t m = input_tensors[0].shape[1];
  size_t n = input_tensors[1].shape[2];
  size_t k = input_tensors[0].shape[2];

  void* cublas_workspace = nullptr;
  size_t workspace_size = 0;
  if (workspace_buffer_ == nullptr || workspace_buffer_->GetTotalBytes() == 0) {
    KLLM_LOG_DEBUG << "No workspace can be reused for batched matmul layer.";
  } else {
    cublas_workspace = workspace_buffer_->GetPtr<void>();
    workspace_size = workspace_buffer_->GetTotalBytes();
  }

  // Note(TJ): can get best workspace_size and algo from cublasLtMatmulAlgoGetHeuristic
  // Note(TJ): add search algorithm ?
  cublasLtMatmulAlgo_t* cublaslt_algo_ptr = nullptr;
  InvokeBatchedMatMul<T>(context_->ext->GetCublasHandles()[rank_], context_->ext->GetCublasLtHandles()[rank_], bs, m, n,
                         k, reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                         reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()),
                         output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_].Get(), cublas_workspace,
                         workspace_size, cublaslt_algo_ptr);

  output_tensors[0].shape = {bs, m, n};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template class BatchedMatMulLayer<float>;
template class BatchedMatMulLayer<half>;
template class BatchedMatMulLayer<__nv_bfloat16>;

}  // namespace ksana_llm
