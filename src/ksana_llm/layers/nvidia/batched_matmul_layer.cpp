/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/batched_matmul_layer.h"
#include <vector>
#include "ksana_llm/profiler/profile_event.h"
#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status BatchedMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;
  return Status();
}

Status BatchedMatMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
cudaDataType_t GetCublasDataType();
#define GET_Cublas_DATA_TYPE(T, Cublas_TYPE) template <> cudaDataType_t GetCublasDataType<T>() {return Cublas_TYPE;}
GET_Cublas_DATA_TYPE(float, CUDA_R_32F);
GET_Cublas_DATA_TYPE(half, CUDA_R_16F);
GET_Cublas_DATA_TYPE(__nv_bfloat16, CUDA_R_16BF);
#undef GET_Cublas_DATA_TYPE

template <typename T>
void FindBestMatmulAlgos(
  cublasLtHandle_t cublasLtHandle,
  size_t bs, size_t m, size_t n, size_t k,
  void* workspace_ptr, size_t workspace_size,
  float f_alpha, float f_beta,
  cudaDataType_t compute_type,
  cublasOperation_t transa = CUBLAS_OP_N,
  cublasOperation_t transb = CUBLAS_OP_N,
  size_t req_algos = 1,  // Maximum number of returned algorithms
  cublasLtMatmulAlgo_t* b_algo = nullptr
) {
  half h_alpha = static_cast<half>(f_alpha);
  half h_beta = static_cast<half>(f_beta);

  int32_t is_fp16_compute_type = (compute_type == CUDA_R_16F) ? 1 : 0;

  const void* alpha = is_fp16_compute_type ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
  const void* beta = is_fp16_compute_type ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasComputeType_t inner_compute_type = is_fp16_compute_type ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;

  cudaDataType_t a_type = GetCublasDataType<T>();
  cudaDataType_t b_type = GetCublasDataType<T>();
  cudaDataType_t c_type = GetCublasDataType<T>();

  int32_t lda = n;
  int32_t ldb = k;
  int32_t ldc = n;

  cublasLtMatrixLayoutCreate(&a_desc, a_type, (transa == CUBLAS_OP_N) ? n : k, (transa == CUBLAS_OP_N) ? k : n, lda);
  cublasLtMatrixLayoutCreate(&b_desc, b_type, (transb == CUBLAS_OP_N) ? k : m, (transb == CUBLAS_OP_N) ? m : k, ldb);
  cublasLtMatrixLayoutCreate(&c_desc, c_type, n, m, ldc);

  size_t strideA = n * k;
  size_t strideB = k * m;
  size_t strideC = m * n;

  cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bs, sizeof(bs));
  cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bs, sizeof(bs));
  cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bs, sizeof(bs));

  cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA));
  cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB));
  cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC));

  cublasLtMatmulDescCreate(&operation_desc, inner_compute_type, scale_type);
  cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

  cublasLtMatmulPreference_t preference = NULL;
  cublasLtMatmulPreferenceCreate(&preference);
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    &workspace_size, sizeof(workspace_size));

  cublasLtMatmulHeuristicResult_t heuristicResult[req_algos] = {0};
  int returned_count = 0;

  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
      cublasLtHandle, operation_desc, a_desc, b_desc, c_desc, c_desc, preference, req_algos,
      heuristicResult, &returned_count);

  KLLM_CHECK_WITH_INFO(status == CUBLAS_STATUS_SUCCESS,
    FormatStr("[ERROR] cublasLtMatmulAlgoGetHeuristic failed, status =  %s.", status));

  b_algo = &heuristicResult[0].algo;
}

template <typename T>
Status BatchedMatMulLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
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

  cublas_workspace_size_ = 0;
  if (workspace_buffer_ == nullptr || workspace_buffer_->GetTotalBytes() == 0) {
    KLLM_LOG_DEBUG << "No workspace can be reused for batched matmul layer.";
  } else {
    cublas_workspace_ptr_ = workspace_buffer_->GetPtr<void>();
    cublas_workspace_size_ = workspace_buffer_->GetTotalBytes();
  }
  // Note(JW): Based on experimental testing, the BMM module may encounter bad cases during algorithm search
  // when the batch size exceeds 5000. Therefore, a threshold is set to optimize for this scenario in the future.
  if (m > 5000) {
    FindBestMatmulAlgos<T>(context_->ext->GetCublasLtHandles()[rank_],
    bs, m, n, k, cublas_workspace_ptr_, cublas_workspace_size_,
    1.0f, 1.0f, CUDA_R_32F, CUBLAS_OP_N, CUBLAS_OP_N, 1, cublaslt_algo_ptr_);
  }

  InvokeBatchedMatMul<T>(context_->ext->GetCublasHandles()[rank_], context_->ext->GetCublasLtHandles()[rank_],
                        bs, m, n, k, reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                        reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>()),
                        output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_].Get(),
                        cublas_workspace_ptr_, cublas_workspace_size_, cublaslt_algo_ptr_);
  output_tensors[0].shape = {bs, m, n};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}
}  // namespace ksana_llm
