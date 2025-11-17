/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_FP8
#  include "ksana_llm/layers/fp8_matmul_layer.h"

#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status Fp8MatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  STATUS_CHECK_FAILURE(BaseLayer::Init(parameters, runtime_config, context, rank));
  int parameter_index = 0;
  max_m_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_k_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  cublas_workspace_size_ = InvokeGetCublasWorkspaceSize();

  return Status();
}

size_t Fp8MatMulLayer::GetWorkspaceSize(const int m, const int k) {
  size_t input_size = m * k * GetTypeSize(TYPE_FP8_E4M3);
  size_t scale_size = GetTypeSize(TYPE_FP32);
  size_t workspace_size = input_size + scale_size + cublas_workspace_size_;
  return workspace_size;
}

size_t Fp8MatMulLayer::GetWorkspaceSize() {
  size_t workspace_size = GetWorkspaceSize(max_m_, max_k_);
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for Fp8MatMulLayer", rank_, workspace_size);
  return workspace_size;
}

Status Fp8MatMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status Fp8MatMulLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int m = input_tensors[0].shape[0];
  int k = input_tensors[0].shape[1];
  int n = input_tensors[1].shape[0];
  const T* input = static_cast<const T*>(input_tensors[0].GetPtr<void>());
  size_t workspace_size = GetWorkspaceSize(m, k);
  if (workspace_size > workspace_buffer_->GetTotalBytes()) {
    KLLM_THROW(fmt::format("workspace size {} > buffer size {}", workspace_size, workspace_buffer_->GetTotalBytes()));
  }
  void* cublas_workspace = workspace_buffer_->GetPtr<void>();
  void* input_quant = cublas_workspace + cublas_workspace_size_;
  const void* weight_quant = input_tensors[1].GetPtr<const void>();
  const void* weight_scale = input_tensors[1].weight_scales->GetPtr<const void>();
  if (weight_scale == nullptr) {
    KLLM_THROW("Cannot load weight_scale. Weight_scale is nullptr.");
  }
  T* output = static_cast<T*>(output_tensors[0].GetPtr<void>());
  output_tensors[0].shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
  output_tensors[0].dtype = input_tensors[0].dtype;
  float* input_scale = nullptr;
  if (input_tensors[1].input_scales) {
    input_scale = static_cast<float*>(input_tensors[1].input_scales->GetPtr<void>());
    Fp8E4m3Quantize<T>(1, m * k, input, input_quant, input_scale, true, context_->GetComputeStreams()[rank_].Get());
  } else {
    input_scale = static_cast<float*>(input_quant + GetTypeSize(TYPE_FP8_E4M3) * m * k);
    Fp8E4m3Quantize<T>(1, m * k, input, input_quant, input_scale, false, context_->GetComputeStreams()[rank_].Get());
  }

  if (context_->ext->GetGPUGemmAlgoHelper().IsInit()) {
    uint32_t sm = context_->ext->GetComputeCapacity();
    uint32_t cuda_ver = context_->ext->GetCudaVersion();
    // NOTE(karlluo): for continue batching, there is not batch size here.
    constexpr uint64_t BATCH_SIZE = 1ull;
    // NOTE(karlluo): using an impossible type to make sure no algo can be search for an unregistered type.
    cudaDataType_t c_dtype = CUDA_R_32F;
    if (std::is_same<T, float>::value) {
      c_dtype = CUDA_R_32F;
    } else if (std::is_same<T, half>::value) {
      c_dtype = CUDA_R_16F;
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      c_dtype = CUDA_R_16BF;
    }
    llm_kernels::nvidia::GemmAlgoInfo gemm_algo_info = context_->ext->GetGPUGemmAlgoHelper().GetGemmAlgo(
        sm, cuda_ver, BATCH_SIZE, input_tensors[0].shape[0], input_tensors[1].shape[1], input_tensors[0].shape[1],
        CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, c_dtype, CUDA_R_32F, CUBLAS_OP_T, CUBLAS_OP_N);
    if (gemm_algo_info.gemm_op_type != llm_kernels::nvidia::DEFAULT_GEMM_ALGO) {
      cublaslt_algo_ptr_ = &(gemm_algo_info.cublaslt_algo);
    }
  }

  Fp8QuantizedMatMul<T>(context_->ext->GetCublasHandles()[rank_], context_->ext->GetCublasLtHandles()[rank_], m, n, k,
                        input_quant, input_scale, weight_quant, weight_scale, output,
                        context_->GetComputeStreams()[rank_].Get(), cublaslt_algo_ptr_, cublas_workspace,
                        cublas_workspace_size_);
  cublaslt_algo_ptr_ = nullptr;
  return Status();
}

}  // namespace ksana_llm
#endif
