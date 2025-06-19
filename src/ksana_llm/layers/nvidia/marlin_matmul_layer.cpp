/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/marlin_matmul_layer.h"
#include "csrc/kernels/nvidia/gptq_marlin/marlin_wrapper.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T, DataType WT>
Status MarlinMatMulLayer<T, WT>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                      int rank) {
  context_ = context;
  rank_ = rank;

  int parameter_index = 0;
  max_m_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_n_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_k_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  groupsize_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  is_awq_ = std::any_cast<const bool>(parameters[parameter_index++]);
  is_gptq_desc_ = std::any_cast<const bool>(parameters[parameter_index++]);
  is_k_full_ = std::any_cast<const bool>(parameters[parameter_index++]);
  bool cutlass_use_gemv_cuda_core_ = std::any_cast<const bool>(parameters[parameter_index++]);  // unused

  // double check some parameter
  is_k_full_ = is_gptq_desc_ ? is_k_full_ : true;

  return Status();
}

inline size_t AlignAddress(size_t size) { return (size + 255) & (~255); }

template <typename T, DataType WT>
size_t MarlinMatMulLayer<T, WT>::GetWorkSpaceSize() {
  if constexpr (WT == TYPE_I4_GROUP) {
    llm_kernels::nvidia::marlin::WorkspaceInfo info = GetMarlinWorkspace<T>(true, is_gptq_desc_, rank_, max_m_, max_k_);

    marlin_workspace_size_ = AlignAddress(info.workspace_size);
    marlin_input_tmp_size_ = AlignAddress(info.a_tmp_size);
    marlin_output_tmp_size_ = AlignAddress(info.c_tmp_size);

    marlin_workspace_offset_ = 0;
    marlin_input_tmp_offset_ = marlin_workspace_offset_ + marlin_workspace_size_;
    marlin_output_tmp_offset_ = marlin_input_tmp_offset_ + marlin_input_tmp_size_;

    size_t max_ws_bytes = marlin_workspace_size_ + marlin_input_tmp_size_ + marlin_output_tmp_size_;

    KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for MarlinMatMulLayer", rank_, max_ws_bytes);
    return max_ws_bytes;
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. MarlinMatMul only supports TYPE_I4_GROUP.", WT));
  }
}

template <typename T, DataType WT>
Status MarlinMatMulLayer<T, WT>::Forward(const std::vector<Tensor>& input_tensors,
                                         std::vector<Tensor>& output_tensors) {
  if constexpr (WT == TYPE_I4_GROUP) {
    const Tensor& weight_tensor = input_tensors[1];
    void* p_qweight_tensor = weight_tensor.GetPtr<void>();
    void* p_scales_tensor = weight_tensor.scales->GetPtr<void>();
    void* p_zeros_tensor = is_awq_ ? weight_tensor.zeros->GetPtr<void>() : nullptr;
    void* p_g_idx_tensor = is_gptq_desc_ ? weight_tensor.g_idx->GetPtr<void>() : nullptr;
    void* p_perm_tensor = is_gptq_desc_ ? weight_tensor.perm->GetPtr<void>() : nullptr;

    const size_t m = input_tensors[0].shape[0];
    const size_t n = max_n_;
    const size_t k = max_k_;

    const size_t num_group = weight_tensor.scales->shape[0];
    MemsetAsync(workspace_buffer_->GetPtr<void>() + marlin_workspace_offset_, 0, marlin_workspace_size_,
                context_->GetComputeStreams()[rank_]);
    InvokeMarlinGemm<T>(input_tensors[0].GetPtr<void>(), workspace_buffer_->GetPtr<void>() + marlin_input_tmp_offset_,
                        p_qweight_tensor, p_scales_tensor, p_zeros_tensor, p_g_idx_tensor, p_perm_tensor,
                        workspace_buffer_->GetPtr<void>() + marlin_workspace_offset_, output_tensors[0].GetPtr<void>(),
                        workspace_buffer_->GetPtr<void>() + marlin_output_tmp_offset_, m, n, k, num_group, is_k_full_,
                        true, true, false, is_awq_, is_gptq_desc_, is_awq_, rank_,
                        context_->GetComputeStreams()[rank_].Get());

    output_tensors[0].shape = {m, n};
    output_tensors[0].dtype = input_tensors[0].dtype;
    return Status();
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. MarlinMatMul only supports TYPE_I4_GROUP.", WT));
  }
}

template class MarlinMatMulLayer<float, TYPE_I4_GROUP>;
template class MarlinMatMulLayer<half, TYPE_I4_GROUP>;
#ifdef ENABLE_BFLOAT16
template class MarlinMatMulLayer<__nv_bfloat16, TYPE_I4_GROUP>;
#endif

}  // namespace ksana_llm
