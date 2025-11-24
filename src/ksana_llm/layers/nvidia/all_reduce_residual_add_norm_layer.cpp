/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/all_reduce_residual_add_norm_layer.h"

#include "csrc/kernels/nvidia/others/vllm/main/tokenweave/tokenweave_fused_kernels.h"
#include "ksana_llm/kernels/nvidia/basic_kernel_wrapper.h"

namespace ksana_llm {

Status AllReduceResidualAddNormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                           std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  rms_norm_weight_ = std::any_cast<Tensor>(parameters[parameter_index++]);
  return Status();
}

Status AllReduceResidualAddNormLayer::Forward(const std::vector<Tensor>& input_tensors,
                                              std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status AllReduceResidualAddNormLayer::ForwardT(const std::vector<Tensor>& input_tensors,
                                               std::vector<Tensor>& output_tensors) {
  const size_t token_num = input_tensors[0].shape[0];
  const size_t hidden_dim = input_tensors[0].shape[1];
  void* input = input_tensors[0].GetPtr<void>();
  void* residual_in_ptr = input_tensors.size() == 1 ? nullptr : input_tensors[1].GetPtr<void>();
  void* layernorm_weight_ptr = rms_norm_weight_.GetPtr<void>();
  void* norm_out_ptr = output_tensors[0].GetPtr<void>();

  // Fused the next three steps into one kernel operation
  // Step 1: tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0)
  // Step 2: adds->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer)
  // Step 3 (optional): layernorm_layer_->Forward(residual_buffer, hidden_buffer_tensors_0)
  if (context_->ext->IsMulticastSupported()) {
    // Each rank processes an equal portion of all tokens
    // Last rank may process extra tokens, but with no effect
    const size_t world_size = context_->GetTensorParallelSize();
    const size_t token_num_per_rank = (token_num + world_size - 1) / world_size;
    const size_t offset = token_num_per_rank * rank_ * hidden_dim * sizeof(T);
    auto nvls_mcast_memory = NvlsMcastMemory::GetInstance();
    if (residual_in_ptr == nullptr) {
      RunTokenWeaveFusedAllReduceResidual<T>(nvls_mcast_memory->GetNvlsHandles()[rank_]->mc_ptr + offset,
                                             norm_out_ptr + offset, nvls_mcast_memory->GetSignalPadsDev()[rank_], rank_,
                                             world_size, token_num_per_rank, hidden_dim,
                                             context_->GetComputeStreams()[rank_].Get());
    } else {
      RunTokenWeaveFusedAllReduceResidualNorm<T>(
          nvls_mcast_memory->GetNvlsHandles()[rank_]->mc_ptr + offset, residual_in_ptr + offset, layernorm_weight_ptr,
          nvls_mcast_memory->GetSignalPadsDev()[rank_], rank_, world_size, rms_norm_eps_, token_num_per_rank,
          hidden_dim, context_->GetComputeStreams()[rank_].Get());
    }
    // Multimem allreduce is in-place, requiring an extra copy to the output tensor
    // TODO(yfnjin): Optimize this copy
    if (reinterpret_cast<uintptr_t>(norm_out_ptr) != nvls_mcast_memory->GetNvlsHandles()[rank_]->uc_ptr) {
      MemcpyAsync(norm_out_ptr, reinterpret_cast<void*>(nvls_mcast_memory->GetNvlsHandles()[rank_]->uc_ptr),
                  token_num * hidden_dim * sizeof(T), MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
    }
  } else {
    if (!is_init_) {
      InitTrtAllReduceWorkspace(rank_, context_->ext->GetTrtAllReduceBuffers(), context_->ext->GetTrtAllReduceFlags(),
                                context_->ext->GetTrtAllReduceWorkspaces(), context_->GetComputeStreams()[rank_].Get());
      is_init_ = true;
    }

    RunTrtFusedAllReduceResidualNorm<T>(input, rank_, token_num, hidden_dim, context_->ext->GetTrtAllReduceWorkspaces(),
                                        layernorm_weight_ptr, rms_norm_eps_, residual_in_ptr, residual_in_ptr,
                                        norm_out_ptr, context_->GetComputeStreams()[rank_].Get());
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
