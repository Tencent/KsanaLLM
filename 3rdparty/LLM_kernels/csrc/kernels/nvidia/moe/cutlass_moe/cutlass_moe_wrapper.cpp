/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/moe/cutlass_moe/cutlass_moe_wrapper.h"

#if ENABLE_CUTLASSMOE
#  include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/moeOp.h"
#  include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/utils.h"
#endif
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;
using namespace llm_kernels::nvidia::tensorrt_llm::dev;

namespace llm_kernels {
namespace nvidia {

CutlassMoeWrapper::CutlassMoeWrapper(size_t tp_size, size_t tp_rank, size_t ep_size, size_t ep_rank,
                                     size_t cluster_size, size_t cluster_rank, size_t top_k)
    : tp_size_(tp_size),
      tp_rank_(tp_rank),
      ep_size_(ep_size),
      ep_rank_(ep_rank),
      cluster_size_(cluster_size),
      cluster_rank_(cluster_rank),
      top_k_(top_k) {}

template <typename Atype>
void CutlassMoeWrapper::Init() {
#if ENABLE_CUTLASSMOE
  fused_moe_runner_ = std::make_unique<FusedMoeRunner>(GetScalarType<Atype>(), ScalarType::QUInt4x2,
                                                       GetScalarType<Atype>(), false, true, false);
#else
  KLLM_KERNEL_THROW("ENABLE_CUTLASSMOE=0, skipping Cutlass Moe kernel.");
#endif
}

CutlassMoeWrapper::~CutlassMoeWrapper() = default;

size_t CutlassMoeWrapper::GetWorkspaceSize(size_t num_token, size_t experts_per_token, size_t num_experts_on_rank,
                                           size_t hidden_size, size_t inter_size) {
#if ENABLE_CUTLASSMOE
  return fused_moe_runner_->getRuntimeWorkspaceInfo(experts_per_token, num_token, hidden_size, inter_size,
                                                    num_experts_on_rank, tp_size_, tp_rank_, ep_size_, ep_rank_, false,
                                                    {});
#else
  KLLM_KERNEL_THROW("ENABLE_CUTLASSMOE=0, skipping Cutlass Moe kernel.");
#endif
}

void CutlassMoeWrapper::SetWorkspacePtr(void* workspace_ptr) {
#if ENABLE_CUTLASSMOE
  fused_moe_runner_->setRuntimeWorkspaceInfo(workspace_ptr);
#else
  KLLM_KERNEL_THROW("ENABLE_CUTLASSMOE=0, skipping Cutlass Moe kernel.");
#endif
}

void CutlassMoeWrapper::Forward(Tensor& output, const Tensor& input, const Tensor& token_selected_experts,
                                const std::optional<Tensor>& token_final_scales, const Tensor& fc1_expert_weights,
                                const Tensor& fc2_expert_weights, const std::vector<Tensor>& quant_scales,
                                const std::vector<int64_t>& profile_ids, cudaStream_t stream) {
#if ENABLE_CUTLASSMOE
  fused_moe_runner_->runMoe(output, input, token_selected_experts, token_final_scales, fc1_expert_weights, std::nullopt,
                            fc2_expert_weights, std::nullopt, quant_scales, std::nullopt, tp_size_, tp_rank_, ep_size_,
                            ep_rank_, cluster_size_, cluster_rank_, false, false, profile_ids, stream);
#else
  KLLM_KERNEL_THROW("ENABLE_CUTLASSMOE=0, skipping Cutlass Moe kernel.");
#endif
}

int64_t CutlassMoeWrapper::GetTacticNum() {
#if ENABLE_CUTLASSMOE
  return fused_moe_runner_->getTacticNum();
#else
  KLLM_KERNEL_THROW("ENABLE_CUTLASSMOE=0, skipping Cutlass Moe kernel.");
#endif
}

size_t CutlassMoeWrapper::GetProfileWorkspace(const Tensor& fc1_expert_weights,
                                              const std::optional<Tensor>& fc1_expert_biases,
                                              const Tensor& fc2_expert_weights,
                                              const std::optional<Tensor>& fc2_expert_biases, const int64_t num_rows,
                                              bool const enable_alltoall, bool const min_latency_mode,
                                              int64_t const gemm_idx, int64_t const profile_id,
                                              bool const do_preparation, cudaStream_t stream) {
#if ENABLE_CUTLASSMOE
  return fused_moe_runner_->getProfileWorkspace(fc1_expert_weights, fc1_expert_biases, fc2_expert_weights,
                                                fc2_expert_biases, num_rows, top_k_, tp_size_, tp_rank_, ep_size_,
                                                ep_rank_, cluster_size_, cluster_rank_, enable_alltoall,
                                                min_latency_mode, gemm_idx, profile_id, do_preparation, stream);
#else
  KLLM_KERNEL_THROW("ENABLE_CUTLASSMOE=0, skipping Cutlass Moe kernel.");
#endif
}

void CutlassMoeWrapper::SetProfileWorkspace(void* profile_workspace_ptr, const Tensor& fc1_expert_weights,
                                            const std::optional<Tensor>& fc1_expert_biases,
                                            const Tensor& fc2_expert_weights,
                                            const std::optional<Tensor>& fc2_expert_biases, const int64_t num_rows,
                                            bool const enable_alltoall, bool const min_latency_mode,
                                            int64_t const gemm_idx, int64_t const profile_id, bool const do_preparation,
                                            cudaStream_t stream) {
#if ENABLE_CUTLASSMOE
  fused_moe_runner_->setProfileWorkspace(profile_workspace_ptr, fc1_expert_weights, fc1_expert_biases,
                                         fc2_expert_weights, fc2_expert_biases, num_rows, top_k_, tp_size_, tp_rank_,
                                         ep_size_, ep_rank_, cluster_size_, cluster_rank_, enable_alltoall,
                                         min_latency_mode, gemm_idx, profile_id, do_preparation, stream);
#else
  KLLM_KERNEL_THROW("ENABLE_CUTLASSMOE=0, skipping Cutlass Moe kernel.");
#endif
}

void CutlassMoeWrapper::RunGemmProfile(const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases,
                                       const Tensor& fc2_expert_weights, const std::optional<Tensor>& fc2_expert_biases,
                                       const int64_t num_rows, bool const enable_alltoall, bool const min_latency_mode,
                                       int64_t const gemm_idx, int64_t const profile_id, bool const do_preparation,
                                       cudaStream_t stream) {
#if ENABLE_CUTLASSMOE
  fused_moe_runner_->runGemmProfile(fc1_expert_weights, fc1_expert_biases, fc2_expert_weights, fc2_expert_biases,
                                    num_rows, top_k_, tp_size_, tp_rank_, ep_size_, ep_rank_, cluster_size_,
                                    cluster_rank_, enable_alltoall, min_latency_mode, gemm_idx, profile_id,
                                    do_preparation, stream);
#else
  KLLM_KERNEL_THROW("ENABLE_CUTLASSMOE=0, skipping Cutlass Moe kernel.");
#endif
}

template void CutlassMoeWrapper::Init<float>();
template void CutlassMoeWrapper::Init<__nv_bfloat16>();
template void CutlassMoeWrapper::Init<half>();

}  // namespace nvidia
}  // namespace llm_kernels