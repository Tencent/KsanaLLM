/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <fmt/format.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

namespace tensorrt_llm {
namespace dev {
class FusedMoeRunner;
struct Tensor;
}  // namespace dev
}  // namespace tensorrt_llm

using namespace llm_kernels::nvidia::tensorrt_llm::dev;

class CutlassMoeWrapper {
 public:
  CutlassMoeWrapper(size_t tp_size, size_t tp_rank, size_t ep_size, size_t ep_rank, size_t cluster_size,
                    size_t cluster_rank, size_t top_k);

  ~CutlassMoeWrapper();

  template <typename Atype>
  void Init();

  size_t GetWorkspaceSize(size_t num_token, size_t experts_per_token, size_t num_experts_on_rank, size_t hidden_size,
                          size_t inter_size);

  void SetWorkspacePtr(void* workspace_ptr);

  void Forward(Tensor& output, const Tensor& input, const Tensor& token_selected_experts,
               const std::optional<Tensor>& token_final_scales, const Tensor& fc1_expert_weights,
               const Tensor& fc2_expert_weights, const std::vector<Tensor>& quant_scales,
               const std::vector<int64_t>& profile_ids, cudaStream_t stream);

  int64_t GetTacticNum();

  size_t GetProfileWorkspace(const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases,
                             const Tensor& fc2_expert_weights, const std::optional<Tensor>& fc2_expert_biases,
                             const int64_t num_rows, bool const enable_alltoall, bool const min_latency_mode,
                             int64_t const gemm_idx, int64_t const profile_id, bool const do_preparation,
                             cudaStream_t stream);

  void SetProfileWorkspace(void* profile_workspace_ptr, const Tensor& fc1_expert_weights,
                           const std::optional<Tensor>& fc1_expert_biases, const Tensor& fc2_expert_weights,
                           const std::optional<Tensor>& fc2_expert_biases, const int64_t num_rows,
                           bool const enable_alltoall, bool const min_latency_mode, int64_t const gemm_idx,
                           int64_t const profile_id, bool const do_preparation, cudaStream_t stream);

  void RunGemmProfile(const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases,
                      const Tensor& fc2_expert_weights, const std::optional<Tensor>& fc2_expert_biases,
                      const int64_t num_rows, bool const enable_alltoall, bool const min_latency_mode,
                      int64_t const gemm_idx, int64_t const profile_id, bool const do_preparation, cudaStream_t stream);

 private:
#if ENABLE_CUTLASSMOE
  std::unique_ptr<FusedMoeRunner> fused_moe_runner_;
#endif

  size_t tp_size_;
  size_t tp_rank_;
  size_t ep_size_;
  size_t ep_rank_;
  size_t cluster_size_;
  size_t cluster_rank_;

  size_t top_k_;
};

}  // namespace nvidia
}  // namespace llm_kernels