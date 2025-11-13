/*
 * Adapted from
 * [TensorRT-LLM Project]
 * https://github.com/NVIDIA/TensorRT-LLM/tree/v1.0.0rc3
 */

/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/include/moe_kernels.h"
// Always include the public header for moe_gemm_kernels.h
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/include/moe_gemm_kernels.h"

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/common/workspace.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/include/cutlass_kernel_selector.h"

#include <fmt/format.h>

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/utils.h"
#include "csrc/utils/nvidia/assert.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels::nvidia::tensorrt_llm::dev {

namespace internal {
namespace common = llm_kernels::nvidia::tensorrt_llm::dev::common;
namespace kernels = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE;
using ActivationType = CUTLASS_MOE_GEMM_NAMESPACE::ActivationType;
// Always use public header as it is just utility functions and types
using TmaWarpSpecializedGroupedGemmInput =
    llm_kernels::nvidia::tensorrt_llm::dev::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;
using profiler_backend = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::GemmProfilerBackend;
}  // namespace internal

struct WorkspaceInfo {
  void* workspace{};
  void* src_to_dest_map{};
};

class FusedMoeRunner {
 public:
  template <typename TypeAct, typename TypeWeight, bool NeedQuant = false>
  std::unique_ptr<internal::kernels::CutlassMoeFCRunnerInterface> switch_output_type(ScalarType output_type);

  FusedMoeRunner(ScalarType activation_dtype, ScalarType weight_dtype, ScalarType output_dtype,
                 bool use_deepseek_fp8_block_scale, bool use_w4a8_group_scaling, bool use_mxfp8_act_scaling);

  FusedMoeRunner(FusedMoeRunner const&) = delete;
  void operator=(FusedMoeRunner const&) = delete;

  size_t getRuntimeWorkspaceInfo(const Tensor& input, const Tensor& token_selected_experts,
                                 const Tensor& fc2_expert_weights, int64_t const tp_size, int64_t const tp_rank,
                                 int64_t const ep_size, int64_t const ep_rank, bool min_latency_mode,
                                 const std::vector<int64_t>& profile_ids);

  size_t getRuntimeWorkspaceInfo(const size_t experts_per_token, const size_t num_rows, const size_t hidden_size,
                                 const size_t inter_size, const size_t num_experts_on_rank, int64_t const tp_size,
                                 int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank,
                                 bool min_latency_mode, const std::vector<int64_t>& profile_ids);

  void setRuntimeWorkspaceInfo(void* workspace_ptr);

  void runMoe(Tensor& output, const Tensor& input, const Tensor& token_selected_experts,
              const std::optional<Tensor>& token_final_scales, const Tensor& fc1_expert_weights,
              const std::optional<Tensor>& fc1_expert_biases, const Tensor& fc2_expert_weights,
              const std::optional<Tensor>& fc2_expert_biases, const std::vector<Tensor>& quant_scales,
              const std::optional<Tensor>& input_sf, int64_t const tp_size, int64_t const tp_rank,
              int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size, int64_t const cluster_rank,
              bool const enable_alltoall, bool min_latency_mode, const std::vector<int64_t>& profile_ids,
              cudaStream_t stream);

  void runMoeMinLantency(Tensor& output, Tensor& num_active_experts_per_node, Tensor& experts_to_token_score,
                         Tensor& active_expert_global_ids, const Tensor& input, const Tensor& token_selected_experts,
                         const std::optional<Tensor>& token_final_scales, const Tensor& fc1_expert_weights,
                         const std::optional<Tensor>& fc1_expert_biases, const Tensor& fc2_expert_weights,
                         const std::optional<Tensor>& fc2_expert_biases, const std::vector<Tensor>& quant_scales,
                         const std::optional<Tensor>& input_sf, int64_t const tp_size, int64_t const tp_rank,
                         int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
                         int64_t const cluster_rank, bool const enable_alltoall, bool min_latency_mode,
                         const std::vector<int64_t>& profile_ids, cudaStream_t stream);

  int64_t getTacticNum();

  size_t getProfileWorkspace(const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases,
                             const Tensor& fc2_expert_weights, const std::optional<Tensor>& fc2_expert_biases,
                             const int64_t num_rows, int64_t const top_k, int64_t const tp_size, int64_t const tp_rank,
                             int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
                             int64_t const cluster_rank, bool const enable_alltoall, bool const min_latency_mode,
                             int64_t const gemm_idx, int64_t const profile_id, bool const do_preparation,
                             cudaStream_t stream);

  void setProfileWorkspace(void* profile_workspace_ptr, const Tensor& fc1_expert_weights,
                           const std::optional<Tensor>& fc1_expert_biases, const Tensor& fc2_expert_weights,
                           const std::optional<Tensor>& fc2_expert_biases, const int64_t num_rows, int64_t const top_k,
                           int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank,
                           int64_t const cluster_size, int64_t const cluster_rank, bool const enable_alltoall,
                           bool const min_latency_mode, int64_t const gemm_idx, int64_t const profile_id,
                           bool const do_preparation, cudaStream_t stream);

  void runGemmProfile(const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases,
                      const Tensor& fc2_expert_weights, const std::optional<Tensor>& fc2_expert_biases,
                      const int64_t num_rows, int64_t const top_k, int64_t const tp_size, int64_t const tp_rank,
                      int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
                      int64_t const cluster_rank, bool const enable_alltoall, bool const min_latency_mode,
                      int64_t const gemm_idx, int64_t const profile_id, bool const do_preparation, cudaStream_t stream);

 private:
  std::shared_ptr<internal::kernels::CutlassMoeFCRunnerInterface> mKernelRunner;
  std::shared_ptr<internal::kernels::GemmProfilerBackend> mProfiler;
  ScalarType mActivationDtype;
  ScalarType mWeightDtype;
  ScalarType mOutputDtype;
  // number of elements packed into the inner dimension of a matrix
  // e.g. 16 nvfp4 elements are packed into a single int64 element
  int64_t mInnerDimMultiplier;
  char* mProfileWorkspace = nullptr;

  bool mUseDeepSeekFP8BlockScaling = false;
  bool mUseW4A8GroupScaling = false;
  bool mUseMxfp8ActScaling = false;

  using Profile = llm_kernels::nvidia::tensorrt_llm::dev::cutlass_extensions::CutlassGemmConfig;
  std::vector<Profile> mAllProfiles;

  WorkspaceInfo runtime_workspace;
  size_t moe_workspace_size;
  size_t src_to_dest_map_size;

  size_t getWorkspaceInfo(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int num_experts,
                          int experts_per_token, internal::ActivationType activation_type,
                          internal::kernels::MOEParallelismConfig const& parallelismConfig, bool min_latency_mode);

  void setRunnerProfiles(const std::vector<int64_t>& profile_ids);

  internal::kernels::QuantParams getQuantParams(int64_t const num_experts_on_rank, int64_t const hidden_size,
                                                int64_t const inter_size,
                                                const std::vector<Tensor>& quant_scales) const;

  bool isFp8Quant() const {
    return !mUseDeepSeekFP8BlockScaling && mActivationDtype == ScalarType::Float8_e4m3fn &&
           mWeightDtype == ScalarType::Float8_e4m3fn;
  }

  bool isNvfp4Quant() const {
    return mWeightDtype == ScalarType::Long &&
           mActivationDtype != ScalarType::Float8_e4m3fn;  // FP8 activation does not use FP4
  }

  bool isInt4Quant() const { return mWeightDtype == ScalarType::QUInt4x2; }

  bool isW4AFp8Quant() const { return mActivationDtype == ScalarType::Float8_e4m3fn && isInt4Quant(); }

  bool isWMxfp4AFp8Quant() const {
    return mActivationDtype == ScalarType::Float8_e4m3fn && mWeightDtype == ScalarType::Long && !mUseMxfp8ActScaling;
  }

  bool isWMxfp4AMxfp8Quant() const {
    return mActivationDtype == ScalarType::Float8_e4m3fn && mWeightDtype == ScalarType::Long && mUseMxfp8ActScaling;
  }

  nvinfer1::DataType GetNvinferDataType(ScalarType scalarType) {
    switch (scalarType) {
      case ScalarType::Long:
        return nvinfer1::DataType::kINT64;
      case ScalarType::Float8_e4m3fn:
        return nvinfer1::DataType::kFP8;
      case ScalarType::QUInt4x2:
        return nvinfer1::DataType::kINT4;
      case ScalarType::Int:
        return nvinfer1::DataType::kINT32;
      case ScalarType::Float:
        return nvinfer1::DataType::kFLOAT;
      case ScalarType::BFloat16:
        return nvinfer1::DataType::kBF16;
      case ScalarType::Float16:
        return nvinfer1::DataType::kHALF;
      default:
        KLLM_KERNEL_THROW("Unknown ScalarType");
    }
  }

  size_t GetElementSize(ScalarType scalarType) {
    switch (scalarType) {
      case ScalarType::Long:
        return 8;
      case ScalarType::Float8_e4m3fn:
        return 1;
      // TODO(jinxcwu) torch.quint4x2不确定是1字节还是0.5字节
      case ScalarType::QUInt4x2:
        return 1;
      case ScalarType::Int:
        return 4;
      case ScalarType::Float:
        return 4;
      case ScalarType::BFloat16:
        return 2;
      case ScalarType::Float16:
        return 2;
      default:
        KLLM_KERNEL_THROW("Unknown ScalarType");
    }
  }
};

}  // namespace llm_kernels::nvidia::tensorrt_llm::dev
