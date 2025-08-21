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

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/moeOp.h"

namespace llm_kernels::nvidia::tensorrt_llm::dev {

template <typename TypeAct, typename TypeWeight, bool NeedQuant>
std::unique_ptr<internal::kernels::CutlassMoeFCRunnerInterface> FusedMoeRunner::switch_output_type(
    ScalarType output_type) {
  switch (output_type) {
    case ScalarType::Long:  // INT64 == FP4
    case ScalarType::Float8_e4m3fn:
      // TODO We need an atomic FP8 reduction for the finalize fusions
      KLLM_KERNEL_THROW(fmt::format("Outputting {} directly is not currently supported", output_type));
      // return std::make_unique<internal::kernels::CutlassMoeFCRunner<Type, Type>>();
    case ScalarType::Half:
      if constexpr (NeedQuant) {
        return std::make_unique<internal::kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, half, half>>();
      } else {
        return std::make_unique<internal::kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, half, TypeAct>>();
      }
    case ScalarType::BFloat16:
      if constexpr (NeedQuant) {
        return std::make_unique<
            internal::kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, __nv_bfloat16, __nv_bfloat16>>();
      } else {
        return std::make_unique<internal::kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, __nv_bfloat16, TypeAct>>();
      }
    default:
      KLLM_KERNEL_THROW(fmt::format("Invalid output type {} specified for {}", output_type, mActivationDtype));
  }
}

FusedMoeRunner::FusedMoeRunner(ScalarType activation_dtype, ScalarType weight_dtype, ScalarType output_dtype,
                               bool use_deepseek_fp8_block_scale, bool use_w4a8_group_scaling,
                               bool use_mxfp8_act_scaling) {
  mActivationDtype = activation_dtype;
  mWeightDtype = weight_dtype;
  mOutputDtype = output_dtype;
  mUseDeepSeekFP8BlockScaling = use_deepseek_fp8_block_scale;
  mUseW4A8GroupScaling = use_w4a8_group_scaling;
  mUseMxfp8ActScaling = use_mxfp8_act_scaling;
  mInnerDimMultiplier = 1;

  // keep consistent with cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp
  if (mActivationDtype == ScalarType::Half && mWeightDtype == ScalarType::Half) {
    mKernelRunner = std::make_shared<internal::kernels::CutlassMoeFCRunner<half, half>>();
  } else if (mActivationDtype == ScalarType::BFloat16 && mWeightDtype == ScalarType::BFloat16) {
    mKernelRunner = std::make_shared<internal::kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
  }
#ifdef ENABLE_FP8
  else if (mActivationDtype == ScalarType::BFloat16 && mWeightDtype == ScalarType::Float8_e4m3fn) {
    mKernelRunner = std::make_unique<internal::kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3>>();
  }
#endif

#ifdef ENABLE_FP8
  if (isFp8Quant()) {
    mKernelRunner = switch_output_type<__nv_fp8_e4m3, __nv_fp8_e4m3>(mOutputDtype);
  }
#endif
#ifdef ENABLE_FP4
  if (isWMxfp4AMxfp8Quant() || isWMxfp4AFp8Quant()) {
    mInnerDimMultiplier = 16;  // 16 FP4 -> 1 LONG
    mKernelRunner = switch_output_type<__nv_fp8_e4m3, __nv_fp4_e2m1>(mOutputDtype);
  }

  if (isNvfp4Quant()) {
    mInnerDimMultiplier = 16;
    switch (mActivationDtype) {
      case ScalarType::Half:
      case ScalarType::BFloat16:
        mKernelRunner = switch_output_type<__nv_fp4_e2m1, __nv_fp4_e2m1, true>(mOutputDtype);
        break;
      default:
        mKernelRunner = switch_output_type<__nv_fp4_e2m1, __nv_fp4_e2m1, false>(mOutputDtype);
    }
  }
#endif
  if (isInt4Quant()) {
    mInnerDimMultiplier = 2;
    if (mActivationDtype == ScalarType::Half) {
#ifdef ENABLE_FP8
      if (mUseW4A8GroupScaling) {
        mKernelRunner =
            std::make_unique<internal::kernels::CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>>();
      } else {
        mKernelRunner = std::make_shared<internal::kernels::CutlassMoeFCRunner<half, cutlass::uint4b_t>>();
      }
#else
      mKernelRunner = std::make_shared<internal::kernels::CutlassMoeFCRunner<half, cutlass::uint4b_t>>();
#endif
    } else if (mActivationDtype == ScalarType::BFloat16) {
#ifdef ENABLE_FP8
      if (mUseW4A8GroupScaling) {
        mKernelRunner = std::make_unique<
            internal::kernels::CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16, __nv_bfloat16>>();
      } else {
        mKernelRunner = std::make_shared<internal::kernels::CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>>();
      }
#else
      mKernelRunner = std::make_shared<internal::kernels::CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>>();
#endif
    }
  }
  if (!mKernelRunner) {
    KLLM_KERNEL_THROW(
        fmt::format("Could not construct fused moe op with the requested input combination Activation: {}, Weight: "
                    "{}, Output: {}",
                    mActivationDtype, mWeightDtype, mOutputDtype));
  }

  mProfiler = std::make_shared<internal::kernels::GemmProfilerBackend>();
  mAllProfiles = mKernelRunner->getTactics();
}

size_t FusedMoeRunner::getRuntimeWorkspaceInfo(const Tensor& input, const Tensor& token_selected_experts,
                                               const Tensor& fc2_expert_weights, int64_t const tp_size,
                                               int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank,
                                               bool min_latency_mode, const std::vector<int64_t>& profile_ids) {
  int experts_per_token = token_selected_experts.shape[1];
  int64_t num_rows = input.shape[0];
  int64_t hidden_size = fc2_expert_weights.shape[1];
  int64_t inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;

  int const num_experts_on_rank = fc2_expert_weights.shape[0];
  auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
  auto parallelism_config = internal::kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
  auto activation_type = internal::ActivationType::Swiglu;

  setRunnerProfiles(profile_ids);

  return getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
                          activation_type, parallelism_config, min_latency_mode);
}

void FusedMoeRunner::runMoe(Tensor& output, const Tensor& input, const Tensor& token_selected_experts,
                            const std::optional<Tensor>& token_final_scales, const Tensor& fc1_expert_weights,
                            const std::optional<Tensor>& fc1_expert_biases, const Tensor& fc2_expert_weights,
                            const std::optional<Tensor>& fc2_expert_biases, const std::vector<Tensor>& quant_scales,
                            const std::optional<Tensor>& input_sf, int64_t const tp_size, int64_t const tp_rank,
                            int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
                            int64_t const cluster_rank, bool const enable_alltoall, bool min_latency_mode,
                            const std::vector<int64_t>& profile_ids, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mMutex);

  KLLM_KERNEL_CHECK_WITH_INFO(cluster_size == 1 && cluster_rank == 0, "smart_router is supported in min_latency mode");

  KLLM_KERNEL_CHECK(input.dtype == mActivationDtype);
  KLLM_KERNEL_CHECK(token_selected_experts.dtype == ScalarType::Int);
  if (token_final_scales.has_value()) {
    KLLM_KERNEL_CHECK(token_final_scales.value().dtype == ScalarType::Float);
  }
  KLLM_KERNEL_CHECK(fc1_expert_weights.dtype == mWeightDtype);
  KLLM_KERNEL_CHECK(fc2_expert_weights.dtype == mWeightDtype);

  KLLM_KERNEL_CHECK_WITH_INFO(input.shape.size() == 2, "input must be 2D.");
  KLLM_KERNEL_CHECK_WITH_INFO(token_selected_experts.shape.size() == 2, "token_selected_experts must be 2D.");

  KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_weights.shape.size() == 3, "fc1_expert_weights must be 3D.");
  KLLM_KERNEL_CHECK_WITH_INFO(fc2_expert_weights.shape.size() == 3, "fc2_expert_weights must be 3D.");

  if (fc1_expert_biases.has_value() || fc2_expert_biases.has_value()) {
    KLLM_KERNEL_CHECK(fc1_expert_biases.value().dtype == mOutputDtype);
    KLLM_KERNEL_CHECK(fc2_expert_biases.value().dtype == mOutputDtype);
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_biases.value().shape.size() == 2, "fc1_expert_biases must be 2D.");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_expert_biases.value().shape.size() == 2, "fc2_expert_biases must be 2D.");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_weights.shape[0] == fc1_expert_biases.value().shape[0],
                                "fc1_expert_weights and fc1_expert_biases must have the same number of experts.");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_expert_weights.shape[0] == fc2_expert_biases.value().shape[0],
                                "fc2_expert_weights and fc2_expert_biases must have the same number of experts.");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_biases.value().shape[1] == fc1_expert_weights.shape[1],
                                "fc1_expert_biases should match fc1_expert_weights output shape.");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_expert_biases.value().shape[1] == fc2_expert_weights.shape[1],
                                "fc2_expert_biases should match fc2_expert_weights output shape.");
  }

  KLLM_KERNEL_CHECK_WITH_INFO(input.shape[0] == token_selected_experts.shape[0],
                              "input and token_selected_experts must have the same num tokens.");
  if (token_final_scales.has_value()) {
    KLLM_KERNEL_CHECK_WITH_INFO(token_final_scales.value().shape.size() == 2,
                                "token_selected_experts_probs must be 2D.");
    KLLM_KERNEL_CHECK_WITH_INFO(input.shape[0] == token_final_scales.value().shape[0],
                                "input and token_selected_experts_probs must have the same num tokens.");
    KLLM_KERNEL_CHECK_WITH_INFO(
        token_selected_experts.shape[1] == token_final_scales.value().shape[1],
        "token_selected_experts and token_final_scales must have the same number of experts per token.");
  }
  KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_weights.shape[0] == fc2_expert_weights.shape[0],
                              "fc1_expert_weights and fc2_expert_weights must have the same number of experts.");
  KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_weights.shape[1] == fc2_expert_weights.shape[2] * mInnerDimMultiplier * 2,
                              "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.");

  int experts_per_token = token_selected_experts.shape[1];
  int64_t num_rows = input.shape[0];
  int64_t hidden_size = fc2_expert_weights.shape[1];
  int64_t inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;

  if (isWMxfp4AMxfp8Quant() || isWMxfp4AFp8Quant()) {
    // MXFP4 weights are required to bealigned to 128 bytes
    KLLM_KERNEL_CHECK_WITH_INFO(hidden_size % 128 == 0, "hidden_size must be divisible by 128 for MXFP4 weights");
    KLLM_KERNEL_CHECK_WITH_INFO(inter_size % 128 == 0, "inter_size must be divisible by 128 for MXFP4 weights");
  } else {
    // TMA requires at least 128 bit alignment
    auto min_alignment = 128 / (8 * std::min(GetElementSize(mActivationDtype), GetElementSize(mWeightDtype)));
    KLLM_KERNEL_CHECK_WITH_INFO(hidden_size % min_alignment == 0, "hidden_size ", hidden_size, " must be divisible by ",
                                min_alignment, " for weights");
    KLLM_KERNEL_CHECK_WITH_INFO(inter_size % min_alignment == 0, "inter_size ", inter_size, " must be divisible by ",
                                min_alignment, " for weights");
  }

  int const num_experts_on_rank = fc2_expert_weights.shape[0];
  auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
  auto parallelism_config = internal::kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
  auto activation_type = internal::ActivationType::Swiglu;

  setRunnerProfiles(profile_ids);

  output.shape = {static_cast<size_t>(num_rows), static_cast<size_t>(hidden_size)};
  output.dtype = mOutputDtype;

  auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);
  internal::kernels::MoeMinLatencyParams min_latency_params{};

  // TODO: support lora in the future
  ::llm_kernels::nvidia::tensorrt_llm::dev::kernels::LoraParams lora_params{};
  mKernelRunner->runMoe(
      input.data, input_sf.has_value() ? input_sf.value().data : nullptr,
      reinterpret_cast<int const*>(token_selected_experts.data),
      token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().data) : nullptr,
      fc1_expert_weights.data, fc1_expert_biases.has_value() ? fc1_expert_biases.value().data : nullptr,
      activation_type, fc2_expert_weights.data,
      fc2_expert_biases.has_value() ? fc2_expert_biases.value().data : nullptr, quant_params, num_rows, hidden_size,
      inter_size, num_experts_total, static_cast<int>(experts_per_token),
      static_cast<char*>(runtime_workspace.workspace), output.data,
      static_cast<int*>(runtime_workspace.src_to_dest_map), parallelism_config, enable_alltoall, false, lora_params,
      mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
}

void FusedMoeRunner::runMoeMinLantency(Tensor& output, Tensor& num_active_experts_per_node,
                                       Tensor& experts_to_token_score, Tensor& active_expert_global_ids,
                                       const Tensor& input, const Tensor& token_selected_experts,
                                       const std::optional<Tensor>& token_final_scales,
                                       const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases,
                                       const Tensor& fc2_expert_weights, const std::optional<Tensor>& fc2_expert_biases,
                                       const std::vector<Tensor>& quant_scales, const std::optional<Tensor>& input_sf,
                                       int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size,
                                       int64_t const ep_rank, int64_t const cluster_size, int64_t const cluster_rank,
                                       bool const enable_alltoall, bool min_latency_mode,
                                       const std::vector<int64_t>& profile_ids, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mMutex);

  KLLM_KERNEL_CHECK(input.dtype == mActivationDtype);
  KLLM_KERNEL_CHECK(token_selected_experts.dtype == ScalarType::Int);
  if (token_final_scales.has_value()) {
    KLLM_KERNEL_CHECK(token_final_scales.value().dtype == ScalarType::Float);
  }
  KLLM_KERNEL_CHECK(fc1_expert_weights.dtype == mWeightDtype);
  KLLM_KERNEL_CHECK(fc2_expert_weights.dtype == mWeightDtype);

  KLLM_KERNEL_CHECK_WITH_INFO(input.shape.size() == 2, "input must be 2D.");
  KLLM_KERNEL_CHECK_WITH_INFO(token_selected_experts.shape.size() == 2, "token_selected_experts must be 2D.");

  KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_weights.shape.size() == 3, "fc1_expert_weights must be 3D.");
  KLLM_KERNEL_CHECK_WITH_INFO(fc2_expert_weights.shape.size() == 3, "fc2_expert_weights must be 3D.");

  if (fc1_expert_biases.has_value() || fc2_expert_biases.has_value()) {
    KLLM_KERNEL_CHECK(fc1_expert_biases.value().dtype == mOutputDtype);
    KLLM_KERNEL_CHECK(fc2_expert_biases.value().dtype == mOutputDtype);
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_biases.value().shape.size() == 2, "fc1_expert_biases must be 2D.");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_expert_biases.value().shape.size() == 2, "fc2_expert_biases must be 2D.");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_weights.shape[0] == fc1_expert_biases.value().shape[0],
                                "fc1_expert_weights and fc1_expert_biases must have the same number of experts.");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_expert_weights.shape[0] == fc2_expert_biases.value().shape[0],
                                "fc2_expert_weights and fc2_expert_biases must have the same number of experts.");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_biases.value().shape[1] == fc1_expert_weights.shape[1],
                                "fc1_expert_biases should match fc1_expert_weights output shape.");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_expert_biases.value().shape[1] == fc2_expert_weights.shape[1],
                                "fc2_expert_biases should match fc2_expert_weights output shape.");
  }

  KLLM_KERNEL_CHECK_WITH_INFO(input.shape[0] == token_selected_experts.shape[0],
                              "input and token_selected_experts must have the same num tokens.");
  if (token_final_scales.has_value()) {
    KLLM_KERNEL_CHECK_WITH_INFO(token_final_scales.value().shape.size() == 2,
                                "token_selected_experts_probs must be 2D.");
    KLLM_KERNEL_CHECK_WITH_INFO(input.shape[0] == token_final_scales.value().shape[0],
                                "input and token_selected_experts_probs must have the same num tokens.");
    KLLM_KERNEL_CHECK_WITH_INFO(
        token_selected_experts.shape[1] == token_final_scales.value().shape[1],
        "token_selected_experts and token_final_scales must have the same number of experts per token.");
  }
  KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_weights.shape[0] == fc2_expert_weights.shape[0],
                              "fc1_expert_weights and fc2_expert_weights must have the same number of experts.");
  KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_weights.shape[1] == fc2_expert_weights.shape[2] * mInnerDimMultiplier * 2,
                              "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.");

  KLLM_KERNEL_CHECK_WITH_INFO(!input_sf.has_value() || isWMxfp4AMxfp8Quant() || isNvfp4Quant(),
                              "Block-scaling factors provided for non block-scaling quantization");

  int experts_per_token = token_selected_experts.shape[1];
  int64_t num_rows = input.shape[0];
  int64_t hidden_size = fc2_expert_weights.shape[1];
  int64_t inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
  int const num_experts_on_rank = fc2_expert_weights.shape[0];
  auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
  auto parallelism_config =
      internal::kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank, cluster_size, cluster_rank);
  auto activation_type = internal::ActivationType::Swiglu;

  setRunnerProfiles(profile_ids);

  output.shape = {static_cast<size_t>(num_rows * num_experts_on_rank), static_cast<size_t>(hidden_size)};
  output.dtype = mOutputDtype;

  num_active_experts_per_node.shape = {1};
  num_active_experts_per_node.dtype = ScalarType::Int;

  experts_to_token_score.shape = {static_cast<size_t>(num_experts_on_rank), static_cast<size_t>(num_rows)};
  experts_to_token_score.dtype = ScalarType::Float;

  active_expert_global_ids.shape = {static_cast<size_t>(num_experts_on_rank)};
  active_expert_global_ids.dtype = ScalarType::Int;

  internal::kernels::MoeMinLatencyParams min_latency_params{};
  min_latency_params.num_active_experts_per_node = static_cast<int*>(num_active_experts_per_node.data);
  min_latency_params.experts_to_token_score = static_cast<float*>(experts_to_token_score.data);
  min_latency_params.active_expert_global_ids = static_cast<int*>(active_expert_global_ids.data);

  auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);

  // TODO: support lora in the future
  ::llm_kernels::nvidia::tensorrt_llm::dev::kernels::LoraParams lora_params{};
  mKernelRunner->runMoe(
      input.data, input_sf.has_value() ? input_sf.value().data : nullptr,
      reinterpret_cast<int const*>(token_selected_experts.data),
      token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().data) : nullptr,
      fc1_expert_weights.data, fc1_expert_biases.has_value() ? fc1_expert_biases.value().data : nullptr,
      activation_type, fc2_expert_weights.data,
      fc2_expert_biases.has_value() ? fc2_expert_biases.value().data : nullptr, quant_params, num_rows, hidden_size,
      inter_size, num_experts_total, static_cast<int>(experts_per_token),
      static_cast<char*>(runtime_workspace.workspace), output.data,
      static_cast<int*>(runtime_workspace.src_to_dest_map), parallelism_config, enable_alltoall, false, lora_params,
      mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
}

int64_t FusedMoeRunner::getTacticNum() {
  std::lock_guard<std::mutex> lock(mMutex);
  return mAllProfiles.size();
}

size_t FusedMoeRunner::getProfileWorkspace(
    const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases, const Tensor& fc2_expert_weights,
    const std::optional<Tensor>& fc2_expert_biases, const int64_t num_rows, int64_t const top_k, int64_t const tp_size,
    int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
    int64_t const cluster_rank, bool const enable_alltoall, bool const min_latency_mode, int64_t const gemm_idx,
    int64_t const profile_id, bool const do_preparation, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mMutex);

  // TODO: support profiling under fp8 block scaling in the future
  if (mUseDeepSeekFP8BlockScaling) {
    return 0;
  }

  int64_t const hidden_size = fc2_expert_weights.shape[1];
  int64_t const inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
  int64_t const group_size = isInt4Quant() ? 128 : -1;
  int const num_experts = static_cast<int>(fc2_expert_weights.shape[0] * ep_size);

  // Get specific profile configs according to the profile_id.
  // Fallback tactic is set to be 0
  // TODO: use the best tactic id found offline for a better default inference perf
  auto const& profile = profile_id == -1 ? mAllProfiles.front() : mAllProfiles[profile_id];

  auto const* expert_weights_ptr = (gemm_idx == 1) ? fc1_expert_weights.data : fc2_expert_weights.data;

  // Preparation phase, only enabled during autotuning warmup phase.
  // Set profiled gemm idx
  mProfiler->mGemmToProfile = (gemm_idx == 1) ? internal::profiler_backend::GemmToProfile::GEMM_1
                                              : internal::profiler_backend::GemmToProfile::GEMM_2;

  // mProfiler init
  auto parallelism_config = internal::kernels::MOEParallelismConfig(
      static_cast<int>(tp_size), static_cast<int>(tp_rank), static_cast<int>(ep_size), static_cast<int>(ep_rank),
      static_cast<int>(cluster_size), static_cast<int>(cluster_rank));

  bool const USE_BIAS = fc1_expert_biases.has_value() || fc2_expert_biases.has_value();
  bool const USE_LORA = false;
  auto activation_dtype = mUseW4A8GroupScaling ? ScalarType::Float8_e4m3fn : mActivationDtype;
  activation_dtype = isNvfp4Quant() ? ScalarType::Long : activation_dtype;
  mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile, GetNvinferDataType(activation_dtype),
                  GetNvinferDataType(mWeightDtype), GetNvinferDataType(mOutputDtype), num_experts,
                  static_cast<int>(top_k), hidden_size, inter_size, group_size, internal::ActivationType::Swiglu,
                  USE_BIAS, USE_LORA, min_latency_mode,
                  /*need_weights*/ false, parallelism_config, enable_alltoall);

  return mProfiler->getWorkspaceSize(num_rows);
}

void FusedMoeRunner::setProfileWorkspace(
    void* profile_workspace_ptr, const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases,
    const Tensor& fc2_expert_weights, const std::optional<Tensor>& fc2_expert_biases, const int64_t num_rows,
    int64_t const top_k, int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank,
    int64_t const cluster_size, int64_t const cluster_rank, bool const enable_alltoall, bool const min_latency_mode,
    int64_t const gemm_idx, int64_t const profile_id, bool const do_preparation, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mMutex);

  // TODO: support profiling under fp8 block scaling in the future
  if (mUseDeepSeekFP8BlockScaling) {
    return;
  }

  int64_t const hidden_size = fc2_expert_weights.shape[1];
  int64_t const inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
  int64_t const group_size = isInt4Quant() ? 128 : -1;
  int const num_experts = static_cast<int>(fc2_expert_weights.shape[0] * ep_size);

  // Get specific profile configs according to the profile_id.
  // Fallback tactic is set to be 0
  // TODO: use the best tactic id found offline for a better default inference perf
  auto const& profile = profile_id == -1 ? mAllProfiles.front() : mAllProfiles[profile_id];

  auto const* expert_weights_ptr = (gemm_idx == 1) ? fc1_expert_weights.data : fc2_expert_weights.data;

  // Preparation phase, only enabled during autotuning warmup phase.
  // Set profiled gemm idx
  mProfiler->mGemmToProfile = (gemm_idx == 1) ? internal::profiler_backend::GemmToProfile::GEMM_1
                                              : internal::profiler_backend::GemmToProfile::GEMM_2;

  // mProfiler init
  auto parallelism_config = internal::kernels::MOEParallelismConfig(
      static_cast<int>(tp_size), static_cast<int>(tp_rank), static_cast<int>(ep_size), static_cast<int>(ep_rank),
      static_cast<int>(cluster_size), static_cast<int>(cluster_rank));

  bool const USE_BIAS = fc1_expert_biases.has_value() || fc2_expert_biases.has_value();
  bool const USE_LORA = false;
  auto activation_dtype = mUseW4A8GroupScaling ? ScalarType::Float8_e4m3fn : mActivationDtype;
  activation_dtype = isNvfp4Quant() ? ScalarType::Long : activation_dtype;
  mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile, GetNvinferDataType(activation_dtype),
                  GetNvinferDataType(mWeightDtype), GetNvinferDataType(mOutputDtype), num_experts,
                  static_cast<int>(top_k), hidden_size, inter_size, group_size, internal::ActivationType::Swiglu,
                  USE_BIAS, USE_LORA, min_latency_mode,
                  /*need_weights*/ false, parallelism_config, enable_alltoall);

  mProfileWorkspace = static_cast<char*>(profile_workspace_ptr);

  mProfiler->prepare(num_rows, mProfileWorkspace, expert_weights_ptr, stream);

  // Profile specific tactic. Assuming at least one preparation phase has been executed already.
  mProfiler->runProfiler(num_rows, profile, mProfileWorkspace, expert_weights_ptr, stream);
}

void FusedMoeRunner::runGemmProfile(const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases,
                                    const Tensor& fc2_expert_weights, const std::optional<Tensor>& fc2_expert_biases,
                                    const int64_t num_rows, int64_t const top_k, int64_t const tp_size,
                                    int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank,
                                    int64_t const cluster_size, int64_t const cluster_rank, bool const enable_alltoall,
                                    bool const min_latency_mode, int64_t const gemm_idx, int64_t const profile_id,
                                    bool const do_preparation, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mMutex);

  // TODO: support profiling under fp8 block scaling in the future
  if (mUseDeepSeekFP8BlockScaling) {
    return;
  }

  int64_t const hidden_size = fc2_expert_weights.shape[1];
  int64_t const inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
  int64_t const group_size = isInt4Quant() ? 128 : -1;
  int const num_experts = static_cast<int>(fc2_expert_weights.shape[0] * ep_size);

  // Get specific profile configs according to the profile_id.
  // Fallback tactic is set to be 0
  // TODO: use the best tactic id found offline for a better default inference perf
  auto const& profile = profile_id == -1 ? mAllProfiles.front() : mAllProfiles[profile_id];

  auto const* expert_weights_ptr = (gemm_idx == 1) ? fc1_expert_weights.data : fc2_expert_weights.data;

  // Profile specific tactic. Assuming at least one preparation phase has been executed already.
  mProfiler->runProfiler(num_rows, profile, mProfileWorkspace, expert_weights_ptr, stream);
}

void FusedMoeRunner::setRunnerProfiles(const std::vector<int64_t>& profile_ids) {
  if (mUseDeepSeekFP8BlockScaling) {
    auto config = llm_kernels::nvidia::tensorrt_llm::dev::cutlass_extensions::CutlassGemmConfig(
        llm_kernels::nvidia::tensorrt_llm::dev::cutlass_extensions::CutlassTileConfigSM90::CtaShape128x16x128B,
        llm_kernels::nvidia::tensorrt_llm::dev::cutlass_extensions::MainloopScheduleType::AUTO,
        llm_kernels::nvidia::tensorrt_llm::dev::cutlass_extensions::EpilogueScheduleType::AUTO,
        llm_kernels::nvidia::tensorrt_llm::dev::cutlass_extensions::ClusterShape::ClusterShape_1x1x1);
    mKernelRunner->setTactic(config, config);
    return;
  }

  auto best_gemm1_profile = mAllProfiles.front();
  auto best_gemm2_profile = mAllProfiles.front();
  if (!profile_ids.empty()) {
    KLLM_KERNEL_CHECK_WITH_INFO(profile_ids.size() == 2, "Expecting 2 profile ids");
    best_gemm1_profile = profile_ids[0] == -1 ? best_gemm1_profile : mAllProfiles.at(profile_ids[0]);
    best_gemm2_profile = profile_ids[1] == -1 ? best_gemm2_profile : mAllProfiles.at(profile_ids[1]);
  }
  mKernelRunner->setTactic(best_gemm1_profile, best_gemm2_profile);
}

size_t FusedMoeRunner::getWorkspaceInfo(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                                        int num_experts, int experts_per_token,
                                        internal::ActivationType activation_type,
                                        internal::kernels::MOEParallelismConfig const& parallelismConfig,
                                        bool min_latency_mode) {
  moe_workspace_size = mKernelRunner->getWorkspaceSize(
      num_rows, hidden_size, inter_size, num_experts, experts_per_token, activation_type, parallelismConfig,
      /* use_lora */ false, mUseDeepSeekFP8BlockScaling, min_latency_mode, mUseW4A8GroupScaling);
  src_to_dest_map_size = experts_per_token * num_rows * sizeof(int);

  std::vector<size_t> workspaces{moe_workspace_size, src_to_dest_map_size};

  return internal::common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
}

void FusedMoeRunner::setRuntimeWorkspaceInfo(void* workspace_ptr) {
  runtime_workspace.workspace = workspace_ptr;
  runtime_workspace.src_to_dest_map =
      internal::common::nextWorkspacePtr(static_cast<int8_t*>(runtime_workspace.workspace), moe_workspace_size);
}

internal::kernels::QuantParams FusedMoeRunner::getQuantParams(int64_t const num_experts_on_rank,
                                                              int64_t const hidden_size, int64_t const inter_size,
                                                              const std::vector<Tensor>& quant_scales) const {
  if (isFp8Quant()) {
    KLLM_KERNEL_CHECK_WITH_INFO(!quant_scales.empty(), "Expecting quant scales for fp8 quantization");
    KLLM_KERNEL_CHECK_WITH_INFO(quant_scales.size() == 4, "Expecting 4 quant scales for fp8 quantization");

    auto const fc1_dequant = quant_scales[0];
    auto const fc2_quant = quant_scales[1];
    auto const fc2_dequant = quant_scales[2];
    auto const fc1_input_dequant = quant_scales[3];

    // Check types
    KLLM_KERNEL_CHECK(fc1_dequant.dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK(fc2_quant.dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK(fc2_dequant.dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK(fc1_input_dequant.dtype == ScalarType::Float);
    // Check ranks
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_dequant.shape.size() == 1, "fc1 dequant must be 1D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_quant.shape.size() == 0 || fc2_quant.shape.size() == 1,
                                "fc2 quant must be a scalar or 1-D tensor");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_dequant.shape.size() == 1, "fc2 quant must be 1D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_input_dequant.shape.size() == 0, "fc1 input dequant must be a scalar tensor");
    // Check shapes
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_dequant.shape[0] == num_experts_on_rank,
                                "fc1 dequant size must be (num_experts_on_rank,)");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_quant.shape.size() == 0 || fc2_quant.shape[0] == num_experts_on_rank,
                                "fc2 quant must be scalar or (num_experts_on_rank,)");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_dequant.shape[0] == num_experts_on_rank,
                                "fc2 dequant size must be (num_experts_on_rank,)");

    return internal::kernels::QuantParams::FP8(
        static_cast<float const*>(fc1_dequant.data), static_cast<float const*>(fc2_quant.data),
        static_cast<float const*>(fc2_dequant.data),
        /* fp8 output quant scale */ nullptr, static_cast<float const*>(fc1_input_dequant.data),
        fc2_quant.shape.size() == 1);
  } else if (isWMxfp4AFp8Quant()) {
    KLLM_KERNEL_CHECK_WITH_INFO(!quant_scales.empty(), "Expecting quant scales for W4A8_MXFP4_MXF8 quantization");
    KLLM_KERNEL_CHECK_WITH_INFO(quant_scales.size() == 5, "Expecting 5 quant scales for W4A8_MXFP4_FP8 quantization");

    auto const fc1_weight_block = quant_scales[0];
    auto const fc1_global = quant_scales[1];
    auto const fc2_act_global = quant_scales[2];
    auto const fc2_weight_block = quant_scales[3];
    auto const fc2_global = quant_scales[4];

    // The input for scale fc1_weight_block / fc2_weight_block is packed into INT32
    constexpr int FP8_PER_INT32 = 4;
    // Check types
    KLLM_KERNEL_CHECK(fc1_weight_block.dtype == ScalarType::Int);
    KLLM_KERNEL_CHECK(fc1_global.dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK(fc2_act_global.dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK(fc2_weight_block.dtype == ScalarType::Int);
    KLLM_KERNEL_CHECK(fc2_global.dtype == ScalarType::Float);
    // Check ranks
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_weight_block.shape.size() == 3, "fc1 weight block must be #D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_global.shape.size() == 1, "fc1 global must be 1D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_act_global.shape.size() == 0 || fc2_act_global.shape.size() == 1,
                                "fc2 act global must be a scalar or 1-D tensor");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_weight_block.shape.size() == 3, "fc2 weight block must be 3D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_global.shape.size() == 1, "fc2 global must be 1D");
    // Check shapes
    KLLM_KERNEL_CHECK_WITH_INFO(
        fc1_weight_block.shape[0] == num_experts_on_rank &&
            fc1_weight_block.shape[1] ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    inter_size, internal::TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) *
                    2 &&
            fc1_weight_block.shape[2] * FP8_PER_INT32 *
                    internal::TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    hidden_size, internal::TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX),
        "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // 4 // "
        "block_scale_vector_size)");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_global.shape[0] == num_experts_on_rank,
                                "fc1 global size must be (num_experts_on_rank,)");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_act_global.shape.size() == 0 || fc2_act_global.shape[0] == num_experts_on_rank,
                                "fc2 act global must be scalar or (num_experts_on_rank,)");
    KLLM_KERNEL_CHECK_WITH_INFO(
        fc2_weight_block.shape[0] == num_experts_on_rank &&
            fc2_weight_block.shape[1] ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    hidden_size, internal::TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) &&
            fc2_weight_block.shape[2] * FP8_PER_INT32 *
                    internal::TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    inter_size, internal::TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX),
        "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
        "block_scale_vector_size)");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_global.shape[0] == num_experts_on_rank,
                                "fc2 global size must be (num_experts_on_rank,)");

    return internal::kernels::QuantParams::FP8MXFP4(
        nullptr, static_cast<internal::TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data),
        static_cast<float const*>(fc1_global.data), static_cast<float const*>(fc2_act_global.data),
        static_cast<internal::TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data),
        static_cast<float const*>(fc2_global.data), false, fc2_act_global.shape.size() == 1);
  } else if (isWMxfp4AMxfp8Quant()) {
    KLLM_KERNEL_CHECK_WITH_INFO(!quant_scales.empty(), "Expecting quant scales for W4A8_MXFP4_MXFP8 quantization");
    KLLM_KERNEL_CHECK_WITH_INFO(quant_scales.size() == 4, "Expecting 4 quant scales for W4A8_MXFP4_MXFP8 quantization");

    auto const fc1_weight_block = quant_scales[0];
    auto const fc1_global = quant_scales[1];
    auto const fc2_weight_block = quant_scales[2];
    auto const fc2_global = quant_scales[3];

    // The input for scale fc1_weight_block / fc2_weight_block is packed into INT32
    constexpr int FP8_PER_INT32 = 4;
    KLLM_KERNEL_CHECK(fc1_weight_block.dtype == ScalarType::Int);
    KLLM_KERNEL_CHECK(fc1_global.dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK(fc2_weight_block.dtype == ScalarType::Int);
    KLLM_KERNEL_CHECK(fc2_global.dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_weight_block.shape.size() == 3, "fc1 weight block must be #D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_global.shape.size() == 1, "fc1 global must be 1D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_weight_block.shape.size() == 3, "fc2 weight block must be 3D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_global.shape.size() == 1, "fc2 global must be 1D");
    KLLM_KERNEL_CHECK_WITH_INFO(
        fc1_weight_block.shape[0] == num_experts_on_rank &&
            fc1_weight_block.shape[1] ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    inter_size, internal::TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) *
                    2 &&
            fc1_weight_block.shape[2] * FP8_PER_INT32 *
                    internal::TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    hidden_size, internal::TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX) *
                    internal::TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX,
        "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // 4 // "
        "block_scale_vector_size)");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_global.shape[0] == num_experts_on_rank,
                                "fc1 global size must be (num_experts_on_rank,)");
    KLLM_KERNEL_CHECK_WITH_INFO(
        fc2_weight_block.shape[0] == num_experts_on_rank &&
            fc2_weight_block.shape[1] ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    hidden_size, internal::TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) &&
            fc2_weight_block.shape[2] * FP8_PER_INT32 *
                    internal::TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    inter_size, internal::TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX),
        "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
        "block_scale_vector_size)");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_global.shape[0] == num_experts_on_rank,
                                "fc2 global size must be (num_experts_on_rank,)");

    return internal::kernels::QuantParams::MXFP8MXFP4(
        static_cast<internal::TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data),
        static_cast<float const*>(fc1_global.data),
        static_cast<internal::TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data),
        static_cast<float const*>(fc2_global.data));
  } else if (isNvfp4Quant()) {
    KLLM_KERNEL_CHECK_WITH_INFO(!quant_scales.empty(), "Expecting quant scales for nvfp4 quantization");
    KLLM_KERNEL_CHECK_WITH_INFO(quant_scales.size() == 6, "Expecting 6 quant scales for nvfp4 quantization");

    auto const fc1_act_global = quant_scales[0];
    auto const fc1_weight_block = quant_scales[1];
    auto const fc1_global = quant_scales[2];
    auto const fc2_act_global = quant_scales[3];
    auto const fc2_weight_block = quant_scales[4];
    auto const fc2_global = quant_scales[5];

    // The input for scale fc1_weight_block / fc2_weight_block is packed into INT32
    constexpr int FP8_PER_INT32 = 4;
    // Check types
    KLLM_KERNEL_CHECK(fc1_act_global.dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK(fc1_weight_block.dtype == ScalarType::Int);
    KLLM_KERNEL_CHECK(fc1_global.dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK(fc2_act_global.dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK(fc2_weight_block.dtype == ScalarType::Int);
    KLLM_KERNEL_CHECK(fc2_global.dtype == ScalarType::Float);
    // Check ranks
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_act_global.shape.size() == 0 || fc1_act_global.shape.size() == 1,
                                "fc1 act global must be a scalar or 1-D tensor");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_weight_block.shape.size() == 3, "fc1 weight block must be #D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_global.shape.size() == 1, "fc1 global must be 1D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_act_global.shape.size() == 0 || fc2_act_global.shape.size() == 1,
                                "fc2 act global must be a scalar or 1-D tensor");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_weight_block.shape.size() == 3, "fc2 weight block must be 3D");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_global.shape.size() == 1, "fc2 global must be 1D");
    // Check shapes
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_act_global.shape.size() == 0 || fc1_act_global.shape[0] == num_experts_on_rank,
                                "fc1 act global must be scalar or (num_experts_on_rank,)");
    KLLM_KERNEL_CHECK_WITH_INFO(
        fc1_weight_block.shape[0] == num_experts_on_rank &&
            fc1_weight_block.shape[1] ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    inter_size, internal::TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4) *
                    2 &&
            fc1_weight_block.shape[2] * FP8_PER_INT32 *
                    internal::TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    hidden_size, internal::TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4),
        "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // 4 // "
        "block_scale_vector_size)");
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_global.shape[0] == num_experts_on_rank,
                                "fc1 global size must be (num_experts_on_rank,)");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_act_global.shape.size() == 0 || fc2_act_global.shape[0] == num_experts_on_rank,
                                "fc2 act global must be scalar or (num_experts_on_rank,)");
    KLLM_KERNEL_CHECK_WITH_INFO(
        fc2_weight_block.shape[0] == num_experts_on_rank &&
            fc2_weight_block.shape[1] ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    hidden_size, internal::TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4) &&
            fc2_weight_block.shape[2] * FP8_PER_INT32 *
                    internal::TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize ==
                internal::TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    inter_size, internal::TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4),
        "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
        "block_scale_vector_size)");
    KLLM_KERNEL_CHECK_WITH_INFO(fc2_global.shape[0] == num_experts_on_rank,
                                "fc2 global size must be (num_experts_on_rank,)");

    return internal::kernels::QuantParams::FP4(
        static_cast<float const*>(fc1_act_global.data),
        static_cast<internal::TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data),
        static_cast<float const*>(fc1_global.data), static_cast<float const*>(fc2_act_global.data),
        static_cast<internal::TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data),
        static_cast<float const*>(fc2_global.data), fc1_act_global.shape.size() == 1, fc2_act_global.shape.size() == 1);
  } else if (mUseDeepSeekFP8BlockScaling) {
    auto& fc1_scales = quant_scales[0];
    auto& fc2_scales = quant_scales[1];
    return internal::kernels::QuantParams::FP8BlockScaling(static_cast<float const*>(fc1_scales.data),
                                                           static_cast<float const*>(fc2_scales.data));
  } else if (isInt4Quant()) {
    KLLM_KERNEL_CHECK_WITH_INFO(!quant_scales.empty(), "Expecting quant scales for INT4 quantization");
    KLLM_KERNEL_CHECK_WITH_INFO(quant_scales.size() == 8, "Expecting 8 quant scales for INT4 quantization");
    auto& fc1_weight_scales = quant_scales[0];
    auto& fc2_weight_scales = quant_scales[1];
    auto& fc1_act_scales = quant_scales[2];
    auto& fc2_act_scales = quant_scales[3];
    auto& fc1_weight_zeros = quant_scales[4];
    auto& fc2_weight_zeros = quant_scales[5];
    auto& fc1_alpha = quant_scales[6];
    auto& fc2_alpha = quant_scales[7];
    int group_size = 128;
    return internal::kernels::QuantParams::GroupWise(
        group_size, static_cast<void const*>(fc1_weight_scales.data), static_cast<void const*>(fc2_weight_scales.data),
        static_cast<void const*>(fc1_act_scales.data), static_cast<void const*>(fc2_act_scales.data),
        static_cast<void const*>(fc1_weight_zeros.data), static_cast<void const*>(fc2_weight_zeros.data),
        static_cast<float const*>(fc1_alpha.data), static_cast<float const*>(fc2_alpha.data));
  } else {
    return internal::kernels::QuantParams{};
  }
}

}  // namespace llm_kernels::nvidia::tensorrt_llm::dev