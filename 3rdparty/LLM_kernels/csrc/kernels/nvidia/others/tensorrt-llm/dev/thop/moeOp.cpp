/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 *
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
 *
 * Adapted from [TensorRT-LLM Project]
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.1.0rc4/cpp/tensorrt_llm/thop/moeOp.cpp
 */

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/moeOp.h"

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
    case ScalarType::Float16:
      if constexpr (NeedQuant) {
        return std::make_unique<internal::kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, half, half>>();
      } else {
        return std::make_unique<internal::kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, half, TypeAct>>();
      }
#ifdef ENABLE_BF16
    case ScalarType::BFloat16:
      if constexpr (NeedQuant) {
        return std::make_unique<
            internal::kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, __nv_bfloat16, __nv_bfloat16>>();
      } else {
        return std::make_unique<internal::kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, __nv_bfloat16, TypeAct>>();
      }
#endif
    default:
      KLLM_KERNEL_THROW(fmt::format("Invalid output type {} specified for {}", output_type, mActivationDtype));
  }
}

template <typename TypeAct>
std::unique_ptr<internal::kernels::CutlassMoeFCRunnerInterface> FusedMoeRunner::create_weight_quant_runner() {
  if (isInt8Quant()) {
    return std::make_unique<internal::kernels::CutlassMoeFCRunner<TypeAct, uint8_t>>();
  } else if (isInt4Quant()) {
#ifdef ENABLE_FP8
    if (mUseW4GroupScaling) {
      return std::make_unique<
          internal::kernels::CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, TypeAct, TypeAct>>();
    }
#endif
    return std::make_unique<internal::kernels::CutlassMoeFCRunner<TypeAct, cutlass::uint4b_t>>();
  } else {
    KLLM_KERNEL_THROW("Unsupported weight quantization type");
  }
}

FusedMoeRunner::FusedMoeRunner(ScalarType activation_dtype, ScalarType weight_dtype, ScalarType output_dtype,
                               bool use_deepseek_fp8_block_scale, bool use_w4_group_scaling,
                               bool use_int8_woq_per_channel, bool use_mxfp8_act_scaling, bool use_fused_finalize) {
  mActivationDtype = activation_dtype;
  mWeightDtype = weight_dtype;
  mOutputDtype = output_dtype;
  mUseDeepSeekFP8BlockScaling = use_deepseek_fp8_block_scale;
  mUseW4GroupScaling = use_w4_group_scaling;
  mUseINT8WoqPerChannel = use_int8_woq_per_channel;
  mUseMxfp8ActScaling = use_mxfp8_act_scaling;
  mUseFusedFinalize = use_fused_finalize;
  mInnerDimMultiplier = 1;

  // keep consistent with cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp
  if (mActivationDtype == ScalarType::Float16 && mWeightDtype == ScalarType::Float16) {
    mKernelRunner = std::make_shared<internal::kernels::CutlassMoeFCRunner<half, half>>();
  }
#ifdef ENABLE_BF16
  else if (mActivationDtype == ScalarType::BFloat16 && mWeightDtype == ScalarType::BFloat16) {
    mKernelRunner = std::make_shared<internal::kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
  }
#  ifdef ENABLE_FP8
  else if (mActivationDtype == ScalarType::BFloat16 && mWeightDtype == ScalarType::Float8_e4m3fn) {
    mKernelRunner = std::make_unique<internal::kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3>>();
  }
#  endif
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
    mInnerDimMultiplier = 16;  // 16 FP4 -> 1 LONG
    switch (mActivationDtype) {
      case ScalarType::Float16:
#  ifdef ENABLE_BF16
      case ScalarType::BFloat16:
#  endif
        mKernelRunner = switch_output_type<__nv_fp4_e2m1, __nv_fp4_e2m1, true>(mOutputDtype);
        break;
      default:
        mKernelRunner = switch_output_type<__nv_fp4_e2m1, __nv_fp4_e2m1, false>(mOutputDtype);
    }
  }
  if (isWFP4A16Quant()) {
    mInnerDimMultiplier = 2;
    if (mActivationDtype == ScalarType::Float16) {
      mKernelRunner = std::make_shared<internal::kernels::CutlassMoeFCRunner<half, __nv_fp4_e2m1>>();
    }
#  ifdef ENABLE_BF16
    else if (mActivationDtype == ScalarType::BFloat16) {
      mKernelRunner = std::make_shared<internal::kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_fp4_e2m1>>();
    }
#  endif
  }
#endif
  if (isIntWeightOnlyQuant()) {
    if (isInt4Quant()) {
      mInnerDimMultiplier = 2;  // 2 INT4 -> 1 INT8
    }
    switch (mActivationDtype) {
      case ScalarType::Float16:
        mKernelRunner = create_weight_quant_runner<half>();
        break;
      case ScalarType::BFloat16:
        mKernelRunner = create_weight_quant_runner<__nv_bfloat16>();
        break;
      default:
        KLLM_KERNEL_THROW("Unsupported activation type for int-type weight");
    }
  }
  if (!mKernelRunner) {
    KLLM_KERNEL_THROW(
        fmt::format("Could not construct fused moe op with the requested input combination Activation: {}, Weight: "
                    "{}, Output: {}",
                    mActivationDtype, mWeightDtype, mOutputDtype));
  }

  mKernelRunner->use_fused_finalize_ = mUseFusedFinalize;

  mProfiler = std::make_shared<internal::kernels::GemmProfilerBackend>();
  mGemm1Profiles = mKernelRunner->getTactics(internal::MoeGemmId::GEMM_1);
  mGemm2Profiles = mKernelRunner->getTactics(internal::MoeGemmId::GEMM_2);
}

size_t FusedMoeRunner::getRuntimeWorkspaceInfo(const Tensor& input, const Tensor& token_selected_experts,
                                               const Tensor& fc2_expert_weights,
                                               const std::optional<Tensor>& swiglu_alpha,
                                               const std::optional<Tensor>& swiglu_beta,
                                               const std::optional<Tensor>& swiglu_limit, int64_t const tp_size,
                                               int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank,
                                               bool min_latency_mode, const std::vector<int64_t>& profile_ids,
                                               const std::optional<int64_t>& unpadded_hidden_size) {
  int experts_per_token = token_selected_experts.shape[1];
  int64_t num_rows = input.shape[0];
  int64_t hidden_size = fc2_expert_weights.shape[1];
  int64_t unpadded_hidden_size_val = unpadded_hidden_size.has_value() ? unpadded_hidden_size.value() : hidden_size;
  int64_t inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
  if (mUseINT8WoqPerChannel) {
    // Note: The weight shape for INT8 weight only quantization is different, e.g., fc2_expert_weights:
    // [num_experts, inter_size, hidden_size]
    hidden_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
    inter_size = fc2_expert_weights.shape[1];
  }

  int const num_experts_on_rank = fc2_expert_weights.shape[0];
  return getRuntimeWorkspaceInfo(swiglu_alpha, swiglu_beta, swiglu_limit, experts_per_token, num_rows, hidden_size,
                                 inter_size, num_experts_on_rank, tp_size, tp_rank, ep_size, ep_rank, min_latency_mode,
                                 profile_ids, unpadded_hidden_size);
}

size_t FusedMoeRunner::getRuntimeWorkspaceInfo(
    const std::optional<Tensor>& swiglu_alpha, const std::optional<Tensor>& swiglu_beta,
    const std::optional<Tensor>& swiglu_limit, const size_t experts_per_token, const size_t num_rows,
    const size_t hidden_size, const size_t inter_size, const size_t num_experts_on_rank, int64_t const tp_size,
    int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank, bool min_latency_mode,
    const std::vector<int64_t>& profile_ids, const std::optional<int64_t>& unpadded_hidden_size) {
  auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
  auto parallelism_config = internal::kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
  internal::ActivationType base_activation_type = internal::ActivationType::Swiglu;
  if (swiglu_alpha.has_value()) {
    base_activation_type = internal::ActivationType::SwigluBias;
  }
  if (swiglu_beta.has_value()) {
    base_activation_type = internal::ActivationType::SwigluBias;
  }
  if (swiglu_limit.has_value()) {
    base_activation_type = internal::ActivationType::SwigluBias;
  }
  auto activation_params = internal::ActivationParams(
      base_activation_type,
      reinterpret_cast<float const*>(swiglu_alpha.has_value() ? swiglu_alpha.value().data : nullptr),
      reinterpret_cast<float const*>(swiglu_beta.has_value() ? swiglu_beta.value().data : nullptr),
      reinterpret_cast<float const*>(swiglu_limit.has_value() ? swiglu_limit.value().data : nullptr));

  setRunnerProfiles(profile_ids);

  return getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
                          base_activation_type, parallelism_config, min_latency_mode);
}

void FusedMoeRunner::runMoe(Tensor& output, const Tensor& input, const Tensor& token_selected_experts,
                            const std::optional<Tensor>& token_final_scales, const Tensor& fc1_expert_weights,
                            const std::optional<Tensor>& fc1_expert_biases, const Tensor& fc2_expert_weights,
                            const std::optional<Tensor>& fc2_expert_biases, const std::vector<Tensor>& quant_scales,
                            const std::optional<Tensor>& input_sf, const bool swizzled_input_sf,
                            const std::optional<Tensor>& swiglu_alpha, const std::optional<Tensor>& swiglu_beta,
                            const std::optional<Tensor>& swiglu_limit, int64_t const tp_size, int64_t const tp_rank,
                            int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
                            int64_t const cluster_rank, bool const enable_alltoall, bool min_latency_mode,
                            const std::vector<int64_t>& profile_ids, std::optional<int64_t> const& unpadded_hidden_size,
                            cudaStream_t stream) {
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

  if (mUseINT8WoqPerChannel) {
    // Note: The weight shape for INT8 weight only quantization is different, e.g., fc2_expert_weights:
    // [num_experts, inter_size, hidden_size]
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_weights.shape[2] == fc2_expert_weights.shape[1] * mInnerDimMultiplier * 2,
                                "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.");
  } else {
    KLLM_KERNEL_CHECK_WITH_INFO(fc1_expert_weights.shape[1] == fc2_expert_weights.shape[2] * mInnerDimMultiplier * 2,
                                "fc1_expert_weights inter size must be fc2_expert_weights inter size.");
  }

  int experts_per_token = token_selected_experts.shape[1];
  int64_t num_rows = input.shape[0];
  int64_t hidden_size = fc2_expert_weights.shape[1];
  int64_t unpadded_hidden_size_val = unpadded_hidden_size.has_value() ? unpadded_hidden_size.value() : hidden_size;
  int64_t inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
  if (mUseINT8WoqPerChannel) {
    // Note: The weight shape for INT8 weight only quantization is different, e.g., fc2_expert_weights:
    // [num_experts, inter_size, hidden_size]
    hidden_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
    inter_size = fc2_expert_weights.shape[1];
  }

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
  internal::ActivationType base_activation_type = internal::ActivationType::Swiglu;
  if (swiglu_alpha.has_value()) {
    KLLM_KERNEL_CHECK(swiglu_alpha.value().dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK_WITH_INFO(swiglu_alpha.value().shape[0] == num_experts_on_rank,
                                "swiglu_alpha must have num_experts_on_rank elements.");
    base_activation_type = internal::ActivationType::SwigluBias;
  }
  if (swiglu_beta.has_value()) {
    KLLM_KERNEL_CHECK(swiglu_beta.value().dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK_WITH_INFO(swiglu_beta.value().shape[0] == num_experts_on_rank,
                                "swiglu_beta must have num_experts_on_rank elements.");
    base_activation_type = internal::ActivationType::SwigluBias;
  }
  if (swiglu_limit.has_value()) {
    KLLM_KERNEL_CHECK(swiglu_limit.value().dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK_WITH_INFO(swiglu_limit.value().shape[0] == num_experts_on_rank,
                                "swiglu_limit must have num_experts_on_rank elements.");
    base_activation_type = internal::ActivationType::SwigluBias;
  }
  auto activation_params = internal::ActivationParams(
      base_activation_type,
      reinterpret_cast<float const*>(swiglu_alpha.has_value() ? swiglu_alpha.value().data : nullptr),
      reinterpret_cast<float const*>(swiglu_beta.has_value() ? swiglu_beta.value().data : nullptr),
      reinterpret_cast<float const*>(swiglu_limit.has_value() ? swiglu_limit.value().data : nullptr));

  setRunnerProfiles(profile_ids);

  output.shape = {static_cast<size_t>(num_rows), static_cast<size_t>(unpadded_hidden_size_val)};
  output.dtype = mOutputDtype;

  auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);
  internal::kernels::MoeMinLatencyParams min_latency_params{};

  // TODO: support lora in the future
  ::llm_kernels::nvidia::tensorrt_llm::dev::kernels::LoraParams lora_params{};
  mKernelRunner->runMoe(
      input.data, input_sf.has_value() ? input_sf.value().data : nullptr, swizzled_input_sf,
      reinterpret_cast<int const*>(token_selected_experts.data),
      token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().data) : nullptr,
      fc1_expert_weights.data, fc1_expert_biases.has_value() ? fc1_expert_biases.value().data : nullptr,
      activation_params, fc2_expert_weights.data,
      fc2_expert_biases.has_value() ? fc2_expert_biases.value().data : nullptr, quant_params, num_rows, hidden_size,
      unpadded_hidden_size_val, inter_size, num_experts_total, static_cast<int>(experts_per_token),
      static_cast<char*>(runtime_workspace.workspace), output.data,
      static_cast<int*>(runtime_workspace.src_to_dest_map), parallelism_config, enable_alltoall, false, lora_params,
      mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
}

void FusedMoeRunner::runMoeMinLantency(
    Tensor& output, Tensor& num_active_experts_per_node, Tensor& experts_to_token_score,
    Tensor& active_expert_global_ids, const Tensor& input, const Tensor& token_selected_experts,
    const std::optional<Tensor>& token_final_scales, const Tensor& fc1_expert_weights,
    const std::optional<Tensor>& fc1_expert_biases, const Tensor& fc2_expert_weights,
    const std::optional<Tensor>& fc2_expert_biases, const std::vector<Tensor>& quant_scales,
    const std::optional<Tensor>& input_sf, const bool swizzled_input_sf, const std::optional<Tensor>& swiglu_alpha,
    const std::optional<Tensor>& swiglu_beta, const std::optional<Tensor>& swiglu_limit, int64_t const tp_size,
    int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
    int64_t const cluster_rank, bool const enable_alltoall, bool min_latency_mode,
    const std::vector<int64_t>& profile_ids, std::optional<int64_t> const& unpadded_hidden_size, cudaStream_t stream) {
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
  int64_t unpadded_hidden_size_val = unpadded_hidden_size.has_value() ? unpadded_hidden_size.value() : hidden_size;
  int64_t inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
  int const num_experts_on_rank = fc2_expert_weights.shape[0];
  auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
  auto parallelism_config =
      internal::kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank, cluster_size, cluster_rank);
  internal::ActivationType base_activation_type = internal::ActivationType::Swiglu;
  if (swiglu_alpha.has_value()) {
    KLLM_KERNEL_CHECK(swiglu_alpha.value().dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK_WITH_INFO(swiglu_alpha.value().shape[0] == num_experts_on_rank,
                                "swiglu_alpha must have num_experts_on_rank elements.");
    base_activation_type = internal::ActivationType::SwigluBias;
  }
  if (swiglu_beta.has_value()) {
    KLLM_KERNEL_CHECK(swiglu_beta.value().dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK_WITH_INFO(swiglu_beta.value().shape[0] == num_experts_on_rank,
                                "swiglu_beta must have num_experts_on_rank elements.");
    base_activation_type = internal::ActivationType::SwigluBias;
  }
  if (swiglu_limit.has_value()) {
    KLLM_KERNEL_CHECK(swiglu_limit.value().dtype == ScalarType::Float);
    KLLM_KERNEL_CHECK_WITH_INFO(swiglu_limit.value().shape[0] == num_experts_on_rank,
                                "swiglu_limit must have num_experts_on_rank elements.");
    base_activation_type = internal::ActivationType::SwigluBias;
  }
  auto activation_params = internal::ActivationParams(
      base_activation_type,
      reinterpret_cast<float const*>(swiglu_alpha.has_value() ? swiglu_alpha.value().data : nullptr),
      reinterpret_cast<float const*>(swiglu_beta.has_value() ? swiglu_beta.value().data : nullptr),
      reinterpret_cast<float const*>(swiglu_limit.has_value() ? swiglu_limit.value().data : nullptr));

  setRunnerProfiles(profile_ids);

  output.shape = {static_cast<size_t>(num_rows * num_experts_on_rank), static_cast<size_t>(unpadded_hidden_size_val)};
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
      input.data, input_sf.has_value() ? input_sf.value().data : nullptr, swizzled_input_sf,
      reinterpret_cast<int const*>(token_selected_experts.data),
      token_final_scales.has_value() ? reinterpret_cast<float const*>(token_final_scales.value().data) : nullptr,
      fc1_expert_weights.data, fc1_expert_biases.has_value() ? fc1_expert_biases.value().data : nullptr,
      activation_params, fc2_expert_weights.data,
      fc2_expert_biases.has_value() ? fc2_expert_biases.value().data : nullptr, quant_params, num_rows, hidden_size,
      unpadded_hidden_size_val, inter_size, num_experts_total, static_cast<int>(experts_per_token),
      static_cast<char*>(runtime_workspace.workspace), output.data,
      static_cast<int*>(runtime_workspace.src_to_dest_map), parallelism_config, enable_alltoall, false, lora_params,
      mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, stream);
}

int64_t FusedMoeRunner::getTacticNum(int64_t const gemm_idx) {
  KLLM_KERNEL_CHECK_WITH_INFO(gemm_idx == 1 || gemm_idx == 2, "gemm_idx must be 1 or 2");
  return (gemm_idx == 1) ? mGemm1Profiles.size() : mGemm2Profiles.size();
}

size_t FusedMoeRunner::getProfileWorkspace(
    const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases, const Tensor& fc2_expert_weights,
    const std::optional<Tensor>& fc2_expert_biases, const int64_t num_rows, int64_t const top_k, int64_t const tp_size,
    int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank, int64_t const cluster_size,
    int64_t const cluster_rank, bool const enable_alltoall, bool const min_latency_mode, int64_t const gemm_idx,
    int64_t const profile_id, bool const do_preparation, int64_t const unpadded_hidden_size, cudaStream_t stream) {
  // TODO: support profiling under fp8 block scaling in the future
  if (mUseDeepSeekFP8BlockScaling) {
    return 0;
  }

  int64_t hidden_size = fc2_expert_weights.shape[1];
  int64_t inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
  if (mUseINT8WoqPerChannel) {
    // Note: The weight shape for INT8 weight only quantization is different, e.g., fc2_expert_weights:
    // [num_experts, inter_size, hidden_size]
    hidden_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
    inter_size = fc2_expert_weights.shape[1];
  }
  int64_t const group_size_ =
      isInt4Quant() ? internal::TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size : -1;
  int64_t const group_size = isWFP4A16Quant()
                                 ? internal::TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size
                                 : group_size_;
  int const num_experts = static_cast<int>(fc2_expert_weights.shape[0] * ep_size);

  auto const gemm_to_profile = (gemm_idx == 1) ? internal::profiler_backend::GemmToProfile::GEMM_1
                                               : internal::profiler_backend::GemmToProfile::GEMM_2;
  auto const& profiles = (gemm_idx == 1) ? mGemm1Profiles : mGemm2Profiles;

  // Get specific profile configs according to the profile_id.
  // Fallback tactic is set to be 0
  // TODO: use the best tactic id found offline for a better default inference perf
  auto const& profile = profile_id == -1 ? profiles.front() : profiles[profile_id];

  auto const* expert_weights_ptr = (gemm_idx == 1) ? fc1_expert_weights.data : fc2_expert_weights.data;

  // Set profiled gemm idx
  mProfiler->mGemmToProfile = gemm_to_profile;

  // mProfiler init
  auto parallelism_config = internal::kernels::MOEParallelismConfig(
      static_cast<int>(tp_size), static_cast<int>(tp_rank), static_cast<int>(ep_size), static_cast<int>(ep_rank),
      static_cast<int>(cluster_size), static_cast<int>(cluster_rank));

  bool const USE_BIAS = fc1_expert_biases.has_value() || fc2_expert_biases.has_value();
  bool const USE_LORA = false;
  auto activation_dtype = (mUseW4GroupScaling && !isWFP4A16Quant()) ? ScalarType::Float8_e4m3fn : mActivationDtype;
  activation_dtype = isNvfp4Quant() ? ScalarType::Long : activation_dtype;

  mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile, GetNvinferDataType(activation_dtype),
                  GetNvinferDataType(mWeightDtype), GetNvinferDataType(mOutputDtype), num_experts,
                  static_cast<int>(top_k), hidden_size, unpadded_hidden_size > 0 ? unpadded_hidden_size : hidden_size,
                  inter_size, group_size, internal::ActivationType::Swiglu, USE_BIAS, USE_LORA, min_latency_mode,
                  /*need_weights*/ false, parallelism_config, enable_alltoall);

  return mProfiler->getWorkspaceSize(num_rows);
}

// TODO Update this to be able to tell if we are profiling swiglu bias
void FusedMoeRunner::setProfileWorkspace(
    void* profile_workspace_ptr, const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases,
    const Tensor& fc2_expert_weights, const std::optional<Tensor>& fc2_expert_biases, const int64_t num_rows,
    int64_t const top_k, int64_t const tp_size, int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank,
    int64_t const cluster_size, int64_t const cluster_rank, bool const enable_alltoall, bool const min_latency_mode,
    int64_t const gemm_idx, int64_t const profile_id, bool const do_preparation, int64_t const unpadded_hidden_size,
    cudaStream_t stream) {
  // TODO: support profiling under fp8 block scaling in the future
  if (mUseDeepSeekFP8BlockScaling) {
    return;
  }

  int64_t hidden_size = fc2_expert_weights.shape[1];
  int64_t inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
  if (mUseINT8WoqPerChannel) {
    // Note: The weight shape for INT8 weight only quantization is different, e.g., fc2_expert_weights:
    // [num_experts, inter_size, hidden_size]
    hidden_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
    inter_size = fc2_expert_weights.shape[1];
  }
  int64_t const group_size_ =
      isInt4Quant() ? internal::TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size : -1;
  int64_t const group_size = isWFP4A16Quant()
                                 ? internal::TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size
                                 : group_size_;
  int const num_experts = static_cast<int>(fc2_expert_weights.shape[0] * ep_size);

  auto const gemm_to_profile = (gemm_idx == 1) ? internal::profiler_backend::GemmToProfile::GEMM_1
                                               : internal::profiler_backend::GemmToProfile::GEMM_2;
  auto const& profiles = (gemm_idx == 1) ? mGemm1Profiles : mGemm2Profiles;

  // Get specific profile configs according to the profile_id.
  // Fallback tactic is set to be 0
  // TODO: use the best tactic id found offline for a better default inference perf
  auto const& profile = profile_id == -1 ? profiles.front() : profiles[profile_id];

  auto const* expert_weights_ptr = (gemm_idx == 1) ? fc1_expert_weights.data : fc2_expert_weights.data;

  // Preparation phase, only enabled during autotuning warmup phase.
  // Set profiled gemm idx
  mProfiler->mGemmToProfile = gemm_to_profile;

  // mProfiler init
  auto parallelism_config = internal::kernels::MOEParallelismConfig(
      static_cast<int>(tp_size), static_cast<int>(tp_rank), static_cast<int>(ep_size), static_cast<int>(ep_rank),
      static_cast<int>(cluster_size), static_cast<int>(cluster_rank));

  bool const USE_BIAS = fc1_expert_biases.has_value() || fc2_expert_biases.has_value();
  bool const USE_LORA = false;
  auto activation_dtype = (mUseW4GroupScaling && !isWFP4A16Quant()) ? ScalarType::Float8_e4m3fn : mActivationDtype;
  activation_dtype = isNvfp4Quant() ? ScalarType::Long : activation_dtype;

  mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile, GetNvinferDataType(activation_dtype),
                  GetNvinferDataType(mWeightDtype), GetNvinferDataType(mOutputDtype), num_experts,
                  static_cast<int>(top_k), hidden_size, unpadded_hidden_size > 0 ? unpadded_hidden_size : hidden_size,
                  inter_size, group_size, internal::ActivationType::Swiglu, USE_BIAS, USE_LORA, min_latency_mode,
                  /*need_weights*/ false, parallelism_config, enable_alltoall);

  mProfileWorkspace = static_cast<char*>(profile_workspace_ptr);

  mProfiler->prepare(num_rows, mProfileWorkspace, expert_weights_ptr, stream);

  // Profile specific tactic. Assuming at least one preparation phase has been executed already.
  mProfiler->runProfiler(num_rows, profile, mProfileWorkspace, expert_weights_ptr, stream);
}

// TODO Update this to be able to tell if we are profiling swiglu bias
void FusedMoeRunner::runGemmProfile(const Tensor& fc1_expert_weights, const std::optional<Tensor>& fc1_expert_biases,
                                    const Tensor& fc2_expert_weights, const std::optional<Tensor>& fc2_expert_biases,
                                    const int64_t num_rows, int64_t const top_k, int64_t const tp_size,
                                    int64_t const tp_rank, int64_t const ep_size, int64_t const ep_rank,
                                    int64_t const cluster_size, int64_t const cluster_rank, bool const enable_alltoall,
                                    bool const min_latency_mode, int64_t const gemm_idx, int64_t const profile_id,
                                    bool const do_preparation, int64_t const unpadded_hidden_size,
                                    cudaStream_t stream) {
  // TODO: support profiling under fp8 block scaling in the future
  if (mUseDeepSeekFP8BlockScaling) {
    return;
  }

  int64_t hidden_size = fc2_expert_weights.shape[1];
  int64_t inter_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
  if (mUseINT8WoqPerChannel) {
    // Note: The weight shape for INT8 weight only quantization is different, e.g., fc2_expert_weights:
    // [num_experts, inter_size, hidden_size]
    hidden_size = fc2_expert_weights.shape[2] * mInnerDimMultiplier;
    inter_size = fc2_expert_weights.shape[1];
  }
  int64_t const group_size_ =
      isInt4Quant() ? internal::TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size : -1;
  int64_t const group_size = isWFP4A16Quant()
                                 ? internal::TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size
                                 : group_size_;
  int const num_experts = static_cast<int>(fc2_expert_weights.shape[0] * ep_size);

  auto const gemm_to_profile = (gemm_idx == 1) ? internal::profiler_backend::GemmToProfile::GEMM_1
                                               : internal::profiler_backend::GemmToProfile::GEMM_2;
  auto const& profiles = (gemm_idx == 1) ? mGemm1Profiles : mGemm2Profiles;

  // Get specific profile configs according to the profile_id.
  // Fallback tactic is set to be 0
  // TODO: use the best tactic id found offline for a better default inference perf
  auto const& profile = profile_id == -1 ? profiles.front() : profiles[profile_id];

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

  auto best_gemm1_profile = mGemm1Profiles.front();
  auto best_gemm2_profile = mGemm2Profiles.front();
  if (!profile_ids.empty()) {
    KLLM_KERNEL_CHECK_WITH_INFO(profile_ids.size() == 2, "Expecting 2 profile ids");
    best_gemm1_profile = profile_ids[0] == -1 ? best_gemm1_profile : mGemm1Profiles.at(profile_ids[0]);
    best_gemm2_profile = profile_ids[1] == -1 ? best_gemm2_profile : mGemm2Profiles.at(profile_ids[1]);
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
      /* use_lora */ false, mUseDeepSeekFP8BlockScaling, min_latency_mode, mUseW4GroupScaling);
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
                    hidden_size, internal::TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX),
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
  } else if (isWFP4A16Quant()) {
    KLLM_KERNEL_CHECK_WITH_INFO(!quant_scales.empty(), "Expecting quant scales for weight only quantization");
    KLLM_KERNEL_CHECK_WITH_INFO(quant_scales.size() == 2, "Expecting 2 quant scales for W4A16 quantization");

    auto& fc1_weight_scales = quant_scales[0];
    auto& fc2_weight_scales = quant_scales[1];
    int group_size = internal::TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size;
    return internal::kernels::QuantParams::GroupWise(group_size, static_cast<void const*>(fc1_weight_scales.data),
                                                     static_cast<void const*>(fc2_weight_scales.data), nullptr, nullptr,
                                                     nullptr, nullptr, nullptr, nullptr);
  } else if (isIntWeightOnlyQuant()) {
    KLLM_KERNEL_CHECK_WITH_INFO(!quant_scales.empty(), "Expecting quant scales for weight only quantization");
    if (mUseINT8WoqPerChannel) {
      KLLM_KERNEL_CHECK_WITH_INFO(quant_scales.size() == 2,
                                  "Expecting 2 quant scales for INT8 weight only quantization");
      auto& fc1_weight_scales = quant_scales[0];
      auto& fc2_weight_scales = quant_scales[1];
      return internal::kernels::QuantParams::Int(static_cast<float const*>(fc1_weight_scales.data),
                                                 static_cast<float const*>(fc2_weight_scales.data));
    } else if (isInt4Quant() && mUseW4GroupScaling) {
      KLLM_KERNEL_CHECK_WITH_INFO(quant_scales.size() == 8, "Expecting 8 quant scales for W4A8 quantization");
      auto& fc1_weight_scales = quant_scales[0];
      auto& fc2_weight_scales = quant_scales[1];
      auto& fc1_act_scales = quant_scales[2];
      auto& fc2_act_scales = quant_scales[3];
      auto& fc1_weight_zeros = quant_scales[4];
      auto& fc2_weight_zeros = quant_scales[5];
      auto& fc1_alpha = quant_scales[6];
      auto& fc2_alpha = quant_scales[7];
      int group_size = internal::TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size;
      return internal::kernels::QuantParams::GroupWise(
          group_size, static_cast<void const*>(fc1_weight_scales.data),
          static_cast<void const*>(fc2_weight_scales.data), static_cast<void const*>(fc1_act_scales.data),
          static_cast<void const*>(fc2_act_scales.data), static_cast<void const*>(fc1_weight_zeros.data),
          static_cast<void const*>(fc2_weight_zeros.data), static_cast<float const*>(fc1_alpha.data),
          static_cast<float const*>(fc2_alpha.data));
    } else {
      KLLM_KERNEL_CHECK_WITH_INFO(false, "Unsupported weight only quantization");
    }
  } else {
    return internal::kernels::QuantParams{};
  }
}

}  // namespace llm_kernels::nvidia::tensorrt_llm::dev