/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "csrc/kernels/nvidia/asymmetric_gemm/moe_gemm/moe_gemm_kernels.h"
#include "csrc/kernels/nvidia/mixture_of_experts/moe_norm_config.h"
#include "csrc/utils/nvidia/assert.h"
#include "csrc/utils/nvidia/quantization.h"
#include "cutlass/gemm/gemm.h"
// TODO(winminkong): support lora
// #include "lora/lora.h"
#include <cuda_runtime_api.h>
#include <optional>
#include <random>

namespace llm_kernels {
namespace nvidia {

static inline size_t pad_to_multiple_of_16(size_t const& input) {
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

class CubKeyValueSorter {
 public:
  CubKeyValueSorter();

  CubKeyValueSorter(int const num_experts);

  void updateNumExperts(int const num_experts);

  static size_t getWorkspaceSize(size_t const num_key_value_pairs, int const num_experts);

  void run(void* workspace, size_t const workspace_size, int const* keys_in, int* keys_out, int const* values_in,
           int* values_out, size_t const num_key_value_pairs, cudaStream_t stream);

 private:
  int num_experts_;
  int num_bits_;
};

/**
 * \brief Describes what parallelism mode the MoE is using
 *
 * Tensor Parallelism refers to the mode where the weight matrices for each expert are sliced up between nodes.
 * Each node will handle part of each expert, the final result is achieved by summing the result.
 * The inter_size dimension should be divided by the number of nodes prior to passing it to the MoE plugin, only the
 * required slice of the weights should be provided to the plugin FC1 is a ColumnLinear and FC2 is a RowLinear, see
 * tensorrt_llm/mlp/mlp.py for an example of how this works for a single MLP
 *
 * NOTE: The bias for fc2 is only applied on rank 0. If we added it on all nodes the allreduce() would contain multiple
 * copies of the bias. The bias on other node will be ignored, and may be set to nullptr
 *
 * Expert Parallelism refers to the mode where experts are divided between the nodes. Each node will handle only the
 * tokens that are routed to the experts it is assigned to. Only the weights for the node's experts should be provided
 * to the plugin For example, with #experts = 8, expert parallelism = 2: Node 0 would handle experts 0-3, and node 1
 * would handle experts 4-7
 *
 * Regardless of parallelism mode:
 *  * The input routing values must be the complete routing for all tokens/experts (required for softmax)
 *  * An allreduce must be run on the result to combine the results from different nodes if parallelism > 1
 */
struct MOEParallelismConfig {
  int tp_size = 1;
  int tp_rank = 0;
  int ep_size = 1;
  int ep_rank = 0;

  MOEParallelismConfig() = default;

  MOEParallelismConfig(int tp_size, int tp_rank, int ep_size, int ep_rank)
      : tp_size(tp_size), tp_rank(tp_rank), ep_size(ep_size), ep_rank(ep_rank) {
    // Do some basic sanity checks
    KLLM_KERNEL_CHECK(tp_rank < tp_size);
    KLLM_KERNEL_CHECK(tp_rank >= 0);
    KLLM_KERNEL_CHECK(tp_size >= 1);
    KLLM_KERNEL_CHECK(ep_rank < ep_size);
    KLLM_KERNEL_CHECK(ep_rank >= 0);
    KLLM_KERNEL_CHECK(ep_size >= 1);
  }

  bool operator==(MOEParallelismConfig const& other) const {
    return tp_size == other.tp_size && tp_rank == other.tp_rank && ep_size == other.ep_size && ep_rank == other.ep_rank;
  }

  friend std::ostream& operator<<(std::ostream& os, MOEParallelismConfig const& config) {
    os << "tp_size: " << config.tp_size << ", tp_rank: " << config.tp_rank << ", ep_size: " << config.ep_size
       << ", ep_rank: " << config.ep_rank;
    return os;
  }
};

struct QuantParams {
  // Int weight only quantization params
  void const* fc1_weight_scales = nullptr;
  void const* fc2_weight_scales = nullptr;

  // FP8 quantization params
  float const* dequant_fc1 = nullptr;
  float const* quant_fc2 = nullptr;
  float const* dequant_fc2 = nullptr;
  float const* quant_final = nullptr;

  static QuantParams FP8(float const* dequant_fc1, float const* quant_fc2, float const* dequant_fc2,
                         float const* quant_final = nullptr) {
    return QuantParams{nullptr, nullptr, dequant_fc1, quant_fc2, dequant_fc2, quant_final};
  }

  static QuantParams Int(void const* fc1_weight_scales, void const* fc2_weight_scales) {
    return QuantParams{fc1_weight_scales, fc2_weight_scales, nullptr, nullptr, nullptr, nullptr};
  }
};

struct LoraParams {
  // TODO(winminkong): support moe lora
  LoraParams() = default;
};

class CutlassMoeFCRunnerInterface {
 public:
  virtual ~CutlassMoeFCRunnerInterface() = default;
  virtual size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                                  int const num_experts, int const k, ActivationType activation_type,
                                  MOEParallelismConfig parallelism_config, bool use_lora) const = 0;
  virtual void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
                         std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config) = 0;
  virtual std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() = 0;

  virtual void runMoe(void const* input_activations, float const* gating_output, void const* fc1_expert_weights,
                      void const* fc1_expert_biases, ActivationType fc1_activation_type, void const* fc2_expert_weights,
                      void const* fc2_expert_biases, QuantParams quant_params, int64_t const num_rows,
                      int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const k,
                      char* workspace_ptr, void* final_output, bool const* finished, int64_t const active_rows,
                      void* token_topk_unpermuted_scales, int* expanded_source_row_to_expanded_dest_row,
                      int* expert_for_source_row, MOEParallelismConfig parallelism_config,
                      MOEExpertScaleNormalizationMode normalization_mode, bool use_lora, LoraParams& lora_params,
                      cudaStream_t stream,
                      RoutingFunctionType custom_routing_function = RoutingFunctionType::GREEDY_TOPK_SOFTMAX_SCORE,
                      bool apply_weight = false) = 0;

  // Aliases for profiling the gemms
  virtual void gemm1(void const* const input, void* const output, void* const intermediate_result,
                     int64_t const* const expert_first_token_offset, HopperGroupedGemmInput hopper_input_template,
                     void const* const fc1_expert_weights, void const* const fc1_expert_biases,
                     int64_t const* const num_valid_tokens_ptr, void const* const fc1_int_scales,
                     float const* const fc1_fp8_dequant, float const* const fc2_fp8_quant,
                     float const* const token_topk_unpermuted_scales, float const* const token_topk_permuted_scales,
                     int const* const expanded_source_row_to_expanded_dest_row,
                     int const* expanded_dest_row_to_expanded_source_row, int const* const expert_for_source_row,
                     int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
                     int const num_experts_per_node, ActivationType fc1_activation_type,
                     bool using_hopper_fused_finalize, float const** alpha_scale_ptr_array, bool bias_is_broadcast,
                     cudaStream_t stream, MOEParallelismConfig parallelism_config,
                     cutlass_extensions::CutlassGemmConfig config, bool apply_weight = false) = 0;

  virtual void gemm2(void const* const input, void* const gemm_output, void* const final_output,
                     int64_t const* const expert_first_token_offset, HopperGroupedGemmInput const hopper_input_template,
                     void const* const fc2_expert_weights, void const* const fc2_expert_biases,
                     void const* const fc2_int_scales, float const* const fc2_fp8_dequant,
                     float const* const token_topk_unpermuted_scales, float const* const token_topk_permuted_scales,
                     int const* const expanded_source_row_to_expanded_dest_row,
                     int const* expanded_dest_row_to_expanded_source_row, int const* const expert_for_source_row,
                     int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, int64_t const expanded_num_rows,
                     int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
                     int64_t const k, bool using_hopper_fused_finalize, float const** alpha_scale_ptr_array,
                     bool use_lora, void* fc2_lora, cudaStream_t stream, MOEParallelismConfig parallelism_config,
                     cutlass_extensions::CutlassGemmConfig config, bool apply_weight = false) = 0;

  virtual size_t getGemmWorkspaceSize(int num_experts) const = 0;
  virtual void setGemmWorkspaceSizes(std::vector<size_t>& workspace_sizes) = 0;

  bool is_profiler = false;
  bool use_deterministic_hopper_reduce_ = false;
};

// Assumes inputs activations are row major. Weights need to be preprocessed by th_op/weight_quantize.cc .
// Nested in a class to avoid multiple calls to cudaGetDeviceProperties as this call can be expensive.
// Avoid making several duplicates of this class.
template <typename T,                          /*The type used for activations*/
          typename WeightType,                 /* The type for the MoE weights */
          typename OutputType = T,             /* The type for the MoE final output */
          typename ScaleBiasType = OutputType, /* The type for scales and bias */
          typename Enable = void>
class CutlassMoeFCRunner : public CutlassMoeFCRunnerInterface {
  using Self = CutlassMoeFCRunner<T, WeightType, OutputType>;
#if defined(ENABLE_FP8)
  static constexpr bool use_fp8 = std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, __nv_fp8_e5m2>::value;
#else
  static constexpr bool use_fp8 = false;
#endif

  // This should leave the variable unchanged in any currently supported configuration
  using UnfusedGemmOutputType = typename HopperGroupedGemmInput::OutputTypeAdaptor_t<OutputType>;

  static_assert(!std::is_same_v<OutputType, __nv_fp8_e4m3>, "Current logic requires output type to be non-FP8");
  // We introduce this as a separate parameter, so that if we ever remove the above condition we can decouple
  // ScaleBiasType and OutputType easily. For now these are required to be equivalent
  static_assert(std::is_same_v<OutputType, ScaleBiasType>, "Scale and bias types must match OutputType");

 public:
  CutlassMoeFCRunner() = default;

  ~CutlassMoeFCRunner() override = default;

  static_assert(std::is_same_v<T, WeightType> || !std::is_same_v<T, float>,
                "Does not support float with quantized weights");

  size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size, int64_t const fc1_output_size,
                          int const num_experts, int const k, ActivationType activation_type,
                          MOEParallelismConfig parallelism_config, bool use_lora) const override;

  void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
                 std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config) override {
    gemm1_config_ = std::move(gemm1_config);
    gemm2_config_ = std::move(gemm2_config);
  }

  std::vector<cutlass_extensions::CutlassGemmConfig> getTactics() override { return moe_gemm_runner_.getConfigs(); }

  static std::vector<cutlass_extensions::CutlassGemmConfig> getTactics(int sm) {
    using RunnerType = decltype(moe_gemm_runner_);
    return RunnerType::getConfigs(sm);
  }

  std::vector<cutlass_extensions::CutlassGemmConfig> getFilteredTactics(int sm, bool is_fp8);

  void runMoe(void const* input_activations, float const* gating_output, void const* fc1_expert_weights,
              void const* fc1_expert_biases, ActivationType fc1_activation_type, void const* fc2_expert_weights,
              void const* fc2_expert_biases, QuantParams quant_params, int64_t const num_rows,
              int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const k,
              char* workspace_ptr, void* final_output, bool const* finished, int64_t const active_rows,
              void* token_topk_unpermuted_scales, int* expanded_source_row_to_expanded_dest_row,
              int* expert_for_source_row, MOEParallelismConfig parallelism_config,
              MOEExpertScaleNormalizationMode normalization_mode, bool use_lora, LoraParams& lora_params,
              cudaStream_t stream,
              RoutingFunctionType custom_routing_function = RoutingFunctionType::GREEDY_TOPK_SOFTMAX_SCORE,
              bool apply_weight = false) override;

  // We make these GEMM1 & GEMM2 static because they need to be stateless for the profiler to work
  static void gemm1(MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner, T const* const input,
                    T* const output, void* const intermediate_result, int64_t const* const expert_first_token_offset,
                    HopperGroupedGemmInput const hopper_input_template, WeightType const* const fc1_expert_weights,
                    ScaleBiasType const* const fc1_expert_biases, int64_t const* const num_valid_tokens_ptr,
                    ScaleBiasType const* const fc1_int_scales, float const* const fc1_fp8_dequant,
                    float const* const fc2_fp8_quant, float const* const token_topk_unpermuted_scales,
                    float const* const token_topk_permuted_scales,
                    int const* const expanded_source_row_to_expanded_dest_row,
                    int const* expanded_dest_row_to_expanded_source_row, int const* const expert_for_source_row,
                    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
                    int const num_experts_per_node, ActivationType fc1_activation_type,
                    bool using_hopper_fused_finalize, float const** alpha_scale_ptr_array, bool bias_is_broadcast,
                    cudaStream_t stream, MOEParallelismConfig parallelism_config,
                    cutlass_extensions::CutlassGemmConfig config, bool apply_weight = false);

  static void gemm2(MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner, T const* const input,
                    void* const gemm_output, OutputType* const final_output,
                    int64_t const* const expert_first_token_offset, HopperGroupedGemmInput const hopper_input_template,
                    WeightType const* const fc2_expert_weights, ScaleBiasType const* const fc2_expert_biases,
                    ScaleBiasType const* const fc2_int_scales, float const* const fc2_fp8_dequant,
                    float const* const token_topk_unpermuted_scales, float const* const token_topk_permuted_scales,
                    int const* const expanded_source_row_to_expanded_dest_row,
                    int const* expanded_dest_row_to_expanded_source_row, int const* const expert_for_source_row,
                    int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, int64_t const expanded_num_rows,
                    int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
                    int64_t const k, bool using_hopper_fused_finalize, float const** alpha_scale_ptr_array,
                    bool use_lora, void* fc2_lora, cudaStream_t stream, MOEParallelismConfig parallelism_config,
                    cutlass_extensions::CutlassGemmConfig config, bool apply_weight = false);

  // Overrides to allow us to forward on to the internal functions with the pointers using the correct type
  void gemm1(void const* const input, void* const output, void* const intermediate_result,
             int64_t const* const expert_first_token_offset, HopperGroupedGemmInput hopper_input_template,
             void const* const fc1_expert_weights, void const* const fc1_expert_biases,
             int64_t const* const num_valid_tokens_ptr, void const* const fc1_int_scales,
             float const* const fc1_fp8_dequant, float const* const fc2_fp8_quant,
             float const* const token_topk_unpermuted_scales, float const* const token_topk_permuted_scales,
             int const* const expanded_source_row_to_expanded_dest_row,
             int const* expanded_dest_row_to_expanded_source_row, int const* const expert_for_source_row,
             int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
             int const num_experts_per_node, ActivationType fc1_activation_type, bool using_hopper_fused_finalize,
             float const** alpha_scale_ptr_array, bool bias_is_broadcast, cudaStream_t stream,
             MOEParallelismConfig parallelism_config, cutlass_extensions::CutlassGemmConfig config,
             bool apply_weight = false) override {
    return Self::gemm1(
        moe_gemm_runner_, static_cast<T const*>(input), static_cast<T*>(output), intermediate_result,
        expert_first_token_offset, hopper_input_template, static_cast<WeightType const*>(fc1_expert_weights),
        static_cast<ScaleBiasType const*>(fc1_expert_biases), num_valid_tokens_ptr,
        static_cast<ScaleBiasType const*>(fc1_int_scales), fc1_fp8_dequant, fc2_fp8_quant, token_topk_unpermuted_scales,
        token_topk_permuted_scales, expanded_source_row_to_expanded_dest_row, expanded_dest_row_to_expanded_source_row,
        expert_for_source_row, expanded_num_rows, hidden_size, inter_size, num_experts_per_node, fc1_activation_type,
        using_hopper_fused_finalize, alpha_scale_ptr_array, bias_is_broadcast, stream, parallelism_config, config,
        apply_weight);
  }

  void gemm2(void const* const input, void* const gemm_output, void* const final_output,
             int64_t const* const expert_first_token_offset, HopperGroupedGemmInput const hopper_input_template,
             void const* const fc2_expert_weights, void const* const fc2_expert_biases,
             void const* const fc2_int_scales, float const* const fc2_fp8_dequant,
             float const* const token_topk_unpermuted_scales, float const* const token_topk_permuted_scales,
             int const* const expanded_source_row_to_expanded_dest_row,
             int const* expanded_dest_row_to_expanded_source_row, int const* const expert_for_source_row,
             int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, int64_t const expanded_num_rows,
             int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node, int64_t const k,
             bool using_hopper_fused_finalize, float const** alpha_scale_ptr_array, bool use_lora, void* fc2_lora,
             cudaStream_t stream, MOEParallelismConfig parallelism_config, cutlass_extensions::CutlassGemmConfig config,
             bool apply_weight = false) override {
    return Self::gemm2(moe_gemm_runner_, static_cast<T const*>(input), gemm_output,
                       static_cast<OutputType*>(final_output), expert_first_token_offset, hopper_input_template,
                       static_cast<WeightType const*>(fc2_expert_weights),
                       static_cast<ScaleBiasType const*>(fc2_expert_biases),
                       static_cast<ScaleBiasType const*>(fc2_int_scales), fc2_fp8_dequant, token_topk_unpermuted_scales,
                       token_topk_permuted_scales, expanded_source_row_to_expanded_dest_row,
                       expanded_dest_row_to_expanded_source_row, expert_for_source_row, num_valid_tokens_ptr, num_rows,
                       expanded_num_rows, hidden_size, inter_size, num_experts_per_node, k, using_hopper_fused_finalize,
                       alpha_scale_ptr_array, use_lora, fc2_lora, stream, parallelism_config, config, apply_weight);
  }

  virtual size_t getGemmWorkspaceSize(int num_experts) const override {
    return moe_gemm_runner_.getMaxWorkspaceSize(num_experts);
  }


  virtual void setGemmWorkspaceSizes(std::vector<size_t>& workspace_sizes) override {
    workspace_sizes_ = workspace_sizes;
  }

  std::vector<size_t> getWorkspaceBufferSizes(int64_t const num_rows, int64_t const hidden_size,
                                              int64_t const inter_size, int const num_experts,
                                              int const num_experts_per_node, int const k,
                                              ActivationType activation_type, bool use_lora) const;

 private:
  static HopperGroupedGemmInput computeStridesHopper(int64_t const* expert_first_token_offset,
                                                     HopperGroupedGemmInput layout_info, int64_t gemm_n, int64_t gemm_k,
                                                     int const num_experts, T const* in, WeightType const* weights,
                                                     float const* fp8_dequant, T const* bias,
                                                     UnfusedGemmOutputType* output, cudaStream_t stream);
  void configureWsPtrs(char* ws_ptr, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
                       int const num_experts, int const num_experts_per_node, int const k,
                       ActivationType activation_type, bool use_lora);

 private:
  bool mayHaveDifferentGEMMOutputType() const {
    // We just check if its supported because we need to know when calculating workspace size
    return ((moe_gemm_runner_.supportsHopperSpecialisation() && !std::is_same_v<T, UnfusedGemmOutputType>) || use_fp8);
  }

  bool mayHaveFinalizeFused() const {
    return moe_gemm_runner_.supportsHopperSpecialisation() && !use_deterministic_hopper_reduce_;
  }

  bool setupLoraWorkspace(int64_t expanded_num_rows, int64_t num_rows, bool is_gated_activation,
                          int num_experts_per_node, bool needs_num_valid, LoraParams& lora_params, cudaStream_t stream);

  T const* loraFC1(int64_t expanded_num_rows, int64_t inter_size, int64_t hidden_size, int num_experts_per_node,
                   int start_expert, int64_t const* num_valid_tokens_ptr, bool is_gated_activation,
                   ScaleBiasType const* fc1_expert_biases, LoraParams& lora_params, cudaStream_t stream);

  void loraFC2(int64_t inter_size, int64_t hidden_size, int num_experts_per_node, int start_expert,
               int64_t const* num_valid_tokens_ptr, LoraParams& lora_params, cudaStream_t stream);

  CubKeyValueSorter sorter_;
  MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType> moe_gemm_runner_;

  std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config_;
  std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config_;

  // Pointers
  int* source_rows_{};
  int* permuted_rows_{};
  int* permuted_experts_{};
  char* sorter_ws_{};
  T* permuted_data_{};
  float* softmax_out_{};
  float* permuted_scales_{};

  int64_t* expert_first_token_offset_{};

  void* glu_inter_result_{};
  void* fc2_result_{};
  T* fc1_result_{};
  float const** alpha_scale_ptr_array_ = nullptr;
  T* lora_fc1_result_{};
  T* lora_add_bias_{};
  T* lora_fc2_result_{};

  HopperGroupedGemmInput hopper_grouped_gemm_input_;

  struct HostLoraWorkspace {
    std::vector<int> host_permuted_rows;
    std::vector<void const*> host_permuted_fc1_weight_ptrs;
    std::vector<void const*> host_permuted_fc2_weight_ptrs;
    std::vector<void const*> host_permuted_gated_weight_ptrs;
    std::vector<int32_t> host_permuted_fc1_lora_ranks;
    std::vector<int32_t> host_permuted_fc2_lora_ranks;
    std::vector<int32_t> host_permuted_gated_lora_ranks;
    std::vector<int64_t> host_expert_first_token_offset;
  };

  HostLoraWorkspace host_lora_workspace_;
  RoutingFunctionType custom_routing_function_;
  std::vector<size_t> workspace_sizes_;
};

void topkGatingSoftmaxKernelLauncher(float const* input, bool const* finished, float* output,
                                     float* softmax_temp_output, int* indices, int* source_row, int64_t const num_rows,
                                     int const num_experts, int const k, int const start_expert, int const end_expert,
                                     MOEExpertScaleNormalizationMode renorm_mode, cudaStream_t stream);

template <typename T, typename OutputType>
void doGatedActivation(T* output, OutputType const* gemm_result, int64_t const* num_valid_tokens_ptr,
                       int64_t inter_size, int64_t num_tokens, ActivationType activation_type, cudaStream_t stream,
                       bool is_up_first = true);

}  // namespace nvidia
}  // namespace llm_kernels
