/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/moe_layer.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/layers/grouped_topk_layer.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {
#ifdef ENABLE_CUDA
template <typename T>
Status MoeLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;

  static const char* env_int4_fp8_moe_compute = std::getenv("EXPERIMENTAL_INT4_FP8_MOE");
  bool use_int4_fp8_moe_compute = env_int4_fp8_moe_compute != nullptr && strcmp(env_int4_fp8_moe_compute, "1") == 0;

  int parameter_index = 0;
  moe_scale_norm_mode_ = std::any_cast<const MoeScaleNormMode>(parameters[parameter_index++]);
  max_token_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_hidden_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_inter_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  expert_topk_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  tp_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  use_vllm_moe_ = std::any_cast<bool>(parameters[parameter_index++]);
  num_expert_group_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  expert_groups_topk_ = std::any_cast<uint32_t>(parameters[parameter_index++]);
  scoring_func_ = std::any_cast<std::string>(parameters[parameter_index++]);
  topk_method_ = std::any_cast<std::string>(parameters[parameter_index++]);
  norm_topk_prob_ = std::any_cast<bool>(parameters[parameter_index++]);
  routed_scaling_factor_ = std::any_cast<float>(parameters[parameter_index++]);
  use_e_score_correction_bias_ = std::any_cast<bool>(parameters[parameter_index++]);
  DataType fp8_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  DataType int_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  int group_size = std::any_cast<int>(parameters[parameter_index++]);

  // 权重&计算类型处理
  weight_dtype_ = GetDataType<T>();
  compute_dtype_ = GetDataType<T>();
  if (fp8_weight_dtype == DataType::TYPE_BLOCK_FP8_E4M3) {
    weight_dtype_ = DataType::TYPE_BLOCK_FP8_E4M3;
    compute_dtype_ = DataType::TYPE_BLOCK_FP8_E4M3;
  } else if (int_weight_dtype == DataType::TYPE_I4_GROUP) {
    weight_dtype_ = DataType::TYPE_I4_GROUP;
    compute_dtype_ = use_int4_fp8_moe_compute ? DataType::TYPE_BLOCK_FP8_E4M3 : compute_dtype_;
  }

  block_shape_.resize(2);
  if (weight_dtype_ == DataType::TYPE_BLOCK_FP8_E4M3) {
    block_shape_ = {128, 128};
  } else if (weight_dtype_ == DataType::TYPE_I4_GROUP) {
    block_shape_ = {0, group_size};
  }

  apply_weight_ = std::any_cast<bool>(parameters[parameter_index++]);

  // 初始化 GroupedTopkLayer
  grouped_topk_layer_ = std::make_shared<GroupedTopkLayer<T>>();
  std::vector<std::any> grouped_topk_params = {
      static_cast<int>(expert_topk_),        norm_topk_prob_, static_cast<int>(num_expert_group_),
      static_cast<int>(expert_groups_topk_), scoring_func_,   routed_scaling_factor_,
      use_e_score_correction_bias_};
  grouped_topk_layer_->Init(grouped_topk_params, context, rank);

  return Status();
}
#  define VLLM_FUSED_MOE_CHUNK_SIZE ((size_t)(32 * 1024))

inline size_t AlignAddress(size_t size) { return (size + 255) & (~255); }

template <typename T>
size_t MoeLayer<T>::GetWorkSpaceSize() {
  GetMoeGemmWorkspaceSize<T, T, T>(max_token_num_, expert_num_, expert_hidden_size_, expert_inter_size_, expert_topk_,
                                   tp_size_, rank_, use_lora_, max_ws_bytes_);
  if (use_vllm_moe_) {
    size_t m = std::min(VLLM_FUSED_MOE_CHUNK_SIZE, max_token_num_);
    topk_weights_ptr_size = AlignAddress(max_token_num_ * expert_topk_ * sizeof(float));
    topk_ids_ptr_size = AlignAddress(max_token_num_ * expert_topk_ * sizeof(int64_t));
    max_fused_id_buffer_size = 2 * m * expert_topk_ * sizeof(int32_t);
    intermediate_cache1_size = AlignAddress(m * expert_topk_ * expert_inter_size_ * 2 * sizeof(T));
    intermediate_cache2_size = AlignAddress(m * expert_topk_ * expert_inter_size_ * sizeof(T));
    intermediate_cache3_size = AlignAddress(m * expert_topk_ * expert_hidden_size_ * sizeof(T));
    intermediate_cache1_and_cache3_size = std::max(intermediate_cache1_size, intermediate_cache3_size);  // 共享
    if (compute_dtype_ == DataType::TYPE_BLOCK_FP8_E4M3) {
      a1_q_size = AlignAddress(m * expert_hidden_size_ * sizeof(char));
      a2_q_size = AlignAddress(m * expert_topk_ * expert_inter_size_ * sizeof(char));
      a1_scale_size = AlignAddress(m * expert_hidden_size_ / 128 * sizeof(float));
      a2_scale_size = AlignAddress(m * expert_topk_ * expert_inter_size_ / 128 * sizeof(float));
      a1_and_a2_q_size = std::max(a1_q_size, a2_q_size);              // 共享
      a1_and_a2_scale_size = std::max(a1_scale_size, a2_scale_size);  // 共享
      if (weight_dtype_ == DataType::TYPE_I4_GROUP) {
        // TODO(jinxcwu) too large
        dequant_workspace_size =
            AlignAddress(expert_num_ * expert_hidden_size_ * expert_inter_size_ * 2 * sizeof(char));
      }
    }
    max_ws_bytes_ = topk_weights_ptr_size + topk_ids_ptr_size + max_fused_id_buffer_size +
                    intermediate_cache1_and_cache3_size + intermediate_cache2_size + a1_and_a2_q_size +
                    a1_and_a2_scale_size + dequant_workspace_size;
  }
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for MoeLayer", rank_, max_ws_bytes_);
  return max_ws_bytes_;
}

template <typename T>
Status MoeLayer<T>::SetWorkSpaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) {
  workspace_buffer_ = workspace_buffer;
  scale_probabilities_size_ = max_token_num_ * expert_num_ * sizeof(float);
  src_to_dest_map_size_ = expert_topk_ * max_token_num_ * sizeof(int);
  selected_expert_size_ = expert_topk_ * max_token_num_ * sizeof(int);
  lora_workspace_size_ = 0;  // NO support for lora
  moe_workspace_size_ =
      max_ws_bytes_ - scale_probabilities_size_ - src_to_dest_map_size_ - selected_expert_size_ - lora_workspace_size_;

  return Status();
}

template <typename T>
Status MoeLayer<T>::Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) {
  config_map_.resize(runtime_config.max_batch_size + 1);
  for (size_t m = 1; m <= static_cast<size_t>(runtime_config.max_batch_size); m++) {
    size_t best_config_index = InvokeMoeGemmConfigProfile<T, T, T>();
    config_map_[m] = best_config_index;
  }
  return Status();
}

template <typename T>
Status MoeLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  const size_t num_tokens = input_tensors[0].shape[0];
  size_t best_config_index = 0;  // TODO(winminkong): op optimization
  void* e_score_correction_bias_weight_void = nullptr;
  if (use_e_score_correction_bias_) {
    e_score_correction_bias_weight_void = input_tensors[5].GetPtr<void>();
  }

  // SetWorkSpaceBuffer只保证空间足够大，不能保证后续地址不发生改变，要写死就只能在Forward中做
  if (set_workspace_buffer_info_) {
    set_workspace_buffer_info_ = false;

    workspace_info_.size = max_ws_bytes_;
    workspace_info_.workspace = workspace_buffer_->GetPtr<void>();
    workspace_info_.scale_probs =
        llm_kernels::utils::nextWorkspacePtr(reinterpret_cast<int8_t*>(workspace_info_.workspace), moe_workspace_size_);
    workspace_info_.src_to_dest_map = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(workspace_info_.scale_probs), scale_probabilities_size_);
    workspace_info_.selected_experts = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(workspace_info_.src_to_dest_map), src_to_dest_map_size_);
    workspace_info_.lora_workspace = llm_kernels::utils::nextWorkspacePtr(
        reinterpret_cast<int8_t*>(workspace_info_.selected_experts), selected_expert_size_);

    if (use_vllm_moe_) {
      topk_weights_ptr_ = workspace_buffer_->GetPtr<void>();
      topk_ids_ptr_ = topk_weights_ptr_ + topk_weights_ptr_size;
      fused_id_buffer_ = topk_ids_ptr_ + topk_ids_ptr_size;
      intermediate_cache1_ = fused_id_buffer_ + max_fused_id_buffer_size;
      intermediate_cache2_ = intermediate_cache1_ + intermediate_cache1_and_cache3_size;
      intermediate_cache3_ = intermediate_cache1_;  // 共享
      if (compute_dtype_ == DataType::TYPE_BLOCK_FP8_E4M3) {
        a1_q_ = intermediate_cache2_ + intermediate_cache2_size;
        a2_q_ = a1_q_;  // 共享
        a1_scale_ = a1_q_ + a1_and_a2_q_size;
        a2_scale_ = a1_scale_;  // 共享
        if (weight_dtype_ == DataType::TYPE_I4_GROUP) {
          dequant_workspace_ = a1_scale_ + a1_and_a2_scale_size;
        }
      }
    }
  }

  if (use_vllm_moe_) {
    // input_tensors: 0.hidden states 1.routing_out 2.up_gate_experts 3.down_experts 4.bias
    void* w1_scale = nullptr;
    void* w2_scale = nullptr;
    if (weight_dtype_ == DataType::TYPE_BLOCK_FP8_E4M3) {
      w1_scale = input_tensors[2].weight_scales->GetPtr<void>();
      w2_scale = input_tensors[3].weight_scales->GetPtr<void>();
    } else if (weight_dtype_ == DataType::TYPE_I4_GROUP) {
      w1_scale = input_tensors[2].scales->GetPtr<void>();
      w2_scale = input_tensors[3].scales->GetPtr<void>();
    }

    // 使用 GroupedTopkLayer 计算 topk
    int num_tokens = input_tensors[0].shape[0];
    ExecuteGroupedTopk(input_tensors, num_tokens);
    size_t expert_para_size = Singleton<Environment>::GetInstance()->GetExpertParallelSize() *
                              Singleton<Environment>::GetInstance()->GetExpertWorldSize();
    if (expert_para_size == 1) {
      InvokeFusedMoe<T, false>(input_tensors[0].GetPtr<void>(),              // hidden_states
                               input_tensors[2].GetPtr<void>(),              // w1
                               input_tensors[3].GetPtr<void>(),              // w2
                               input_tensors[4].GetPtr<int>(),               // expert_map
                               expert_topk_,                                 // topk
                               norm_topk_prob_,                              // renormalize
                               scoring_func_,                                // scoring_func_
                               e_score_correction_bias_weight_void,          // e_bias
                               true,                                         // inplace
                               num_expert_group_ != 1,                       // use_grouped_topk
                               num_expert_group_,                            // num_expert_group
                               expert_groups_topk_,                          // topk_group
                               weight_dtype_,                                // weight_dtype
                               compute_dtype_,                               // compute_dtype
                               false,                                        // is_marlin
                               false,                                        // use_triton
                               w1_scale,                                     // w1_scale
                               w2_scale,                                     // w2_scale
                               nullptr,                                      // w1_zp
                               nullptr,                                      // w2_zp
                               a1_q_,                                        // a1_q
                               a2_q_,                                        // a2_q
                               a1_scale_,                                    // a1_scale
                               a2_scale_,                                    // a2_scale
                               block_shape_,                                 // block_shape
                               topk_weights_ptr_,                            // topk_weights_ptr
                               topk_ids_ptr_,                                // topk_ids_ptr
                               routed_scaling_factor_,                       // routed_scaling_factor
                               output_tensors[0].GetPtr<void>(),             // output_hidden_states
                               intermediate_cache1_,                         // intermediate_cache1
                               intermediate_cache2_,                         // intermediate_cache2
                               intermediate_cache3_,                         // intermediate_cache3
                               fused_id_buffer_,                             // buffer_of_ids_in_kernel
                               num_tokens,                                   // num_tokens
                               expert_num_,                                  // num_experts
                               expert_hidden_size_,                          // hidden_size
                               expert_inter_size_,                           // inter_size
                               dequant_workspace_,                           // dequant_workspace
                               rank_,                                        // rank
                               context_->GetComputeStreams()[rank_].Get());  // stream
    } else {
      InvokeFusedMoe<T, true>(input_tensors[0].GetPtr<void>(),              // hidden_states
                              input_tensors[2].GetPtr<void>(),              // w1
                              input_tensors[3].GetPtr<void>(),              // w2
                              input_tensors[4].GetPtr<int>(),               // expert_map
                              expert_topk_,                                 // topk
                              norm_topk_prob_,                              // renormalize
                              scoring_func_,                                // scoring_func_
                              e_score_correction_bias_weight_void,          // e_bias
                              true,                                         // inplace
                              num_expert_group_ != 1,                       // use_grouped_topk
                              num_expert_group_,                            // num_expert_group
                              expert_groups_topk_,                          // topk_group
                              weight_dtype_,                                // weight_dtype
                              compute_dtype_,                               // compute_dtype
                              false,                                        // is_marlin
                              false,                                        // use_triton
                              w1_scale,                                     // w1_scale
                              w2_scale,                                     // w2_scale
                              nullptr,                                      // w1_zp
                              nullptr,                                      // w2_zp
                              a1_q_,                                        // a1_q
                              a2_q_,                                        // a2_q
                              a1_scale_,                                    // a1_scale
                              a2_scale_,                                    // a2_scale
                              block_shape_,                                 // block_shape
                              topk_weights_ptr_,                            // topk_weights_ptr
                              topk_ids_ptr_,                                // topk_ids_ptr
                              routed_scaling_factor_,                       // routed_scaling_factor
                              output_tensors[0].GetPtr<void>(),             // output_hidden_states
                              intermediate_cache1_,                         // intermediate_cache1
                              intermediate_cache2_,                         // intermediate_cache2
                              intermediate_cache3_,                         // intermediate_cache3
                              fused_id_buffer_,                             // buffer_of_ids_in_kernel
                              num_tokens,                                   // num_tokens
                              expert_num_,                                  // num_experts
                              expert_hidden_size_,                          // hidden_size
                              expert_inter_size_,                           // inter_size
                              dequant_workspace_,                           // dequant_workspace
                              rank_,                                        // rank
                              context_->GetComputeStreams()[rank_].Get());  // stream
    }
    // template void InvokeFusedMoe<T>(
    //   void* hidden_states, void* w1, void* w2, int* expert_map, int topk, bool renormalize,
    //   const std::string& scoring_func_, void* e_bias, bool inplace, bool use_grouped_topk, int num_expert_group,
    //   int topk_group, bool use_fp8_w8a8, bool use_int8_w8a16, bool use_int4_w4a16, bool is_marlin,
    //   bool use_triton, void* w1_scale, void* w2_scale, void* w1_zp, void* w2_zp, void* a1_q, void* a2_q,
    //   void* a1_scale, void* a2_scale, std::vector<int> block_shape, void* topk_weights_ptr, void* topk_ids_ptr,
    //   float routed_scaling_factor, void* output_hidden_states, void* intermediate_cache1, void* intermediate_cache2,
    //   void* intermediate_cache3, int num_tokens, int num_experts, int hidden_size, int inter_size, int rank,
    //   cudaStream_t stream)
  } else {
    // input_tensors: 0.hidden states 1.routing_out 2.up_gate_experts 3.down_experts 4.bias
    if (moe_scale_norm_mode_ == MoeScaleNormMode::RE_NORM) {
      InvokeMoeCutlassGemm<T, T, T, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE>(
          input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
          input_tensors[3].GetPtr<void>(), e_score_correction_bias_weight_void, num_tokens, expert_hidden_size_,
          expert_inter_size_, expert_num_, expert_topk_, static_cast<char*>(workspace_info_.workspace),
          output_tensors[0].GetPtr<void>(), workspace_info_.scale_probs,
          static_cast<int*>(workspace_info_.src_to_dest_map), static_cast<int*>(workspace_info_.selected_experts),
          tp_size_, rank_, use_lora_, best_config_index, use_vllm_moe_, num_expert_group_, expert_groups_topk_,
          scoring_func_, topk_method_, norm_topk_prob_, routed_scaling_factor_, use_e_score_correction_bias_,
          context_->GetComputeStreams()[rank_].Get(), false, nullptr, nullptr, nullptr, apply_weight_);
    } else if (moe_scale_norm_mode_ == MoeScaleNormMode::NO_NORM) {
      InvokeMoeCutlassGemm<T, T, T, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE>(
          input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
          input_tensors[3].GetPtr<void>(), e_score_correction_bias_weight_void, num_tokens, expert_hidden_size_,
          expert_inter_size_, expert_num_, expert_topk_, static_cast<char*>(workspace_info_.workspace),
          output_tensors[0].GetPtr<void>(), workspace_info_.scale_probs,
          static_cast<int*>(workspace_info_.src_to_dest_map), static_cast<int*>(workspace_info_.selected_experts),
          tp_size_, rank_, use_lora_, best_config_index, use_vllm_moe_, num_expert_group_, expert_groups_topk_,
          scoring_func_, topk_method_, norm_topk_prob_, routed_scaling_factor_, use_e_score_correction_bias_,
          context_->GetComputeStreams()[rank_].Get(), false, nullptr, nullptr, nullptr, apply_weight_);
    }
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;

  return Status();
}

template <typename T>
Status MoeLayer<T>::ExecuteGroupedTopk(const std::vector<Tensor>& input_tensors, int num_tokens) {
  // 准备 GroupedTopkLayer 的输入和输出张量
  std::vector<Tensor> grouped_topk_input_tensors;
  std::vector<Tensor> grouped_topk_output_tensors;

  // 输入: gating_output
  grouped_topk_input_tensors.push_back(input_tensors[1]);

  // 输入: e_bias (直接传递，让 GroupedTopkLayer 内部判断是否使用)
  if (input_tensors.size() > 5) {
    grouped_topk_input_tensors.push_back(input_tensors[5]);
  }

  // 输出: topk_weights_ptr
  Tensor topk_weights_tensor(input_tensors[1].location, TYPE_FP32,
                             {static_cast<size_t>(num_tokens), static_cast<size_t>(expert_topk_)},
                             input_tensors[1].device_id, topk_weights_ptr_);
  grouped_topk_output_tensors.push_back(topk_weights_tensor);

  // 输出: topk_ids_ptr
  Tensor topk_ids_tensor(input_tensors[1].location, TYPE_INT32,
                         {static_cast<size_t>(num_tokens), static_cast<size_t>(expert_topk_)},
                         input_tensors[1].device_id, topk_ids_ptr_);
  grouped_topk_output_tensors.push_back(topk_ids_tensor);

  // 调用 GroupedTopkLayer
  return grouped_topk_layer_->Forward(grouped_topk_input_tensors, grouped_topk_output_tensors);
}

template class MoeLayer<float>;
template class MoeLayer<half>;
template class MoeLayer<__nv_bfloat16>;

#endif
}  // namespace ksana_llm
