/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/moe_layer.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#include "ksana_llm/data_hub/expert_data_hub.h"
#include "ksana_llm/kernels/nvidia/basic_kernel_wrapper.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/layers/grouped_topk_layer.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

// MOE layer tensor数量常量
constexpr size_t kA1QTensorIndex = 2;        // a1_q_tensor在output_tensors中的索引
constexpr size_t kA1ScaleTensorIndex = 3;    // a1_scale_tensor在output_tensors中的索引
constexpr size_t kWorkspaceTensorIndex = 4;  // workspace_tensor在output_tensors中的索引

Status MoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) {
  inter_data_type_ = runtime_config.inter_data_type;
  DISPATCH_BY_3_DTYPE(inter_data_type_, InitT, parameters, runtime_config, context, rank);
}

size_t MoeLayer::GetWorkspaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

Status MoeLayer::Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, PreprocessT, model_config, runtime_config);
}

Status MoeLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status MoeLayer::InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                       std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;

  int parameter_index = 0;
  moe_scale_norm_mode_ = std::any_cast<const MoeScaleNormMode>(parameters[parameter_index++]);
  max_token_num_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  layer_idx_ = std::any_cast<int>(parameters[parameter_index++]);

  expert_num_per_node_ = std::any_cast<const size_t>(parameters[parameter_index++]);
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
  enable_full_shared_expert_ = std::any_cast<bool>(parameters[parameter_index++]);
  DataType fp8_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  DataType int_weight_dtype = std::any_cast<DataType>(parameters[parameter_index++]);
  int group_size = std::any_cast<int>(parameters[parameter_index++]);
  apply_weight_ = std::any_cast<bool>(parameters[parameter_index++]);

  w4afp8_moe_backend_ = runtime_config.w4afp8_moe_backend;

  // 权重&计算类型处理
  weight_dtype_ = GetDataType<T>();
  compute_dtype_ = GetDataType<T>();
  if (fp8_weight_dtype == DataType::TYPE_BLOCK_FP8_E4M3) {
    weight_dtype_ = DataType::TYPE_BLOCK_FP8_E4M3;
    compute_dtype_ = DataType::TYPE_BLOCK_FP8_E4M3;
  } else if (int_weight_dtype == DataType::TYPE_UINT4x2 || int_weight_dtype == DataType::TYPE_INT4x2) {
    weight_dtype_ = int_weight_dtype;
    compute_dtype_ = (w4afp8_moe_backend_ == W4AFP8_MOE_BACKEND::GroupTriton ||
                      w4afp8_moe_backend_ == W4AFP8_MOE_BACKEND::TensorTriton)
                         ? DataType::TYPE_BLOCK_FP8_E4M3
                         : compute_dtype_;
  }

  // fp8 blockwise处理
  block_shape_.resize(2);
  if (weight_dtype_ == DataType::TYPE_BLOCK_FP8_E4M3) {
    block_shape_ = {128, 128};
  } else if (weight_dtype_ == DataType::TYPE_UINT4x2 || weight_dtype_ == DataType::TYPE_INT4x2) {
    block_shape_ = {0, group_size};
  }

  // 计算全局ep_size和ep_rank
  global_expert_para_size_ = runtime_config.parallel_basic_config.expert_parallel_size *
                             runtime_config.parallel_basic_config.expert_world_size;
  size_t global_expert_para_rank_ =
      context_->GetExpertParallelExpertNodeRank() * runtime_config.parallel_basic_config.expert_parallel_size + rank_;

  // 多机 + DP + EP 才使用deepep
  // TODO(jinxcwu) 后续需要修改到更高层的配置中，例如context
  using_deepep_ = runtime_config.parallel_basic_config.expert_world_size > 1 &&
                  runtime_config.parallel_basic_config.attn_data_parallel_size > 1 &&
                  runtime_config.parallel_basic_config.expert_parallel_size > 1;

  // Initialize GroupedTopkLayer
  grouped_topk_layer_ = std::make_shared<GroupedTopkLayer>();
  std::vector<std::any> grouped_topk_params = {
      static_cast<int>(expert_topk_),        norm_topk_prob_, static_cast<int>(num_expert_group_),
      static_cast<int>(expert_groups_topk_), scoring_func_,   routed_scaling_factor_,
      use_e_score_correction_bias_};
  grouped_topk_layer_->Init(grouped_topk_params, runtime_config, context, rank);

  // EPLB data dump is off by default; when enabled, pick up the output directory
  // from env-var DUMP_EPLB_PATH or fall back to the canonical EPLB cache path.
  enable_dump_eplb_data_ = runtime_config.enable_dump_eplb_data;
  if (enable_dump_eplb_data_) {
    const char* eplb_dump_path = std::getenv("DUMP_EPLB_PATH");
    if (eplb_dump_path) {
      eplb_dump_path_ = eplb_dump_path;
    } else {
      const char* home_dir = std::getenv("HOME");
      eplb_dump_path_ = home_dir ? fmt::format("{}/.cache/KsanaLLM/EPLB/", std::string(home_dir)) : "./EPLB/";
    }
  }

  // When EPLB acceleration is enabled, initialize the expert_map_.
  enable_load_eplb_weight_ = runtime_config.enable_load_eplb_weight;
  if (enable_load_eplb_weight_ || !using_deepep_) {
    expert_map_ = std::make_shared<llm_kernels::nvidia::moe::ExpertMap>(
        global_expert_para_size_, global_expert_para_rank_, expert_num_per_node_ * global_expert_para_size_);
  }

  KLLM_LOG_INFO << fmt::format("Rank[{}] MoeLayer Init with weight_dtype:{} compute_dtype:{}", rank_, weight_dtype_,
                               compute_dtype_);
  return Status();
}

#define VLLM_FUSED_MOE_CHUNK_SIZE ((size_t)(32 * 1024))

inline size_t AlignAddress(size_t size) { return (size + 255) & (~255); }

template <typename T>
size_t MoeLayer::GetWorkspaceSizeT() {
  GetMoeGemmWorkspaceSize<T, T, T>(max_token_num_, expert_num_per_node_, expert_hidden_size_, expert_inter_size_,
                                   expert_topk_, tp_size_, rank_, use_lora_, max_ws_bytes_,
                                   workspace_info_.workspace_sizes);
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
      if (using_deepep_) {
        // 提前传输fp8量化后的数据，空间大小按照max_token_num_计算
        // TODO(zakwang): 优化空间使用，复用已有的空间
        m = std::max(m, max_token_num_);
        KLLM_LOG_INFO << fmt::format("using_deepep_, m = {}", m);
      }
      a1_q_size = AlignAddress(m * expert_hidden_size_ * sizeof(char));
      a2_q_size = AlignAddress(m * expert_topk_ * expert_inter_size_ * sizeof(char));
      a1_scale_size = AlignAddress(m * expert_hidden_size_ / 128 * sizeof(float));
      a2_scale_size = AlignAddress(m * expert_topk_ * expert_inter_size_ / 128 * sizeof(float));
      a1_and_a2_q_size = std::max(a1_q_size, a2_q_size);              // 共享
      a1_and_a2_scale_size = std::max(a1_scale_size, a2_scale_size);  // 共享
      if (weight_dtype_ == DataType::TYPE_UINT4x2 || weight_dtype_ == DataType::TYPE_INT4x2) {
        // TODO(jinxcwu) too large
        dequant_workspace_size =
            AlignAddress(expert_num_per_node_ * expert_hidden_size_ * expert_inter_size_ * 2 * sizeof(char));
      }
    }
    max_ws_bytes_ = topk_weights_ptr_size + topk_ids_ptr_size + max_fused_id_buffer_size +
                    intermediate_cache1_and_cache3_size + intermediate_cache2_size + a1_and_a2_q_size +
                    a1_and_a2_scale_size + dequant_workspace_size;
  }
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for MoeLayer", rank_, max_ws_bytes_);
  return max_ws_bytes_;
}

Status MoeLayer::SetWorkspaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) {
  workspace_buffer_ = workspace_buffer;
  scale_probabilities_size_ = max_token_num_ * expert_num_per_node_ * sizeof(float);
  src_to_dest_map_size_ = expert_topk_ * max_token_num_ * sizeof(int);
  selected_expert_size_ = expert_topk_ * max_token_num_ * sizeof(int);
  lora_workspace_size_ = 0;  // NO support for lora
  moe_workspace_size_ =
      max_ws_bytes_ - scale_probabilities_size_ - src_to_dest_map_size_ - selected_expert_size_ - lora_workspace_size_;

  return Status();
}

template <typename T>
Status MoeLayer::PreprocessT(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) {
  config_map_.resize(runtime_config.max_batch_size + 1);
  for (size_t m = 1; m <= static_cast<size_t>(runtime_config.max_batch_size); m++) {
    size_t best_config_index = InvokeMoeGemmConfigProfile<T, T, T>(tactics_);
    config_map_[m] = best_config_index;
  }
  return Status();
}

template <typename T>
Status MoeLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //  0: hidden_states
  //  1: gating_output
  //  2: up_gate_proj_weight
  //  3: down_proj_weight
  //  4: eplb_expert_map
  //  (*)5: e_score_correction_bias_weight
  const size_t num_tokens = input_tensors[0].shape[0];
  const size_t hidden_size = input_tensors[0].shape[1];
  size_t best_config_index = 0;  // TODO(winminkong): op optimization
  void* e_score_correction_bias_weight_void = nullptr;
  if (use_e_score_correction_bias_) {
    e_score_correction_bias_weight_void = input_tensors[5].GetPtr<void>();
  }

  // SetWorkspaceBuffer只保证空间足够大，不能保证后续地址不发生改变，要写死就只能在Forward中做
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
        if (weight_dtype_ == DataType::TYPE_UINT4x2 || weight_dtype_ == DataType::TYPE_INT4x2) {
          dequant_workspace_ = a1_scale_ + a1_and_a2_scale_size;
        }
      }
    }
  }

  if (use_vllm_moe_) {
    void* w1_scale = nullptr;
    void* w2_scale = nullptr;
    if (weight_dtype_ == DataType::TYPE_BLOCK_FP8_E4M3) {
      w1_scale = input_tensors[2].weight_scales->GetPtr<void>();
      w2_scale = input_tensors[3].weight_scales->GetPtr<void>();
    } else if (weight_dtype_ == DataType::TYPE_UINT4x2 || weight_dtype_ == DataType::TYPE_INT4x2) {
      w1_scale = input_tensors[2].scales->GetPtr<void>();
      w2_scale = input_tensors[3].scales->GetPtr<void>();
    }
    void* w1_input_scale = input_tensors[2].input_scales ? input_tensors[2].input_scales->GetPtr<void>() : nullptr;
    void* w2_input_scale = input_tensors[3].input_scales ? input_tensors[3].input_scales->GetPtr<void>() : nullptr;
    void* w1_input_alpha = input_tensors[2].input_alpha ? input_tensors[2].input_alpha->GetPtr<void>() : nullptr;
    void* w2_input_alpha = input_tensors[3].input_alpha ? input_tensors[3].input_alpha->GetPtr<void>() : nullptr;

    if (using_deepep_ && compute_dtype_ == DataType::TYPE_BLOCK_FP8_E4M3) {
      Tensor workspace_tensor(input_tensors[1].location, TYPE_FP8_E4M3, {num_tokens, hidden_size},
                              input_tensors[1].device_id, workspace_buffer_->GetPtr<void>());
      Tensor a1_q_tensor(input_tensors[1].location, TYPE_FP8_E4M3, {num_tokens, hidden_size},
                         input_tensors[1].device_id, a1_q_);
      Tensor a1_scale_tensor(input_tensors[1].location, TYPE_FP32, {num_tokens, hidden_size / 128},
                             input_tensors[1].device_id, a1_scale_);
      output_tensors.push_back(a1_q_tensor);
      output_tensors.push_back(a1_scale_tensor);
      output_tensors.push_back(workspace_tensor);
    }
    // 使用 GroupedTopkLayer 计算 topk
    ExecuteGroupedTopk<T>(input_tensors, output_tensors);

    auto AutoInvoke = [&](auto template_param, void* hidden_states_ptr, void* output_hidden_states_ptr,
                          int num_tokens) {
      InvokeFusedMoe<T, template_param>(hidden_states_ptr,                // hidden_states
                                        input_tensors[2].GetPtr<void>(),  // w1
                                        input_tensors[3].GetPtr<void>(),  // w2
                                        expert_topk_,                     // topk
                                        weight_dtype_,                    // weight_dtype
                                        compute_dtype_,                   // compute_dtype
                                        false,                            // is_marlin
                                        false,                            // use_triton
                                        w1_scale,                         // w1_scale
                                        w2_scale,                         // w2_scale
                                        nullptr,                          // w1_zp
                                        nullptr,                          // w2_zp
                                        a1_q_,                            // a1_q
                                        a2_q_,                            // a2_q
                                        w1_input_scale,                   // w1_input_scale
                                        w2_input_scale,                   // w2_input_scale
                                        a1_scale_,                        // a1_scale
                                        a2_scale_,                        // a2_scale
                                        w1_input_alpha,                   // w1_input_alpha
                                        w2_input_alpha,                   // w2_input_alpha
                                        block_shape_,                     // block_shape
                                        topk_weights_ptr_,                // topk_weights_ptr
                                        topk_ids_ptr_,                    // topk_ids_ptr
                                        routed_scaling_factor_,           // routed_scaling_factor
                                        output_hidden_states_ptr,         // output_hidden_states
                                        intermediate_cache1_,             // intermediate_cache1
                                        intermediate_cache2_,             // intermediate_cache2
                                        intermediate_cache3_,             // intermediate_cache3
                                        fused_id_buffer_,                 // buffer_of_ids_in_kernel
                                        num_tokens,                       // num_tokens
                                        expert_num_per_node_,             // num_experts_per_node
                                        expert_hidden_size_,              // hidden_size
                                        expert_inter_size_,               // inter_size
                                        global_expert_para_size_,         // expert_para_size * expert_world_size
                                        dequant_workspace_,               // dequant_workspace
                                        w4afp8_moe_backend_,              // w4afp8_moe_backend
                                        rank_,                            // rank
                                        context_->GetComputeStreams()[rank_].Get());  // stream
    };

    if (global_expert_para_size_ == 1) {  // 单卡情况
      AutoInvoke(std::false_type{}, input_tensors[0].GetPtr<void>(), output_tensors[0].GetPtr<void>(), num_tokens);
    } else if (!using_deepep_) {  // EP但不使用deepep
      expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(topk_ids_ptr_), num_tokens * expert_topk_,
                                          context_->GetComputeStreams()[rank_].Get());
      AutoInvoke(std::true_type{}, input_tensors[0].GetPtr<void>(), output_tensors[0].GetPtr<void>(), num_tokens);
    } else {  // EP使用deepep
      const size_t dispatch_num_tokens = output_tensors[1].shape[0];
      if (dispatch_num_tokens > 0) {
        void* hidden_states_ptr = output_tensors[1].GetPtr<void>();
        if (using_deepep_ && compute_dtype_ == DataType::TYPE_BLOCK_FP8_E4M3) {
          // 使用fp8的a1_q，hidden_states设置为nullptr
          hidden_states_ptr = nullptr;
        }
        AutoInvoke(std::true_type{}, hidden_states_ptr, output_tensors[1].GetPtr<void>(), dispatch_num_tokens);
      }
      // Combine is an in-place operation
      Combine({output_tensors[1]}, output_tensors);
    }
  } else {
    // input_tensors: 0.hidden states 1.routing_out 2.up_gate_experts 3.down_experts 4.bias
    if (moe_scale_norm_mode_ == MoeScaleNormMode::RE_NORM) {
      InvokeMoeCutlassGemm<T, T, T, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE>(
          input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
          input_tensors[3].GetPtr<void>(), e_score_correction_bias_weight_void, num_tokens, expert_hidden_size_,
          expert_inter_size_, expert_num_per_node_, expert_topk_, workspace_info_.workspace_sizes,
          static_cast<char*>(workspace_info_.workspace), output_tensors[0].GetPtr<void>(), workspace_info_.scale_probs,
          static_cast<int*>(workspace_info_.src_to_dest_map), static_cast<int*>(workspace_info_.selected_experts),
          tp_size_, rank_, use_lora_, best_config_index, tactics_, use_vllm_moe_, num_expert_group_,
          expert_groups_topk_, scoring_func_, topk_method_, norm_topk_prob_, routed_scaling_factor_,
          use_e_score_correction_bias_, context_->GetComputeStreams()[rank_].Get(), false, nullptr, nullptr, nullptr,
          apply_weight_);
    } else if (moe_scale_norm_mode_ == MoeScaleNormMode::NO_NORM) {
      InvokeMoeCutlassGemm<T, T, T, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE>(
          input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
          input_tensors[3].GetPtr<void>(), e_score_correction_bias_weight_void, num_tokens, expert_hidden_size_,
          expert_inter_size_, expert_num_per_node_, expert_topk_, workspace_info_.workspace_sizes,
          static_cast<char*>(workspace_info_.workspace), output_tensors[0].GetPtr<void>(), workspace_info_.scale_probs,
          static_cast<int*>(workspace_info_.src_to_dest_map), static_cast<int*>(workspace_info_.selected_experts),
          tp_size_, rank_, use_lora_, best_config_index, tactics_, use_vllm_moe_, num_expert_group_,
          expert_groups_topk_, scoring_func_, topk_method_, norm_topk_prob_, routed_scaling_factor_,
          use_e_score_correction_bias_, context_->GetComputeStreams()[rank_].Get(), false, nullptr, nullptr, nullptr,
          apply_weight_);
    }
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template <typename T>
Status MoeLayer::ExecuteGroupedTopk(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // 准备 GroupedTopkLayer 的输入和输出张量
  std::vector<Tensor> grouped_topk_input_tensors;
  std::vector<Tensor> grouped_topk_output_tensors;

  size_t num_tokens = input_tensors[0].shape[0];

  // 输入: gating_output
  grouped_topk_input_tensors.push_back(input_tensors[1]);

  // 输入: e_bias (直接传递，让 GroupedTopkLayer 内部判断是否使用)
  if (input_tensors.size() > 5) {
    grouped_topk_input_tensors.push_back(input_tensors[5]);
  }

  // 输出: topk_weights_ptr
  Tensor topk_weights_tensor(input_tensors[1].location, TYPE_FP32, {num_tokens, expert_topk_},
                             input_tensors[1].device_id, topk_weights_ptr_);
  grouped_topk_output_tensors.push_back(topk_weights_tensor);

  // 输出: topk_ids_ptr
  Tensor topk_ids_tensor(input_tensors[1].location, TYPE_INT32, {num_tokens, expert_topk_}, input_tensors[1].device_id,
                         topk_ids_ptr_);
  grouped_topk_output_tensors.push_back(topk_ids_tensor);

  // 调用 GroupedTopkLayer
  Status status = grouped_topk_layer_->Forward(grouped_topk_input_tensors, grouped_topk_output_tensors);
  if (!status.OK()) {
    KLLM_LOG_ERROR << fmt::format("ExecuteGroupedTopk ERROR: failed to forward grouped_topk_layer");
    return status;
  }

  if (enable_dump_eplb_data_) {
    DumpEplbData(topk_ids_tensor);
  }

  if (enable_load_eplb_weight_ && input_tensors[4].shape[0] > layer_idx_) {
    size_t num_experts = input_tensors[4].shape[1];
    void* layer_expert_ptr = input_tensors[4].GetPtr<void>() + layer_idx_ * num_experts * sizeof(int);
    expert_map_->InvokeExpertMapInplace(static_cast<int32_t*>(topk_ids_ptr_), num_tokens * expert_topk_,
                                        reinterpret_cast<int32_t*>(layer_expert_ptr),
                                        context_->GetComputeStreams()[rank_].Get());
  }

  if (using_deepep_) {
    // 调用 Dispatch 分发
    // 分发结果将被存储到 common_mlp_tensor, topk_ids, topk_weights 中
    std::vector<Tensor> deepep_input_tensors = {input_tensors[0], topk_ids_tensor, topk_weights_tensor};
    std::vector<Tensor>& deepep_output_tensors = output_tensors;
    // 在 InvokeFusedMoe 前进行首次量化（当数据类型为 TYPE_BLOCK_FP8_E4M3 时）
    bool use_scales = output_tensors.size() > kWorkspaceTensorIndex;
    if (use_scales) {
      Tensor a1_q_tensor = output_tensors[kA1QTensorIndex];
      Tensor a1_scale_tensor = output_tensors[kA1ScaleTensorIndex];
      Tensor workspace_tensor = output_tensors[kWorkspaceTensorIndex];
      InvokePerTokenGroupQuantFp8E4m3<T>(input_tensors[0].GetPtr<void>(), a1_q_tensor.GetPtr<void>(),
                                         a1_scale_tensor.GetPtr<void>(), num_tokens, expert_hidden_size_, false,
                                         context_->GetComputeStreams()[rank_].Get());
      deepep_input_tensors.push_back(a1_q_tensor);
      deepep_input_tensors.push_back(a1_scale_tensor);
      deepep_input_tensors.push_back(workspace_tensor);
    }
    KLLM_LOG_DEBUG << fmt::format("ExecuteGroupedTopk: Dispatch shape {} {}", deepep_input_tensors[0].shape[0],
                                  deepep_input_tensors[1].shape[0]);
    Dispatch(deepep_input_tensors, deepep_output_tensors);
    KLLM_LOG_DEBUG << fmt::format("ExecuteGroupedTopk: Dispatch output shape {} {}", deepep_output_tensors[0].shape[0],
                                  deepep_output_tensors[1].shape[0]);
    topk_ids_tensor.shape[0] = deepep_output_tensors[0].shape[0];
    topk_weights_tensor.shape[0] = deepep_output_tensors[0].shape[0];
  }
  return Status();
}

Status MoeLayer::Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (GetExpertParallelDeepepWrapper()) {
    GetExpertParallelDeepepWrapper()->Dispatch(input_tensors, output_tensors, rank_);
  }
  return Status();
}

Status MoeLayer::Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (GetExpertParallelDeepepWrapper()) {
    GetExpertParallelDeepepWrapper()->Combine(input_tensors, output_tensors, rank_);
  }
  return Status();
}

Status MoeLayer::DumpEplbData(Tensor& topk_ids) {
  // TODO(zezhao): 高并发的 SaveToNpyFile 会出现
  topk_ids.SaveToNpyFile(
      fmt::format("{}/layer_{}/topk_ids_{}_{}.npy", eplb_dump_path_, layer_idx_, eplb_dump_step_, rank_));
  eplb_dump_step_ += 1;
  return Status();
}

}  // namespace ksana_llm
