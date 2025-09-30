/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/cutlass_moe_layer.h"

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/utils.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/layers/grouped_topk_layer.h"
#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/search_status.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

using KTensor = llm_kernels::nvidia::tensorrt_llm::dev::Tensor;
using KScalarType = llm_kernels::nvidia::tensorrt_llm::dev::ScalarType;

static const std::unordered_map<DataType, KScalarType> DataTypeToScalarTypeMap = {
    {DataType::TYPE_INT64, KScalarType::Long},     {DataType::TYPE_FP8_E4M3, KScalarType::Float8_e4m3fn},
    {DataType::TYPE_UINT8, KScalarType::QUInt4x2},  // NOTE(jinxcwu) 特殊配置的，需要注意
    {DataType::TYPE_INT8, KScalarType::QUInt4x2},   // NOTE(jinxcwu) 特殊配置的，需要注意
    {DataType::TYPE_INT32, KScalarType::Int},      {DataType::TYPE_FP32, KScalarType::Float},
    {DataType::TYPE_BF16, KScalarType::BFloat16},  {DataType::TYPE_FP16, KScalarType::Half}};

inline KTensor TensorToKTensor(const Tensor& tensor) {
  return KTensor(tensor.GetPtr<void>(), tensor.shape, DataTypeToScalarTypeMap.at(tensor.dtype));
}

Status CutlassMoeLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                             std::shared_ptr<Context> context, int rank) {
  inter_data_type_ = runtime_config.inter_data_type;
  DISPATCH_BY_3_DTYPE(inter_data_type_, InitT, parameters, runtime_config, context, rank);
}

template <typename T>
Status CutlassMoeLayer::InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
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
  group_size_ = std::any_cast<int>(parameters[parameter_index++]);
  apply_weight_ = std::any_cast<bool>(parameters[parameter_index++]);

  // 初始化 GroupedTopkLayer
  grouped_topk_layer_ = std::make_shared<GroupedTopkLayer>();
  std::vector<std::any> grouped_topk_params = {
      static_cast<int>(expert_topk_),        norm_topk_prob_, static_cast<int>(num_expert_group_),
      static_cast<int>(expert_groups_topk_), scoring_func_,   routed_scaling_factor_,
      use_e_score_correction_bias_};
  grouped_topk_layer_->Init(grouped_topk_params, runtime_config, context, rank);

  // 强制要求TP=EP
  if (runtime_config.parallel_basic_config.tensor_parallel_size !=
      runtime_config.parallel_basic_config.expert_parallel_size) {
    KLLM_THROW("Currently, CutlassMoeLayer strictly enforces TP=EP.");
  }

  // 计算全局ep_size和ep_rank
  size_t ep_size = runtime_config.parallel_basic_config.expert_parallel_size *
                   runtime_config.parallel_basic_config.expert_world_size;
  size_t ep_rank =
      context_->GetExpertParallelExpertNodeRank() * runtime_config.parallel_basic_config.expert_parallel_size + rank_;
  cutlass_moe_wrapper_ =
      std::make_shared<llm_kernels::nvidia::CutlassMoeWrapper>(1, 0, ep_size, ep_rank, 1, 0, expert_topk_);
  cutlass_moe_wrapper_->Init<T>();

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
  if (enable_load_eplb_weight_ || using_deepep_) {
    size_t total_expert_num = expert_num_per_node_ * global_expert_para_size_;
    expert_map_ = std::make_shared<llm_kernels::nvidia::moe::ExpertMap>(global_expert_para_size_,
                                                                        global_expert_para_rank_, total_expert_num);
  }
  return Status();
}

#define TRTLLM_CUTLASS_MOE_CHUNK_SIZE ((size_t)(8 * 1024))

size_t CutlassMoeLayer::GetWorkSpaceSize() {
  topk_weights_ptr_size = RoundUp(max_token_num_ * expert_topk_ * sizeof(float), 256);
  topk_ids_ptr_size = RoundUp(max_token_num_ * expert_topk_ * sizeof(int32_t), 256);
  kernel_workspace_size = RoundUp(
      cutlass_moe_wrapper_->GetWorkspaceSize(std::min(TRTLLM_CUTLASS_MOE_CHUNK_SIZE, max_token_num_), expert_topk_,
                                             expert_num_per_node_, expert_hidden_size_, expert_inter_size_),
      256);
  max_ws_bytes_ = topk_weights_ptr_size + topk_ids_ptr_size + kernel_workspace_size;

  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for CutlassMoeLayer", rank_, max_ws_bytes_);
  return max_ws_bytes_;
}

Status CutlassMoeLayer::Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
  const size_t record_iters = GetEnvAsPositiveInt("QUANT_PROFILE", 10);
  if (record_iters == 0) {
    KLLM_LOG_DEBUG << "$QUANT_PROFILE==0, Skipping CutlassMoeLayer Preprocess";
    return Status();
  }
  const size_t warmup_iters = std::max(1UL, record_iters / 2);  // warmup不能为0

  static std::mutex g_mtx;
  std::lock_guard<std::mutex> guard(g_mtx);

  const auto start_time = ProfileTimer::GetCurrentTime();

  size_t max_profile_token = static_cast<size_t>(std::pow(2, std::ceil(std::log2(runtime_config.max_batch_size))));
  // 检查是否可以跳过
  if (Singleton<CutlassMoeSearchStatus>::GetInstance()->IsCutlassMoeScheduleContain(
          expert_topk_, expert_num_per_node_, expert_hidden_size_, expert_inter_size_)) {
    config_map_ = Singleton<CutlassMoeSearchStatus>::GetInstance()->GetCutlassMoeSchedule(
        expert_topk_, expert_num_per_node_, expert_hidden_size_, expert_inter_size_);
    KLLM_LOG_INFO << fmt::format("Reusing Profile CutlassMoeLayer Layer in rank:{}, token=({}~{}),({},{},{},{})", rank_,
                                 1, max_profile_token, expert_topk_, expert_num_per_node_, expert_hidden_size_,
                                 expert_inter_size_);
    return Status();
  }
  config_map_.resize(max_profile_token + 1);

  // 创建一份假权重
  Tensor w1_w3_weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                               {expert_num_per_node_, expert_inter_size_ * 2, expert_hidden_size_ / 2}, rank_);
  Tensor w2_weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                            {expert_num_per_node_, expert_hidden_size_, expert_inter_size_ / 2}, rank_);
  KTensor fc1_expert_weights = TensorToKTensor(w1_w3_weight);
  KTensor fc2_expert_weights = TensorToKTensor(w2_weight);

  for (size_t profile_token = 1; profile_token <= max_profile_token; profile_token *= 2) {
    auto get_gemm_best_tactic = [&](int64_t gemm_idx, size_t warmup_iters, size_t profile_iters) -> int64_t {
      // 获取workspace
      size_t profile_workspace_size = cutlass_moe_wrapper_->GetProfileWorkspace(
          fc1_expert_weights, std::nullopt, fc2_expert_weights, std::nullopt, profile_token, false, false, gemm_idx, -1,
          true, context_->GetComputeStreams()[rank_].Get());
      // 开辟workspace
      Tensor profile_workspace = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {profile_workspace_size}, rank_);
      // 设置workspace
      cutlass_moe_wrapper_->SetProfileWorkspace(profile_workspace.GetPtr<void>(), fc1_expert_weights, std::nullopt,
                                                fc2_expert_weights, std::nullopt, profile_token, false, false, gemm_idx,
                                                -1, true, context_->GetComputeStreams()[rank_].Get());
      // 获取最优tactic
      int64_t best_tactic = -1;
      float best_tactic_time = std::numeric_limits<float>::max();
      int64_t tactic_num = cutlass_moe_wrapper_->GetTacticNum();
      for (int64_t tactic = 0; tactic < tactic_num; tactic++) {
        auto kernel = [&]() {
          cutlass_moe_wrapper_->RunGemmProfile(fc1_expert_weights, std::nullopt, fc2_expert_weights, std::nullopt,
                                               profile_token, false, false, gemm_idx, tactic, false,
                                               context_->GetComputeStreams()[rank_].Get());
        };
        float tactic_time =
            MeasureCudaExecutionTime(kernel, context_->GetComputeStreams()[rank_].Get(), warmup_iters, profile_iters);
        if (tactic_time < best_tactic_time) {
          best_tactic_time = tactic_time;
          best_tactic = tactic;
        }
      }
      // 结束
      return best_tactic;
    };
    int64_t gemm1_tactic = get_gemm_best_tactic(1, warmup_iters, record_iters);
    int64_t gemm2_tactic = get_gemm_best_tactic(2, warmup_iters, record_iters);
    config_map_[profile_token] = {gemm1_tactic, gemm2_tactic};
  }
  for (size_t profile_token = 1; profile_token <= max_profile_token; profile_token++) {
    config_map_[profile_token] = config_map_[std::pow(2, static_cast<int>(std::log2(profile_token)))];
  }

  Singleton<CutlassMoeSearchStatus>::GetInstance()->AddCutlassMoeSchedule(
      expert_topk_, expert_num_per_node_, expert_hidden_size_, expert_inter_size_, config_map_);

  KLLM_LOG_INFO << fmt::format("Rank[{}] CutlassMoeLayer Preprocess cost time: {} s", rank_,
                               ProfileTimer::GetCurrentTime() - start_time);
  return Status();
}

Status CutlassMoeLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //  0: hidden_states
  //  1: gating_output
  //  2: up_gate_proj_weight
  //  3: down_proj_weight
  //  (*)4: e_score_correction_bias_weight
  if (set_workspace_buffer_info_) {
    set_workspace_buffer_info_ = false;
    topk_weights_ptr_ = workspace_buffer_->GetPtr<void>();
    topk_ids_ptr_ = topk_weights_ptr_ + topk_weights_ptr_size;
    kernel_workspace_ptr_ = topk_ids_ptr_ + topk_ids_ptr_size;
    cutlass_moe_wrapper_->SetWorkspacePtr(kernel_workspace_ptr_);
  }

  // 使用 GroupedTopkLayer 计算 topk
  size_t total_tokens = input_tensors[0].shape[0];
  ExecuteGroupedTopk(input_tensors, total_tokens);

  // 获取并转换权重
  const Tensor& up_gate_experts_weight = input_tensors[2];
  const Tensor& down_experts_weight = input_tensors[3];
  KTensor fc1_expert_weights_ktensor = TensorToKTensor(up_gate_experts_weight);
  KTensor fc2_expert_weights_ktensor = TensorToKTensor(down_experts_weight);
  KTensor fc1_weight_scales_ktensor = TensorToKTensor(*(up_gate_experts_weight.scales));
  KTensor fc2_weight_scales_ktensor = TensorToKTensor(*(down_experts_weight.scales));
  KTensor fc1_act_scales_ktensor = TensorToKTensor(*(up_gate_experts_weight.input_scales));
  KTensor fc2_act_scales_ktensor = TensorToKTensor(*(down_experts_weight.input_scales));
  KTensor fc1_weight_zeros_ktensor;
  KTensor fc2_weight_zeros_ktensor;
  KTensor fc1_alpha_ktensor = TensorToKTensor(*(up_gate_experts_weight.input_alpha));
  KTensor fc2_alpha_ktensor = TensorToKTensor(*(down_experts_weight.input_alpha));

  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;

  // chunk 分块处理
  for (size_t chunk_token = 0; chunk_token < total_tokens; chunk_token += TRTLLM_CUTLASS_MOE_CHUNK_SIZE) {
    size_t start_token = chunk_token;
    size_t end_token = std::min(chunk_token + TRTLLM_CUTLASS_MOE_CHUNK_SIZE, total_tokens);
    size_t solve_token = end_token - start_token;

    Tensor chunk_input(
        input_tensors[0].location, input_tensors[0].dtype, {solve_token, input_tensors[0].shape[1]},
        input_tensors[0].device_id,
        input_tensors[0].GetPtr<void>() + start_token * input_tensors[0].shape[1] * input_tensors[0].GetDTypeSize());
    Tensor chunk_output(
        output_tensors[0].location, output_tensors[0].dtype, {solve_token, output_tensors[0].shape[1]},
        output_tensors[0].device_id,
        output_tensors[0].GetPtr<void>() + start_token * output_tensors[0].shape[1] * output_tensors[0].GetDTypeSize());
    KTensor input_ktensor = TensorToKTensor(chunk_input);
    KTensor output_ktensor = TensorToKTensor(chunk_output);

    Tensor chunk_topk_weights(input_tensors[1].location, TYPE_FP32, {solve_token, expert_topk_},
                              input_tensors[1].device_id,
                              topk_weights_ptr_ + start_token * expert_topk_ * sizeof(float));
    Tensor chunk_topk_ids(input_tensors[1].location, TYPE_INT32, {solve_token, expert_topk_},
                          input_tensors[1].device_id, topk_ids_ptr_ + start_token * expert_topk_ * sizeof(int32_t));
    KTensor token_selected_experts_ktensor = TensorToKTensor(chunk_topk_ids);
    KTensor token_final_scales_ktensor = TensorToKTensor(chunk_topk_weights);

    std::vector<int64_t> best_config = {};
    if (!config_map_.empty()) {
      if (solve_token < config_map_.size()) {
        best_config = config_map_[solve_token];
      } else {
        best_config = config_map_[config_map_.size() - 1];
      }
    }
    cutlass_moe_wrapper_->Forward(
        output_ktensor, input_ktensor, token_selected_experts_ktensor, token_final_scales_ktensor,
        fc1_expert_weights_ktensor, fc2_expert_weights_ktensor,
        {fc1_weight_scales_ktensor, fc2_weight_scales_ktensor, fc1_act_scales_ktensor, fc2_act_scales_ktensor,
         fc1_weight_zeros_ktensor, fc2_weight_zeros_ktensor, fc1_alpha_ktensor, fc2_alpha_ktensor},
        best_config, context_->GetComputeStreams()[rank_].Get());
  }

  return Status();
}

Status CutlassMoeLayer::ExecuteGroupedTopk(const std::vector<Tensor>& input_tensors, int num_tokens) {
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

  // TODO(jinxcwu) 支持deepep的量化传输
  if (using_deepep_) {
    // LCOV_EXCL_START
    // 调用 Dispatch 分发
    // 分发结果将被存储到 common_mlp_tensor, topk_ids, topk_weights 中
    std::vector<Tensor> deepep_input_tensors = {input_tensors[0], topk_ids_tensor, topk_weights_tensor};
    std::vector<Tensor>& deepep_output_tensors = output_tensors;
    KLLM_LOG_DEBUG << fmt::format("ExecuteGroupedTopk: Dispatch shape {} {}", deepep_input_tensors[0].shape[0],
                                  deepep_input_tensors[1].shape[0]);
    Dispatch(deepep_input_tensors, deepep_output_tensors);
    KLLM_LOG_DEBUG << fmt::format("ExecuteGroupedTopk: Dispatch output shape {} {}", deepep_output_tensors[0].shape[0],
                                  deepep_output_tensors[1].shape[0]);
    topk_ids_tensor.shape[0] = deepep_output_tensors[0].shape[0];
    topk_weights_tensor.shape[0] = deepep_output_tensors[0].shape[0];
    // LCOV_EXCL_STOP
  }
  return Status();
}

Status CutlassMoeLayer::DumpEplbData(Tensor& topk_ids) {
  topk_ids.SaveToNpyFile(
      fmt::format("{}/layer_{}/topk_ids_{}_{}.npy", eplb_dump_path_, layer_idx_, eplb_dump_step_, rank_));
  eplb_dump_step_ += 1;
  return Status();
}

// LCOV_EXCL_START
Status CutlassMoeLayer::Dispatch(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (GetExpertParallelDeepepWrapper()) {
    GetExpertParallelDeepepWrapper()->Dispatch(input_tensors, output_tensors, rank_);
  }
  return Status();
}

Status CutlassMoeLayer::Combine(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (GetExpertParallelDeepepWrapper()) {
    GetExpertParallelDeepepWrapper()->Combine(input_tensors, output_tensors, rank_);
  }
  return Status();
}
// LCOV_EXCL_STOP

}  // namespace ksana_llm
