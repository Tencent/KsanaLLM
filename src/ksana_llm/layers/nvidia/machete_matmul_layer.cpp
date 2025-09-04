/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/machete_matmul_layer.h"

#include <unordered_map>

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/search_status.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/utils.h"

using namespace llm_kernels::nvidia;

namespace ksana_llm {

// vllm_dtype::ScalarType::Id到DataType的映射
static const std::unordered_map<vllm_dtype::ScalarType::Id, DataType> MacheteTypeIdToDataType = {
    {vllm_dtype::kHalf.id(), DataType::TYPE_FP16}, {vllm_dtype::kBFloat16.id(), DataType::TYPE_BF16}};

Status MacheteMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;

  int parameter_index = 0;
  max_m_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_n_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_k_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  groupsize_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  is_awq_ = std::any_cast<const bool>(parameters[parameter_index++]);
  [[maybe_unused]] bool is_gptq_desc_ = std::any_cast<const bool>(parameters[parameter_index++]);
  [[maybe_unused]] bool is_k_full_ = std::any_cast<const bool>(parameters[parameter_index++]);
  [[maybe_unused]] bool cutlass_use_gemv_cuda_core_ = std::any_cast<const bool>(parameters[parameter_index++]);
  weight_data_type_ = std::any_cast<const DataType>(parameters[parameter_index++]);

  return Status();
}

size_t MacheteMatMulLayer::GetWorkSpaceSize() {
  if (weight_data_type_ == TYPE_I4_GROUP) {
    // shape配置
    const size_t bits = 4;
    const size_t pack_factor = 32 / bits;
    const size_t max_posible_m = max_m_;
    const size_t posible_n = max_n_;
    const size_t posible_k = max_k_;

    static std::mutex g_mtx;
    std::lock_guard<std::mutex> guard(g_mtx);
    // 检查是否可以跳过
    if (Singleton<MacheteSearchStatus>::GetInstance()->IsMacheteWorkspaceContain(max_posible_m, posible_n, posible_k)) {
      KLLM_LOG_INFO << fmt::format("Reusing MacheteMatMul Layer Workspace in rank:{}, mnk=({},{},{})", rank_,
                                   max_posible_m, posible_n, posible_k);
      return Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteWorkspace(max_posible_m, posible_n, posible_k);
    }
    size_t max_ws_bytes = 0;

    // 根据模板类型确定各类型
    vllm_dtype::ScalarType::Id activation_type_id;
    if (inter_data_type_ == DataType::TYPE_FP16) {
      activation_type_id = vllm_dtype::kHalf.id();
    } else if (inter_data_type_ == DataType::TYPE_BF16) {
      activation_type_id = vllm_dtype::kBFloat16.id();
    } else {
      KLLM_THROW("MacheteMatMul only supports half and bfloat16 activation types");
    }
    vllm_dtype::ScalarType::Id weight_type_id = is_awq_ ? vllm_dtype::kU4.id() : vllm_dtype::kU4B8.id();
    vllm_dtype::ScalarType::Id scale_type_id = activation_type_id;
    vllm_dtype::ScalarType::Id zero_type_id = activation_type_id;

    // schedule与shape无关，只与类型有关
    std::vector<std::string> schedules = GetMacheteSupportedSchedules(
        vllm_dtype::ScalarType::from_id(activation_type_id), vllm_dtype::ScalarType::from_id(weight_type_id),
        vllm_dtype::ScalarType::from_id(scale_type_id),
        is_awq_ ? std::optional<vllm_dtype::ScalarType>(vllm_dtype::ScalarType::from_id(zero_type_id)) : std::nullopt);

    // 使用find方法安全地访问映射表，并添加错误处理
    auto activation_type_it = MacheteTypeIdToDataType.find(activation_type_id);
    if (activation_type_it == MacheteTypeIdToDataType.end()) {
      KLLM_THROW(fmt::format("Unsupported activation type ID: {}", activation_type_id));
    }
    DataType buffer_input_activation_type = activation_type_it->second;

    auto scale_type_it = MacheteTypeIdToDataType.find(scale_type_id);
    if (scale_type_it == MacheteTypeIdToDataType.end()) {
      KLLM_THROW(fmt::format("Unsupported scale type ID: {}", scale_type_id));
    }
    DataType buffer_input_scales_type = scale_type_it->second;

    auto zero_type_it = MacheteTypeIdToDataType.find(zero_type_id);
    if (zero_type_it == MacheteTypeIdToDataType.end()) {
      KLLM_THROW(fmt::format("Unsupported zero type ID: {}", zero_type_id));
    }
    DataType buffer_input_zeros_type = zero_type_it->second;

    DataType buffer_output_type = buffer_input_activation_type;

    Tensor buffer_input_activation(MemoryLocation::LOCATION_DEVICE, buffer_input_activation_type, {max_m_, posible_k},
                                   rank_);
    Tensor buffer_input_weight(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT32,
                               {posible_k / pack_factor, posible_n}, rank_);
    Tensor buffer_input_scales(MemoryLocation::LOCATION_DEVICE, buffer_input_scales_type,
                               {posible_k / groupsize_, posible_n}, rank_);
    Tensor buffer_input_zeros(MemoryLocation::LOCATION_DEVICE, buffer_input_zeros_type,
                              {posible_k / groupsize_, posible_n}, rank_);
    Tensor buffer_output(MemoryLocation::LOCATION_DEVICE, buffer_output_type, {max_m_, posible_n}, rank_);

    for (std::string curr_schedule : schedules) {                        // 遍历所有schedule
      for (size_t posible_m = 1; posible_m < max_m_ + 1; posible_m++) {  // 遍历所有m
        int64_t current_workspace_size = -1;
        // 实际只获取workspace，不会消耗时间
        InvokeMacheteGemm(current_workspace_size, nullptr, context_->GetComputeStreams()[rank_].Get(), posible_m,
                          posible_n, posible_k, buffer_input_activation.GetPtr<void>(),
                          buffer_input_weight.GetPtr<void>(), buffer_output.GetPtr<void>(),
                          vllm_dtype::ScalarType::from_id(activation_type_id),
                          vllm_dtype::ScalarType::from_id(weight_type_id), buffer_input_scales.GetPtr<void>(),
                          buffer_input_scales.shape, vllm_dtype::ScalarType::from_id(scale_type_id),
                          is_awq_ ? std::optional<void*>(buffer_input_zeros.GetPtr<void>()) : std::nullopt,
                          is_awq_ ? std::optional<std::vector<size_t>>(buffer_input_zeros.shape) : std::nullopt,
                          is_awq_ ? std::optional<vllm_dtype::ScalarType>(vllm_dtype::ScalarType::from_id(zero_type_id))
                                  : std::nullopt,
                          groupsize_, curr_schedule);

        // 检查current_workspace_size是否成功有效
        KLLM_CHECK_WITH_INFO(current_workspace_size != -1,
                             fmt::format("Machete get workspace size faild with ({},{},{}) and [{}].", posible_m,
                                         posible_n, posible_k, curr_schedule));
        max_ws_bytes = std::max(max_ws_bytes, static_cast<size_t>(current_workspace_size));
      }
    }

    Singleton<MacheteSearchStatus>::GetInstance()->AddMacheteWorkspace(max_posible_m, posible_n, posible_k,
                                                                       max_ws_bytes);

    KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for MacheteMatMulLayer", rank_, max_ws_bytes);
    return max_ws_bytes;
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. MacheteMatMul only supports TYPE_I4_GROUP.",
                           weight_data_type_));
  }
}

Status MacheteMatMulLayer::Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) {
  const size_t record_iters = GetEnvAsPositiveInt("QUANT_PROFILE", 5);
  if (record_iters == 0) {
    KLLM_LOG_INFO << "$QUANT_PROFILE==0, Skipping MacheteMatMulLayer Preprocess";
    return Status();
  }
  const size_t warmup_iters = std::max(1UL, record_iters / 2);  // warmup不能为0

  if (weight_data_type_ == TYPE_I4_GROUP) {
    // shape配置
    const size_t bits = 4;
    const size_t pack_factor = 32 / bits;
    const size_t max_posible_m = std::min(runtime_config.max_batch_size, 256);
    const size_t posible_n = max_n_;
    const size_t posible_k = max_k_;

    static std::mutex g_mtx;
    std::lock_guard<std::mutex> guard(g_mtx);
    // 检查是否可以跳过
    if (Singleton<MacheteSearchStatus>::GetInstance()->IsMacheteScheduleContain(posible_n, posible_k)) {
      machete_schedule_map_ = Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(posible_n, posible_k);
      KLLM_LOG_INFO << fmt::format("Reusing Profile MacheteMatMul Layer in rank:{}, mnk=({}~{},{},{})", rank_, 1,
                                   max_posible_m, posible_n, posible_k);
      return Status();
    }
    machete_schedule_map_.resize(max_posible_m + 1);  // start from 1 but not 0

    // 确定类型
    vllm_dtype::ScalarType::Id activation_type_id;
    if (inter_data_type_ == DataType::TYPE_FP16) {
      activation_type_id = vllm_dtype::kHalf.id();
    } else if (inter_data_type_ == DataType::TYPE_BF16) {
      activation_type_id = vllm_dtype::kBFloat16.id();
    } else {
      KLLM_THROW("MacheteMatMul only supports half and bfloat16 activation types");
    }
    vllm_dtype::ScalarType::Id weight_type_id = is_awq_ ? vllm_dtype::kU4.id() : vllm_dtype::kU4B8.id();
    vllm_dtype::ScalarType::Id scale_type_id = activation_type_id;
    vllm_dtype::ScalarType::Id zero_type_id = activation_type_id;

    auto activation_type_it = MacheteTypeIdToDataType.find(activation_type_id);
    if (activation_type_it == MacheteTypeIdToDataType.end()) {
      KLLM_THROW(fmt::format("Unsupported activation type ID: {}", activation_type_id));
    }
    DataType buffer_input_activation_type = activation_type_it->second;

    auto scale_type_it = MacheteTypeIdToDataType.find(scale_type_id);
    if (scale_type_it == MacheteTypeIdToDataType.end()) {
      KLLM_THROW(fmt::format("Unsupported scale type ID: {}", scale_type_id));
    }
    DataType buffer_input_scales_type = scale_type_it->second;

    auto zero_type_it = MacheteTypeIdToDataType.find(zero_type_id);
    if (zero_type_it == MacheteTypeIdToDataType.end()) {
      KLLM_THROW(fmt::format("Unsupported zero type ID: {}", zero_type_id));
    }
    DataType buffer_input_zeros_type = zero_type_it->second;

    DataType buffer_output_type = buffer_input_activation_type;

    // 开辟最大空间
    Tensor buffer_input_activation(MemoryLocation::LOCATION_DEVICE, buffer_input_activation_type,
                                   {max_posible_m, posible_k}, rank_);
    Tensor buffer_input_weight(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT32,
                               {posible_k / pack_factor, posible_n}, rank_);
    Tensor buffer_input_scales(MemoryLocation::LOCATION_DEVICE, buffer_input_scales_type,
                               {posible_k / groupsize_, posible_n}, rank_);
    Tensor buffer_input_zeros(MemoryLocation::LOCATION_DEVICE, buffer_input_zeros_type,
                              {posible_k / groupsize_, posible_n}, rank_);
    Tensor buffer_output(MemoryLocation::LOCATION_DEVICE, buffer_output_type, {max_posible_m, posible_n}, rank_);

    // 开始搜索
    auto start_profile_time = std::chrono::high_resolution_clock::now();
    for (size_t posible_m = 1; posible_m < machete_schedule_map_.size(); posible_m++) {
      machete_schedule_map_[posible_m] = GetMacheteBestSchedule(
          warmup_iters, record_iters, workspace_buffer_->GetPtr<void>(), context_->GetComputeStreams()[rank_].Get(),
          posible_m, posible_n, posible_k, buffer_input_activation.GetPtr<void>(), buffer_input_weight.GetPtr<void>(),
          buffer_output.GetPtr<void>(), vllm_dtype::ScalarType::from_id(activation_type_id),
          vllm_dtype::ScalarType::from_id(weight_type_id), buffer_input_scales.GetPtr<void>(),
          buffer_input_scales.shape, vllm_dtype::ScalarType::from_id(scale_type_id),
          is_awq_ ? std::optional<void*>(buffer_input_zeros.GetPtr<void>()) : std::nullopt,
          is_awq_ ? std::optional<std::vector<size_t>>(buffer_input_zeros.shape) : std::nullopt,
          is_awq_ ? std::optional<vllm_dtype::ScalarType>(vllm_dtype::ScalarType::from_id(zero_type_id)) : std::nullopt,
          groupsize_);

      KLLM_LOG_DEBUG << fmt::format("Generate best schedule for mnk=({},{},{}) is {}", posible_m, posible_n, posible_k,
                                    machete_schedule_map_[posible_m]);
    }

    Singleton<MacheteSearchStatus>::GetInstance()->AddMacheteSchedule(posible_n, posible_k, machete_schedule_map_);

    auto end_profile_time = std::chrono::high_resolution_clock::now();
    auto duration_profile_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_profile_time - start_profile_time);
    KLLM_LOG_INFO << fmt::format(
        "Profile MacheteMatMul Layer in rank:{}, mnk=({}~{},{},{}), warmup:{}, record:{}, cost:{}ms", rank_, 1,
        runtime_config.max_batch_size, posible_n, posible_k, warmup_iters, record_iters, duration_profile_ms.count());

    return Status();
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. MacheteMatMul only supports TYPE_I4_GROUP.",
                           weight_data_type_));
  }
}

Status MacheteMatMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (weight_data_type_ == TYPE_I4_GROUP) {
    // 根据模板类型确定各类型
    vllm_dtype::ScalarType::Id activation_type_id;
    if (inter_data_type_ == DataType::TYPE_FP16) {
      activation_type_id = vllm_dtype::kHalf.id();
    } else if (inter_data_type_ == DataType::TYPE_BF16) {
      activation_type_id = vllm_dtype::kBFloat16.id();
    } else {
      KLLM_THROW("MacheteMatMul only supports half and bfloat16 activation types");
    }
    vllm_dtype::ScalarType::Id weight_type_id = is_awq_ ? vllm_dtype::kU4.id() : vllm_dtype::kU4B8.id();
    vllm_dtype::ScalarType::Id scale_type_id = activation_type_id;
    vllm_dtype::ScalarType::Id zero_type_id = activation_type_id;

    const Tensor& weight_tensor = input_tensors[1];
    void* p_qweight_tensor = weight_tensor.GetPtr<void>();
    void* p_scales_tensor = weight_tensor.scales->GetPtr<void>();
    void* p_zeros_tensor = is_awq_ ? weight_tensor.zeros->GetPtr<void>() : nullptr;

    const size_t m = input_tensors[0].shape[0];
    const size_t n = max_n_;
    const size_t k = max_k_;

    std::optional<std::string> best_schedule = std::nullopt;
    if (m < machete_schedule_map_.size()) {
      best_schedule = std::optional<std::string>(machete_schedule_map_[m]);
    }

    int64_t workspace_size = 0;  // 不等于-1即可
    InvokeMacheteGemm(
        workspace_size, workspace_buffer_->GetPtr<void>(), context_->GetComputeStreams()[rank_].Get(), m, n, k,
        input_tensors[0].GetPtr<void>(), p_qweight_tensor, output_tensors[0].GetPtr<void>(),
        vllm_dtype::ScalarType::from_id(activation_type_id), vllm_dtype::ScalarType::from_id(weight_type_id),
        p_scales_tensor, weight_tensor.scales->shape, vllm_dtype::ScalarType::from_id(scale_type_id),
        is_awq_ ? std::optional<void*>(p_zeros_tensor) : std::nullopt,
        is_awq_ ? std::optional<std::vector<size_t>>(weight_tensor.zeros->shape) : std::nullopt,
        is_awq_ ? std::optional<vllm_dtype::ScalarType>(vllm_dtype::ScalarType::from_id(zero_type_id)) : std::nullopt,
        groupsize_, best_schedule);

    output_tensors[0].shape = {m, n};
    output_tensors[0].dtype = input_tensors[0].dtype;
    return Status();
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. MacheteMatMul only supports TYPE_I4_GROUP.",
                           weight_data_type_));
  }
}

}  // namespace ksana_llm
