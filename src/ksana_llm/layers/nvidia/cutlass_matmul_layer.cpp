/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/cutlass_matmul_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/search_status.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T, DataType WT>
Status CutlassMatMulLayer<T, WT>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                       int rank) {
  context_ = context;
  rank_ = rank;

  int parameter_index = 0;
  max_m_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_n_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_k_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  groupsize_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  is_awq_ = std::any_cast<const bool>(parameters[parameter_index++]);
  [[maybe_unused]] bool is_gptq_desc_ = std::any_cast<const bool>(parameters[parameter_index++]);  // unused
  [[maybe_unused]] bool is_k_full_ = std::any_cast<const bool>(parameters[parameter_index++]);     // unused
  cutlass_use_gemv_cuda_core_ = std::any_cast<const bool>(parameters[parameter_index++]);

  // double check some parameter
  if (cutlass_use_gemv_cuda_core_) {
    cutlass_use_gemv_cuda_core_ = GetFpAIntBGroupCudaGemmSupported<T, llm_kernels::nvidia::WeightType::INT4>();
  }
  return Status();
}

template <typename T, DataType WT>
size_t CutlassMatMulLayer<T, WT>::GetWorkSpaceSize() {
  if constexpr (WT == TYPE_I4_GROUP) {
    static std::mutex g_mtx;
    std::lock_guard<std::mutex> guard(g_mtx);
    // 检查是否可以跳过
    if (Singleton<CutlassSearchStatus>::GetInstance()->IsCutlassWorkspaceContain(max_m_, max_n_, max_k_)) {
      KLLM_LOG_DEBUG << fmt::format("Reusing CutlassMatMul Layer Workspace in rank:{}, mnk=({},{},{})", rank_, max_m_,
                                    max_n_, max_k_);
      return Singleton<CutlassSearchStatus>::GetInstance()->GetCutlassWorkspace(max_m_, max_n_, max_k_);
    }
    size_t max_ws_bytes = 0;

    GetFpAIntBGroupCutlassGemmWorkspaceSize<T, llm_kernels::nvidia::WeightType::INT4>(max_m_, max_n_, max_k_,
                                                                                      max_ws_bytes);

    Singleton<CutlassSearchStatus>::GetInstance()->AddCutlassWorkspace(max_m_, max_n_, max_k_, max_ws_bytes);

    KLLM_LOG_INFO << fmt::format("Rank[{}] Request {} for CutlassMatMulLayer", rank_, max_ws_bytes);
    return max_ws_bytes;
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. CutlassMatMul only supports TYPE_I4_GROUP.", WT));
  }
}

template <typename T, DataType WT>
Status CutlassMatMulLayer<T, WT>::Preprocess(const ModelConfig& model_config_) {
  const size_t record_iters = GetEnvAsPositiveInt("QUANT_PROFILE", 5);
  if (record_iters == 0) {
    KLLM_LOG_DEBUG << "$QUANT_PROFILE==0, Skipping CutlassMatMulLayer Preprocess";
    return Status();
  }
  const size_t warmup_iters = std::max(1UL, record_iters / 2);  // warmup不能为0

  if constexpr (WT == TYPE_I4_GROUP) {
    const size_t max_posible_m = model_config_.max_batch_size;
    const size_t posible_n = max_n_;
    const size_t posible_k = max_k_;

    static std::mutex g_mtx;
    std::lock_guard<std::mutex> guard(g_mtx);
    // 检查是否可以跳过
    if (Singleton<CutlassSearchStatus>::GetInstance()->IsCutlassScheduleContain(posible_n, posible_k)) {
      cutlass_config_map_ = Singleton<CutlassSearchStatus>::GetInstance()->GetCutlassSchedule(posible_n, posible_k);
      KLLM_LOG_DEBUG << fmt::format("Reusing Profile CutlassMatMul Layer in rank:{}, mnk=({}~{},{},{})", rank_, 1,
                                    max_posible_m, posible_n, posible_k);
      return Status();
    }
    cutlass_config_map_.resize(max_posible_m + 1);  // start from 1 but not 0

    Tensor buffer_input_activation(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP16, {max_posible_m, posible_k},
                                   rank_);
    Tensor buffer_input_weight(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_UINT8, {posible_k, posible_n / 2},
                               rank_);
    Tensor buffer_input_scales(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP16,
                               {posible_k / groupsize_, posible_n}, rank_);
    Tensor buffer_input_zeros(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP16, {posible_k / groupsize_, posible_n},
                              rank_);
    Tensor buffer_output(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP16, {max_posible_m, posible_n}, rank_);
    void* zeros_ptr = buffer_input_zeros.GetPtr<void>();
    if (model_config_.quant_config.method == QUANT_GPTQ) {
      zeros_ptr = nullptr;
    }

    // 开始搜索
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t posible_m = 1; posible_m < cutlass_config_map_.size(); posible_m++) {
      cutlass_config_map_[posible_m] =
          InvokeFpAIntBGroupCutlassGemmConfigProfile<T, llm_kernels::nvidia::WeightType::INT4>(
              warmup_iters, record_iters, buffer_output.GetPtr<void>(), buffer_input_activation.GetPtr<void>(),
              buffer_input_weight.GetPtr<void>(), buffer_input_scales.GetPtr<void>(), zeros_ptr,
              workspace_buffer_->GetPtr<void>(), posible_m, posible_n, posible_k, groupsize_,
              context_->GetComputeStreams()[rank_].Get());

      KLLM_LOG_DEBUG << fmt::format("Generate best config index for mnk=({},{},{}) is {}", posible_m, posible_n,
                                    posible_k, cutlass_config_map_[posible_m]);
    }

    Singleton<CutlassSearchStatus>::GetInstance()->AddCutlassSchedule(posible_n, posible_k, cutlass_config_map_);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    KLLM_LOG_INFO << fmt::format(
        "Profile CutlassMatMul Layer in rank:{}, mnk=({}~{},{},{}), warmup:{}, record:{}, cost:{}ms", rank_, 1,
        model_config_.max_batch_size, posible_n, posible_k, warmup_iters, record_iters, duration_ms.count());

    return Status();
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. CutlassMatMul only supports TYPE_I4_GROUP.", WT));
  }
}

template <typename T, DataType WT>
Status CutlassMatMulLayer<T, WT>::Forward(const std::vector<Tensor>& input_tensors,
                                          std::vector<Tensor>& output_tensors) {
  if constexpr (WT == TYPE_I4_GROUP) {
    const Tensor& weight_tensor = input_tensors[1];
    void* p_qweight_tensor = weight_tensor.GetPtr<void>();
    void* p_scales_tensor = weight_tensor.scales->GetPtr<void>();
    void* p_zeros_tensor = is_awq_ ? weight_tensor.zeros->GetPtr<void>() : nullptr;

    const size_t m = input_tensors[0].shape[0];
    const size_t n = max_n_;
    const size_t k = max_k_;

    if (cutlass_use_gemv_cuda_core_ && m < 5) {
      InvokeFpAIntBGroupCudaGemm<T, llm_kernels::nvidia::WeightType::INT4>(
          output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), p_qweight_tensor, p_scales_tensor,
          p_zeros_tensor, m, n, k, groupsize_, context_->GetComputeStreams()[rank_].Get());
    } else {
      size_t best_config_index = 0;
      if (m < cutlass_config_map_.size()) {
        best_config_index = cutlass_config_map_[m];
      }
      InvokeFpAIntBGroupCutlassGemm<T, llm_kernels::nvidia::WeightType::INT4>(
          output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), p_qweight_tensor, p_scales_tensor,
          p_zeros_tensor, workspace_buffer_->GetPtr<void>(), m, n, k, groupsize_, best_config_index,
          context_->GetComputeStreams()[rank_].Get());
    }

    output_tensors[0].shape = {m, n};
    output_tensors[0].dtype = input_tensors[0].dtype;
    return Status();
  } else {
    KLLM_THROW(fmt::format("Not supported weight data type: {}. CutlassMatMul only supports TYPE_I4_GROUP.", WT));
  }
}

template class CutlassMatMulLayer<float, TYPE_I4_GROUP>;
template class CutlassMatMulLayer<half, TYPE_I4_GROUP>;
template class CutlassMatMulLayer<__nv_bfloat16, TYPE_I4_GROUP>;

}  // namespace ksana_llm