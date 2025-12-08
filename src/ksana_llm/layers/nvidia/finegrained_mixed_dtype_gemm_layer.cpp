/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/finegrained_mixed_dtype_gemm_layer.h"

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/thop/utils.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/utils/search_status.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

Status FinegrainedMixedDtypeGemmLayer::Init(const std::vector<std::any>& parameters,
                                            const RuntimeConfig& runtime_config, std::shared_ptr<Context> context,
                                            int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  params_ = std::any_cast<const FinegrainedMixedDtypeGemmLayerParameters>(parameters.front());

  const auto quant_mode =
      params_.has_zero ? llm_kernels::nvidia::QuantMode::WITH_ZERO : llm_kernels::nvidia::QuantMode::NO_ZERO;
  wrapper_ = std::make_shared<llm_kernels::nvidia::FinegrainedMixedDtypeGemmWrapper>(
      DataTypeToScalarTypeMap.at(params_.activation_type), DataTypeToScalarTypeMap.at(params_.output_type), quant_mode,
      params_.group_size);

  KLLM_LOG_INFO << fmt::format("Rank[{}] FinegrainedMixedDtypeGemmLayer Init with {}", rank_, params_.ToString());
  return Status();
}

inline size_t AlignTo256(size_t size) { return (size + 255) & (~255); }

size_t FinegrainedMixedDtypeGemmLayer::GetWorkspaceSize() {
  workspace_info_.kernel_workspace_size_ = AlignTo256(GetKernelWorkspaceSize());
  workspace_info_.quantized_size_ = AlignTo256(params_.m * params_.k * sizeof(char));
  size_t workspace_size = workspace_info_.GetTotalSize();
  KLLM_LOG_INFO << fmt::format("Rank[{}] Request {} for FinegrainedMixedDtypeGemmLayer", rank_, workspace_size);
  return workspace_size;
}

size_t FinegrainedMixedDtypeGemmLayer::GetKernelWorkspaceSize() {
  static std::mutex g_mtx;
  std::lock_guard<std::mutex> guard(g_mtx);
  // 检索
  const std::string key = fmt::format("{}_{}", typeid(*this).name(), params_.ToString());
  auto cache = Singleton<std::unordered_map<std::string, int64_t>>::GetInstance();
  auto it = cache->find(key);
  if (it != cache->end()) {
    return it->second;
  }
  // 生成
  int64_t workspace_size = 0;
  for (size_t possible_m = 1; possible_m <= params_.m; possible_m++) {
    workspace_size =
        std::max(workspace_size, static_cast<int64_t>(wrapper_->GetWorkspaceSize(possible_m, params_.n, params_.k)));
  }
  // 存储
  (*cache)[key] = workspace_size;
  return workspace_size;
}

Status FinegrainedMixedDtypeGemmLayer::Preprocess(const ModelConfig& model_config,
                                                  const RuntimeConfig& runtime_config) {
  static std::mutex g_mtx;
  std::lock_guard<std::mutex> guard(g_mtx);
  // 获取搜索次数
  const size_t record_iters = GetEnvAsPositiveInt("QUANT_PROFILE", 10);
  if (record_iters == 0) {
    KLLM_LOG_DEBUG << "$QUANT_PROFILE==0, Skipping FinegrainedMixedDtypeGemmLayer Preprocess";
    return Status();
  }
  const size_t warmup_iters = std::max(1UL, record_iters / 2);  // warmup不能为0

  const auto start_time = ProfileTimer::GetCurrentTime();

  // 检索
  using tactic_type = std::vector<std::vector<int64_t>>;
  const std::string key = fmt::format("{}_{}", typeid(*this).name(), params_.ToString());
  auto cache = Singleton<std::unordered_map<std::string, tactic_type>>::GetInstance();
  auto it = cache->find(key);
  if (it != cache->end()) {
    tactics_ = it->second;
    return Status();
  }
  // 开辟空间
  float alpha = 1.1f;
  Tensor workspace = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {workspace_info_.workspace_size_}, rank_);
  Tensor input = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP8_E4M3, {params_.m, params_.k}, rank_);
  Tensor weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {params_.k, params_.n / 2}, rank_);
  Tensor weight_scale =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {params_.k / params_.group_size, params_.n}, rank_);
  Tensor output = Tensor(MemoryLocation::LOCATION_DEVICE, params_.output_type, {params_.m, params_.n}, rank_);
  KTensor weight_kt = TensorToKTensor(weight);
  KTensor weight_scale_kt = TensorToKTensor(weight_scale);
  // decode搜索
  std::vector<size_t> decode_m(runtime_config.max_batch_size);
  std::vector<int64_t> decode_tactic(runtime_config.max_batch_size + 1, -1);
  std::iota(decode_m.begin(), decode_m.end(), 1);  // 生成连续递增m
  for (const size_t& token : decode_m) {
    KTensor quantized_input_kt(input.GetPtr<void>(), {token, params_.k}, KScalarType::Float8_e4m3fn);
    KTensor output_kt(output.GetPtr<void>(), {token, params_.n}, DataTypeToScalarTypeMap.at(params_.output_type));
    decode_tactic[token] = wrapper_->Profile(context_->GetComputeStreams()[rank_].Get(), output_kt,
                                             workspace.GetPtr<void>(), quantized_input_kt, weight_kt, weight_scale_kt,
                                             std::nullopt, std::nullopt, alpha, warmup_iters, record_iters);
  }
  // prefill搜索
  std::vector<int64_t> prefill_m = [](int64_t start, int64_t end) {
    std::vector<int64_t> vec;
    for (int64_t current = start; current <= end; current += (current < 1024 ? 128 : 1024)) {
      vec.push_back(current);
    }
    return vec;
  }(runtime_config.max_batch_size, params_.m);
  std::vector<int64_t> prefill_tactic(prefill_m.size(), -1);
  for (size_t idx = 0; idx < prefill_m.size(); idx++) {
    const size_t token = prefill_m[idx];
    KTensor quantized_input_kt(input.GetPtr<void>(), {token, params_.k}, KScalarType::Float8_e4m3fn);
    KTensor output_kt(output.GetPtr<void>(), {token, params_.n}, DataTypeToScalarTypeMap.at(params_.output_type));
    prefill_tactic[idx] = wrapper_->Profile(context_->GetComputeStreams()[rank_].Get(), output_kt,
                                            workspace.GetPtr<void>(), quantized_input_kt, weight_kt, weight_scale_kt,
                                            std::nullopt, std::nullopt, alpha, warmup_iters, record_iters);
  }

  // 存储
  tactics_ = {decode_tactic, prefill_m, prefill_tactic};
  (*cache)[key] = tactics_;

  KLLM_LOG_INFO << fmt::format("Rank[{}] FinegrainedMixedDtypeGemmLayer Preprocess cost time: {} s", rank_,
                               ProfileTimer::GetCurrentTime() - start_time);
  return Status();
}

int64_t FinegrainedMixedDtypeGemmLayer::GetBestTactic(const size_t& m) {
  if (tactics_.empty()) {
    return -1;
  } else {
    const std::vector<int64_t>& decode_tactic = tactics_[0];
    const std::vector<int64_t>& prefill_m = tactics_[1];
    const std::vector<int64_t>& prefill_tactic = tactics_[2];
    if (m < decode_tactic.size()) {
      // decode情况直接获取
      return decode_tactic[m];
    } else {
      // prefill情况需要搜索
      auto it = std::upper_bound(prefill_m.begin(), prefill_m.end(), m);
      size_t idx = (it == prefill_m.begin()) ? 0 : std::distance(prefill_m.begin(), it) - 1;
      return prefill_tactic[idx];
    }
  }
}

Status FinegrainedMixedDtypeGemmLayer::Forward(const std::vector<Tensor>& input_tensors,
                                               std::vector<Tensor>& output_tensors) {
  workspace_info_.SetPtr(workspace_buffer_->GetPtr<void>());

  const size_t m = input_tensors[0].shape[0];
  const size_t n = 2 * input_tensors[1].shape[1];
  const size_t k = input_tensors[0].shape[1];

  output_tensors[0].shape = {m, n};
  output_tensors[0].dtype = input_tensors[0].dtype;

  KTensor input_kt = TensorToKTensor(input_tensors[0]);
  KTensor output_kt = TensorToKTensor(output_tensors[0]);
  KTensor weight_kt = TensorToKTensor(input_tensors[1]);
  KTensor input_scale_kt = TensorToKTensor(*(input_tensors[1].input_scales));
  KTensor weight_scale_kt = TensorToKTensor(*(input_tensors[1].weight_scales));
  const float alpha = input_tensors[1].alpha;
  KTensor quantized_input_kt(workspace_info_.quantized_input_ptr_, {m, k}, KScalarType::Float8_e4m3fn);

  if (input_tensors[1].pre_quant_scales) {
    KTensor pre_quant_scale_kt = TensorToKTensor(*(input_tensors[1].pre_quant_scales));
    wrapper_->StaticQuantizeE4M3PerChannel(quantized_input_kt, input_kt, pre_quant_scale_kt,
                                           context_->GetComputeStreams()[rank_].Get());
  } else {
    wrapper_->StaticQuantizeE4M3PerTensor(quantized_input_kt, input_kt, input_scale_kt,
                                          context_->GetComputeStreams()[rank_].Get());
  }
  const int64_t best_tactic = GetBestTactic(m);
  wrapper_->Forward(context_->GetComputeStreams()[rank_].Get(), output_kt, workspace_info_.kernel_workspace_ptr_,
                    quantized_input_kt, weight_kt, weight_scale_kt, best_tactic, std::nullopt, std::nullopt, alpha);
  return Status();
}

}  // namespace ksana_llm
