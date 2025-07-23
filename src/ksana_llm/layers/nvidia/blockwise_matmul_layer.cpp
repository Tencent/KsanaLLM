/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_FP8
#  include "ksana_llm/layers/blockwise_matmul_layer.h"

#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

#  include "ksana_llm/profiler/timer.h"
#  include "ksana_llm/runtime/threadpool.h"

namespace ksana_llm {

template <typename T>
Status BlockwiseMatMulLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                     std::shared_ptr<Context> context, int rank) {
  STATUS_CHECK_FAILURE(BaseLayer::Init(parameters, runtime_config, context, rank));
  int parameter_index = 0;
  max_m_ = std::any_cast<size_t>(parameters[parameter_index++]);
  n_ = std::any_cast<size_t>(parameters[parameter_index++]);
  k_ = std::any_cast<size_t>(parameters[parameter_index++]);
  block_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  size_t tp_size = std::any_cast<size_t>(parameters[parameter_index++]);

  // currently, DeepGEMM only support bfloat16
  if (std::is_same<T, __nv_bfloat16>::value && std::getenv("DISABLE_DEEPGEMM") == nullptr) {
    if (std::getenv("DEEPGEMM_MAX_M_THRESHOLD") != nullptr) {
      kDeepGemmMaxMThreshold_ = std::stoi(std::getenv("DEEPGEMM_MAX_M_THRESHOLD"));
    } else {
      kDeepGemmMaxMThreshold_ = 256;  // default value
    }
    const size_t align_m = 4;
    if (max_m_ % align_m != 0) {
      KLLM_THROW(
          fmt::format("max_m {} is not aligned to {}, please set it to a multiple of {}", max_m_, align_m, align_m));
    }
    if (kDeepGemmMaxMThreshold_ % align_m != 0) {
      KLLM_THROW(fmt::format("DEEPGEMM_MAX_M_THRESHOLD {} is not aligned to {}, please set it to a multiple of {}",
                             kDeepGemmMaxMThreshold_, align_m, align_m));
    }
    const size_t tuned_m_step = 64;
    std::unordered_set<size_t> m_set = {kDeepGemmMaxMThreshold_};
    for (size_t cur_m = tuned_m_step; cur_m <= kDeepGemmMaxMThreshold_; cur_m += tuned_m_step) {
      m_set.insert(cur_m);
    }
    const auto start_time = ProfileTimer::GetCurrentTimeInMs();
    int thread_num = 1;  // default thread number, too high may cause OOM
    if (std::getenv("DEEPGEMM_TUNER_THREAD_NUM") != nullptr) {
      thread_num = std::stoi(std::getenv("DEEPGEMM_TUNER_THREAD_NUM"));
    }
    std::shared_ptr<ThreadPool> tuner_threadpool_ = std::make_shared<ThreadPool>(thread_num * tp_size);
    tuner_threadpool_->Start();
    std::vector<std::future<void>> tune_tasks;
    std::mutex map_mutex;
    size_t thread_id = 0;
    for (size_t cur_m : m_set) {
      tune_tasks.push_back(tuner_threadpool_->Submit([cur_m, context, tp_size, thread_id, &map_mutex, this]() {
        try {
          std::unique_ptr<llm_kernels::nvidia::DeepGEMMAOTWrapper> deepgemm_aot_wrapper =
              std::make_unique<llm_kernels::nvidia::DeepGEMMAOTWrapper>(
                  cur_m, n_, k_, /*need_generate_kernel*/ rank_ == 0, /*tuner_device_id*/ thread_id % tp_size);

          std::lock_guard<std::mutex> lock(map_mutex);
          m_to_deepgemm_aot_wrapper_[cur_m] = std::move(deepgemm_aot_wrapper);
        } catch (const std::exception& e) {
          KLLM_THROW(fmt::format("Failed to initialize DeepGEMMAOTWrapper for m: {}, n: {}, k: {}. Error: {}", cur_m,
                                 n_, k_, e.what()));
        }
      }));
      thread_id++;
    }
    for (auto&& tune_task : tune_tasks) {
      tune_task.get();
    }
    tuner_threadpool_->Stop();

    KLLM_LOG_DEBUG << fmt::format("Rank[{}] DeepGemmMatMulLayer Init cost time: {} ms", rank_,
                                  ProfileTimer::GetCurrentTimeInMs() - start_time);
  }

  return Status();
}

template <typename T>
size_t BlockwiseMatMulLayer<T>::GetWorkSpaceSize() {
  size_t input_size = max_m_ * k_ * GetTypeSize(TYPE_FP8_E4M3);
  size_t scale_size = max_m_ * DivRoundUp(k_, block_size_) * GetTypeSize(TYPE_FP32);
  size_t cutlass_buffer_size = max_m_ * k_ * GetTypeSize(TYPE_FP8_E4M3);
  workspace_size_ = input_size + scale_size + cutlass_buffer_size;
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for BlockwiseMatMulLayer", rank_, workspace_size_);
  return workspace_size_;
}

template <typename T>
Status BlockwiseMatMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  int m = input_tensors[0].shape[0];
  int k = input_tensors[0].shape[1];
  int n = input_tensors[1].shape[0];
  if (workspace_size_ > workspace_buffer_->GetTotalBytes()) {
    KLLM_THROW(fmt::format("workspace size {} > buffer size {}", workspace_size_, workspace_buffer_->GetTotalBytes()));
  }
  if (m <= kDeepGemmMaxMThreshold_) {
    const size_t align_m = 4;
    int aligned_m = std::ceil(static_cast<float>(m) / align_m) * align_m;
    T* a = static_cast<T*>(input_tensors[0].GetPtr<void>());
    void* a_q = workspace_buffer_->GetPtr<void>();
    void* a_s = a_q + GetTypeSize(TYPE_FP8_E4M3) * aligned_m * k;

    InvokePerTokenGroupQuantFp8E4m3<T>(a, a_q, a_s, aligned_m, k, true, context_->GetComputeStreams()[rank_].Get(),
                                       block_size_);

    void* b = input_tensors[1].GetPtr<void>();
    void* b_scale = input_tensors[1].weight_scales->GetPtr<void>();

    void* output = output_tensors[0].GetPtr<void>();
    output_tensors[0].shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
    output_tensors[0].dtype = input_tensors[0].dtype;

    auto it = m_to_deepgemm_aot_wrapper_.lower_bound(aligned_m);
    if (it == m_to_deepgemm_aot_wrapper_.end()) {
      KLLM_THROW(fmt::format("No DeepGEMMAOTWrapper found for aligned_m: {}", aligned_m));
    }
    auto& deepgemm_aot_wrapper = it->second;
    deepgemm_aot_wrapper->Forward(a_q, a_s, b, b_scale, output, aligned_m, context_->GetComputeStreams()[rank_].Get());
  } else {
    T* a = static_cast<T*>(input_tensors[0].GetPtr<void>());
    void* a_q = workspace_buffer_->GetPtr<void>();
    float* a_s = static_cast<float*>(a_q + GetTypeSize(TYPE_FP8_E4M3) * m * k);
    void* cutlass_buffer = a_s + m * DivRoundUp(k, block_size_) * GetTypeSize(TYPE_FP32);
    size_t cutlass_buffer_size = m * k * GetTypeSize(TYPE_FP8_E4M3);

    InvokePerTokenGroupQuantFp8E4m3<T>(a, a_q, a_s, m, k, true, context_->GetComputeStreams()[rank_].Get(),
                                       block_size_);

    void* b = input_tensors[1].GetPtr<void>();
    float* b_scale = static_cast<float*>(input_tensors[1].weight_scales->GetPtr<void>());

    T* output = static_cast<T*>(output_tensors[0].GetPtr<void>());
    output_tensors[0].shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
    output_tensors[0].dtype = input_tensors[0].dtype;
    InvokeBlockGemm<T>(a_q, a_s, b, b_scale, output, m, k, n, context_->GetComputeStreams()[rank_].Get(),
                       cutlass_buffer, cutlass_buffer_size);
  }
  return Status();
}

template class BlockwiseMatMulLayer<float>;
template class BlockwiseMatMulLayer<half>;
template class BlockwiseMatMulLayer<__nv_bfloat16>;

}  // namespace ksana_llm
#endif
