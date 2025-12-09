/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_FP8
#  include "ksana_llm/layers/blockwise_matmul_layer.h"

#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#  include "ksana_llm/profiler/timer.h"
#  include "ksana_llm/utils/search_status.h"
#  include "ksana_llm/utils/singleton.h"
#  include "ksana_llm/utils/utils.h"

namespace ksana_llm {

Status BlockwiseMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                  std::shared_ptr<Context> context, int rank) {
  STATUS_CHECK_FAILURE(BaseLayer::Init(parameters, runtime_config, context, rank));

  size_t parameter_index = 0;
  max_m_ = std::any_cast<size_t>(parameters[parameter_index++]);
  n_ = std::any_cast<size_t>(parameters[parameter_index++]);
  k_ = std::any_cast<size_t>(parameters[parameter_index++]);
  block_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  weight_ = std::any_cast<Tensor>(parameters[parameter_index++]);
  if (parameter_index < parameters.size()) {
    skip_quant_ = std::any_cast<const bool>(parameters[parameter_index++]);
  }

  // currently, DeepGEMM only support bfloat16
  deepgemm_enabled_ = ((inter_data_type_ == DataType::TYPE_BF16) && std::getenv("DISABLE_DEEPGEMM") == nullptr);
  if (deepgemm_enabled_) {
    if (max_m_ % kAlignSize_ != 0) {
      KLLM_THROW(fmt::format("max_m {} is not aligned to {}, please set it to a multiple of {}", max_m_, kAlignSize_,
                             kAlignSize_));
    }
    deepgemm_wrapper_ = std::make_shared<llm_kernels::nvidia::DeepGEMMWrapper>(rank);
  }

  return Status();
}

size_t BlockwiseMatMulLayer::GetWorkspaceSize() { DISPATCH_BY_3_DTYPE(inter_data_type_, GetWorkspaceSizeT); }

template <typename T>
size_t BlockwiseMatMulLayer::GetCachedCutlassBufferSize() {
  // Check if buffer size is already cached
  if (Singleton<BlockwiseMatmulSearchStatus>::GetInstance()->IsCutlassBufferSizeContain(inter_data_type_, max_m_, k_,
                                                                                        n_)) {
    size_t cached_size =
        Singleton<BlockwiseMatmulSearchStatus>::GetInstance()->GetCutlassBufferSize(inter_data_type_, max_m_, k_, n_);
    KLLM_LOG_DEBUG << fmt::format("Rank[{}] Using cached cutlass buffer size: {}", rank_, cached_size);
    return cached_size;
  }

  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Computing cutlass buffer size for max_m={}, k={}, n={}", rank_, max_m_, k_,
                                n_);
  size_t cutlass_buffer_size = 0;
  for (size_t m = 1; m <= max_m_; m++) {
    cutlass_buffer_size = std::max(cutlass_buffer_size, InvokeGetBlockGemmWorkspaceSize<T>(m, k_, n_));
  }

  // Cache the computed buffer size
  Singleton<BlockwiseMatmulSearchStatus>::GetInstance()->AddCutlassBufferSize(inter_data_type_, max_m_, k_, n_,
                                                                              cutlass_buffer_size);
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Cached cutlass buffer size: {}", rank_, cutlass_buffer_size);

  return cutlass_buffer_size;
}

Status BlockwiseMatMulLayer::Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) {
  if (!deepgemm_enabled_) {
    SetGemmType(BlockwiseMatMulLayer::Fp8GemmType::Cutlass);
    return Status();
  }

  const auto start_time = ProfileTimer::GetCurrentTimeInMs();

  static std::mutex g_mtx;
  std::lock_guard<std::mutex> guard(g_mtx);

  // Check if GEMM selection thresholds are already cached
  if (Singleton<BlockwiseMatmulSearchStatus>::GetInstance()->IsGemmSelectionThresholdContain(inter_data_type_, max_m_,
                                                                                             k_, n_)) {
    auto [deepgemm_threshold, swap_ab_thresholds] =
        Singleton<BlockwiseMatmulSearchStatus>::GetInstance()->GetGemmSelectionThreshold(inter_data_type_, max_m_, k_,
                                                                                         n_);
    deepgemm_max_m_threshold_ = deepgemm_threshold;
    swap_ab_max_m_thresholds_ = std::move(swap_ab_thresholds);
    KLLM_LOG_DEBUG << fmt::format("Reusing Profile BlockwiseMatMulLayer in rank:{}, ({},{},{},{})", rank_,
                                  inter_data_type_, max_m_, k_, n_);
  } else {
    const size_t kPerfIters = GetEnvAsPositiveInt("QUANT_PROFILE", 20);
    if (kPerfIters == 0) {
      return Status();
    }
    const size_t kWarmupIters = std::max(1UL, kPerfIters / 2);
    // compute the threshold
    const size_t kSmallStep = 8;
    const size_t kUseSmallStepThreshold = 2048;

    // generate random input and output
    auto input = torch::randn({static_cast<int64_t>(max_m_), static_cast<int64_t>(k_)},
                              torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
    Tensor output_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {max_m_, n_}, rank_);
    // 自适应GEMM算子选择阈值搜索算法
    // 目标：通过性能测试确定三种GEMM实现（cutlass、deepgemm、deepgemm_swap_ab）的最优切换阈值
    //
    // 搜索策略：
    //   1. 从 m=kAlignSize 开始，逐步增大 m 值
    //   2. 对每个 m 值，分别测量三种算子的执行时间
    //   3. 当 deepgemm 和 deepgemm_swap_ab 均不优于 cutlass 时，终止搜索
    //   4. 否则更新 deepgemm_max_m_threshold_
    //   5. 当 deepgemm_swap_ab 性能优于 deepgemm 时，更新 swap_ab_max_m_threshold_
    //
    // 算法假设：
    //   1. m 越大，cutlass 相比 deepgemm 优势越大，因此当 m 大于
    //      deepgemm_max_m_threshold_ 时使用 cutlass
    //   2. deepgemm 使用的 wgmma 指令固定 m=kWgmmaBlockM_=64，导致执行时间按 64 呈阶梯式增长。
    //      在每个 64 的分段内，m 越大，deepgemm 相比 deepgemm_swap_ab 优势越大，因此当 m
    //      大于对应分段的 swap_ab_max_m_threshold_ 时使用 deepgemm，否则使用 deepgemm_swap_ab。
    //      为避免段数过多，当 m 大于 kUseSmallStepThreshold=2048 时，固定使用 deepgemm
    size_t m = kAlignSize_;
    while (m <= max_m_) {
      Tensor input_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {m, k_}, rank_,
                          static_cast<void*>(input.data_ptr<torch::BFloat16>()));
      std::vector<Tensor> input_tensors = {input_tensor, weight_};
      std::vector<Tensor> output_tensors = {output_tensor};

      auto cutlass_kernel = [&]() {
        SetGemmType(BlockwiseMatMulLayer::Fp8GemmType::Cutlass);
        this->Forward(input_tensors, output_tensors);
      };
      auto deepgemm_kernel = [&]() {
        SetGemmType(BlockwiseMatMulLayer::Fp8GemmType::DeepGemm);
        this->Forward(input_tensors, output_tensors);
      };
      auto deepgemm_swap_ab_kernel = [&]() {
        SetGemmType(BlockwiseMatMulLayer::Fp8GemmType::DeepGemmSwapAB);
        this->Forward(input_tensors, output_tensors);
      };

      float cutlass_time = MeasureCudaExecutionTime(cutlass_kernel, context_->GetComputeStreams()[rank_].Get(),
                                                    kWarmupIters, kPerfIters);
      float deepgemm_time = MeasureCudaExecutionTime(deepgemm_kernel, context_->GetComputeStreams()[rank_].Get(),
                                                     kWarmupIters, kPerfIters);
      float deepgemm_swap_ab_time = MeasureCudaExecutionTime(
          deepgemm_swap_ab_kernel, context_->GetComputeStreams()[rank_].Get(), kWarmupIters, kPerfIters);

      if (std::min(deepgemm_time, deepgemm_swap_ab_time) < cutlass_time) {
        deepgemm_max_m_threshold_ = m;
      } else {
        break;
      }
      if (m <= kUseSmallStepThreshold) {
        if ((m - kAlignSize_) / kWgmmaBlockM_ == swap_ab_max_m_thresholds_.size()) {
          swap_ab_max_m_thresholds_.push_back((m - kAlignSize_) / kWgmmaBlockM_ * kWgmmaBlockM_);
        }
        if (deepgemm_swap_ab_time < deepgemm_time) {
          swap_ab_max_m_thresholds_.back() = m;
        }
      }

      m += m < kUseSmallStepThreshold ? std::min(m, kSmallStep) : m;
    }

    // Cache the computed thresholds
    Singleton<BlockwiseMatmulSearchStatus>::GetInstance()->AddGemmSelectionThreshold(
        inter_data_type_, max_m_, k_, n_, deepgemm_max_m_threshold_, swap_ab_max_m_thresholds_);
    KLLM_LOG_INFO << fmt::format(
        "Set deepgemm_max_m_threshold_ to {} and swap_ab_max_m_thresholds_ to {} "
        "for max_m={}, k={}, n={}",
        deepgemm_max_m_threshold_, Vector2Str(swap_ab_max_m_thresholds_), max_m_, k_, n_);
  }

  BuildDeepGemmKernels();
  SetGemmType(BlockwiseMatMulLayer::Fp8GemmType::Dynamic);
  KLLM_LOG_INFO << fmt::format("Rank[{}] BlockwiseMatMulLayer Preprocess cost time: {} ms", rank_,
                               ProfileTimer::GetCurrentTimeInMs() - start_time);
  return Status();
}

void BlockwiseMatMulLayer::BuildDeepGemmKernels() {
  if (!deepgemm_enabled_) {
    return;
  }

  const auto start_time = ProfileTimer::GetCurrentTimeInMs();

  // Build kernel
  static std::mutex g_mtx;
  std::lock_guard<std::mutex> guard(g_mtx);

  static_assert(kWgmmaBlockM_ % kAlignSize_ == 0);
  for (size_t begin_m = kAlignSize_; begin_m <= deepgemm_max_m_threshold_; begin_m += kWgmmaBlockM_) {
    const size_t swap_ab_max_m_threshold = begin_m > swap_ab_max_m_thresholds_.back()
                                               ? begin_m - kAlignSize_
                                               : swap_ab_max_m_thresholds_[(begin_m - kAlignSize_) / kWgmmaBlockM_];
    // Build SwapAB kernel
    for (size_t m = begin_m; m <= swap_ab_max_m_threshold; m += kAlignSize_) {
      deepgemm_wrapper_->BuildGemmSwapABKernel(m, n_, k_);
    }
    // Build regular kernel
    for (size_t m = swap_ab_max_m_threshold + kAlignSize_;
         m < begin_m + kWgmmaBlockM_ && m <= deepgemm_max_m_threshold_; m += kAlignSize_) {
      deepgemm_wrapper_->BuildGemmKernel(m, n_, k_);
    }
  }

  KLLM_LOG_INFO << fmt::format("Rank[{}] BlockwiseMatMulLayer BuildDeepGemmKernels cost time: {} ms", rank_,
                               ProfileTimer::GetCurrentTimeInMs() - start_time);
}

void BlockwiseMatMulLayer::SetGemmType(BlockwiseMatMulLayer::Fp8GemmType gemm_type) { gemm_type_ = gemm_type; }

BlockwiseMatMulLayer::Fp8GemmType BlockwiseMatMulLayer::PickGemmType(size_t m) {
  if (gemm_type_ != BlockwiseMatMulLayer::Fp8GemmType::Dynamic) {
    return gemm_type_;
  }
  if (const size_t aligned_m = RoundUp(m, kAlignSize_); aligned_m > deepgemm_max_m_threshold_) {
    return BlockwiseMatMulLayer::Fp8GemmType::Cutlass;
  } else if (aligned_m > swap_ab_max_m_thresholds_.back() ||
             aligned_m > swap_ab_max_m_thresholds_[(aligned_m - kAlignSize_) / kWgmmaBlockM_]) {
    return BlockwiseMatMulLayer::Fp8GemmType::DeepGemm;
  } else {
    return BlockwiseMatMulLayer::Fp8GemmType::DeepGemmSwapAB;
  }
}

template <typename T>
size_t BlockwiseMatMulLayer::GetWorkspaceSizeT() {
  size_t input_size = max_m_ * k_ * GetTypeSize(TYPE_FP8_E4M3);
  size_t scale_size = max_m_ * DivRoundUp(k_, block_size_) * GetTypeSize(TYPE_FP32);
  input_buffer_size_ = input_size + scale_size;
  cutlass_gemm_workspace_size_ = GetCachedCutlassBufferSize<T>();
  size_t deepgemm_workspace_size = 0;
  workspace_size_ = input_size + scale_size + std::max(cutlass_gemm_workspace_size_, deepgemm_workspace_size);
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for BlockwiseMatMulLayer", rank_, workspace_size_);
  return workspace_size_;
}

Status BlockwiseMatMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status BlockwiseMatMulLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (workspace_size_ > workspace_buffer_->GetTotalBytes()) {
    KLLM_THROW(fmt::format("workspace size {} > buffer size {}", workspace_size_, workspace_buffer_->GetTotalBytes()));
  }
  const size_t m = input_tensors[0].shape[0];
  const size_t k = input_tensors[0].shape[1];
  // input_tensors[0].shape[1] is k_ (normal case) or 2*k_ (need to do silu mul first)
  // input_tensors[1].shape[0] is n_
  BlockwiseMatMulLayer::Fp8GemmType gemm_type = PickGemmType(m);
  size_t cur_m = m;
  if (gemm_type == BlockwiseMatMulLayer::Fp8GemmType::DeepGemmSwapAB ||
      gemm_type == BlockwiseMatMulLayer::Fp8GemmType::DeepGemm) {
    cur_m = RoundUp(m, kAlignSize_);
  }
  T* a = static_cast<T*>(input_tensors[0].GetPtr<void>());
  void* a_q = workspace_buffer_->GetPtr<void>();
  void* a_s = a_q + GetTypeSize(TYPE_FP8_E4M3) * cur_m * k_;

  if (!skip_quant_) {
    InvokePerTokenGroupQuantFp8E4m3<T>(a, a_q, a_s, cur_m, k_, /*is_column_major*/ true,
                                       context_->GetComputeStreams()[rank_].Get(), block_size_,
                                       PerTokenGroupQuantFusionParams{.fuse_silu_mul = (k == 2 * k_)});
  }

  void* b = input_tensors[1].GetPtr<void>();
  void* b_s = input_tensors[1].weight_scales->GetPtr<void>();

  void* out = output_tensors[0].GetPtr<void>();

  // Execute GEMM operation based on selected type
  cudaStream_t compute_stream = context_->GetComputeStreams()[rank_].Get();
  switch (gemm_type) {
    case BlockwiseMatMulLayer::Fp8GemmType::DeepGemmSwapAB:
      deepgemm_wrapper_->GemmSwapAB(a_q, a_s, b, b_s, out, cur_m, n_, k_, compute_stream);
      break;
    case BlockwiseMatMulLayer::Fp8GemmType::DeepGemm:
      deepgemm_wrapper_->Gemm(a_q, a_s, b, b_s, out, cur_m, n_, k_, compute_stream);
      break;
    case BlockwiseMatMulLayer::Fp8GemmType::Cutlass: {
      void* cutlass_buffer = workspace_buffer_->GetPtr<void>() + input_buffer_size_;
      InvokeBlockGemm<T>(a_q, static_cast<float*>(a_s), b, static_cast<float*>(b_s), out, cur_m, k_, n_, compute_stream,
                         cutlass_buffer, cutlass_gemm_workspace_size_);
      break;
    }
    default:
      KLLM_THROW(fmt::format("Invalid or unsupported gemm type: {}", static_cast<int>(gemm_type)));
      break;
  }

  output_tensors[0].shape = {m, n_};
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

}  // namespace ksana_llm
#endif
