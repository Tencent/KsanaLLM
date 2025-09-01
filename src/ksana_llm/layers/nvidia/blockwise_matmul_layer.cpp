/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_FP8
#  include "ksana_llm/layers/blockwise_matmul_layer.h"
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#  include "ksana_llm/profiler/timer.h"
#  include "ksana_llm/runtime/threadpool.h"
#  include "ksana_llm/utils/utils.h"

namespace ksana_llm {

Status BlockwiseMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                  std::shared_ptr<Context> context, int rank) {
  STATUS_CHECK_FAILURE(BaseLayer::Init(parameters, runtime_config, context, rank));

  int parameter_index = 0;
  max_m_ = std::any_cast<size_t>(parameters[parameter_index++]);
  n_ = std::any_cast<size_t>(parameters[parameter_index++]);
  k_ = std::any_cast<size_t>(parameters[parameter_index++]);
  block_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);

  // currently, DeepGEMM only support bfloat16
  if ((inter_data_type_ == DataType::TYPE_BF16) && std::getenv("DISABLE_DEEPGEMM") == nullptr) {
    kDeepGemmMaxMThreshold_ = GetEnvAsPositiveInt("DEEPGEMM_MAX_M_THRESHOLD", 256);
    const size_t align_m = 4;
    if (max_m_ % align_m != 0) {
      KLLM_THROW(
          fmt::format("max_m {} is not aligned to {}, please set it to a multiple of {}", max_m_, align_m, align_m));
    }
    if (kDeepGemmMaxMThreshold_ % align_m != 0) {
      KLLM_THROW(fmt::format("DEEPGEMM_MAX_M_THRESHOLD {} is not aligned to {}, please set it to a multiple of {}",
                             kDeepGemmMaxMThreshold_, align_m, align_m));
    }

    std::vector<std::any> deepgemm_matmul_params;
    deepgemm_matmul_params.push_back(kDeepGemmMaxMThreshold_);
    deepgemm_matmul_params.push_back(n_);
    deepgemm_matmul_params.push_back(k_);
    deepgemm_matmul_params.push_back(block_size_);
    deepgemm_matmul_layer_.Init(deepgemm_matmul_params, runtime_config, context, rank);

    KLLM_LOG_DEBUG << fmt::format("Rank[{}] DeepGemmMatMulLayer Init", rank_);
  }

  return Status();
}

size_t BlockwiseMatMulLayer::GetWorkSpaceSize() {
  size_t input_size = max_m_ * k_ * GetTypeSize(TYPE_FP8_E4M3);
  size_t scale_size = max_m_ * DivRoundUp(k_, block_size_) * GetTypeSize(TYPE_FP32);
  size_t cutlass_buffer_size = max_m_ * k_ * GetTypeSize(TYPE_FP8_E4M3);
  workspace_size_ = input_size + scale_size + cutlass_buffer_size;
  if (kDeepGemmMaxMThreshold_ > 0) {
    workspace_size_ = std::max(workspace_size_, deepgemm_matmul_layer_.GetWorkSpaceSize());
  }
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for BlockwiseMatMulLayer", rank_, workspace_size_);
  return workspace_size_;
}

Status BlockwiseMatMulLayer::SetWorkSpaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) {
  workspace_buffer_ = workspace_buffer;
  if (kDeepGemmMaxMThreshold_ > 0) {
    deepgemm_matmul_layer_.SetWorkSpaceBuffer(workspace_buffer);
  }
  return Status();
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
  // TODO(jinxcwu) 要设计一种dispatch逻辑
  if (m <= kDeepGemmMaxMThreshold_) {
    deepgemm_matmul_layer_.Forward(input_tensors, output_tensors);
  } else {
    const size_t k = input_tensors[0].shape[1];
    // input_tensors[0].shape[1] is k_ (normal case) or 2*k_ (need to do silu mul first)
    // input_tensors[1].shape[0] is n_

    T* a = static_cast<T*>(input_tensors[0].GetPtr<void>());
    void* a_q = workspace_buffer_->GetPtr<void>();
    float* a_s = static_cast<float*>(a_q + GetTypeSize(TYPE_FP8_E4M3) * m * k_);
    void* cutlass_buffer = a_s + m * DivRoundUp(k_, block_size_) * GetTypeSize(TYPE_FP32);
    size_t cutlass_buffer_size = m * k_ * GetTypeSize(TYPE_FP8_E4M3);

    InvokePerTokenGroupQuantFp8E4m3<T>(a, a_q, a_s, m, k_, /*is_column_major*/ true,
                                       context_->GetComputeStreams()[rank_].Get(), block_size_,
                                       PerTokenGroupQuantFusionParams{.fuse_silu_mul = (k == 2 * k_)});

    void* b = input_tensors[1].GetPtr<void>();
    float* b_scale = static_cast<float*>(input_tensors[1].weight_scales->GetPtr<void>());

    T* output = static_cast<T*>(output_tensors[0].GetPtr<void>());
    InvokeBlockGemm<T>(a_q, a_s, b, b_scale, output, m, k_, n_, context_->GetComputeStreams()[rank_].Get(),
                       cutlass_buffer, cutlass_buffer_size);

    output_tensors[0].shape = {m, n_};
    output_tensors[0].dtype = input_tensors[0].dtype;
  }
  return Status();
}

}  // namespace ksana_llm
#endif
