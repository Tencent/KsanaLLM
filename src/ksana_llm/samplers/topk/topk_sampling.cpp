/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/samplers/topk/topk_sampling.h"

#include <cstdint>
#include <random>

#include "ksana_llm/kernels/argmax.h"
#include "ksana_llm/utils/logger.h"
#ifdef ENABLE_CUDA
#  include "3rdparty/LLM_kernels/csrc/kernels/nvidia/samplers/sampling_topk_kernels.h"
#endif

namespace ksana_llm {
TopkSampling::TopkSampling(size_t max_batch_size, size_t max_vocab_size, RandState* device_curandstates)
    : BaseSampling(max_batch_size, max_vocab_size) {
#ifdef ENABLE_CUDA
  float* logits = nullptr;
  CUDA_CHECK_LAST_ERROR(tensorrt_llm::kernels::invoke_batch_topk_sampling(
      nullptr, workspace_size_, logits, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1024, nullptr, 0,
      nullptr, static_cast<int>(max_vocab_size), nullptr, nullptr, nullptr, static_cast<int>(max_batch_size), 0,
      nullptr, false, false));

  KLLM_LOG_DEBUG << "TopkSampling workspace_size_ " << workspace_size_;

  Malloc(&workspace_, workspace_size_ + sizeof(uint64_t) * max_batch_size);
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  std::vector<uint64_t> host_random_seeds(max_batch_size);
  for (auto& i : host_random_seeds) i = dis(gen);
  Memcpy(workspace_ + workspace_size_, host_random_seeds.data(), sizeof(uint64_t) * max_batch_size,
         MEMCPY_HOST_TO_DEVICE);

  CUDA_CHECK_LAST_ERROR(tensorrt_llm::kernels::InvokeCurandBatchInitialize(
      device_curandstates, nullptr, max_batch_size, static_cast<uint64_t*>(workspace_ + workspace_size_), 0));
#elif defined(ENABLE_ACL)
  // TODO(karlluo): support topk > 1 and toppp
  int32_t rank;
  GetDevice(&rank);
  atb_executor_.Init(rank, max_batch_size);
  atb_executors_ptr_ = &atb_executor_;
#endif
}

TopkSampling::~TopkSampling() {
  if (workspace_size_ > 0) {
    Free(workspace_);
  }
}

Status TopkSampling::RunSampling(float* logits, uint32_t* output_token, const SamplingConfig* sampling_config,
                                 SamplingDeviceParameter sampling_device_parameter, const ModelConfig* model_config,
                                 Stream& stream) {
  if (sampling_device_parameter.device_topKs == nullptr) {
#ifdef ENABLE_CUDA
    return ArgMax(logits, sampling_device_parameter.bs, sampling_device_parameter.vocab_size, output_token, stream);
#elif defined(ENABLE_ACL)
    return ArgMax(logits, sampling_device_parameter.bs, sampling_device_parameter.vocab_size, output_token, stream,
                  atb_executors_ptr_);
#endif
  } else {
#ifdef ENABLE_CUDA
    CUDA_CHECK_LAST_ERROR(tensorrt_llm::kernels::invoke_batch_topk_sampling(
        workspace_, workspace_size_, logits, sampling_device_parameter.device_output_tokens_ptrs, nullptr, nullptr,
        nullptr, nullptr, nullptr, sampling_device_parameter.device_curandstates, sampling_device_parameter.max_topK,
        sampling_device_parameter.device_topKs, 1.0, sampling_device_parameter.device_topPs,
        static_cast<int>(sampling_device_parameter.vocab_size), nullptr, nullptr, stream.Get(),
        static_cast<int>(sampling_device_parameter.bs), 0, nullptr, false, sampling_device_parameter.logits_softmax));
    return Status();
#else
    KLLM_THROW("Not support topk > 1 in Ascend NPU.");
#endif
  }
}

}  // namespace ksana_llm
