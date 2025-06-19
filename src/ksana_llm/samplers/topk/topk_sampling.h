/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/samplers/base/base_sampling.h"

#ifdef ENABLE_ACL
#  include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#endif

namespace ksana_llm {

class TopkSampling : public BaseSampling {
 public:
  TopkSampling(size_t max_batch_size, size_t max_vocab_size, RandState* device_curandstates = nullptr);

  ~TopkSampling();

 private:
  Status RunSampling(float* logits, uint32_t* output_token, const SamplingConfig* sampling_config,
                     SamplingDeviceParameter sampling_device_parameter, const ModelConfig* model_config,
                     Stream& stream) override;

  void* workspace_ = nullptr;
  size_t workspace_size_{0ul};

#ifdef ENABLE_ACL
  ArgMaxATBExecutor<float> atb_executor_;
  void* atb_executors_ptr_ = nullptr;
#endif  // ENABLE_ACL
};

}  // namespace ksana_llm
