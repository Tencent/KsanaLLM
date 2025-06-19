/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/samplers/base/base_sampling.h"

namespace ksana_llm {

Status BaseSampling::Forward(float* logits, uint32_t* output_token, const SamplingConfig* sampling_config,
                             SamplingDeviceParameter sampling_device_parameter, const ModelConfig* model_config,
                             Stream& stream) {
  // the same for all type sampling

  // specialized sampling for different type
  STATUS_CHECK_RETURN(
      RunSampling(logits, output_token, sampling_config, sampling_device_parameter, model_config, stream));
  return Status();
}

}  // namespace ksana_llm
