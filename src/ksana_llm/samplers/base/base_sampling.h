/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"

struct curandStateXORWOW;
typedef struct curandStateXORWOW curandState_t;

#ifdef ENABLE_CUDA
typedef curandState_t RandState;
#else
// TODO(karlluo): need implement ascend random
typedef int RandState;
#endif

namespace ksana_llm {
struct SamplingDeviceParameter {
  int* device_topKs = nullptr;
  float* device_topPs = nullptr;
  float* device_temperatures = nullptr;
  int** device_output_tokens_ptrs = nullptr;
  RandState* device_curandstates = nullptr;
  // Whether to perform softmax on logits.
  bool logits_softmax{false};
  // Whether to do sampling, i.e., get tokens based on logits.
  bool do_sampling{false};
  int vocab_size = 0;
  int max_topK = 0;
  int bs = 0;
};

class BaseSampling {
 public:
  BaseSampling(size_t max_batch_size, size_t max_vocab_size)
      : max_batch_size_(max_batch_size), max_vocab_size_(max_vocab_size) {}
  Status Forward(float* logits, uint32_t* output_token, const SamplingConfig* sampling_config,
                 SamplingDeviceParameter sampling_device_parameter, const ModelConfig* model_config, Stream& stream);
  virtual ~BaseSampling() {}

 protected:
  virtual Status RunSampling(float* logits, uint32_t* output_token, const SamplingConfig* sampling_config,
                             SamplingDeviceParameter sampling_device_parameter, const ModelConfig* model_config,
                             Stream& stream) = 0;

  // The max batch size.
  size_t max_batch_size_ = 8;

  // The max vocab size.
  size_t max_vocab_size_ = 0;
};

}  // namespace ksana_llm
