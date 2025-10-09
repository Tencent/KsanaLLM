/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/samplers/base/base_sampling.h"
#include "ksana_llm/samplers/topk/topk_sampling.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class Sampler {
 public:
  Sampler(const BatchSchedulerConfig& batch_scheduler_config, const int rank, std::shared_ptr<Context> context);
  ~Sampler();
  Status Sampling(size_t multi_batch_id, std::vector<SamplingRequest>& sampling_reqs, Stream& stream);
  Status SamplingAndCalcLogprobs(std::vector<SamplingRequest>& sampling_reqs, float* device_logits,
                                 SamplingDeviceParameter& sampling_device_parameter, Stream& stream);

  void SamplingParameterToDevice(bool use_top_k, bool use_top_p, bool use_temperature,
                                 SamplingDeviceParameter& sampling_device_parameter, Stream& stream);

  Status PrepareDeviceLogitsAndParameter(std::vector<SamplingRequest>& sampling_reqs,
                                         SamplingDeviceParameter& sampling_device_parameter, float*& device_logits,
                                         Stream& stream);

  // Copies the probabilities from the logits buffer to the output vector for each sampling request.
  std::function<void()> CopyProbsOutput(std::vector<SamplingRequest>& sampling_reqs, Stream& stream,
                                        std::vector<std::vector<float>>& probs_output);

  void ApplyRepetitionPenalty(float* logits, std::vector<int>* input_tokens, std::vector<int>* output_tokens,
                              const int vocab_size, const float repetition_penalty, Stream& stream);

  void CopyProbsOutputToRequests(std::vector<SamplingRequest>& sampling_reqs,
                                 std::vector<std::vector<float>>& probs_output, Stream& stream);

  void GetNgrams(const int ngram_size, const int cur_output_size, const std::vector<int>* output_tokens,
                 NgramDict* ngram_dict);

  void BanRepeatTokens(float* logits, const int ngram_size, const int cur_output_size,
                       const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                       Stream& stream);

  void NoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                              const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                              size_t last_step_token_num, Stream& stream);

  void EncoderNoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                                     const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                                     Stream& stream);

  void DecoderNoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                                     const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                                     size_t last_step_token_num, Stream& stream);

  // Apply grammar constraints to logits for requests with grammar matchers
  void ApplyGrammarMask(std::vector<SamplingRequest>& sampling_reqs, float* device_logits,
                        const SamplingDeviceParameter& sampling_device_parameter, Stream& stream);

  // Update grammar state after token selection for grammar-enabled requests
  void UpdateGrammarState(std::vector<SamplingRequest>& sampling_reqs);

  // Apply token bitmask selectively to grammar-enabled requests only
  void ApplyTokenBitmaskSelective(float* logits, void* bitmask_data, int vocab_size,
                                  const std::vector<size_t>& logits_offsets, Stream& stream);

 private:
  const BatchSchedulerConfig batch_schedule_config_;
  const int rank_;
  TopkSampling* topk_sampling_{nullptr};
  void* device_buffer_ = nullptr;
  uint32_t* device_output_tokens_;
  uint32_t* device_offset_;
  int* device_topKs_;
  float* device_topPs_;
  float* device_temperatures_;
  int** device_output_tokens_ptrs_;
  float* device_repetition_processor_;
  float* device_prob_;
  float** device_prob_ptrs_;
  RandState* device_curandstates_{nullptr};
  std::vector<int> host_output_tokens_;
  std::vector<int> host_topKs_;
  std::vector<float> host_topPs_;
  std::vector<float> host_temperatures_;
  std::vector<const float*> host_logits_;

  // The context
  std::shared_ptr<Context> context_;
  std::vector<float> inv_repetition_penalties_;
  std::vector<float> norepeat_ngrams_;

  // Grammar-related buffers
  int32_t* device_vocab_mask_ = nullptr;
  std::vector<int32_t> host_vocab_mask_;
};

}  // namespace ksana_llm
