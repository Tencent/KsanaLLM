/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_CUDA
#  include <curand_kernel.h>
#  include "3rdparty/LLM_kernels/csrc/kernels/nvidia/samplers/copy_elements.cuh"
#  include "3rdparty/LLM_kernels/csrc/kernels/nvidia/samplers/decoding_common.h"
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"

namespace ksana_llm {

static size_t kCudaMemAlignmentSize = alignof(std::max_align_t);

Sampler::Sampler(const BatchSchedulerConfig& batch_scheduler_config, const int rank, std::shared_ptr<Context> context)
    : batch_schedule_config_(batch_scheduler_config), rank_(rank), context_(context) {
  KLLM_CHECK_WITH_INFO(sizeof(uint32_t) == sizeof(int),
                       fmt::format("sizeof(uint32_t)({}) != sizeof(int)({})", sizeof(uint32_t), sizeof(int)));

  // need to allocate device buffer for sampling
  SetDevice(rank_);
  const size_t max_logits_num =
      batch_scheduler_config.max_batch_size * batch_schedule_config_.max_decode_tokens_per_req;
  AlignedMemoryQueue aligned_memory_queue(kCudaMemAlignmentSize, [this](const size_t size) {
    SetDevice(rank_);
    Malloc(&device_buffer_, size);
    return device_buffer_;
  });
  aligned_memory_queue.Add(device_output_tokens_, max_logits_num);
  aligned_memory_queue.Add(device_topKs_, max_logits_num);
  aligned_memory_queue.Add(device_topPs_, max_logits_num);
  aligned_memory_queue.Add(device_temperatures_, max_logits_num);
  aligned_memory_queue.Add(device_curandstates_, max_logits_num);
  aligned_memory_queue.Add(device_output_tokens_ptrs_, max_logits_num);
  aligned_memory_queue.Add(device_repetition_processor_, batch_schedule_config_.max_vocab_size);
  aligned_memory_queue.Add(device_prob_, max_logits_num);
  aligned_memory_queue.Add(device_prob_ptrs_, max_logits_num);
  aligned_memory_queue.AllocateAndAlign();

  inv_repetition_penalties_.resize(batch_schedule_config_.max_vocab_size);
  norepeat_ngrams_.resize(batch_schedule_config_.max_vocab_size);

  std::vector<uint32_t*> output_tokens_ptrs_host(max_logits_num);
  iota(output_tokens_ptrs_host.begin(), output_tokens_ptrs_host.end(), device_output_tokens_);
  MemcpyAsync(device_output_tokens_ptrs_, output_tokens_ptrs_host.data(),
              sizeof(decltype(output_tokens_ptrs_host)::value_type) * output_tokens_ptrs_host.size(),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);

  host_topKs_.resize(max_logits_num);
  host_topPs_.resize(max_logits_num);
  host_temperatures_.resize(max_logits_num);
  host_output_tokens_.resize(max_logits_num);

  topk_sampling_ = new TopkSampling(max_logits_num, batch_scheduler_config.max_vocab_size, device_curandstates_);
}

Sampler::~Sampler() {
  // free device buffer of output tokens
  SetDevice(rank_);
  if (topk_sampling_ != nullptr) {
    delete topk_sampling_;
  }
  if (device_buffer_ != nullptr) {
    Free(device_buffer_);
    device_buffer_ = nullptr;
  }
}

void Sampler::ApplyRepetitionPenalty(float* logits, std::vector<int>* input_tokens, std::vector<int>* output_tokens,
                                     const int vocab_size, const float repetition_penalty, Stream& stream) {
  // inv_repetition_penalties_ is filled with 1.0f
  std::fill(inv_repetition_penalties_.begin(), inv_repetition_penalties_.end(), 1.0f);
  // If a token has appeared before, repetition_penalties is inv_repetition_penalty.
  const float inv_repetition_penalty = 1.0f / repetition_penalty;
  for (size_t i = 0; i < input_tokens->size(); ++i) {
    inv_repetition_penalties_[input_tokens->at(i)] = inv_repetition_penalty;
  }
  for (size_t i = 0; i < output_tokens->size(); ++i) {
    inv_repetition_penalties_[output_tokens->at(i)] = inv_repetition_penalty;
  }
  // copy inv_repetition_penalties_ to device
  MemcpyAsync(device_repetition_processor_, inv_repetition_penalties_.data(), sizeof(float) * vocab_size,
              MEMCPY_HOST_TO_DEVICE, stream);
  // logits = mul(logits, device_repetition_processor_)
#ifdef ENABLE_CUDA
  InvokeMul(logits, device_repetition_processor_, logits, vocab_size, rank_);
#endif
}

void Sampler::GetNgrams(const int ngram_size, const int cur_output_size, const std::vector<int>* output_tokens,
                        NgramDict* ngram_dict) {
  if (!ngram_dict->empty()) {
    return;
  }

  std::vector<std::vector<int>> ngrams;
  // for tokens recompute
  for (int i = 0; i <= cur_output_size - ngram_size; ++i) {
    std::vector<int> sub_ngram(output_tokens->begin() + i, output_tokens->begin() + i + ngram_size);
    ngrams.push_back(sub_ngram);
  }

  for (const auto& ngram : ngrams) {
    std::vector<int> ngram_excluding_last(ngram.begin(), ngram.end() - 1);
    int last_elem = ngram.back();
    (*ngram_dict)[ngram_excluding_last].push_back(last_elem);
  }
}

void Sampler::BanRepeatTokens(float* logits, const int ngram_size, const int cur_output_size,
                              const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                              Stream& stream) {
  std::vector<int> repeat_ids;
  int start_idx = cur_output_size - ngram_size + 1;
  std::vector<int> ngram_idx(output_tokens->begin() + start_idx, output_tokens->begin() + cur_output_size);
  if (ngram_dict->find(ngram_idx) != ngram_dict->end()) {
    repeat_ids = (*ngram_dict)[ngram_idx];
  } else {
    repeat_ids = {};
  }

  if (repeat_ids.size() > 0) {
    std::fill(norepeat_ngrams_.begin(), norepeat_ngrams_.end(), 0.0f);
    for (size_t i = 0; i < repeat_ids.size(); ++i) {
      norepeat_ngrams_[repeat_ids[i]] = -std::numeric_limits<float>::infinity();
    }
    MemcpyAsync(device_repetition_processor_, norepeat_ngrams_.data(), sizeof(float) * vocab_size,
                MEMCPY_HOST_TO_DEVICE, stream);
#ifdef ENABLE_CUDA
    InvokeAddBiasResidual<float>(logits, device_repetition_processor_, nullptr, 1, vocab_size, logits, stream.Get());
#endif
  }
}

void Sampler::NoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                                     const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                                     size_t last_step_token_num, Stream& stream) {
  int cur_output_size = output_tokens->size();
  if (ngram_size > cur_output_size) {
    KLLM_LOG_WARNING << fmt::format(
        "The no_repeat_ngram_size must be less than the number of tokens output by the Forward. {} < {}", ngram_size,
        cur_output_size);
    return;
  }
  if (input_tokens_size == cur_output_size) {
    KLLM_LOG_DEBUG << "for input and output tokens no repeat ngram sample";
    // TODO(winminkong): consider the computational approach for ngrams with re-computation.
    GetNgrams(ngram_size, cur_output_size, output_tokens, ngram_dict);
  } else if (input_tokens_size < cur_output_size) {
    for (size_t i = 0; i < last_step_token_num; ++i) {  // For MTP
      std::vector<int> sub_ngram(output_tokens->end() - ngram_size - i, output_tokens->end() - i);
      std::vector<int> ngram_excluding_last(sub_ngram.begin(), sub_ngram.end() - 1);
      int last_elem = sub_ngram.back();
      (*ngram_dict)[ngram_excluding_last].push_back(last_elem);
    }
  }
  BanRepeatTokens(logits, ngram_size, cur_output_size, output_tokens, ngram_dict, vocab_size, stream);
}

void Sampler::EncoderNoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                                            const std::vector<int>* output_tokens, NgramDict* ngram_dict,
                                            const int vocab_size, Stream& stream) {
  int cur_output_size = output_tokens->size();
  if (ngram_size > cur_output_size) {
    KLLM_LOG_WARNING << fmt::format(
        "The encoder_no_repeat_ngram_size must be less than the number of tokens output by the Forward. {} < {}",
        ngram_size, cur_output_size);
    return;
  }
  if (input_tokens_size == cur_output_size) {
    KLLM_LOG_DEBUG << "for input tokens no repeat ngram sample";
    GetNgrams(ngram_size, cur_output_size, output_tokens, ngram_dict);
  }
  BanRepeatTokens(logits, ngram_size, cur_output_size, output_tokens, ngram_dict, vocab_size, stream);
}

void Sampler::DecoderNoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                                            const std::vector<int>* output_tokens, NgramDict* ngram_dict,
                                            const int vocab_size, size_t last_step_token_num, Stream& stream) {
  int cur_output_size = output_tokens->size();
  if (ngram_size > cur_output_size - input_tokens_size) {
    KLLM_LOG_WARNING << fmt::format(
        "The decoder_no_repeat_ngram_size must be less than the number of tokens output by the Forward. {} < {}",
        ngram_size, cur_output_size - input_tokens_size);
    return;
  } else {
    for (size_t i = 0; i < last_step_token_num; ++i) {  // For MTP
      std::vector<int> sub_ngram(output_tokens->end() - ngram_size - i, output_tokens->end() - i);
      std::vector<int> ngram_excluding_last(sub_ngram.begin(), sub_ngram.end() - 1);
      int last_elem = sub_ngram.back();
      (*ngram_dict)[ngram_excluding_last].push_back(last_elem);
    }
  }
  BanRepeatTokens(logits, ngram_size, cur_output_size, output_tokens, ngram_dict, vocab_size, stream);
}

void Sampler::CopyProbsOutputToRequests(std::vector<SamplingRequest>& sampling_reqs,
                                        std::vector<std::vector<float>>& probs_output, Stream& stream) {
  auto copy_probs_after_synchronize = CopyProbsOutput(sampling_reqs, stream, probs_output);
  StreamSynchronize(stream);
  copy_probs_after_synchronize();
  for (size_t i = 0; i < sampling_reqs.size(); i++) {
    auto& req = sampling_reqs[i];
    req.sampling_result_tokens->insert(req.sampling_result_tokens->end(),
                                       host_output_tokens_.begin() + req.logits_offset,
                                       host_output_tokens_.begin() + req.logits_offset + req.sampling_token_num);
    if (req.request_target != nullptr) {
      auto it = req.request_target->find("logits");
      if (it != req.request_target->end()) {
        if (it->second.token_reduce_mode == TokenReduceMode::GATHER_ALL) {
          continue;
        }
      }
    }
    if (!probs_output[i].empty()) {
      PythonTensor& ret_tensor = (*req.response)["logits"];
      ret_tensor.shape = {probs_output[i].size()};
      ret_tensor.dtype = GetTypeString(TYPE_FP32);
      ret_tensor.data.resize(probs_output[i].size() * sizeof(float));
      memcpy(ret_tensor.data.data(), probs_output[i].data(), ret_tensor.data.size());
    }
  }
}

Status Sampler::SamplingAndCalcLogprobs(std::vector<SamplingRequest>& sampling_reqs, float* device_logits,
                                        SamplingDeviceParameter& sampling_device_parameter, Stream& stream) {
  for (auto& sampling_req : sampling_reqs) {
    auto& logprobs_num = sampling_req.sampling_config->logprobs_num;
    if (logprobs_num == 0) {
      sampling_req.logprobs->emplace_back();
      continue;
    }

    std::vector<float> logprobs(logprobs_num);
    std::vector<int64_t> token_ids(logprobs_num);
#ifdef ENABLE_CUDA
    auto& offset = sampling_req.logits_offset;
    auto& vocab_size = sampling_device_parameter.vocab_size_padded;
    float* device_temperatures_ptr = sampling_device_parameter.device_temperatures == nullptr
                                         ? nullptr
                                         : sampling_device_parameter.device_temperatures + offset;
    CalcLogprobs(device_logits + (offset * vocab_size), device_temperatures_ptr, vocab_size, 1,
                 sampling_req.sampling_config->logprobs_num, logprobs.data(), token_ids.data());
#endif
    std::vector<std::pair<int, float>> logprobs_output;
    for (int logprobs_index = 0; logprobs_index < sampling_req.sampling_config->logprobs_num; logprobs_index++) {
      logprobs_output.push_back({token_ids[logprobs_index], logprobs[logprobs_index]});
    }
    sampling_req.logprobs->emplace_back(logprobs_output);
  }
  return Status();
}

// Copies the probabilities from the logits buffer to the output vector for each sampling request.
std::function<void()> Sampler::CopyProbsOutput(std::vector<SamplingRequest>& sampling_reqs, Stream& stream,
                                               std::vector<std::vector<float>>& probs_output) {
  // Vectors to hold source and destination pointers for copying.
  std::vector<float*> src_ptr_vector;
  std::vector<float*> dst_ptr_vector;
  for (size_t i = 0; i < sampling_reqs.size(); i++) {
    if (sampling_reqs[i].logits_custom_length > 0) {
      if (sampling_reqs[i].request_target != nullptr) {
        auto it = sampling_reqs[i].request_target->find("logits");
        if (it != sampling_reqs[i].request_target->end()) {
          if (it->second.token_reduce_mode == TokenReduceMode::GATHER_ALL) {
            continue;
          }
        }
      }
      probs_output[i].resize(sampling_reqs[i].logits_custom_length);
      auto& input_tokens = *sampling_reqs[i].input_tokens;
      auto& vocab_size = batch_schedule_config_.max_vocab_size;
      size_t probs_index = 0;
      for (auto [l, r] : sampling_reqs[i].request_target->at("logits").slice_pos) {
        for (auto index = l; index <= r; index++) {
          size_t req_logits_offset = (sampling_reqs[i].logits_offset + probs_index) * vocab_size;
          // Add destination and source pointers for copying.
          dst_ptr_vector.push_back(probs_output[i].data() + probs_index);
          src_ptr_vector.push_back(sampling_reqs[i].logits_buf[rank_] + req_logits_offset + input_tokens[index + 1]);
          probs_index++;
        }
      }
    }
  }

  std::vector<float> dst_vector(src_ptr_vector.size());
#ifdef ENABLE_CUDA
  // Copy source pointers to device memory asynchronously.
  MemcpyAsync(device_prob_ptrs_, src_ptr_vector.data(), sizeof(float*) * src_ptr_vector.size(), MEMCPY_HOST_TO_DEVICE,
              stream);
  // Invoke kernel to copy elements from source to a temporary device buffer.
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokeCopyElements(device_prob_ptrs_, device_prob_, src_ptr_vector.size(), stream.Get()));
  // Copy the temporary device buffer to host memory asynchronously.
  MemcpyAsync(dst_vector.data(), device_prob_, sizeof(float) * src_ptr_vector.size(), MEMCPY_DEVICE_TO_HOST, stream);
#endif
  return [dst_vector = std::move(dst_vector), dst_ptr_vector = std::move(dst_ptr_vector)]() mutable {
    for (size_t i = 0; i < dst_ptr_vector.size(); i++) {
      *dst_ptr_vector[i] = dst_vector[i];
    }
  };
}

// Transfer sampling parameters to the device
void Sampler::SamplingParameterToDevice(bool use_top_k, bool use_top_p, bool use_temperature,
                                        SamplingDeviceParameter& sampling_device_parameter, Stream& stream) {
  if (use_top_k) {
    MemcpyAsync(device_topKs_, host_topKs_.data(), sizeof(int) * sampling_device_parameter.bs, MEMCPY_HOST_TO_DEVICE,
                stream);
    sampling_device_parameter.device_topKs = device_topKs_;
    sampling_device_parameter.device_output_tokens_ptrs = device_output_tokens_ptrs_;
    sampling_device_parameter.device_curandstates = device_curandstates_;
  }
  if (use_top_p) {
    MemcpyAsync(device_topPs_, host_topPs_.data(), sizeof(float) * sampling_device_parameter.bs, MEMCPY_HOST_TO_DEVICE,
                stream);
    sampling_device_parameter.device_topPs = device_topPs_;
  }
  if (use_temperature) {
    MemcpyAsync(device_temperatures_, host_temperatures_.data(), sizeof(float) * sampling_device_parameter.bs,
                MEMCPY_HOST_TO_DEVICE, stream);
    sampling_device_parameter.device_temperatures = device_temperatures_;
  }
}

Status Sampler::PrepareDeviceLogitsAndParameter(std::vector<SamplingRequest>& sampling_reqs,
                                                SamplingDeviceParameter& sampling_device_parameter,
                                                float*& device_logits, Stream& stream) {
  PROFILE_EVENT_SCOPE(PrepareDeviceLogitsAndParameter, "PrepareDeviceLogitsAndParameter", rank_);
  bool use_top_k = false;
  bool use_top_p = false;
  bool use_temperature = false;
  sampling_device_parameter.logits_softmax = false;
  sampling_device_parameter.do_sampling = false;
  const size_t max_logits_num =
      batch_schedule_config_.max_batch_size * batch_schedule_config_.max_decode_tokens_per_req;

  for (size_t req_index = 0; req_index < sampling_reqs.size(); ++req_index) {
    const auto& sampling_req = sampling_reqs[req_index];
    SamplingConfig* sampling_config = sampling_req.sampling_config;
    STATUS_CHECK_RETURN(sampling_config->VerifyArgs());
    sampling_device_parameter.logits_softmax |= sampling_req.logits_custom_length > 0;
    sampling_device_parameter.do_sampling |= sampling_req.logits_custom_length == 0;
    // In cases of logits_custom_length and speculative decoding, a single request may correspond to multiple logits
    sampling_device_parameter.bs += sampling_req.sampling_token_num;
    float* logits = sampling_req.logits_buf[rank_];
    if (device_logits != logits && device_logits != nullptr) {
      return Status(RET_SEGMENT_FAULT, "sampling for different logits not implemented");
    }
    device_logits = logits;
    sampling_device_parameter.vocab_size_padded = batch_schedule_config_.max_vocab_size;
    const size_t offset = sampling_req.logits_offset;
    if (offset >= max_logits_num) {
      return Status(RET_SEGMENT_FAULT, "sampling check sampling_req.logits_offset >= max_logits_num");
    }
    for (size_t sampling_index = 0; sampling_index < sampling_req.sampling_token_num; sampling_index++) {
      host_topKs_[offset + sampling_index] = sampling_config->topk;
      host_topPs_[offset + sampling_index] = sampling_config->topp;
      host_temperatures_[offset + sampling_index] = sampling_config->temperature;
    }
    if (sampling_device_parameter.max_topK < sampling_config->topk) {
      sampling_device_parameter.max_topK = sampling_config->topk;
    }
    use_top_k |= sampling_config->topk > 1;
    use_top_p |= sampling_config->topp != 1.0f;
    use_temperature |= sampling_config->temperature != 1.0f;

    const int vocab_size = batch_schedule_config_.max_vocab_size;
    if (sampling_config->repetition_penalty != 1.0f) {
      for (size_t sampling_index = 0; sampling_index < sampling_req.sampling_token_num; sampling_index++) {
        ApplyRepetitionPenalty(logits + (offset + sampling_index) * vocab_size, sampling_req.input_tokens,
                               sampling_req.sampling_result_tokens, vocab_size, sampling_config->repetition_penalty,
                               stream);
      }
    }

    const int input_tokens_size = sampling_req.input_tokens->size();
    // NOTE(winminkong): When MTP is enabled, the NoRepeatNgram sampling is applied only to the first token generated.
    if (sampling_config->no_repeat_ngram_size > 0) {
      NoRepeatNgramProcessor(logits + offset * vocab_size, sampling_config->no_repeat_ngram_size, input_tokens_size,
                             sampling_req.forwarding_tokens, sampling_req.ngram_dict, vocab_size,
                             sampling_req.last_step_token_num, stream);
    } else if (sampling_config->encoder_no_repeat_ngram_size > 0) {
      EncoderNoRepeatNgramProcessor(logits + offset * vocab_size, sampling_config->encoder_no_repeat_ngram_size,
                                    input_tokens_size, sampling_req.forwarding_tokens, sampling_req.ngram_dict,
                                    vocab_size, stream);
    } else if (sampling_config->decoder_no_repeat_ngram_size > 0) {
      DecoderNoRepeatNgramProcessor(logits + offset * vocab_size, sampling_config->decoder_no_repeat_ngram_size,
                                    input_tokens_size, sampling_req.forwarding_tokens, sampling_req.ngram_dict,
                                    vocab_size, sampling_req.last_step_token_num, stream);
    }
  }

  // top_p and temperature are applyed on the logits after softmax.
  sampling_device_parameter.logits_softmax |= use_top_p | use_temperature;
  SamplingParameterToDevice(use_top_k, use_top_p, use_temperature, sampling_device_parameter, stream);
  return Status();
}

Status Sampler::Sampling(size_t multi_batch_id, std::vector<SamplingRequest>& sampling_reqs, Stream& stream) {
  if (rank_ != 0) {
    return Status();
  }
  PROFILE_EVENT_SCOPE(Sampling_, fmt::format("Sampling_{}_{}", multi_batch_id, rank_), rank_);
  float* device_logits = nullptr;
  SamplingDeviceParameter sampling_device_parameter;
  STATUS_CHECK_RETURN(PrepareDeviceLogitsAndParameter(sampling_reqs, sampling_device_parameter, device_logits, stream));

  SamplingAndCalcLogprobs(sampling_reqs, device_logits, sampling_device_parameter, stream);
  // Apply softmax on logits.
  if (sampling_device_parameter.logits_softmax) {
#ifdef ENABLE_CUDA
    CUDA_CHECK_LAST_ERROR(tensorrt_llm::kernels::InvokeAddBiasSoftMax<float>(
        device_logits, nullptr, sampling_device_parameter.device_temperatures, nullptr, nullptr, nullptr, nullptr,
        sampling_device_parameter.bs, 0, 1, sampling_device_parameter.vocab_size_padded,
        sampling_device_parameter.vocab_size_padded, false, true, stream.Get()));
#else
    KLLM_THROW("Softmax is not supported on NPU.");
#endif
  }
  // Get the next tokens based on logits and the sampling parameters.
  if (sampling_device_parameter.do_sampling) {
    STATUS_CHECK_RETURN(topk_sampling_->Forward(device_logits, device_output_tokens_, nullptr,
                                                sampling_device_parameter, nullptr, stream));
    MemcpyAsync(host_output_tokens_.data(), device_output_tokens_, sizeof(uint32_t) * sampling_device_parameter.bs,
                MEMCPY_DEVICE_TO_HOST, stream);
  }
  std::vector<std::vector<float>> probs_output(sampling_reqs.size());
  CopyProbsOutputToRequests(sampling_reqs, probs_output, stream);
  return Status();
}

}  // namespace ksana_llm
