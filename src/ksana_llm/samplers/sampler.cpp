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
static constexpr float kBitsPerInt = 32.0f;  // Number of bits in an integer for bitmask calculations

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

  // vocab mask buffer
  if (batch_schedule_config_.enable_xgrammar && rank_ == 0) {
    const int bitmask_elements = static_cast<int>(std::ceil(batch_schedule_config_.max_vocab_size / kBitsPerInt));
    const size_t vocab_mask_elements = batch_schedule_config_.max_batch_size * bitmask_elements;
    aligned_memory_queue.Add(device_vocab_mask_, vocab_mask_elements);
    host_vocab_mask_.resize(vocab_mask_elements);
    KLLM_LOG_INFO << "Grammar vocab mask buffers initialized successfully with " << vocab_mask_elements
                  << " elements (batch_size=" << batch_schedule_config_.max_batch_size
                  << ", bitmask_elements=" << bitmask_elements << ")";
  }

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

void Sampler::ApplyRepetitionPenalty(float* logits, const std::vector<int>* input_tokens,
                                     std::vector<int>* output_tokens, const int vocab_size,
                                     const float repetition_penalty, Stream& stream) {
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

void Sampler::CopyProbsOutputToRequests(std::vector<SamplingRequest*>& sampling_reqs, Stream& stream) {
  std::vector<std::vector<float>> probs_output(sampling_reqs.size());
  auto copy_probs_after_synchronize = CopyProbsOutput(sampling_reqs, stream, probs_output);
  StreamSynchronize(stream);
  copy_probs_after_synchronize();
  for (size_t i = 0; i < sampling_reqs.size(); i++) {
    auto& req = *sampling_reqs[i];
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

Status Sampler::SamplingAndCalcLogprobs(std::vector<SamplingRequest*>& sampling_reqs, float* device_logits,
                                        SamplingDeviceParameter& sampling_device_parameter, Stream& stream) {
  for (auto& sampling_req : sampling_reqs) {
    // TODO(winminkong): A batch of requests is judged and calculated only once.
    if (sampling_req->enable_mtp_sampler) {
      // Do not calculate output token logprobs when sample req is mtp req.
      continue;
    }

    auto& logprobs_num = sampling_req->sampling_config->logprobs_num;
    if (logprobs_num == 0) {
      sampling_req->logprobs->emplace_back();
      continue;
    }
    int universal_logprobs_num = logprobs_num > kMinLogprobsNum ? logprobs_num : kMinLogprobsNum;
    std::vector<float> logprobs(universal_logprobs_num);
    std::vector<int64_t> token_ids(universal_logprobs_num);
#ifdef ENABLE_CUDA
    auto& offset = sampling_req->logits_offset;
    auto& vocab_size = sampling_device_parameter.vocab_size;
    float* device_temperatures_ptr = sampling_device_parameter.device_temperatures == nullptr
                                         ? nullptr
                                         : sampling_device_parameter.device_temperatures + offset;
    for (size_t sampling_index = 0; sampling_index < sampling_req->sampling_token_num; sampling_index++) {
      CalcLogprobs(device_logits + (offset + sampling_index) * vocab_size, device_temperatures_ptr, vocab_size, 1,
                   universal_logprobs_num, logprobs.data(), token_ids.data());
#endif
      std::vector<std::pair<int, float>> logprobs_output;
      for (int logprobs_index = 0; logprobs_index < universal_logprobs_num; logprobs_index++) {
        logprobs_output.push_back({token_ids[logprobs_index], logprobs[logprobs_index]});
      }
      sampling_req->logprobs->emplace_back(logprobs_output);
#ifdef ENABLE_CUDA
    }
#endif
  }
  return Status();
}

// Copies the probabilities from the logits buffer to the output vector for each sampling request.
std::function<void()> Sampler::CopyProbsOutput(std::vector<SamplingRequest*>& sampling_reqs, Stream& stream,
                                               std::vector<std::vector<float>>& probs_output) {
  // Vectors to hold source and destination pointers for copying.
  std::vector<float*> src_ptr_vector;
  std::vector<float*> dst_ptr_vector;
  for (size_t i = 0; i < sampling_reqs.size(); i++) {
    auto& sampling_req = *sampling_reqs[i];
    if (sampling_req.logits_custom_length > 0 && sampling_req.request_target != nullptr) {
      const auto it = sampling_req.request_target->find("logits");
      if (it != sampling_req.request_target->end()) {
        if (it->second.token_reduce_mode == TokenReduceMode::GATHER_ALL) {
          continue;
        }
      }
      probs_output[i].resize(sampling_req.logits_custom_length);
      auto& input_tokens = *sampling_req.input_tokens;
      auto& vocab_size = batch_schedule_config_.max_vocab_size;
      size_t probs_index = 0;
      for (auto [l, r] : sampling_req.request_target->at("logits").slice_pos) {
        for (auto index = l; index <= r; index++) {
          size_t req_logits_offset = (sampling_req.logits_offset + probs_index) * vocab_size;
          // Add destination and source pointers for copying.
          dst_ptr_vector.push_back(probs_output[i].data() + probs_index);
          // For any part that exceeds the input token size, directly take the value of the zeroth position.
          size_t token_idx_offset = (index + 1) < input_tokens.size() ? input_tokens[index + 1] : 0;
          src_ptr_vector.push_back(sampling_req.logits_buf[rank_] + req_logits_offset + token_idx_offset);
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

Status Sampler::PrepareDeviceLogitsAndParameter(std::vector<SamplingRequest*>& sampling_reqs,
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

  for (auto& sampling_req : sampling_reqs) {
    SamplingConfig* const sampling_config = sampling_req->sampling_config;
    STATUS_CHECK_RETURN(sampling_config->VerifyArgs());
    sampling_device_parameter.logits_softmax |= sampling_req->logits_custom_length > 0;
    sampling_device_parameter.do_sampling |= sampling_req->logits_custom_length == 0;
    // In cases of logits_custom_length and speculative decoding, a single request may correspond to multiple logits
    sampling_device_parameter.bs += sampling_req->sampling_token_num;
    float* const logits = sampling_req->logits_buf[rank_];
    if (device_logits != logits && device_logits != nullptr) {
      return Status(RET_SEGMENT_FAULT, "sampling for different logits not implemented");
    }
    device_logits = logits;
    sampling_device_parameter.vocab_size = batch_schedule_config_.max_vocab_size;
    const size_t offset = sampling_req->logits_offset;
    if (offset >= max_logits_num) {
      return Status(RET_SEGMENT_FAULT, "sampling check sampling_req->logits_offset >= max_logits_num");
    }
    for (size_t sampling_index = 0; sampling_index < sampling_req->sampling_token_num; sampling_index++) {
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
    if (const auto it = sampling_req->request_target->find("logits"); it != sampling_req->request_target->end()) {
      const int input_top_logprobs_num = it->second.input_top_logprobs_num;
      sampling_req->input_top_logprobs_num = input_top_logprobs_num;
      sampling_device_parameter.max_input_top_logprobs_num =
          std::max(sampling_device_parameter.max_input_top_logprobs_num, input_top_logprobs_num);
    }

    const int vocab_size = batch_schedule_config_.max_vocab_size;
    if (sampling_config->repetition_penalty != 1.0f) {
      for (size_t sampling_index = 0; sampling_index < sampling_req->sampling_token_num; sampling_index++) {
        ApplyRepetitionPenalty(logits + (offset + sampling_index) * vocab_size, sampling_req->input_tokens,
                               sampling_req->sampling_result_tokens, vocab_size, sampling_config->repetition_penalty,
                               stream);
      }
    }

    const int input_tokens_size = sampling_req->input_tokens->size();
    // NOTE(winminkong): Do not apply NoRepeatNgram sampling when sample req is mtp req.
    if (sampling_req->enable_mtp_sampler) {
      continue;
    }
    // NOTE(winminkong): When mtp_step_num > 0, the NoRepeatNgram sampling is applied only to the first token generated.
    if (sampling_config->no_repeat_ngram_size > 0) {
      NoRepeatNgramProcessor(logits + offset * vocab_size, sampling_config->no_repeat_ngram_size, input_tokens_size,
                             sampling_req->forwarding_tokens, sampling_req->ngram_dict, vocab_size,
                             sampling_req->last_step_token_num, stream);
    } else if (sampling_config->encoder_no_repeat_ngram_size > 0) {
      EncoderNoRepeatNgramProcessor(logits + offset * vocab_size, sampling_config->encoder_no_repeat_ngram_size,
                                    input_tokens_size, sampling_req->forwarding_tokens, sampling_req->ngram_dict,
                                    vocab_size, stream);
    } else if (sampling_config->decoder_no_repeat_ngram_size > 0) {
      DecoderNoRepeatNgramProcessor(logits + offset * vocab_size, sampling_config->decoder_no_repeat_ngram_size,
                                    input_tokens_size, sampling_req->forwarding_tokens, sampling_req->ngram_dict,
                                    vocab_size, sampling_req->last_step_token_num, stream);
    }
  }

  // top_p and temperature are applyed on the logits after softmax.
  sampling_device_parameter.logits_softmax |= use_top_p | use_temperature;
  sampling_device_parameter.logits_softmax &= (sampling_device_parameter.max_input_top_logprobs_num == 0);
  SamplingParameterToDevice(use_top_k, use_top_p, use_temperature, sampling_device_parameter, stream);
  return Status();
}

Status Sampler::Sampling(size_t multi_batch_id, std::vector<SamplingRequest*>& sampling_reqs, Stream& stream) {
  if (rank_ != 0) {
    StreamSynchronize(context_->GetComputeStreams()[rank_]);
    return Status();
  }
  PROFILE_EVENT_SCOPE(Sampling_, fmt::format("Sampling_{}_{}", multi_batch_id, rank_), rank_);
  float* device_logits = nullptr;
  SamplingDeviceParameter sampling_device_parameter;
  STATUS_CHECK_RETURN(PrepareDeviceLogitsAndParameter(sampling_reqs, sampling_device_parameter, device_logits, stream));

  SamplingAndCalcLogprobs(sampling_reqs, device_logits, sampling_device_parameter, stream);

  // Apply grammar mask after logits processing
  ApplyGrammarMask(sampling_reqs, device_logits, sampling_device_parameter, stream);

  // Apply softmax on logits.
  if (sampling_device_parameter.logits_softmax) {
#ifdef ENABLE_CUDA
    CUDA_CHECK_LAST_ERROR(tensorrt_llm::kernels::InvokeAddBiasSoftMax<float>(
        device_logits, nullptr, sampling_device_parameter.device_temperatures, nullptr, nullptr, nullptr, nullptr,
        sampling_device_parameter.bs, 0, 1, sampling_device_parameter.vocab_size, sampling_device_parameter.vocab_size,
        false, true, stream.Get()));
#else
    KLLM_THROW("Softmax is not supported on NPU.");
#endif
  }

  if (sampling_device_parameter.max_input_top_logprobs_num > 0) {
    int max_top_num = sampling_device_parameter.max_input_top_logprobs_num;
    std::vector<std::vector<std::pair<int, float>>> input_top_logprobs_res(
        sampling_device_parameter.bs, std::vector<std::pair<int, float>>(max_top_num));
#ifdef ENABLE_CUDA
    CalcInputLogprobs(device_logits, sampling_device_parameter.device_temperatures,
                      sampling_device_parameter.vocab_size, sampling_device_parameter.bs, input_top_logprobs_res,
                      max_top_num);
#else
    KLLM_THROW("Input logprobs calculation is not supported on NPU.");
#endif
    int pruned_len = 0;
    for (size_t req_index = 0; req_index < sampling_reqs.size(); ++req_index) {
      auto& sampling_req = *sampling_reqs[req_index];
      if (sampling_req.enable_mtp_sampler) {
        KLLM_LOG_WARNING << "MTP sampler not support input_top_logprobs, please set mtp_step_num = 0";
        continue;
      }
      sampling_req.logprobs->clear();
      if (sampling_req.logits_custom_length <= 0) {
        sampling_req.logprobs->emplace_back();
      } else {
        for (int i = 0; i < sampling_req.logits_custom_length; ++i) {
          sampling_req.logprobs->emplace_back(
              input_top_logprobs_res[pruned_len + i].begin(),
              input_top_logprobs_res[pruned_len + i].begin() + sampling_req.input_top_logprobs_num);
        }
        pruned_len += sampling_req.logits_custom_length;
      }
    }
  }

  // Get the next tokens based on logits and the sampling parameters.
  if (sampling_device_parameter.do_sampling) {
    STATUS_CHECK_RETURN(topk_sampling_->Forward(device_logits, device_output_tokens_, nullptr,
                                                sampling_device_parameter, nullptr, stream));
    MemcpyAsync(host_output_tokens_.data(), device_output_tokens_, sizeof(uint32_t) * sampling_device_parameter.bs,
                MEMCPY_DEVICE_TO_HOST, stream);
  }

  CopyProbsOutputToRequests(sampling_reqs, stream);

  return Status();
}

void Sampler::ApplyGrammarMask(std::vector<SamplingRequest*>& sampling_reqs, float* device_logits,
                               const SamplingDeviceParameter& sampling_device_parameter, Stream& stream) {
  if (!batch_schedule_config_.enable_xgrammar) {
    return;
  }

  // allocate vocab mask
  const int bitmask_elements = static_cast<int>(std::ceil(batch_schedule_config_.max_vocab_size / kBitsPerInt));
  const int32_t full_mask = -1;  // All bits set to 1

  // Track which requests have grammar constraints
  std::vector<size_t> structured_req_indices;
  std::vector<size_t> structured_req_offsets;

  // Process each sampling request to identify structured-enabled requests
  for (size_t req_idx = 0; req_idx < sampling_reqs.size(); ++req_idx) {
    auto& req = *sampling_reqs[req_idx];

    if (!req.structured_generator || !req.apply_structured_constraint) {
      continue;
    }

    // Record the request index and its logits offset (only first token position for MTP)
    structured_req_indices.push_back(req_idx);
    structured_req_offsets.push_back(req.logits_offset);
  }

  if (structured_req_indices.empty()) {
    return;  // No requests with structured constraints
  }

  // Allocate bitmask only for structured-enabled requests
  const size_t structured_req_num = structured_req_indices.size();
  std::fill(host_vocab_mask_.begin(), host_vocab_mask_.begin() + structured_req_num * bitmask_elements, full_mask);

  // Fill bitmasks for structured-enabled requests
  bool has_active_constraint = false;
  for (size_t i = 0; i < structured_req_num; ++i) {
    size_t req_idx = structured_req_indices[i];
    auto& req = *sampling_reqs[req_idx];

    // Fill the bitmask at position i (not req_idx)
    int32_t* batch_bitmask = host_vocab_mask_.data() + i * bitmask_elements;
    bool needs_mask = req.structured_generator->FillNextTokenBitmask(batch_bitmask);

    if (needs_mask) {
      has_active_constraint = true;
      KLLM_LOG_DEBUG << fmt::format("Structured constraint applied: req={} idx={}/{}, sampling_token_num={}",
                                    req.req_id, req_idx, i, req.sampling_token_num);
    }
  }

  if (has_active_constraint) {
    KLLM_LOG_DEBUG << "Applying structured constraint mask: " << structured_req_num << "/" << sampling_reqs.size()
                   << " requests";

    // Copy bitmask to device (only for structured requests)
    const size_t vocab_mask_size = structured_req_num * bitmask_elements * sizeof(int32_t);
    MemcpyAsync(device_vocab_mask_, host_vocab_mask_.data(), vocab_mask_size, MEMCPY_HOST_TO_DEVICE, stream);

    // Apply bitmask only to structured-enabled requests
    ApplyTokenBitmaskSelective(device_logits, device_vocab_mask_, sampling_device_parameter.vocab_size,
                               structured_req_offsets, stream);
  }
}

void Sampler::ApplyTokenBitmaskSelective(float* logits, void* bitmask_data, int vocab_size,
                                         const std::vector<size_t>& logits_offsets, Stream& stream) {
  KLLM_LOG_DEBUG << "BitmaskSelective parameters: vocab_size=" << vocab_size
                 << ", num_requests=" << logits_offsets.size()
                 << ", bitmask_stride=" << static_cast<int>(std::ceil(vocab_size / kBitsPerInt));

#ifdef ENABLE_CUDA
  const int bitmask_stride = static_cast<int>(std::ceil(vocab_size / kBitsPerInt));

  // Apply bitmask to each grammar-enabled request individually
  for (size_t i = 0; i < logits_offsets.size(); ++i) {
    float* request_logits = logits + logits_offsets[i] * vocab_size;
    int32_t* request_bitmask = static_cast<int32_t*>(bitmask_data) + i * bitmask_stride;

    // Apply bitmask for single request
    InvokeApplyTokenBitmaskInplace<float>(request_logits, request_bitmask, nullptr, vocab_size, vocab_size,
                                          bitmask_stride, 1, stream.Get());
  }
#else
  // NPU implementation: empty implementation for now
  KLLM_THROW("ApplyTokenBitmask is not supported on NPU.");
#endif
}
}  // namespace ksana_llm
