/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/decoding_common.h
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

#include <curand_kernel.h>

namespace tensorrt_llm {
namespace kernels {

class FinishedState {
 public:
  static auto constexpr empty() { return FinishedState{0}; }

  static auto constexpr finished() { return FinishedState{kFinished}; }

  static auto constexpr skipDecoding() { return FinishedState{kSkipDecoding}; }

  static auto constexpr finishedEOS() { return FinishedState{kFinishedEos}; }

  static auto constexpr finishedMaxLength() { return FinishedState{kFinishedMaxLength}; }

  static auto constexpr finishedStopWords() { return FinishedState{kFinishedStopWords}; }

  __host__ __device__ void constexpr setFinishedEOS() { mState |= kFinishedEos; }

  __host__ __device__ bool constexpr isFinishedEOS() { return anyBitSet(kFinishedEos); }

  __host__ __device__ void constexpr setFinishedStopWords() { mState |= kFinishedStopWords; }

  __host__ __device__ bool constexpr isFinishedStopWords() { return anyBitSet(kFinishedStopWords); }

  __host__ __device__ void constexpr setFinishedMaxLength() { mState |= kFinishedMaxLength; }

  __host__ __device__ bool constexpr isFinishedMaxLength() { return anyBitSet(kFinishedMaxLength); }

  __host__ __device__ void constexpr setFinished() { mState |= kFinished; }

  __host__ __device__ bool constexpr isFinished() const { return anyBitSet(kFinished); }

  __host__ __device__ void constexpr setSkipDecoding() { mState = kSkipDecoding; }

  __host__ __device__ bool constexpr isSkipDecoding() const { return anyBitSet(kSkipDecoding); }

  using UnderlyingType = uint8_t;

 private:
  // The default state is interpreted as not finished.
  __host__ __device__ constexpr FinishedState(UnderlyingType state) : mState(state) {}

  // Request has finished based on the generation of EOS token
  static UnderlyingType constexpr kFinishedEos{1u << 0};
  // Request has finished based on the generation of stop words
  static UnderlyingType constexpr kFinishedStopWords{1u << 1};
  // Request has finished based on reaching max sequence length
  static UnderlyingType constexpr kFinishedMaxLength{1u << 2};
  // Finished by any condition
  static UnderlyingType constexpr kFinished{kFinishedEos | kFinishedStopWords | kFinishedMaxLength};
  // Skip decoding. E.g. used for not accepted tokens in speculative decoding
  static UnderlyingType constexpr kSkipDecoding{1u << 3};

  __host__ __device__ bool constexpr anyBitSet(UnderlyingType bits) const { return (mState & bits) != 0; }

  UnderlyingType mState{};
};

static_assert(!FinishedState::empty().isFinished());
static_assert(!FinishedState::empty().isSkipDecoding());
static_assert(FinishedState::finished().isFinished());
static_assert(FinishedState::skipDecoding().isSkipDecoding());
static_assert(FinishedState::finishedEOS().isFinishedEOS());
static_assert(FinishedState::finishedStopWords().isFinishedStopWords());
static_assert(FinishedState::finishedMaxLength().isFinishedMaxLength());

//! \brief Initialize batch_size curand states with given seed.
//!
//! \param state output buffer [max_batch_size]. Curand states to be initialized
//! \param batch_slots input buffer[batch_size], optional. Indices of rows of data in memory pool
//! \param batch_size number of states to initialize
//! \param random_seed seed to initialize states
//! \param stream stream
void InvokeCurandInitialize(curandState_t* state, const int* batch_slots, const size_t batch_size, uint64_t random_seed,
                            cudaStream_t stream);

//! \brief Initialize batch_size curand states with given seed per request.
//!
//! \param state output buffer [max_batch_size] of curand states to be initialized
//! \param batch_slots input buffer[batch_size], optional. Indices of rows of data in memory pool
//! \param batch_size number of states to initialize
//! \param random_seeds input buffer [max_batch_size] with seeds
//! \param stream stream
void InvokeCurandBatchInitialize(curandState_t* states, const int* batch_slots, const size_t batch_size,
                                 const uint64_t* random_seeds, cudaStream_t stream);

//! \brief Applies mask, adds bias to logits and computes softmax values.
//! Sets -MAX_FLT value for tokens in range [vocab_size; vocab_size_padded) to prevent them from being chosen.
//! If request finished the generation, sets MAX_FLT to endId token and -MAX_FLT to all other tokens forcing to choose
//! endId token. Otherwise, adds bias per token if bias pointer is not nullptr.
//!
//! \param logits input/output buffer [max_batch_size, vocab_size]. Logits to be modified by mask and bias.
//! If nullptr, logits_ptrs has to be provided.
//! \param logits_ptrs input/output buffer [max_batch_size][vocab_size]. Vector of pointers to the logits.
//! If nullptr, logits has to be provided.
//! \param temperatures
//! Can be the same pointer as logits
//! \param bias input buffer [vocab_size]. Bias to logit per token. Ignored if nullptr
//! \param end_ids input buffer [max_batch_size]. EOS token ids per request
//! \param finished input buffer [max_batch_size] with flags set to true if request has finished the generation
//! \param batch_slots input buffer[batch_size], optional. Indices of rows of data in memory pool
//! \param batch_size current batch size
//! \param max_batch_size max batch size
//! \param beam_width beam width
//! \param vocab_size unpadded vocab size
//! \param vocab_size_padded padded vocab size
//! \param skip_soft_max flag to skip softmax computation
//! \param batch_slots_logits flag to use batchSlot as index for logits and probs
//! \param stream stream
template <typename T>
void InvokeAddBiasSoftMax(T* logits, T** logits_ptrs, T* temperatures, T const* bias, int32_t const* end_ids,
                          FinishedState const* finished, int32_t const* batch_slots, int32_t batch_size,
                          int32_t max_batch_size, int32_t beam_width, int32_t vocab_size, int32_t vocab_size_padded,
                          bool skip_soft_max, bool batch_slots_logits, cudaStream_t stream);
}  // namespace kernels
}  // namespace tensorrt_llm
