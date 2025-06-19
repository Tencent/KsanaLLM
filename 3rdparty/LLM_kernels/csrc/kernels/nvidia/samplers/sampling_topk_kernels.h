/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/sampling_topk_kernels.h
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <curand_kernel.h>

#include "decoding_common.h"

namespace tensorrt_llm {
namespace kernels {
// clang-format off
//! \brief Given logProbs, performs top-K **and** top-P sampling at the same time. Fills sampled tokens to outputIds.
//! Computes sequenceLength, finished state, cumLogProbs in-place.
//! Sampling per request can be controlled using skipDecode, topPs and topKs parameters.
//! Function sets workspaceSize and exits early if workspace is nullptr.
//! If logits are NaN, we set output token to be the last in the vocabulary.
//!
//! \param workspace pointer to the workspace. Has to be pre-allocated by caller. Function does not take ownership of the
//! buffer.
//! \param workspace_size size of the workspace in bytes
//! \param log_probs input buffer [batchSize x vocabSizePadded].
//! Log probabilities of each token in the vocab. If logits_has_probs is true,
//! log_probs must contain **just** probabilities instead of log probabilities.
//! \param ids output buffer [maxBatchSize][maxSeqLen]. Contains pointers to rows with output tokens per request
//! \param sequence_lengths input/output buffer [maxBatchSize]. Current sequence length of the request up to, but excluding endId token
//! \param finished_input input buffer [maxBatchSize]. If true, request exits early.
//! \param finished_output output buffer [maxBatchSize]. Set flag if sequence has finished (if finished || outputId == endId).
//! \param cum_log_probs input/output buffer [maxBatchSize]. Cumulative log probability of selected tokens. Ignored if nullptr
//! \param output_log_probs output buffer [maxBatchSize]. Log probabilities induced by the top-K sampling.
//! If normalize_log_probs is true, we normalize the probability 'expLogit' of the selected token by the probability 's_sum' of a set of top-K
//! tokens, meaning the logProb is the probability of the selected token, conditioned on the event that it is selected,
//! i.e., log_prob = log P(i | i is in top-K) = log(expLogit / s_sum).
//! Ignored if nullptr.
//! \param curand_state input buffer [maxBatchSize]. Curand states properly
//! initialized using InvokeCurandInitialize per request.
//! \param max_topk maximum among all topks K for top-K sampling
//! \param topks input buffer [maxBatchSize]. K for top-K sampling per request.
//! Supported K is in range [1; 1024]. Where K=1 is greedy search. If nullptr, max_topk is used for all requests.
//! \param topp probability for top-P sampling.
//! \param topps input buffer [maxBatchSize]. Probability for top-P sampling per request.
//! Supported P is in range (0.0, 1.0]. If nullptr, topp is used for all requests
//! \param vocab_size_padded size of padded vocabulary
//! \param end_ids input buffer [maxBatchSize]. EOS token IDs per request
//! \param batch_slots input buffer [batchSize], optional. Indices of rows of data in memory pool
//! \param stream CUDA stream
//! \param batch_size batch size
//! \param max_batch_size maximum batch size
//! \param skip_decode input buffer [maxBatchSize]. Flags whether to skip decoding per request
//! \param normalize_log_probs when set to true, output_log_probs are normalized to top-K
//! \param logits_has_probs flag to indicate that log_probs contains probabilities
// clang-format on
template <typename T>
void invoke_batch_topk_sampling(void* workspace, size_t& workspace_size, const T* log_probs, int** ids,
                                int* sequence_lengths, const FinishedState* finished_input,
                                FinishedState* finished_output, float* cum_log_probs, float* output_log_probs,
                                curandState_t* curand_state, const int max_topk, const int* topks, const float topp,
                                const float* topps, const int vocab_size_padded, const int* end_ids,
                                const int* batch_slots, cudaStream_t stream, const int batch_size, int max_batch_size,
                                const bool* skip_decode, const bool normalize_log_probs, const bool logits_has_probs);

//! \brief Simplified version of invoke_batch_topk_sampling with single topK and topP values for all requests
//!
//! \param workspace pointer to the workspace. Has to be pre-allocated by caller. Function does not take ownership of
//! the buffer. \param workspace_size size of the workspace in bytes \param log_probs input buffer [batchSize x
//! vocabSizePadded]. Log probabilities of each token in the vocab. If logits_has_probs is true, log_probs must contain
//! **just** probabilities instead of log probabilities. \param ids output buffer [maxBatchSize][maxSeqLen]. Contains
//! pointers to rows with output tokens per request \param sequence_lengths input/output buffer [maxBatchSize]. Current
//! sequence length of the request up to, but excluding endId token \param finished_input input buffer [maxBatchSize].
//! If true, request exits early. \param finished_output output buffer [maxBatchSize]. Set flag if sequence has finished
//! (if finished || outputId == endId). \param cum_log_probs input/output buffer [maxBatchSize]. Cumulative log
//! probability of selected tokens. Ignored if nullptr \param output_log_probs output buffer [maxBatchSize]. Log
//! probabilities induced by the top-K sampling. If normalize_log_probs is true, we normalize the probability 'expLogit'
//! of the selected token by the probability 's_sum' of a set of top-K tokens, meaning the logProb is the probability of
//! the selected token, conditioned on the event that it is selected, i.e., log_prob = log P(i | i is in top-K) =
//! log(expLogit / s_sum). Ignored if nullptr. \param curand_state input buffer [maxBatchSize]. Curand states properly
//! initialized using InvokeCurandInitialize per request.
//! \param topk K value for top-K sampling. Supported K is in range [1; 1024]. Where K=1 is greedy search.
//! \param topp probability for top-P sampling. Supported P is in range (0.0, 1.0].
//! \param vocab_size_padded size of padded vocabulary
//! \param end_ids input buffer [maxBatchSize]. EOS token IDs per request
//! \param batch_slots input buffer [batchSize], optional. Indices of rows of data in memory pool
//! \param stream CUDA stream
//! \param batch_size batch size
//! \param max_batch_size maximum batch size
//! \param skip_decode input buffer [maxBatchSize]. Flags whether to skip decoding per request
//! \param normalize_log_probs when set to true, output_log_probs are normalized to top-K
//! \param logits_has_probs flag to indicate that log_probs contains probabilities
template <typename T>
void invoke_topk_sampling(void* workspace, size_t& workspace_size, const T* log_probs, int** ids, int* sequence_lengths,
                          const FinishedState* finished_input, FinishedState* finished_output, float* cum_log_probs,
                          float* output_log_probs, curandState_t* curand_state, const int topk, const float topp,
                          const int vocab_size_padded, const int* end_ids, const int* batch_slots, cudaStream_t stream,
                          const int batch_size, int max_batch_size, const bool* skip_decode,
                          const bool normalize_log_probs, const bool logits_has_probs);

}  // namespace kernels
}  // namespace tensorrt_llm
