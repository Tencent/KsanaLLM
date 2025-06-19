/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/samplingTopKKernels.cu
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/common/reduceKernelUtils.cuh
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cfloat>
#include <stdexcept>

#include <cub/cub.cuh>

#include "sampling_topk_kernels.h"

namespace tensorrt_llm {
namespace kernels {

static float constexpr HALF_FLT_MAX = 65504.F;
template <typename T>
struct TopK_2 {
  int p = -1;
  T u = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);

  __device__ __forceinline__ void insert(T elem, int elem_id) {
    if (elem > u) {
      u = elem;
      p = elem_id;
    }
  }

  __device__ __forceinline__ void init() {
    u = -((std::is_same<T, half>::value) ? HALF_FLT_MAX : FLT_MAX);
    p = -1;
  }
};

template <typename T>
__device__ __forceinline__ TopK_2<T> reduce_topk_op_2(const TopK_2<T>& a, const TopK_2<T>& b) {
  return a.u > b.u ? a : b;
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage1(const T* __restrict log_probs, T* tmp_log_probs, int* topk_tmp_id_buf, T* topk_tmp_val_buf,
                            const FinishedState* finished, const int max_topk, const int* topks, const int vocab_size,
                            const int* end_ids, const bool* skip_decode, const int* batch_slots) {
  typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int const tid = threadIdx.x;
  int const bid = blockIdx.x;

  auto const batch_id = bid / BLOCKS_PER_BEAM_;  // row id for log_probs
  auto const batch_slot = batch_slots != nullptr ? batch_slots[batch_id] : batch_id;
  FinishedState const finish_state = finished != nullptr ? finished[batch_slot] : FinishedState::empty();
  if ((skip_decode != nullptr && skip_decode[batch_slot]) || (finish_state.isSkipDecoding())) {
    return;
  }
  const int block_lane = bid % BLOCKS_PER_BEAM_;                    // block id for a beam
  const int k = (topks != nullptr) ? topks[batch_slot] : max_topk;  // batch_id = batch index

  const int log_buf_index = batch_id * vocab_size;
  const int tmp_log_buf_index = batch_id * vocab_size;
  const int tmp_topk_buf_index = batch_id * BLOCKS_PER_BEAM_ * max_topk + block_lane * k;

  TopK_2<T> partial;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  if (finished != nullptr && finish_state.isFinished()) {
    if (tid < k) {
      const int index = tmp_topk_buf_index + tid;
      if (block_lane == 0 && tid == 0) {
        const int end_id = end_ids[batch_slot];
        topk_tmp_id_buf[index] = tmp_log_buf_index + end_id;
        topk_tmp_val_buf[index] = log_probs[log_buf_index + end_id];
      } else {
        topk_tmp_id_buf[index] = -1;
        topk_tmp_val_buf[index] = -MAX_T_VAL;
      }
    }
    return;
  }

  for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size; elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
    int local_index = elem_id + tmp_log_buf_index;
    int global_index = elem_id + log_buf_index;
    tmp_log_probs[local_index] = log_probs[global_index];
  }

  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
         elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
      int index = elem_id + tmp_log_buf_index;
      partial.insert(tmp_log_probs[index], index);
    }

    TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

    if (tid == 0) {
      const int index = tmp_topk_buf_index + ite;
      topk_tmp_id_buf[index] = total.p;
      topk_tmp_val_buf[index] = total.u;
      if (total.p >= 0) {
        tmp_log_probs[total.p] = -MAX_T_VAL;
      }
    }
    __syncthreads();
  }
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage2_sampling(const int* __restrict topk_tmp_id_buf, T* topk_tmp_val_buf, int** ids,
                                     int* sequence_lengths, const FinishedState* finished_input,
                                     FinishedState* finished_output, float* cum_log_probs, float* output_log_probs,
                                     const int max_topk, const int* topks, const float topp, const float* topps,
                                     curandState_t* curand_state, const int* end_ids, const int vocab_size,
                                     const bool* skip_decode, const int* batch_slots, int max_batch_size,
                                     const bool normalize_log_probs, const bool logitHasProbs) {
  bool const IS_FP16 = std::is_same<T, half>::value;
  T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  int const tid = threadIdx.x;
  auto const batch_idx = blockIdx.x;
  auto const batch_slot = batch_slots != nullptr ? batch_slots[batch_idx] : batch_idx;
  FinishedState const finish_state = finished_input != nullptr ? finished_input[batch_slot] : FinishedState::empty();
  if ((skip_decode != nullptr && skip_decode[batch_slot]) || (finish_state.isSkipDecoding())) {
    return;
  }

  const int k = (topks != nullptr) ? topks[batch_slot] : max_topk;
  const float prob_threshold = (topps != nullptr) ? topps[batch_slot] : topp;
  const int size = k * BLOCKS_PER_BEAM_;
  const int stride = max_topk * BLOCKS_PER_BEAM_;

  typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  extern __shared__ char array[];
  __shared__ float s_sum;
  T* s_val = topk_tmp_val_buf + batch_idx * stride;
  int* s_id = reinterpret_cast<int*>(array);
  if (tid == 0) {
    s_sum = 0.0f;
  }
  TopK_2<float> partial;

  if (finish_state.isFinished()) {
    if (finished_output != nullptr) {
      finished_output[batch_slot] = finish_state;
    }
    return;
  }

  float* s_val2 = reinterpret_cast<float*>(s_id + k);
  float maxLogit;
  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int i = tid; i < size; i += BLOCK_SIZE_) {
      partial.insert((float)s_val[i], i);
    }

    TopK_2<float> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<float>);

    if (tid == 0) {
      if (ite == 0) {
        maxLogit = total.u;
      }
      s_id[ite] = total.p;
      s_val[total.p] = -MAX_T_VAL;

      // when cum_log_probs are computed, topk_tmp_val_buf (logits_buf_) are
      // already pre-processed by softmax_kernel
      if (!logitHasProbs) {
        total.u = __expf(total.u - maxLogit);
      }
      s_val2[ite] = total.u;
      s_sum += total.u;
    }
    __syncthreads();
  }

  if (tid == 0) {
    float rand_num = (float)curand_uniform(curand_state + batch_slot) * prob_threshold * s_sum;
    for (int i = 0; i < k; i++) {
      float expLogit = s_val2[i];
      rand_num = rand_num - expLogit;
      if (rand_num <= 0.0f || i == k - 1) {
        int idx = s_id[i];
        // If s_id is -1 here we force output token to the last from vocabulary to get vivid indicator of smth
        // going wrong for the debug
        auto outputId = idx != -1 ? topk_tmp_id_buf[batch_idx * stride + idx] % vocab_size : vocab_size - 1;
        auto cur_seqlen = 0;
        if (sequence_lengths != nullptr) {
          cur_seqlen = sequence_lengths[batch_slot];
        }
        ids[batch_slot][cur_seqlen] = outputId;
        if (cum_log_probs != nullptr || output_log_probs != nullptr) {
          float log_prob = logf(expLogit);
          if (cum_log_probs != nullptr) {
            cum_log_probs[batch_slot] += log_prob;
          }
          if (output_log_probs != nullptr) {
            // 'output_log_probs' is the probability induced by the top-k sampling:
            // NOT normalized (same way as OpenAI does):
            // log_prob = log P(i | i is in top-k) = log(expLogit)
            // normalized:
            // log_prob = log P(i | i is in top-k) = log(expLogit / sum)
            output_log_probs[cur_seqlen * max_batch_size + batch_slot] =
                normalize_log_probs ? log_prob - logf(s_sum) : log_prob;
          }
        }
        break;
      }
    }
    if (sequence_lengths != nullptr && finished_output != nullptr) {
      const int seq_len = sequence_lengths[batch_slot];
      if (ids[batch_slot][seq_len] == end_ids[batch_slot]) {
        finished_output[batch_slot].setFinishedEOS();
        // Do not increase seq len when EOS is generated. Seq len should always contain only tokens to be
        // outputted
      } else {
        // We don't need to set output finished state as it is assumed to be in non finished state
        sequence_lengths[batch_slot] += 1;
      }
    }
  }
}

#define CASE_K(K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_, normalize_log_probs)                             \
  topk_stage1<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_><<<batch_size * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>(        \
      log_probs, temp_log_probs, topk_tmp_id_buf, topk_tmp_val_buf, finished_input, max_topk, topks, vocab_size,       \
      end_ids, skip_decode, batch_slots);                                                                              \
  topk_stage2_sampling<T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                                                             \
      <<<batch_size, BLOCK_SIZE_2_, K_MAX * sizeof(int) + K_MAX * sizeof(float), stream>>>(                            \
          topk_tmp_id_buf, topk_tmp_val_buf, ids, sequence_lengths, finished_input, finished_output, cum_log_probs,    \
          output_log_probs, max_topk, topks, topp, topps, curand_state, end_ids, vocab_size, skip_decode, batch_slots, \
          max_batch_size, normalize_log_probs, logits_has_probs);                                                      \
  break;

template <typename T>
void invoke_batch_topk_sampling(void* workspace, size_t& workspace_size, const T* log_probs, int** ids,
                                int* sequence_lengths, const FinishedState* finished_input,
                                FinishedState* finished_output, float* cum_log_probs, float* output_log_probs,
                                curandState_t* curand_state, const int max_topk, const int* topks, const float topp,
                                const float* topps, const int vocab_size_padded, const int* end_ids,
                                const int* batch_slots, cudaStream_t stream, const int batch_size, int max_batch_size,
                                const bool* skip_decode, const bool normalize_log_probs, const bool logits_has_probs) {
  // Not allow an ambiguous inputs topp and topps.
  assert(topp == 1.0f || topps == nullptr);
  const int vocab_size = vocab_size_padded;
  const int max_block_per_beam = 8;
  int temp_log_probs_buf_size = batch_size * vocab_size;                   // type float
  int topk_tmp_ids_buf_size = batch_size * max_topk * max_block_per_beam;  // type int
  int topk_tmp_val_buf_size = batch_size * max_topk * max_block_per_beam;  // type float

  // prevent memory misaligned address
  temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
  topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
  topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

  if (workspace == nullptr) {
    workspace_size =
        sizeof(T) * temp_log_probs_buf_size + sizeof(int) * topk_tmp_ids_buf_size + sizeof(T) * topk_tmp_val_buf_size;
    return;
  }

  if (max_topk == 0) {
    return;
  }

  T* temp_log_probs = (T*)workspace;
  int* topk_tmp_id_buf = (int*)(temp_log_probs + temp_log_probs_buf_size);
  T* topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);

  int log_max_topk(0);
  int recursor(max_topk - 1);
  while (recursor >>= 1) ++log_max_topk;
  switch (log_max_topk) {
    case 0:
    case 1:
    case 2:
    case 3:  // 0 < max_topk <= 16
      CASE_K(16, 128, 128, 8, normalize_log_probs);
    case 4:  // 16 < max_topk <= 32
      CASE_K(32, 256, 128, 8, normalize_log_probs);
    case 5:  // 32 < max_topk <= 64
      CASE_K(64, 256, 256, 8, normalize_log_probs);
    case 6:
    case 7:
    case 8:
    case 9:  // 64 < max_topk <= 1024
      CASE_K(1024, 256, 256, 8, normalize_log_probs);
    default:
      throw std::domain_error("top-k kernel supports 1<=k<=1024 but got k=" + std::to_string(max_topk));
  }
}

#undef CASE_K

template void invoke_batch_topk_sampling(void* workspace, size_t& workspace_size, const float* log_probs, int** ids,
                                         int* sequence_lengths, const FinishedState* finished_input,
                                         FinishedState* finished_output, float* cum_log_probs, float* output_log_probs,
                                         curandState_t* curand_state, const int max_topk, const int* topks,
                                         const float topp, const float* topps, const int vocab_size_padded,
                                         const int* end_ids, const int* batch_slots, cudaStream_t stream,
                                         const int batch_size, int max_batch_size, const bool* skip_decode,
                                         const bool normalize_log_probs, const bool logits_has_probs);

template void invoke_batch_topk_sampling(void* workspace, size_t& workspace_size, const half* log_probs, int** ids,
                                         int* sequence_lengths, const FinishedState* finished_input,
                                         FinishedState* finished_output, float* cum_log_probs, float* output_log_probs,
                                         curandState_t* curand_state, const int max_topk, const int* topks,
                                         const float topp, const float* topps, const int vocab_size_padded,
                                         const int* end_ids, const int* batch_slots, cudaStream_t stream,
                                         const int batch_size, int max_batch_size, const bool* skip_decode,
                                         const bool normalize_log_probs, const bool logits_has_probs);

template <typename T>
void invoke_topk_sampling(void* workspace, size_t& workspace_size, const T* log_probs, int** ids, int* sequence_lengths,
                          const FinishedState* finished_input, FinishedState* finished_output, float* cum_log_probs,
                          float* output_log_probs, curandState_t* curand_state, const int topk, const float topp,
                          const int vocab_size_padded, const int* end_ids, const int* batch_slots, cudaStream_t stream,
                          const int batch_size, int max_batch_size, const bool* skip_decode,
                          const bool normalize_log_probs, const bool logits_has_probs) {
  invoke_batch_topk_sampling(workspace, workspace_size, log_probs, ids, sequence_lengths, finished_input,
                             finished_output, cum_log_probs, output_log_probs, curand_state, topk, nullptr, topp,
                             nullptr, vocab_size_padded, end_ids, batch_slots, stream, batch_size, max_batch_size,
                             skip_decode, normalize_log_probs, logits_has_probs);
}

template void invoke_topk_sampling(void* workspace, size_t& workspace_size, const float* log_probs, int** ids,
                                   int* sequence_lengths, const FinishedState* finished_input,
                                   FinishedState* finished_output, float* cum_log_probs, float* output_log_probs,
                                   curandState_t* curand_state, const int topk, const float topp,
                                   const int vocab_size_padded, const int* end_ids, const int* batch_slots,
                                   cudaStream_t stream, const int batch_size, int max_batch_size,
                                   const bool* skip_decode, const bool normalize_log_probs,
                                   const bool logits_has_probs);

template void invoke_topk_sampling(void* workspace, size_t& workspace_size, const half* log_probs, int** ids,
                                   int* sequence_lengths, const FinishedState* finished_input,
                                   FinishedState* finished_output, float* cum_log_probs, float* output_log_probs,
                                   curandState_t* curand_state, const int topk, const float topp,
                                   const int vocab_size_padded, const int* end_ids, const int* batch_slots,
                                   cudaStream_t stream, const int batch_size, int max_batch_size,
                                   const bool* skip_decode, const bool normalize_log_probs,
                                   const bool logits_has_probs);

}  // namespace kernels
}  // namespace tensorrt_llm
