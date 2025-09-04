/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h
 *
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

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace llm_kernels {
namespace nvidia {

template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half> {
  static constexpr int value = 8;
};

template <>
struct ElemsPerAccess<nv_bfloat16> {
  static constexpr int value = 8;
};

template <>
struct ElemsPerAccess<float> {
  static constexpr int value = 4;
};

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;
// This is a conservative value
static constexpr int kOneShotMaxToken = 128;
static constexpr int kBarrierFlagCount = 256;
static constexpr int kLamportAlignSize = 1 << 21;
static constexpr int kMaxCommSize = 2147483647 & ~(kLamportAlignSize - 1);  // MAX_INT32 rounded down to 2MB

enum class AllReduceFusionPattern : int {
  kAllReduce = 0,          // Basic all-reduce pattern
  kARResidualRMSNorm = 1,  // All-reduce followed by residual add and RMS norm
};

enum class QuantType : int {
  kNone = 0,
};

template <AllReduceFusionPattern Pattern>
struct FusionPatternTraits;

#define DEFINE_FUSION_PATTERN_TRAITS(pattern, hasAllReduceOut, hasResidual, hasResidualOut, hasRMSNorm, hasNormOut, \
                                     quantType)                                                                     \
  template <>                                                                                                       \
  struct FusionPatternTraits<pattern> {                                                                             \
    static constexpr bool kHasAllReduceOut = hasAllReduceOut;                                                       \
    static constexpr bool kHasResidual = hasResidual;                                                               \
    static constexpr bool kHasResidualOut = hasResidualOut;                                                         \
    static constexpr bool kHasRMSNorm = hasRMSNorm;                                                                 \
    static constexpr bool kHasNormOut = hasNormOut;                                                                 \
    static constexpr QuantType kQuantType = quantType;                                                              \
  };

DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kAllReduce, true, false, false, false, false, QuantType::kNone);
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNorm, false, true, true, true, true,
                             QuantType::kNone);
#undef DEFINE_FUSION_PATTERN_TRAITS

template <AllReduceFusionPattern Pattern>
constexpr bool HasResidual = FusionPatternTraits<Pattern>::kHasResidual;
template <AllReduceFusionPattern Pattern>
constexpr bool HasRMSNorm = FusionPatternTraits<Pattern>::kHasRMSNorm;
template <AllReduceFusionPattern Pattern>
constexpr bool HasAllReduceOut = FusionPatternTraits<Pattern>::kHasAllReduceOut;
template <AllReduceFusionPattern Pattern>
constexpr bool HasResidualOut = FusionPatternTraits<Pattern>::kHasResidualOut;
template <AllReduceFusionPattern Pattern>
constexpr bool HasNormOut = FusionPatternTraits<Pattern>::kHasNormOut;
template <AllReduceFusionPattern Pattern>
constexpr QuantType GetQuantType = FusionPatternTraits<Pattern>::kQuantType;

template <typename T>
struct AllReduceFusionParams {
  int nranks;                            // the size of the allreduce group
  int rank;                              // the current rank in the group
  int size;                              // the number of elements in allreduce: `token_num * hidden_dim`
  int hidden_dim;                        // the dimension of the hidden states
  void** workspace{nullptr};             // the workspace pointers: [3 * nranks + 1]
  void* allreduce_in{nullptr};           // the input tensor: [token_num, hidden_dim]
  void* residual_in{nullptr};            // the residual input tensor: [token_num, hidden_dim]
  void* allreduce_out{nullptr};          // the output tensor: [token_num, hidden_dim]
  void* residual_out{nullptr};           // the residual output tensor: [token_num, hidden_dim]
  void* norm_out{nullptr};               // the norm output tensor: [token_num, hidden_dim]
  void* rms_gamma{nullptr};              // the rms gamma tensor: [hidden_dim]
  float rms_eps{1e-6};                   // the rms epsilon value
  bool use_oneshot{false};               // whether to use oneshot
  cudaStream_t stream;                   // the cuda stream
  AllReduceFusionPattern pattern;        // the fusion pattern in allreduce
  bool trigger_completion_at_end{true};  // whether to trigger completion at the end
};

inline bool use_oneshot(const int token_num) { return token_num <= kOneShotMaxToken; }

template <typename T>
void allreduce_fusion_op(AllReduceFusionParams<T> const& params);

void AllocTrtAllReduceWorkspace(const int nranks, const int rank, const int max_token_num, const int hidden_dim,
                                const int data_type_size, std::vector<void*>& buffer_d_ptrs,
                                std::vector<void*>& flag_d_ptrs, std::vector<void*>& workspace_d_ptrs,
                                cudaStream_t stream);

void InitTrtAllReduceWorkspace(const int nranks, const int rank, const std::vector<void*>& buffer_d_ptrs,
                               const std::vector<void*>& flag_d_ptrs, const std::vector<void*>& workspace_d_ptrs,
                               cudaStream_t stream);

void FreeTrtAllReduceWorkspace(const int nranks, const int rank, const std::vector<void*>& buffer_d_ptrs,
                               const std::vector<void*>& flag_d_ptrs, const std::vector<void*>& workspace_d_ptrs,
                               cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
