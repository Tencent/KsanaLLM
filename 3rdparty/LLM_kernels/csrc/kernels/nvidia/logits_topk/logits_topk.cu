/*
 * Copyright 2025 vLLM Team
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
 *
 * Adapted from
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/9798b2fb0052092a6420172e41c0c8a307eedfa6/csrc/quantization/cutlass_w8a8/c3x/cutlass_gemm_caller.cuh
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace llm_kernels {
namespace nvidia {

static inline __device__ uint16_t extractBinIdx(float x) {
  union {
    __half h;
    uint16_t u16;
  } tmp;
  tmp.h = __float2half_rn(x);
  tmp.u16 = (x < 0.f) ? (~tmp.u16 & 0xffff) : (tmp.u16 | 0x8000);
  return 511 - (tmp.u16 >> 7);
}

template <int kNumThreadsPerBlock = 512>
static __global__ void topKPerRow(const float* logits, const int32_t* rowStarts,
                                  const int32_t* rowEnds, int32_t* outIndices,
                                  int64_t stride0, int64_t stride1) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 512;

  // The top-k width.
  static constexpr int kTopK = 2048;
  // The number of elements per thread for the final top-k sort.
  static constexpr int kNumTopKItemsPerThread = kTopK / kNumThreadsPerBlock;
  // The class to sort the elements during the final top-k sort.
  using TopKSort = cub::BlockRadixSort<float, kNumThreadsPerBlock,
                                       kNumTopKItemsPerThread, int32_t>;

  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = 3072;
  // The number of elements per thread for the final sort.
  static constexpr int kNumFinalItemsPerThread =
      kNumFinalItems / kNumThreadsPerBlock;
  // The class to sort the elements during the final pass.
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock,
                                        kNumFinalItemsPerThread, int32_t>;

  // The class to compute the inclusive prefix-sum over the histogram.
  using Scan = cub::BlockScan<int32_t, kNumThreadsPerBlock>;

  // Shared memory to compute the block scan.
  __shared__ typename Scan::TempStorage smemScan;

  // The structure to store the final items (for the final pass).
  struct FinalItems {
    // Shared memory to store the indices for the final pass.
    int32_t indices[kNumFinalItems];
    // Shared memory to store the logits for the final pass.
    float logits[kNumFinalItems];
  };

  // Shared memory to compute the block sort.
  __shared__ union {
    FinalItems items;
    typename FinalSort::TempStorage finalSort;
    typename TopKSort::TempStorage topKSort;
  } smemFinal;

  // Shared memory to store the histogram.
  __shared__ int32_t smemHistogram[kNumBins];
  // Shared memory to store the selected indices.
  __shared__ int32_t smemIndices[kTopK];
  // Shared memory to store the selected logits.
  __shared__ float smemLogits[kTopK];
  // Shared memory to store the threshold bin.
  __shared__ int32_t smemThresholdBinIdx[1];
  // Shared memory counter to register the candidates for the final phase.
  __shared__ int32_t smemFinalDstIdx[1];

  // The row computed by this block.
  int32_t rowIdx = blockIdx.x;
  // The range of logits within the row.
  int32_t rowStart = rowStarts[rowIdx], rowEnd = rowEnds[rowIdx];
  // The length of the row.
  int32_t rowLen = rowEnd - rowStart;

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (rowLen <= kTopK) {
    for (int32_t rowIt = threadIdx.x; rowIt < rowLen;
         rowIt += kNumThreadsPerBlock) {
      int32_t idx = rowStart + rowIt;
      outIndices[rowIdx * kTopK + rowIt] = idx - rowStart;
    }
    for (int32_t rowIt = rowLen + threadIdx.x; rowIt < kTopK;
         rowIt += kNumThreadsPerBlock) {
      outIndices[rowIdx * kTopK + rowIt] = -1;
    }
    return;
  }

  // Clear the histogram.
  if (threadIdx.x < kNumBins) {
    smemHistogram[threadIdx.x] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Fetch elements one-by-one.
  for (int32_t rowIt = rowStart + threadIdx.x; rowIt < rowEnd;
       rowIt += kNumThreadsPerBlock) {
    uint16_t idx = extractBinIdx(logits[rowIdx * stride0 + rowIt * stride1]);
    atomicAdd(&smemHistogram[idx], 1);
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Read the values from SMEM.
  int32_t binCount{0};
  if (threadIdx.x < kNumBins) {
    binCount = smemHistogram[threadIdx.x];
  }

  // Make sure each thread has read its value.
  __syncthreads();

  // Compute the prefix sum.
  int32_t prefixSum{0}, totalSum{0};
  Scan(smemScan).ExclusiveSum(binCount, prefixSum, totalSum);

  // Update the histogram with the prefix sums.
  if (threadIdx.x < kNumBins) {
    smemHistogram[threadIdx.x] = prefixSum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // Find the last valid bin.
  if (threadIdx.x < kNumBins) {
    int32_t nextPrefixSum =
        threadIdx.x == kNumBins - 1 ? totalSum : smemHistogram[threadIdx.x + 1];
    if (prefixSum < kTopK && nextPrefixSum >= kTopK) {
      smemThresholdBinIdx[0] = threadIdx.x;
    }
  }

  // Clear the counter to store the items for the final phase.
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The threshold bin.
  int32_t thresholdBinIdx = smemThresholdBinIdx[0];

  // Fetch elements one-by-one and populate the shared memory buffers.
  for (int32_t rowIt = rowStart + threadIdx.x; rowIt < rowEnd;
       rowIt += kNumThreadsPerBlock) {
    float logit = logits[rowIdx * stride0 + rowIt * stride1];
    uint16_t idx = extractBinIdx(logit);
    if (idx < thresholdBinIdx) {
      int32_t dstIdx = atomicAdd(&smemHistogram[idx], 1);
      smemLogits[dstIdx] = logit;
      smemIndices[dstIdx] = rowIt;
    } else if (idx == thresholdBinIdx) {
      int32_t dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
      if (dstIdx < kNumFinalItems) {
        smemFinal.items.logits[dstIdx] = logit;
        smemFinal.items.indices[dstIdx] = rowIt;
      }
    }
  }

  // Make sure the elements are in shared memory.
  __syncthreads();

  // The logits of the elements to be sorted in the final pass.
  float finalLogits[kNumFinalItemsPerThread];
  // The indices of the elements to be sorted in the final pass.
  int32_t finalIndices[kNumFinalItemsPerThread];

// Init.
#pragma unroll
  for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
    finalLogits[ii] = -FLT_MAX;
  }

// Read the elements from SMEM.
#pragma unroll
  for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
    int32_t srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
    if (srcIdx < smemFinalDstIdx[0]) {
      finalLogits[ii] = smemFinal.items.logits[srcIdx];
      finalIndices[ii] = smemFinal.items.indices[srcIdx];
    }
  }

  // Make sure the shared memory has been read.
  __syncthreads();

  // Sort the elements.
  FinalSort(smemFinal.finalSort)
      .SortDescendingBlockedToStriped(finalLogits, finalIndices);

  // Copy the data back to the shared memory storage.
  int32_t baseIdx = thresholdBinIdx > 0 ? smemHistogram[thresholdBinIdx - 1] : 0;
#pragma unroll
  for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
    int32_t srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
    int32_t dstIdx = baseIdx + srcIdx;
    if (dstIdx < kTopK) {
      smemLogits[dstIdx] = finalLogits[ii];
      smemIndices[dstIdx] = finalIndices[ii];
    }
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The topK logits.
  float topKLogits[kNumTopKItemsPerThread];
  // The topK indices.
  int32_t topKIndices[kNumTopKItemsPerThread];

// Load from shared memory.
#pragma unroll
  for (int ii = 0; ii < kNumTopKItemsPerThread; ++ii) {
    topKLogits[ii] = smemLogits[ii * kNumThreadsPerBlock + threadIdx.x];
    topKIndices[ii] = smemIndices[ii * kNumThreadsPerBlock + threadIdx.x];
  }

  // Sort the elements.
  TopKSort(smemFinal.topKSort)
      .SortDescendingBlockedToStriped(topKLogits, topKIndices);

// Store to global memory.
#pragma unroll
  for (int ii = 0; ii < kNumTopKItemsPerThread; ++ii) {
    int32_t offset = rowIdx * kTopK + ii * kNumThreadsPerBlock + threadIdx.x;
    outIndices[offset] = topKIndices[ii] - rowStart;
  }
}

void InvokeFastTopK(const float* logits, const int32_t* rowStarts, const int32_t* rowEnds,
                    int32_t* indices, int64_t numRows,
                    int64_t stride0, int64_t stride1, cudaStream_t stream) {
  // Launch the main topK kernel using cudaLaunchKernelEx for PDL (Programmatic Dependent Launch)
  constexpr int kNumThreadsPerBlock = 512;
  
  // Configure launch parameters
  cudaLaunchConfig_t config = {
      dim3(numRows, 1, 1),              // gridDim
      dim3(kNumThreadsPerBlock, 1, 1),  // blockDim
      0,                                // dynamicSmemBytes
      stream,                           // stream
      nullptr,                          // attrs
      0                                 // numAttrs
  };
  
  // Launch kernel using cudaLaunchKernelEx with direct parameter passing
  cudaLaunchKernelEx(&config, topKPerRow<kNumThreadsPerBlock>, logits, rowStarts, rowEnds, indices, stride0, stride1);
}

}  // namespace nvidia
}  // namespace llm_kernels
