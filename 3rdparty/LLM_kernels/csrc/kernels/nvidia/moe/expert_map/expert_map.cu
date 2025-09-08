/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/moe/expert_map/expert_map.h"

#include "csrc/utils/nvidia/assert.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace moe {

__global__ void ExpertMapKernel(int32_t* data, size_t data_size, int32_t start_expert, int32_t end_expert) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < data_size) {
    if (start_expert <= data[idx] && data[idx] < end_expert) {
      data[idx] -= start_expert;
    } else {
      data[idx] = -1;
    }
  }
}

ExpertMap::ExpertMap(size_t ep_size, size_t ep_rank, size_t expert_num) {
  KLLM_KERNEL_CHECK_WITH_INFO(expert_num % ep_size == 0, "expert_num must be divisible by ep_size");
  size_t expert_per_rank = expert_num / ep_size;
  start_expert_ = ep_rank * expert_per_rank;
  end_expert_ = start_expert_ + expert_per_rank;
}

void ExpertMap::InvokeExpertMapInplace(int32_t* data, size_t data_size, cudaStream_t stream) {
  const int threads_per_block = 128;
  const int blocks = (data_size + threads_per_block - 1) / threads_per_block;
  ExpertMapKernel<<<blocks, threads_per_block, 0, stream>>>(data, data_size, start_expert_, end_expert_);
}

}  // namespace moe
}  // namespace nvidia
}  // namespace llm_kernels