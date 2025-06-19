/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "dequant.h"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace dequant {

template <size_t THREAD, size_t LEN>
__global__ void dequant_int4_fp8_kernel(const uint16_t* __restrict__ input, __nv_fp8x4_e4m3* __restrict__ output) {
  for (size_t i = threadIdx.x; i < LEN; i += THREAD) {
    const size_t offset = blockIdx.x * LEN + i;

    uint16_t data = input[offset];

    float4 data4;
    data4.x = static_cast<float>(data & 0xF) - 8;
    data4.y = static_cast<float>((data >> 4) & 0xF) - 8;
    data4.z = static_cast<float>((data >> 8) & 0xF) - 8;
    data4.w = static_cast<float>((data >> 12) & 0xF) - 8;

    output[offset] = static_cast<__nv_fp8x4_e4m3>(data4);
  }
}

void dequant_int4_fp8(cudaStream_t stream, void* output, const void* input, size_t datasize) {
#if defined(ENABLE_COMMON_INT4_FP8_DEQUANT)
  const size_t pack_factor = 2;
  const size_t blocksize = 64;
  const size_t datasize_per_grid = 1024;
  KLLM_KERNEL_CHECK(datasize % (pack_factor * blocksize) == 0);

  datasize /= pack_factor;
  if (datasize <= datasize_per_grid) {
    const size_t gridsize = datasize / blocksize;
    dequant_int4_fp8_kernel<blocksize, blocksize><<<gridsize, blocksize, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(input), reinterpret_cast<__nv_fp8x4_e4m3*>(output));

  } else {
    const size_t gridsize = datasize / datasize_per_grid;
    dequant_int4_fp8_kernel<blocksize, datasize_per_grid><<<gridsize, blocksize, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(input), reinterpret_cast<__nv_fp8x4_e4m3*>(output));
  }
#else
  KLLM_KERNEL_THROW("SM version is lower than 90. skipping dequant kernel.");
#endif
}

}  // namespace dequant
}  // namespace nvidia
}  // namespace llm_kernels