/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "dequant.h"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace dequant {

template <size_t THREAD, size_t LEN>
__global__ void dequant_uint4_fp8_kernel(const uint16_t* __restrict__ input, __nv_fp8x4_e4m3* __restrict__ output) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

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

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void dequant_uint4_fp8_launcher(cudaStream_t stream, void* output, const void* input, size_t datasize) {
#if defined(ENABLE_FUSED_MOE_GPTQ_INT4_FP8)
  constexpr size_t kPackFactor = 2;
  constexpr size_t kBlockSize = 64;
  constexpr size_t kDatasizePerGrid = 1024;
  KLLM_KERNEL_CHECK(datasize % (kPackFactor * kBlockSize) == 0);

  datasize = datasize / kPackFactor;
  size_t gridsize = datasize / kDatasizePerGrid;
  auto fn = &dequant_uint4_fp8_kernel<kBlockSize, kDatasizePerGrid>;
  if (datasize <= kDatasizePerGrid) {
    gridsize = datasize / kBlockSize;
    fn = &dequant_uint4_fp8_kernel<kBlockSize, kBlockSize>;
  }

  cudaLaunchConfig_t config;
  config.gridDim = gridsize;
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = true;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, fn, reinterpret_cast<const uint16_t*>(input), reinterpret_cast<__nv_fp8x4_e4m3*>(output));

#else
  KLLM_KERNEL_THROW("SM version is lower than 90. skipping dequant kernel.");
#endif
}

template <size_t THREAD, size_t LEN>
__global__ void dequant_int4_fp8_kernel(const uint16_t* __restrict__ input, __nv_fp8x4_e4m3* __restrict__ output) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (size_t i = threadIdx.x; i < LEN; i += THREAD) {
    const size_t offset = blockIdx.x * LEN + i;

    uint16_t data = input[offset];
    uint32_t n0 = data & 0xF;
    uint32_t n1 = (data >> 4) & 0xF;
    uint32_t n2 = (data >> 8) & 0xF;
    uint32_t n3 = (data >> 12) & 0xF;

    float4 data4;
    data4.x = static_cast<float>(static_cast<int32_t>(n0 << 28) >> 28);
    data4.y = static_cast<float>(static_cast<int32_t>(n1 << 28) >> 28);
    data4.z = static_cast<float>(static_cast<int32_t>(n2 << 28) >> 28);
    data4.w = static_cast<float>(static_cast<int32_t>(n3 << 28) >> 28);

    output[offset] = static_cast<__nv_fp8x4_e4m3>(data4);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void dequant_int4_fp8_launcher(cudaStream_t stream, void* output, const void* input, size_t datasize) {
#if defined(ENABLE_FUSED_MOE_GPTQ_INT4_FP8)
  constexpr size_t kPackFactor = 2;
  constexpr size_t kBlockSize = 64;
  constexpr size_t kDatasizePerGrid = 1024;
  KLLM_KERNEL_CHECK(datasize % (kPackFactor * kBlockSize) == 0);

  datasize = datasize / kPackFactor;
  size_t gridsize = datasize / kDatasizePerGrid;
  auto fn = &dequant_int4_fp8_kernel<kBlockSize, kDatasizePerGrid>;
  if (datasize <= kDatasizePerGrid) {
    gridsize = datasize / kBlockSize;
    fn = &dequant_int4_fp8_kernel<kBlockSize, kBlockSize>;
  }

  cudaLaunchConfig_t config;
  config.gridDim = gridsize;
  config.blockDim = kBlockSize;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = true;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, fn, reinterpret_cast<const uint16_t*>(input), reinterpret_cast<__nv_fp8x4_e4m3*>(output));

#else
  KLLM_KERNEL_THROW("SM version is lower than 90. skipping dequant kernel.");
#endif
}

}  // namespace dequant
}  // namespace nvidia
}  // namespace llm_kernels