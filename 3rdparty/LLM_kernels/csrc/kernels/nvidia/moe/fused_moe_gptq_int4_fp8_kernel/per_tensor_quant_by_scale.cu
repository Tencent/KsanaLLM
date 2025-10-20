/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/moe/fused_moe_gptq_int4_fp8_kernel/per_tensor_quant_by_scale.h"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/common/envUtils.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

namespace llm_kernels {
namespace nvidia {

template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
  return converter(input);
}

template <class InputT, class OutputT, int THREADS_PER_BLOCK, int ELE_PER_THREAD, int ELEM_PER_PACK>
__global__ void per_tensor_quant_by_scale_kernel(OutputT* output, const InputT* input, const InputT* scales) {
  using InputElem = cutlass::Array<InputT, ELEM_PER_PACK>;
  using ComputeElem = cutlass::Array<float, ELEM_PER_PACK>;
  using OutputElem = cutlass::Array<OutputT, ELEM_PER_PACK>;
  constexpr size_t groups = ELE_PER_THREAD / ELEM_PER_PACK;

  register const float scale = static_cast<float>(*scales);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (size_t i = threadIdx.x; i < groups; i += THREADS_PER_BLOCK) {
    const size_t offset = blockIdx.x * ELE_PER_THREAD + i * ELEM_PER_PACK;

    const InputElem* in_ptr = reinterpret_cast<const InputElem*>(input + offset);
    OutputElem* out_ptr = reinterpret_cast<OutputElem*>(output + offset);
    auto inter_val = arrayConvert<InputElem, ComputeElem>(*in_ptr);
    auto result_val = inter_val * scale;
    (*out_ptr) = arrayConvert<ComputeElem, OutputElem>(result_val);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <class InputT, class OutputT>
void per_tensor_quant_by_scale_launcher(void* output, const void* input, const void* scales, const size_t num_elements,
                                        cudaStream_t stream) {
#if defined(ENABLE_FUSED_MOE_GPTQ_INT4_FP8)
  constexpr int64_t kElemPerPack = 8;
  constexpr size_t kThreadsPerBlock = 128;
  constexpr size_t kElemPerThread = 1024;
  KLLM_KERNEL_CHECK(num_elements % kElemPerThread == 0);

  const size_t blocks = num_elements / kElemPerThread;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = kThreadsPerBlock;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = true;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config,
                     &per_tensor_quant_by_scale_kernel<InputT, OutputT, kThreadsPerBlock, kElemPerThread, kElemPerPack>,
                     reinterpret_cast<OutputT*>(output), reinterpret_cast<const InputT*>(input),
                     reinterpret_cast<const InputT*>(scales));

#else
  KLLM_KERNEL_THROW("SM version is lower than 90. skipping per_tensor_quant_by_scale kernel.");
#endif
}

#define PerTokenQuantByScaleLauncher(InputT, OutputT)                \
  template void per_tensor_quant_by_scale_launcher<InputT, OutputT>( \
      void* output, const void* input, const void* scales, const size_t num_elements, cudaStream_t stream);
PerTokenQuantByScaleLauncher(half, __nv_fp8_e4m3);
PerTokenQuantByScaleLauncher(__nv_bfloat16, __nv_fp8_e4m3);
PerTokenQuantByScaleLauncher(float, __nv_fp8_e4m3);
#undef PerTokenQuantByScaleLauncher

}  // namespace nvidia
}  // namespace llm_kernels