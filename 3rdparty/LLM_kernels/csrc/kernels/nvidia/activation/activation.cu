/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "csrc/kernels/nvidia/activation/activation.h"

#include "csrc/utils/nvidia/cuda_bf16_fallbacks.cuh"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

__forceinline__ __device__ float CopySignFPos(float a, float b) {
  float r;
  r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
  return r;
}

__inline__ __device__ float TanhOpt(float x) {
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
  float r;
  asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  return r;
#else
  const float exp_val = -1.f * fabs(2 * x);
  return CopySignFPos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template <typename T>
struct GeluActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T& val) {
    // approximate calculation formula
    const float cdf = 0.5f * (1.0f + TanhOpt((0.7978845608028654f * (val + 0.044715f * val * val * val))));
    return val * cdf;
  }
};

template <>
struct GeluActivation<half2> {
  using return_type = half2;
  static __device__ __forceinline__ half2 apply(const half2& val) {
    half2 val_pow3 = __hmul2(val, __hmul2(val, val));
    float2 tmp_pow = __half22float2(val_pow3);
    float2 tmp = __half22float2(val);

    // approximate calculation formula
    tmp.x = 0.5f * (1.0f + TanhOpt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
    tmp.y = 0.5f * (1.0f + TanhOpt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
    return __hmul2(val, __float22half2_rn(tmp));
  }
};

#ifdef ENABLE_BF16
template <>
struct GeluActivation<__nv_bfloat162> {
  using return_type = __nv_bfloat162;
  static __device__ __forceinline__ __nv_bfloat162 apply(const __nv_bfloat162& val) {
    __nv_bfloat162 val_pow3 = bf16hmul2(val, bf16hmul2(val, val));
    float2 tmp_pow = bf1622float2(val_pow3);
    float2 tmp = bf1622float2(val);

    // approximate calculation formula
    tmp.x = 0.5f * (1.0f + TanhOpt((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
    tmp.y = 0.5f * (1.0f + TanhOpt((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
    return bf16hmul2(val, __floats2bfloat162_rn(tmp.x, tmp.y));
  }
};
#endif

template <typename T>
struct ReluActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T& val) {
    return val > static_cast<T>(0.0f) ? val : static_cast<T>(0.0f);
  }
};

template <>
struct ReluActivation<half2> {
  using return_type = half2;
  static __device__ __forceinline__ half2 apply(const half2& val) {
    const half zero_half = static_cast<half>(0.0f);
    return make_half2(val.x > zero_half ? val.x : zero_half, val.y > zero_half ? val.y : zero_half);
  }
};

#ifdef ENABLE_BF16
template <>
struct ReluActivation<__nv_bfloat162> {
  using return_type = __nv_bfloat162;
  static __device__ __forceinline__ __nv_bfloat162 apply(const __nv_bfloat162& val) {
    const __nv_bfloat16 zero_bf16 = static_cast<__nv_bfloat16>(0.0f);
    return make_bfloat162(val.x > zero_bf16 ? val.x : zero_bf16, val.y > zero_bf16 ? val.y : zero_bf16);
  }
};
#endif

template <typename T>
struct SiluActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T& val) { return (T)((float)val / (1.0f + __expf((float)-val))); }
};

template <>
struct SiluActivation<half2> {
  using return_type = float2;
  static __device__ __forceinline__ float2 apply(const half2& val) {
    return make_float2(SiluActivation<float>::apply(val.x), SiluActivation<float>::apply(val.y));
  }
};

#ifdef ENABLE_BF16
template <>
struct SiluActivation<__nv_bfloat162> {
  using return_type = float2;
  static __device__ __forceinline__ float2 apply(const __nv_bfloat162& val) {
    return make_float2(SiluActivation<float>::apply(val.x), SiluActivation<float>::apply(val.y));
  }
};
#endif  // ENABLE_BF16

template <typename T>
struct IdentityActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T& val) { return val; }
};

template <template <typename T> class Activation, typename T>
__global__ void RowBasedActivationKernel(T* out, const T* __restrict input, int32_t m, int32_t n, int32_t stride,
                                         int32_t for_range, int32_t packed_elems) {
  using ActType = typename Activation<T>::return_type;

  int32_t row_id = blockIdx.x;
  int32_t output_col_num = n / 2;
  for (int32_t offset = 0; offset < for_range && threadIdx.x * for_range + offset < output_col_num; ++offset) {
    int32_t col_id = threadIdx.x * for_range + offset;
    if (col_id >= output_col_num) {
      return;
    }
    int32_t input_idx = row_id * n + col_id;
    T val = input[input_idx];

    T gated_val = input[input_idx + stride];

    val = CastCudaDataType<T>(Activation<T>::apply(val) * CastCudaDataType<ActType>(gated_val));

    out[row_id * output_col_num + col_id] = val;
  }
}

template <template <typename T> class Activation, typename T>
void InvokeRowBasedActivation(T* out, const T* input, const int32_t m, const int32_t n, cudaStream_t& stream) {
  using PT = typename PackTypeAlign<T>::type;
  int32_t packed_elems = ElemsNum<PT>::value;
  int32_t divided_column_num = n / 2;
  if (divided_column_num % packed_elems != 0) {
    packed_elems = 1;
  }

  const size_t BLOCK_SIZE = 256;
  dim3 grid(m), block(BLOCK_SIZE);
  size_t for_range = ceil(static_cast<float>(divided_column_num) / packed_elems / BLOCK_SIZE);
  if (packed_elems == 1) {  // not using PT
    RowBasedActivationKernel<Activation, T>
        <<<grid, block, 0, stream>>>(reinterpret_cast<T*>(out), reinterpret_cast<const T*>(input), m, n / packed_elems,
                                     divided_column_num / packed_elems, for_range, packed_elems);
  } else {
    RowBasedActivationKernel<Activation, PT>
        <<<grid, block, 0, stream>>>(reinterpret_cast<PT*>(out), reinterpret_cast<const PT*>(input), m,
                                     n / packed_elems, divided_column_num / packed_elems, for_range, packed_elems);
  }
}

#define INSTANTIATE_ROW_BASED_ACTIVATION(Activation, T)                                                            \
  template void InvokeRowBasedActivation<Activation, T>(T * out, const T* input, const int32_t m, const int32_t n, \
                                                            cudaStream_t& stream);

INSTANTIATE_ROW_BASED_ACTIVATION(SiluActivation, float);
INSTANTIATE_ROW_BASED_ACTIVATION(SiluActivation, half);
#ifdef ENABLE_BF16
INSTANTIATE_ROW_BASED_ACTIVATION(SiluActivation, __nv_bfloat16);
#endif

template <template <typename T> class Activation, typename T, typename BT>
__global__ void InvokeGenericActivationKernel(T* out, const BT* __restrict bias, const T* __restrict gated_weights,
                                              const BT* __restrict gated_bias, const int32_t* __restrict ia3_tasks,
                                              const T* __restrict ia3_weights, const int32_t int8_mode,
                                              const float* __restrict activation_in,
                                              const float* __restrict activation_out,
                                              const int32_t* __restrict padding_offset, const int32_t seq_len,
                                              int32_t m, int32_t n) {
  constexpr size_t packed_elems = ElemsNum<T>::value;

  const bool with_bias = bias != nullptr;
  const bool with_gate = gated_weights != nullptr;
  const bool with_ia3 = ia3_tasks != nullptr;

  using ActType = typename Activation<T>::return_type;
  using FloatType = typename PackType<float, packed_elems>::type;
  using PackedInt8Type = typename PackType<int8_t, packed_elems>::type;

  for (int32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
    T val;
    if (int8_mode == 2) {
      val = CastCudaDataType<T>(CastCudaDataType<FloatType>(reinterpret_cast<PackedInt8Type*>(out)[id]) *
                                activation_in[0]);
    } else {
      val = out[id];
    }

    T gated_val;
    if (with_gate) {
      gated_val = gated_weights[id];
    }

    if (with_bias) {
      const T reg_bias = static_cast<T>(bias[id % n]);
      val = val + reg_bias;

      if (with_gate) {
        const T reg_gated_bias = static_cast<T>(gated_bias[id % n]);
        gated_val = gated_val + reg_gated_bias;
      }
    }

    if (with_gate) {
      val = CastCudaDataType<T>(Activation<T>::apply(val) * CastCudaDataType<ActType>(gated_val));
    } else {
      val = CastCudaDataType<T>(Activation<T>::apply(val));
    }

    if (with_ia3) {
      const int32_t word_id = id / n;
      const int32_t offset = padding_offset == nullptr ? 0 : padding_offset[word_id];
      const int32_t batch_id = (word_id + offset) / seq_len;
      const int32_t task = ia3_tasks[batch_id];
      val = val * ia3_weights[task * n + (id % n)];
    }

    if (int8_mode != 2) {
      out[id] = val;
    } else {
      reinterpret_cast<PackedInt8Type*>(out)[id] =
          CastCudaDataType<PackedInt8Type>(CastCudaDataType<FloatType>(val) * activation_out[0]);
    }
  }
}

template <template <typename T> class Activation, typename T, typename BT>
void InvokeGenericActivation(T* out, const BT* bias, const T* gated_weights, const BT* gated_bias,
                             const int32_t* ia3_tasks, const T* ia3_weights, const int32_t m, const int32_t n,
                             const int32_t int8_mode, const float* activation_in, const float* activation_out,
                             const int32_t* padding_offset, const int32_t seq_len, cudaStream_t& stream) {
  using PT = typename PackTypeAlign<T>::type;
  constexpr int32_t packed_elems = ElemsNum<PT>::value;
  using PBT = typename PackType<BT, packed_elems>::type;

  dim3 block, grid;
  if (n / 4 / packed_elems <= 1024) {
    block.x = n / 4 / packed_elems;
    grid.x = m;
  } else {
    block.x = 1024;
    grid.x = ceil(m * n / 1024.);
  }
  InvokeGenericActivationKernel<Activation><<<grid, block, 0, stream>>>(
      reinterpret_cast<PT*>(out), reinterpret_cast<const PBT*>(bias), reinterpret_cast<const PT*>(gated_weights),
      reinterpret_cast<const PBT*>(gated_bias), ia3_tasks, reinterpret_cast<const PT*>(ia3_weights), int8_mode,
      activation_in, activation_out, padding_offset, seq_len, m, n / packed_elems);
}

#define INSTANTIATE_GENERIC_ACTIVATION(Activation, T, BT)                                                          \
  template void InvokeGenericActivation<Activation, T, BT>(                                                        \
      T * out, const BT* bias, const T* gated_weights, const BT* gated_bias, const int32_t* ia3_tasks,             \
      const T* ia3_weights, const int32_t m, const int32_t n, const int32_t int8_mode, const float* activation_in, \
      const float* activation_out, const int32_t* padding_offset, const int32_t seq_len, cudaStream_t& stream);

INSTANTIATE_GENERIC_ACTIVATION(GeluActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(GeluActivation, half, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(GeluActivation, __nv_bfloat16, __nv_bfloat16);
#endif

INSTANTIATE_GENERIC_ACTIVATION(ReluActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(ReluActivation, half, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(ReluActivation, __nv_bfloat16, __nv_bfloat16);
#endif

INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, half, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(SiluActivation, __nv_bfloat16, __nv_bfloat16);
#endif

INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, float, float);
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, half, half);
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, float, half);
#ifdef ENABLE_BF16
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, __nv_bfloat16, __nv_bfloat16);
INSTANTIATE_GENERIC_ACTIVATION(IdentityActivation, float, __nv_bfloat16);
#endif
#undef INSTANCIATE_GENERIC_ACTIVATION

template <typename T>
__global__ void AddBiasTanhKernel(T* out, const T* __restrict bias, int32_t m, int32_t n) {
  for (int32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
    T val = out[id];
    if (bias != nullptr) {
      val = val + ldg(&bias[id % n]);
    }
    out[id] = tanhf(val);
  }
}

template <>
__global__ void AddBiasTanhKernel(half* out, const half* __restrict bias, int32_t m, int32_t n) {
  half2* out_ptr = (half2*)out;
  const half2* bias_ptr = (half2*)bias;

  for (int32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
    half2 val = out_ptr[id];
    if (bias != nullptr) {
      val = val + __ldg(&bias_ptr[id % n]);
    }
    val.x = tanhf(val.x);
    val.y = tanhf(val.y);
    out_ptr[id] = val;
  }
}

#ifdef ENABLE_BF16
template <>
__global__ void AddBiasTanhKernel(__nv_bfloat16* out, const __nv_bfloat16* __restrict bias, int32_t m, int32_t n) {
  __nv_bfloat162* out_ptr = (__nv_bfloat162*)out;
  const __nv_bfloat162* bias_ptr = (__nv_bfloat162*)bias;

  for (int32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < m * n; id += blockDim.x * gridDim.x) {
    __nv_bfloat162 val = out_ptr[id];
    if (bias != nullptr) {
      val = bf16hadd2(val, ldg(&bias_ptr[id % n]));
    }
    val.x = tanhf(val.x);
    val.y = tanhf(val.y);
    out_ptr[id] = val;
  }
}
#endif

template <typename T>
void InvokeAddBiasTanh(T* out, const T* bias, const int32_t m, const int32_t n, cudaStream_t& stream) {
  // 1 for fp32, 2 for fp16 and bf16
  const int32_t data_type_factor = 4 / sizeof(T);
  constexpr int block_threads_num = 1024;
  dim3 block, grid;
  if (n / 4 / data_type_factor <= block_threads_num) {
    block.x = n / 4 / data_type_factor;
    grid.x = m;
  } else {
    block.x = block_threads_num;
    grid.x = ceil(m * n / static_cast<float>(block_threads_num));
  }
  AddBiasTanhKernel<T><<<grid, block, 0, stream>>>(out, bias, m, n / data_type_factor);
}

template void InvokeAddBiasTanh(float* out, const float* bias, const int32_t m, const int32_t n, cudaStream_t& stream);
template void InvokeAddBiasTanh(half* out, const half* bias, const int32_t m, const int32_t n, cudaStream_t& stream);
#ifdef ENABLE_BF16
template void InvokeAddBiasTanh(__nv_bfloat16* out, const __nv_bfloat16* bias, const int32_t m, const int32_t n,
                                cudaStream_t& stream);
#endif

template <typename T2, int32_t N>
__global__ void AddBiasGeluV2Kernel(T2* out, const T2* __restrict bias, const int32_t* ia3_tasks, const T2* ia3_weights,
                                    const int32_t size, const int32_t* padding_offset, const int32_t seq_len) {
  const bool with_ia3 = ia3_tasks != nullptr;
  for (int32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x) {
    T2 val = out[id];
    if (bias != nullptr) {
      T2 reg_bias = ldg(&bias[id % N]);
      val = hadd2(val, reg_bias);
    }
    val = GeluActivation<T2>::apply(val);
    if (with_ia3) {
      const int32_t word_id = id / N;
      const int32_t offset = padding_offset == nullptr ? 0 : padding_offset[word_id];
      const int32_t batch_id = (word_id + offset) / seq_len;
      const int32_t task = ia3_tasks[batch_id];
      val = val * ia3_weights[task * N + (id % N)];
    }
    out[id] = val;
  }
}

template <typename T2, int32_t N, int32_t ELEMENT_PER_ROUND>
__global__ void AddBiasGeluV3Kernel(T2* out, const T2* __restrict bias, const int32_t* ia3_tasks, const T2* ia3_weights,
                                    const int32_t size, const int32_t* padding_offset, const int32_t seq_len) {
  const bool with_ia3 = ia3_tasks != nullptr;
  T2 buffer[ELEMENT_PER_ROUND];
  T2 tmp_bias[ELEMENT_PER_ROUND];
  for (int32_t id = blockIdx.x * blockDim.x * ELEMENT_PER_ROUND + threadIdx.x * ELEMENT_PER_ROUND; id < size;
       id += blockDim.x * gridDim.x * ELEMENT_PER_ROUND) {
#pragma unroll
    for (int32_t i = 0; i < ELEMENT_PER_ROUND; i++) {
      buffer[i] = out[id + i];
      if (bias != nullptr) {
        tmp_bias[i] = ldg(&bias[(id + i) % N]);
      }
    }
#pragma unroll
    for (int32_t i = 0; i < ELEMENT_PER_ROUND; i++) {
      if (bias != nullptr) {
        buffer[i] = hadd2(buffer[i], tmp_bias[i]);
      }
      buffer[i] = GeluActivation<T2>::apply(buffer[i]);
      if (with_ia3) {
        const int32_t word_id = (id + i) / N;
        const int32_t offset = padding_offset == nullptr ? 0 : padding_offset[word_id];
        const int32_t batch_id = (word_id + offset) / seq_len;
        const int32_t task = ia3_tasks[batch_id];
        buffer[i] = buffer[i] * ia3_weights[task * N + ((id + i) % N)];
      }
      out[id + i] = buffer[i];
    }
  }
}

#define ADD_BIAS_GELU(HALF_N, ELEMENT_PER_ROUND)                                                        \
  case HALF_N:                                                                                          \
    if (ELEMENT_PER_ROUND > 1) {                                                                        \
      grid.x = grid.x / ELEMENT_PER_ROUND;                                                              \
      AddBiasGeluV3Kernel<T2, HALF_N, ELEMENT_PER_ROUND><<<grid, block, 0, stream>>>(                   \
          (T2*)out, (const T2*)bias, ia3_tasks, (T2*)ia3_weights, m * half_n, padding_offset, seq_len); \
    } else {                                                                                            \
      AddBiasGeluV2Kernel<T2, HALF_N><<<grid, block, 0, stream>>>(                                      \
          (T2*)out, (const T2*)bias, ia3_tasks, (T2*)ia3_weights, m * half_n, padding_offset, seq_len); \
    }                                                                                                   \
    break;

template <typename T>
void InvokeAddBiasGeluV2(T* out, const T* bias, const int32_t* ia3_tasks, const T* ia3_weights,
                         const int32_t* padding_offset, const int32_t seq_len, const int32_t m, const int32_t n,
                         cudaStream_t& stream) {
  if (n % 2 == 0 && sizeof(T) == 2) {
    const int32_t half_n = n / 2;
    dim3 block, grid;
    block.x = std::min(half_n, 512);
    grid.x = (m * half_n + (block.x - 1)) / block.x;
    using T2 = typename TypeConverter<T>::Type;

    if (grid.x >= 512) {
      switch (half_n) {
        ADD_BIAS_GELU(256, 1)
        ADD_BIAS_GELU(512, 1)
        ADD_BIAS_GELU(1024, 1)
        ADD_BIAS_GELU(1536, 1)
        ADD_BIAS_GELU(2048, 1)
        ADD_BIAS_GELU(4096, 2)
        ADD_BIAS_GELU(8192, 2)
        ADD_BIAS_GELU(16384, 2)
        ADD_BIAS_GELU(24576, 2)
        ADD_BIAS_GELU(40960, 4)
        default:
          InvokeGenericActivation<GeluActivation>(out, bias, (T*)nullptr, (T*)nullptr, ia3_tasks, ia3_weights, m, n, 0,
                                                  (float*)nullptr, (float*)nullptr, padding_offset, seq_len, stream);
          break;
      }
    } else {
      switch (half_n) {
        ADD_BIAS_GELU(256, 1)
        ADD_BIAS_GELU(512, 1)
        ADD_BIAS_GELU(1024, 1)
        ADD_BIAS_GELU(1536, 1)
        ADD_BIAS_GELU(2048, 1)
        ADD_BIAS_GELU(4096, 1)
        ADD_BIAS_GELU(8192, 2)
        ADD_BIAS_GELU(16384, 2)
        ADD_BIAS_GELU(24576, 2)
        ADD_BIAS_GELU(40960, 2)
        default:
          InvokeGenericActivation<GeluActivation>(out, bias, (T*)nullptr, (T*)nullptr, ia3_tasks, ia3_weights, m, n, 0,
                                                  (float*)nullptr, (float*)nullptr, padding_offset, seq_len, stream);
          break;
      }
    }
  } else {
    InvokeGenericActivation<GeluActivation>(out, bias, (T*)nullptr, (T*)nullptr, ia3_tasks, ia3_weights, m, n, 0,
                                            (float*)nullptr, (float*)nullptr, padding_offset, seq_len, stream);
  }
}

#undef ADD_BIAS_GELU

template void InvokeAddBiasGeluV2(float* out, const float* bias, const int32_t* ia3_tasks, const float* ia3_weights,
                                  const int32_t* padding_offset, const int32_t seq_len, const int32_t m,
                                  const int32_t n, cudaStream_t& stream);
template void InvokeAddBiasGeluV2(half* out, const half* bias, const int32_t* ia3_tasks, const half* ia3_weights,
                                  const int32_t* padding_offset, const int32_t seq_len, const int32_t m,
                                  const int32_t n, cudaStream_t& stream);
#ifdef ENABLE_BF16
template void InvokeAddBiasGeluV2(__nv_bfloat16* out, const __nv_bfloat16* bias, const int32_t* ia3_tasks,
                                  const __nv_bfloat16* ia3_weights, const int32_t* padding_offset,
                                  const int32_t seq_len, const int32_t m, const int32_t n, cudaStream_t& stream);
#endif  // ENABLE_BF16

template <typename T>
__global__ void InvokeSigmoidKernel(T* data, const int32_t size, const float scale) {
  const int32_t index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) {
    float val = CastCudaDataType<float>(data[index]);
    val = 1.0f / (1.0f + exp(-val)) * scale;
    data[index] = T(val);
  }
}

template <typename T>
__global__ void InvokeOptiSigmoidKernel(T* data, const int32_t size, const float scale) {
  const int32_t index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size / 2) {
    T val = data[index];
    float2 val_float2 = CastCudaDataType<float2>(val);
    val_float2.x = 1.0f / (1.0f + exp(-val_float2.x)) * scale;
    val_float2.y = 1.0f / (1.0f + exp(-val_float2.y)) * scale;
    data[index] = CastCudaDataType<T>(val_float2);
  }
}

template <typename T>
void InvokeSigmoid(T* data, const int32_t size, const float scale, cudaStream_t& stream) {
  constexpr int block_threads_num = 128;
  constexpr int grid_blocks_num = 256;
  if (std::is_same<T, float>::value || (size % 2 != 0)) {
    dim3 block(block_threads_num);
    dim3 grid((size + block_threads_num - 1) / block_threads_num);
    InvokeSigmoidKernel<<<grid, block, 0, stream>>>(data, size, scale);
  } else {
    // NOTE(karlluo): each instrinct can handle two elements, so we just need half blocks num
    dim3 block(block_threads_num);
    dim3 grid((size + grid_blocks_num - 1) / grid_blocks_num);
#ifdef ENABLE_BF16
    if (std::is_same<T, __nv_bfloat16>::value) {
      InvokeOptiSigmoidKernel<<<grid, block, 0, stream>>>((__nv_bfloat162*)data, size, scale);
    } else {
#endif
      InvokeOptiSigmoidKernel<<<grid, block, 0, stream>>>((half2*)data, size, scale);
#ifdef ENABLE_BF16
    }
#endif
  }
}

template void InvokeSigmoid(float* data, const int32_t size, const float scale, cudaStream_t& stream);
template void InvokeSigmoid(half* data, const int32_t size, const float scale, cudaStream_t& stream);
#ifdef ENABLE_BF16
template void InvokeSigmoid(__nv_bfloat16* data, const int32_t size, const float scale, cudaStream_t& stream);
#endif  // ENABLE_BF16

}  // namespace nvidia
}  // namespace llm_kernels
