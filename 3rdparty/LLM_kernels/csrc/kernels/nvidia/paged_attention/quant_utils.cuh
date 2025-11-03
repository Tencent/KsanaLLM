/*
 * Adapted from
 * https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fp8/nvidia/quant_utils.cuh
 * and
 * https://github.com/vllm-project/vllm/blob/main/csrc/attention/dtype_fp8.cuh
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "dtype_float32.cuh"
#include "dtype_float16.cuh"
#include "dtype_bfloat16.cuh"
#include "dtype_fp8.cuh"
#include "paged_attention_generic.cuh"
#include "csrc/utils/quant_type.h"

#include <stdint.h>
#include <stdio.h>
#include <assert.h>

namespace llm_kernels {
namespace nvidia {

namespace fp8 {

/* Scaled and vectorized conversions, for data exchange between high and low
   precision domains Convention of the scale in API, e.g: FP8_data =
   Quantization( High_Precision_data / scale ) s.t. Quantize(HP / scale) => FP8
     Dequant(FP8) * scale =>  HP
 */

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(const Tin &x, const float scale,
                                                 const __nv_fp8_interpretation_t fp8_type)
{
    return x;
}

// fp8 -> half
template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, uint8_t>(const uint8_t &a, const float scale,
                                                                        const __nv_fp8_interpretation_t fp8_type)
{
    __half_raw tmp = __nv_cvt_fp8_to_halfraw(a, fp8_type);
    return float_to_half(half_to_float(tmp.x) * scale);
}

// fp8 -> half
template <>
__inline__ __device__ half scaled_vec_conversion<half, uint8_t>(const uint8_t &a, const float scale,
                                                                        const __nv_fp8_interpretation_t fp8_type)
{
    __half_raw tmp = __nv_cvt_fp8_to_halfraw(a, fp8_type);
    uint16_t tmp2 = float_to_half(half_to_float(tmp.x) * scale);
    return *reinterpret_cast<half*>(&tmp2);
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, uint16_t>(const uint16_t &a, const float scale,
                                                                         const __nv_fp8_interpretation_t fp8_type)
{
    union {
        uint16_t u16[2];
        uint32_t u32;
    } tmp;
    __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, fp8_type);
    tmp.u16[0] = float_to_half(half_to_float(res.x) * scale);
    tmp.u16[1] = float_to_half(half_to_float(res.y) * scale);
    return tmp.u32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint32_t>(const uint32_t &a, const float scale,
                                                                   const __nv_fp8_interpretation_t fp8_type)
{
    union {
        uint2    u32x2;
        uint32_t u32[2];
    } tmp;
    tmp.u32[0] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)a, scale, fp8_type);
    tmp.u32[1] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U), scale, fp8_type);
    return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4 scaled_vec_conversion<uint4, uint2>(const uint2 &a, const float scale,
                                                                const __nv_fp8_interpretation_t fp8_type)
{
    union {
        uint4 u64x2;
        uint2 u64[2];
    } tmp;
    tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale, fp8_type);
    tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale, fp8_type);
    return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16 scaled_vec_conversion<__nv_bfloat16, uint8_t>(
    const uint8_t &a, const float scale, const __nv_fp8_interpretation_t fp8_type)
{
    // Note there is no direct convert function from fp8 to bf16.
    // fp8 -> half
    __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
    // half -> float -> bf16
    float tmp = half_to_float(res.x);
    return __float2bfloat16(tmp * scale);
}

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162 scaled_vec_conversion<__nv_bfloat162, uint16_t>(
    const uint16_t &a, const float scale, const __nv_fp8_interpretation_t fp8_type)
{
    __nv_bfloat162 res;
    res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, scale, fp8_type);
    res.y = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U), scale, fp8_type);
    return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t scaled_vec_conversion<bf16_4_t, uint32_t>(const uint32_t &a, const float scale,
                                                                         const __nv_fp8_interpretation_t fp8_type)
{
    bf16_4_t res;
    res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, scale, fp8_type);
    res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U), scale, fp8_type);
    return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t scaled_vec_conversion<bf16_8_t, uint2>(const uint2 &a, const float scale,
                                                                      const __nv_fp8_interpretation_t fp8_type)
{
    bf16_4_t tmp1, tmp2;
    tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale, fp8_type);
    tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale, fp8_type);
    bf16_8_t res;
    res.x = tmp1.x;
    res.y = tmp1.y;
    res.z = tmp2.x;
    res.w = tmp2.y;
    return res;
}

// fp8 -> float
template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(const uint8_t &a, const float scale,
                                                                  const __nv_fp8_interpretation_t fp8_type)
{
    // fp8 -> half
    __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
    uint16_t tmp = res.x;

    // half -> float
    return half_to_float(tmp) * scale;
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2 scaled_vec_conversion<float2, uint16_t>(const uint16_t &a, const float scale,
                                                                     const __nv_fp8_interpretation_t fp8_type)
{
    // fp8x2 -> half2
    uint32_t tmp = scaled_vec_conversion<uint32_t, uint16_t>(a, scale, fp8_type);
    // half2 -> float2
    return half2_to_float2(tmp);
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_ scaled_vec_conversion<Float4_, uint32_t>(const uint32_t &a, const float scale,
                                                                       const __nv_fp8_interpretation_t fp8_type)
{
    Float4_ res;
    res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t)a, scale, fp8_type);
    res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), scale, fp8_type);
    return res;
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_ scaled_vec_conversion<Float8_, uint2>(const uint2 &a, const float scale,
                                                                    const __nv_fp8_interpretation_t fp8_type)
{
    Float4_ tmp1, tmp2;
    tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale, fp8_type);
    tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale, fp8_type);
    Float8_ res;
    res.x = tmp1.x;
    res.y = tmp1.y;
    res.z = tmp2.x;
    res.w = tmp2.y;
    return res;
}

// half -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, uint16_t>(const uint16_t &a, const float scale,
                                                                       const __nv_fp8_interpretation_t fp8_type)
{
    __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(half_to_float(a) / scale, __NV_SATFINITE, fp8_type);
    return (uint8_t)res;
}

// half -> fp8
template<>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, half>(const half& a, const float scale,
                                                                   const __nv_fp8_interpretation_t fp8_type)
{
    __half_raw tmp(a);
    //tmp.x = a;
    __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(half_to_float(tmp.x) / scale, __NV_SATFINITE, fp8_type);
    return (uint8_t)res;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(const __nv_bfloat16 &a, const float scale,
                                                                            const __nv_fp8_interpretation_t fp8_type)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    assert(false);
#else
    __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale, __NV_SATFINITE, fp8_type);
    return (uint8_t)res;
#endif
    __builtin_unreachable();
}

// bf16x4 -> fp8x4
template <>
__inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, bf16_4_t>(const bf16_4_t &a, const float scale,
                                                                          const __nv_fp8_interpretation_t fp8_type)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    assert(false);
#else
    union {
        uint8_t  u8[4];
        uint32_t u32;
    } tmp;
    
    float inv_scale = 1.0f / scale;
    float2 f01 = __bfloat1622float2(a.x);
    float2 f23 = __bfloat1622float2(a.y);
    
    tmp.u8[0] = (uint8_t)__nv_cvt_float_to_fp8(f01.x * inv_scale, __NV_SATFINITE, fp8_type);
    tmp.u8[1] = (uint8_t)__nv_cvt_float_to_fp8(f01.y * inv_scale, __NV_SATFINITE, fp8_type);
    tmp.u8[2] = (uint8_t)__nv_cvt_float_to_fp8(f23.x * inv_scale, __NV_SATFINITE, fp8_type);
    tmp.u8[3] = (uint8_t)__nv_cvt_float_to_fp8(f23.y * inv_scale, __NV_SATFINITE, fp8_type);

    return tmp.u32;
#endif
    __builtin_unreachable();
}

// bf16x8 -> fp8x8 
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, bf16_8_t>(const bf16_8_t &a, const float scale,
                                                                     const __nv_fp8_interpretation_t fp8_type)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    assert(false);
#else
    union {
        uint8_t  u8[8];
        uint2    u64;
    } tmp;
    
    float inv_scale = 1.0f / scale;
    float2 f01 = __bfloat1622float2(a.x);
    float2 f23 = __bfloat1622float2(a.y);
    float2 f45 = __bfloat1622float2(a.z);
    float2 f67 = __bfloat1622float2(a.w);
    
    tmp.u8[0] = (uint8_t)__nv_cvt_float_to_fp8(f01.x * inv_scale, __NV_SATFINITE, fp8_type);
    tmp.u8[1] = (uint8_t)__nv_cvt_float_to_fp8(f01.y * inv_scale, __NV_SATFINITE, fp8_type);
    tmp.u8[2] = (uint8_t)__nv_cvt_float_to_fp8(f23.x * inv_scale, __NV_SATFINITE, fp8_type);
    tmp.u8[3] = (uint8_t)__nv_cvt_float_to_fp8(f23.y * inv_scale, __NV_SATFINITE, fp8_type);
    tmp.u8[4] = (uint8_t)__nv_cvt_float_to_fp8(f45.x * inv_scale, __NV_SATFINITE, fp8_type);
    tmp.u8[5] = (uint8_t)__nv_cvt_float_to_fp8(f45.y * inv_scale, __NV_SATFINITE, fp8_type);
    tmp.u8[6] = (uint8_t)__nv_cvt_float_to_fp8(f67.x * inv_scale, __NV_SATFINITE, fp8_type);
    tmp.u8[7] = (uint8_t)__nv_cvt_float_to_fp8(f67.y * inv_scale, __NV_SATFINITE, fp8_type);
    
    return tmp.u64;
#endif
    __builtin_unreachable();
}

// float -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(const float &a, const float scale,
                                                                    const __nv_fp8_interpretation_t fp8_type)
{
    __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, fp8_type);
    return (uint8_t)res;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4 scaled_vec_conversion<float4, uint32_t>(const uint32_t &a, const float scale,
                                                                     const __nv_fp8_interpretation_t fp8_type)
{
    Float4_ tmp = scaled_vec_conversion<Float4_, uint32_t>(a, scale, fp8_type);
    float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
    return res;
}

template <typename Tout, typename Tin, llm_kernels::utils::KVCacheType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin &x, const float scale) {
    if constexpr (kv_dt == llm_kernels::utils::KVCacheType::kFp8E4M3) {
        return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E4M3);
    } else if constexpr (kv_dt == llm_kernels::utils::KVCacheType::kFp8E5M2) {
        return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E5M2);
    }
    assert(false);
    __builtin_unreachable();
}

} // namespace fp8

}  // namespace nvidia
}  // namespace llm_kernels
