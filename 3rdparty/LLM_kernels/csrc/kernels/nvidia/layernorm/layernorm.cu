/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "layernorm.h"

#include <cub/cub.cuh>

#include "csrc/kernels/nvidia/common/reduce_kernel_utils.cuh"
#include "csrc/utils/nvidia/cuda_type_utils.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

#include <flashinfer/norm.cuh>

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

// * Note that typename T is half2 or bfloat2 type
template <typename T, bool IS_OUTPUT, bool IS_BIAS, int32_t RESIDUAL_NUM, bool IS_BETA, int32_t UNROLL_FACTOR>
__global__ void InvokeAddBiasResLayerNormOptKernel(T* normed_output, T* output, const T* __restrict input,
                                                   const T* __restrict bias, const T* __restrict residual1,
                                                   const T* __restrict residual2, const T* __restrict gamma,
                                                   const T* __restrict beta, const float layernorm_eps, int32_t m,
                                                   int32_t n, const float* scale_inter, const float* scale_out,
                                                   const float* scale, float* dynamic_scale, const int32_t int8_mode) {
  extern __shared__ __align__(sizeof(float)) char _shmem[];  // Align on largest type
  T* shmem = reinterpret_cast<T*>(_shmem);

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  using Int8_Packed_T = typename PackType<int8_t, ElemsNum<T>::value>::type;
  using Int32_Packed_T = typename PackType<int32_t, ElemsNum<T>::value>::type;
  using Float_Packed_T = typename PackType<float, ElemsNum<T>::value>::type;
  using Scalar_T = typename PackType<T, 1>::type;

  const bool scale_input = int8_mode == 2 && scale_inter != nullptr;
  const bool dynamic_scaling = dynamic_scale != nullptr;

  T local_sum = CastCudaDataType<T>(0.0f);

  const Float_Packed_T scale_from_int =
      CastCudaDataType<Float_Packed_T>(scale_input ? (*scale_inter) * (*scale_out) : 0.0f);
  const Float_Packed_T scale_to_int = CastCudaDataType<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);

#pragma unroll
  for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const int32_t index = blockIdx.x * n + i;
    T val = CastCudaDataType<T>(0.0f);

    if (IS_BIAS) {
      val = hadd2(val, ldg(&bias[i]));
    }
    if (RESIDUAL_NUM == 1) {
      val = hadd2(val, ldg(&residual1[index]));
    } else if (RESIDUAL_NUM == 2) {
      val = hadd2(hadd2(val, ldg(&residual1[index])), ldg(&residual2[index]));
    }

    if (IS_OUTPUT) {
      T in_val;
      if (scale_input) {
        in_val = CastCudaDataType<T>(
            CastCudaDataType<Float_Packed_T>(reinterpret_cast<const Int32_Packed_T*>(input)[index]) * scale_from_int);
      } else {
        in_val = input[index];
      }
      val = hadd2(val, in_val);
    }
    shmem[i] = val;
    output[index] = val;
    local_sum = hadd2(local_sum, val);
  }

  mean = BlockReduceSum((float)(local_sum.x + local_sum.y));

  if (threadIdx.x == 0) {
    s_mean = mean / n / 2;
  }
  __syncthreads();

  float local_var_sum = 0.0f;
#pragma unroll UNROLL_FACTOR
  for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
    T val = input[blockIdx.x * n + i];
    float diff_1 = (float)(val.x) - s_mean;
    float diff_2 = (float)(val.y) - s_mean;
    local_var_sum += (diff_1 * diff_1 + diff_2 * diff_2);
  }
  variance = BlockReduceSum(local_var_sum);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / n / 2 + layernorm_eps);
  }
  __syncthreads();

  T mean_2 = CastCudaDataType<T>(s_mean);
  T var_2 = CastCudaDataType<T>(s_variance);

  Scalar_T abs_max = 1e-6f;

#pragma unroll UNROLL_FACTOR
  for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const int32_t index = blockIdx.x * n + i;
    T val = hmul2(hsub2(shmem[i], mean_2), var_2, ldg(&gamma[i]));
    if (IS_BETA) {
      val = hadd2(val, ldg(&beta[i]));
    }

    if (dynamic_scaling) {
      abs_max = CudaMax(CudaMax<Scalar_T>(cuda_abs(val)), abs_max);
      shmem[i] = val;
    } else if (int8_mode == 2) {
      reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
          CastCudaDataType<Int8_Packed_T>(CastCudaDataType<Float_Packed_T>(val) * scale_to_int);
    } else {
      normed_output[index] = val;
    }
  }

  if (dynamic_scaling) {
    float abs_max_f = BlockAllReduceMax(CastCudaDataType<float>(abs_max));
    const float dynamic_per_token_scale = 127. / abs_max_f;
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
      const int32_t index = blockIdx.x * n + i;
      reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = CastCudaDataType<Int8_Packed_T>(
          CastCudaDataType<Float_Packed_T>(shmem[i]) * CastCudaDataType<Float_Packed_T>(dynamic_per_token_scale));
    }
    if (threadIdx.x == 0) {
      dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
    }
  }
}

// * Note that typename T is half2 or bfloat2 type
template <typename T, bool IS_OUTPUT, bool IS_BIAS, int32_t RESIDUAL_NUM, bool IS_BETA, int32_t UNROLL_FACTOR>
__global__ void InvokeAddBiasResLayerNormOpt2Kernel(T* normed_output, T* output, const T* __restrict input,
                                                    const T* __restrict bias, const T* __restrict residual1,
                                                    const T* __restrict residual2, const T* __restrict gamma,
                                                    const T* __restrict beta, const float layernorm_eps, int32_t m,
                                                    int32_t n, const float* scale_inter, const float* scale_out,
                                                    const float* scale, float* dynamic_scale, const int32_t int8_mode) {
  extern __shared__ __align__(sizeof(float)) char _shmem[];
  T* shmem = reinterpret_cast<T*>(_shmem);

  __shared__ float s_mean;
  __shared__ float s_variance;
  float x_sum = 0.0f;
  float x2_sum = 0.0f;
  const int32_t b_offset = blockIdx.x * n;

  using T1 = typename TypeConverter<T>::Type;
  using Int8_Packed_T = typename PackType<int8_t, ElemsNum<T>::value>::type;
  using Int32_Packed_T = typename PackType<int32_t, ElemsNum<T>::value>::type;
  using Float_Packed_T = typename PackType<float, ElemsNum<T>::value>::type;
  using Scalar_T = typename PackType<T, 1>::type;

  const bool scale_input = int8_mode == 2 && scale_inter != nullptr;
  const Float_Packed_T scale_vec_in =
      CastCudaDataType<Float_Packed_T>(scale_input ? (*scale_inter) * (*scale_out) : 0.0f);
  const Float_Packed_T scale_vec = CastCudaDataType<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);
  const bool dynamic_scaling = dynamic_scale != nullptr;

#pragma unroll UNROLL_FACTOR
  for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const int32_t index = b_offset + i;
    float val_1 = 0.0f;
    float val_2 = 0.0f;
    T tmp;

    if (IS_BIAS) {
      tmp = ldg(&bias[i]);
      val_1 += static_cast<float>(tmp.x);
      val_2 += static_cast<float>(tmp.y);
    }
    if (RESIDUAL_NUM == 1) {
      tmp = ldg(&residual1[index]);
      val_1 += static_cast<float>(tmp.x);
      val_2 += static_cast<float>(tmp.y);
    } else if (RESIDUAL_NUM == 2) {
      tmp = ldg(&residual1[index]);
      T tmp2 = ldg(&residual2[index]);
      val_1 += (static_cast<float>(tmp.x) + static_cast<float>(tmp2.x));
      val_2 += (static_cast<float>(tmp.y) + static_cast<float>(tmp2.y));
    }

    if (IS_OUTPUT) {
      if (scale_input) {
        tmp = CastCudaDataType<T>(
            CastCudaDataType<Float_Packed_T>(reinterpret_cast<const Int32_Packed_T*>(input)[index]) * scale_vec_in);
      } else {
        tmp = ldg(&input[index]);
      }
      val_1 += static_cast<float>(tmp.x);
      val_2 += static_cast<float>(tmp.y);
    }
    tmp.x = CastCudaDataType<T1>(val_1);
    tmp.y = CastCudaDataType<T1>(val_2);
    shmem[i] = tmp;
    output[index] = tmp;
    x_sum += val_1 + val_2;
    x2_sum += val_1 * val_1 + val_2 * val_2;
  }
  float sums[2];
  sums[0] = x_sum;
  sums[1] = x2_sum;
  blockReduceSumV2<float, 2>(sums);

  if (threadIdx.x == 0) {
    s_mean = sums[0] / n / 2;
    s_variance = rsqrtf(sums[1] / n / 2 - s_mean * s_mean + layernorm_eps);
  }
  __syncthreads();

  T mean_2 = CastCudaDataType<T>(s_mean);
  T var_2 = CastCudaDataType<T>(s_variance);

  Scalar_T abs_max = 1e-6f;

#pragma unroll UNROLL_FACTOR
  for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const int32_t index = blockIdx.x * n + i;
    T val = hmul2(hsub2(shmem[i], mean_2), var_2, ldg(&gamma[i]));
    if (IS_BETA) {
      val = hadd2(val, ldg(&beta[i]));
    }

    if (dynamic_scaling) {
      abs_max = CudaMax(CudaMax<Scalar_T>(cuda_abs(val)), abs_max);
      shmem[i] = val;
    } else if (int8_mode == 2) {
      reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
          CastCudaDataType<Int8_Packed_T>(CastCudaDataType<Float_Packed_T>(val) * scale_vec);
    } else {
      normed_output[index] = val;
    }
  }

  if (dynamic_scaling) {
    float abs_max_f = BlockAllReduceMax(CastCudaDataType<float>(abs_max));
    const float dynamic_per_token_scale = 127. / abs_max_f;
    for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
      const int32_t index = blockIdx.x * n + i;
      reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = CastCudaDataType<Int8_Packed_T>(
          CastCudaDataType<Float_Packed_T>(shmem[i]) * CastCudaDataType<Float_Packed_T>(dynamic_per_token_scale));
    }
    if (threadIdx.x == 0) {
      dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
    }
  }
}

template <typename T, bool IS_OUTPUT, bool IS_BIAS, int32_t UNROLL_FACTOR, int32_t RESIDUAL_NUM>
void DispatchAddBiasResidualLayerNormOptWithOptVersion(T* norm_output, T* output, const T* input, const T* bias,
                                                       const T* residual1, const T* residual2, const T* gamma,
                                                       const T* beta, float layernorm_eps, int32_t m, int32_t half_n,
                                                       const float* scale_inter, const float* scale_out,
                                                       const float* scale, float* dynamic_scale, int32_t int8_mode,
                                                       dim3 grid, dim3 block, cudaStream_t stream,
                                                       int32_t opt_version) {
  size_t maxbytes = half_n * sizeof(T);
  if (opt_version == 1) {
    if (maxbytes >= (48 << 10)) {
      CHECK_NVIDIA_CUDA_ERROR(cudaFuncSetAttribute(
          InvokeAddBiasResLayerNormOptKernel<T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM, true, UNROLL_FACTOR>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
    }
    InvokeAddBiasResLayerNormOptKernel<T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM, true, UNROLL_FACTOR>
        <<<grid, block, maxbytes, stream>>>(norm_output, output, input, bias, residual1, residual2, gamma, beta,
                                            layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
                                            int8_mode);
  } else if (opt_version == 2) {
    if (maxbytes >= (48 << 10)) {
      CHECK_NVIDIA_CUDA_ERROR(cudaFuncSetAttribute(
          InvokeAddBiasResLayerNormOpt2Kernel<T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM, true, UNROLL_FACTOR>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
    }
    InvokeAddBiasResLayerNormOpt2Kernel<T, IS_OUTPUT, IS_BIAS, RESIDUAL_NUM, true, UNROLL_FACTOR>
        <<<grid, block, maxbytes, stream>>>(norm_output, output, input, bias, residual1, residual2, gamma, beta,
                                            layernorm_eps, m, half_n, scale_inter, scale_out, scale, dynamic_scale,
                                            int8_mode);
  } else {
    throw std::runtime_error("opt_num must be 1 or 2");
  }
}

template <typename T, bool IS_BIAS, int32_t UNROLL_FACTOR, int32_t RESIDUAL_NUM>
void DispatchAddBiasResidualLayerNormOptWithOutputFlag(T* norm_output, T* output, const T* input, const T* bias,
                                                       const T* residual1, const T* residual2, const T* gamma,
                                                       const T* beta, float layernorm_eps, int32_t m, int32_t half_n,
                                                       const float* scale_inter, const float* scale_out,
                                                       const float* scale, float* dynamic_scale, int32_t int8_mode,
                                                       dim3 grid, dim3 block, cudaStream_t stream, int32_t opt_version,
                                                       bool is_output) {
  if (is_output) {
    DispatchAddBiasResidualLayerNormOptWithOptVersion<T, true, IS_BIAS, UNROLL_FACTOR, RESIDUAL_NUM>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta, layernorm_eps, m, half_n, scale_inter,
        scale_out, scale, dynamic_scale, int8_mode, grid, block, stream, opt_version);
  } else {
    DispatchAddBiasResidualLayerNormOptWithOptVersion<T, false, IS_BIAS, UNROLL_FACTOR, RESIDUAL_NUM>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta, layernorm_eps, m, half_n, scale_inter,
        scale_out, scale, dynamic_scale, int8_mode, grid, block, stream, opt_version);
  }
}

template <typename T, int32_t UNROLL_FACTOR, int32_t RESIDUAL_NUM>
void dispatchAddBiasResidualLayerNormOptWithBias(T* norm_output, T* output, const T* input, const T* bias,
                                                 const T* residual1, const T* residual2, const T* gamma, const T* beta,
                                                 float layernorm_eps, int32_t m, int32_t half_n,
                                                 const float* scale_inter, const float* scale_out, const float* scale,
                                                 float* dynamic_scale, int32_t int8_mode, dim3 grid, dim3 block,
                                                 cudaStream_t stream, int32_t opt_version, bool is_output) {
  if (bias != nullptr) {
    DispatchAddBiasResidualLayerNormOptWithOutputFlag<T, true, UNROLL_FACTOR, RESIDUAL_NUM>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta, layernorm_eps, m, half_n, scale_inter,
        scale_out, scale, dynamic_scale, int8_mode, grid, block, stream, opt_version, is_output);
  } else {
    DispatchAddBiasResidualLayerNormOptWithOutputFlag<T, false, UNROLL_FACTOR, RESIDUAL_NUM>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta, layernorm_eps, m, half_n, scale_inter,
        scale_out, scale, dynamic_scale, int8_mode, grid, block, stream, opt_version, is_output);
  }
}

template <typename T, int32_t UNROLL_FACTOR>
void DispatchAddBiasResidualLayerNormOptWithResNum(T* norm_output, T* output, const T* input, const T* bias,
                                                   const T* residual1, const T* residual2, const T* gamma,
                                                   const T* beta, float layernorm_eps, int32_t m, int32_t half_n,
                                                   const float* scale_inter, const float* scale_out, const float* scale,
                                                   float* dynamic_scale, int32_t int8_mode, dim3 grid, dim3 block,
                                                   cudaStream_t stream, int32_t opt_version, bool is_output,
                                                   int32_t residual_num) {
  if (residual_num == 1) {
    dispatchAddBiasResidualLayerNormOptWithBias<T, UNROLL_FACTOR, 1>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta, layernorm_eps, m, half_n, scale_inter,
        scale_out, scale, dynamic_scale, int8_mode, grid, block, stream, opt_version, is_output);
  } else if (residual_num == 2) {
    dispatchAddBiasResidualLayerNormOptWithBias<T, UNROLL_FACTOR, 2>(
        norm_output, output, input, bias, residual1, residual2, gamma, beta, layernorm_eps, m, half_n, scale_inter,
        scale_out, scale, dynamic_scale, int8_mode, grid, block, stream, opt_version, is_output);
  } else {
    throw std::runtime_error("residual_num must be 1 or 2");
  }
}

template <typename T>
void DispatchGeneralAddBiasResidualLayerNormOptUnrollFactor(
    T* norm_output, T* output, const T* input, const T* bias, const T* residual1, const T* residual2, const T* gamma,
    const T* beta, float layernorm_eps, int32_t m, int32_t half_n, const float* scale_inter, const float* scale_out,
    const float* scale, float* dynamic_scale, int32_t int8_mode, dim3 grid, dim3 block, cudaStream_t stream,
    int32_t opt_version, bool is_output, int32_t residual_num, int32_t unroll_factor) {
  switch (unroll_factor) {
    case 1:
      DispatchAddBiasResidualLayerNormOptWithResNum<T, 1>(
          norm_output, output, input, bias, residual1, residual2, gamma, beta, layernorm_eps, m, half_n, scale_inter,
          scale_out, scale, dynamic_scale, int8_mode, grid, block, stream, opt_version, is_output, residual_num);
      break;
    case 2:
      DispatchAddBiasResidualLayerNormOptWithResNum<T, 2>(
          norm_output, output, input, bias, residual1, residual2, gamma, beta, layernorm_eps, m, half_n, scale_inter,
          scale_out, scale, dynamic_scale, int8_mode, grid, block, stream, opt_version, is_output, residual_num);
      break;
    case 4:
      DispatchAddBiasResidualLayerNormOptWithResNum<T, 4>(
          norm_output, output, input, bias, residual1, residual2, gamma, beta, layernorm_eps, m, half_n, scale_inter,
          scale_out, scale, dynamic_scale, int8_mode, grid, block, stream, opt_version, is_output, residual_num);
      break;
    case 8:
      DispatchAddBiasResidualLayerNormOptWithResNum<T, 8>(
          norm_output, output, input, bias, residual1, residual2, gamma, beta, layernorm_eps, m, half_n, scale_inter,
          scale_out, scale, dynamic_scale, int8_mode, grid, block, stream, opt_version, is_output, residual_num);
      break;
    default:
      throw std::runtime_error("unroll_factor must be 1, 2, 4 or 8");
  }
}

template <typename T, bool DYNAMIC_SCALING = false>
__global__ void InvokeLayerNormWithBiasKernel(const T* __restrict input, const T* __restrict gamma,
                                              const T* __restrict beta, T* normed_output, const float layernorm_eps,
                                              int32_t m, int32_t n, float* scale, float* dynamic_scale,
                                              const int32_t int8_mode) {
  const int32_t tid = threadIdx.x;

  extern __shared__ __align__(sizeof(float)) char _shmem[];
  T* shmem = reinterpret_cast<T*>(_shmem);

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  using Int8_Packed_T = typename PackType<int8_t, ElemsNum<T>::value>::type;
  using Int32_Packed_T = typename PackType<int32_t, ElemsNum<T>::value>::type;
  using Float_Packed_T = typename PackType<float, ElemsNum<T>::value>::type;
  using Scalar_T = typename PackType<T, 1>::type;

  const Float_Packed_T scale_to_int = CastCudaDataType<Float_Packed_T>(int8_mode == 2 ? *scale : 0.0f);

  float local_sum = 0.0f;
  for (int32_t i = tid; i < n; i += blockDim.x) {
    local_sum += (float)(ldg(&input[blockIdx.x * n + i]));
  }

  mean = BlockReduceSum(local_sum);

  if (threadIdx.x == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  float local_var_sum = 0.0f;
  for (int32_t i = tid; i < n; i += blockDim.x) {
    float diff = (float)(ldg(&input[blockIdx.x * n + i])) - s_mean;
    local_var_sum += diff * diff;
  }
  variance = BlockReduceSum(local_var_sum);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / n + layernorm_eps);
  }
  __syncthreads();

  Scalar_T abs_max = 1e-6f;

  for (int32_t i = tid; i < n; i += blockDim.x) {
    const int32_t index = blockIdx.x * n + i;
    float beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
    T val = (T)((((float)input[index] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);

    if (DYNAMIC_SCALING) {
      abs_max = CudaMax(CudaMax<Scalar_T, T>(cuda_abs(val)), abs_max);
      shmem[i] = val;
    } else if (int8_mode == 2) {
      reinterpret_cast<Int8_Packed_T*>(normed_output)[index] =
          CastCudaDataType<Int8_Packed_T>(CastCudaDataType<Float_Packed_T>(val) * scale_to_int);
    } else {
      normed_output[index] = val;
    }
  }

  if (DYNAMIC_SCALING) {
    float abs_max_f = BlockAllReduceMax(CastCudaDataType<float>(abs_max));
    const Scalar_T dynamic_per_token_scale = 127. / abs_max_f;
    for (int32_t i = tid; i < n; i += blockDim.x) {
      const int32_t index = blockIdx.x * n + i;
      reinterpret_cast<Int8_Packed_T*>(normed_output)[index] = CastCudaDataType<Int8_Packed_T>(
          CastCudaDataType<Float_Packed_T>(shmem[i]) * CastCudaDataType<Float_Packed_T>(dynamic_per_token_scale));
    }
    if (threadIdx.x == 0) {
      dynamic_scale[blockIdx.x] = (*scale * abs_max_f) / 127.f;
    }
  }
}

template <typename T>
void InvokeLayerNormWithBeta(T* out, const T* input, const T* gamma, const T* beta, const float layernorm_eps,
                             const int32_t m, const int32_t n, float* scale, float* dynamic_scale,
                             const int32_t int8_mode, cudaStream_t stream, int32_t opt_version) {
  dim3 grid(m);
  const bool dynamic_quant = dynamic_scale != nullptr;
  if (n % 2 == 0 && (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && opt_version > 0) {
    int32_t half_n = n / 2;
    int32_t half_n_32 = (half_n + 31) / 32 * 32;
    dim3 block(min(half_n_32, 512));
    int32_t rolls_per_thread = half_n / block.x;
    int32_t unroll_factor = 8;
    while (unroll_factor > rolls_per_thread && unroll_factor > 1) {
      unroll_factor /= 2;
    }
    using T2 = typename TypeConverter<T>::Type;

    // we launch (and instantiate) the kernel by specializing for unroll_factor -> residual_num -> is_bias ->
    // opt_version
    DispatchGeneralAddBiasResidualLayerNormOptUnrollFactor(
        (T2*)out, (T2*)out, (const T2*)out, (const T2*)nullptr, (const T2*)input, (const T2*)nullptr, (const T2*)gamma,
        (const T2*)beta, layernorm_eps, m, half_n, nullptr, nullptr, scale, dynamic_scale, int8_mode, grid, block,
        stream, opt_version,
        false,  // is_output
        1,      // residual_num
        unroll_factor);
  } else {
    dim3 block(min(n, 1024));

    // For general cases, n is equal to hidden_units, e.g., 512/1024. Since we have warp shuffle inside the code,
    // block.x % 32 should be 0.
    if (n % 32 != 0) {
      block.x = 1024;
    }

    // should pay attention to the rsqrt precision
    if (dynamic_quant) {
      size_t maxbytes = n * sizeof(T);
      if (maxbytes >= (48 << 10)) {
        CHECK_NVIDIA_CUDA_ERROR(cudaFuncSetAttribute(InvokeLayerNormWithBiasKernel<T, true>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes));
      }
      InvokeLayerNormWithBiasKernel<T, true><<<grid, block, maxbytes, stream>>>(
          input, gamma, beta, out, layernorm_eps, m, n, scale, dynamic_scale, int8_mode);  // For gpt-3
    } else {
      InvokeLayerNormWithBiasKernel<T, false><<<grid, block, 0, stream>>>(
          input, gamma, beta, out, layernorm_eps, m, n, scale, dynamic_scale, int8_mode);  // For gpt-3
    }
  }
}

template void InvokeLayerNormWithBeta(float* out, const float* input, const float* gamma, const float* beta,
                                      const float layernorm_eps, const int32_t m, const int32_t n, float* scale,
                                      float* dynamic_scale, const int32_t int8_mode, cudaStream_t stream,
                                      int32_t opt_version);
template void InvokeLayerNormWithBeta(half* out, const half* input, const half* gamma, const half* beta,
                                      const float layernorm_eps, const int32_t m, const int32_t n, float* scale,
                                      float* dynamic_scale, const int32_t int8_mode, cudaStream_t stream,
                                      int32_t opt_version);
template void InvokeLayerNormWithBeta(__nv_bfloat16* out, const __nv_bfloat16* input, const __nv_bfloat16* gamma,
                                      const __nv_bfloat16* beta, const float layernorm_eps, const int32_t m,
                                      const int32_t n, float* scale, float* dynamic_scale, const int32_t int8_mode,
                                      cudaStream_t stream, int32_t opt_version);

template <typename T>
__global__ void InvokeRMSNormKernel(const T* __restrict input, const T* __restrict gamma, T* output,
                                    const float layernorm_eps, int32_t m, int32_t n) {
  const int32_t tid = threadIdx.x;

  __shared__ float s_variance;
  float variance = 0.0f;

  float local_var_sum = 0.0f;
  for (int32_t i = tid; i < n; i += blockDim.x) {
    float diff = (float)(ldg(&input[blockIdx.x * n + i]));
    local_var_sum += diff * diff;
  }
  variance = BlockReduceSum(local_var_sum);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / (float)n + layernorm_eps);
  }
  __syncthreads();

  for (int32_t i = tid; i < n; i += blockDim.x) {
    output[blockIdx.x * n + i] =
        ClampInfForHalf<T>((((float)input[blockIdx.x * n + i]) * s_variance) * (float)(ldg(&gamma[i])));
  }
}

template <typename T>
void InvokeLayerNorm(T* out, const T* input, const T* gamma, const T* beta, float layernorm_eps, int32_t m, int32_t n,
                     cudaStream_t stream) {
  InvokeLayerNormWithBeta(out, input, gamma, beta, layernorm_eps, m, n, (float*)nullptr, 0, stream);
}

#define INVOKE_LAYER_NORM(T)                                                                                           \
  template void InvokeLayerNorm(T* out, const T* input, const T* gamma, const T* beta, float layernorm_eps, int32_t m, \
                                int32_t n, cudaStream_t stream)
INVOKE_LAYER_NORM(float);
INVOKE_LAYER_NORM(half);
INVOKE_LAYER_NORM(__nv_bfloat16);
#undef INVOKE_LAYER_NORM

template <typename T>
void InvokeRMSNorm(T* out, T* input, T* gamma, float layernorm_eps, int32_t m, int32_t n, bool enable_pdl,
                   cudaStream_t stream) {
#if defined(ENABLE_FLASHINFER)
  flashinfer::norm::RMSNorm(input, gamma, out, m, n,
                            /*stride_input*/ n, /*stride_output*/ n, layernorm_eps, enable_pdl, stream);
  return;
#endif

  dim3 grid(m);
  dim3 block(min(n, 1024));

  // For general cases, n is equal to hidden_units, e.g., 512/1024.
  // Since we have warp shuffle inside the code, block.x % 32 should be 0.
  if (n % 32 != 0) {
    block.x = 1024;
  }

  block.x = block.x / (4 / sizeof(T));  // if using half, only need half of block.x

  // should pay attention to the rsqrt precision
  InvokeRMSNormKernel<T><<<grid, block, 0, stream>>>(input, gamma, out, layernorm_eps, m, n);  // For gpt-3
}

#define INVOKE_RMS_NORM(T)                                                                                            \
  template void InvokeRMSNorm(T* out, T* input, T* gamma, float layernorm_eps, int32_t m, int32_t n, bool enable_pdl, \
                              cudaStream_t stream)
INVOKE_RMS_NORM(float);
INVOKE_RMS_NORM(half);
INVOKE_RMS_NORM(__nv_bfloat16);
#undef INVOKE_RMS_NORM

template <typename T>
__global__ void InvokeAddResLayerNormKernel(const T* __restrict input, const T* __restrict gamma, T* output,
                                            T* norm_output, const float layernorm_eps, int32_t m, int32_t n) {
  // layernorm module in the T5 style No bias and no subtraction of mean.
  __shared__ float s_variance;
  float variance = 0.0f;

  float local_var_sum = 0.0f;
  for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
    output[blockIdx.x * n + i] =
        ClampInfForHalf<T>((float)ldg(&input[blockIdx.x * n + i]) + (float)output[blockIdx.x * n + i]);

    float diff = (float)(output[blockIdx.x * n + i]);
    local_var_sum += diff * diff;
  }
  variance = BlockReduceSum(local_var_sum);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / (float)n + layernorm_eps);
  }
  __syncthreads();

  for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
    norm_output[blockIdx.x * n + i] =
        ClampInfForHalf<T>((((float)output[blockIdx.x * n + i]) * s_variance) * (float)(ldg(&gamma[i])));
  }
}

template <typename T>
void InvokeAddResPreLayerNorm(T* output, T* norm_output, const T* input, const T* gamma, const float layernorm_eps,
                              int32_t m, int32_t n, cudaStream_t stream) {
  dim3 grid(m);
  dim3 block(min(n, 1024));

  // For general cases, n is equal to hidden_units, e.g., 512/1024. Since we have warp shuffle inside the code, block.x
  // % 32 should be 0.
  if (n % 32 != 0) {
    block.x = 1024;
  }

  // TODO(karlluo): add 16bitx2 implementation should pay attention to the rsqrt precision
  InvokeAddResLayerNormKernel<T><<<grid, block, 0, stream>>>(input, gamma, output, norm_output, layernorm_eps, m, n);
}

template void InvokeAddResPreLayerNorm(float* output, float* norm_output, const float* input, const float* gamma,
                                       const float layernorm_eps, int32_t m, int32_t n, cudaStream_t stream);

template void InvokeAddResPreLayerNorm(half* output, half* norm_output, const half* input, const half* gamma,
                                       const float layernorm_eps, int32_t m, int32_t n, cudaStream_t stream);

template void InvokeAddResPreLayerNorm(__nv_bfloat16* output, __nv_bfloat16* norm_output, const __nv_bfloat16* input,
                                       const __nv_bfloat16* gamma, const float layernorm_eps, int32_t m, int32_t n,
                                       cudaStream_t stream);

template <typename T>
__global__ void InvokeRmsNorm3DKernel(const T* __restrict input, const T* __restrict gamma, T* output,
                                      const float layernorm_eps, int32_t len, int32_t m, int32_t n, int32_t start,
                                      int32_t end, const int64_t* __restrict__ mask) {
  const int32_t tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  // Filter out some unnecessary 2D matrices.
  int64_t mask_i = mask[token_idx];
  if (mask_i == 0) {
    return;
  }
  __shared__ float s_variance;
  // The starting index of each row in a 2D matrix
  int32_t matrix_offset = blockIdx.x * m * n + blockIdx.y * n;

  // Set start and end rows
  if (blockIdx.y >= start && blockIdx.y < end) {
    float local_var_sum = 0.0f;
    for (int32_t i = tid; i < n; i += blockDim.x) {
      float diff = (float)(ldg(&input[matrix_offset + i]));
      local_var_sum += diff * diff;
    }
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    local_var_sum = BlockReduce(reduceStore).Reduce(local_var_sum, cub::Sum{}, blockDim.x);

    if (threadIdx.x == 0) {
      s_variance = rsqrtf(local_var_sum / (float)n + layernorm_eps);
    }
    __syncthreads();

    if (gamma != nullptr) {
      for (int32_t i = tid; i < n; i += blockDim.x) {
        output[matrix_offset + i] =
            ClampInfForHalf<T>((((float)input[matrix_offset + i]) * s_variance) * (float)(ldg(&gamma[i])));
      }
    } else {
      for (int32_t i = tid; i < n; i += blockDim.x) {
        output[matrix_offset + i] = ClampInfForHalf<T>((((float)input[matrix_offset + i]) * s_variance));
      }
    }
  }
}

// For 3dim rmsnorm. The input shape is [len, m, n], performs rmsnorm on the last dimension 'n' of the input matrix.
// param 'mask': Allowing for the filtering of certain l-dimension 2D matrices based on a custom 'mask' 1D array.
// param 'start' 'end': For each 2D matrix, defined 'start' and 'end' params to apply rmsnorm to a specific range of
// consecutive rows.
template <typename T>
void InvokeRmsNorm3D(T* out, const T* input, const T* gamma, const float layernorm_eps, const int32_t total_tokens,
                     const int32_t m, const int32_t n, const int32_t start, const int32_t end, const int64_t* mask,
                     cudaStream_t stream) {
  dim3 grid(total_tokens, m);
  dim3 block(min(n, 1024));

  InvokeRmsNorm3DKernel<T>
      <<<grid, block, 0, stream>>>(input, gamma, out, layernorm_eps, total_tokens, m, n, start, end, mask);
}

template void InvokeRmsNorm3D(float* out, const float* input, const float* gamma, const float layernorm_eps,
                              const int32_t total_tokens, const int32_t m, const int32_t n, const int32_t start,
                              const int32_t end, const int64_t* mask, cudaStream_t stream);
template void InvokeRmsNorm3D(half* out, const half* input, const half* gamma, const float layernorm_eps,
                              const int32_t total_tokens, const int32_t m, const int32_t n, const int32_t start,
                              const int32_t end, const int64_t* mask, cudaStream_t stream);
template void InvokeRmsNorm3D(__nv_bfloat16* out, const __nv_bfloat16* input, const __nv_bfloat16* gamma,
                              const float layernorm_eps, const int32_t total_tokens, const int32_t m, const int32_t n,
                              const int32_t start, const int32_t end, const int64_t* mask, cudaStream_t stream);

template <typename T>
__global__ void InvokeFusedQKVRmsNormKernel(const T* __restrict input, const T* __restrict q_gamma,
                                            const T* __restrict k_gamma, T* output, const float layernorm_eps,
                                            const int32_t total_tokens, const int32_t num_heads,
                                            const int32_t num_kv_heads, const int32_t head_size,
                                            const int64_t* __restrict__ mask) {
  const int32_t tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  // Filter out some unnecessary 2D matrices.
  int64_t mask_i = mask[token_idx];
  if (mask_i == 0) {
    return;
  }

  __shared__ float s_variance;
  // The starting index of each row in a 2D matrix
  int32_t matrix_offset = blockIdx.x * (num_heads + 2 * num_kv_heads) * head_size + blockIdx.y * head_size;

  float local_var_sum = 0.0f;
  for (int32_t i = tid; i < head_size; i += blockDim.x) {
    float diff = (float)(ldg(&input[matrix_offset + i]));
    local_var_sum += diff * diff;
  }
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  local_var_sum = BlockReduce(reduceStore).Reduce(local_var_sum, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(local_var_sum / (float)head_size + layernorm_eps);
  }
  __syncthreads();

  for (int32_t i = tid; i < head_size; i += blockDim.x) {
    float gamma = 1.0f;
    if (blockIdx.y >= 0 && blockIdx.y < num_heads) {
      if (q_gamma != nullptr) {
        gamma = (float)(ldg(&q_gamma[i]));
      }
    } else {
      if (k_gamma != nullptr) {
        gamma = (float)(ldg(&k_gamma[i]));
      }
    }
    output[matrix_offset + i] = ClampInfForHalf<T>((((float)input[matrix_offset + i]) * s_variance) * gamma);
  }
}

template <typename T>
void InvokeFusedQKVRmsNorm(T* out, const T* input, const T* q_gamma, const T* k_gamma, const float layernorm_eps,
                           const int32_t total_tokens, const int32_t num_heads, const int32_t num_kv_heads,
                           const int32_t head_size, const int64_t* mask, cudaStream_t stream) {
  dim3 grid(total_tokens, (num_heads + num_kv_heads));
  dim3 block(min(head_size, 1024));

  InvokeFusedQKVRmsNormKernel<T><<<grid, block, 0, stream>>>(input, q_gamma, k_gamma, out, layernorm_eps, total_tokens,
                                                             num_heads, num_kv_heads, head_size, mask);
}

template void InvokeFusedQKVRmsNorm(float* out, const float* input, const float* q_gamma, const float* k_gamma,
                                    const float layernorm_eps, const int32_t total_tokens, const int32_t num_heads,
                                    const int32_t num_kv_heads, const int32_t head_size, const int64_t* mask,
                                    cudaStream_t stream);

template void InvokeFusedQKVRmsNorm(half* out, const half* input, const half* q_gamma, const half* k_gamma,
                                    const float layernorm_eps, const int32_t total_tokens, const int32_t num_heads,
                                    const int32_t num_kv_heads, const int32_t head_size, const int64_t* mask,
                                    cudaStream_t stream);
template void InvokeFusedQKVRmsNorm(__nv_bfloat16* out, const __nv_bfloat16* input, const __nv_bfloat16* q_gamma,
                                    const __nv_bfloat16* k_gamma, const float layernorm_eps, const int32_t total_tokens,
                                    const int32_t num_heads, const int32_t num_kv_heads, const int32_t head_size,
                                    const int64_t* mask, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
