/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

// This file is adopt
// from
// https://github.com/vllm-project/vllm/blob/891070257c145b506a20666a3cb70afcf674d4ca/csrc/punica/bgmv/vec_dtypes.cuh#L13

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifdef FLASHINFER_USE_FP8
#  include <cuda_fp8.h>
#endif
#include <cuda_runtime.h>

#include <type_traits>

#define INFER_INLINE_SPECIFIER inline __attribute__((always_inline)) __device__ __host__

namespace llm_kernels {
namespace nvidia {

template <typename float_t, size_t vec_size>
struct vec_t {
  INFER_INLINE_SPECIFIER float_t &operator[](size_t i);
  INFER_INLINE_SPECIFIER const float_t &operator[](size_t i) const;
  INFER_INLINE_SPECIFIER void fill(float_t val);
  INFER_INLINE_SPECIFIER void load(const float_t *ptr);
  INFER_INLINE_SPECIFIER void store(float_t *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, vec_size> &src);
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr);
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const;
  INFER_INLINE_SPECIFIER static void memcpy(float_t *dst, const float_t *src);
};

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<src_float_t, vec_size> &src, vec_t<tgt_float_t, vec_size> &dst) {
#pragma unroll
  for (size_t i = 0; i < vec_size; ++i) {
    dst[i] = tgt_float_t(src[i]);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
INFER_INLINE_SPECIFIER void cast_load_impl(const src_float_t *src_ptr, vec_t<tgt_float_t, vec_size> &dst) {
  if constexpr (std::is_same<src_float_t, tgt_float_t>::value) {
    dst.load(src_ptr);
  } else {
    vec_t<src_float_t, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
INFER_INLINE_SPECIFIER void cast_store_impl(const vec_t<src_float_t, vec_size> &src, tgt_float_t *dst_ptr) {
  if constexpr (std::is_same<src_float_t, tgt_float_t>::value) {
    src.store(dst_ptr);
  } else {
    vec_t<tgt_float_t, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}

#ifdef FLASHINFER_USE_FP8
/******************* vec_t<__nv_fp8_e4m3> *******************/

// __nv_fp8_e4m3 x 1
template <>
struct vec_t<__nv_fp8_e4m3, 1> {
  __nv_fp8_e4m3 data;

  INFER_INLINE_SPECIFIER __nv_fp8_e4m3 &operator[](size_t i) { return ((__nv_fp8_e4m3 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const __nv_fp8_e4m3 &operator[](size_t i) const { return ((const __nv_fp8_e4m3 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(__nv_fp8_e4m3 val);
  INFER_INLINE_SPECIFIER void load(const __nv_fp8_e4m3 *ptr);
  INFER_INLINE_SPECIFIER void store(__nv_fp8_e4m3 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 1> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src);
};

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 1>::fill(__nv_fp8_e4m3 val) { data = val; }

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 1>::load(const __nv_fp8_e4m3 *ptr) { data = *ptr; }

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 1>::store(__nv_fp8_e4m3 *ptr) const { *ptr = data; }

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 1>::memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src) {
  *dst = *src;
}

// __nv_fp8_e4m3 x 2
template <>
struct vec_t<__nv_fp8_e4m3, 2> {
  __nv_fp8x2_e4m3 data;

  INFER_INLINE_SPECIFIER __nv_fp8_e4m3 &operator[](size_t i) { return ((__nv_fp8_e4m3 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const __nv_fp8_e4m3 &operator[](size_t i) const { return ((const __nv_fp8_e4m3 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(__nv_fp8_e4m3 val);
  INFER_INLINE_SPECIFIER void load(const __nv_fp8_e4m3 *ptr);
  INFER_INLINE_SPECIFIER void store(__nv_fp8_e4m3 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 2> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src);
};

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 2>::fill(__nv_fp8_e4m3 val) {
  data.__x = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 2>::load(const __nv_fp8_e4m3 *ptr) {
  data = *((__nv_fp8x2_e4m3 *)ptr);
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 2>::store(__nv_fp8_e4m3 *ptr) const {
  *((__nv_fp8x2_e4m3 *)ptr) = data;
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 2>::memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src) {
  *((__nv_fp8x2_e4m3 *)dst) = *((__nv_fp8x2_e4m3 *)src);
}

// __nv_fp8_e4m3 x 4

template <>
struct vec_t<__nv_fp8_e4m3, 4> {
  __nv_fp8x4_e4m3 data;

  INFER_INLINE_SPECIFIER __nv_fp8_e4m3 &operator[](size_t i) { return ((__nv_fp8_e4m3 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const __nv_fp8_e4m3 &operator[](size_t i) const { return ((const __nv_fp8_e4m3 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(__nv_fp8_e4m3 val);
  INFER_INLINE_SPECIFIER void load(const __nv_fp8_e4m3 *ptr);
  INFER_INLINE_SPECIFIER void store(__nv_fp8_e4m3 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 4> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src);
};

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 4>::fill(__nv_fp8_e4m3 val) {
  data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
             (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 4>::load(const __nv_fp8_e4m3 *ptr) {
  data = *((__nv_fp8x4_e4m3 *)ptr);
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 4>::store(__nv_fp8_e4m3 *ptr) const {
  *((__nv_fp8x4_e4m3 *)ptr) = data;
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 4>::memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src) {
  *((__nv_fp8x4_e4m3 *)dst) = *((__nv_fp8x4_e4m3 *)src);
}

// __nv_fp8_e4m3 x 8

template <>
struct vec_t<__nv_fp8_e4m3, 8> {
  uint2 data;

  INFER_INLINE_SPECIFIER __nv_fp8_e4m3 &operator[](size_t i) { return ((__nv_fp8_e4m3 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const __nv_fp8_e4m3 &operator[](size_t i) const { return ((const __nv_fp8_e4m3 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(__nv_fp8_e4m3 val);
  INFER_INLINE_SPECIFIER void load(const __nv_fp8_e4m3 *ptr);
  INFER_INLINE_SPECIFIER void store(__nv_fp8_e4m3 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 8> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src);
};

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 8>::fill(__nv_fp8_e4m3 val) {
  ((__nv_fp8x4_e4m3 *)(&data.x))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
                                        (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
  ((__nv_fp8x4_e4m3 *)(&data.y))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
                                        (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 8>::load(const __nv_fp8_e4m3 *ptr) { data = *((uint2 *)ptr); }

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 8>::store(__nv_fp8_e4m3 *ptr) const { *((uint2 *)ptr) = data; }

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e4m3, 8>::memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src) {
  *((__nv_fp8_e4m3 *)dst) = *((__nv_fp8_e4m3 *)src);
}

// __nv_fp8_e4m3 x 16 or more
template <size_t vec_size>
struct vec_t<__nv_fp8_e4m3, vec_size> {
  uint4 data[vec_size / 16];

  INFER_INLINE_SPECIFIER __nv_fp8_e4m3 &operator[](size_t i) { return ((__nv_fp8_e4m3 *)data)[i]; }
  INFER_INLINE_SPECIFIER const __nv_fp8_e4m3 &operator[](size_t i) const { return ((const __nv_fp8_e4m3 *)data)[i]; }
  INFER_INLINE_SPECIFIER void fill(__nv_fp8_e4m3 val) {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((__nv_fp8x4_e4m3 *)(&(data[i].x)))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 16) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3 *)(&(data[i].y)))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 16) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3 *)(&(data[i].z)))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 16) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3 *)(&(data[i].w)))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 16) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
    }
  }
  INFER_INLINE_SPECIFIER void load(const __nv_fp8_e4m3 *ptr) {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      data[i] = ((uint4 *)ptr)[i];
    }
  }
  INFER_INLINE_SPECIFIER void store(__nv_fp8_e4m3 *ptr) const {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((uint4 *)ptr)[i] = data[i];
    }
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, vec_size> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(__nv_fp8_e4m3 *dst, const __nv_fp8_e4m3 *src) {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((uint4 *)dst)[i] = ((uint4 *)src)[i];
    }
  }
};

/******************* vec_t<__nv_fp8_e5m2> *******************/

// __nv_fp8_e5m2 x 1
template <>
struct vec_t<__nv_fp8_e5m2, 1> {
  __nv_fp8_e5m2 data;

  INFER_INLINE_SPECIFIER __nv_fp8_e5m2 &operator[](size_t i) { return ((__nv_fp8_e5m2 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const __nv_fp8_e5m2 &operator[](size_t i) const { return ((const __nv_fp8_e5m2 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(__nv_fp8_e5m2 val);
  INFER_INLINE_SPECIFIER void load(const __nv_fp8_e5m2 *ptr);
  INFER_INLINE_SPECIFIER void store(__nv_fp8_e5m2 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 1> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src);
};

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 1>::fill(__nv_fp8_e5m2 val) { data = val; }

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 1>::load(const __nv_fp8_e5m2 *ptr) { data = *ptr; }

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 1>::store(__nv_fp8_e5m2 *ptr) const { *ptr = data; }

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 1>::memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src) {
  *dst = *src;
}

// __nv_fp8_e5m2 x 2
template <>
struct vec_t<__nv_fp8_e5m2, 2> {
  __nv_fp8x2_e5m2 data;

  INFER_INLINE_SPECIFIER __nv_fp8_e5m2 &operator[](size_t i) { return ((__nv_fp8_e5m2 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const __nv_fp8_e5m2 &operator[](size_t i) const { return ((const __nv_fp8_e5m2 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(__nv_fp8_e5m2 val);
  INFER_INLINE_SPECIFIER void load(const __nv_fp8_e5m2 *ptr);
  INFER_INLINE_SPECIFIER void store(__nv_fp8_e5m2 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 2> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src);
};

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 2>::fill(__nv_fp8_e5m2 val) {
  data.__x = (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 2>::load(const __nv_fp8_e5m2 *ptr) {
  data = *((__nv_fp8x2_e5m2 *)ptr);
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 2>::store(__nv_fp8_e5m2 *ptr) const {
  *((__nv_fp8x2_e5m2 *)ptr) = data;
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 2>::memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src) {
  *((__nv_fp8x2_e5m2 *)dst) = *((__nv_fp8x2_e5m2 *)src);
}

// __nv_fp8_e5m2 x 4

template <>
struct vec_t<__nv_fp8_e5m2, 4> {
  __nv_fp8x4_e5m2 data;

  INFER_INLINE_SPECIFIER __nv_fp8_e5m2 &operator[](size_t i) { return ((__nv_fp8_e5m2 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const __nv_fp8_e5m2 &operator[](size_t i) const { return ((const __nv_fp8_e5m2 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(__nv_fp8_e5m2 val);
  INFER_INLINE_SPECIFIER void load(const __nv_fp8_e5m2 *ptr);
  INFER_INLINE_SPECIFIER void store(__nv_fp8_e5m2 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 4> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src);
};

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 4>::fill(__nv_fp8_e5m2 val) {
  data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
             (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 4>::load(const __nv_fp8_e5m2 *ptr) {
  data = *((__nv_fp8x4_e5m2 *)ptr);
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 4>::store(__nv_fp8_e5m2 *ptr) const {
  *((__nv_fp8x4_e5m2 *)ptr) = data;
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 4>::memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src) {
  *((__nv_fp8x4_e5m2 *)dst) = *((__nv_fp8x4_e5m2 *)src);
}

// __nv_fp8_e5m2 x 8

template <>
struct vec_t<__nv_fp8_e5m2, 8> {
  uint2 data;

  INFER_INLINE_SPECIFIER __nv_fp8_e5m2 &operator[](size_t i) { return ((__nv_fp8_e5m2 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const __nv_fp8_e5m2 &operator[](size_t i) const { return ((const __nv_fp8_e5m2 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(__nv_fp8_e5m2 val);
  INFER_INLINE_SPECIFIER void load(const __nv_fp8_e5m2 *ptr);
  INFER_INLINE_SPECIFIER void store(__nv_fp8_e5m2 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 8> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src);
};

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 8>::fill(__nv_fp8_e5m2 val) {
  ((__nv_fp8x4_e5m2 *)(&data.x))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
                                        (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
  ((__nv_fp8x4_e5m2 *)(&data.y))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) | (__nv_fp8x4_storage_t(val.__x) << 16) |
                                        (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
}

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 8>::load(const __nv_fp8_e5m2 *ptr) { data = *((uint2 *)ptr); }

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 8>::store(__nv_fp8_e5m2 *ptr) const { *((uint2 *)ptr) = data; }

INFER_INLINE_SPECIFIER void vec_t<__nv_fp8_e5m2, 8>::memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src) {
  *((__nv_fp8_e5m2 *)dst) = *((__nv_fp8_e5m2 *)src);
}

// __nv_fp8_e5m2 x 16 or more

template <size_t vec_size>
struct vec_t<__nv_fp8_e5m2, vec_size> {
  uint4 data[vec_size / 16];

  INFER_INLINE_SPECIFIER __nv_fp8_e5m2 &operator[](size_t i) { return ((__nv_fp8_e5m2 *)data)[i]; }
  INFER_INLINE_SPECIFIER const __nv_fp8_e5m2 &operator[](size_t i) const { return ((const __nv_fp8_e5m2 *)data)[i]; }
  INFER_INLINE_SPECIFIER void fill(__nv_fp8_e5m2 val) {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((__nv_fp8x4_e5m2 *)(&(data[i].x)))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 16) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e5m2 *)(&(data[i].y)))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 16) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e5m2 *)(&(data[i].z)))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 16) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e5m2 *)(&(data[i].w)))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 16) |
                                                 (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
    }
  }
  INFER_INLINE_SPECIFIER void load(const __nv_fp8_e5m2 *ptr) {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      data[i] = ((uint4 *)ptr)[i];
    }
  }
  INFER_INLINE_SPECIFIER void store(__nv_fp8_e5m2 *ptr) const {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((uint4 *)ptr)[i] = data[i];
    }
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, vec_size> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(__nv_fp8_e5m2 *dst, const __nv_fp8_e5m2 *src) {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((uint4 *)dst)[i] = ((uint4 *)src)[i];
    }
  }
};
#endif

/******************* vec_t<half> *******************/

// half x 1
template <>
struct vec_t<half, 1> {
  half data;

  INFER_INLINE_SPECIFIER half &operator[](size_t i) { return ((half *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const half &operator[](size_t i) const { return ((const half *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(half val);
  INFER_INLINE_SPECIFIER void load(const half *ptr);
  INFER_INLINE_SPECIFIER void store(half *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 1> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(half *dst, const half *src);
};

INFER_INLINE_SPECIFIER void vec_t<half, 1>::fill(half val) { data = val; }

INFER_INLINE_SPECIFIER void vec_t<half, 1>::load(const half *ptr) { data = *ptr; }

INFER_INLINE_SPECIFIER void vec_t<half, 1>::store(half *ptr) const { *ptr = data; }

INFER_INLINE_SPECIFIER void vec_t<half, 1>::memcpy(half *dst, const half *src) { *dst = *src; }

// half x 2
template <>
struct vec_t<half, 2> {
  half2 data;

  INFER_INLINE_SPECIFIER half &operator[](size_t i) { return ((half *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const half &operator[](size_t i) const { return ((const half *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(half val);
  INFER_INLINE_SPECIFIER void load(const half *ptr);
  INFER_INLINE_SPECIFIER void store(half *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 2> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(half *dst, const half *src);
};

INFER_INLINE_SPECIFIER void vec_t<half, 2>::fill(half val) { data = make_half2(val, val); }

INFER_INLINE_SPECIFIER void vec_t<half, 2>::load(const half *ptr) { data = *((half2 *)ptr); }

INFER_INLINE_SPECIFIER void vec_t<half, 2>::store(half *ptr) const { *((half2 *)ptr) = data; }

INFER_INLINE_SPECIFIER void vec_t<half, 2>::memcpy(half *dst, const half *src) { *((half2 *)dst) = *((half2 *)src); }

// half x 4

template <>
struct vec_t<half, 4> {
  uint2 data;

  INFER_INLINE_SPECIFIER half &operator[](size_t i) { return ((half *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const half &operator[](size_t i) const { return ((const half *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(half val);
  INFER_INLINE_SPECIFIER void load(const half *ptr);
  INFER_INLINE_SPECIFIER void store(half *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 4> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(half *dst, const half *src);
};

INFER_INLINE_SPECIFIER void vec_t<half, 4>::fill(half val) {
  *(half2 *)(&data.x) = make_half2(val, val);
  *(half2 *)(&data.y) = make_half2(val, val);
}

INFER_INLINE_SPECIFIER void vec_t<half, 4>::load(const half *ptr) { data = *((uint2 *)ptr); }

INFER_INLINE_SPECIFIER void vec_t<half, 4>::store(half *ptr) const { *((uint2 *)ptr) = data; }

INFER_INLINE_SPECIFIER void vec_t<half, 4>::memcpy(half *dst, const half *src) { *((uint2 *)dst) = *((uint2 *)src); }

// half x 8 or more

template <size_t vec_size>
struct vec_t<half, vec_size> {
  uint4 data[vec_size / 8];
  INFER_INLINE_SPECIFIER half &operator[](size_t i) { return ((half *)data)[i]; }
  INFER_INLINE_SPECIFIER const half &operator[](size_t i) const { return ((const half *)data)[i]; }
  INFER_INLINE_SPECIFIER void fill(half val) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      *(half2 *)(&(data[i].x)) = make_half2(val, val);
      *(half2 *)(&(data[i].y)) = make_half2(val, val);
      *(half2 *)(&(data[i].z)) = make_half2(val, val);
      *(half2 *)(&(data[i].w)) = make_half2(val, val);
    }
  }
  INFER_INLINE_SPECIFIER void load(const half *ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4 *)ptr)[i];
    }
  }
  INFER_INLINE_SPECIFIER void store(half *ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4 *)ptr)[i] = data[i];
    }
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, vec_size> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(half *dst, const half *src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4 *)dst)[i] = ((uint4 *)src)[i];
    }
  }
};

/******************* vec_t<nv_bfloat16> *******************/

// nv_bfloat16 x 1
template <>
struct vec_t<nv_bfloat16, 1> {
  nv_bfloat16 data;

  INFER_INLINE_SPECIFIER nv_bfloat16 &operator[](size_t i) { return ((nv_bfloat16 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const nv_bfloat16 &operator[](size_t i) const { return ((const nv_bfloat16 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(nv_bfloat16 val);
  INFER_INLINE_SPECIFIER void load(const nv_bfloat16 *ptr);
  INFER_INLINE_SPECIFIER void store(nv_bfloat16 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 1> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src);
};

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 1>::fill(nv_bfloat16 val) { data = val; }

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 1>::load(const nv_bfloat16 *ptr) { data = *ptr; }

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 1>::store(nv_bfloat16 *ptr) const { *ptr = data; }

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 1>::memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src) { *dst = *src; }

// nv_bfloat16 x 2
template <>
struct vec_t<nv_bfloat16, 2> {
  nv_bfloat162 data;

  INFER_INLINE_SPECIFIER nv_bfloat16 &operator[](size_t i) { return ((nv_bfloat16 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const nv_bfloat16 &operator[](size_t i) const { return ((const nv_bfloat16 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(nv_bfloat16 val);
  INFER_INLINE_SPECIFIER void load(const nv_bfloat16 *ptr);
  INFER_INLINE_SPECIFIER void store(nv_bfloat16 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 2> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src);
};

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 2>::fill(nv_bfloat16 val) { data = make_bfloat162(val, val); }

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 2>::load(const nv_bfloat16 *ptr) { data = *((nv_bfloat162 *)ptr); }

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 2>::store(nv_bfloat16 *ptr) const { *((nv_bfloat162 *)ptr) = data; }

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 2>::memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src) {
  *((nv_bfloat162 *)dst) = *((nv_bfloat162 *)src);
}

// nv_bfloat16 x 4

template <>
struct vec_t<nv_bfloat16, 4> {
  uint2 data;

  INFER_INLINE_SPECIFIER nv_bfloat16 &operator[](size_t i) { return ((nv_bfloat16 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const nv_bfloat16 &operator[](size_t i) const { return ((const nv_bfloat16 *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(nv_bfloat16 val);
  INFER_INLINE_SPECIFIER void load(const nv_bfloat16 *ptr);
  INFER_INLINE_SPECIFIER void store(nv_bfloat16 *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 4> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src);
};

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 4>::fill(nv_bfloat16 val) {
  *(nv_bfloat162 *)(&data.x) = make_bfloat162(val, val);
  *(nv_bfloat162 *)(&data.y) = make_bfloat162(val, val);
}

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 4>::load(const nv_bfloat16 *ptr) { data = *((uint2 *)ptr); }

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 4>::store(nv_bfloat16 *ptr) const { *((uint2 *)ptr) = data; }

INFER_INLINE_SPECIFIER void vec_t<nv_bfloat16, 4>::memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src) {
  *((uint2 *)dst) = *((uint2 *)src);
}

// nv_bfloat16 x 8 or more

template <size_t vec_size>
struct vec_t<nv_bfloat16, vec_size> {
  uint4 data[vec_size / 8];

  INFER_INLINE_SPECIFIER nv_bfloat16 &operator[](size_t i) { return ((nv_bfloat16 *)data)[i]; }
  INFER_INLINE_SPECIFIER const nv_bfloat16 &operator[](size_t i) const { return ((const nv_bfloat16 *)data)[i]; }
  INFER_INLINE_SPECIFIER void fill(nv_bfloat16 val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(nv_bfloat162 *)(&(data[i].x)) = make_bfloat162(val, val);
      *(nv_bfloat162 *)(&(data[i].y)) = make_bfloat162(val, val);
      *(nv_bfloat162 *)(&(data[i].z)) = make_bfloat162(val, val);
      *(nv_bfloat162 *)(&(data[i].w)) = make_bfloat162(val, val);
    }
  }
  INFER_INLINE_SPECIFIER void load(const nv_bfloat16 *ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((uint4 *)ptr)[i];
    }
  }
  INFER_INLINE_SPECIFIER void store(nv_bfloat16 *ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4 *)ptr)[i] = data[i];
    }
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, vec_size> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(nv_bfloat16 *dst, const nv_bfloat16 *src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((uint4 *)dst)[i] = ((uint4 *)src)[i];
    }
  }
};

/******************* vec_t<float> *******************/

// float x 1

template <>
struct vec_t<float, 1> {
  float data;

  INFER_INLINE_SPECIFIER float &operator[](size_t i) { return ((float *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const float &operator[](size_t i) const { return ((const float *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(float val);
  INFER_INLINE_SPECIFIER void load(const float *ptr);
  INFER_INLINE_SPECIFIER void store(float *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 1> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }

  INFER_INLINE_SPECIFIER static void memcpy(float *dst, const float *src);
};

INFER_INLINE_SPECIFIER void vec_t<float, 1>::fill(float val) { data = val; }

INFER_INLINE_SPECIFIER void vec_t<float, 1>::load(const float *ptr) { data = *ptr; }

INFER_INLINE_SPECIFIER void vec_t<float, 1>::store(float *ptr) const { *ptr = data; }

INFER_INLINE_SPECIFIER void vec_t<float, 1>::memcpy(float *dst, const float *src) { *dst = *src; }

// float x 2

template <>
struct vec_t<float, 2> {
  float2 data;

  INFER_INLINE_SPECIFIER float &operator[](size_t i) { return ((float *)(&data))[i]; }
  INFER_INLINE_SPECIFIER const float &operator[](size_t i) const { return ((const float *)(&data))[i]; }
  INFER_INLINE_SPECIFIER void fill(float val);
  INFER_INLINE_SPECIFIER void load(const float *ptr);
  INFER_INLINE_SPECIFIER void store(float *ptr) const;
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, 2> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }
  INFER_INLINE_SPECIFIER static void memcpy(float *dst, const float *src);
};

INFER_INLINE_SPECIFIER void vec_t<float, 2>::fill(float val) { data = make_float2(val, val); }

INFER_INLINE_SPECIFIER void vec_t<float, 2>::load(const float *ptr) { data = *((float2 *)ptr); }

INFER_INLINE_SPECIFIER void vec_t<float, 2>::store(float *ptr) const { *((float2 *)ptr) = data; }

INFER_INLINE_SPECIFIER void vec_t<float, 2>::memcpy(float *dst, const float *src) {
  *((float2 *)dst) = *((float2 *)src);
}

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  float4 data[vec_size / 4];

  INFER_INLINE_SPECIFIER float &operator[](size_t i) { return ((float *)(data))[i]; }
  INFER_INLINE_SPECIFIER const float &operator[](size_t i) const { return ((const float *)(data))[i]; }
  INFER_INLINE_SPECIFIER void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  INFER_INLINE_SPECIFIER void load(const float *ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4 *)ptr)[i];
    }
  }
  INFER_INLINE_SPECIFIER void store(float *ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4 *)ptr)[i] = data[i];
    }
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_from(const vec_t<T, vec_size> &src) {
    cast_from_impl(src, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_load(const T *ptr) {
    cast_load_impl(ptr, *this);
  }
  template <typename T>
  INFER_INLINE_SPECIFIER void cast_store(T *ptr) const {
    cast_store_impl(*this, ptr);
  }
  INFER_INLINE_SPECIFIER static void memcpy(float *dst, const float *src) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4 *)dst)[i] = ((float4 *)src)[i];
    }
  }
};

/******************* vec_t type cast *******************/

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<half, vec_size> &src, vec_t<float, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = float(src.data);
  } else {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((float2 *)(&dst.data))[i] = __half22float2(((half2 *)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<float, vec_size> &src, vec_t<half, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = half(src.data);
  } else {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((half2 *)(&dst.data))[i] = __float22half2_rn(((float2 *)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<nv_bfloat16, vec_size> &src, vec_t<float, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = float(src.data);
  } else {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((float2 *)(&dst.data))[i] = __bfloat1622float2(((nv_bfloat162 *)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<float, vec_size> &src, vec_t<nv_bfloat16, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = nv_bfloat16(src.data);
  } else {
#pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((nv_bfloat162 *)(&dst.data))[i] = __float22bfloat162_rn(((float2 *)(&src.data))[i]);
    }
  }
}

#ifdef FLASHINFER_USE_FP8

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<__nv_fp8_e4m3, vec_size> &src, vec_t<float, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = float(src.data);
  } else if constexpr (vec_size == 2) {
    *(float2 *)(&dst.data) = float2(*(__nv_fp8x2_e4m3 *)(&src.data));
  } else {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4 *)(&dst.data))[i] = float4(((__nv_fp8x4_e4m3 *)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<__nv_fp8_e4m3, vec_size> &src, vec_t<half, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = float(src.data);
  } else {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((half2 *)(&dst.data))[i] = half2(((__nv_fp8x2_e4m3 *)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<float, vec_size> &src, vec_t<__nv_fp8_e4m3, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = __nv_fp8_e4m3(src.data);
  } else if constexpr (vec_size == 2) {
    *(__nv_fp8x2_e4m3 *)(&dst.data) = __nv_fp8x2_e4m3(*(float2 *)(&src.data));
  } else {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((__nv_fp8x4_e4m3 *)(&dst.data))[i] = __nv_fp8x4_e4m3(((float4 *)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<half, vec_size> &src, vec_t<__nv_fp8_e4m3, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = __nv_fp8_e4m3(src.data);
  } else if constexpr (vec_size == 2) {
    *(__nv_fp8x2_e4m3 *)(&dst.data) = __nv_fp8x2_e4m3(*(half2 *)(&src.data));
  } else {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      // NOTE(Zihao): need to double check if we properly handle flo and fhi
      ((__nv_fp8x4_e4m3 *)(&dst.data))[i] =
          __nv_fp8x4_e4m3(((half2 *)(&src.data))[i * 2], ((half2 *)(&src.data))[i * 2 + 1]);
    }
  }
}

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<__nv_fp8_e5m2, vec_size> &src, vec_t<float, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = float(src.data);
  } else if constexpr (vec_size == 2) {
    *(float2 *)(&dst.data) = float2(*(__nv_fp8x2_e5m2 *)(&src.data));
  } else {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4 *)(&dst.data))[i] = float4(((__nv_fp8x4_e5m2 *)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<__nv_fp8_e5m2, vec_size> &src, vec_t<half, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = float(src.data);
  } else {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 2; ++i) {
      ((half2 *)(&dst.data))[i] = half2(((__nv_fp8x2_e5m2 *)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<float, vec_size> &src, vec_t<__nv_fp8_e5m2, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = __nv_fp8_e5m2(src.data);
  } else if constexpr (vec_size == 2) {
    *(__nv_fp8x2_e5m2 *)(&dst.data) = __nv_fp8x2_e5m2(*(float2 *)(&src.data));
  } else {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((__nv_fp8x4_e5m2 *)(&dst.data))[i] = __nv_fp8x4_e5m2(((float4 *)(&src.data))[i]);
    }
  }
}

template <size_t vec_size>
INFER_INLINE_SPECIFIER void cast_from_impl(const vec_t<half, vec_size> &src, vec_t<__nv_fp8_e5m2, vec_size> &dst) {
  if constexpr (vec_size == 1) {
    dst.data = __nv_fp8_e4m3(src.data);
  } else if constexpr (vec_size == 2) {
    *(__nv_fp8x2_e5m2 *)(&dst.data) = __nv_fp8x2_e5m2(*(half2 *)(&src.data));
  } else {
#  pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      // NOTE(Zihao): need to double check if we properly handle flo and fhi
      ((__nv_fp8x4_e5m2 *)(&dst.data))[i] =
          __nv_fp8x4_e5m2(((half2 *)(&src.data))[i * 2], ((half2 *)(&src.data))[i * 2 + 1]);
    }
  }
}

#endif  // FLASHINFER_USE_FP8

}  // namespace nvidia
}  // namespace llm_kernels