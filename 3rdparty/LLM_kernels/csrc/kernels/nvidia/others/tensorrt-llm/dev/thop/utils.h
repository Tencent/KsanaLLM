/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "csrc/utils/nvidia/assert.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels::nvidia::tensorrt_llm::dev {

/** ScalarType: 轻量数据类型枚举，用于内部表示（替代 torch::ScalarType），含常用浮点/整型与设备/量化格式。 */
enum ScalarType { Long, Int, Int8, UInt8, QUInt4x2, Float, BFloat16, Float16, Float8_e4m3fn, Byte, Char };

template <typename T>
inline ScalarType GetScalarType();
#define GET_SCALAR_TYPE(T, DATA_TYPE)    \
  template <>                            \
  inline ScalarType GetScalarType<T>() { \
    return DATA_TYPE;                    \
  }
GET_SCALAR_TYPE(float, ScalarType::Float);
GET_SCALAR_TYPE(half, ScalarType::Float16);
GET_SCALAR_TYPE(__nv_bfloat16, ScalarType::BFloat16);
GET_SCALAR_TYPE(__nv_fp8_e4m3, ScalarType::Float8_e4m3fn);
GET_SCALAR_TYPE(int32_t, ScalarType::Int);
GET_SCALAR_TYPE(char, ScalarType::Int8);
GET_SCALAR_TYPE(int8_t, ScalarType::Int8);
#undef GET_SCALAR_TYPE

struct Tensor {
  void* data;
  std::vector<size_t> shape;
  ScalarType dtype;

  inline Tensor(void* data, const std::vector<size_t>& shape, ScalarType dtype)
      : data(data), shape(shape), dtype(dtype) {}

  inline Tensor() : data(nullptr), dtype(ScalarType::Float) {}
};

}  // namespace llm_kernels::nvidia::tensorrt_llm::dev
