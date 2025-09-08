/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "csrc/utils/nvidia/assert.h"
#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels::nvidia::tensorrt_llm::dev {

enum ScalarType { Long, Float8_e4m3fn, QUInt4x2, Int, Float, BFloat16, Half };

template <typename T>
inline ScalarType GetScalarType();
#define GET_SCALAR_TYPE(T, DATA_TYPE)    \
  template <>                            \
  inline ScalarType GetScalarType<T>() { \
    return DATA_TYPE;                    \
  }
GET_SCALAR_TYPE(float, ScalarType::Float);
GET_SCALAR_TYPE(half, ScalarType::Half);
GET_SCALAR_TYPE(__nv_bfloat16, ScalarType::BFloat16);
GET_SCALAR_TYPE(int32_t, ScalarType::Int);
#undef GET_SCALAR_TYPE

struct Tensor {
  void* data;
  std::vector<size_t> shape;
  ScalarType dtype;

  inline Tensor(void* data, const std::vector<size_t>& shape, ScalarType dtype)
      : data(data), shape(shape), dtype(dtype) {}

  inline Tensor() : data(nullptr), dtype(ScalarType::Float) {}
};

struct WorkspaceInfo {
  void* workspace{};
  void* src_to_dest_map{};
};

}  // namespace llm_kernels::nvidia::tensorrt_llm::dev
