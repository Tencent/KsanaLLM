/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <torch/torch.h>

template <typename T>
torch::ScalarType GetTorchDataType();
#define GET_TORCH_DATA_TYPE(T, TORCH_TYPE)  \
  template <>                               \
  torch::ScalarType GetTorchDataType<T>() { \
    return TORCH_TYPE;                      \
  }
GET_TORCH_DATA_TYPE(float, torch::kFloat32);
GET_TORCH_DATA_TYPE(half, torch::kFloat16);
GET_TORCH_DATA_TYPE(__nv_bfloat16, torch::kBFloat16);
GET_TORCH_DATA_TYPE(int32_t, torch::kInt32);
#undef GET_TORCH_DATA_TYPE

template <typename T>
std::string GetDataTypeName();
#define GET_DATA_TYPE_NAME(T, NAME)  \
  template <>                        \
  std::string GetDataTypeName<T>() { \
    return NAME;                     \
  }
GET_DATA_TYPE_NAME(half, " float16");
GET_DATA_TYPE_NAME(__nv_bfloat16, "bfloat16");
#undef GET_DATA_TYPE_NAME
