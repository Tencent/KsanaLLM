/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/device_types.h"

namespace ksana_llm {

size_t GetTypeSize(DataType dtype) {
  static const std::unordered_map<DataType, size_t> type_map{
      {TYPE_INVALID, 0},
      {TYPE_BOOL, sizeof(bool)},
      {TYPE_BYTES, sizeof(char)},
      {TYPE_UINT8, sizeof(uint8_t)},
      {TYPE_UINT16, sizeof(uint16_t)},
      {TYPE_UINT32, sizeof(uint32_t)},
      {TYPE_UINT64, sizeof(uint64_t)},
      {TYPE_INT8, sizeof(int8_t)},
      {TYPE_INT16, sizeof(int16_t)},
      {TYPE_INT32, sizeof(int32_t)},
      {TYPE_INT64, sizeof(int64_t)},
      {TYPE_FP16, sizeof(float16)},
      {TYPE_FP32, sizeof(float)},
      {TYPE_FP64, sizeof(double)},
      {TYPE_POINTER, sizeof(void*)},
      {TYPE_BF16, sizeof(bfloat16)},
#ifdef ENABLE_FP8
      {TYPE_FP8_E4M3, sizeof(__nv_fp8_e4m3)},
      {TYPE_FP8_E5M2, sizeof(__nv_fp8_e5m2)},
#else
      {TYPE_FP8_E4M3, sizeof(uint8_t)},
      {TYPE_FP8_E5M2, sizeof(uint8_t)},
#endif
  };
  return type_map.at(dtype);
}

c10::ScalarType GetTorchTypeFromDataType(const DataType& data_type) {
  c10::ScalarType torch_type = torch::kFloat32;
  switch (data_type) {
    case TYPE_BF16:
      torch_type = c10::kBFloat16;
      break;
    case TYPE_FP16:
      torch_type = torch::kFloat16;
      break;
    case TYPE_FP32:
      torch_type = torch::kFloat32;
      break;
    case TYPE_INT32:
      torch_type = torch::kInt32;
      break;
#if defined(ENABLE_CUDA)
    case TYPE_INT8:
      torch_type = torch::kInt8;
      break;
    case TYPE_UINT8:
      torch_type = torch::kUInt8;
      break;
#  ifdef ENABLE_FP8_TORCH
    case TYPE_FP8_E4M3:
      torch_type = torch::kFloat8_e4m3fn;
      break;
    case TYPE_FP8_E5M2:
      torch_type = torch::kFloat8_e5m2;
      break;
#  endif
#endif
    default:
      break;
  }
  return torch_type;
}

}  // namespace ksana_llm
