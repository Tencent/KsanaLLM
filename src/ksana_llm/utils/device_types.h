/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>

// All supported device type.
#define DEVICE_TYPE_NVIDIA 0
#define DEVICE_TYPE_ASCEND 1
#define DEVICE_TYPE_ZIXIAO 2

// Unknown device type.
#define DEVICE_TYPE_UNKNOWN -1

// Select active device type.
#ifdef ENABLE_CUDA
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_NVIDIA
#elif defined(ENABLE_ACL)
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_ASCEND
#elif defined(ENABLE_TOPS)
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_ZIXIAO
#else
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_UNKNOWN
#endif

// Include necessary header files.
#ifdef ENABLE_CUDA
#  include <cublasLt.h>
#  include <cublas_v2.h>
#  include <cuda_runtime.h>
// Include cuda fp8 header
#  ifdef ENABLE_FP8
#    include <cuda_fp8.h>
#  endif
#endif

#ifdef ENABLE_ACL
#  include <acl/acl.h>
#  include <acl/acl_base.h>
#  include <acl/acl_rt.h>
#endif

namespace ksana_llm {

// The type define.
#if defined(ENABLE_CUDA)
typedef half float16;
#  ifdef ENABLE_BFLOAT16
typedef __nv_bfloat16 bfloat16;
#  endif
#  ifdef ENABLE_FP8
typedef __nv_fp8_e4m3 fp8e4m3;
typedef __nv_fp8_e5m2 fp8e5m2;
#  endif
#elif defined(ENABLE_ACL)
typedef aclFloat16 float16;
#  ifdef ENABLE_BFLOAT16
typedef int16_t bfloat16;
#  endif
#elif defined(ENABLE_TOPS)
typedef int16_t float16;
#  ifdef ENABLE_BFLOAT16
typedef uint16_t bfloat16;
#  endif
#endif

// All the available data format, for ascend aclTensor.
enum DataFormat {
#if defined(ENABLE_CUDA)
  FORMAT_DEFAULT
#elif defined(ENABLE_ACL)
  FORMAT_DEFAULT = aclFormat::ACL_FORMAT_ND,
  FORMAT_ND = aclFormat::ACL_FORMAT_ND,
  FORMAT_NZ = aclFormat::ACL_FORMAT_FRACTAL_NZ
#elif defined(ENABLE_TOPS)
  FORMAT_DEFAULT
#endif
};

#ifdef ENABLE_ACL
inline aclFormat GetACLFormat(DataFormat data_format) {
  switch (data_format) {
    case FORMAT_ND:
      return aclFormat::ACL_FORMAT_ND;
    case FORMAT_NZ:
      return aclFormat::ACL_FORMAT_FRACTAL_NZ;
    default:
      return aclFormat::ACL_FORMAT_ND;
  }
}

inline DataFormat GetTensorFormat(aclFormat data_format) {
  switch (data_format) {
    case aclFormat::ACL_FORMAT_ND:
      return FORMAT_DEFAULT;
    case aclFormat::ACL_FORMAT_FRACTAL_NZ:
      return FORMAT_NZ;
    default:
      return FORMAT_DEFAULT;
  }
}
#endif

// All the available data types.
enum DataType {
#if defined(ENABLE_CUDA)
  TYPE_INVALID,
  TYPE_BOOL,
  TYPE_UINT8,
  TYPE_UINT16,
  TYPE_UINT32,
  TYPE_UINT64,
  TYPE_INT8,
  TYPE_INT16,
  TYPE_INT32,
  TYPE_INT64,
  TYPE_BF16,
  TYPE_FP16,
  TYPE_FP32,
  TYPE_FP64,
  TYPE_BYTES,
  TYPE_FP8_E4M3,
  TYPE_I4_GROUP,
  TYPE_BLOCK_FP8_E4M3,
  TYPE_FP8_E5M2,
  TYPE_VOID,
  TYPE_POINTER
#elif defined(ENABLE_ACL)
  TYPE_INVALID = aclDataType::ACL_DT_UNDEFINED,
  TYPE_BOOL = aclDataType::ACL_BOOL,
  TYPE_UINT8 = aclDataType::ACL_INT8,
  TYPE_UINT16 = aclDataType::ACL_UINT16,
  TYPE_UINT32 = aclDataType::ACL_UINT32,
  TYPE_UINT64 = aclDataType::ACL_INT64,
  TYPE_INT8 = aclDataType::ACL_INT8,
  TYPE_INT16 = aclDataType::ACL_INT16,
  TYPE_INT32 = aclDataType::ACL_INT32,
  TYPE_INT64 = aclDataType::ACL_INT64,
  TYPE_BF16 = aclDataType::ACL_BF16,
  TYPE_FP16 = aclDataType::ACL_FLOAT16,
  TYPE_FP32 = aclDataType::ACL_FLOAT,
  TYPE_FP64 = aclDataType::ACL_DOUBLE,
  TYPE_BYTES = aclDataType::ACL_STRING,
  TYPE_FP8_E4M3 = aclDataType::ACL_DT_UNDEFINED - 1,
  TYPE_FP8_E5M2 = aclDataType::ACL_DT_UNDEFINED - 2,
  TYPE_I4_GROUP = aclDataType::ACL_DT_UNDEFINED - 3,
  TYPE_BLOCK_FP8_E4M3 = aclDataType::ACL_DT_UNDEFINED - 4,
  TYPE_VOID = aclDataType::ACL_DT_UNDEFINED - 5,
  TYPE_POINTER = aclDataType::ACL_DT_UNDEFINED - 6
#elif defined(ENABLE_TOPS)
  TYPE_INVALID,
  TYPE_FP32,
  TYPE_FP16,
  TYPE_BF16,
  TYPE_TF24,
  TYPE_TF32,
  TYPE_INT8,
  TYPE_UINT8,
  TYPE_INT32,
  TYPE_INT64,
  TYPE_UINT32,
  TYPE_UINT64,
  TYPE_BOOL,
  TYPE_BIT,
  TYPE_BIT2,
  TYPE_BIT4,
  TYPE_INT16,
  TYPE_FP8_E4M3,
  TYPE_FP8_E5M2,
  TYPE_FP64,
  TYPE_UINT16,
  TYPE_BYTES,
  TYPE_VOID,
  TYPE_POINTER,
  TYPE_I4_GROUP,
  TYPE_BLOCK_FP8_E4M3,
#endif
};

__attribute__((unused)) static std::string GetTypeString(DataType dtype) {
  static const std::unordered_map<DataType, std::string> type_map{{TYPE_INVALID, "invalid"},
                                                                  {TYPE_BOOL, "bool"},
                                                                  {TYPE_BYTES, "bytes"},
                                                                  {TYPE_UINT8, "uint8"},
                                                                  {TYPE_UINT16, "uint16"},
                                                                  {TYPE_UINT32, "uint32"},
                                                                  {TYPE_UINT64, "uint64"},
                                                                  {TYPE_INT8, "int8"},
                                                                  {TYPE_INT16, "int16"},
                                                                  {TYPE_INT32, "int32"},
                                                                  {TYPE_INT64, "int64"},
                                                                  {TYPE_FP16, "float16"},
                                                                  {TYPE_BF16, "bfloat16"},
                                                                  {TYPE_FP32, "float32"},
                                                                  {TYPE_FP64, "float64"},
                                                                  {TYPE_BYTES, "bytes"},
                                                                  {TYPE_FP8_E4M3, "fp8_e4m3"},
                                                                  {TYPE_I4_GROUP, "int4_group"},
                                                                  {TYPE_BLOCK_FP8_E4M3, "block_fp8_e4m3"},
                                                                  {TYPE_FP8_E5M2, "fp8_e5m2"},
                                                                  {TYPE_VOID, "void"},
                                                                  {TYPE_POINTER, "pointer"}};
  return type_map.count(dtype) ? type_map.at(dtype) : "invalid";
}

size_t GetTypeSize(DataType dtype);

// The memory device.
enum MemoryDevice { MEMORY_HOST, MEMORY_DEVICE };

// The memory location.
enum MemoryLocation { LOCATION_UNKNOWN, LOCATION_HOST, LOCATION_DEVICE, LOCATION_NVSHMEM };

// A dummy class used as a real defined class.
struct DummyClass {};

c10::ScalarType GetTorchTypeFromDataType(const DataType& data_type);

}  // namespace ksana_llm
