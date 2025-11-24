/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <nvml.h>

#include "fmt/core.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

#define NCCL_CHECK(cmd)                                                                                        \
  do {                                                                                                         \
    ncclResult_t r = cmd;                                                                                      \
    if (r != ncclSuccess) {                                                                                    \
      KLLM_LOG_ERROR << fmt::format("NCCL runtime error:{} {}:{}", ncclGetErrorString(r), __FILE__, __LINE__); \
      exit(RetCode::RET_INVALID_ARGUMENT);                                                                     \
    }                                                                                                          \
  } while (0)

#define NCCL_CHECK_CALL(cmd, call)                                                                           \
  do {                                                                                                       \
    ncclResult_t r = cmd;                                                                                    \
    if (r != ncclSuccess) {                                                                                  \
      KLLM_LOG_ERROR << fmt::format("NCCL {} runtime error:{} {}:{}", call, ncclGetErrorString(r), __FILE__, \
                                    __LINE__);                                                               \
      exit(RetCode::RET_INVALID_ARGUMENT);                                                                   \
    }                                                                                                        \
  } while (0)

#define NVML_CHECK(cmd)                                                                                     \
  do {                                                                                                      \
    nvmlReturn_t r = cmd;                                                                                   \
    if (r != NVML_SUCCESS) {                                                                                \
      KLLM_LOG_ERROR << fmt::format("NVML runtime error:{} {}:{}", nvmlErrorString(r), __FILE__, __LINE__); \
      exit(RetCode::RET_INVALID_ARGUMENT);                                                                  \
    }                                                                                                       \
  } while (0)

struct NCCLParam {
  int rank{0};
  int world_size{1};

  ncclUniqueId nccl_uid;
  ncclComm_t nccl_comm{nullptr};

  NCCLParam() : rank(0), world_size(1), nccl_comm(nullptr) {}
  NCCLParam(int rank, int world_size) : rank(rank), world_size(world_size) {}
  NCCLParam(NCCLParam const& param)
      : rank(param.rank), world_size(param.world_size), nccl_uid(param.nccl_uid), nccl_comm(param.nccl_comm) {}
  std::string toString() {
    return fmt::format("NCCLParam: rank={}, world_size=%{}, nccl_comm={}", rank, world_size,
                       reinterpret_cast<uintptr_t>(nccl_comm));
  }
};

template <typename T>
ncclDataType_t CastToNCCLDataType() {
  ncclDataType_t nccl_data_type;
  if (std::is_same<T, float>::value) {
    nccl_data_type = ncclFloat;
  } else if (std::is_same<T, half>::value) {
    nccl_data_type = ncclHalf;
  } else if (std::is_same<T, __nv_bfloat16>::value) {
    nccl_data_type = ncclBfloat16;
  } else if (std::is_same<T, int>::value) {
    nccl_data_type = ncclInt;
  } else if (std::is_same<T, char>::value) {
    nccl_data_type = ncclChar;
  } else if (std::is_same<T, bool>::value) {
    nccl_data_type = ncclInt8;
  } else {
    KLLM_LOG_ERROR << "Not supported type for casting to NCCL data type";
    exit(RetCode::RET_INVALID_ARGUMENT);
  }
  return nccl_data_type;
}

ncclResult_t DestroyNCCLParam(NCCLParam& param);

ncclUniqueId GenerateNCCLUniqueID();

// Convert data type to nccl data type.
Status GetNcclDataType(DataType dtype, ncclDataType_t& nccl_dtype);

// Convert data type to nccl data type.
inline ncclDataType_t GetNcclDataType(DataType dtype) {
  ncclDataType_t nccl_dtype;
  Status status = GetNcclDataType(dtype, nccl_dtype);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Not supported type for casting to NCCL data type. msg" << status.GetMessage();
    exit(RetCode::RET_INVALID_ARGUMENT);
  }
  return nccl_dtype;
}

}  // namespace ksana_llm
