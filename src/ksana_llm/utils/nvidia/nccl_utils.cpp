/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "src/ksana_llm/utils/nvidia/nccl_utils.h"

namespace ksana_llm {

ncclResult_t DestroyNCCLParam(NCCLParam& param) {
  ncclResult_t status = ncclSuccess;
  if (param.nccl_comm != nullptr) {
    status = ncclCommDestroy(param.nccl_comm);
    param.nccl_comm = nullptr;
  }
  return status;
}

ncclUniqueId GenerateNCCLUniqueID() {
  ncclUniqueId nccl_uid;
  NCCL_CHECK(ncclGetUniqueId(&nccl_uid));
  return nccl_uid;
}

Status GetNcclDataType(DataType dtype, ncclDataType_t& nccl_dtype) {
  switch (dtype) {
    case DataType::TYPE_BYTES:
    case DataType::TYPE_BOOL: {
      nccl_dtype = ncclDataType_t::ncclChar;
      return Status();
    }
    case DataType::TYPE_INT8: {
      nccl_dtype = ncclDataType_t::ncclInt8;
      return Status();
    }
    case DataType::TYPE_UINT8: {
      nccl_dtype = ncclDataType_t::ncclUint8;
      return Status();
    }
    case DataType::TYPE_UINT32: {
      nccl_dtype = ncclDataType_t::ncclUint32;
      return Status();
    }
    case DataType::TYPE_UINT64: {
      nccl_dtype = ncclDataType_t::ncclUint64;
      return Status();
    }
    case DataType::TYPE_INT32: {
      nccl_dtype = ncclDataType_t::ncclInt32;
      return Status();
    }
    case DataType::TYPE_INT64: {
      nccl_dtype = ncclDataType_t::ncclInt64;
      return Status();
    }
    case DataType::TYPE_BF16: {
      nccl_dtype = ncclDataType_t::ncclBfloat16;
      return Status();
    }
    case DataType::TYPE_FP16: {
      nccl_dtype = ncclDataType_t::ncclFloat16;
      return Status();
    }
    case DataType::TYPE_FP32: {
      nccl_dtype = ncclDataType_t::ncclFloat32;
      return Status();
    }
    case DataType::TYPE_FP64: {
      nccl_dtype = ncclDataType_t::ncclFloat64;
      return Status();
    }
    case DataType::TYPE_BLOCK_FP8_E4M3:
    case DataType::TYPE_INT16:
    case DataType::TYPE_UINT16:
    case DataType::TYPE_INVALID:
    case DataType::TYPE_FP8_E4M3:
    case DataType::TYPE_I4_GROUP:
    case DataType::TYPE_FP8_E5M2:
    case DataType::TYPE_VOID:
    case DataType::TYPE_POINTER: {
      return Status(RET_INVALID_ARGUMENT, FormatStr("Not supported dtype %d", static_cast<int>(dtype)));
    }
    default: {
      return Status(RET_INVALID_ARGUMENT, FormatStr("Unknown dtype %d", static_cast<int>(dtype)));
    }
  }
}

}  // namespace ksana_llm
