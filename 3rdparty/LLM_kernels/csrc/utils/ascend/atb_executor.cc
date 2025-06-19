/**
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "atb_executor.h"

#include "atb/utils.h"

namespace llm_kernels {
namespace utils {

void ATBOperationExecutor::ResetVariantPack() {
  variant_pack_.inTensors.resize(0);
  variant_pack_.outTensors.resize(0);
  workspace_ptr_ = nullptr;
  workspace_size_ = 0;
  in_tensor_num_ = 0;
  out_tensor_num_ = 0;
}

void ATBOperationExecutor::SetInputTensor(void* addr_ptr, const std::vector<size_t> shape, const aclDataType dtype,
                                          const aclFormat format) {
  variant_pack_.inTensors.resize(in_tensor_num_ + 1);
  atb::Tensor& atb_tensor = variant_pack_.inTensors[in_tensor_num_++];
  atb_tensor.desc.format = format;
  atb_tensor.desc.shape.dimNum = shape.size();
  for (auto i = 0; i < shape.size(); ++i) {
    atb_tensor.desc.shape.dims[i] = shape[i];
  }
  atb_tensor.desc.dtype = dtype;
  atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
  atb_tensor.hostData = const_cast<void*>(addr_ptr);
  atb_tensor.deviceData = const_cast<void*>(addr_ptr);
}

void ATBOperationExecutor::SetOutputTensor(void* addr_ptr, const std::vector<size_t> shape, const aclDataType dtype,
                                           const aclFormat format) {
  variant_pack_.outTensors.resize(out_tensor_num_ + 1);
  atb::Tensor& atb_tensor = variant_pack_.outTensors[out_tensor_num_++];
  atb_tensor.desc.format = format;
  atb_tensor.desc.shape.dimNum = shape.size();
  for (auto i = 0; i < shape.size(); ++i) {
    atb_tensor.desc.shape.dims[i] = shape[i];
  }
  atb_tensor.desc.dtype = dtype;
  atb_tensor.dataSize = atb::Utils::GetTensorSize(atb_tensor);
  atb_tensor.hostData = const_cast<void*>(addr_ptr);
  atb_tensor.deviceData = const_cast<void*>(addr_ptr);
}

void ATBOperationExecutor::Run(atb::Context* context, void (*ws_func)(size_t, void**)) {
  ATB_CHECK_RET(operation_->Setup(variant_pack_, workspace_size_, context));
  ws_func(workspace_size_, &workspace_ptr_);
  ATB_CHECK_RET(
      operation_->Execute(variant_pack_, reinterpret_cast<uint8_t*>(workspace_ptr_), workspace_size_, context));
}

size_t ATBOperationExecutor::GetWorkSpaceSize(atb::Context* context) {
  ATB_CHECK_RET(operation_->Setup(variant_pack_, workspace_size_, context));
  return workspace_size_;
}

void ATBOperationExecutor::Run(atb::Context* context, void* workspace_ptr) {
  ATB_CHECK_RET(
      operation_->Execute(variant_pack_, reinterpret_cast<uint8_t*>(workspace_ptr), workspace_size_, context));
}

}  // namespace utils
}  // namespace llm_kernels
