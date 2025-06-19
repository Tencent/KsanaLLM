/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/ascend/ascend_context.h"

#include <numeric>

#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <>
void ContextT<DEVICE_TYPE_ASCEND>::InitializeExtension() {
  ext = new AscendContextExtension<DEVICE_TYPE_ASCEND>(this);
  ext->Initialize();
}

template <>
void ContextT<DEVICE_TYPE_ASCEND>::DestroyExtension() {
  ext->Destroy();
  delete ext;
}

template <int T>
void AscendContextExtension<T>::InitHcclParam() {
  KLLM_LOG_DEBUG << "Init ascend hccl param.";
  if (base_ptr_->tensor_parallel_size_ <= 1) {
    return;
  }
  rank_ids_.resize(base_ptr_->tensor_parallel_size_, int32_t(0));
  std::iota(rank_ids_.begin() + 1, rank_ids_.end(), int32_t(1));
  hccl_params_.resize(base_ptr_->tensor_parallel_size_);

  HCCL_CHECK(HcclCommInitAll(base_ptr_->tensor_parallel_size_, rank_ids_.data(), hccl_params_.data()));
}

template <int T>
void AscendContextExtension<T>::Initialize() {
  InitHcclParam();

  // reset device id
  ACL_CHECK(aclrtSetDevice(base_ptr_->defalt_device_id_));
}

template <int T>
void AscendContextExtension<T>::Destroy() {
  KLLM_LOG_DEBUG << "Destroy ascend context.";
  if (base_ptr_->tensor_parallel_size_ <= 1) {
    return;
  }
  for (int worker_id = 0; worker_id < base_ptr_->tensor_parallel_size_; ++worker_id) {
    HCCL_CHECK(HcclCommDestroy(hccl_params_[worker_id]));
  }
}

}  // namespace ksana_llm
