/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/ascend/hccl_utils.h"
#include "ksana_llm/utils/common_context.h"

namespace ksana_llm {

// The class used for ascend extension.
template <int T>
struct AscendContextExtension {
 public:
  explicit AscendContextExtension(ContextT<T>* base_ptr) { base_ptr_ = base_ptr; }

  // Initialize and destroy extension.
  void Initialize();
  void Destroy();

  std::vector<HcclComm>& GetHCCLComm() { return hccl_params_; }

  const bool IsMulticastSupported() const { return false; }

 private:
  ContextT<T>* base_ptr_ = nullptr;

  // init nccl handle
  void InitHcclParam();

  // hccl comms
  std::vector<HcclComm> hccl_params_;
  std::vector<int32_t> rank_ids_;
};

template <>
struct ExtensionTypeTraits<DEVICE_TYPE_ASCEND> {
  typedef AscendContextExtension<DEVICE_TYPE_ASCEND> value_type;
};

// 构造扩展类对象
template <>
void ContextT<DEVICE_TYPE_ASCEND>::InitializeExtension();

// 销毁扩展类对象
template <>
void ContextT<DEVICE_TYPE_ASCEND>::DestroyExtension();

}  // namespace ksana_llm
