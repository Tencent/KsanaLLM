/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "atb/atb_infer.h"

#include "common.h"

namespace llm_kernels {
namespace utils {

class ATBOperationExecutor {
 public:
  ATBOperationExecutor(){};

  ~ATBOperationExecutor() {
    ResetVariantPack();
    if (operation_ != nullptr) {
      ATB_CHECK_RET(atb::DestroyOperation(operation_));
    }
  };

  template <typename OPParamType>
  void Init(const int rank, const OPParamType& param) {
    rank_ = rank;
    ATB_CHECK_RET(atb::CreateOperation(param, &operation_));
  }

  void ResetVariantPack();

  void SetInputTensor(void* addr_ptr, const std::vector<size_t> shape, const aclDataType dtype,
                      const aclFormat format = aclFormat::ACL_FORMAT_ND);

  void SetOutputTensor(void* addr_ptr, const std::vector<size_t> shape, const aclDataType dtype,
                       const aclFormat format = aclFormat::ACL_FORMAT_ND);

  void Run(atb::Context* context, void (*ws_func)(size_t, void**));

  void SetOperation(atb::Operation* operation) { operation_ = operation; }

  atb::Operation* GetOperation() { return operation_; }

  void SetRank(int rank) { rank_ = rank; }

  int GetRank() { return rank_; }

  size_t GetWorkspaceSize(atb::Context* context);

  void Run(atb::Context* context, void* workspace_ptr);

 private:
  int rank_{0};
  atb::Operation* operation_{nullptr};
  void* workspace_ptr_{nullptr};
  size_t workspace_size_{0};
  uint64_t in_tensor_num_{0};
  uint64_t out_tensor_num_{0};
  atb::VariantPack variant_pack_;
};

}  // namespace utils
}  // namespace llm_kernels