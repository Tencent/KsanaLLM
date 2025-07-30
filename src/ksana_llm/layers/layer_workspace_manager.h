/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/batched_matmul_layer.h"
#include "ksana_llm/layers/blockwise_matmul_layer.h"
#include "ksana_llm/layers/cutlass_matmul_layer.h"
#include "ksana_llm/layers/fp8_matmul_layer.h"
#include "ksana_llm/layers/fp8_moe_layer.h"
#include "ksana_llm/layers/machete_matmul_layer.h"
#include "ksana_llm/layers/marlin_matmul_layer.h"
#include "ksana_llm/layers/marlin_moe_layer.h"
#include "ksana_llm/layers/matmul_layer.h"
#include "ksana_llm/layers/moe_layer.h"
#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// Assumption: workspaces of all layers can be reused.
// TODO(robertyuan): minimize workspace buffer
template <typename T>
class LayerWorkspaceManager {
 public:
  explicit LayerWorkspaceManager(const int rank) : rank_(rank) {}

  ~LayerWorkspaceManager() {}

  std::shared_ptr<Tensor>& GetWorkspace(size_t workspace_size) {
    if (workspace_buffer_ == nullptr) {
      if (workspace_size > 0) {
        KLLM_LOG_DEBUG << fmt::format("Rank[{}] Create WorkSpace Buffer: {}", rank_, workspace_size);
        workspace_buffer_ = std::shared_ptr<Tensor>(
            new Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT8, {workspace_size}, rank_));
      } else {
        KLLM_LOG_DEBUG << fmt::format("Rank[{}] No need any WorkSpace Buffer", rank_);
      }
    } else {
      if (workspace_buffer_->GetTotalBytes() < workspace_size) {
        KLLM_LOG_DEBUG << fmt::format("Rank[{}] Increase WorkSpace Buffer from: {} to: {}", rank_,
                                      workspace_buffer_->GetTotalBytes(), workspace_size);
        workspace_buffer_.reset();
        workspace_buffer_ = std::shared_ptr<Tensor>(
            new Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT8, {workspace_size}, rank_));
      } else {
        KLLM_LOG_DEBUG << fmt::format("Rank[{}] WorkSpace Buffer {} is big enough", rank_,
                                      workspace_buffer_->GetTotalBytes());
      }
    }
    return workspace_buffer_;
  }

  std::shared_ptr<Tensor> GetWorkspaceBuffer() const { return workspace_buffer_; }

 private:
  int rank_;
  std::shared_ptr<Tensor> workspace_buffer_ = nullptr;
};

}  // namespace ksana_llm
