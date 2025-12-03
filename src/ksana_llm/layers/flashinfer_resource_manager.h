/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <unordered_map>
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// Unified manager for all FlashInfer runtime resources.
// Manages workspace and prefill helper instances shared across layers on the same rank.
class FlashInferResourceManager {
 public:
  // Resource for a single rank.
  struct Resources {
    // Workspace resources.
    void* pinned_host_workspace = nullptr;
    std::shared_ptr<Tensor> device_workspace_buffer = nullptr;

    // (float buffer / init buffer) default value
    size_t kPinnedWorkspaceSize = 8 * 1024 * 1024;  // Recommended by FlashInfer doc.
    size_t device_workspace_size = 0;
    bool is_workspace_initialized = false;

    // Helper resource.
    std::shared_ptr<void> prefill_helper = nullptr;
    bool is_helper_initialized = false;
  };

  // Get pinned host workspace pointer for specific rank.
  // This workspace is shared across all layers on the same rank.
  static void* GetPinnedHostWorkspace(int num_heads, int num_kv_heads, int head_dim, int rank);

  // Get GPU device workspace buffer for specific rank.
  // This workspace is shared across all layers on the same rank.
  static std::shared_ptr<Tensor>& GetDeviceWorkspace(int num_heads, int num_kv_heads, int head_dim, int rank);

  // Set prefill helper for specific rank.
  // This helper is shared across all layers on the same rank.
  static void SetPrefillHelper(int rank, std::shared_ptr<void> helper);

  // Get prefill helper for specific rank.
  // Returns nullptr if helper has not been set.
  static std::shared_ptr<void> GetPrefillHelper(int rank);

  // Free up all resources.
  static void FreeAllResources();

  // Free up resources for specific rank.
  static void FreeRankResources(int rank);

 private:
  // Initialize workspace resources for specific rank.
  static void InitializeWorkspaceInternal(int num_heads, int num_kv_heads, int head_dim, int rank);

  // Static storage for all rank resources.
  static inline std::unordered_map<int, Resources> rank_resources_;
};

}  // namespace ksana_llm
