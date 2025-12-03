/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flashinfer_resource_manager.h"
#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/attention/flashinfer_attention/flashinfer_prefill.h"
#  include "ksana_llm/utils/nvidia/cuda_utils.h"
#endif

namespace ksana_llm {

void* FlashInferResourceManager::GetPinnedHostWorkspace(int num_heads, int num_kv_heads, int head_dim, int rank) {
  auto it = rank_resources_.find(rank);
  if (it == rank_resources_.end() || !it->second.is_workspace_initialized) {
    InitializeWorkspaceInternal(num_heads, num_kv_heads, head_dim, rank);
  }

  return rank_resources_[rank].pinned_host_workspace;
}

std::shared_ptr<Tensor>& FlashInferResourceManager::GetDeviceWorkspace(int num_heads, int num_kv_heads, int head_dim,
                                                                       int rank) {
  auto it = rank_resources_.find(rank);
  if (it == rank_resources_.end() || !it->second.is_workspace_initialized) {
    InitializeWorkspaceInternal(num_heads, num_kv_heads, head_dim, rank);
  }

  return rank_resources_[rank].device_workspace_buffer;
}

void FlashInferResourceManager::SetPrefillHelper(int rank, std::shared_ptr<void> helper) {
  auto it = rank_resources_.find(rank);
  if (it != rank_resources_.end() && it->second.is_helper_initialized) {
    KLLM_THROW(
        fmt::format("Rank[{}] FlashInfer prefill helper already initialized. "
                    "Each rank should only have one unique helper.",
                    rank));
  }

  rank_resources_[rank].prefill_helper = helper;
  rank_resources_[rank].is_helper_initialized = true;
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Set FlashInfer prefill helper", rank);
}

std::shared_ptr<void> FlashInferResourceManager::GetPrefillHelper(int rank) {
  auto it = rank_resources_.find(rank);
  if (it == rank_resources_.end() || !it->second.is_helper_initialized) {
    return nullptr;
  }

return it->second.prefill_helper;
}

void FlashInferResourceManager::FreeRankResources(int rank) {
  auto it = rank_resources_.find(rank);
  if (it == rank_resources_.end()) {
    return;
  }

  auto& resource_data = it->second;

#ifdef ENABLE_CUDA
  if (resource_data.is_workspace_initialized) {
      // Free pinned host memory.
    cudaFreeHost(resource_data.pinned_host_workspace);
    resource_data.pinned_host_workspace = nullptr;
    KLLM_LOG_DEBUG << fmt::format("Rank[{}] Freed up pinned host workspace", rank);

    // Free device workspace buffer.
    resource_data.device_workspace_buffer.reset();
    resource_data.is_workspace_initialized = false;
    KLLM_LOG_DEBUG << fmt::format("Rank[{}] Freed up device workspace buffer", rank);
  }
#endif

  // Free helper.
  if (resource_data.is_helper_initialized) {
    resource_data.prefill_helper.reset();
    resource_data.is_helper_initialized = false;
    KLLM_LOG_DEBUG << fmt::format("Rank[{}] Reset prefill helper", rank);
  }

  // Remove from map.
  rank_resources_.erase(it);
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Freed up all FlashInfer resources", rank);
}

void FlashInferResourceManager::FreeAllResources() {
  std::vector<int> ranks;
  for (const auto& pair : rank_resources_) {
    ranks.push_back(pair.first);
  }

  // Free up FlashInfer resources of each rank.
  for (int rank : ranks) {
    FreeRankResources(rank);
  }

  KLLM_LOG_DEBUG << "Freed up FlashInfer resources of all ranks";
}

void FlashInferResourceManager::InitializeWorkspaceInternal(int num_heads, int num_kv_heads, int head_dim, int rank) {
  auto& resource_data = rank_resources_[rank];

  if (resource_data.is_workspace_initialized) {
    return;
  }
  // Compute the GPU memory size required by FlashInfer.
  resource_data.device_workspace_size =
      llm_kernels::nvidia::GetFlashInferDeviceWorkspaceSize(num_heads, num_kv_heads, head_dim);

#ifdef ENABLE_CUDA
  // Allocate pinned host memory workspace for FlashInfer kernels.
  CUDA_CHECK(cudaMallocHost(&resource_data.pinned_host_workspace, resource_data.kPinnedWorkspaceSize));
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Allocated pinned host workspace: {} bytes for FlashInfer", rank,
                                resource_data.kPinnedWorkspaceSize);

  // Allocate GPU device memory workspace for FlashInfer kernels.
  resource_data.device_workspace_buffer =
      std::make_shared<Tensor>(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT8,
                               std::vector<size_t>{resource_data.device_workspace_size}, rank);
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Allocated device workspace: {} bytes for FlashInfer", rank,
                                resource_data.device_workspace_size);

  resource_data.is_workspace_initialized = true;
#else
  KLLM_THROW(fmt::format("Rank[{}] FlashInfer workspace requires CUDA support, but CUDA is not enabled", rank));
#endif
}

}  // namespace ksana_llm
