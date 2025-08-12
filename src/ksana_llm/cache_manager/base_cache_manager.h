/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ksana_llm/cache_manager/block_allocator/block_allocator_manager.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

// Used to describe memory block swappness.
struct RequestMemoryBlockSwappinessTask {
  std::vector<int> device_memory_blocks;

  // Eevery device block need multiple host blocks.
  std::vector<std::vector<int>> host_memory_blocks;
};

template <class CachedBlockType, class CachedRequestType>
class BaseCacheManager {
 public:
  explicit BaseCacheManager(const CacheManagerConfig& cache_manager_config,
                            std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group) {
    cache_manager_config_ = cache_manager_config;
    block_allocator_group_ = block_allocator_group;

    block_device_num_ = block_allocator_group_->GetBlockAllocatorDevices().size();

    threadpool_ = std::make_shared<ThreadPool>(cache_manager_config_.swap_threadpool_size);
    threadpool_->Start();
  }

  ~BaseCacheManager() { threadpool_->Stop(); }

  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const { return block_allocator_group_; }

  // The value is from block mamanger.
  size_t GetHostFreeBlockNumber() { return block_allocator_group_->GetHostBlockAllocator()->GetFreeBlockNumber(); }

  size_t GetDeviceUsedBlockNumber() { return block_allocator_group_->GetDeviceBlockAllocator()->GetUsedBlockNumber(); }

  // Get block number that not usable now, but will be usable in future.
  // That is, the blocks used by swapout, but not merged yet.
  size_t GetFutureFreeBlockNumber() {
    size_t future_block_num = 0;
    for (const auto& [req_id, swapout_blocks] : swapout_cached_block_buffer_) {
      future_block_num += swapout_blocks.size();
    }
    return future_block_num;
  }

  // Waiting until at least one swap out/in task done, return the pending task number.
  Status WaitSwapoutRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking = true) {
    return WaitTaskDone(swapout_task_queue_, finish_swapout_request_, req_ids, left_req_num, blocking);
  }

  Status WaitSwapinRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking = true) {
    return WaitTaskDone(swapin_task_queue_, finish_swapin_request_, req_ids, left_req_num, blocking);
  }

  // Merge the swapped out blocks to free list, no need to get host block ids.
  // The swapout of the request's block must be done before call this.
  Status MergeSwapoutRequest(int64_t req_id) {
    std::vector<CachedBlockType*> swapout_blocks;

    swapout_blocks.swap(swapout_cached_block_buffer_[req_id]);
    swapout_cached_block_buffer_.erase(req_id);

    // Move reusable node to free list.
    for (CachedBlockType* cb : swapout_blocks) {
      free_cached_blocks_.push(cb);
    }

    // Remove from finished queue.
    auto it = std::find(finish_swapout_request_.begin(), finish_swapout_request_.end(), req_id);
    if (it != finish_swapout_request_.end()) {
      finish_swapout_request_.erase(it);
    }

    return Status();
  }

 protected:
  // Create a new cached request instance.
  CachedRequestType* CreateCachedRequest(int64_t req_id) {
    // New created request is in waiting state, and have no cached blocks.PrefixCacheManager
    auto cached_request = std::make_unique<CachedRequestType>();
    cached_request->req_id = req_id;

    // Get a raw pointer from the unique pointer of CachedRequestType without ownership.
    CachedRequestType* cached_request_ptr = cached_request.get();
    // Append to request list.
    cached_requests_.emplace(req_id, std::move(cached_request));

    return cached_request_ptr;
  }

  // Wait until at least one request is done.
  Status WaitTaskDone(std::unordered_map<int64_t, std::future<void>>& task_queue,
                      std::vector<int64_t>& finish_task_request, std::vector<int64_t>& req_ids, size_t& left_task_num,
                      bool blocking = true) {
    // Return immediately if no pending task.
    req_ids.clear();
    if (task_queue.empty()) {
      left_task_num = 0;
      req_ids = finish_task_request;
      return Status();
    }

    for (auto& pair : task_queue) {
      // Fetch all finished task.
      if (pair.second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        pair.second.get();
        req_ids.push_back(pair.first);
        continue;
      }

      // Return immediately if in unblocking mode.
      if (!blocking) {
        break;
      }

      // Waiting the first one if on task done.
      if (req_ids.empty()) {
        try {
          pair.second.get();
        } catch (const std::exception& e) {
          KLLM_LOG_FATAL << "Exception in swapout, info: " << e.what();
        }
        req_ids.push_back(pair.first);
        continue;
      }

      // return immediately if any task done.
      break;
    }

    for (auto id : req_ids) {
      task_queue.erase(id);
      finish_task_request.push_back(id);
    }

    left_task_num = task_queue.size();
    return Status();
  }

  // Swap out/in a cached block.
  Status SwapoutCachedBlock(CachedBlockType* dev_cached_block, CachedBlockType* host_cached_block) {
    // For swapout, every block is process by only one thread, no need to protect.
    std::vector<int> dev_block_ids = dev_cached_block->memory_block_ids;

    // Swap out cached block to host.
    for (size_t i = 0; i < block_device_num_; ++i) {
      block_allocator_group_->SwapOut(i, host_cached_block->memory_block_ids[i], dev_cached_block->memory_block_ids[i]);
      dev_cached_block->memory_block_ids[i] = host_cached_block->memory_block_ids[i];
    }
    dev_cached_block->is_device_location = false;

    // Reuse the memory blocks.
    host_cached_block->block_id = dev_cached_block->block_id;
    host_cached_block->memory_block_ids = dev_block_ids;
    host_cached_block->is_device_location = true;

    return Status();
  }

  Status SwapinCachedBlock(CachedBlockType* dev_cached_block, CachedBlockType* host_cached_block) {
    // Check whether current block has been processed by another thread.
    if (host_cached_block->is_device_location) {
      return Status();
    }

    // Swap in cached block to dev.
    for (size_t i = 0; i < block_device_num_; ++i) {
      block_allocator_group_->SwapIn(i, dev_cached_block->memory_block_ids[i], host_cached_block->memory_block_ids[i]);

      // Free the host memory.
      block_allocator_group_->GetHostBlockAllocator()->FreeBlocks({host_cached_block->memory_block_ids[i]});

      host_cached_block->memory_block_ids[i] = dev_cached_block->memory_block_ids[i];
    }
    host_cached_block->block_id = dev_cached_block->block_id;
    host_cached_block->is_device_location = true;

    // Remove usless cached block.
    delete dev_cached_block;

    return Status();
  }

  // Swap out/in memory blocks referenced by req_id.
  Status SwapoutRequestMemoryBlockAsync(int64_t req_id, const std::vector<int>& memory_block_ids) {
    RequestMemoryBlockSwappinessTask req_memory_block_swappiness;

    req_memory_block_swappiness.device_memory_blocks = memory_block_ids;
    for (size_t i = 0; i < memory_block_ids.size(); ++i) {
      std::vector<int> host_block_ids;
      Status status =
          block_allocator_group_->GetHostBlockAllocator()->AllocateBlocks(block_device_num_, host_block_ids);
      if (!status.OK()) {
        return status;
      }

      req_memory_block_swappiness.host_memory_blocks.push_back(host_block_ids);
    }
    request_memory_block_swap_task_[req_id] = req_memory_block_swappiness;

    request_memory_block_swap_result_[req_id] = threadpool_->Submit([=] {
      for (size_t i = 0; i < memory_block_ids.size(); ++i) {
        int dev_block_id = memory_block_ids[i];
        for (size_t j = 0; j < block_device_num_; ++j) {
          block_allocator_group_->SwapOut(j, req_memory_block_swappiness.host_memory_blocks[i][j], dev_block_id);
        }
      }
    });

    return Status();
  }

  Status SwapinRequestMemoryBlockAsync(int64_t req_id, const std::vector<int>& memory_block_ids) {
    auto it = request_memory_block_swap_task_.find(req_id);
    if (it == request_memory_block_swap_task_.end()) {
      return Status(RET_RUNTIME_FAILED, FormatStr("SwapinRequestMemoryBlockAsync req_id %d not found.", req_id));
    }

    auto it2 = request_memory_block_swap_result_.find(req_id);
    if (it2 != request_memory_block_swap_result_.end()) {
      return Status(RET_RUNTIME_FAILED,
                    FormatStr("SwapinRequestMemoryBlockAsync Swapin req %d fail, please merge swapout first.", req_id));
    }

    RequestMemoryBlockSwappinessTask& req_memory_block_swappiness = it->second;
    req_memory_block_swappiness.device_memory_blocks = memory_block_ids;

    request_memory_block_swap_result_[req_id] = threadpool_->Submit([=] {
      for (int i = 0; i < memory_block_ids.size(); ++i) {
        for (size_t j = 0; j < block_device_num_; ++j) {
          block_allocator_group_->SwapIn(j, req_memory_block_swappiness.device_memory_blocks[j],
                                         req_memory_block_swappiness.host_memory_blocks[i][j]);
          block_allocator_group_->GetHostBlockAllocator()->FreeBlocks(
              {req_memory_block_swappiness.host_memory_blocks[i][j]});
        }
      }
    });

    return Status();
  }

  // Wait until all memory block swappness referenced by req_ids finished.
  Status WaitSwappinessRequestMemoryBlock(const std::vector<int64_t>& req_ids) {
    for (int64_t req_id : req_ids) {
      auto it = request_memory_block_swap_result_.find(req_id);
      if (it == request_memory_block_swap_result_.end()) {
        KLLM_LOG_WARNING << "Wait swapout request " << req_id << " not found.";
        continue;
      }

      try {
        it->second.get();
      } catch (const std::exception& e) {
        KLLM_LOG_FATAL << "Exception in swapout, info: " << e.what();
      }
    }

    return Status();
  }

  // Wait until all memory block swappness referenced by req_ids finished.
  // The task will be removed after wait() finished.
  Status WaitSwapoutRequestMemoryBlock(const std::vector<int64_t>& req_ids) {
    Status status = WaitSwappinessRequestMemoryBlock(req_ids);
    for (int64_t req_id : req_ids) {
      // Keep request_memory_block_swap_task_ for following swapin.
      request_memory_block_swap_result_.erase(req_id);
    }

    return Status();
  }

  // Wait until all memory block swappness referenced by req_ids finished.
  // The task will be removed after wait() finished.
  Status WaitSwapinRequestMemoryBlock(const std::vector<int64_t>& req_ids) {
    Status status = WaitSwappinessRequestMemoryBlock(req_ids);
    for (int64_t req_id : req_ids) {
      request_memory_block_swap_task_.erase(req_id);
      request_memory_block_swap_result_.erase(req_id);
    }

    return Status();
  }

  bool IsPrefixCachingEnabled() { return cache_manager_config_.enable_prefix_caching; }

 protected:
  CacheManagerConfig cache_manager_config_;

  // The block allocators for this cache manager instance.
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;

  // The devices of current cache manager.
  size_t block_device_num_;

  // The cached blocks that have no computed data, and could be reused freely.
  std::queue<CachedBlockType*> free_cached_blocks_;

  // All requests, the key is request id.
  std::unordered_map<int64_t, std::unique_ptr<CachedRequestType>> cached_requests_;

  // Threadpool used to swap in/out.
  std::shared_ptr<ThreadPool> threadpool_ = nullptr;

  // The swap out/in requests, all blocks have copied.
  std::vector<int64_t> finish_swapout_request_;
  std::vector<int64_t> finish_swapin_request_;

  // The future queue for swap out/in, the key is requst id.
  std::unordered_map<int64_t, std::future<void>> swapout_task_queue_;
  std::unordered_map<int64_t, std::future<void>> swapin_task_queue_;

  // The cached blocks that have been swapped out/in by async task.
  // The key is request id.
  std::unordered_map<int64_t, std::vector<CachedBlockType*>> swapout_cached_block_buffer_;
  std::unordered_map<int64_t, std::vector<CachedBlockType*>> swapin_cached_block_buffer_;

  // The future result of memory swap.
  std::unordered_map<int64_t, RequestMemoryBlockSwappinessTask> request_memory_block_swap_task_;
  std::unordered_map<int64_t, std::future<void>> request_memory_block_swap_result_;
};

}  // namespace ksana_llm
