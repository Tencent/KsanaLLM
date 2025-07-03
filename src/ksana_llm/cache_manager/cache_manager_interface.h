/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ksana_llm/cache_manager/block_allocator/block_allocator_manager.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

class CacheManagerInterface {
 public:
  // Initialize all the memory blocks.
  virtual void InitializeCachedBlocks() = 0;

  virtual std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const = 0;

  // Get all usable block number, including free and reusable ones.
  virtual size_t GetUsableBlockNumber() = 0;

  // Get block number that not usable now, but will be usable in future.
  // That is, the blocks used by swapout, but not merged yet.
  virtual size_t GetFutureFreeBlockNumber() = 0;

  // The value is from block mamanger.
  virtual size_t GetHostFreeBlockNumber() = 0;

  // Calculate the actual number of unallocated blocks by passing the input length and obtaining the required block
  // number for the specific request.
  virtual size_t GetRequestStepBlockNumber(int64_t req_id, size_t input_token_lens) = 0;

  // Get the needed block num for specific request if only one next token.
  virtual size_t GetRequestStepBlockNumberForOneNextToken(int64_t req_id) = 0;

  // Get the usable block num for specific request.
  virtual size_t GetRequestUsableBlockNumber(int64_t req_id) = 0;

  // Check the shared and unique block num of specific request, the token number must be enough for next generation.
  // The unique_block_num will always large than 0.
  virtual Status GetRequestPrefixBlockNumber(int64_t req_id, const std::vector<int>& input_token_ids,
                                             size_t& shared_block_num, size_t& unique_block_num,
                                             size_t& shared_token_num) = 0;

  // Allocate new blocks for request, called only when req is running.
  virtual Status AllocateRequestBlocks(int64_t req_id, size_t block_num,
                                       std::vector<std::vector<int>>& req_block_ids) = 0;

  virtual void UpdateFlexibleCache(int64_t req_id, const std::vector<int>& token_ids, int shared_token_num,
                                   std::vector<FlexibleCachedCopyTask>& flexible_cached_copy_tasks) = 0;

  // Update the token ids of this request.
  // This method will update request memory blocks if the origin block is merged.
  virtual Status UpdateRequestTokens(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                                     size_t shareable_kvcache_token_num,
                                     std::vector<std::vector<int>>& req_block_ids) = 0;

  // Get the freeable/needed block num if swap out/in a request.
  virtual Status GetRequestFreeableBlockNum(int64_t req_id, size_t& block_num) = 0;
  virtual Status GetRequestNeededBlockNumForOneNextToken(int64_t req_id, size_t& block_num) = 0;

  // Swap out/in specific request async.
  virtual Status SwapoutRequestAsync(int64_t req_id, size_t& swapped_block_num, size_t& free_block_num,
                                     std::vector<int>& swapped_memory_block_ids) = 0;
  virtual Status SwapinRequestAsync(int64_t req_id, size_t& block_num, std::vector<std::vector<int>>& req_block_ids,
                                    std::vector<int>& swapped_memory_block_ids) = 0;

  // Waiting until at lease on swap out/in task done, return the pending task number.
  virtual Status WaitSwapoutRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking = true) = 0;
  virtual Status WaitSwapinRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking = true) = 0;

  // Swap out/in memory blocks referenced by req_id.
  virtual Status SwapoutRequestMemoryBlockAsync(int64_t req_id, const std::vector<int>& memory_block_ids) = 0;
  virtual Status SwapinRequestMemoryBlockAsync(int64_t req_id, const std::vector<int>& memory_block_ids) = 0;

  // Wait until all memory block swappness referenced by req_ids finished.
  virtual Status WaitSwapoutRequestMemoryBlock(const std::vector<int64_t>& req_ids) = 0;
  virtual Status WaitSwapinRequestMemoryBlock(const std::vector<int64_t>& req_ids) = 0;

  // Merge the swapped out blocks to free list, no need to get host block ids.
  // The swapout of the request's block must be done before call this.
  virtual Status MergeSwapoutRequest(int64_t req_id) = 0;

  // Merge the swapped in block to the tree, update block ids for infer request.
  // The swapin of the request's block must be done before call this.
  virtual Status MergeSwapinRequest(int64_t req_id, std::vector<std::vector<int>>& req_block_ids) = 0;

  // Drop a swaped cached request.
  virtual void DestroySwappedRequest(int64_t req_id) = 0;

  // Update internal state after request finished.
  virtual void DestroyFinishedRequest(int64_t req_id) = 0;

  virtual bool IsPrefixCachingEnabled() = 0;
};

}  // namespace ksana_llm
