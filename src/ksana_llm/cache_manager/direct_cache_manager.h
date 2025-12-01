/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/cache_manager/base_cache_manager.h"
#include "ksana_llm/cache_manager/cache_manager_interface.h"

#include "ksana_llm/runtime/request_state.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

struct DirectCachedRequest;

// Describe a cached data, on either device or host.
struct DirectCachedBlock {
  // The unique id of this block, independent with content, not changed after created.
  size_t block_id = 0;

  // Whether the block is on device or host.
  bool is_device_location = true;

  // The block id of every device, the index is device_id。
  // If this cached block is swaped out, the value is host block id of every device.
  std::vector<int> memory_block_ids;
};

// Describe the cache information for a infer request.
struct DirectCachedRequest {
  // The id of this request, as same as the id of InferRequest.
  int64_t req_id;

  // kvcached token num, used to get next step block number.
  size_t kvcached_token_num;

  // The cached blocks associated with this request, not include root node.
  // Contain all blocks even when request is waiting, or blocks that haved been swapped.
  std::vector<DirectCachedBlock*> cached_blocks;
};

// Used to support prefix caching.
class DirectCacheManager : public CacheManagerInterface,
                           public BaseCacheManager<DirectCachedBlock, DirectCachedRequest> {
 public:
  explicit DirectCacheManager(const CacheManagerConfig& cache_manager_config,
                              std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group = nullptr);
  ~DirectCacheManager();

  // Initialize all the memory blocks.
  void InitializeCachedBlocks() override;

  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const override;

  // Get block number that not usable now, but will be usable in future.
  // That is, the blocks used by swapout, but not merged yet.
  size_t GetFutureFreeBlockNumber() override;

  // Get all usable block number, including free and reusable ones.
  size_t GetUsableBlockNumber() override;

  // The value is from block manager.
  size_t GetHostFreeBlockNumber() override;

  // The value is from block manager.
  size_t GetUsedBlockNumber() override;

  // Calculate the actual number of unallocated blocks by passing the input length and obtaining the required block
  // number for the specific request.
  size_t GetRequestStepBlockNumber(int64_t req_id, size_t input_token_lens) override;

  // Get the needed block num for specific request if only one next token.
  size_t GetRequestStepBlockNumberForOneNextToken(int64_t req_id) override;

  // Get the usable block num for specific request.
  size_t GetRequestUsableBlockNumber(int64_t req_id) override;

  // Check the block num of specific request, the token number must be enough for next generation.
  // The shared block num always 0.
  Status GetRequestPrefixBlockNumber(int64_t req_id, const std::vector<int>& input_token_ids, size_t check_token_num,
                                     size_t& shared_block_num, size_t& unique_block_num,
                                     size_t& shared_token_num) override;

  // Allocate new blocks for request, called only when req is running.
  Status AllocateRequestBlocks(int64_t req_id, size_t block_num, std::vector<std::vector<int>>& req_block_ids) override;

  // Update the token ids of this request.
  Status UpdateRequestTokens(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                             size_t shareable_kvcache_token_num, std::vector<std::vector<int>>& req_block_ids,
                             bool& block_merged) override;

  void UpdateFlexibleCache(int64_t req_id, const std::vector<int>& token_ids, int shared_token_num,
                           std::vector<FlexibleCachedCopyTask>& flexible_cached_copy_tasks) override {}
  // Get the freeable/needed block num if swap out/in a request.
  Status GetRequestFreeableBlockNum(int64_t req_id, size_t& block_num) override;
  Status GetRequestNeededBlockNumForOneNextToken(int64_t req_id, size_t& block_num) override;

  // Swap out/in specific request async.
  Status SwapoutRequestAsync(int64_t req_id, size_t& swapped_block_num, size_t& free_block_num,
                             std::vector<int>& swapped_memory_block_ids) override;
  Status SwapinRequestAsync(int64_t req_id, size_t& block_num, std::vector<std::vector<int>>& req_block_ids,
                            std::vector<int>& swapped_memory_block_ids) override;

  // Waiting until at lease on swap out/in task done, return the pending task number.
  Status WaitSwapoutRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking = true) override;
  Status WaitSwapinRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking = true) override;

  // Merge the swapped out blocks to free list, no need to get host block ids.
  // The swapout of the request's block must be done before call this.
  Status MergeSwapoutRequest(int64_t req_id) override;

  // Merge the swapped in block to the tree, update block ids for infer request.
  // The swapin of the request's block must be done before call this.
  Status MergeSwapinRequest(int64_t req_id, std::vector<std::vector<int>>& req_block_ids) override;

  // Drop a swaped cached request.
  void DestroySwappedRequest(int64_t req_id) override;

  // Update internal state after request finished.
  void DestroyFinishedRequest(int64_t req_id) override;

  // Swap out/in memory blocks referenced by req_id.
  Status SwapoutRequestMemoryBlockAsync(int64_t req_id, const std::vector<int>& memory_block_ids) override;
  Status SwapinRequestMemoryBlockAsync(int64_t req_id, const std::vector<int>& memory_block_ids) override;

  // Wait until all memory block swappness referenced by req_ids finished.
  Status WaitSwapoutRequestMemoryBlock(const std::vector<int64_t>& req_ids) override;
  Status WaitSwapinRequestMemoryBlock(const std::vector<int64_t>& req_ids) override;

  bool IsPrefixCachingEnabled() override;

 private:
  // Create a new cached block.
  DirectCachedBlock* CreateCachedBlock(size_t block_id);

  // Create a temporarily cached block, no block id generated.
  DirectCachedBlock* CreateEmptyCachedBlock();
};

}  // namespace ksana_llm
