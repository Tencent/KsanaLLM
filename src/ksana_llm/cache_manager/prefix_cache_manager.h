/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/cache_manager/base_cache_manager.h"
#include "ksana_llm/cache_manager/cache_manager_interface.h"

#include "ksana_llm/runtime/request_state.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/calc_intvec_hash.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

struct PrefixCachedRequest;

// Describe a cached data, on either device or host.
struct PrefixCachedBlock {
  // ///////////////////////////////////////////////////////////////////////////
  // Common field for base class.

  // The unique id of this block, independent with content, not changed after created.
  size_t block_id = 0;

  // Whether the block is on device or host.
  bool is_device_location = true;

  // The block id of every device, the index is device_id。
  // If this cached block is swaped out, the value is host block id of every device.
  std::vector<int> memory_block_ids;

  // ///////////////////////////////////////////////////////////////////////////
  // Prefix cached block only.

  // The hash code of this block's content.
  size_t hash_code = 0;

  // Whether it is the root node.
  bool is_root = false;

  // The parent block.
  PrefixCachedBlock* parent = nullptr;

  // The child blocks, key is hash code of tokens.
  // std::unordered_map<size_t, std::list<PrefixCachedBlock*>> children;
  std::unordered_map<std::vector<int>, PrefixCachedBlock*, TokensHash, TokensEqual> children;

  // Whether this block contain valid cache data and be shareable.
  bool is_shareable = false;

  // The token ids of this block.
  std::vector<int> token_ids;

  // All running or finished requests that reference this block.
  // pair.first is position index of current block in request.
  std::unordered_map<int, std::pair<int, PrefixCachedRequest*>> active_requests;

  // All waiting or swapped requests that reference this block.
  // The block could not be reuse if any swapped request reference it.
  // pair.first is position index of current block in request.
  std::unordered_map<int, std::pair<int, PrefixCachedRequest*>> inactive_requests;

  // Used to protect swapin operation,
  std::mutex swapin_mutex;
};

// Describe the cache information for a infer request.
struct PrefixCachedRequest {
  // The id of this request, as same as the id of InferRequest.
  int64_t req_id;

  // The request state, waiting or running or finished.
  RequestState req_state = RequestState::kWaiting;

  // Used to update filled cached block, increased only if one block is full.
  size_t shared_block_num = 0;

  // The cached blocks associated with this request, not include root node.
  // Contain all blocks even when request is waiting, or blocks that haved been swapped.
  std::vector<PrefixCachedBlock*> cached_blocks;
};

// Used to support prefix caching.
class PrefixCacheManager : public CacheManagerInterface,
                           public BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest> {
 public:
  explicit PrefixCacheManager(const CacheManagerConfig& cache_manager_config,
                              std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group = nullptr);
  virtual ~PrefixCacheManager();

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
  // The method will exclude the cached blocks of this request.
  size_t GetRequestUsableBlockNumber(int64_t req_id) override;

  // WARNING: Allocated blocks may be truncated according to check_token_num.
  Status GetRequestPrefixBlockNumber(int64_t req_id, const std::vector<int>& input_token_ids, size_t check_token_num,
                                     size_t& shared_block_num, size_t& unique_block_num,
                                     size_t& shared_token_num) override;

  // Allocate new blocks for request, called only when req is running.
  Status AllocateRequestBlocks(int64_t req_id, size_t block_num, std::vector<std::vector<int>>& req_block_ids) override;

  // Update the token ids of this request.
  // This method will update request memory blocks if the origin block is merged and set block_merged as true
  Status UpdateRequestTokens(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                             size_t shareable_kvcache_token_num, std::vector<std::vector<int>>& req_block_ids,
                             bool& block_merged) override;

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

  // Drop a swapped cached request.
  void DestroySwappedRequest(int64_t req_id) override;

  // Update internal state after request finished.
  void DestroyFinishedRequest(int64_t req_id) override;

  // Free at least block_num cached blocks that could resued, reserve some blocks if needed.
  bool FreeCachedBlocks(size_t block_num, size_t& free_block_num,
                        const std::vector<PrefixCachedBlock*>& reserved_blocks = {});

  // Swap out/in memory blocks referenced by req_id.
  Status SwapoutRequestMemoryBlockAsync(int64_t req_id, const std::vector<int>& memory_block_ids) override;
  Status SwapinRequestMemoryBlockAsync(int64_t req_id, const std::vector<int>& memory_block_ids) override;

  // Wait until all memory block swappness referenced by req_ids finished.
  Status WaitSwapoutRequestMemoryBlock(const std::vector<int64_t>& req_ids) override;
  Status WaitSwapinRequestMemoryBlock(const std::vector<int64_t>& req_ids) override;

  bool IsPrefixCachingEnabled() override;

 private:
  // Whether the block token is equal to specific ones.
  bool CheckSameTokens(const PrefixCachedBlock* block, const int* start, size_t len);

  // Find tokens from cached blocks, return the matched blocks, return nullptr if not found.
  PrefixCachedBlock* FindChildCacheBlock(PrefixCachedBlock* block, const int* start, size_t len);

  // Create a new cached block.
  PrefixCachedBlock* CreateCachedBlock(size_t block_id);

  // Create a temporarily cached block, no block id generated.
  PrefixCachedBlock* CreateEmptyCachedBlock();

  // The cached block must be reset before reused, keep memory block id unchanged.
  void ResetCachedBlock(PrefixCachedBlock* cached_block);

  // Recursive free block and its all children.
  void FreeCachedBlockRecursively(PrefixCachedBlock* cached_block, std::vector<PrefixCachedBlock*>& free_blocks,
                                  size_t& free_num);

  // Append or remove cached block to/from time block list.
  void AppendCachedBlockToTimedList(PrefixCachedBlock* cached_block);
  void RemoveCachedBlockFromTimedList(PrefixCachedBlock* cached_block);

  // Append specific filled block to the tree.
  Status AppendFilledCachedBlock(PrefixCachedRequest* cached_request, size_t block_index,
                                 PrefixCachedBlock* cached_block, std::vector<std::vector<int>>& req_block_ids,
                                 bool& block_merged);

  // Merge specific filled block to another.
  Status MergeFilledCachedBlocks(PrefixCachedRequest* cached_request, size_t block_index,
                                 PrefixCachedBlock* dst_cached_block, PrefixCachedBlock* src_cached_block,
                                 std::vector<std::vector<int>>& req_block_ids);

  // Merge specific swapped in block to the tree.
  Status AppendSwapinCachedBlock(PrefixCachedRequest* cached_request, size_t block_index,
                                 PrefixCachedBlock* cached_block, std::vector<std::vector<int>>& req_block_ids);

  // Merge two cached blocks.
  Status MergeSwapinCachedBlocks(PrefixCachedRequest* cached_request, size_t block_index,
                                 PrefixCachedBlock* dst_cached_block, PrefixCachedBlock* src_cached_block,
                                 std::vector<std::vector<int>>& req_block_ids);

  // Update the internal state and ref count of specific request.
  Status UpdateCachedRequestState(int64_t req_id, RequestState req_state);
  void QueryIncompleteBlock(PrefixCachedBlock* last_block, const std::vector<int>& token_ids, int start_idx,
                            PrefixCachedBlock*& prefix_append_block, size_t& prefix_append_len);
  void EraseFlexibleCacheMap(PrefixCachedBlock* cached_block);
  void InsertFlexibleCacheMap(PrefixCachedBlock* cached_block);
  void UpdateFlexibleCache(int64_t req_id, const std::vector<int>& token_ids, int shared_token_num,
                           std::vector<FlexibleCachedCopyTask>& flexible_cached_copy_tasks,
                           size_t& req_flexible_cache_len);
  int GetPrefixLen(PrefixCachedBlock* block);

 private:
  // The root block of block tree, not contain any memory block.
  // The tree struct only contain computed blocks, non-computed block is maintain by cached request itself.
  PrefixCachedBlock* root_cached_block_;
  std::unordered_map<std::vector<int>, std::unordered_set<PrefixCachedBlock*>, TokensHash, TokensEqual>
      flexible_cache_map_;
  // The cached blocks that have filled, but no request referenced it.
  std::unordered_set<PrefixCachedBlock*> reusable_cached_blocks_;

  // All the cached block in tree, sort by time when that node added into the tree.
  // Not contain the cached block that not in node tree.
  std::list<PrefixCachedBlock*> timed_cached_blocks_;
  std::unordered_map<PrefixCachedBlock*, std::list<PrefixCachedBlock*>::iterator> timed_cached_block_iters_;
};

}  // namespace ksana_llm
