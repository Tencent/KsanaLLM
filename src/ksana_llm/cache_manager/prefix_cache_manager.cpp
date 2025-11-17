/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/cache_manager/prefix_cache_manager.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>

#include "ksana_llm/cache_manager/base_cache_manager.h"
#include "ksana_llm/runtime/request_state.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

PrefixCacheManager::PrefixCacheManager(const CacheManagerConfig& cache_manager_config,
                                       std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group)
    : BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>(cache_manager_config, block_allocator_group) {
  cache_manager_config_ = cache_manager_config;

  root_cached_block_ = new PrefixCachedBlock();
  root_cached_block_->is_root = true;
}

PrefixCacheManager::~PrefixCacheManager() {
  KLLM_LOG_DEBUG << "PrefixCacheManager destroyed.";
  delete root_cached_block_;
}

void PrefixCacheManager::InitializeCachedBlocks() {
  size_t total_device_block_num = block_allocator_group_->GetDeviceBlockAllocator()->GetFreeBlockNumber();

  // block id 0 is root, so here start with block id 1.
  for (size_t i = 1; i <= total_device_block_num; ++i) {
    PrefixCachedBlock* cached_block = CreateCachedBlock(i);

    // allocate memory block on every device.
    for (size_t j = 0; j < block_device_num_; ++j) {
      std::vector<int> blocks;
      block_allocator_group_->GetDeviceBlockAllocator(j)->AllocateBlocks(1, blocks);
      cached_block->memory_block_ids[j] = blocks[0];
    }
    free_cached_blocks_.push(cached_block);
  }
  KLLM_LOG_DEBUG << "PrefixCacheManager initialized, device num:" << block_device_num_
                 << ", device block num:" << free_cached_blocks_.size()
                 << ", host block num:" << block_allocator_group_->GetHostBlockAllocator()->GetFreeBlockNumber();
}

std::shared_ptr<BlockAllocatorGroupInterface> PrefixCacheManager::GetBlockAllocatorGroup() const {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::GetBlockAllocatorGroup();
}

size_t PrefixCacheManager::GetFutureFreeBlockNumber() {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::GetFutureFreeBlockNumber();
}

size_t PrefixCacheManager::GetUsableBlockNumber() {
  return free_cached_blocks_.size() + reusable_cached_blocks_.size();
}

size_t PrefixCacheManager::GetRequestUsableBlockNumber(int64_t req_id) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get usable block number for req %d error, req not exist.", req_id));
  }
  PrefixCachedRequest* cached_request = it->second.get();

  size_t reserved_block_num = 0;
  for (PrefixCachedBlock* cached_block : cached_request->cached_blocks) {
    if (reusable_cached_blocks_.find(cached_block) != reusable_cached_blocks_.end()) {
      ++reserved_block_num;
    }
  }

  return free_cached_blocks_.size() + reusable_cached_blocks_.size() - reserved_block_num;
}

size_t PrefixCacheManager::GetHostFreeBlockNumber() {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::GetHostFreeBlockNumber();
}

size_t PrefixCacheManager::GetUsedBlockNumber() {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::GetDeviceUsedBlockNumber() - GetUsableBlockNumber();
}

size_t PrefixCacheManager::GetRequestStepBlockNumber(int64_t req_id, size_t input_token_lens) {
  const auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get step block number for req %d error, req not exist.", req_id));
  }
  const size_t block_token_num = cache_manager_config_.block_token_num;
  const size_t total_require_block_num = (input_token_lens + block_token_num) / block_token_num;
  return total_require_block_num - std::min(total_require_block_num, it->second->cached_blocks.size());
}

size_t PrefixCacheManager::GetRequestStepBlockNumberForOneNextToken(int64_t req_id) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get step block number for req %d error, req not exist.", req_id));
  }
  return it->second->cached_blocks.back()->is_shareable ? 1 : 0;
}

Status PrefixCacheManager::GetRequestPrefixBlockNumber(int64_t req_id, const std::vector<int>& input_token_ids,
                                                       size_t check_token_num, size_t& shared_block_num,
                                                       size_t& unique_block_num, size_t& shared_token_num) {
  auto it = cached_requests_.find(req_id);
  PrefixCachedRequest* request = (it == cached_requests_.end()) ? CreateCachedRequest(req_id) : it->second.get();
  KLLM_CHECK(input_token_ids.size() >= check_token_num);

  // Check whether existed blocks is still valid.
  if (!request->cached_blocks.empty()) {
    for (size_t i = 0; i < request->cached_blocks.size(); ++i) {
      PrefixCachedBlock* cb = request->cached_blocks[i];
      PrefixCachedBlock* cb_prev = (i == 0) ? root_cached_block_ : request->cached_blocks[i - 1];

      // Make sure the block is not changed. Not check token size, just believe it.
      if (cb->parent == cb_prev && cb->is_shareable && cb->is_device_location &&
          CheckSameTokens(cb, input_token_ids.data() + (i * cache_manager_config_.block_token_num),
                          cache_manager_config_.block_token_num)) {
        continue;
      }

      // block changed, remove left blocks and research.
      request->cached_blocks.erase(request->cached_blocks.begin() + i, request->cached_blocks.end());
      break;
    }
  }

  // Research from last block, because new matched block maybe generated yet.
  PrefixCachedBlock* cur_block = request->cached_blocks.empty() ? root_cached_block_ : request->cached_blocks.back();
  for (size_t i = request->cached_blocks.size(); i < check_token_num / cache_manager_config_.block_token_num; ++i) {
    cur_block = FindChildCacheBlock(cur_block, input_token_ids.data() + (i * cache_manager_config_.block_token_num),
                                    cache_manager_config_.block_token_num);
    if (cur_block != nullptr) {
      // Update matched prefill blocks, but do not update block's requests info.
      request->cached_blocks.push_back(cur_block);
      continue;
    }
    break;
  }

  shared_block_num = request->cached_blocks.size();
  unique_block_num = (check_token_num + cache_manager_config_.block_token_num) / cache_manager_config_.block_token_num -
                     shared_block_num;
  shared_token_num = shared_block_num * cache_manager_config_.block_token_num;
  request->shared_block_num = shared_block_num;

  return Status();
}

bool PrefixCacheManager::CheckSameTokens(const PrefixCachedBlock* block, const int* start, size_t len) {
  return std::memcmp(block->token_ids.data(), start, len * sizeof(int)) == 0;
}

PrefixCachedBlock* PrefixCacheManager::FindChildCacheBlock(PrefixCachedBlock* block, const int* start, size_t len) {
  std::vector<int> token_ids(start, start + len);

  auto it = block->children.find(token_ids);
  if (it == block->children.end()) {
    return nullptr;
  }

  return it->second;
}

PrefixCachedBlock* PrefixCacheManager::CreateEmptyCachedBlock() {
  PrefixCachedBlock* cached_block = new PrefixCachedBlock();
  cached_block->memory_block_ids.resize(block_device_num_);
  cached_block->token_ids.resize(cache_manager_config_.block_token_num);

  cached_block->is_shareable = false;
  cached_block->is_device_location = true;

  return cached_block;
}

PrefixCachedBlock* PrefixCacheManager::CreateCachedBlock(size_t block_id) {
  PrefixCachedBlock* cached_block = CreateEmptyCachedBlock();
  cached_block->block_id = block_id;
  return cached_block;
}

void PrefixCacheManager::ResetCachedBlock(PrefixCachedBlock* cached_block) {
  // Keep block id unchanged, reset hashcode.

  cached_block->is_shareable = false;
  cached_block->is_device_location = true;
  EraseFlexibleCacheMap(cached_block);
  cached_block->token_ids.clear();
  cached_block->token_ids.resize(cache_manager_config_.block_token_num);

  cached_block->active_requests.clear();
  cached_block->inactive_requests.clear();

  cached_block->parent = nullptr;
  cached_block->children.clear();
}

/**
 * @brief Find the longest matching prefix block among children of the last block.
 */
void PrefixCacheManager::QueryIncompleteBlock(PrefixCachedBlock* last_block, const std::vector<int>& token_ids,
                                              int start_idx, PrefixCachedBlock*& prefix_append_block,
                                              size_t& prefix_append_len) {
  for (auto& try_block : last_block->children) {
    if (!try_block.second->inactive_requests.empty()) {
      continue;
    }
    size_t i = 0;
    for (; i < cache_manager_config_.block_token_num; i++) {
      if (try_block.second->token_ids[i] != token_ids[start_idx + i]) {
        break;
      }
    }
    if (prefix_append_len < i) {
      prefix_append_len = i;
      prefix_append_block = try_block.second;
    }
  }
}

int PrefixCacheManager::GetPrefixLen(PrefixCachedBlock* block) {
  // Remove root node
  int len = -1 * cache_manager_config_.block_token_num;
  if (block == nullptr) {
    return 0;
  }
  while (block->parent) {
    len += cache_manager_config_.block_token_num;
    block = block->parent;
  }
  return len;
}

/**
 * @brief Updates the flexible cache for a given request by finding and copying matching data blocks.
 *
 * This function searches for the longest prefix match and extends it to form a longer matching sequence
 * within the provided token IDs. It then copies the matched data blocks into the flexible cache task list
 * for subsequent processing.
 */
void PrefixCacheManager::UpdateFlexibleCache(int64_t req_id, const std::vector<int>& token_ids, int shared_token_num,
                                             std::vector<FlexibleCachedCopyTask>& flexible_cached_copy_tasks) {
  // Check if the minimum flexible cache size is set and greater than 0
  size_t min_flexible_cache_num = cache_manager_config_.min_flexible_cache_num;
  if (min_flexible_cache_num == 0) {
    return;
  }

  size_t flexible_cache_len = 0;
  size_t flexible_cache_begin = 0;
  auto& block_token_num = cache_manager_config_.block_token_num;
  // Check if there are enough tokens for flexible caching
  if (shared_token_num + min_flexible_cache_num + block_token_num * 2 > token_ids.size() || shared_token_num == 0) {
    return;
  }
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_THROW(FormatStr("Get usable block number for req %d error, req not exist.", req_id));
  }
  PrefixCachedRequest* cached_request = it->second.get();

  // Find the longest matching prefix tokens among children of the last block
  std::vector<PrefixCachedBlock*> cached_blocks;
  size_t prefix_append_len = 0;
  PrefixCachedBlock* prefix_append_block = nullptr;
  if (shared_token_num > 0) {
    PrefixCachedBlock* last_block = cached_request->cached_blocks[shared_token_num / block_token_num - 1];
    QueryIncompleteBlock(last_block, token_ids, shared_token_num, prefix_append_block, prefix_append_len);
  }

  // Update shared_token_num with the length of the prefix append block
  auto new_shared_token_num = shared_token_num + prefix_append_len;

  // Iterate through the token_ids to find the best matching block for flexible caching
  for (uint32_t offset = new_shared_token_num; offset < new_shared_token_num + block_token_num; offset++) {
    if (min_flexible_cache_num + offset > token_ids.size()) {
      break;
    }
    PrefixCachedBlock* cached_block = nullptr;
    std::vector<int> dst_token_ids(token_ids.begin() + offset, token_ids.begin() + offset + block_token_num);
    auto map_it = flexible_cache_map_.find(dst_token_ids);
    if (map_it != flexible_cache_map_.end()) {
      for (auto block : map_it->second) {
        size_t hit_num = block_token_num;
        uint32_t start_idx = block_token_num - (offset - new_shared_token_num);
        cached_block = block;
        if (!cached_block->inactive_requests.empty()) {
          continue;
        }
        std::vector<PrefixCachedBlock*> new_cached_blocks;
        if (offset != new_shared_token_num) {
          if (cached_block->parent == nullptr) {
            continue;
          }
          if (!cached_block->parent->inactive_requests.empty()) {
            continue;
          }
          // skip if the parent is root block
          if (cached_block->parent->token_ids.empty()) {
            continue;
          }
          for (; start_idx < block_token_num; start_idx++) {
            if (cached_block->parent->token_ids[start_idx] != token_ids[offset - block_token_num + start_idx]) {
              break;
            }
          }
          if (start_idx != block_token_num) {
            continue;
          }
          new_cached_blocks.push_back(cached_block->parent);
        }
        new_cached_blocks.push_back(cached_block);
        bool partial_match = false;
        for (; offset + hit_num + block_token_num < token_ids.size(); hit_num += block_token_num) {
          std::memcpy(dst_token_ids.data(), token_ids.data() + offset + hit_num, block_token_num * sizeof(int));
          auto it = cached_block->children.find(dst_token_ids);
          if (it != cached_block->children.end()) {
            cached_block = it->second;
            if (!cached_block->inactive_requests.empty()) {
              continue;
            }
            new_cached_blocks.push_back(cached_block);
          } else {
            partial_match = true;
            break;
          }
        }
        if (partial_match) {
          PrefixCachedBlock* flexible_append_block = nullptr;
          size_t flexible_append_len = 0;
          QueryIncompleteBlock(cached_block, dst_token_ids, 0, flexible_append_block, flexible_append_len);
          if (flexible_append_len > 0) {
            hit_num += flexible_append_len;
            new_cached_blocks.push_back(flexible_append_block);
          }
        }

        if (hit_num >= min_flexible_cache_num && hit_num > flexible_cache_len) {
          cached_blocks = new_cached_blocks;
          flexible_cache_begin = (block_token_num - (offset - new_shared_token_num)) % block_token_num;
          flexible_cache_len = offset - new_shared_token_num + hit_num;
        }
      }
    }
  }

  // If no matching blocks are found, return early
  if (cached_blocks.empty()) {
    return;
  }

  // Calculate the number of flexible cached tokens to resize the tasks vector
  size_t new_flexible_cached_tokens = flexible_cache_len + prefix_append_len;
  flexible_cached_copy_tasks.resize(new_flexible_cached_tokens);
  size_t idx = 0;
  int src_shared_token_num = GetPrefixLen(prefix_append_block);
  // Copy the prefix append block tokens to the flexible cache tasks
  for (; idx < prefix_append_len; idx++) {
    flexible_cached_copy_tasks[idx].Update(
        shared_token_num + idx, src_shared_token_num + idx,
        cached_request->cached_blocks[shared_token_num / block_token_num]->memory_block_ids,
        prefix_append_block->memory_block_ids);
  }
  src_shared_token_num = GetPrefixLen(cached_blocks[0]);
  // Copy the remaining cached block tokens to the flexible cache tasks
  for (; idx < new_flexible_cached_tokens; idx++) {
    auto cached_token_idx = (idx - prefix_append_len + flexible_cache_begin);
    auto cached_blocks_idx = cached_token_idx / block_token_num;
    flexible_cached_copy_tasks[idx].Update(
        shared_token_num + idx, src_shared_token_num + cached_token_idx,
        cached_request->cached_blocks[(shared_token_num + idx) / block_token_num]->memory_block_ids,
        cached_blocks[cached_blocks_idx]->memory_block_ids);
  }
}

void PrefixCacheManager::EraseFlexibleCacheMap(PrefixCachedBlock* cached_block) {
  auto map_it = flexible_cache_map_.find(cached_block->token_ids);
  if (map_it != flexible_cache_map_.end()) {
    auto block_set_it = map_it->second.find(cached_block);
    if (block_set_it != map_it->second.end()) {
      map_it->second.erase(block_set_it);
    }
    if (map_it->second.empty()) {
      flexible_cache_map_.erase(map_it);
    }
  }
}

void PrefixCacheManager::InsertFlexibleCacheMap(PrefixCachedBlock* cached_block) {
  flexible_cache_map_[cached_block->token_ids].insert(cached_block);
}

void PrefixCacheManager::FreeCachedBlockRecursively(PrefixCachedBlock* cached_block,
                                                    std::vector<PrefixCachedBlock*>& free_blocks, size_t& free_num) {
  if (cached_block->children.empty()) {
    // From reusable to free
    reusable_cached_blocks_.erase(cached_block);

    ResetCachedBlock(cached_block);
    free_cached_blocks_.push(cached_block);
    free_blocks.push_back(cached_block);
    free_num += 1;
    return;
  }

  // First process all children.
  for (auto pair : cached_block->children) {
    FreeCachedBlockRecursively(pair.second, free_blocks, free_num);
  }

  // Force reuse itself, from reusable to free
  reusable_cached_blocks_.erase(cached_block);

  ResetCachedBlock(cached_block);
  free_cached_blocks_.push(cached_block);
  free_blocks.push_back(cached_block);
  free_num += 1;
}

bool PrefixCacheManager::FreeCachedBlocks(size_t block_num, size_t& free_block_num,
                                          const std::vector<PrefixCachedBlock*>& reserved_blocks) {
  free_block_num = 0;
  std::vector<PrefixCachedBlock*> free_blocks;

  // Loop timeline from end to begin.
  for (auto it = timed_cached_blocks_.rbegin(); it != timed_cached_blocks_.rend(); ++it) {
    PrefixCachedBlock* cb = *it;

    // Skip reserved blocks.
    if (!reserved_blocks.empty()) {
      if (std::find(reserved_blocks.begin(), reserved_blocks.end(), cb) != reserved_blocks.end()) {
        continue;
      }
    }

    // Skip if unfilled, unexpected, all block in timeline should be filled.
    if (!cb->is_shareable) {
      KLLM_LOG_DEBUG << "FreeCachedBlocks should not arrive here, unfilled block in timeline.";
      continue;
    }

    if (cb->active_requests.empty() && cb->inactive_requests.empty()) {
      // Take it and all its children from tree.
      auto it2 = cb->parent->children.find(cb->token_ids);
      if (it2 != cb->parent->children.end()) {
        cb->parent->children.erase(it2);
      }
      FreeCachedBlockRecursively(cb, free_blocks, free_block_num);
    }

    if (free_block_num >= block_num) {
      break;
    }
  }

  // Remove from timed block list here, because could not erase in itself's loop.
  for (PrefixCachedBlock* cb : free_blocks) {
    RemoveCachedBlockFromTimedList(cb);
  }

  return free_block_num >= block_num;
}

Status PrefixCacheManager::AllocateRequestBlocks(int64_t req_id, size_t block_num,
                                                 std::vector<std::vector<int>>& req_block_ids) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Allocate block for req %d error, req not exist.", req_id));
  }
  PrefixCachedRequest* cached_request = it->second.get();

  // Try to allocate from block manager first.
  size_t needed_blocks = 0;
  if (block_num > free_cached_blocks_.size()) {
    needed_blocks = block_num - free_cached_blocks_.size();
  }

  // The matched prefix blocks could not be free for waiting request.
  std::vector<PrefixCachedBlock*> reserved_blocks =
      (cached_request->req_state == RequestState::kWaiting || cached_request->req_state == RequestState::kRunning)
          ? cached_request->cached_blocks
          : std::vector<PrefixCachedBlock*>();

  // Reuse some unreferenced blocks.
  size_t free_block_num = 0;
  if (needed_blocks > 0 && !FreeCachedBlocks(needed_blocks, free_block_num, reserved_blocks)) {
    return Status(RET_OUT_OF_DEVICE_MEMORY,
                  FormatStr("Allocate %d blocks for req %d error, no more usable blocks, free:%d.", block_num, req_id,
                            free_cached_blocks_.size()));
  }

  // I donot want to check req block size here, because caller should guard this.
  const size_t req_block_num = req_block_ids[0].size();
  if (req_block_num == 0) {
    cached_request->req_state = RequestState::kRunning;
  }

  // The prefix cache may be invalid. Refer to cached_request for the authoritative data and clean up the invalid parts.
  for (auto& req_block_ids_tp : req_block_ids) {
    req_block_ids_tp.resize(std::min(req_block_ids_tp.size(), cached_request->cached_blocks.size()));
  }

  for (size_t i = req_block_num; i < cached_request->cached_blocks.size(); ++i) {
    PrefixCachedBlock* cb = cached_request->cached_blocks[i];

    // Fill prefix memory blocks.
    for (size_t j = 0; j < block_device_num_; ++j) {
      req_block_ids[j].push_back(cb->memory_block_ids[j]);
    }

    // Not reusable if no request associated with it.
    if (cb->active_requests.empty() && cb->inactive_requests.empty()) {
      reusable_cached_blocks_.erase(cb);
    }

    // Associate request to prefill cached block.
    cb->active_requests[req_id] = std::make_pair(i, cached_request);
  }

  // Try to allocate from free list.
  for (size_t i = 0; i < block_num; ++i) {
    PrefixCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();

    // Fill memory block ids,
    for (size_t j = 0; j < block_device_num_; ++j) {
      req_block_ids[j].push_back(cached_block->memory_block_ids[j]);
    }

    // Not trace request info before a block is merge to tree.
    cached_request->cached_blocks.push_back(cached_block);
  }

  return Status();
}

void PrefixCacheManager::DestroyFinishedRequest(int64_t req_id) {
  const auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "DestroyFinishedRequest req " << req_id << " not exists.";
    return;
  }

  PrefixCachedRequest& cached_request = *it->second.get();
  // If not waiting, remove request from associated cache blocks.
  if (cached_request.req_state != RequestState::kWaiting) {
    for (const auto cached_block : cached_request.cached_blocks) {
      // Move unfilled block to free list if existed.
      if (!cached_block->is_shareable) {
        ResetCachedBlock(cached_block);
        free_cached_blocks_.push(cached_block);
        continue;
      }

      auto& request =
          (cached_request.req_state == RequestState::kRunning || cached_request.req_state == RequestState::kFinished)
              ? cached_block->active_requests
              : cached_block->inactive_requests;
      request.erase(cached_request.req_id);

      // Make reusable if no request left.
      if (cached_block->active_requests.empty() && cached_block->inactive_requests.empty()) {
        reusable_cached_blocks_.insert(cached_block);
      }
    }
  }

  // Remove blocks from request.
  cached_request.cached_blocks.clear();
  cached_requests_.erase(it);
  KLLM_LOG_DEBUG << "DestroyFinishedRequest req " << req_id << " removed from cached requests.";
}

Status PrefixCacheManager::MergeFilledCachedBlocks(PrefixCachedRequest* cached_request, size_t block_index,
                                                   PrefixCachedBlock* dst_cached_block,
                                                   PrefixCachedBlock* src_cached_block,
                                                   std::vector<std::vector<int>>& req_block_ids) {
  // No children, no requests, just update request which reference src cached block.
  dst_cached_block->active_requests[cached_request->req_id] = std::make_pair(block_index, cached_request);

  // Update request's block to new one.
  cached_request->cached_blocks[block_index] = dst_cached_block;

  // Update internal memory block ids of the request.
  for (size_t i = 0; i < block_device_num_; ++i) {
    req_block_ids[i][block_index] = dst_cached_block->memory_block_ids[i];
  }

  return Status();
}

Status PrefixCacheManager::AppendFilledCachedBlock(PrefixCachedRequest* cached_request, size_t block_index,
                                                   PrefixCachedBlock* cached_block,
                                                   std::vector<std::vector<int>>& req_block_ids) {
  PrefixCachedBlock* shadow_parent =
      (block_index == 0) ? root_cached_block_ : cached_request->cached_blocks[block_index - 1];

  // Try to find a brother block that have same data
  auto it = shadow_parent->children.find(cached_block->token_ids);
  if (it != shadow_parent->children.end()) {
    // Found it, merge to existed block.
    MergeFilledCachedBlocks(cached_request, block_index, it->second, cached_block, req_block_ids);
    // Reset it and move to free list.
    ResetCachedBlock(cached_block);
    free_cached_blocks_.push(cached_block);
    return Status();
  }

  // Not found, mark it shareable, trace request info.
  cached_block->is_shareable = true;
  cached_block->active_requests[cached_request->req_id] = std::make_pair(block_index, cached_request);

  // add to tree block.
  cached_block->parent = shadow_parent;
  cached_block->parent->children[cached_block->token_ids] = cached_block;
  InsertFlexibleCacheMap(cached_block);
  // Record timeline.
  AppendCachedBlockToTimedList(cached_block);

  // Not reusable now.
  reusable_cached_blocks_.erase(cached_block);

  return Status();
}

Status PrefixCacheManager::MergeSwapinCachedBlocks(PrefixCachedRequest* cached_request, size_t block_index,
                                                   PrefixCachedBlock* dst_cached_block,
                                                   PrefixCachedBlock* src_cached_block,
                                                   std::vector<std::vector<int>>& req_block_ids) {
  // No children, no active request, process inactive requests only.
  // If the block is not filled, no requests will be found.
  for (auto& pair : src_cached_block->inactive_requests) {
    dst_cached_block->inactive_requests.insert(pair);

    // Update request's block list which reference src cached block.
    PrefixCachedRequest* cached_request = pair.second.second;
    cached_request->cached_blocks[pair.second.first] = dst_cached_block;
  }

  // Update internal memory block ids of current request.
  // Other request's memory blocks will be updated in its self merge progress.
  for (size_t i = 0; i < block_device_num_; ++i) {
    req_block_ids[i][block_index] = dst_cached_block->memory_block_ids[i];
  }

  return Status();
}

Status PrefixCacheManager::AppendSwapinCachedBlock(PrefixCachedRequest* cached_request, size_t block_index,
                                                   PrefixCachedBlock* cached_block,
                                                   std::vector<std::vector<int>>& req_block_ids) {
  PrefixCachedBlock* shadow_parent =
      (block_index == 0) ? root_cached_block_ : cached_request->cached_blocks[block_index - 1];

  // Try to find a brother block that have same data.
  auto it = shadow_parent->children.find(cached_block->token_ids);
  if (it != shadow_parent->children.end()) {
    // Found it, merge to existed block.
    MergeSwapinCachedBlocks(cached_request, block_index, it->second, cached_block, req_block_ids);

    // Reset it and move to free list.
    ResetCachedBlock(cached_block);
    free_cached_blocks_.push(cached_block);
    return Status();
  }

  // Not found, the node is shareable if arrive here, so add to the tree.
  cached_block->parent = shadow_parent;
  cached_block->parent->children[cached_block->token_ids] = cached_block;
  InsertFlexibleCacheMap(cached_block);

  // Update internal memory block ids of current request.
  for (size_t i = 0; i < block_device_num_; ++i) {
    req_block_ids[i][block_index] = cached_block->memory_block_ids[i];
  }

  // Record timeline.
  AppendCachedBlockToTimedList(cached_block);

  // The swapped block come from free list, no need to update reusable lisst.
  return Status();
}

Status PrefixCacheManager::UpdateRequestTokens(int64_t req_id, const std::vector<int>& kvcached_token_ids,
                                               size_t shareable_kvcache_token_num,
                                               std::vector<std::vector<int>>& req_block_ids) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Update request token error, req %d is not found.", req_id));
  }

  PrefixCachedRequest* cached_request = it->second.get();

  if (kvcached_token_ids.size() < shareable_kvcache_token_num) {
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Update request token error,  kvcached_token_ids.size=%d < shareable_kvcache_token_num=%d.",
                            kvcached_token_ids.size(), shareable_kvcache_token_num));
  }

  // Return if block is not full.
  size_t filled_block_num = shareable_kvcache_token_num / cache_manager_config_.block_token_num;
  if (filled_block_num <= cached_request->shared_block_num) {
    return Status();
  }

  if (cached_request->cached_blocks.size() < filled_block_num) {
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Update request token error,  req %d block size %d less than filled block num %d.", req_id,
                            cached_request->cached_blocks.size(), filled_block_num));
  }

  // Maybe there are more than one blocks should be process.
  for (size_t i = cached_request->shared_block_num; i < filled_block_num; ++i) {
    // Fill token ids
    memcpy(cached_request->cached_blocks[i]->token_ids.data(),
           kvcached_token_ids.data() + (i * cache_manager_config_.block_token_num),
           cache_manager_config_.block_token_num * sizeof(int));

    // Append new filled block to tree.
    AppendFilledCachedBlock(cached_request, i, cached_request->cached_blocks[i], req_block_ids);
  }
  cached_request->shared_block_num = filled_block_num;

  return Status();
}

Status PrefixCacheManager::UpdateCachedRequestState(int64_t req_id, RequestState req_state) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Update request state error, req %d is not exist.", req_id));
  }

  // Only swap out/in should call this.
  if (req_state == RequestState::kRunning) {
    if (it->second->req_state == RequestState::kSwapped) {
      it->second->req_state = req_state;
      for (PrefixCachedBlock* cb : it->second->cached_blocks) {
        // For shareable block, update request state from inactive to active.
        if (cb->is_shareable) {
          cb->active_requests[req_id] = cb->inactive_requests[req_id];
          cb->inactive_requests.erase(req_id);
        }
      }
    }
    return Status();
  }

  // Remove request from all blocks before swapped.
  if (req_state == RequestState::kSwapped) {
    if (it->second->req_state == RequestState::kRunning) {
      it->second->req_state = req_state;
      for (PrefixCachedBlock* cb : it->second->cached_blocks) {
        // For shareable block, update request state from active to inactive.
        if (cb->is_shareable) {
          cb->inactive_requests[req_id] = cb->active_requests[req_id];
          cb->active_requests.erase(req_id);
        }
      }
    }
    return Status();
  }
  return Status();
}

void PrefixCacheManager::AppendCachedBlockToTimedList(PrefixCachedBlock* cached_block) {
  auto it = timed_cached_block_iters_.find(cached_block);
  if (it != timed_cached_block_iters_.end()) {
    timed_cached_blocks_.erase(it->second);
  }

  timed_cached_blocks_.push_back(cached_block);
  timed_cached_block_iters_[cached_block] = --timed_cached_blocks_.end();
}

void PrefixCacheManager::RemoveCachedBlockFromTimedList(PrefixCachedBlock* cached_block) {
  auto it = timed_cached_block_iters_.find(cached_block);
  if (it != timed_cached_block_iters_.end()) {
    timed_cached_blocks_.erase(it->second);
    timed_cached_block_iters_.erase(it);
  }
}

Status PrefixCacheManager::GetRequestFreeableBlockNum(int64_t req_id, size_t& block_num) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Get freeable block num of req error, req %d is not exist.", req_id));
  }

  block_num = 0;
  for (PrefixCachedBlock* cb : it->second->cached_blocks) {
    // Block could be swapped even if inactive requests referenced it.
    // unfilled block's active requests is empty.
    if (cb->is_device_location && cb->active_requests.size() <= 1) {
      ++block_num;
    }
  }

  return Status();
}

Status PrefixCacheManager::GetRequestNeededBlockNumForOneNextToken(int64_t req_id, size_t& block_num) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Get needed block num of req error, req %d is not exist.", req_id));
  }

  block_num = 0;
  for (PrefixCachedBlock* cb : it->second->cached_blocks) {
    // All bocks on host should be swapped back.
    if (!cb->is_device_location) {
      ++block_num;
    }
  }

  // Make sure there is enough blocks for next step.
  block_num += GetRequestStepBlockNumberForOneNextToken(req_id);

  return Status();
}

Status PrefixCacheManager::SwapoutRequestAsync(int64_t req_id, size_t& swapped_block_num, size_t& free_block_num,
                                               std::vector<int>& swapped_memory_block_ids) {
  if (!swapin_task_queue_.empty() || !swapin_cached_block_buffer_.empty() || !finish_swapin_request_.empty()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Cannot swapout req %d, some swapin jobs is in progress.", req_id));
  }

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Swapout req %d error, req is not exist.", req_id));
  }

  // Note: here must from tail to head. If swap out failed, do not change anything.
  std::vector<PrefixCachedBlock*> dev_swapout_blocks;
  for (auto it2 = it->second->cached_blocks.rbegin(); it2 != it->second->cached_blocks.rend(); ++it2) {
    PrefixCachedBlock* cb = *it2;

    // Swapout block only device block.
    if (cb->is_device_location) {
      // Skip if this block is still referenced by other active request.
      if (cb->active_requests.size() > 1) {
        continue;
      }

      // If block is already filled, remove from parent's children list
      if (cb->is_shareable) {
        dev_swapout_blocks.push_back(cb);
      } else {
        // request's unique block, no children, no requests, swap directly.
        dev_swapout_blocks.push_back(cb);
      }

      // Assume every device have same memory block id.
      swapped_memory_block_ids.push_back(cb->memory_block_ids[0]);
    }
  }

  swapped_block_num = dev_swapout_blocks.size();
  if (block_allocator_group_->GetHostBlockAllocator()->GetFreeBlockNumber() < block_device_num_ * swapped_block_num) {
    return Status(RET_OUT_OF_DEVICE_MEMORY,
                  FormatStr("Swap out req %d error, no more host blocks, needed: %d, free: %d.", req_id,
                            block_device_num_ * swapped_block_num,
                            block_allocator_group_->GetHostBlockAllocator()->GetFreeBlockNumber()));
  }

  // Update request state from all associated blocks.
  UpdateCachedRequestState(req_id, RequestState::kSwapped);

  for (PrefixCachedBlock* cb : dev_swapout_blocks) {
    // If block is already filled, remove from parent's children list
    if (cb->is_shareable) {
      auto it3 = cb->parent->children.find(cb->token_ids);
      if (it3 != cb->parent->children.end()) {
        cb->parent->children.erase(it3);
      }
      cb->parent = nullptr;

      // Free all its children if existed.
      if (!cb->children.empty()) {
        std::vector<PrefixCachedBlock*> free_blocks;
        for (auto pair : cb->children) {
          FreeCachedBlockRecursively(pair.second, free_blocks, free_block_num);
        }

        // Remove from timed block list, no processed in FreeCachedBlockRecursively.
        for (PrefixCachedBlock* cb2 : free_blocks) {
          RemoveCachedBlockFromTimedList(cb2);
        }

        cb->children.clear();
      }

      // remove block itself from tree.
      // Still referenced by inactive requests, no needed to update reusable list.
      RemoveCachedBlockFromTimedList(cb);
    }
  }

  std::vector<PrefixCachedBlock*> host_swapout_blocks;
  for (size_t i = 0; i < dev_swapout_blocks.size(); ++i) {
    PrefixCachedBlock* cached_block = CreateEmptyCachedBlock();
    block_allocator_group_->GetHostBlockAllocator()->AllocateBlocks(block_device_num_, cached_block->memory_block_ids);
    host_swapout_blocks.push_back(cached_block);

    // Append new cached block to buffer list.
    swapout_cached_block_buffer_[req_id].push_back(cached_block);
  }

  swapout_task_queue_[req_id] = threadpool_->Submit([=] {
    for (size_t i = 0; i < dev_swapout_blocks.size(); ++i) {
      SwapoutCachedBlock(dev_swapout_blocks[i], host_swapout_blocks[i]);
    }
  });

  return Status();
}

Status PrefixCacheManager::SwapinRequestAsync(int64_t req_id, size_t& block_num,
                                              std::vector<std::vector<int>>& req_block_ids,
                                              std::vector<int>& swapped_memory_block_ids) {
  if (!swapout_task_queue_.empty() || !swapout_cached_block_buffer_.empty() || !finish_swapout_request_.empty()) {
    return Status(RET_RUNTIME_FAILED,
                  FormatStr("Swap in req %d error, some swapout jobs is in progress. swapout_task_queue_=%d, "
                            "swapout_cached_block_buffer_=%d, finish_swapout_request_=%d",
                            req_id, swapout_task_queue_.size(), swapout_cached_block_buffer_.size(),
                            finish_swapout_request_.size()));
  }

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Swap in req %d error, req is not exist.", req_id));
  }

  std::vector<PrefixCachedBlock*> swapin_host_blocks;
  for (PrefixCachedBlock* cb : it->second->cached_blocks) {
    // Swapin block only if location is host.
    if (!cb->is_device_location) {
      swapin_host_blocks.push_back(cb);
    }
  }

  // Allocate block for next step.
  size_t step_block_num = GetRequestStepBlockNumberForOneNextToken(req_id);

  // Check whether enough memory exist, do not change anything swapin failed.
  size_t free_block_num = 0;
  size_t swapin_block_num = swapin_host_blocks.size() + step_block_num;
  if (free_cached_blocks_.size() < swapin_block_num &&
      !FreeCachedBlocks(swapin_block_num - free_cached_blocks_.size(), free_block_num)) {
    return Status(RET_OUT_OF_DEVICE_MEMORY, FormatStr("Swap in req %d error, No more free blocks.", req_id));
  }
  block_num = swapin_block_num;

  std::vector<PrefixCachedBlock*> swapin_dev_blocks;
  for (size_t i = 0; i < swapin_host_blocks.size(); ++i) {
    // Pick a empty cached block, replace it.
    PrefixCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();
    swapin_dev_blocks.push_back(cached_block);

    // Assume every device have same memory block id.
    swapped_memory_block_ids.push_back(cached_block->memory_block_ids[0]);

    // Append to buffer list.
    swapin_cached_block_buffer_[req_id].push_back(swapin_host_blocks[i]);
  }

  // Allocate next step block.
  for (size_t i = 0; i < step_block_num; ++i) {
    PrefixCachedBlock* cached_block = free_cached_blocks_.front();
    free_cached_blocks_.pop();

    // Fill memory block ids,
    for (size_t j = 0; j < block_device_num_; ++j) {
      req_block_ids[j].push_back(cached_block->memory_block_ids[j]);
    }

    // Not trace request info before a block is merge to tree.
    it->second->cached_blocks.push_back(cached_block);
  }

  swapin_task_queue_[req_id] = threadpool_->Submit([=] {
    for (size_t i = 0; i < swapin_host_blocks.size(); ++i) {
      {
        // For swapin, every block maybe processed by multiple thread, protect it.
        std::lock_guard<std::mutex> guard(swapin_host_blocks[i]->swapin_mutex);
        SwapinCachedBlock(swapin_dev_blocks[i], swapin_host_blocks[i]);
      }
    }
  });

  return Status();
}

Status PrefixCacheManager::WaitSwapoutRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking) {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::WaitSwapoutRequests(req_ids, left_req_num, blocking);
}

Status PrefixCacheManager::WaitSwapinRequests(std::vector<int64_t>& req_ids, size_t& left_req_num, bool blocking) {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::WaitSwapinRequests(req_ids, left_req_num, blocking);
}

Status PrefixCacheManager::MergeSwapoutRequest(int64_t req_id) {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::MergeSwapoutRequest(req_id);
}

Status PrefixCacheManager::MergeSwapinRequest(int64_t req_id, std::vector<std::vector<int>>& req_block_ids) {
  std::vector<PrefixCachedBlock*> swapin_blocks;

  swapin_blocks.swap(swapin_cached_block_buffer_[req_id]);
  swapin_cached_block_buffer_.erase(req_id);

  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    return Status(RET_RUNTIME_FAILED, FormatStr("Merge swapped req %d error, req is not exist.", req_id));
  }

  // Note: must from begin to end.
  for (size_t i = 0; i < it->second->cached_blocks.size(); ++i) {
    // Merge every node to the tree.
    PrefixCachedBlock* cb = it->second->cached_blocks[i];

    // Already on the tree, this node maybe merged by other request.
    if (cb->parent != nullptr) {
      // Update internal memory block ids of merged block.
      for (size_t j = 0; j < block_device_num_; ++j) {
        req_block_ids[j][i] = cb->memory_block_ids[j];
      }
      continue;
    }

    // Merge cb to existed node only if shareable.
    if (cb->is_shareable) {
      AppendSwapinCachedBlock(it->second.get(), i, cb, req_block_ids);
    } else {
      // Update internal memory block ids of unfilled block.
      for (size_t j = 0; j < block_device_num_; ++j) {
        req_block_ids[j][i] = cb->memory_block_ids[j];
      }
    }
  }

  // Remove from finished queue.
  auto it2 = std::find(finish_swapin_request_.begin(), finish_swapin_request_.end(), req_id);
  if (it2 != finish_swapin_request_.end()) {
    finish_swapin_request_.erase(it2);
  }

  // Update to running state.
  UpdateCachedRequestState(req_id, RequestState::kRunning);

  return Status();
}

void PrefixCacheManager::DestroySwappedRequest(int64_t req_id) {
  auto it = cached_requests_.find(req_id);
  if (it == cached_requests_.end()) {
    KLLM_LOG_ERROR << "DestroyFinishedRequest error, req " << req_id << " is not found in cached queue.";
    return;
  }

  PrefixCachedRequest* cached_request = it->second.get();
  for (PrefixCachedBlock* cb : cached_request->cached_blocks) {
    // If on host.
    if (!cb->is_device_location) {
      // free unique block directly, may be filled or unfilled.
      if (cb->active_requests.empty() && cb->inactive_requests.size() <= 1) {
        block_allocator_group_->GetHostBlockAllocator()->FreeBlocks(cb->memory_block_ids);
        delete cb;
        continue;
      }

      // Remove current reqeust from block if associated with other request.
      cb->inactive_requests.erase(req_id);
      continue;
    }

    // block on device.
    // The block on device of a swapped req must be shareable and on the tree, no need to check.
    // And maybe associated with a active request after a inactive request associated with it swapped out.

    // no other request associated with it, make reusable.
    if (cb->active_requests.empty() && cb->inactive_requests.size() == 1) {
      reusable_cached_blocks_.insert(cb);
      cb->inactive_requests.erase(req_id);
      continue;
    }

    // There are some other request associated with it, remove reqeust from block.
    cb->inactive_requests.erase(req_id);
  }

  cached_requests_.erase(it);
}

Status PrefixCacheManager::SwapoutRequestMemoryBlockAsync(int64_t req_id, const std::vector<int>& memory_block_ids) {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::SwapoutRequestMemoryBlockAsync(req_id,
                                                                                                  memory_block_ids);
}

Status PrefixCacheManager::SwapinRequestMemoryBlockAsync(int64_t req_id, const std::vector<int>& memory_block_ids) {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::SwapinRequestMemoryBlockAsync(req_id,
                                                                                                 memory_block_ids);
}

Status PrefixCacheManager::WaitSwapoutRequestMemoryBlock(const std::vector<int64_t>& req_ids) {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::WaitSwapoutRequestMemoryBlock(req_ids);
}

Status PrefixCacheManager::WaitSwapinRequestMemoryBlock(const std::vector<int64_t>& req_ids) {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::WaitSwapinRequestMemoryBlock(req_ids);
}

bool PrefixCacheManager::IsPrefixCachingEnabled() {
  return BaseCacheManager<PrefixCachedBlock, PrefixCachedRequest>::IsPrefixCachingEnabled();
}

}  // namespace ksana_llm
