/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/cache_manager/block_allocator/block_allocator.h"

#include "fmt/core.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

BlockAllocator::BlockAllocator(MemoryLocation location, size_t block_num, size_t block_size, int rank,
                               std::shared_ptr<MemoryAllocatorInterface> memory_allocator,
                               std::shared_ptr<Context> context, size_t convert_size)
    : location_(location),
      block_num_(block_num),
      block_size_(block_size),
      convert_size_(convert_size),
      rank_(rank),
      memory_allocator_(memory_allocator),
      context_(context) {
  using namespace std::placeholders;
  if (location_ == MemoryLocation::LOCATION_HOST) {
    malloc_fn_ = std::bind(&MemoryAllocatorInterface::HostAlloc, memory_allocator.get(), _1, _2);
    free_fn_ = std::bind(&MemoryAllocatorInterface::HostFree, memory_allocator.get(), _1);
  } else if (location_ == MemoryLocation::LOCATION_DEVICE) {
    malloc_fn_ = std::bind(&MemoryAllocatorInterface::MallocAsync, memory_allocator.get(), _1, _2,
                           context_->GetMemoryManageStreams()[rank_]);
    free_fn_ = std::bind(&MemoryAllocatorInterface::FreeAsync, memory_allocator.get(), _1,
                         context_->GetMemoryManageStreams()[rank_]);
  } else {
    KLLM_THROW("The MemoryLocation is not supported.");
  }
}

BlockAllocator::~BlockAllocator() { Clear(); }

void BlockAllocator::PreAllocateBlocks() {
  bool use_continuous_memory = false;
  uint8_t* base_mem_ptr = nullptr;

  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    SetDevice(rank_);
  }

#if defined(ENABLE_ACL) || defined(ENABLE_FLASH_ATTN_WITH_CACHE)
  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    use_continuous_memory = true;
    malloc_fn_(reinterpret_cast<void**>(&base_mem_ptr), (block_num_ + 1) * (block_size_ + convert_size_));
    blocks_base_ptr_ = base_mem_ptr;
  }
#endif

  // NOTE: Make sure block ids on all worker nodes have same id range.
  free_blocks_.reserve(block_num_);
  void* memory_ptr = nullptr;
  for (size_t block_id = 0; block_id < block_num_; ++block_id) {
    if (use_continuous_memory) {
      memory_ptr = base_mem_ptr + block_id * block_size_;
    } else {
      malloc_fn_(&memory_ptr, block_size_);
    }
    free_blocks_.emplace(block_id, memory_ptr);
  }
}

void BlockAllocator::Clear() {
  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    SetDevice(rank_);
  }

#if defined(ENABLE_ACL_ATB) || defined(ENABLE_FLASH_ATTN_WITH_CACHE)
  if (location_ == MemoryLocation::LOCATION_DEVICE) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (blocks_base_ptr_ != nullptr) {
      free_fn_(blocks_base_ptr_);
      blocks_base_ptr_ = nullptr;
    }
    free_blocks_.clear();
    used_blocks_.clear();
    return;
  }
#endif

  {
    auto clear_fn = [&](std::unordered_map<int, void*>& blocks) -> void {
      for (auto it = blocks.begin(); it != blocks.end();) {
        free_fn_(it->second);
        it = blocks.erase(it);
      }
    };

    std::unique_lock<std::mutex> lock(mutex_);
    clear_fn(free_blocks_);
    clear_fn(used_blocks_);
  }
}

Status BlockAllocator::AllocateBlocks(size_t block_num, std::vector<int>& blocks) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (block_num > free_blocks_.size()) {
    return Status(RET_DEVICE_MEM_ALLOCATE_FAILED,
                  FormatStr("No more free blocks, expect %d, free %d", block_num, free_blocks_.size()));
  }

  blocks.clear();
  blocks.reserve(block_num);
  auto it = free_blocks_.begin();
  while (block_num--) {
    used_blocks_.insert(*it);
    blocks.push_back(it->first);
    it = free_blocks_.erase(it);
  }
  return Status();
}

Status BlockAllocator::FreeBlocks(const std::vector<int>& blocks) {
  std::unique_lock<std::mutex> lock(mutex_);

  for (auto block_id : blocks) {
    auto it = used_blocks_.find(block_id);
    if (it != used_blocks_.end()) {
      free_blocks_.insert(*it);
      used_blocks_.erase(it);
    } else {
      return Status(RET_DEVICE_MEM_FREE_FAILED, fmt::format("Double free error, block id {}", block_id));
    }
  }
  return Status();
}

Status BlockAllocator::GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) {
  std::unique_lock<std::mutex> lock(mutex_);

  addrs.clear();
  addrs.reserve(blocks.size());
  for (auto block_id : blocks) {
    auto it = used_blocks_.find(block_id);
    if (it != used_blocks_.end()) {
      addrs.push_back(it->second);
      continue;
    }

    // For distributed worker node, get from free blocks.
    if (!context_->IsChief()) {
      auto it2 = free_blocks_.find(block_id);
      if (it2 != free_blocks_.end()) {
        addrs.push_back(it2->second);
        continue;
      }
    }

    KLLM_LOG_ERROR << "Get block id " << block_id << " address error on device " << rank_;
    return Status(RET_SEGMENT_FAULT, FormatStr("Get block address error, block id {}", block_id));
  }
  return Status();
}

void* BlockAllocator::GetBlocksBasePtr() { return blocks_base_ptr_; }

int BlockAllocator::GetBlocksBaseId() { return block_base_id_; }

size_t BlockAllocator::GetFreeBlockNumber() {
  std::unique_lock<std::mutex> lock(mutex_);
  return free_blocks_.size();
}

size_t BlockAllocator::GetUsedBlockNumber() {
  std::unique_lock<std::mutex> lock(mutex_);
  return used_blocks_.size();
}

}  // namespace ksana_llm
