/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>
#include "ksana_llm/cache_manager/block_allocator/block_allocator_interface.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/memory_allocator_interface.h"

namespace ksana_llm {

class BlockAllocator : public BlockAllocatorInterface {
 public:
  BlockAllocator(MemoryLocation location, size_t block_num, size_t block_size, int rank = 0,
                 std::shared_ptr<MemoryAllocatorInterface> memory_allocator = nullptr,
                 std::shared_ptr<Context> context = nullptr, size_t convert_size = 0);

  virtual ~BlockAllocator();

  virtual void PreAllocateBlocks() override;

  // Free all blocks.
  virtual void Clear() override;

  // Allocate blocked memory.
  virtual Status AllocateBlocks(size_t block_num, std::vector<int>& blocks) override;

  // Free blocked memory.
  virtual Status FreeBlocks(const std::vector<int>& blocks) override;

  // Get memory address of blocked memory.
  virtual Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) override;

  // Used for ATB mode, all blocks is part of a whole flatten memory space
  virtual void* GetBlocksBasePtr() override;

  // Used for ATB mode, return the first allocated block id
  virtual int GetBlocksBaseId() override;

  // Get number of free blocked memory.
  virtual size_t GetFreeBlockNumber() override;

  // Get number of used blocked memory.
  virtual size_t GetUsedBlockNumber() override;

 private:
  MemoryLocation location_ = MemoryLocation::LOCATION_HOST;

  size_t block_num_;
  size_t block_size_;
  // The buffer size required for dequantization operations, in bytes.
  size_t convert_size_ = 0;

  // device rank, -1 for host.
  int rank_ = -1;

  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;

  // The malloc function.
  std::function<void(void**, size_t)> malloc_fn_;
  std::function<void(void*)> free_fn_;

  // block id to address
  std::unordered_map<int, void*> free_blocks_;
  std::unordered_map<int, void*> used_blocks_;

  // Make thread-safe.
  std::mutex mutex_;

  // blocks base pointer used for project kvcache mem to NPU k/vcache mem
  void* blocks_base_ptr_ = nullptr;

  // blocks base id for project kvcache mem to NPU k/vcache mem
  int block_base_id_ = 0;
};

}  // namespace ksana_llm
