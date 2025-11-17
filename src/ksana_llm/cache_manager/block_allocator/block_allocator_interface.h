/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The unified interface of all cache block manager.
class BlockAllocatorInterface {
 public:
  // pre-allocate all blocks.
  virtual void PreAllocateBlocks() = 0;

  // Free all blocks.
  virtual void Clear() = 0;

  // Allocate blocked memory.
  virtual Status AllocateBlocks(size_t block_num, std::vector<int>& blocks) = 0;

  // Free blocked memory.
  virtual Status FreeBlocks(const std::vector<int>& blocks) = 0;

  // Get memory address of blocked memory.
  virtual Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) = 0;

  // Append new memory address of blocked memory.
  virtual Status AppendBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) = 0;

  // Used for ATB mode, all blocks is part of a whole flatten memory space
  virtual void* GetBlocksBasePtr() = 0;

  // Used for ATB mode, return the first allocated block id
  virtual int GetBlocksBaseId() = 0;

  // Get number of free blocked memory.
  virtual size_t GetFreeBlockNumber() = 0;

  // Get number of used blocked memory.
  virtual size_t GetUsedBlockNumber() = 0;
};

}  // namespace ksana_llm
