/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ksana_llm {

//
// A dynamic buffer pool implementation.
// Manage all the tensor buffer as a whole memory area,
// provide allocate and free through offset.
// +----------------------------+-------------------------+
// | dynamic memory ->          |      <- resident memory |
// +----------------------------+-------------------------+
// low address                                hight address
//
class DynamicMemoryPool {
 public:
  struct MemoryBlock {
    // The block offset of this block.
    size_t offset;

    // The total block number of this block.
    size_t length;

    // Whether current block is free.
    bool is_free = false;

    // The index of free block list of current block, for free block only.
    int slot = -1;

    // the iterator of current block on free block list, for free block only.
    std::list<std::list<MemoryBlock>::iterator>::iterator iter;
  };

 public:
  DynamicMemoryPool(void *base_memory_ptr, size_t memory_bytes, size_t block_size);

  void *Allocate(size_t memory_bytes, bool resident = false);
  void Free(void *memory_ptr);

  // Get bock num info.
  size_t GetTotalBlockNum();
  size_t GetFreeBlockNum();
  size_t GetUsedBlockNum();

  bool CheckValidate();

  // Get max used or free continuous bytes.
  size_t GetTotalByte();
  size_t GetMaxUsedByte();
  size_t GetMaxContinuousFreeByte(bool resident = false);

  // Get base address.
  void *GetBasePtr();

 private:
  void *AlignMemoryAddress(void *addr);
  size_t AlignDownMemorySize(size_t size);
  size_t AlignUpMemorySize(size_t size);

  // Allocate memory on resident area.
  void *AllocateResident(size_t memory_bytes);

  // Allocate memory on dynamic area.
  void *AllocateDynamic(size_t memory_bytes);

  // Get address of memory block.
  void *GetMemoryBlockPtr(const MemoryBlock &memory_block);

  // Get index of free list of current block number.
  size_t GetFreeBlockIndex(size_t block_num);

 private:
  void *base_memory_ptr_ = nullptr;
  size_t block_size_ = 4096;

  // The total block number.
  size_t total_block_num_;

  // For stat.
  size_t max_resident_block_ = 0;
  size_t max_dynamic_block_ = 0;

  // All memory blocks, including free and allocated.
  std::list<MemoryBlock> memory_blocks_;

  // All the allocated memory blocks.
  std::unordered_map<void *, std::list<MemoryBlock>::iterator> allocated_blocks_;

  // all the allocated resident blocks.
  std::unordered_map<void *, MemoryBlock> resident_blocks_;

  // The free memory blocks that block number between [2^i, 2^i+1),
  // i is vector index.
  std::vector<std::list<std::list<MemoryBlock>::iterator>> free_blocks_;

  // A faked pointer address for zero bytes, distinguish with nullptr.
  // This pointer is just a placeholder, it should not be read or write.
  void *dummpy_zero_address_ = reinterpret_cast<void *>(0x1);
};

class DeviceMemoryPool {
 public:
  // Set memory pool for specific device rank.
  static void SetMemoryPool(int rank, void *base_ptr, size_t bytes, size_t block_size);

  // Get memory pool of specific device rank.
  static std::shared_ptr<DynamicMemoryPool> &GetMemoryPool(size_t rank);

  // Whether current memory is empty.
  static bool Empty() { return memory_pools_.empty(); }

  static void Enable() { enabled_ = true; }
  static void Disable() { enabled_ = false; }
  static bool IsEnabled() { return enabled_; }
  static bool IsDisabled() { return !enabled_; }

 private:
  // The memory pools, shared by all tensors.
  static inline std::vector<std::shared_ptr<DynamicMemoryPool>> memory_pools_;

  // Whther the memory pool is enabled.
  static inline bool enabled_ = true;
};

}  // namespace ksana_llm
