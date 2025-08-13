/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/dynamic_memory_pool.h"

#include <cstdint>
#include <stdexcept>

#include "fmt/format.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

DynamicMemoryPool::DynamicMemoryPool(void *base_memory_ptr, size_t memory_bytes, size_t block_size) {
  block_size_ = block_size;
  base_memory_ptr_ = AlignMemoryAddress(base_memory_ptr);

  const size_t padding =
      reinterpret_cast<std::uintptr_t>(base_memory_ptr_) - reinterpret_cast<std::uintptr_t>(base_memory_ptr);
  total_block_num_ = AlignDownMemorySize(memory_bytes - padding) / block_size_;
  int free_index = GetFreeBlockIndex(total_block_num_);

  // Calc the size of free block slot.
  free_blocks_.resize(free_index + 1);

  MemoryBlock memory_block{0, total_block_num_, true, free_index};
  memory_blocks_.push_back(memory_block);

  free_blocks_[free_index].push_back(memory_blocks_.begin());
  memory_blocks_.begin()->iter = free_blocks_[free_index].begin();
}

void *DynamicMemoryPool::GetBasePtr() { return base_memory_ptr_; }

size_t DynamicMemoryPool::GetTotalBlockNum() { return total_block_num_; }

size_t DynamicMemoryPool::GetFreeBlockNum() {
  size_t result = 0;
  for (auto &blocks : free_blocks_) {
    for (auto it : blocks) {
      result += it->length;
    }
  }
  return result;
}

size_t DynamicMemoryPool::GetUsedBlockNum() {
  size_t result = 0;
  for (auto &pair : resident_blocks_) {
    result += pair.second.length;
  }
  for (auto &pair : allocated_blocks_) {
    result += pair.second->length;
  }
  return result;
}

bool DynamicMemoryPool::CheckValidate() {
  bool check_succ = true;
  std::vector<void *> block_used(total_block_num_, nullptr);
  for (const auto &pair : allocated_blocks_) {
    for (size_t i = pair.second->offset; i < pair.second->length; ++i) {
      if (block_used[i] != nullptr) {
        KLLM_LOG_WARNING << "WARNING! duplicate memory, block addr: " << pair.first
                         << ", another block addr:" << block_used[i] << std::endl;
        check_succ = false;
      }
      block_used[i] = pair.first;
    }
  }
  return check_succ;
}

size_t DynamicMemoryPool::GetTotalByte() { return total_block_num_ * block_size_; }

size_t DynamicMemoryPool::GetMaxUsedByte() { return (max_resident_block_ + max_dynamic_block_) * block_size_; }

size_t DynamicMemoryPool::GetMaxContinuousFreeByte(bool resident) {
  if (resident) {
    MemoryBlock &memory_block = memory_blocks_.back();
    return memory_block.length * block_size_;
  }

  size_t max_block_num = 0;
  for (auto &blocks : free_blocks_) {
    for (auto it : blocks) {
      max_block_num = std::max(max_block_num, it->length);
    }
  }
  return max_block_num * block_size_;
}

void *DynamicMemoryPool::AlignMemoryAddress(void *addr) {
  const auto intptr = reinterpret_cast<std::uintptr_t>(addr);
  const auto aligned = (intptr - 1u + block_size_) & -block_size_;

  // Convert pointer to integer and back will lost some addtional information
  const auto diff = aligned - intptr;
  return reinterpret_cast<void *>(reinterpret_cast<char *>(addr) + diff);
}

size_t DynamicMemoryPool::AlignDownMemorySize(size_t size) { return size & -block_size_; }

size_t DynamicMemoryPool::AlignUpMemorySize(size_t size) { return (size - 1u + block_size_) & -block_size_; }

void *DynamicMemoryPool::GetMemoryBlockPtr(const MemoryBlock &memory_block) {
  return reinterpret_cast<uint8_t *>(base_memory_ptr_) + (memory_block.offset * block_size_);
}

void *DynamicMemoryPool::Allocate(size_t memory_bytes, bool resident) {
  if (memory_bytes == 0) {
    // KLLM_LOG_WARNING << "WARNING! Allocate zero bytes, resident: " << resident << std::endl;
    return dummpy_zero_address_;
  }

  if (resident) {
    return AllocateResident(memory_bytes);
  }
  return AllocateDynamic(memory_bytes);
}

void *DynamicMemoryPool::AllocateResident(size_t memory_bytes) {
  size_t allocate_size = (memory_bytes < block_size_) ? block_size_ : memory_bytes;
  size_t block_num = AlignUpMemorySize(allocate_size) / block_size_;

  // At least one block exists, no check for performance.
  MemoryBlock &memory_block = memory_blocks_.back();
  if (memory_block.length < block_num) {
    throw std::runtime_error(fmt::format(
        "Not enough resident memory block, bytes {}, block num {}, total: {}, free: {} used: {}, max_used_byte: {}",
        memory_bytes, block_num, GetTotalBlockNum(), GetFreeBlockNum(), GetUsedBlockNum(), GetMaxUsedByte()));
  }

  // New allocated memory block.
  MemoryBlock allocated_block;
  allocated_block.offset = memory_block.offset + memory_block.length - block_num;
  allocated_block.length = block_num;
  allocated_block.is_free = false;

  // Record maximum resident size.
  max_resident_block_ = std::max(max_resident_block_, total_block_num_ - allocated_block.offset);

  // Shrink free memory block.
  memory_block.length -= block_num;

  void *result_ptr = GetMemoryBlockPtr(allocated_block);
  resident_blocks_[result_ptr] = allocated_block;
  return result_ptr;
}

size_t DynamicMemoryPool::GetFreeBlockIndex(size_t block_num) { return std::log2(block_num); }

void *DynamicMemoryPool::AllocateDynamic(size_t memory_bytes) {
  size_t allocate_size = (memory_bytes < block_size_) ? block_size_ : memory_bytes;
  size_t block_num = AlignUpMemorySize(allocate_size) / block_size_;

  size_t start_index = GetFreeBlockIndex(block_num);

  void *result_ptr = nullptr;
  for (size_t index = start_index; index < free_blocks_.size(); ++index) {
    std::list<std::list<MemoryBlock>::iterator> &free_blocks = free_blocks_[index];

    bool terminated = false;
    for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
      if ((*it)->length < block_num) {
        continue;
      }
      terminated = true;

      // full matched, just change block state.
      if ((*it)->length == block_num) {
        (*it)->is_free = false;
        (*it)->iter = std::list<std::list<MemoryBlock>::iterator>::iterator();

        // Change to allocated list.
        result_ptr = GetMemoryBlockPtr(**it);
        allocated_blocks_[result_ptr] = *it;

        // Record maximum dynamic size.
        max_dynamic_block_ = std::max(max_dynamic_block_, (*it)->offset + (*it)->length);

        // Remove from free list
        free_blocks.erase(it);
        break;
      } else {
        // Split free block, new allocated block is on the head.
        MemoryBlock allocated_block;
        allocated_block.offset = (*it)->offset;
        allocated_block.length = block_num;
        allocated_block.is_free = false;

        // Change size of original free block
        (*it)->offset += block_num;
        (*it)->length -= block_num;

        // Add new block to block list, before current block.
        std::list<MemoryBlock>::iterator new_it = memory_blocks_.insert(*it, allocated_block);

        // Add new block to allocated list.
        result_ptr = GetMemoryBlockPtr(allocated_block);
        allocated_blocks_[result_ptr] = new_it;

        // Record maximum dynamic size.
        max_dynamic_block_ = std::max(max_dynamic_block_, allocated_block.offset + allocated_block.length);

        // Move old free block to corrent list.
        size_t free_index = GetFreeBlockIndex((*it)->length);
        if (free_index != index) {
          free_blocks_[free_index].push_back(*it);

          // Update old free block's slot & iter.
          (*it)->slot = free_index;
          (*it)->iter = --free_blocks_[free_index].end();

          // Remove from current free list
          free_blocks.erase(it);
        }

        break;
      }
    }

    if (terminated) {
      break;
    }
  }

  if (result_ptr == nullptr) {
    throw std::runtime_error(fmt::format(
        "Not enough dynamic memory block, bytes {}, block num {}, total: {}, free: {} used: {}, max_used_byte: {}",
        memory_bytes, block_num, GetTotalBlockNum(), GetFreeBlockNum(), GetUsedBlockNum(), GetMaxUsedByte()));
  }

  return result_ptr;
}

void DynamicMemoryPool::Free(void *memory_ptr) {
  if (memory_ptr == nullptr) {
    throw std::runtime_error("The memory ptr is nullptr.");
  } else if (memory_ptr == dummpy_zero_address_) {
    // skip empty memory block.
    return;
  }

  auto it = allocated_blocks_.find(memory_ptr);
  if (it == allocated_blocks_.end()) {
    if (resident_blocks_.find(memory_ptr) == resident_blocks_.end()) {
      KLLM_LOG_ERROR << "ERROR! The memory ptr " << memory_ptr << " is not found in allocated blocks";
    }
    return;
  }

  auto block_it = it->second;
  block_it->is_free = true;

  // Merge previous block.
  std::list<MemoryBlock>::iterator prev_it = block_it;
  if (prev_it != memory_blocks_.begin()) {
    --prev_it;
    while (true) {
      if (!prev_it->is_free) {
        break;
      }
      bool is_begin = (prev_it == memory_blocks_.begin());

      // Merge previous block memory.
      block_it->offset = prev_it->offset;
      block_it->length += prev_it->length;

      auto new_prev_it = prev_it;
      if (!is_begin) {
        --new_prev_it;
      }

      // Remove merged node on free list.
      size_t index = prev_it->slot;

      std::list<std::list<MemoryBlock>::iterator>::iterator &iter = prev_it->iter;

      free_blocks_[index].erase(iter);

      // Remove merged node on block list.
      memory_blocks_.erase(prev_it);
      prev_it = new_prev_it;
      if (is_begin) {
        break;
      }
    }
  }

  // Merge post block.
  std::list<MemoryBlock>::iterator post_it = block_it;
  ++post_it;
  while (post_it != memory_blocks_.end()) {
    if (post_it == memory_blocks_.end() || !post_it->is_free) {
      break;
    }

    // Merge previous block memory.
    block_it->length += post_it->length;

    auto new_post_it = post_it;
    ++new_post_it;

    // Remove merged node on free list.
    size_t index = post_it->slot;

    std::list<std::list<MemoryBlock>::iterator>::iterator &iter = post_it->iter;

    free_blocks_[index].erase(iter);

    // Remove merged node on block list.
    memory_blocks_.erase(post_it);
    post_it = new_post_it;
  }

  // Move from allocated to free list.
  int index = GetFreeBlockIndex(block_it->length);
  block_it->slot = index;
  free_blocks_[index].push_back(block_it);
  block_it->iter = --free_blocks_[index].end();
  allocated_blocks_.erase(it);
}

void DeviceMemoryPool::SetMemoryPool(int rank, void *base_ptr, size_t bytes, size_t block_size) {
  if (memory_pools_.empty()) {
    int device_count;
    GetDeviceCount(&device_count);
    memory_pools_.resize(device_count, nullptr);
  }
  memory_pools_[rank] = std::make_shared<DynamicMemoryPool>(base_ptr, bytes, block_size);
}

std::shared_ptr<DynamicMemoryPool> &DeviceMemoryPool::GetMemoryPool(size_t rank) {
  // Do not check rank size, for performance.
  return memory_pools_[rank];
}

}  // namespace ksana_llm
