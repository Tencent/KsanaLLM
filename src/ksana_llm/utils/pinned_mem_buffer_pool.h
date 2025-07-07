/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <atomic>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>
#include "ksana_llm/utils/blocking_queue.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {
struct PinnedMemoryBufferBlock {
  void* host_ptr = nullptr;    // Host-side pinned memory pointer
  void* device_ptr = nullptr;  // Device-side mapped pointer (same memory as host_ptr)
  size_t capacity = 0;
  int device_id = -1;

  PinnedMemoryBufferBlock() : host_ptr(nullptr), device_ptr(nullptr), capacity(0), device_id(-1) {}
};

class PinnedMemoryBufferPool {
 public:
  // Constructor that initializes the pool with blocks for all devices
  // device_count: number of devices to support
  // blocks_per_device: number of blocks per device
  // block_size: size of each block
  PinnedMemoryBufferPool(int device_count, size_t blocks_per_device, size_t block_size)
      : device_count_(device_count),
        blocks_per_device_(blocks_per_device),
        block_size_(block_size),
        available_blocks_(device_count) {
    int count = 0;
    GetDeviceCount(&count);
    if (device_count <= 0 || device_count > count) {
      KLLM_LOG_WARNING << "No CUDA devices found. PinnedMemoryBufferPool will be empty.";
      return;  // Return early with empty pool
    }

    if (block_size_ <= 0) {
      throw std::runtime_error("PinnedMemoryBufferPool: block_size must be > 0");
    }

    // Initialize blocks for all devices
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      SetDevice(device_id);
      // Pre-allocate blocks for this device
      for (size_t i = 0; i < blocks_per_device_; ++i) {
        auto block = new_allocate_block(device_id);
        if (block == nullptr) {
          KLLM_LOG_ERROR << "Failed to allocate pinned memory block on device " << device_id;
          throw std::runtime_error("Failed to allocate pinned memory block");
        }
        available_blocks_[device_id].Put(std::move(block));
      }
    }
  }
  ~PinnedMemoryBufferPool() {
    // Mark as destroyed to prevent new operations
    is_destroyed_.store(true);

    // First, stop all queues to prevent new operations
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      available_blocks_[device_id].Stop();
    }

    // Clean up all allocated pinned host memory
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      SetDevice(device_id);
      // Use NonBlockingGet to avoid waiting indefinitely
      auto block = available_blocks_[device_id].NonBlockingGet();
      while (block) {
        if (block->host_ptr) {
          FreeHostMapped(block->host_ptr, block->device_ptr);
        }
        block = available_blocks_[device_id].NonBlockingGet();
      }
    }
  }

  std::unique_ptr<PinnedMemoryBufferBlock> new_allocate_block(int device_id) {
    // Allocate a new block using heap allocation
    auto new_block = std::make_unique<PinnedMemoryBufferBlock>();

    // Set device before allocation
    SetDevice(device_id);

    HostAllocMapped(&new_block->host_ptr, &new_block->device_ptr, block_size_);

    if (new_block->host_ptr == nullptr || new_block->device_ptr == nullptr) {
      if (new_block->host_ptr) {
        FreeHostMapped(new_block->host_ptr, new_block->device_ptr);
      }
      throw std::runtime_error("Dynamic memory allocation returned null pointer");
    }

    new_block->capacity = block_size_;
    new_block->device_id = device_id;

    return new_block;
  }

  // Get a block from a specific device
  PinnedMemoryBufferBlock* get_block(int device_id) {
    if (is_destroyed_.load()) {
      KLLM_LOG_ERROR << "Attempting to get block from destroyed pool";
      return nullptr;
    }

    if (device_id < 0 || device_id >= device_count_) {
      throw std::invalid_argument("Invalid device_id: " + std::to_string(device_id));
    }

    // Try to get from available blocks for this device
    if (!available_blocks_[device_id].Empty()) {
      auto block = available_blocks_[device_id].Get();
      if (block) {
        PinnedMemoryBufferBlock* raw_ptr = block.release();  // Release ownership
        return raw_ptr;
      }
    }
    return nullptr;  // No available blocks, return nullptr
  }

  // Put a block back to the pool
  void put_block(PinnedMemoryBufferBlock* blk) {
    if (blk == nullptr) {
      KLLM_LOG_ERROR << "Trying to put null block back to pool";
      return;
    }

    if (is_destroyed_.load()) {
      KLLM_LOG_WARNING << "Pool is destroyed, cleaning up block manually";
      SetDevice(blk->device_id);
      if (blk->host_ptr) {
        FreeHostMapped(blk->host_ptr, blk->device_ptr);
      }
      delete blk;
      return;
    }

    int device_id = blk->device_id;
    if (device_id < 0 || device_id >= device_count_) {
      KLLM_LOG_ERROR << "Invalid device_id in block: " << device_id;
      return;
    }

    // Wrap the raw pointer back into unique_ptr and put it back to the queue
    auto block_ptr = std::unique_ptr<PinnedMemoryBufferBlock>(blk);
    available_blocks_[device_id].Put(std::move(block_ptr));
  }

 private:
  int device_count_;
  size_t blocks_per_device_;
  size_t block_size_;
  std::atomic<bool> is_destroyed_{false};

  // Available blocks for each device
  std::vector<BlockingQueue<std::unique_ptr<PinnedMemoryBufferBlock>>> available_blocks_;
};
}  // namespace ksana_llm