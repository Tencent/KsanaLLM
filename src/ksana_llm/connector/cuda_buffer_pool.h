/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>
#include "ksana_llm/utils/blocking_queue.h"
#ifdef ENABLE_CUDA
#  include <cuda_runtime.h>
#  include <nccl.h>
#  include "ksana_llm/utils/nvidia/cuda_utils.h"
#endif

namespace ksana_llm {
#ifdef ENABLE_CUDA
struct BufferBlock {
  void* device_ptr = nullptr;
  size_t capacity = 0;
  int device_id = -1;

  BufferBlock() : device_ptr(nullptr), capacity(0), device_id(-1) {}
};

class BufferPool {
 public:
  BufferPool(int device_id, size_t N, size_t block_size) : device_id_(device_id), block_size_(block_size) {
    CUDA_CHECK(cudaSetDevice(device_id_));

    // Pre-allocate blocks using heap allocation to avoid pointer invalidation
    for (size_t i = 0; i < N; ++i) {
      auto block = std::make_unique<BufferBlock>();

      // Add error checking and debug info
      cudaError_t error = cudaMalloc(&block->device_ptr, block_size_);
      if (error != cudaSuccess) {
        KLLM_LOG_ERROR << "cudaMalloc failed for block " << i << " size " << block_size_ << " on device " << device_id_
                       << ": " << cudaGetErrorString(error);
        throw std::runtime_error("Failed to allocate GPU memory");
      }

      // Verify the allocated pointer is valid
      if (block->device_ptr == nullptr) {
        KLLM_LOG_ERROR << "cudaMalloc returned nullptr for block " << i;
        throw std::runtime_error("cudaMalloc returned null pointer");
      }

      // Log successful allocation for debugging
      KLLM_LOG_DEBUG << "Successfully allocated block " << i << " at device_ptr=" << block->device_ptr
                     << " size=" << block_size_;

      block->capacity = block_size_;
      block->device_id = device_id_;

      BufferBlock* block_ptr = block.get();
      blocks_.push_back(std::move(block));
      queue_.Put(block_ptr);
    }
  }

  ~BufferPool() {
    // Clean up all allocated GPU memory
    for (auto& block : blocks_) {
      if (block->device_ptr) {
        cudaFree(block->device_ptr);
      }
    }
  }

  BufferBlock* get_block() {
    // 首先尝试从队列获取已有的块
    if (!queue_.Empty()) {
      BufferBlock* block = queue_.Get();
      // Verify the block pointer is still valid
      if (block == nullptr) {
        KLLM_LOG_ERROR << "Got nullptr from queue";
        throw std::runtime_error("Queue returned null pointer");
      }
      if (block->device_ptr == nullptr) {
        KLLM_LOG_ERROR << "Block has null device_ptr";
        throw std::runtime_error("Block has null device pointer");
      }

      KLLM_LOG_DEBUG << "Retrieved existing block: device_ptr=" << block->device_ptr << " capacity=" << block->capacity
                     << " device_id=" << block->device_id;
      return block;
    }

    // 如果队列为空，检查 block_size_ 是否有效
    if (block_size_ == 0) {
      throw std::runtime_error("BufferPool not properly initialized: block_size is 0");
    }
    KLLM_LOG_WARNING << "No available blocks in the pool, allocating a new block of size " << block_size_
                     << " on device " << device_id_;

    // 动态分配新的块 - use heap allocation for thread safety
    auto new_block = std::make_unique<BufferBlock>();

    // Set device before allocation
    CUDA_CHECK(cudaSetDevice(device_id_));

    cudaError_t error = cudaMalloc(&new_block->device_ptr, block_size_);
    if (error != cudaSuccess) {
      KLLM_LOG_ERROR << "cudaMalloc failed during dynamic allocation, size " << block_size_ << " on device "
                     << device_id_ << ": " << cudaGetErrorString(error);
      throw std::runtime_error("Failed to dynamically allocate GPU memory");
    }

    if (new_block->device_ptr == nullptr) {
      KLLM_LOG_ERROR << "Dynamic cudaMalloc returned nullptr";
      throw std::runtime_error("Dynamic cudaMalloc returned null pointer");
    }

    new_block->capacity = block_size_;
    new_block->device_id = device_id_;

    KLLM_LOG_DEBUG << "Dynamically allocated block: device_ptr=" << new_block->device_ptr
                   << " capacity=" << new_block->capacity;

    BufferBlock* block_ptr = new_block.get();
    // 将新块添加到 blocks_ 向量中以管理其生命周期
    // Use mutex to protect vector modification in multi-threaded environment
    std::lock_guard<std::mutex> lock(blocks_mutex_);
    blocks_.push_back(std::move(new_block));

    // 返回新分配块的指针
    return block_ptr;
  }

  void put_block(BufferBlock* blk) { queue_.Put(blk); }

 private:
  int device_id_;
  size_t block_size_;
  std::vector<std::unique_ptr<BufferBlock>> blocks_;
  BlockingQueue<BufferBlock*> queue_;
  std::mutex blocks_mutex_;  // Protect blocks_ vector in multi-threaded access
};
#endif
}  // namespace ksana_llm
