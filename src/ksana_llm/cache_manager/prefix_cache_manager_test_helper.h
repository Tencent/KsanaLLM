/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <list>
#include <unordered_map>
#include <vector>

#include "ksana_llm/cache_manager/block_allocator/block_allocator_interface.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/memory_allocator_interface.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

class FakedMemoryAllocator : public MemoryAllocatorInterface {
 public:
  virtual ~FakedMemoryAllocator() {}

  virtual void SetDevice(int device_id) override {}
  virtual void GetDevice(int* device_id) override {}

  virtual void Malloc(void** dev_ptr, size_t size) override {}
  virtual void MallocAsync(void** dev_ptr, size_t size, Stream stream) override {}

  virtual void MemsetAsync(void* dev_ptr, int value, size_t count, Stream stream) override {}
  virtual void Memset(void* dev_ptr, int value, size_t count) override {}

  virtual void MemcpyAsync(void* dst, const void* src, size_t count, enum MemcpyKind kind, Stream stream) override {}
  virtual void Memcpy(void* dst, const void* src, size_t count, enum MemcpyKind kind) override {}

  virtual void Free(void* dev_ptr) override {}
  virtual void FreeAsync(void* dev_ptr, Stream stream) override {}

  virtual void HostAlloc(void** host_ptr, size_t size) override {}

  virtual void HostFree(void* host_ptr) override {}
};

class FakedBlockAllocator : public BlockAllocatorInterface {
 public:
  FakedBlockAllocator(MemoryLocation location, size_t block_num, size_t block_size, int rank = 0,
                      std::shared_ptr<MemoryAllocatorInterface> memory_allocator = nullptr,
                      std::shared_ptr<Context> context = nullptr) {
    location_ = location;
    block_num_ = block_num;
    block_size_ = block_size;
    rank_ = rank;
    memory_allocator_ = memory_allocator;
    context_ = context;
  }

  virtual ~FakedBlockAllocator() {}

  virtual void PreAllocateBlocks() override {
    for (size_t i = 0; i < block_num_; ++i) {
      free_blocks_.push_back(i);
    }
  }

  virtual void Clear() override {
    free_blocks_.clear();
    used_blocks_.clear();
  }

  virtual Status AllocateBlocks(size_t block_num, std::vector<int>& blocks) override {
    blocks.clear();
    size_t needed_block_num = block_num;
    while (needed_block_num > 0 && !free_blocks_.empty()) {
      int block_id = free_blocks_.front();
      blocks.push_back(block_id);

      free_blocks_.pop_front();
      used_blocks_.push_back(block_id);

      --needed_block_num;
    }

    RetCode ret_code = (location_ == LOCATION_HOST) ? RET_OUT_OF_HOST_MEMORY : RET_OUT_OF_DEVICE_MEMORY;
    return (needed_block_num == 0) ? Status() : Status(ret_code, "No more blocks.");
  }

  virtual Status FreeBlocks(const std::vector<int>& blocks) override {
    for (int block_id : blocks) {
      auto it = std::find(used_blocks_.begin(), used_blocks_.end(), block_id);
      if (it != used_blocks_.end()) {
        used_blocks_.erase(it);
      }
      free_blocks_.push_back(block_id);
    }

    return Status();
  }

  virtual Status GetBlockPtrs(const std::vector<int>& blocks, std::vector<void*>& addrs) override {
    addrs.resize(blocks.size(), nullptr);
    return Status();
  }

  virtual void* GetBlocksBasePtr() override { return nullptr; }

  virtual int GetBlocksBaseId() override { return 0; }

  virtual size_t GetFreeBlockNumber() override { return free_blocks_.size(); }

  virtual size_t GetUsedBlockNumber() override { return used_blocks_.size(); }

 private:
  MemoryLocation location_ = MemoryLocation::LOCATION_HOST;

  size_t block_num_;
  size_t block_size_;

  int rank_ = -1;

  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;

  std::list<int> free_blocks_;
  std::list<int> used_blocks_;
};

// A faked token generator.
class FakedTokenGenerator {
 public:
  // Generate some tokens by pair of seed and length.
  void GeneratePromptTokens(const std::vector<std::pair<int, int>>& seeds, std::vector<int>& token_ids) {
    for (auto& pair : seeds) {
      std::srand(pair.first);
      for (int i = 0; i < pair.second; ++i) {
        token_ids.push_back(std::rand() % vocab_size);
      }
    }
  }

  // Generate a random size.
  int GenerateRandomInteger(int min, int max) {
    std::srand(std::time(nullptr));
    return min + (std::rand() % (max - min));
  }

  // Generate a new token.
  void GenerateOneToken(size_t token_num, std::vector<int>& token_ids) {
    for (size_t i = 0; i < token_num; ++i) {
      token_ids.push_back(GenerateRandomInteger(0, vocab_size));
    }
  }

  size_t GetVocabSize() { return vocab_size; }

 private:
  size_t vocab_size = 32000;
};

}  // namespace ksana_llm
