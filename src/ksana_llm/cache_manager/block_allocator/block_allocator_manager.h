/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "ksana_llm/cache_manager/block_allocator/block_allocator_interface.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/memory_allocator_interface.h"

namespace ksana_llm {

// The config of a block allocator group.
struct BlockAllocatorGroupConfig {
  // The device ids of this group.
  std::vector<int> devices;

  // The device block num.
  size_t device_block_num;

  // The host block num.
  size_t host_block_num;

  // The block size, in bytes
  size_t block_size;

  // The buffer size required for dequantization operations, in bytes.
  size_t convert_size = 0;
};

// The config of all block allocator groups.
using BlockAllocatorManagerConfig = std::unordered_map<int, BlockAllocatorGroupConfig>;

// Function used to create block allocator, for testing.
using BlockAllocatorCreationFunc = std::function<std::shared_ptr<BlockAllocatorInterface>(
    MemoryLocation location, size_t block_num, size_t block_size, int rank,
    std::shared_ptr<MemoryAllocatorInterface> memory_allocator, std::shared_ptr<Context> context, size_t convert_size)>;

//
class BlockAllocatorGroupInterface {
 public:
  virtual ~BlockAllocatorGroupInterface() {}

  virtual std::vector<int> GetBlockAllocatorDevices() const = 0;

  virtual std::shared_ptr<BlockAllocatorInterface> GetHostBlockAllocator() const = 0;
  virtual std::shared_ptr<BlockAllocatorInterface> GetDeviceBlockAllocator(int index = 0) const = 0;

  virtual Status SwapIn(int index, int device_block_id, int host_block_id) = 0;
  virtual Status SwapOut(int index, int host_block_id, int device_block_id) = 0;
};

// Function used to create block allocator group, for testing.
using BlockAllocatorGroupCreationFunc = std::function<std::shared_ptr<BlockAllocatorGroupInterface>(
    const BlockAllocatorGroupConfig& block_allocator_group_config,
    std::shared_ptr<MemoryAllocatorInterface> memory_allocator, std::shared_ptr<Context> context,
    BlockAllocatorCreationFunc block_allocator_creation_fn)>;

// A block allocator group.
class BlockAllocatorGroup : public BlockAllocatorGroupInterface {
 public:
  virtual ~BlockAllocatorGroup() {}

  BlockAllocatorGroup(const BlockAllocatorGroupConfig& block_allocator_group_config,
                      std::shared_ptr<MemoryAllocatorInterface> memory_allocator, std::shared_ptr<Context> context,
                      BlockAllocatorCreationFunc block_allocator_creation_fn);

  virtual std::vector<int> GetBlockAllocatorDevices() const override;

  virtual std::shared_ptr<BlockAllocatorInterface> GetHostBlockAllocator() const override;
  virtual std::shared_ptr<BlockAllocatorInterface> GetDeviceBlockAllocator(int index = 0) const override;

  virtual Status SwapIn(int index, int device_block_id, int host_block_id) override;
  virtual Status SwapOut(int index, int host_block_id, int device_block_id) override;

 private:
  void Initialize();

 private:
  BlockAllocatorGroupConfig block_allocator_group_config_;

  // The host allocator
  std::shared_ptr<BlockAllocatorInterface> host_allocator = nullptr;

  // All the device block allocators, the key is device_id.
  std::unordered_map<int, std::shared_ptr<BlockAllocatorInterface>> dev_allocators;

  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;

  std::shared_ptr<Context> context_ = nullptr;

  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
};

// Used to manage all the block allocator.
class BlockAllocatorManager {
 public:
  BlockAllocatorManager(const BlockAllocatorManagerConfig& block_allocator_manager_config,
                        std::shared_ptr<MemoryAllocatorInterface> memory_allocator, std::shared_ptr<Context> context,
                        BlockAllocatorCreationFunc block_allocator_creation_fn = nullptr,
                        BlockAllocatorGroupCreationFunc block_allocator_group_creation_fn = nullptr);

  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup(int group_id);

 private:
  void InitializeBlockAllocatorGroups();

 private:
  BlockAllocatorManagerConfig block_allocator_manager_config_;

  // The key is group_id, for example, every dp is a group.
  std::unordered_map<int, std::shared_ptr<BlockAllocatorGroupInterface>> allocator_groups_;

  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;

  std::shared_ptr<Context> context_ = nullptr;

  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
  BlockAllocatorGroupCreationFunc block_allocator_group_creation_fn_ = nullptr;
};

}  // namespace ksana_llm
