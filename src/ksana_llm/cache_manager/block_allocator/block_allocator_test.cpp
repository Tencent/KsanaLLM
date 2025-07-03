/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/cache_manager/block_allocator/block_allocator.h"
#include <gtest/gtest.h>
#include "ksana_llm/cache_manager/block_allocator/block_allocator_manager.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "test.h"

using namespace ksana_llm;

class BlockAllocatorTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>(device_num_, attn_dp_worker_num_, multi_batch_num_);
    memory_allocator_ = std::make_shared<MemoryAllocator>();
  }

  void TearDown() override {}

 protected:
  int device_num_ = 1;
  uint32_t attn_dp_worker_num_ = 1;
  size_t multi_batch_num_ = 1;

  std::shared_ptr<BlockAllocatorInterface> block_allocator_ = nullptr;

  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;
};

TEST_F(BlockAllocatorTest, TestBlockAllocator) {
  for (MemoryLocation location : {MemoryLocation::LOCATION_HOST, MemoryLocation::LOCATION_DEVICE}) {
    size_t block_num = 20;
    size_t block_size = 1024 * 1024;
    block_allocator_ =
        std::make_shared<BlockAllocator>(location, block_num, block_size, 0, memory_allocator_, context_);
    block_allocator_->PreAllocateBlocks();

    // Check block number.
    EXPECT_EQ(block_allocator_->GetFreeBlockNumber(), 20);
    EXPECT_EQ(block_allocator_->GetUsedBlockNumber(), 0);

    // Allocate
    std::vector<int> blocks;
    Status status = block_allocator_->AllocateBlocks(3, blocks);
    EXPECT_TRUE(status.OK());

    EXPECT_EQ(blocks.size(), 3);
    EXPECT_EQ(block_allocator_->GetFreeBlockNumber(), 17);
    EXPECT_EQ(block_allocator_->GetUsedBlockNumber(), 3);

    // Free a block
    block_allocator_->FreeBlocks({*blocks.begin()});
    blocks.erase(blocks.begin());

    EXPECT_EQ(block_allocator_->GetFreeBlockNumber(), 18);
    EXPECT_EQ(block_allocator_->GetUsedBlockNumber(), 2);

    // Get ptrs
    std::vector<void*> addrs;
    status = block_allocator_->GetBlockPtrs(blocks, addrs);
    EXPECT_TRUE(status.OK());

    EXPECT_EQ(addrs.size(), 2);
    EXPECT_TRUE(addrs[0] != nullptr);
    EXPECT_TRUE(addrs[1] != nullptr);

    // Get base ptr
    void* base_ptr = block_allocator_->GetBlocksBasePtr();
    if (location == MemoryLocation::LOCATION_HOST) {
      EXPECT_EQ(base_ptr, nullptr);
    }

    // Get base id
    int base_id = block_allocator_->GetBlocksBaseId();
    EXPECT_EQ(base_id, 0);

    // Clear
    block_allocator_->Clear();
    EXPECT_EQ(block_allocator_->GetFreeBlockNumber(), 0);
    EXPECT_EQ(block_allocator_->GetUsedBlockNumber(), 0);
  }
}

TEST_F(BlockAllocatorTest, TestBlockAllocatorManager) {
  BlockAllocatorManagerConfig block_allocator_manager_config;

  BlockAllocatorGroupConfig group_1_config;
  group_1_config.devices = {0};
  group_1_config.device_block_num = 10;
  group_1_config.host_block_num = 20;
  group_1_config.block_size = 1 * 1024 * 1024;

  BlockAllocatorGroupConfig group_2_config;
  group_2_config.devices = {0};
  group_2_config.device_block_num = 30;
  group_2_config.host_block_num = 60;
  group_2_config.block_size = 2 * 1024 * 1024;

  block_allocator_manager_config[1] = group_1_config;
  block_allocator_manager_config[2] = group_2_config;

  BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_);

  // Get host allocator
  std::shared_ptr<BlockAllocatorInterface> host_allocator_1 =
      block_allocator_manager.GetBlockAllocatorGroup(1)->GetHostBlockAllocator();
  EXPECT_TRUE(host_allocator_1 != nullptr);
  EXPECT_EQ(host_allocator_1->GetFreeBlockNumber(), 20);

  std::shared_ptr<BlockAllocatorInterface> host_allocator_2 =
      block_allocator_manager.GetBlockAllocatorGroup(2)->GetHostBlockAllocator();
  EXPECT_TRUE(host_allocator_2 != nullptr);
  EXPECT_EQ(host_allocator_2->GetFreeBlockNumber(), 60);

  EXPECT_TRUE(block_allocator_manager.GetBlockAllocatorGroup(3) == nullptr);

  std::shared_ptr<BlockAllocatorInterface> dev_allocator_1 =
      block_allocator_manager.GetBlockAllocatorGroup(1)->GetDeviceBlockAllocator(0);
  EXPECT_TRUE(dev_allocator_1 != nullptr);
  EXPECT_TRUE(block_allocator_manager.GetBlockAllocatorGroup(1)->GetDeviceBlockAllocator(1) == nullptr);

  EXPECT_EQ(dev_allocator_1->GetFreeBlockNumber(), 10);
}
