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
    EXPECT_TRUE(base_ptr != nullptr);

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

TEST_F(BlockAllocatorTest, TestBlockAllocatorMultiDeviceWithDP) {
  // Test with device_num=2 and dp=2
  int test_device_num = 2;
  uint32_t test_attn_dp_worker_num = 2;
  std::shared_ptr<Context> multi_context =
      std::make_shared<Context>(test_device_num, test_attn_dp_worker_num, multi_batch_num_);

  size_t block_num = 30;
  size_t block_size = 2 * 1024 * 1024;

  // Test Device 0 allocator
  std::shared_ptr<BlockAllocatorInterface> device_0_allocator = std::make_shared<BlockAllocator>(
      MemoryLocation::LOCATION_DEVICE, block_num, block_size, 0, memory_allocator_, multi_context);
  device_0_allocator->PreAllocateBlocks();

  EXPECT_EQ(device_0_allocator->GetFreeBlockNumber(), block_num);
  EXPECT_EQ(device_0_allocator->GetUsedBlockNumber(), 0);

  // Allocate blocks on device 0
  std::vector<int> device_0_blocks;
  Status status = device_0_allocator->AllocateBlocks(5, device_0_blocks);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(device_0_blocks.size(), 5);
  EXPECT_EQ(device_0_allocator->GetFreeBlockNumber(), 25);
  EXPECT_EQ(device_0_allocator->GetUsedBlockNumber(), 5);

  // Get block pointers on device 0
  std::vector<void*> device_0_addrs;
  status = device_0_allocator->GetBlockPtrs(device_0_blocks, device_0_addrs);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(device_0_addrs.size(), 5);
  for (void* addr : device_0_addrs) {
    EXPECT_TRUE(addr != nullptr);
  }

  // Test Device 1 allocator
  std::shared_ptr<BlockAllocatorInterface> device_1_allocator = std::make_shared<BlockAllocator>(
      MemoryLocation::LOCATION_DEVICE, block_num, block_size, 1, memory_allocator_, multi_context);
  device_1_allocator->PreAllocateBlocks();

  EXPECT_EQ(device_1_allocator->GetFreeBlockNumber(), block_num);
  EXPECT_EQ(device_1_allocator->GetUsedBlockNumber(), 0);

  // Allocate blocks on device 1
  std::vector<int> device_1_blocks;
  status = device_1_allocator->AllocateBlocks(8, device_1_blocks);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(device_1_blocks.size(), 8);
  EXPECT_EQ(device_1_allocator->GetFreeBlockNumber(), 22);
  EXPECT_EQ(device_1_allocator->GetUsedBlockNumber(), 8);

  // Get block pointers on device 1
  std::vector<void*> device_1_addrs;
  status = device_1_allocator->GetBlockPtrs(device_1_blocks, device_1_addrs);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(device_1_addrs.size(), 8);
  for (void* addr : device_1_addrs) {
    EXPECT_TRUE(addr != nullptr);
  }

  // Test Host allocator with larger block number to support both devices
  size_t host_block_num = 60;
  std::shared_ptr<BlockAllocatorInterface> host_allocator = std::make_shared<BlockAllocator>(
      MemoryLocation::LOCATION_HOST, host_block_num, block_size, 0, memory_allocator_, multi_context);
  host_allocator->PreAllocateBlocks();

  EXPECT_EQ(host_allocator->GetFreeBlockNumber(), host_block_num);
  EXPECT_EQ(host_allocator->GetUsedBlockNumber(), 0);

  // Allocate blocks on host for device 0 swap
  std::vector<int> host_blocks_for_dev0;
  status = host_allocator->AllocateBlocks(10, host_blocks_for_dev0);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(host_blocks_for_dev0.size(), 10);
  EXPECT_EQ(host_allocator->GetFreeBlockNumber(), 50);
  EXPECT_EQ(host_allocator->GetUsedBlockNumber(), 10);

  // Allocate blocks on host for device 1 swap
  std::vector<int> host_blocks_for_dev1;
  status = host_allocator->AllocateBlocks(15, host_blocks_for_dev1);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(host_blocks_for_dev1.size(), 15);
  EXPECT_EQ(host_allocator->GetFreeBlockNumber(), 35);
  EXPECT_EQ(host_allocator->GetUsedBlockNumber(), 25);

  // Get host block pointers
  std::vector<void*> host_addrs_for_dev0;
  status = host_allocator->GetBlockPtrs(host_blocks_for_dev0, host_addrs_for_dev0);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(host_addrs_for_dev0.size(), 10);
  for (void* addr : host_addrs_for_dev0) {
    EXPECT_TRUE(addr != nullptr);
  }

  std::vector<void*> host_addrs_for_dev1;
  status = host_allocator->GetBlockPtrs(host_blocks_for_dev1, host_addrs_for_dev1);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(host_addrs_for_dev1.size(), 15);
  for (void* addr : host_addrs_for_dev1) {
    EXPECT_TRUE(addr != nullptr);
  }

  // Verify host allocator has continuous memory base pointer
  void* host_base_ptr = host_allocator->GetBlocksBasePtr();
  EXPECT_TRUE(host_base_ptr != nullptr);

  // Free some blocks and verify
  device_0_allocator->FreeBlocks({device_0_blocks[0], device_0_blocks[1]});
  EXPECT_EQ(device_0_allocator->GetFreeBlockNumber(), 27);
  EXPECT_EQ(device_0_allocator->GetUsedBlockNumber(), 3);

  device_1_allocator->FreeBlocks({device_1_blocks[0]});
  EXPECT_EQ(device_1_allocator->GetFreeBlockNumber(), 23);
  EXPECT_EQ(device_1_allocator->GetUsedBlockNumber(), 7);

  host_allocator->FreeBlocks({host_blocks_for_dev0[0], host_blocks_for_dev0[1], host_blocks_for_dev1[0]});
  EXPECT_EQ(host_allocator->GetFreeBlockNumber(), 38);
  EXPECT_EQ(host_allocator->GetUsedBlockNumber(), 22);

  // Clean up
  device_0_allocator->Clear();
  device_1_allocator->Clear();
  host_allocator->Clear();

  EXPECT_EQ(device_0_allocator->GetFreeBlockNumber(), 0);
  EXPECT_EQ(device_1_allocator->GetFreeBlockNumber(), 0);
  EXPECT_EQ(host_allocator->GetFreeBlockNumber(), 0);
}

TEST_F(BlockAllocatorTest, TestBlockAllocatorManagerMultiDevice) {
  // Test BlockAllocatorManager with multiple devices
  BlockAllocatorManagerConfig block_allocator_manager_config;

  BlockAllocatorGroupConfig group_config;
  group_config.devices = {0, 1};
  group_config.device_block_num = 20;
  group_config.host_block_num = 50;
  group_config.block_size = 2 * 1024 * 1024;

  block_allocator_manager_config[1] = group_config;

  int test_device_num = 2;
  uint32_t test_attn_dp_worker_num = 2;
  std::shared_ptr<Context> multi_context =
      std::make_shared<Context>(test_device_num, test_attn_dp_worker_num, multi_batch_num_);

  BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, multi_context);

  // Get and verify device allocators
  std::shared_ptr<BlockAllocatorInterface> dev_allocator_0 =
      block_allocator_manager.GetBlockAllocatorGroup(1)->GetDeviceBlockAllocator(0);
  EXPECT_TRUE(dev_allocator_0 != nullptr);
  EXPECT_EQ(dev_allocator_0->GetFreeBlockNumber(), 20);

  std::shared_ptr<BlockAllocatorInterface> dev_allocator_1 =
      block_allocator_manager.GetBlockAllocatorGroup(1)->GetDeviceBlockAllocator(1);
  EXPECT_TRUE(dev_allocator_1 != nullptr);
  EXPECT_EQ(dev_allocator_1->GetFreeBlockNumber(), 20);

  // Verify no device allocator for index 2
  EXPECT_TRUE(block_allocator_manager.GetBlockAllocatorGroup(1)->GetDeviceBlockAllocator(2) == nullptr);

  // Get and verify host allocator
  std::shared_ptr<BlockAllocatorInterface> host_allocator =
      block_allocator_manager.GetBlockAllocatorGroup(1)->GetHostBlockAllocator();
  EXPECT_TRUE(host_allocator != nullptr);
  EXPECT_EQ(host_allocator->GetFreeBlockNumber(), 50);

  // Test allocation on device 0
  std::vector<int> dev_0_blocks;
  Status status = dev_allocator_0->AllocateBlocks(5, dev_0_blocks);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(dev_allocator_0->GetUsedBlockNumber(), 5);

  // Test allocation on device 1
  std::vector<int> dev_1_blocks;
  status = dev_allocator_1->AllocateBlocks(7, dev_1_blocks);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(dev_allocator_1->GetUsedBlockNumber(), 7);

  // Test host allocation
  std::vector<int> host_blocks;
  status = host_allocator->AllocateBlocks(10, host_blocks);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(host_allocator->GetUsedBlockNumber(), 10);
  EXPECT_EQ(host_allocator->GetFreeBlockNumber(), 40);
}
