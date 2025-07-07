/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/pinned_mem_buffer_pool.h"
#include <chrono>
#include <thread>
#include <vector>
#include "ksana_llm/utils/device_utils.h"
#include "tests/test.h"

namespace ksana_llm {

class PinnedMemBufferPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize device context
    int device_count = 0;
    GetDeviceCount(&device_count);
    if (device_count <= 0) {
      GTEST_SKIP() << "No CUDA device available for testing";
    }
    device_id_ = 0;
    device_count_ = device_count;
    try {
      // Try to allocate a pool to ensure CUDA and pinned memory are really available
      PinnedMemoryBufferPool pool(device_count_, 1, 1024);
    } catch (const std::exception& e) {
      GTEST_SKIP() << "PinnedMemoryBufferPool not available: " << e.what();
    }
    SetDevice(device_id_);
  }

  void TearDown() override {
    // Clean up if needed
  }

  int device_count_ = 1;  // Test with single device by default
  int device_id_ = 0;
  static constexpr size_t kTestBlockSize = 1024 * 1024;  // 1MB
  static constexpr size_t kTestBlockNum = 4;
};

// Test basic pool creation and destruction
TEST_F(PinnedMemBufferPoolTest, BasicCreationDestruction) {
  EXPECT_NO_THROW({ PinnedMemoryBufferPool pool(device_count_, kTestBlockNum, kTestBlockSize); });
}

// Test getting and putting blocks
TEST_F(PinnedMemBufferPoolTest, GetPutBlock) {
  PinnedMemoryBufferPool pool(device_count_, kTestBlockNum, kTestBlockSize);

  // Get a block from device 0
  PinnedMemoryBufferBlock* block = pool.get_block(device_id_);
  ASSERT_NE(block, nullptr);
  EXPECT_NE(block->host_ptr, nullptr);
  EXPECT_NE(block->device_ptr, nullptr);
  EXPECT_EQ(block->capacity, kTestBlockSize);
  EXPECT_EQ(block->device_id, device_id_);

  // Put the block back
  EXPECT_NO_THROW(pool.put_block(block));
}

// Test multiple blocks allocation
TEST_F(PinnedMemBufferPoolTest, MultipleBlocks) {
  PinnedMemoryBufferPool pool(device_count_, kTestBlockNum, kTestBlockSize);

  std::vector<PinnedMemoryBufferBlock*> blocks;

  // Get up to kTestBlockNum blocks from device 0
  for (size_t i = 0; i < kTestBlockNum + 2; ++i) {  // Try to get more than pre-allocated
    PinnedMemoryBufferBlock* block = pool.get_block(device_id_);
    if (!block) break;  // pool exhausted
    EXPECT_NE(block->host_ptr, nullptr);
    EXPECT_NE(block->device_ptr, nullptr);
    EXPECT_EQ(block->capacity, kTestBlockSize);
    EXPECT_EQ(block->device_id, device_id_);
    blocks.push_back(block);
  }

  // Should not get more than kTestBlockNum blocks
  EXPECT_LE(blocks.size(), kTestBlockNum);

  // Verify all blocks are different
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (size_t j = i + 1; j < blocks.size(); ++j) {
      EXPECT_NE(blocks[i]->host_ptr, blocks[j]->host_ptr);
      EXPECT_NE(blocks[i]->device_ptr, blocks[j]->device_ptr);
    }
  }

  // Put all blocks back
  for (auto* block : blocks) {
    EXPECT_NO_THROW(pool.put_block(block));
  }
}

// Test memory content consistency
TEST_F(PinnedMemBufferPoolTest, MemoryContentConsistency) {
  PinnedMemoryBufferPool pool(device_count_, kTestBlockNum, kTestBlockSize);

  PinnedMemoryBufferBlock* block = pool.get_block(device_id_);
  ASSERT_NE(block, nullptr);

  // Write some test data to host memory
  uint8_t* host_data = static_cast<uint8_t*>(block->host_ptr);
  for (size_t i = 0; i < std::min(kTestBlockSize, size_t(256)); ++i) {
    host_data[i] = static_cast<uint8_t>(i % 256);
  }

  // Verify the data can be read back
  for (size_t i = 0; i < std::min(kTestBlockSize, size_t(256)); ++i) {
    EXPECT_EQ(host_data[i], static_cast<uint8_t>(i % 256));
  }

  pool.put_block(block);
}

// Test thread safety
TEST_F(PinnedMemBufferPoolTest, ThreadSafety) {
  PinnedMemoryBufferPool pool(device_count_, kTestBlockNum, kTestBlockSize);
  const int num_threads = 4;
  const int operations_per_thread = 10;

  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&pool, &success_count, operations_per_thread, this]() {
      for (int i = 0; i < operations_per_thread; ++i) {
        try {
          // Get a block from device 0
          PinnedMemoryBufferBlock* block = pool.get_block(device_id_);
          if (block && block->host_ptr && block->device_ptr) {
            // Do some work with the block
            std::this_thread::sleep_for(std::chrono::microseconds(10));

            // Put it back
            pool.put_block(block);
            success_count++;
          }
        } catch (const std::exception& e) {
          // Log error but continue
          std::cerr << "Thread operation failed: " << e.what() << std::endl;
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // We should have some successful operations
  EXPECT_GT(success_count.load(), 0);
  std::cout << "Successful thread operations: " << success_count.load() << " out of "
            << (num_threads * operations_per_thread) << std::endl;
}

// Test error conditions
TEST_F(PinnedMemBufferPoolTest, ErrorConditions) {
  // Test with zero block size
  EXPECT_THROW(
      {
        PinnedMemoryBufferPool pool(device_count_, 1, 0);
        pool.get_block(device_id_);
      },
      std::runtime_error);

  // Test with invalid device_id
  PinnedMemoryBufferPool pool(device_count_, 1, kTestBlockSize);
  EXPECT_THROW(
      {
        pool.get_block(999);  // Invalid device ID
      },
      std::invalid_argument);

  EXPECT_THROW(
      {
        pool.get_block(-1);  // Invalid device ID
      },
      std::invalid_argument);
}

// Test reuse of blocks
TEST_F(PinnedMemBufferPoolTest, BlockReuse) {
  PinnedMemoryBufferPool pool(device_count_, 2, kTestBlockSize);  // Small pool

  // Get all pre-allocated blocks
  PinnedMemoryBufferBlock* block1 = pool.get_block(device_id_);
  PinnedMemoryBufferBlock* block2 = pool.get_block(device_id_);

  ASSERT_NE(block1, nullptr);
  ASSERT_NE(block2, nullptr);
  EXPECT_NE(block1->host_ptr, block2->host_ptr);

  // Put them back
  pool.put_block(block1);
  pool.put_block(block2);

  // Get blocks again - should reuse the same ones
  PinnedMemoryBufferBlock* reused_block1 = pool.get_block(device_id_);
  PinnedMemoryBufferBlock* reused_block2 = pool.get_block(device_id_);

  EXPECT_TRUE(reused_block1 == block1 || reused_block1 == block2);
  EXPECT_TRUE(reused_block2 == block1 || reused_block2 == block2);
  EXPECT_NE(reused_block1, reused_block2);

  pool.put_block(reused_block1);
  pool.put_block(reused_block2);
}

// Performance test
TEST_F(PinnedMemBufferPoolTest, Performance) {
  PinnedMemoryBufferPool pool(device_count_, kTestBlockNum, kTestBlockSize);
  const int num_operations = 1000;

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_operations; ++i) {
    PinnedMemoryBufferBlock* block = pool.get_block(device_id_);
    ASSERT_NE(block, nullptr);

    // Simulate some work
    volatile uint8_t* data = static_cast<uint8_t*>(block->host_ptr);
    data[0] = static_cast<uint8_t>(i % 256);

    pool.put_block(block);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  std::cout << "Performance test: " << num_operations << " operations took " << duration.count() << " microseconds"
            << std::endl;
  std::cout << "Average time per operation: " << (static_cast<double>(duration.count()) / num_operations)
            << " microseconds" << std::endl;

  // Should complete within reasonable time (adjust threshold as needed)
  EXPECT_LT(duration.count(), 100000);  // Less than 100ms for 1000 operations
}

// Test multi-device functionality
TEST_F(PinnedMemBufferPoolTest, MultiDevice) {
  int test_device_count = 2;  // Test with 2 devices
  PinnedMemoryBufferPool pool(test_device_count, kTestBlockNum, kTestBlockSize);

  // Get blocks from different devices
  PinnedMemoryBufferBlock* block_dev0 = pool.get_block(0);
  PinnedMemoryBufferBlock* block_dev1 = pool.get_block(1);

  ASSERT_NE(block_dev0, nullptr);
  ASSERT_NE(block_dev1, nullptr);

  EXPECT_EQ(block_dev0->device_id, 0);
  EXPECT_EQ(block_dev1->device_id, 1);

  EXPECT_NE(block_dev0->host_ptr, nullptr);
  EXPECT_NE(block_dev0->device_ptr, nullptr);
  EXPECT_NE(block_dev1->host_ptr, nullptr);
  EXPECT_NE(block_dev1->device_ptr, nullptr);

  // Put blocks back to their respective devices
  pool.put_block(block_dev0);
  pool.put_block(block_dev1);

  // Get blocks again - should work fine
  PinnedMemoryBufferBlock* block_dev0_again = pool.get_block(0);
  PinnedMemoryBufferBlock* block_dev1_again = pool.get_block(1);

  EXPECT_EQ(block_dev0_again->device_id, 0);
  EXPECT_EQ(block_dev1_again->device_id, 1);

  pool.put_block(block_dev0_again);
  pool.put_block(block_dev1_again);
}

// Test error handling for put_block
TEST_F(PinnedMemBufferPoolTest, PutBlockErrorHandling) {
  PinnedMemoryBufferPool pool(device_count_, kTestBlockNum, kTestBlockSize);

  // Test putting null block
  EXPECT_NO_THROW(pool.put_block(nullptr));  // Should not crash, just log error

  // Test putting block with invalid device_id
  auto invalid_block = std::make_unique<PinnedMemoryBufferBlock>();
  invalid_block->device_id = -1;                             // Invalid device_id
  EXPECT_NO_THROW(pool.put_block(invalid_block.release()));  // Should not crash

  // Test putting block with device_id >= device_count
  auto invalid_block2 = std::make_unique<PinnedMemoryBufferBlock>();
  invalid_block2->device_id = 999;                            // Invalid device_id
  EXPECT_NO_THROW(pool.put_block(invalid_block2.release()));  // Should not crash
}

// Test pool exhaustion and dynamic allocation
TEST_F(PinnedMemBufferPoolTest, PoolExhaustionAndDynamicAllocation) {
  // Create a small pool
  PinnedMemoryBufferPool pool(device_count_, 2, kTestBlockSize);  // Only 2 pre-allocated blocks

  std::vector<PinnedMemoryBufferBlock*> blocks;

  // Try to get more blocks than pre-allocated (should return nullptr after pool is exhausted)
  for (int i = 0; i < 5; ++i) {
    PinnedMemoryBufferBlock* block = pool.get_block(device_id_);
    if (!block) break;
    EXPECT_EQ(block->device_id, device_id_);
    EXPECT_EQ(block->capacity, kTestBlockSize);
    blocks.push_back(block);
  }

  // Should not get more than 2 blocks
  EXPECT_LE(blocks.size(), 2u);

  // Verify all blocks are different
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (size_t j = i + 1; j < blocks.size(); ++j) {
      EXPECT_NE(blocks[i], blocks[j]);
    }
  }

  // Put all blocks back
  for (auto* block : blocks) {
    pool.put_block(block);
  }
}

// Test that all pool operations throw in non-CUDA environment
TEST(PinnedMemBufferPoolStandaloneTest, ThrowsOnNoCuda) {
  int device_count = 0;
  ksana_llm::GetDeviceCount(&device_count);
  if (device_count > 0) {
    GTEST_SKIP() << "CUDA device present, skipping non-CUDA error test";
  }
  using namespace ksana_llm;
  constexpr size_t kBlockNum = 2;
  constexpr size_t kBlockSize = 1024;
  // All operations should throw
  EXPECT_THROW({ PinnedMemoryBufferPool pool(1, kBlockNum, kBlockSize); }, std::runtime_error);
  // Even with 0 device, should throw
  EXPECT_THROW({ PinnedMemoryBufferPool pool(0, kBlockNum, kBlockSize); }, std::runtime_error);
  // Try get_block/put_block on a pool constructed with 0 device (should not be possible, but for coverage)
  try {
    PinnedMemoryBufferPool pool(0, kBlockNum, kBlockSize);
    EXPECT_THROW(pool.get_block(0), std::runtime_error);
    EXPECT_THROW(pool.put_block(nullptr), std::runtime_error);
  } catch (...) {
    // Already thrown at construction, OK
  }
}

}  // namespace ksana_llm
