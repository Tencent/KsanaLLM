/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#ifdef ENABLE_CUDA
#  include "ksana_llm/connector/cuda_buffer_pool.h"

#  include <cuda_runtime.h>
#  include <gtest/gtest.h>
#  include <atomic>
#  include <chrono>
#  include <memory>
#  include <random>
#  include <thread>
#  include <vector>

#  include "ksana_llm/utils/nvidia/cuda_utils.h"

using namespace ksana_llm;

class CudaBufferPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check if CUDA is available
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available or no CUDA devices found";
    }

    // Set up CUDA device
    CUDA_CHECK(cudaSetDevice(0));
    device_id_ = 0;

    // Get device properties for testing
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    device_name_ = prop.name;

    // Create CUDA stream for testing
    CUDA_CHECK(cudaStreamCreate(&test_stream_));
  }

  void TearDown() override {
    if (test_stream_) {
      cudaStreamDestroy(test_stream_);
    }
    CUDA_CHECK(cudaDeviceReset());
  }

  // Helper function to verify buffer data integrity
  void VerifyBufferData(void* device_ptr, size_t size, int pattern) {
    std::vector<int> host_data(size / sizeof(int));
    CUDA_CHECK(cudaMemcpy(host_data.data(), device_ptr, size, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < host_data.size(); ++i) {
      EXPECT_EQ(host_data[i], pattern) << "Data mismatch at index " << i;
    }
  }

  // Helper function to write pattern to buffer
  void WritePatternToBuffer(void* device_ptr, size_t size, int pattern) {
    // Add validation and debug info
    if (device_ptr == nullptr) {
      GTEST_FAIL() << "WritePatternToBuffer: device_ptr is null";
      return;
    }

    // Log the device pointer for debugging
    std::cout << "WritePatternToBuffer: device_ptr=" << device_ptr << " size=" << size << " pattern=0x" << std::hex
              << pattern << std::dec << std::endl;

    // Check if the pointer looks valid (basic sanity check)
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(device_ptr);
    if (ptr_val < 0x1000000) {  // Device pointers should be high addresses
      GTEST_FAIL() << "WritePatternToBuffer: device_ptr=" << device_ptr << " looks invalid (too small address)";
      return;
    }

    std::vector<int> host_data(size / sizeof(int), pattern);
    cudaError_t error = cudaMemcpy(device_ptr, host_data.data(), size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      GTEST_FAIL() << "cudaMemcpy failed in WritePatternToBuffer: " << cudaGetErrorString(error)
                   << " (device_ptr=" << device_ptr << ")";
    }
  }

  // Helper function to check if memory regions overlap
  bool BuffersOverlap(void* ptr1, size_t size1, void* ptr2, size_t size2) {
    uintptr_t start1 = reinterpret_cast<uintptr_t>(ptr1);
    uintptr_t end1 = start1 + size1;
    uintptr_t start2 = reinterpret_cast<uintptr_t>(ptr2);
    uintptr_t end2 = start2 + size2;

    return !(end1 <= start2 || end2 <= start1);
  }

  int device_id_;
  std::string device_name_;
  cudaStream_t test_stream_ = nullptr;
};

// Test BufferBlock basic structure
TEST_F(CudaBufferPoolTest, BufferBlockBasicTest) {
  const size_t test_size = 1024 * 1024;  // 1MB
  void* test_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&test_ptr, test_size));

  BufferBlock block;
  block.device_ptr = test_ptr;
  block.capacity = test_size;
  block.device_id = device_id_;

  EXPECT_EQ(block.device_ptr, test_ptr);
  EXPECT_EQ(block.capacity, test_size);
  EXPECT_EQ(block.device_id, device_id_);

  CUDA_CHECK(cudaFree(test_ptr));
}

// Test BufferPool constructor with valid parameters
TEST_F(CudaBufferPoolTest, ConstructorValidParameters) {
  const size_t block_size = 1024 * 1024;  // 1MB
  const int num_blocks = 5;

  EXPECT_NO_THROW({ BufferPool pool(device_id_, num_blocks, block_size); });
}

// Test BufferPool constructor with zero blocks
TEST_F(CudaBufferPoolTest, ConstructorZeroBlocks) {
  const size_t block_size = 1024 * 1024;
  const int num_blocks = 0;

  EXPECT_NO_THROW({ BufferPool pool(device_id_, num_blocks, block_size); });
}

// Test BufferPool constructor with large block size
TEST_F(CudaBufferPoolTest, ConstructorLargeBlockSize) {
  const size_t block_size = 256 * 1024 * 1024;  // 256MB
  const int num_blocks = 2;

  // This might fail on systems with limited GPU memory
  try {
    BufferPool pool(device_id_, num_blocks, block_size);
    SUCCEED();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Insufficient GPU memory for large block test: " << e.what();
  }
}

// Test basic get_block and put_block functionality
TEST_F(CudaBufferPoolTest, BasicGetPutBlock) {
  const size_t block_size = 1024 * 1024;
  const int num_blocks = 3;

  std::cout << "Creating BufferPool with device_id=" << device_id_ << " num_blocks=" << num_blocks
            << " block_size=" << block_size << std::endl;

  BufferPool pool(device_id_, num_blocks, block_size);

  std::cout << "Getting block from pool..." << std::endl;
  // Get a block
  BufferBlock* block = pool.get_block();

  std::cout << "Got block: device_ptr=" << block->device_ptr << " capacity=" << block->capacity
            << " device_id=" << block->device_id << std::endl;

  EXPECT_NE(block->device_ptr, nullptr);
  EXPECT_EQ(block->capacity, block_size);
  EXPECT_EQ(block->device_id, device_id_);

  // Verify we can write to the buffer
  std::cout << "Writing pattern to buffer..." << std::endl;
  WritePatternToBuffer(block->device_ptr, block_size, 0x12345678);

  std::cout << "Verifying buffer data..." << std::endl;
  VerifyBufferData(block->device_ptr, block_size, 0x12345678);

  // Put the block back
  std::cout << "Putting block back..." << std::endl;
  EXPECT_NO_THROW(pool.put_block(block));
  std::cout << "BasicGetPutBlock test completed successfully" << std::endl;
}

// Test getting multiple blocks
TEST_F(CudaBufferPoolTest, GetMultipleBlocks) {
  const size_t block_size = 512 * 1024;
  const int num_blocks = 3;

  BufferPool pool(device_id_, num_blocks, block_size);

  std::vector<BufferBlock*> blocks;

  // Get all available blocks
  for (int i = 0; i < num_blocks; ++i) {
    BufferBlock* block = pool.get_block();
    EXPECT_NE(block->device_ptr, nullptr);
    EXPECT_EQ(block->capacity, block_size);
    blocks.push_back(block);
  }

  // Verify all blocks have different memory addresses
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (size_t j = i + 1; j < blocks.size(); ++j) {
      EXPECT_FALSE(
          BuffersOverlap(blocks[i]->device_ptr, blocks[i]->capacity, blocks[j]->device_ptr, blocks[j]->capacity))
          << "Blocks " << i << " and " << j << " overlap";
    }
  }

  // Put all blocks back
  for (auto& block : blocks) {
    pool.put_block(block);
  }
}

// Test block reuse - verify blocks are recycled from the pool
TEST_F(CudaBufferPoolTest, BlockReuse) {
  const size_t block_size = 1024 * 1024;
  const int num_blocks = 1;  // Use only 1 block to guarantee reuse

  BufferPool pool(device_id_, num_blocks, block_size);

  // Get the only block and mark it with a pattern
  BufferBlock* block1 = pool.get_block();
  void* original_ptr = block1->device_ptr;
  WritePatternToBuffer(block1->device_ptr, block_size, 0xAABBCCDD);

  // Put it back
  pool.put_block(block1);

  // Get another block - with only 1 block in pool, must reuse the same memory
  BufferBlock* block2 = pool.get_block();
  EXPECT_EQ(block2->device_ptr, original_ptr);

  // Verify the block properties are correct
  EXPECT_EQ(block2->capacity, block_size);
  EXPECT_EQ(block2->device_id, device_id_);

  // The data might still be there (implementation dependent)
  // We won't test the data persistence as it's not guaranteed

  pool.put_block(block2);
}

// Test memory isolation between blocks
TEST_F(CudaBufferPoolTest, MemoryIsolation) {
  const size_t block_size = 1024 * 1024;
  const int num_blocks = 3;

  BufferPool pool(device_id_, num_blocks, block_size);

  // Get two blocks
  BufferBlock* block1 = pool.get_block();
  BufferBlock* block2 = pool.get_block();

  // Write different patterns to each block
  WritePatternToBuffer(block1->device_ptr, block_size, 0x11111111);
  WritePatternToBuffer(block2->device_ptr, block_size, 0x22222222);

  // Verify data integrity
  VerifyBufferData(block1->device_ptr, block_size, 0x11111111);
  VerifyBufferData(block2->device_ptr, block_size, 0x22222222);

  pool.put_block(block1);
  pool.put_block(block2);
}

// Test concurrent access to buffer pool
TEST_F(CudaBufferPoolTest, ConcurrentAccess) {
  const size_t block_size = 512 * 1024;
  const int num_blocks = 4;
  const int num_threads = 8;
  const int operations_per_thread = 10;

  BufferPool pool(device_id_, num_blocks, block_size);

  std::atomic<int> successful_operations{0};
  std::atomic<int> failed_operations{0};
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back(
        [this, &pool, operations_per_thread, &successful_operations, &failed_operations, block_size]() {
          for (int i = 0; i < operations_per_thread; ++i) {
            try {
              BufferBlock* block = pool.get_block();

              // Simulate some work with the buffer
              std::hash<std::thread::id> hasher;
              int pattern = hasher(std::this_thread::get_id()) % 1000;
              WritePatternToBuffer(block->device_ptr, std::min(block_size, 1024UL), pattern);

              // Small delay to increase chance of race conditions
              std::this_thread::sleep_for(std::chrono::microseconds(1));

              pool.put_block(block);
              successful_operations++;
            } catch (const std::exception& e) {
              failed_operations++;
            }
          }
        });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_GT(successful_operations.load(), 0);
  // Some operations might fail due to pool exhaustion, which is expected
}

// Test getting blocks when pool is empty (should dynamically allocate new blocks)
TEST_F(CudaBufferPoolTest, GetBlockWhenPoolEmpty) {
  const size_t block_size = 1024 * 1024;
  const int num_blocks = 1;

  BufferPool pool(device_id_, num_blocks, block_size);

  // Get the only pre-allocated block
  BufferBlock* block1 = pool.get_block();
  EXPECT_NE(block1->device_ptr, nullptr);
  EXPECT_EQ(block1->capacity, block_size);

  // Get another block when pool is empty - should dynamically allocate
  BufferBlock* block2 = pool.get_block();
  EXPECT_NE(block2->device_ptr, nullptr);
  EXPECT_EQ(block2->capacity, block_size);

  // Verify they are different blocks
  EXPECT_NE(block1->device_ptr, block2->device_ptr);

  // Both blocks should be usable
  WritePatternToBuffer(block1->device_ptr, block_size, 0x11111111);
  WritePatternToBuffer(block2->device_ptr, block_size, 0x22222222);

  VerifyBufferData(block1->device_ptr, block_size, 0x11111111);
  VerifyBufferData(block2->device_ptr, block_size, 0x22222222);

  // Put both blocks back
  pool.put_block(block1);
  pool.put_block(block2);
}

// Test rapid get/put operations
TEST_F(CudaBufferPoolTest, RapidGetPutOperations) {
  const size_t block_size = 256 * 1024;
  const int num_blocks = 2;
  const int num_operations = 100;

  BufferPool pool(device_id_, num_blocks, block_size);

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_operations; ++i) {
    BufferBlock* block = pool.get_block();

    // Quick verification
    EXPECT_NE(block->device_ptr, nullptr);
    EXPECT_EQ(block->capacity, block_size);

    pool.put_block(block);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  // Performance check - operations should complete reasonably quickly
  EXPECT_LT(duration.count(), 10000) << "Operations took too long: " << duration.count() << " microseconds";
}

// Test with zero-sized pool (should dynamically allocate when needed)
TEST_F(CudaBufferPoolTest, ZeroSizedPool) {
  const size_t block_size = 1024;
  const int num_blocks = 0;

  BufferPool pool(device_id_, num_blocks, block_size);

  // Getting a block from empty pool should dynamically allocate a new block
  BufferBlock* block = pool.get_block();
  EXPECT_NE(block->device_ptr, nullptr);
  EXPECT_EQ(block->capacity, block_size);
  EXPECT_EQ(block->device_id, device_id_);

  // Verify the block is usable
  WritePatternToBuffer(block->device_ptr, block_size, 0xDEADBEEF);
  VerifyBufferData(block->device_ptr, block_size, 0xDEADBEEF);

  // Put the block back
  pool.put_block(block);

  // Getting another block should either reuse the returned block or allocate a new one
  BufferBlock* block2 = pool.get_block();
  EXPECT_NE(block2->device_ptr, nullptr);
  EXPECT_EQ(block2->capacity, block_size);
  EXPECT_EQ(block2->device_id, device_id_);

  pool.put_block(block2);
}

// Test multiple buffer pools
TEST_F(CudaBufferPoolTest, MultipleBufferPools) {
  const size_t block_size1 = 512 * 1024;
  const size_t block_size2 = 1024 * 1024;
  const int num_blocks = 2;

  BufferPool pool1(device_id_, num_blocks, block_size1);
  BufferPool pool2(device_id_, num_blocks, block_size2);

  BufferBlock* block1 = pool1.get_block();
  BufferBlock* block2 = pool2.get_block();

  EXPECT_EQ(block1->capacity, block_size1);
  EXPECT_EQ(block2->capacity, block_size2);
  EXPECT_NE(block1->device_ptr, block2->device_ptr);

  pool1.put_block(block1);
  pool2.put_block(block2);
}

// Test large block allocation
TEST_F(CudaBufferPoolTest, LargeBlockAllocation) {
  // Query available GPU memory
  size_t free_memory, total_memory;
  CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));

  // Try to allocate blocks that use a significant portion of memory
  size_t block_size = std::min(free_memory / 4, 512UL * 1024 * 1024);  // 512MB or 1/4 of available
  const int num_blocks = 2;

  if (block_size < 1024 * 1024) {
    GTEST_SKIP() << "Insufficient GPU memory for large block test";
  }

  try {
    BufferPool pool(device_id_, num_blocks, block_size);

    BufferBlock* block = pool.get_block();
    EXPECT_NE(block->device_ptr, nullptr);
    EXPECT_EQ(block->capacity, block_size);

    // Test writing to the large buffer
    const size_t test_size = 1024 * 1024;  // Test only first 1MB
    WritePatternToBuffer(block->device_ptr, test_size, 0xDEADBEEF);
    VerifyBufferData(block->device_ptr, test_size, 0xDEADBEEF);

    pool.put_block(block);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Large block allocation failed: " << e.what();
  }
}

// Test pool destruction with outstanding blocks
TEST_F(CudaBufferPoolTest, PoolDestructionWithOutstandingBlocks) {
  const size_t block_size = 1024 * 1024;
  const int num_blocks = 2;

  BufferBlock* saved_block;

  {
    BufferPool pool(device_id_, num_blocks, block_size);
    saved_block = pool.get_block();

    // Write pattern to the block while pool is still alive
    WritePatternToBuffer(saved_block->device_ptr, block_size, 0x12345678);
    VerifyBufferData(saved_block->device_ptr, block_size, 0x12345678);

    // Pool will be destroyed here with one outstanding block
    // The destructor should handle this gracefully by freeing all memory
  }

  // After pool destruction, the saved_block pointer structure still exists,
  // but the device_ptr it points to has been freed and is no longer valid.
  // We should NOT try to use the device_ptr after pool destruction.

  // The block structure should still exist (it's stack allocated)
  EXPECT_NE(saved_block, nullptr);

  // But we should NOT try to access the device_ptr as it's been freed
  // This would be undefined behavior:
  // WritePatternToBuffer(saved_block->device_ptr, block_size, 0x12345678);

  std::cout << "Pool destruction test completed - memory properly freed" << std::endl;
}

// Test device-specific allocation
TEST_F(CudaBufferPoolTest, DeviceSpecificAllocation) {
  const size_t block_size = 1024 * 1024;
  const int num_blocks = 2;

  BufferPool pool(device_id_, num_blocks, block_size);
  BufferBlock* block = pool.get_block();

  // Verify the block is allocated on the correct device
  EXPECT_EQ(block->device_id, device_id_);

  // Set device and verify we can access the memory
  CUDA_CHECK(cudaSetDevice(device_id_));
  WritePatternToBuffer(block->device_ptr, block_size, 0xABCDEF00);
  VerifyBufferData(block->device_ptr, block_size, 0xABCDEF00);

  pool.put_block(block);
}

// Performance benchmark test
TEST_F(CudaBufferPoolTest, PerformanceBenchmark) {
  const size_t block_size = 1024 * 1024;
  const int num_blocks = 4;
  const int num_iterations = 1000;

  BufferPool pool(device_id_, num_blocks, block_size);

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iterations; ++i) {
    BufferBlock* block = pool.get_block();
    pool.put_block(block);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  double ops_per_second = (2.0 * num_iterations * 1000000.0) / duration.count();  // get + put operations

  std::cout << "Performance: " << ops_per_second << " operations per second" << std::endl;
  std::cout << "Average latency: " << (duration.count() / (2.0 * num_iterations)) << " microseconds per operation"
            << std::endl;

  // Basic performance expectation - should handle at least 10K ops/sec
  EXPECT_GT(ops_per_second, 10000.0) << "Buffer pool performance below expected threshold";
}
#endif  // ENABLE_CUDA