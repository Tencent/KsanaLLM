/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/dynamic_memory_pool.h"
#include <gtest/gtest.h>

#include <iostream>

#include "ksana_llm/utils/logger.h"
#include "tests/test.h"

namespace ksana_llm {

TEST(TensorMemoryPool, TestAllocateAndFree) {
  size_t block_size = 1024;
  size_t memory_size = 1024 * 1024 + 1023;
  void *base_memory = reinterpret_cast<void *>(1024 * 1024 * 1024);

  DynamicMemoryPool mp1(base_memory, memory_size, block_size);

  size_t init_total_block_num = mp1.GetTotalBlockNum();
  size_t init_free_block_num = mp1.GetFreeBlockNum();
  size_t init_used_block_num = mp1.GetUsedBlockNum();

  // Should be 1024, 1024, 0
  EXPECT_EQ(init_total_block_num, 1024);
  EXPECT_EQ(init_free_block_num, 1024);
  EXPECT_EQ(init_used_block_num, 0);

  base_memory = reinterpret_cast<void *>(1024 * 1024 * 1024 - 1023);
  DynamicMemoryPool mp(base_memory, memory_size, block_size);

  // base_ptr % block_size should be zero.
  void *base_ptr = mp.GetBasePtr();
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(base_ptr) % block_size, 0);

  init_total_block_num = mp.GetTotalBlockNum();
  init_free_block_num = mp.GetFreeBlockNum();
  init_used_block_num = mp.GetUsedBlockNum();

  // Should be 1024, 1024, 0
  EXPECT_EQ(init_total_block_num, 1024);
  EXPECT_EQ(init_free_block_num, 1024);
  EXPECT_EQ(init_used_block_num, 0);

  // Should up to 5 blocks.
  void *resident_1 = mp.Allocate(5 * 1024 - 1, true);
  void *resident_2 = mp.Allocate(4 * 1024 + 1, true);

  init_total_block_num = mp.GetTotalBlockNum();
  init_free_block_num = mp.GetFreeBlockNum();
  init_used_block_num = mp.GetUsedBlockNum();

  // Should be 1024, 1014, 10
  EXPECT_EQ(init_total_block_num, 1024);
  EXPECT_EQ(init_free_block_num, 1014);
  EXPECT_EQ(init_used_block_num, 10);

  // Allocate zero bytes, should not error.
  void *large_resident = mp.Allocate(0, true);
  EXPECT_TRUE(large_resident != nullptr);

  // Allocate large block, should error.
  bool large_resident_error = false;
  try {
    void *large_resident = mp.Allocate(1015 * 1024, true);
  } catch (...) {
    large_resident_error = true;
  }
  EXPECT_TRUE(large_resident_error);

  // Free resident, should do nothing.
  mp.Free(resident_1);
  mp.Free(resident_2);

  init_total_block_num = mp.GetTotalBlockNum();
  init_free_block_num = mp.GetFreeBlockNum();
  init_used_block_num = mp.GetUsedBlockNum();

  // Should be still 1024, 1014, 10
  EXPECT_EQ(init_total_block_num, 1024);
  EXPECT_EQ(init_free_block_num, 1014);
  EXPECT_EQ(init_used_block_num, 10);

  // Free nullptr, should error.
  bool free_nullptr_error = false;
  try {
    mp.Free(nullptr);
  } catch (...) {
    free_nullptr_error = true;
  }
  EXPECT_TRUE(free_nullptr_error);

  // Allocate 500 blocks.
  void *dynamic_1 = mp.Allocate(500 * 1024, false);
  void *dynamic_2 = mp.Allocate(500 * 1024, false);

  init_total_block_num = mp.GetTotalBlockNum();
  init_free_block_num = mp.GetFreeBlockNum();
  init_used_block_num = mp.GetUsedBlockNum();

  // Should be still 1024, 14, 1010
  EXPECT_EQ(init_total_block_num, 1024);
  EXPECT_EQ(init_free_block_num, 14);
  EXPECT_EQ(init_used_block_num, 1010);

  // Allocate 15 block, should error.
  bool allocate_dynamic_error = false;
  try {
    void *dynamic_3 = mp.Allocate(15 * 1024, false);
  } catch (...) {
    allocate_dynamic_error = true;
  }
  EXPECT_TRUE(allocate_dynamic_error);

  // Free dynamic_1
  mp.Free(dynamic_1);

  // Allocate 250 blocks.
  void *dynamic_4 = mp.Allocate(250 * 1024, false);
  mp.Free(dynamic_2);

  // Allocate 750 blocks.
  void *dynamic_5 = mp.Allocate(750 * 1024, false);

  init_total_block_num = mp.GetTotalBlockNum();
  init_free_block_num = mp.GetFreeBlockNum();
  init_used_block_num = mp.GetUsedBlockNum();

  // Should be 1024, 14, 1010
  EXPECT_EQ(init_total_block_num, 1024);
  EXPECT_EQ(init_free_block_num, 14);
  EXPECT_EQ(init_used_block_num, 1010);

  // Free dynamic_4 & dynamic_5
  mp.Free(dynamic_4);
  mp.Free(dynamic_5);

  init_total_block_num = mp.GetTotalBlockNum();
  init_free_block_num = mp.GetFreeBlockNum();
  init_used_block_num = mp.GetUsedBlockNum();

  // Should be 1024, 1014, 10
  EXPECT_EQ(init_total_block_num, 1024);
  EXPECT_EQ(init_free_block_num, 1014);
  EXPECT_EQ(init_used_block_num, 10);

  size_t max_used_byte = mp.GetMaxUsedByte();
  EXPECT_EQ(max_used_byte, 1010 * 1024);
}

}  // namespace ksana_llm
