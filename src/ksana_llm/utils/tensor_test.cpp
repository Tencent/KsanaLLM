/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>
#include <numeric>

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/singleton.h"
#include "tensor.h"
#include "tensor_test_helper.h"
#include "test.h"

namespace ksana_llm {

TEST(WorkspaceTest, CommonTest) {
  InitializeTestEnvironment();
  WorkSpaceFunc f = GetWorkSpaceFunc();

  void* ws_addr_1 = nullptr;
  f(1024, &ws_addr_1);

  void* ws_addr_2 = nullptr;
  f(2048, &ws_addr_2);

  void* ws_addr_3 = nullptr;
  f(1536, &ws_addr_3);
  EXPECT_EQ(ws_addr_2, ws_addr_3);
}

TEST(TensorTest, CommonTest) {
  InitializeTestEnvironment();

  constexpr size_t tensor_parallel_size = 1;
  constexpr int attn_data_parallel_size = 1;
  constexpr int multi_batch_num = 1;
  std::shared_ptr<Context> context =
      std::make_shared<Context>(tensor_parallel_size, attn_data_parallel_size, multi_batch_num);
  constexpr size_t ELEM_NUM = 16;
  constexpr int RANK = 0;

  std::vector<int32_t> src_data(ELEM_NUM, 0);

  std::iota(src_data.begin(), src_data.end(), 1);
  Tensor tensor_with_block_id_on_host;
  Tensor tensor_with_refer_ptr_on_host;
  Tensor tensor_with_block_id_on_dev;
  Tensor tensor_with_refer_ptr_on_dev;
  Tensor output_data;

  tensor_with_block_id_on_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {ELEM_NUM}, RANK);
  tensor_with_refer_ptr_on_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {ELEM_NUM}, RANK, src_data.data());
  for (size_t src_idx = 0; src_idx < src_data.size(); ++src_idx) {
    EXPECT_EQ(tensor_with_refer_ptr_on_host.GetPtr<int32_t>()[src_idx], src_data[src_idx]);
  }
  std::memcpy(tensor_with_block_id_on_host.GetPtr<int32_t>(), tensor_with_refer_ptr_on_host.GetPtr<int32_t>(),
              tensor_with_block_id_on_host.GetTotalBytes());
  for (size_t src_idx = 0; src_idx < src_data.size(); ++src_idx) {
    EXPECT_EQ(tensor_with_block_id_on_host.GetPtr<int32_t>()[src_idx], src_data[src_idx]);
  }

  tensor_with_block_id_on_dev = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {ELEM_NUM}, RANK);
  MemcpyAsync(tensor_with_block_id_on_dev.GetPtr<int32_t>(), src_data.data(),
              tensor_with_block_id_on_dev.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE, context->GetH2DStreams()[RANK]);
  StreamSynchronize(context->GetH2DStreams()[RANK]);
  tensor_with_refer_ptr_on_dev = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {ELEM_NUM}, RANK,
                                        tensor_with_block_id_on_dev.GetPtr<int32_t>());
  MemcpyAsync(tensor_with_refer_ptr_on_dev.GetPtr<int32_t>(), tensor_with_block_id_on_dev.GetPtr<int32_t>(),
              tensor_with_refer_ptr_on_dev.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE, context->GetD2DStreams()[RANK]);
  StreamSynchronize(context->GetD2DStreams()[RANK]);
  output_data = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {ELEM_NUM}, RANK);
  MemcpyAsync(output_data.GetPtr<int32_t>(), tensor_with_refer_ptr_on_dev.GetPtr<int32_t>(),
              tensor_with_refer_ptr_on_dev.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST, context->GetD2HStreams()[RANK]);
  StreamSynchronize(context->GetD2HStreams()[RANK]);
  for (size_t src_idx = 0; src_idx < src_data.size(); ++src_idx) {
    EXPECT_EQ(output_data.GetPtr<int32_t>()[src_idx], src_data[src_idx]);
  }
  EXPECT_EQ(tensor_with_refer_ptr_on_dev.GetPtr<int32_t>(), tensor_with_block_id_on_dev.GetPtr<int32_t>());

  std::string file_path = "test_tensor.npy";
  Tensor tensor_on_device(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {2, 2}, 0);
  Tensor tensor_on_host(MemoryLocation::LOCATION_HOST, TYPE_INT32, {2, 2}, 0);

  // for serialization Save/Load from npy file
  std::vector<int32_t> data = {1, 2, 3, 4};
  MemcpyAsync(tensor_on_device.GetPtr<int32_t>(), data.data(), tensor_on_device.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context->GetD2HStreams()[RANK]);

  StreamSynchronize(context->GetD2HStreams()[RANK]);

  tensor_on_device.SaveToNpyFile(file_path);

  tensor_on_host.LoadFromNpyFile(file_path);
  EXPECT_EQ(tensor_on_host.GetElementNumber(), tensor_on_device.GetElementNumber());
  EXPECT_EQ(tensor_on_host.GetDTypeSize(), tensor_on_device.GetDTypeSize());
  EXPECT_EQ(tensor_on_host.GetTotalBytes(), tensor_on_device.GetTotalBytes());
  for (size_t i = 0; i < tensor_on_host.GetElementNumber(); ++i) {
    EXPECT_EQ(tensor_on_host.GetPtr<int32_t>()[i], data[i]);
  }

  // Check tensor checker.
  MemoryChecker::enabled_ = true;

  Tensor tensor_tmp(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {1024}, RANK);
  MemoryChecker::AddMemoryBlock("test", RANK, tensor_tmp.GetPtr<void>(), 256, tensor_tmp.GetPtr<uint8_t>() + 768, 256,
                                0);
  EXPECT_TRUE(MemoryChecker::Enabled());

  // Should be true
  bool true_check = true;
  Memset(tensor_tmp.GetPtr<void>(), 0, 1024);
  try {
    MemoryChecker::CheckMemory(RANK, false);
  } catch (...) {
    true_check = false;
  }
  EXPECT_TRUE(true_check);

  // Should be false
  Memset(tensor_tmp.GetPtr<void>(), 1, 1024);
  try {
    MemoryChecker::CheckMemory(RANK, false);
  } catch (...) {
    true_check = false;
  }
  EXPECT_FALSE(true_check);
  MemoryChecker::RemoveMemoryBlock("test", RANK);
}
}  // namespace ksana_llm
