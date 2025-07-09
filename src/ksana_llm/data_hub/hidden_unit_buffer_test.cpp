/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

#include "ksana_llm/helpers/environment_test_helper.h"

#ifdef ENABLE_TOPS
#  include "3rdparty/half/include/half.hpp"
#endif

using namespace ksana_llm;

class HiddenUnitBufferTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
  }

  void TearDown() override {}

  void InitBufferSize() {
    ModelConfig model_config;
    Status status = Singleton<Environment>::GetInstance()->GetModelConfig(model_config);
    if (!status.OK()) {
      throw std::runtime_error("No model_config provided.");
    }

    weight_type_ = model_config.weight_data_type;
    tensor_para_size_ = model_config.tensor_para_size;
    max_token_num_ = model_config.max_step_token_num;
    hidden_unit_size_ = model_config.size_per_head * model_config.head_num;
  }

  void SetHiddenUnitBuffer(HiddenUnitHostBuffer* host_hidden_unit, size_t dim0, size_t dim1) {
    host_hidden_unit->shape_dims[0] = dim0;
    host_hidden_unit->shape_dims[1] = dim1;

    if (weight_type_ == DataType::TYPE_FP16) {
      size_t buffer_size =
          host_hidden_unit->shape_dims[0] * host_hidden_unit->shape_dims[1] * GetTypeSize(weight_type_);

      for (size_t i = 0; i < host_hidden_unit->tensor_parallel; ++i) {
#ifdef ENABLE_CUDA
        std::vector<half> vec;
        for (size_t j = 0; j < dim0 * dim1; ++j) {
          vec.push_back(1.0 * (j + 1) * (i + 1));
        }
#endif

#ifdef ENABLE_ACL
        std::vector<aclFloat16> vec;
        for (size_t j = 0; j < dim0 * dim1; ++j) {
          vec.push_back(aclFloatToFloat16(1.0 * (j + 1) * (i + 1)));
        }
#endif

#ifdef ENABLE_TOPS
        std::vector<float16> vec;

        for (size_t j = 0; j < dim0 * dim1; ++j) {
          vec.push_back(half_float::half(1.0 * (j + 1) * (i + 1)));
        }
#endif
        memcpy(host_hidden_unit->data + (i * buffer_size), vec.data(), buffer_size);
      }
    }
  }

  bool CheckHiddenUnitBuffer(HiddenUnitHostBuffer* src_host_hidden_unit, HiddenUnitHostBuffer* dst_host_hidden_unit) {
    if (src_host_hidden_unit->tensor_parallel != dst_host_hidden_unit->tensor_parallel) {
      return false;
    }

    if (src_host_hidden_unit->shape_dims[0] != dst_host_hidden_unit->shape_dims[0] ||
        src_host_hidden_unit->shape_dims[1] != dst_host_hidden_unit->shape_dims[1]) {
      return false;
    }

    size_t buffer_element_num = src_host_hidden_unit->shape_dims[0] * src_host_hidden_unit->shape_dims[1];

    for (size_t i = 0; i < src_host_hidden_unit->tensor_parallel; ++i) {
      for (size_t j = 0; j < (src_host_hidden_unit->shape_dims[0] * src_host_hidden_unit->shape_dims[1]); ++j) {
#ifdef ENABLE_CUDA
        if (src_host_hidden_unit->data[i * buffer_element_num + j] !=
            dst_host_hidden_unit->data[i * buffer_element_num + j]) {
          return false;
        }
#endif

#ifdef ENABLE_ACL
        if (aclFloat16ToFloat(src_host_hidden_unit->data[i * buffer_element_num + j]) !=
            aclFloat16ToFloat(dst_host_hidden_unit->data[i * buffer_element_num + j])) {
          return false;
        }
#endif
      }
    }

    return true;
  }

 protected:
  DataType weight_type_;
  size_t max_token_num_;
  size_t tensor_para_size_;
  size_t hidden_unit_size_;
};

TEST_F(HiddenUnitBufferTest, TestConvert) {
  InitializeHiddenUnitBufferPool();
  InitBufferSize();

  // Get a host buffer.
  Packet* packet = GetHiddenUnitBufferPool()->GetHostBuffer();
  EXPECT_TRUE(packet != nullptr);

  HiddenUnitHostBuffer* host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);

  EXPECT_EQ(host_hidden_unit->shape_dims[0], max_token_num_);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], hidden_unit_size_);
  EXPECT_EQ(host_hidden_unit->tensor_parallel, tensor_para_size_);

  // Get a device buffer.
  HiddenUnitDeviceBuffer* dev_hidden_unit = GetHiddenUnitBufferPool()->GetDeviceBuffer();
  EXPECT_EQ(dev_hidden_unit->tensors.size(), tensor_para_size_);
  EXPECT_EQ(dev_hidden_unit->tensors[0].shape[0], max_token_num_);
  EXPECT_EQ(dev_hidden_unit->tensors[0].shape[1], hidden_unit_size_);

  // Set value.
  SetHiddenUnitBuffer(host_hidden_unit, 4, 3);
  EXPECT_EQ(host_hidden_unit->shape_dims[0], 4);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], 3);

  // Covert to device.
  GetHiddenUnitBufferPool()->ConvertHostBufferToDevice(dev_hidden_unit, host_hidden_unit);
  EXPECT_EQ(host_hidden_unit->shape_dims[0], dev_hidden_unit->tensors[0].shape[0]);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], dev_hidden_unit->tensors[0].shape[1]);

  // Convert back to host.
  Packet* new_packet = GetHiddenUnitBufferPool()->GetHostBuffer();
  EXPECT_TRUE(new_packet != nullptr);
  HiddenUnitHostBuffer* new_host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(new_packet->body);
  GetHiddenUnitBufferPool()->ConvertDeviceBufferToHost(new_host_hidden_unit, dev_hidden_unit);
  EXPECT_EQ(host_hidden_unit->shape_dims[0], new_host_hidden_unit->shape_dims[0]);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], new_host_hidden_unit->shape_dims[1]);

  // Check value.
  EXPECT_TRUE(CheckHiddenUnitBuffer(host_hidden_unit, new_host_hidden_unit));

  // Free buffer.
  GetHiddenUnitBufferPool()->FreeHostBuffer(packet);
  GetHiddenUnitBufferPool()->FreeHostBuffer(new_packet);
  GetHiddenUnitBufferPool()->FreeDeviceBuffer(dev_hidden_unit);

  DestroyHiddenUnitBufferPool();
}

TEST_F(HiddenUnitBufferTest, HiddenUnitBufferCommonTest) {
  InitializeHiddenUnitBufferPool();
  HiddenUnitDeviceBuffer* hidden_unit_buffer = nullptr;

  size_t multi_batch_id = 123;

  hidden_unit_buffer = GetHiddenUnitBufferPool()->GetDeviceBuffer();
  EXPECT_TRUE(hidden_unit_buffer != nullptr);

  hidden_unit_buffer->multi_batch_id = multi_batch_id;
  SetCurrentHiddenUnitBuffer(hidden_unit_buffer);
  EXPECT_TRUE(GetCurrentHiddenUnitBuffer(multi_batch_id) == hidden_unit_buffer);

  GetHiddenUnitBufferPool()->FreeDeviceBuffer(hidden_unit_buffer);
  EXPECT_TRUE(GetHiddenUnitBufferPool()->GetDeviceBuffer() == hidden_unit_buffer);

  GetHiddenUnitBufferPool()->Stop();
  DestroyHiddenUnitBufferPool();
  EXPECT_TRUE(GetHiddenUnitBufferPool() == nullptr);
}

TEST_F(HiddenUnitBufferTest, TestHiddenUnitBufferPool) {
  InitializeHiddenUnitBufferPool();
  InitBufferSize();

  // Get a host buffer.
  Packet* packet = GetHiddenUnitBufferPool()->GetHostBuffer();
  EXPECT_TRUE(packet != nullptr);

  // Assign a id.
  HiddenUnitHostBuffer* host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);
  host_hidden_unit->multi_batch_id = 235;

  // Put to recv queue and get it.
  GetHiddenUnitBufferPool()->PutToHostRecvQueue(packet);
  Packet* recv_packet = GetHiddenUnitBufferPool()->GetFromHostRecvQueue();
  HiddenUnitHostBuffer* recv_host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(recv_packet->body);
  EXPECT_EQ(host_hidden_unit->multi_batch_id, recv_host_hidden_unit->multi_batch_id);

  // Get a device buffer and converted from a host buffer.
  HiddenUnitDeviceBuffer* dev_hidden_unit = GetHiddenUnitBufferPool()->GetDeviceBuffer();
  GetHiddenUnitBufferPool()->ConvertHostBufferToDevice(dev_hidden_unit, recv_host_hidden_unit);
  EXPECT_EQ(host_hidden_unit->multi_batch_id, dev_hidden_unit->multi_batch_id);

  HiddenUnitDeviceBuffer* send_dev_hidden_unit;
  auto send_fn = [&]() {
    send_dev_hidden_unit = GetHiddenUnitBufferPool()->GetFromPendingSendQueue();
    send_dev_hidden_unit->NotifyFinished();
  };
  std::thread send_thread(send_fn);

  // Put to send queue and get it.
  GetHiddenUnitBufferPool()->PutToPendingSendQueue(dev_hidden_unit);

  send_thread.join();
  EXPECT_EQ(host_hidden_unit->multi_batch_id, send_dev_hidden_unit->multi_batch_id);

  // Get Preallocated device buffer.
  GetHiddenUnitBufferPool()->PreAllocateDeviceBuffer();

  // Free buffers.
  GetHiddenUnitBufferPool()->FreeHostBuffer(packet);
  GetHiddenUnitBufferPool()->FreeDeviceBuffer(dev_hidden_unit);

  DestroyHiddenUnitBufferPool();
}

TEST_F(HiddenUnitBufferTest, TestMultiThreadRecvWait) {
  InitializeHiddenUnitBufferPool();
  InitBufferSize();

  // Test with one sender and multiple receivers with different timing
  {
    std::vector<int> num_iterations = {5, 4, 8, 6};  // Number of buffers for every receiver
    const int num_receivers = 4;                     // Number of receiver threads

    std::atomic<int> sender_iteration(0);
    std::vector<std::atomic<int>> receiver_counts(num_receivers);

    auto sender_fn = [&]() {
      int total_iterations = 0;
      std::vector<int> sent_buffer_num(num_receivers);
      for (int r = 0; r < num_receivers; r++) {
        total_iterations += num_iterations[r];
        sent_buffer_num[r] = 0;
      }

      while (total_iterations > 0) {
        for (int r = 0; r < num_receivers; r++) {
          if (sent_buffer_num[r] >= num_iterations[r]) {
            continue;
          }
          // First wait until ready to receive
          auto buffer = GetHiddenUnitBufferPool()->GetDeviceBuffer();
          buffer->multi_batch_id = r;

          // Then put buffer in the queue
          buffer->NotifyFinished();
          GetHiddenUnitBufferPool()->PutToDeviceRecvedQueue(buffer);

          sent_buffer_num[r]++;
          total_iterations--;

          // Add some delay between iterations
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      }
    };

    // Receiver threads: call GetFromDeviceRecvQueue at different times
    std::vector<std::thread> receiver_threads;
    for (int r = 0; r < num_receivers; r++) {
      receiver_threads.emplace_back([&, r]() {
        // Each receiver has a different delay pattern
        int delay_ms = (r * 30) % 100;  // Different initial delay for each receiver

        for (int i = 0; i < num_iterations[r]; i++) {
          // Add some variable delay to create different timing patterns
          std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

          // Try to get the buffer for the current iteration
          try {
            HiddenUnitDeviceBuffer* received_buffer = GetHiddenUnitBufferPool()->GetFromDeviceRecvedQueue(r);
            // Verify the buffer and free it
            EXPECT_EQ(received_buffer->multi_batch_id, r);

            GetHiddenUnitBufferPool()->FreeDeviceBuffer(received_buffer);
            receiver_counts[r]++;
            // Change delay for next iteration to create different timing patterns
            delay_ms = (delay_ms + 23) % 120;
          } catch (const std::exception& e) {
            // If we get an exception, just continue
            KLLM_LOG_ERROR << "Receiver " << r << " got exception: " << e.what();
          }
        }
      });
    }

    // Start sender thread
    std::thread sender_thread(sender_fn);
    sender_thread.join();

    // Wait for receivers to finish
    for (auto& thread : receiver_threads) {
      thread.join();
    }

    // Verify that all receivers got expected number of buffers
    for (int r = 0; r < num_receivers; r++) {
      EXPECT_EQ(num_iterations[r], receiver_counts[r]);
    }
  }

  DestroyHiddenUnitBufferPool();
}
