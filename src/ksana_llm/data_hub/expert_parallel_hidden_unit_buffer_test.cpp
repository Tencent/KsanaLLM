/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/expert_data_hub.h"
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

class ExpertParallelHiddenUnitBufferTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
  }

  void TearDown() override {}

  void InitBufferSize() {
    std::unordered_map<std::string, ModelConfig> model_configs;
    Singleton<Environment>::GetInstance()->GetModelConfigs(model_configs);
    if (model_configs.empty()) {
      throw std::runtime_error("No model_config provided.");
    }

    ModelConfig model_config = model_configs.begin()->second;

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

TEST_F(ExpertParallelHiddenUnitBufferTest, TestConvert) {
  InitializeExpertHiddenUnitBufferPool();
  InitBufferSize();

  int rank = 0;

  // Get a host buffer.
  Packet* packet = GetExpertHiddenUnitBufferPool()->GetHostBuffer();
  EXPECT_TRUE(packet != nullptr);

  HiddenUnitHostBuffer* host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);

  EXPECT_EQ(host_hidden_unit->shape_dims[0], max_token_num_);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], hidden_unit_size_);
  EXPECT_EQ(host_hidden_unit->tensor_parallel, tensor_para_size_);

  // Get a device buffer.
  HiddenUnitDeviceBuffer* dev_hidden_unit = GetExpertHiddenUnitBufferPool()->GetDeviceBuffer(rank);
  EXPECT_EQ(dev_hidden_unit->tensors.size(), tensor_para_size_);
  EXPECT_EQ(dev_hidden_unit->tensors[0].shape[0], max_token_num_);
  EXPECT_EQ(dev_hidden_unit->tensors[0].shape[1], hidden_unit_size_);

  // Set value.
  SetHiddenUnitBuffer(host_hidden_unit, 4, 3);
  EXPECT_EQ(host_hidden_unit->shape_dims[0], 4);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], 3);

  // Covert to device.
  GetExpertHiddenUnitBufferPool()->ConvertHostBufferToDevice(dev_hidden_unit, host_hidden_unit);
  EXPECT_EQ(host_hidden_unit->shape_dims[0], dev_hidden_unit->tensors[0].shape[0]);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], dev_hidden_unit->tensors[0].shape[1]);

  // Convert back to host.
  Packet* new_packet = GetExpertHiddenUnitBufferPool()->GetHostBuffer();
  EXPECT_TRUE(new_packet != nullptr);
  HiddenUnitHostBuffer* new_host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(new_packet->body);
  GetExpertHiddenUnitBufferPool()->ConvertDeviceBufferToHost(new_host_hidden_unit, dev_hidden_unit);
  EXPECT_EQ(host_hidden_unit->shape_dims[0], new_host_hidden_unit->shape_dims[0]);
  EXPECT_EQ(host_hidden_unit->shape_dims[1], new_host_hidden_unit->shape_dims[1]);

  // Check value.
  EXPECT_TRUE(CheckHiddenUnitBuffer(host_hidden_unit, new_host_hidden_unit));

  // Free buffer.
  GetExpertHiddenUnitBufferPool()->FreeHostBuffer(packet);
  GetExpertHiddenUnitBufferPool()->FreeHostBuffer(new_packet);
  GetExpertHiddenUnitBufferPool()->FreeDeviceBuffer(dev_hidden_unit);

  DestroyExpertHiddenUnitBufferPool();
}

TEST_F(ExpertParallelHiddenUnitBufferTest, HiddenUnitBufferCommonTest) {
  InitializeExpertHiddenUnitBufferPool();
  HiddenUnitDeviceBuffer* hidden_unit_buffer = nullptr;

  int rank = 0;
  size_t multi_batch_id = 123;

  hidden_unit_buffer = GetExpertHiddenUnitBufferPool()->GetDeviceBuffer(rank);
  EXPECT_TRUE(hidden_unit_buffer != nullptr);

  hidden_unit_buffer->multi_batch_id = multi_batch_id;
  SetCurrentExpertRecvHiddenUnitBuffer(hidden_unit_buffer);
  EXPECT_TRUE(GetCurrentExpertRecvHiddenUnitBuffer() == hidden_unit_buffer);

  GetExpertHiddenUnitBufferPool()->FreeDeviceBuffer(hidden_unit_buffer);
  EXPECT_TRUE(GetExpertHiddenUnitBufferPool()->GetDeviceBuffer(rank) == hidden_unit_buffer);

  GetExpertHiddenUnitBufferPool()->Stop();
  DestroyExpertHiddenUnitBufferPool();
  EXPECT_TRUE(GetExpertHiddenUnitBufferPool() == nullptr);
}

TEST_F(ExpertParallelHiddenUnitBufferTest, TestHiddenUnitBufferPool) {
  InitializeExpertHiddenUnitBufferPool();
  InitBufferSize();

  int rank = 0;

  // Get a host buffer.
  Packet* packet = GetExpertHiddenUnitBufferPool()->GetHostBuffer();
  EXPECT_TRUE(packet != nullptr);

  // Assign a id.
  HiddenUnitHostBuffer* host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);
  host_hidden_unit->multi_batch_id = 235;

  // Put to recv queue and get it.
  GetExpertHiddenUnitBufferPool()->PutToHostRecvQueue(packet);
  Packet* recv_packet = GetExpertHiddenUnitBufferPool()->GetFromHostRecvQueue();
  HiddenUnitHostBuffer* recv_host_hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(recv_packet->body);
  EXPECT_EQ(host_hidden_unit->multi_batch_id, recv_host_hidden_unit->multi_batch_id);

  // Get a device buffer and converted from a host buffer.
  HiddenUnitDeviceBuffer* dev_hidden_unit = GetExpertHiddenUnitBufferPool()->GetDeviceBuffer(rank);
  GetExpertHiddenUnitBufferPool()->ConvertHostBufferToDevice(dev_hidden_unit, recv_host_hidden_unit);
  EXPECT_EQ(host_hidden_unit->multi_batch_id, dev_hidden_unit->multi_batch_id);

  HiddenUnitDeviceBuffer* send_dev_hidden_unit;
  auto send_fn = [&]() {
    send_dev_hidden_unit = GetExpertHiddenUnitBufferPool()->GetFromSendQueue();
    GetExpertHiddenUnitBufferPool()->NotifySendFinished();
  };
  std::thread send_thread(send_fn);

  // Put to send queue and get it.
  GetExpertHiddenUnitBufferPool()->PutToSendQueue(dev_hidden_unit);

  send_thread.join();
  EXPECT_EQ(host_hidden_unit->multi_batch_id, send_dev_hidden_unit->multi_batch_id);

  // Get Preallocated device buffer.
  GetExpertHiddenUnitBufferPool()->PreAllocateDeviceBuffer();

  // Free buffers.
  GetExpertHiddenUnitBufferPool()->FreeHostBuffer(packet);
  GetExpertHiddenUnitBufferPool()->FreeDeviceBuffer(dev_hidden_unit);

  DestroyExpertHiddenUnitBufferPool();
}
