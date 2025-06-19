/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>
#include "include/gtest/gtest.h"
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

namespace ksana_llm {
class ExpertParallelDataHubTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
  }

  void TearDown() override {}
};

TEST_F(ExpertParallelDataHubTest, TestExpertParallelDataHub) {
  int rank = 0;
  bool is_prefill = false;
  InitializeExpertHiddenUnitBufferPool();

  EXPECT_TRUE(GetExpertHiddenUnitBufferPool() != nullptr);

  // Initialize hidden units with schedule_id
  Status status = InitExpertHiddenUnits();
  EXPECT_TRUE(status.OK());

  // Get hidden_unit_buffer created by InitExpertHiddenUnits().
  HiddenUnitDeviceBuffer* cur_dev_hidden_unit = GetCurrentExpertSendHiddenUnitBuffer();
  EXPECT_TRUE(cur_dev_hidden_unit != nullptr);

  // get from conv queue
  HiddenUnitDeviceBuffer* send_dev_hidden_unit;
  auto send_fn = [&]() {
    send_dev_hidden_unit = GetExpertHiddenUnitBufferPool()->GetFromSendQueue();
    GetExpertHiddenUnitBufferPool()->NotifySendFinished();
  };
  std::thread send_thread(send_fn);

  // Add send task.
  bool is_sync = true;
  status = SendExpertHiddenUnits(cur_dev_hidden_unit, is_sync);
  EXPECT_TRUE(status.OK());

  send_thread.join();

  GetExpertHiddenUnitBufferPool()->PutToDeviceRecvQueue(cur_dev_hidden_unit);
  HiddenUnitDeviceBuffer* recv_hidden_unit = RecvExpertHiddenUnits(rank);
  EXPECT_EQ(recv_hidden_unit, cur_dev_hidden_unit);

  GetExpertHiddenUnitBufferPool()->PutToDeviceRecvQueue(cur_dev_hidden_unit);
  HiddenUnitDeviceBuffer* async_recv_hidden_unit = AsyncRecvExpertHiddenUnits(rank);
  EXPECT_EQ(async_recv_hidden_unit, cur_dev_hidden_unit);

  HiddenUnitDeviceBuffer* send_hidden_unit;
  Tensor tmp_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, cur_dev_hidden_unit->tensors[rank].dtype,
                             cur_dev_hidden_unit->tensors[rank].shape, rank);
  // CopyFromHiddenUnitBuffer(tmp_tensor, GetCurrentHiddenUnitBuffer(SCHEDULE_ID), rank, is_prefill);
  // CopyToHiddenUnitBuffer(GetCurrentHiddenUnitBuffer(SCHEDULE_ID), tmp_tensor, rank, is_prefill);

  // Test FreeHiddenUnits
  Status free_status = FreeExpertRecvHiddenUnits(cur_dev_hidden_unit);
  EXPECT_TRUE(free_status.OK());

  DestroyExpertHiddenUnitBufferPool();
}

}  // namespace ksana_llm