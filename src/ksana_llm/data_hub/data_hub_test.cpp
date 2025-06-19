/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>
#include "include/gtest/gtest.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

#include "ksana_llm/helpers/environment_test_helper.h"

using namespace ksana_llm;

// For data verification.
constexpr size_t SCHEDULE_ID = 235;

class DataHubTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
  }

  void TearDown() override {}
};

TEST_F(DataHubTest, TestDataHub) {
  int rank = 0;
  bool is_prefill = false;
  InitializeScheduleOutputPool();
  InitializeHiddenUnitBufferPool();

  EXPECT_TRUE(GetScheduleOutputPool() != nullptr);
  EXPECT_TRUE(GetHiddenUnitBufferPool() != nullptr);

  // Initialize hidden units with schedule_id
  Status status = InitHiddenUnits(SCHEDULE_ID);
  EXPECT_TRUE(status.OK());

  HiddenUnitDeviceBuffer* cur_dev_hidden_unit = GetCurrentHiddenUnitBuffer(SCHEDULE_ID);
  EXPECT_EQ(cur_dev_hidden_unit->schedule_id, SCHEDULE_ID);

  // Get schedule output.
  ScheduleOutput* schedule_output = GetScheduleOutputPool()->GetScheduleOutput();
  schedule_output->schedule_id = SCHEDULE_ID;

  // Broadcast schedule output.
  BroadcastScheduleOutput(schedule_output);

  // get from send queue.
  Packet* send_schedule_output_packet = GetScheduleOutputPool()->GetFromSendQueue();

  ScheduleOutput* send_schedule_output = new ScheduleOutput();
  ScheduleOutputParser::DeserializeScheduleOutput(send_schedule_output_packet->body, send_schedule_output);

  EXPECT_EQ(send_schedule_output->schedule_id, SCHEDULE_ID);

  // get from conv queue
  HiddenUnitDeviceBuffer* send_dev_hidden_unit;
  auto send_fn = [&]() {
    send_dev_hidden_unit = GetHiddenUnitBufferPool()->GetFromSendQueue();
    GetHiddenUnitBufferPool()->NotifySendFinished();
  };
  std::thread send_thread(send_fn);

  SendHiddenUnits(SCHEDULE_ID);

  send_thread.join();
  EXPECT_EQ(send_dev_hidden_unit->schedule_id, SCHEDULE_ID);

  Tensor tmp_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, GetCurrentHiddenUnitBuffer(SCHEDULE_ID)->tensors[rank].dtype,
             GetCurrentHiddenUnitBuffer(SCHEDULE_ID)->tensors[rank].shape, rank);
  CopyFromHiddenUnitBuffer(tmp_tensor, GetCurrentHiddenUnitBuffer(SCHEDULE_ID), rank, is_prefill);
  CopyToHiddenUnitBuffer(GetCurrentHiddenUnitBuffer(SCHEDULE_ID), tmp_tensor, rank, is_prefill);

  // Test FreeHiddenUnits
  Status free_status = FreeHiddenUnits(SCHEDULE_ID);
  EXPECT_TRUE(free_status.OK());

  GetScheduleOutputPool()->FreeScheduleOutput(schedule_output);

  DestroyScheduleOutputPool();
  DestroyHiddenUnitBufferPool();
}
