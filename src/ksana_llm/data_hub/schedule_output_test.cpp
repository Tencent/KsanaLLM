/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <vector>
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

using namespace ksana_llm;

class ScheduleOutputTest : public testing::Test {
 protected:
  void SetUp() override { schedule_output = new ScheduleOutput(); }

  void TearDown() override { delete schedule_output; }

  void GenerateRequest() {
    std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
    ksana_python_input->model_name = "schedule_output_test";
    ksana_python_input->input_tokens = {1001, 1003, 1005};

    std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx =
        std::make_shared<std::unordered_map<std::string, std::string>>();

    request = std::make_shared<Request>(ksana_python_input, req_ctx);
  }

  void GenerateInferRequest() {
    infer_req = std::make_shared<InferRequest>(request, 0);

    infer_req->req_id = 123;
    infer_req->model_name = "schedule_output_test";
    infer_req->output_tokens = {1001, 1003, 1005};
    infer_req->infer_stage = InferStage::kContext;
    infer_req->step = 3;
    infer_req->kv_cache_blocks = {{101, 103, 105}, {101, 103, 105}};
    infer_req->is_use_prefix_cache = false;
    infer_req->prefix_cache_len = 10;
  }

  bool CheckInferRequest(std::shared_ptr<InferRequest> infer_req,
                         std::shared_ptr<WorkerInferRequest> worker_infer_req) {
    if (infer_req->req_id != worker_infer_req->req_id) {
      return false;
    }

    if (infer_req->model_name != worker_infer_req->model_name) {
      return false;
    }

    if (infer_req->forwarding_tokens != worker_infer_req->forwarding_tokens) {
      return false;
    }

    if (infer_req->infer_stage != worker_infer_req->infer_stage) {
      return false;
    }

    if (infer_req->step != worker_infer_req->step) {
      return false;
    }

    if (infer_req->kv_cache_blocks != worker_infer_req->kv_cache_blocks) {
      return false;
    }

    if (infer_req->is_use_prefix_cache != worker_infer_req->is_use_prefix_cache) {
      return false;
    }

    if (infer_req->prefix_cache_len != worker_infer_req->prefix_cache_len) {
      return false;
    }

    return true;
  }

  bool CheckScheduleOutput(ScheduleOutput* src_schedule_output, ScheduleOutput* dst_schedule_output) {
    if (src_schedule_output->multi_batch_id != dst_schedule_output->multi_batch_id) {
      return false;
    }

    if (!CheckVectorsVector(src_schedule_output->finish_req_ids, dst_schedule_output->finish_req_ids)) {
      return false;
    }

    if (!CheckVectorsVector(src_schedule_output->merged_swapout_req_ids, dst_schedule_output->merged_swapout_req_ids)) {
      return false;
    }

    if (!CheckVectorsVector(src_schedule_output->merged_swapin_req_ids, dst_schedule_output->merged_swapin_req_ids)) {
      return false;
    }

    if (!CheckVectorsKeyToVector(src_schedule_output->swapout_req_block_ids,
                                 dst_schedule_output->swapout_req_block_ids)) {
      return false;
    }

    if (!CheckVectorsKeyToVector(src_schedule_output->swapin_req_block_ids,
                                 dst_schedule_output->swapin_req_block_ids)) {
      return false;
    }

    if (src_schedule_output->running_reqs.size() != dst_schedule_output->worker_running_reqs.size()) {
      return false;
    }

    for (size_t i = 0; i < src_schedule_output->running_reqs.size(); ++i) {
      if (!CheckInferRequest(src_schedule_output->running_reqs[i], dst_schedule_output->worker_running_reqs[i])) {
        return false;
      }
    }

    return true;
  }

  // Check whether two vector<vector> is equal.
  template <typename T>
  bool CheckVectorsVector(const std::vector<std::vector<T>>& src_vec, const std::vector<std::vector<T>>& dst_vec) {
    if (src_vec.size() != dst_vec.size()) {
      return false;
    }

    for (size_t i = 0; i < src_vec.size(); ++i) {
      if (!CheckVector(src_vec[i], dst_vec[i])) {
        return false;
      }
    }

    return true;
  }

  // Check whether two vector is equal.
  template <typename T>
  bool CheckVector(const std::vector<T>& src_vec, const std::vector<T>& dst_vec) {
    if (src_vec.size() != dst_vec.size()) {
      return false;
    }

    for (size_t i = 0; i < src_vec.size(); ++i) {
      if (src_vec[i] != dst_vec[i]) {
        return false;
      }
    }

    return true;
  }

  bool CheckVectorsKeyToVector(const std::vector<std::unordered_map<int64_t, std::vector<int>>>& src_dict,
                               const std::vector<std::unordered_map<int64_t, std::vector<int>>>& dst_dict) {
    if (src_dict.size() != dst_dict.size()) {
      return false;
    }

    for (size_t i = 0; i < src_dict.size(); ++i) {
      if (!CheckKeyToVector(src_dict[i], dst_dict[i])) {
        return false;
      }
    }

    return true;
  }

  bool CheckKeyToVector(const std::unordered_map<int64_t, std::vector<int>>& src_dict,
                        const std::unordered_map<int64_t, std::vector<int>>& dst_dict) {
    if (src_dict.size() != dst_dict.size()) {
      return false;
    }

    for (auto it = src_dict.begin(); it != src_dict.end(); ++it) {
      int64_t key = it->first;
      if (dst_dict.find(key) == dst_dict.end()) {
        return false;
      }

      const std::vector<int>& src_vec = it->second;
      const std::vector<int>& dst_vec = dst_dict.at(key);
      if (!CheckVector(src_vec, dst_vec)) {
        return false;
      }
    }

    return true;
  }

  bool CheckVectorOfVector(const std::vector<std::vector<int>>& src_vecs,
                           const std::vector<std::vector<int>>& dst_vecs) {
    if (src_vecs.size() != dst_vecs.size()) {
      return false;
    }

    for (size_t i = 0; i < src_vecs.size(); ++i) {
      if (!CheckVector(src_vecs[i], dst_vecs[i])) {
        return false;
      }
    }

    return true;
  }

  void GenerateScheduleOutput() {
    GenerateRequest();
    GenerateInferRequest();

    schedule_output->multi_batch_id = 5;

    schedule_output->finish_req_ids = {{1, 3, 5, 7}};

    schedule_output->merged_swapout_req_ids = {{11, 13, 15}};
    schedule_output->merged_swapin_req_ids = {{12, 14, 16}};

    std::unordered_map<int64_t, std::vector<int>> dp_swapout_req_block_ids;
    dp_swapout_req_block_ids[101] = {111, 113, 115};
    schedule_output->swapout_req_block_ids.push_back(dp_swapout_req_block_ids);
    std::unordered_map<int64_t, std::vector<int>> dp_swapin_req_block_ids;
    dp_swapin_req_block_ids[102] = {112, 114, 116};
    schedule_output->swapin_req_block_ids.push_back(dp_swapin_req_block_ids);

    schedule_output->running_reqs = {infer_req};
  }

 protected:
  // Used to initialize infer request.
  std::shared_ptr<Request> request = nullptr;
  std::shared_ptr<InferRequest> infer_req = nullptr;
  std::shared_ptr<WorkerInferRequest> worker_infer_req = nullptr;

  ScheduleOutput* schedule_output = nullptr;

  int device_num_ = -1;
};

TEST_F(ScheduleOutputTest, SerializeVector) {
  char memory[4096];
  std::vector<int> src_vec = {1, 2, 3, 4, 5};

  size_t serialize_bytes;
  ScheduleOutputParser::SerializeVector(src_vec, memory, serialize_bytes);

  size_t deserialize_bytes;
  std::vector<int> dst_vec;
  ScheduleOutputParser::DeserializeVector(memory, dst_vec, deserialize_bytes);

  EXPECT_EQ(serialize_bytes, deserialize_bytes);
  EXPECT_TRUE(CheckVector(src_vec, dst_vec));
}

TEST_F(ScheduleOutputTest, SerializeKeyToVector) {
  char memory[4096];

  std::unordered_map<int64_t, std::vector<int>> src_dict;
  src_dict[1] = {1, 3, 5, 7};
  src_dict[2] = {2, 4, 6, 8};

  size_t serialize_bytes;
  ScheduleOutputParser::SerializeKeyToVector(src_dict, memory, serialize_bytes);

  size_t deserialize_bytes;
  std::unordered_map<int64_t, std::vector<int>> dst_dict;
  ScheduleOutputParser::DeserializeKeyToVector(memory, dst_dict, deserialize_bytes);

  EXPECT_EQ(serialize_bytes, deserialize_bytes);
  EXPECT_TRUE(CheckKeyToVector(src_dict, dst_dict));
}

TEST_F(ScheduleOutputTest, SerializeVectorOfVector) {
  char memory[4096];

  std::vector<std::vector<int>> src_vecs = {{1, 3, 5, 7}, {2, 4, 6, 8}};

  size_t serialize_bytes;
  ScheduleOutputParser::SerializeVectorOfVector(src_vecs, memory, serialize_bytes);

  size_t deserialize_bytes;
  std::vector<std::vector<int>> dst_vecs;
  ScheduleOutputParser::DeserializeVectorOfVector(memory, dst_vecs, deserialize_bytes);

  EXPECT_EQ(serialize_bytes, deserialize_bytes);
  EXPECT_TRUE(CheckVectorOfVector(src_vecs, dst_vecs));
}

TEST_F(ScheduleOutputTest, SerializeInferRequest) {
  GenerateRequest();
  GenerateInferRequest();

  char memory[4096];
  size_t serialize_bytes;

  std::vector<std::shared_ptr<InferRequest>> src_infer_reqs = {infer_req};
  ScheduleOutputParser::SerializeAsWorkerInferRequest(src_infer_reqs, memory, serialize_bytes);

  size_t deserialize_bytes;
  std::vector<std::shared_ptr<WorkerInferRequest>> dst_infer_reqs;
  ScheduleOutputParser::DeserializeWorkerInferRequest(memory, dst_infer_reqs, deserialize_bytes);

  EXPECT_EQ(serialize_bytes, deserialize_bytes);
  EXPECT_EQ(src_infer_reqs.size(), dst_infer_reqs.size());
  EXPECT_TRUE(CheckInferRequest(src_infer_reqs[0], dst_infer_reqs[0]));
}

TEST_F(ScheduleOutputTest, SerializeScheduleOutput) {
  GenerateScheduleOutput();

  size_t bytes = ScheduleOutputParser::GetSerializedSize(schedule_output);
  void* memory = malloc(bytes);

  ScheduleOutputParser::SerializeScheduleOutput(schedule_output, memory);

  ScheduleOutput* new_schedule_output = new ScheduleOutput();
  ScheduleOutputParser::DeserializeScheduleOutput(memory, new_schedule_output);

  EXPECT_TRUE(CheckScheduleOutput(schedule_output, new_schedule_output));

  free(memory);
  delete new_schedule_output;
}

TEST_F(ScheduleOutputTest, ScheduleOutputPool) {
  InitializeScheduleOutputPool();

  // Get a new output.
  ScheduleOutput* schedule_output = GetScheduleOutputPool()->GetScheduleOutput();
  EXPECT_TRUE(schedule_output != nullptr);

  // Set some value.
  schedule_output->multi_batch_id = 235;

  // Put to recv and get again.
  GetScheduleOutputPool()->PutToRecvQueue(schedule_output);
  ScheduleOutput* recv_schedule_output = GetScheduleOutputPool()->GetFromRecvQueue();
  EXPECT_EQ(schedule_output->multi_batch_id, recv_schedule_output->multi_batch_id);

  // Put to send and get again.
  GetScheduleOutputPool()->PutToSendQueue(schedule_output);
  Packet* send_schedule_output_packet = GetScheduleOutputPool()->GetFromSendQueue();

  ScheduleOutput* new_schedule_output = new ScheduleOutput();
  ScheduleOutputParser::DeserializeScheduleOutput(send_schedule_output_packet->body, new_schedule_output);

  EXPECT_EQ(schedule_output->multi_batch_id, new_schedule_output->multi_batch_id);

  // Free to pool.
  GetScheduleOutputPool()->FreeScheduleOutput(new_schedule_output);
  ScheduleOutput* free_schedule_output = GetScheduleOutputPool()->GetScheduleOutput();
  EXPECT_EQ(schedule_output->multi_batch_id, free_schedule_output->multi_batch_id);

  DestroyScheduleOutputPool();
}
