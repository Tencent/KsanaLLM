/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/
#include "ksana_llm/model_performance/model_performance_runner.h"
#include "tests/test.h"

using namespace ksana_llm;

class ModelPerformanceRunnerTest : public testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm_performance_run.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();
    model_performance_runner_ = std::make_shared<ksana_llm::ModelPerformanceRunner>(config_path);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<ksana_llm::ModelPerformanceRunner> model_performance_runner_ = nullptr;
};

TEST_F(ModelPerformanceRunnerTest, Test) {
  // test input
  const std::vector<std::vector<int>>& input_ids_vec = model_performance_runner_->input_ids_vec_;
  const std::vector<ForwardRequest>& forward_reqs = model_performance_runner_->forward_reqs_;
  EXPECT_EQ(forward_reqs.size(), input_ids_vec.size());
  static constexpr size_t expect_multi_token_request_num = 2;
  static constexpr size_t expect_single_token_request_num = 2;
  EXPECT_EQ(expect_single_token_request_num, model_performance_runner_->single_token_request_num_);
  EXPECT_EQ(expect_multi_token_request_num, model_performance_runner_->multi_token_request_num_);
  // test multi token request
  size_t single_token_req_idx = 0;
  EXPECT_EQ(forward_reqs[single_token_req_idx].forwarding_tokens->size(),
            model_performance_runner_->multi_token_request_token_num_);
  // test single token request
  size_t multi_token_req_idx = expect_multi_token_request_num;
  EXPECT_EQ(forward_reqs[multi_token_req_idx].kv_cached_token_num,
            model_performance_runner_->single_token_request_cached_token_num_);
  EXPECT_EQ(forward_reqs[multi_token_req_idx].forwarding_tokens->size(),
            model_performance_runner_->single_token_request_cached_token_num_ + 1);
  // test random input_token
  EXPECT_NE(input_ids_vec[0][0], input_ids_vec[1][0]);

  // test run
  Status status = model_performance_runner_->RunPerformanceForward();
  EXPECT_TRUE(status.OK());
}
