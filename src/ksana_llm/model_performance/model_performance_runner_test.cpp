/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/
#include "ksana_llm/model_performance/model_performance_runner.h"
#include "ksana_llm/model_performance/perf_profile_config_builder_for_json.h"
#include "ksana_llm/utils/dynamic_memory_pool.h"
#include "tests/test.h"

using namespace ksana_llm;

class ModelPerformanceRunnerTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    DeviceMemoryPool::Disable();

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm_tp.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();
    std::filesystem::path profile_json_config_relate = parent_path / "test_config.json";
    std::string profile_json_config_path = std::filesystem::absolute(profile_json_config_relate).string();
    size_t warmup_round = 10;
    size_t profile_round = 100;
    config_builder_ = std::make_shared<ksana_llm::PerfProfileConfigBuilderWithJson>(profile_json_config_path,
                                                                                    warmup_round, profile_round);
    model_performance_runner_ =
        std::make_shared<ksana_llm::ModelPerformanceRunner>(config_path, config_builder_->GetMaxPerfProfileConfig());
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<ksana_llm::PerfProfileConfigBuilderWithJson> config_builder_;
  std::shared_ptr<ksana_llm::ModelPerformanceRunner> model_performance_runner_ = nullptr;
};

TEST_F(ModelPerformanceRunnerTest, Test) {
  auto max_config = config_builder_->GetMaxPerfProfileConfig();
  // test run
  PerfProfileResult result;
  Status status = model_performance_runner_->RunPerformanceForward(max_config, result);
  EXPECT_EQ(max_config.config_id, result.config_id);
  EXPECT_TRUE(status.OK());

  // Check inputs
  const auto& input_ids_map = model_performance_runner_->input_ids_map_;
  const std::vector<std::shared_ptr<InferRequest>>& infer_reqs = model_performance_runner_->infer_reqs_;
  EXPECT_EQ(infer_reqs.size(), input_ids_map.size());

  // With current test_config.json having only dp_0:
  // - dp_0: 2 RequestInfo (request_num=2,1) → 3 InferRequests (2+1)
  // Total: 3 InferRequests
  static constexpr size_t expected_expanded_requests = 3;     // Total after expanding request_num
  static constexpr size_t expected_dp_configs = 1;            // Number of DP configs
  static constexpr size_t expected_request_infos_per_dp = 2;  // RequestInfo count per DP

  EXPECT_EQ(expected_expanded_requests, infer_reqs.size());

  // Check the max_config should have 1 DP config
  ASSERT_FALSE(max_config.req_configs.empty());
  EXPECT_EQ(expected_dp_configs, max_config.req_configs.size());

  // Check dp_0 config
  auto& req_config_dp0 = max_config.req_configs[0];
  EXPECT_EQ(expected_request_infos_per_dp, req_config_dp0.reqs.size());
  EXPECT_EQ(128, req_config_dp0.reqs[0].forwarding_token_num);
  EXPECT_EQ(192, req_config_dp0.reqs[0].sequence_len);
  EXPECT_EQ(2, req_config_dp0.reqs[0].request_num);
  EXPECT_EQ(128, req_config_dp0.reqs[1].forwarding_token_num);
  EXPECT_EQ(192, req_config_dp0.reqs[1].sequence_len);
  EXPECT_EQ(1, req_config_dp0.reqs[1].request_num);

  // Test that requests have different random input tokens
  EXPECT_NE(infer_reqs[0]->forwarding_tokens[0], infer_reqs[1]->forwarding_tokens[0]);
}
