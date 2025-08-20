/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/
#include "ksana_llm/model_performance/model_performance_runner.h"
#include "ksana_llm/model_performance/perf_profile_config_builder_for_csv.h"
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
    std::filesystem::path profile_csv_config_relate = parent_path / "test_config.csv";
    std::string profile_csv_config_path = std::filesystem::absolute(profile_csv_config_relate).string();
    size_t warmup_round = 10;
    size_t profile_round = 100;
    config_builder_ = std::make_shared<ksana_llm::PerfProfileConfigBuilderWithCsv>(profile_csv_config_path,
                                                                                   warmup_round, profile_round);
    model_performance_runner_ =
        std::make_shared<ksana_llm::ModelPerformanceRunner>(config_path, config_builder_->GetMaxPerfProfileConfig());
    config_builder_->SetAttnDpNum(model_performance_runner_->GetAttnDpNum());
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<ksana_llm::PerfProfileConfigBuilderWithCsv> config_builder_;
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
  static constexpr size_t expect_multi_token_request_num = 2;
  static constexpr size_t expect_single_token_request_num = 2;
  auto& req_config = max_config.req_configs[0];
  EXPECT_EQ(expect_single_token_request_num, req_config.single_token_request_num);
  EXPECT_EQ(expect_multi_token_request_num, req_config.multi_token_request_num);
  // test multi token request
  size_t single_token_req_idx = 0;
  EXPECT_EQ(infer_reqs[single_token_req_idx]->forwarding_tokens.size(), req_config.multi_token_request_token_num);
  // test single token request
  size_t multi_token_req_idx = expect_multi_token_request_num;
  EXPECT_EQ(infer_reqs[multi_token_req_idx]->kv_cached_token_num, req_config.single_token_request_cached_token_num);
  EXPECT_EQ(infer_reqs[multi_token_req_idx]->forwarding_tokens.size(),
            req_config.single_token_request_cached_token_num + 1);
  // test random input_token
  EXPECT_NE(infer_reqs[0]->forwarding_tokens[0], infer_reqs[1]->forwarding_tokens[0]);
}
