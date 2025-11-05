/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "ksana_llm/utils/reasoning_config.h"

#include "gflags/gflags.h"
#include "test.h"

#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

class ReasoningConfigTest : public testing::Test {
 protected:
  void SetUp() override {
    env_ = Singleton<Environment>::GetInstance();
  }

  std::shared_ptr<Environment> env_;
};

TEST_F(ReasoningConfigTest, ValidThinkEndTokenId) {
  // Test with a valid think_end_token_id
  int test_token_id = 12345;

  ReasoningConfigManager reasoning_config_manager(test_token_id, env_);

  // Verify the configuration was applied
  ksana_llm::ReasoningConfig retrieved_config;
  env_->GetReasoningConfig(retrieved_config);
  EXPECT_EQ(retrieved_config.think_end_token_id, test_token_id);
}

TEST_F(ReasoningConfigTest, DisabledReasoningMode) {
  // Test with -1 (disabled reasoning mode)
  int test_token_id = -1;
  ksana_llm::ReasoningConfig original_config;
  env_->GetReasoningConfig(original_config);
  ReasoningConfigManager reasoning_config_manager(test_token_id, env_);

  // Configuration should not change when disabled
  ksana_llm::ReasoningConfig retrieved_config;
  env_->GetReasoningConfig(retrieved_config);
  EXPECT_EQ(retrieved_config.think_end_token_id, original_config.think_end_token_id);
}

}  // namespace ksana_llm
