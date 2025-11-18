/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/draft_generator/draft_generator_controller.h"
#include <gtest/gtest.h>
#include "ksana_llm/runtime/draft_generator/ptp_generator.h"

namespace ksana_llm {

class DraftGeneratorControllerTest : public testing::Test {
 protected:
  void SetUp() override {
    draft_generator_controller_ = std::make_shared<DraftGeneratorController>();
  }
 private:
  std::shared_ptr<DraftGeneratorController> draft_generator_controller_;
};

TEST_F(DraftGeneratorControllerTest, PTPGeneratorTest) {
  // 验证 Append 功能
  draft_generator_controller_->AppendDraftGenerator(std::make_shared<PtpGenerator>(3, 1234));
  draft_generator_controller_->AppendDraftGenerator(std::make_shared<PtpGenerator>(2, 2345));
  EXPECT_EQ(draft_generator_controller_->draft_generators_.size(), 2);

  // 验证顺序 Generator 推理
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto infer_request = std::make_shared<InferRequest>(request, 0);
  infer_request->draft_tokens.clear();
  draft_generator_controller_->GenerateDraft(infer_request);
  std::vector<int> target_draft_tokens = {1234, 1234, 1234, 2345, 2345};
  std::vector<int> draft_tokens = infer_request->draft_tokens.GetDraftTokens();
  EXPECT_EQ(draft_tokens.size(), target_draft_tokens.size());
  for (int i = 0; i < target_draft_tokens.size(); ++i) {
    EXPECT_EQ(draft_tokens[i], target_draft_tokens[i]);
  }
}

}  // namespace ksana_llm