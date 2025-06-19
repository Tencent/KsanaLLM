/* Copyright 2024 Tencent Inc.  All rights reserved.
 * ==============================================================================*/

#include "ksana_llm/utils/stop_checker.h"

#include "tests/test.h"

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tokenizer.h"

namespace ksana_llm {

class StopCheckerTest : public testing::Test {
 protected:
  void SetUp() override {
    Singleton<Tokenizer>::GetInstance()->InitTokenizer("/model/llama-hf/7B");

    std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
    std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx =
        std::make_shared<std::unordered_map<std::string, std::string>>();

    request_ = std::make_shared<Request>(ksana_python_input, req_ctx);
    infer_req_ = std::make_shared<InferRequest>(request_, 0);
  }
  void TearDown() override { Singleton<Tokenizer>::GetInstance()->DestroyTokenizer(); }

 protected:
  // Used to initialize infer request.
  std::shared_ptr<Request> request_;
  std::shared_ptr<InferRequest> infer_req_;
  std::shared_ptr<StopChecker> stop_checker_;
};

TEST_F(StopCheckerTest, IncrementalStopStringsTest) {
  infer_req_->sampling_config.stop_strings = {"Shakespeare"};
  Singleton<Tokenizer>::GetInstance()->Encode("My name", infer_req_->input_tokens, true);
  Singleton<Tokenizer>::GetInstance()->Encode("My name is William Shakespeare", infer_req_->output_tokens, true);

  std::vector<int> expected_token_ids = {1, 1619, 1024, 338, 4667, 23688};
  ASSERT_EQ(infer_req_->output_tokens, expected_token_ids);
  stop_checker_->CheckIncrementalStopStrings(infer_req_);
  // expected_token_ids = ["My", "name"]
  expected_token_ids = {1, 1619, 1024};
  ASSERT_EQ(infer_req_->output_tokens, expected_token_ids);
}

TEST_F(StopCheckerTest, CompleteStopStringsTest) {
  infer_req_->sampling_config.stop_strings = {"William"};
  Singleton<Tokenizer>::GetInstance()->Encode("My name", infer_req_->input_tokens, true);
  Singleton<Tokenizer>::GetInstance()->Encode("My name is William Shakespeare", infer_req_->output_tokens, true);
  std::vector<int> expected_token_ids = {1, 1619, 1024, 338, 4667, 23688};
  ASSERT_EQ(infer_req_->output_tokens, expected_token_ids);
  stop_checker_->CheckCompleteStopStrings(infer_req_);
  // expected_token_ids = ["My", "name", "is", " "]
  expected_token_ids = {1, 1619, 1024, 338, 29871};
  ASSERT_EQ(infer_req_->output_tokens, expected_token_ids);
}

}  // namespace ksana_llm
