/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/generation_controller.h"

namespace ksana_llm {

class PTPGenerationController : public GenerationController {
 public:
  explicit PTPGenerationController(std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory);

 private:
  void UpdateGrammarState(std::vector<std::shared_ptr<InferRequest>> &reqs) override;

  void FilterDraftTokens(std::vector<std::shared_ptr<InferRequest>> &reqs) override;

  int ptp_step_num_;
  int ptp_token_id_;
};

}  // namespace ksana_llm
