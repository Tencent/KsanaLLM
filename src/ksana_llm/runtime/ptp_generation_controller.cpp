/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/ptp_generation_controller.h"

#include "ksana_llm/profiler/reporter.h"

namespace ksana_llm {

PTPGenerationController::PTPGenerationController(
    std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory)
    : GenerationController(nullptr) {
  // TODO(zezhao): Async Controller is not yet supported in Parallel-Token-Predict mode.
  if (structured_generator_factory) {
    // TODO(zezhao): Structured generation is not yet supported in Parallel-Token-Predict mode.
    KLLM_LOG_WARNING << "Parallel-Token-Predict does not support structured generation yet.";
  }
}

void PTPGenerationController::UpdateGrammarState(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  for (auto& req : reqs) {
    req->generated_tokens.clear();
    // if no sampling result or is chunked prefill task, don't generate
    // chunked prefill task has sampling_token_num=1 to avoid empty tensor in LmHead()
    if (req->sampling_result_tokens.empty() || (req->forwarding_tokens.size() < req->output_tokens.size())) {
      continue;
    }

    req->generated_tokens = req->sampling_result_tokens;
  }
}

void PTPGenerationController::FilterDraftTokens(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  for (auto& req : reqs) {
    // In PTP mode we keep none of the draft tokens.
    req->accepted_tokens.clear();
    req->draft_tokens.clear();
  }
}

}  // namespace ksana_llm
