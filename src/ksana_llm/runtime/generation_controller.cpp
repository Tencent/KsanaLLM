/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/generation_controller.h"

#include "ksana_llm/profiler/reporter.h"

namespace ksana_llm {

void GenerationController::InitGenerationState(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  if (!structured_generator_factory_) {
    for (auto& req : reqs) {
      req->structured_generator = nullptr;
    }
    return;
  }

  for (auto& req : reqs) {
    if (!req->structured_generator_config.HasConstraint()) {
      continue;
    }

    try {
      auto structured_generator = structured_generator_factory_->CreateGenerator(req->structured_generator_config);
      if (!structured_generator) {
        KLLM_LOG_WARNING << "Failed to create structured generator for request " << req->req_id;
        continue;
      }

      req->structured_generator = std::move(structured_generator);
      KLLM_LOG_DEBUG << "Structured generator created successfully for request " << req->req_id;
    } catch (const std::exception& e) {
      KLLM_LOG_WARNING << "Failed to create structured generator for request " << req->req_id << ": " << e.what();
    }
  }
}

void GenerationController::UpdateGenerationState(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  UpdateGrammarState(reqs);
  FilterDraftTokens(reqs);
}

void GenerationController::UpdateGrammarState(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  for (auto& req : reqs) {
    if (!req->structured_generator || req->sampling_result_tokens.empty()) {
      continue;
    }

    if (req->structured_generator->IsTerminated()) {
      // Note: The request termination should be handled by the caller
      KLLM_LOG_DEBUG << "Structured generation completed for request " << req->req_id;
    }

    int token_id = req->sampling_result_tokens.front();
    bool accepted = req->structured_generator->AcceptToken(token_id);

    if (!accepted) {
      // In production, this should rarely happen if the mask was applied correctly
      KLLM_LOG_WARNING << "Structured constraint rejected token " << token_id << " for request " << req->req_id;
    }
    req->generated_token = token_id;
  }
}

void GenerationController::FilterDraftTokens(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  std::vector<std::pair<size_t, size_t>> hit_count;  // hit_num, draft_num
  for (auto& req : reqs) {
    if (req->sampling_result_tokens.size() - kStepGenerateTokenNum != req->draft_tokens.size()) {
      KLLM_LOG_ERROR << fmt::format(
          "req {} sampling_result_tokens.size = {}, mtp_draft_tokens.size = {}, trie_draft_tokens.size = {}",
          req->req_id, req->sampling_result_tokens.size(), req->draft_tokens.mtp.size(), req->draft_tokens.trie.size());
      continue;
    }
    // Check which tokens are predicted correctly.
    size_t draft_hit_num = 0;
    std::vector<int> draft_tokens = req->draft_tokens.GetDraftTokens();
    for (size_t i = 0; i < draft_tokens.size(); ++i, ++draft_hit_num) {
      if (req->sampling_result_tokens[i] != draft_tokens[i]) {
        break;
      }

      // Structured constraint check for the new generated token
      if (req->structured_generator != nullptr) {
        const int new_token = req->sampling_result_tokens[i + 1];
        if (!req->structured_generator->AcceptToken(new_token)) {
          // Structured constraint rejects the new_token
          KLLM_LOG_DEBUG << "Structured constraint rejected new_token " << new_token << " for request " << req->req_id
                         << ", will not use it as generated_token";
          break;
        }
      }

      // stop if stop token
      if (std::find(req->sampling_config.stop_token_ids.begin(), req->sampling_config.stop_token_ids.end(),
                    draft_tokens[i]) != req->sampling_config.stop_token_ids.end()) {
        break;
      }
    }

    hit_count.resize(std::max(req->draft_tokens.size(), hit_count.size()));
    for (size_t i = 0; i < req->draft_tokens.size(); ++i) {
      auto& [hit_num, draft_num] = hit_count[i];
      ++draft_num;
      if (i < draft_hit_num) {
        ++hit_num;
      }
    }

    req->accepted_tokens.swap(draft_tokens);
    req->accepted_tokens.resize(draft_hit_num);
    // Delete logprobs of tokens that were not accepted.
    size_t unaccepted_num = req->sampling_result_tokens.size() - kStepGenerateTokenNum - draft_hit_num;
    for (size_t i = 0; i < unaccepted_num; ++i) {
      if (!req->logprobs.empty()) req->logprobs.pop_back();
    }
    req->generated_token = req->sampling_result_tokens[draft_hit_num];  // only kStepGenerateTokenNum(1) token now
    req->sampling_result_tokens.clear();
    req->draft_tokens.clear();
  }

  if (hit_count.empty()) {
    return;
  }

  // Log every 10 seconds
  constexpr size_t kReportIntervalMs = 10000;
  static IntervalLogger interval_logger(kReportIntervalMs);
  const bool should_log = interval_logger.ShouldLog();

  size_t total_hit_num = 0, total_draft_num = 0;
  for (size_t i = 0; i < hit_count.size(); ++i) {
    const auto& [hit_num, draft_num] = hit_count[i];
    total_hit_num += hit_num;
    total_draft_num += draft_num;
    if (should_log) {
      KLLM_LOG_INFO << fmt::format("Draft index {} hit rate: {} / {} = {:.3f}%", i, hit_num, draft_num,
                                   static_cast<double>(hit_num) / static_cast<double>(draft_num) * 100.0);
    } else {
      KLLM_LOG_DEBUG << fmt::format("Draft index {} hit rate: {} / {} = {:.3f}%", i, hit_num, draft_num,
                                    static_cast<double>(hit_num) / static_cast<double>(draft_num) * 100.0);
    }
  }

  // Report
  REPORT_METRIC("spec_draft_hit_num", total_hit_num);
  REPORT_METRIC("spec_draft_token_num", total_draft_num);
}

}  // namespace ksana_llm
