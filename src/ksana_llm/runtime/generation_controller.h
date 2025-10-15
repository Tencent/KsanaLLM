/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/structured_generation/structured_generator_factory.h"

namespace ksana_llm {

class GenerationController {
 public:
  explicit GenerationController(std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory)
      : structured_generator_factory_(structured_generator_factory) {}

  // Init generation state before sampling.
  // Suggested to init before forwarding, so bitmask can be used in advance
  void InitGenerationState(std::vector<std::shared_ptr<InferRequest>> &reqs);

  // Update generation state after sampling_result_tokens are generated.
  void UpdateGenerationState(std::vector<std::shared_ptr<InferRequest>> &reqs);

 private:
  void UpdateGrammarState(std::vector<std::shared_ptr<InferRequest>> &reqs);

  void FilterDraftTokens(std::vector<std::shared_ptr<InferRequest>> &reqs);

 private:
  std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory_ = nullptr;
};

}  // namespace ksana_llm
