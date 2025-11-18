/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/draft_generator/draft_generator_controller.h"

namespace ksana_llm {

Status DraftGeneratorController::AppendDraftGenerator(std::shared_ptr<DraftGeneratorInterface> generator) {
  draft_generators_.emplace_back(generator);
  return Status();
}

Status DraftGeneratorController::GenerateDraft(std::shared_ptr<InferRequest> req) {
  for (auto& generator : draft_generators_) {
    generator->GenerateDraft(req);
  }
  return Status();
}

}  // namespace ksana_llm