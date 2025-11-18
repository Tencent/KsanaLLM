/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/draft_generator/draft_generator_interface.h"

namespace ksana_llm {

class DraftGeneratorController {
 public:
  Status AppendDraftGenerator(std::shared_ptr<DraftGeneratorInterface> generator);

  Status GenerateDraft(std::shared_ptr<InferRequest> req);

 private:
  std::vector<std::shared_ptr<DraftGeneratorInterface>> draft_generators_;
};

}  // namespace ksana_llm