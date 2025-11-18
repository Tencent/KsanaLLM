/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {
class DraftGeneratorInterface {
 public:
  virtual ~DraftGeneratorInterface() {}
  virtual void GenerateDraft(std::shared_ptr<InferRequest> req) = 0;
};
}  // namespace ksana_llm