/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {
class DraftGeneratorInterface {
 public:
  virtual ~DraftGeneratorInterface() {}
  virtual void GenerateDraft(const std::vector<int>& input_tokens, int step, int suggested_draft_num,
                             std::vector<int>& draft_tokens, int unverfied_token_num, int accepted_tokens,
                             int req_id) = 0;
};
}  // namespace ksana_llm