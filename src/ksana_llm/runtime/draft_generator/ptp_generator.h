/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/draft_generator/draft_generator_interface.h"

namespace ksana_llm {

class PtpGenerator : public DraftGeneratorInterface {
 public:
  PtpGenerator(size_t ptp_step_num, uint32_t ptp_token_id);

  void GenerateDraft(std::shared_ptr<InferRequest> req) override;

 private:
  size_t ptp_step_num_ = 0;
  uint32_t ptp_token_id_ = 0;
};

}  // namespace ksana_llm