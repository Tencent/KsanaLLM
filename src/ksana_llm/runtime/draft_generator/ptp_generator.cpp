/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/runtime/draft_generator/ptp_generator.h"

namespace ksana_llm {

PtpGenerator::PtpGenerator(size_t ptp_step_num, uint32_t ptp_token_id)
    : ptp_step_num_(ptp_step_num), ptp_token_id_(ptp_token_id) {}

void PtpGenerator::GenerateDraft(std::shared_ptr<InferRequest> req) {
  for (size_t i = 0; i < ptp_step_num_; ++i) {
    req->draft_tokens.ptp.emplace_back(ptp_token_id_);
  }
}

}  // namespace ksana_llm