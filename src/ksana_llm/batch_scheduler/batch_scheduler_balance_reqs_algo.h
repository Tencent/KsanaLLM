/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <vector>

#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {

class BalanceReqsAlgo {
 public:
  BalanceReqsAlgo() = default;
  ~BalanceReqsAlgo() = default;

  // Balance algorithm to distribute requests based on workloads
  void BalanceReqs(const std::vector<float>& workloads,
                   std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>>& tokens_to_req_pairs,
                   std::vector<std::vector<std::shared_ptr<InferRequest>>>& outputs_reqs);
};

}  // namespace ksana_llm
