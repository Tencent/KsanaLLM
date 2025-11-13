/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_scheduler/state/batch_state.h"

namespace ksana_llm {

class PPMultibatchWorkloadBalancer {
 public:
  explicit PPMultibatchWorkloadBalancer(PPMultibatchWBStrategy pp_multibatch_wb_strategy);

  // Balance requests between batch states, considering waiting_reqs
  void BalancePPMultiBatchReqs(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& waiting_reqs,
                               std::vector<std::shared_ptr<BatchState>>& batch_states);

 private:
  // Distribute waiting requests to multiple batch states based on capacity
  void DistributeWaitingReqs(std::vector<std::shared_ptr<InferRequest>>& waiting_reqs,
                             std::vector<std::shared_ptr<BatchState>>& batch_states);

  // Distribute decoding requests to multiple batch states based on capacity
  void DistributeDecodingReqs(std::vector<std::shared_ptr<InferRequest>>& decoding_reqs,
                              std::vector<std::shared_ptr<BatchState>>& batch_states);

  // Offload batch load
  void OffloadBatchWorkload(size_t multi_batch_id, std::vector<std::shared_ptr<BatchState>>& batch_states);

  // Calculate workload for a single request
  int CalculateWorkload(const std::shared_ptr<InferRequest>& req);

  // Calculate workload for requests
  int CalculateWorkloadForReqs(std::vector<std::shared_ptr<InferRequest>>& reqs);

 private:
  PPMultibatchWBStrategy pp_multibatch_wb_strategy_;
  float threshold_factor_;
};

}  // namespace ksana_llm
