/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/batch_scheduler_balance_reqs_algo.h"

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

void BalanceReqsAlgo::BalanceReqs(const std::vector<float>& workloads,
                                  std::vector<std::pair<size_t, std::shared_ptr<InferRequest>>>& tokens_to_req_pairs,
                                  std::vector<std::vector<std::shared_ptr<InferRequest>>>& outputs_reqs) {
  if (tokens_to_req_pairs.empty() || workloads.empty()) {
    return;
  }

  outputs_reqs.resize(workloads.size());
  for (auto& reqs : outputs_reqs) {
    reqs.clear();
  }

  // 按照token数量从大到小排序请求
  std::sort(tokens_to_req_pairs.begin(), tokens_to_req_pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  size_t total_tokens = 0;
  for (const auto& req : tokens_to_req_pairs) {
    total_tokens += req.first;
  }

  std::vector<float> current_workloads(workloads.size(), 0.0);
  for (size_t i = 0; i < workloads.size(); ++i) {
    current_workloads[i] = workloads[i] * total_tokens / tokens_to_req_pairs.size();
  }
  // 分配每个请求到负载最小的组
  for (const auto& req : tokens_to_req_pairs) {
    auto min_it = std::min_element(current_workloads.begin(), current_workloads.end());
    int min_idx = std::distance(current_workloads.begin(), min_it);

    KLLM_LOG_SCHEDULER << "[ Group " << min_idx << " add 1 req, req id: " << req.second->req_id
                       << ", req_tokens: " << req.first << ", workload: " << current_workloads[min_idx] << " -> "
                       << current_workloads[min_idx] + req.first << " ] ";
    current_workloads[min_idx] += 10000 * req.first;
    outputs_reqs[min_idx].push_back(req.second);
  }

  std::stringstream ss;
  for (size_t i = 0; i < current_workloads.size(); ++i) {
    ss << "[ Group " << i << ": input workloads=" << workloads.at(i) << ", result_tokens=" << current_workloads.at(i)
       << ", reqs=" << outputs_reqs.at(i).size() << " ] ";
  }
  KLLM_LOG_SCHEDULER << "BalanceAlgo distribution:" << ss.str();
}

}  // namespace ksana_llm
