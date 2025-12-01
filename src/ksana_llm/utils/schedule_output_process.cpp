/* Copyright 2025 Tencent Inc.
==============================================================================*/

#include "ksana_llm/utils/schedule_output_process.h"

namespace ksana_llm {

void MergeScheduleOutputGroupRunningRequests(std::shared_ptr<ScheduleOutputGroup>& schedule_output_group,
                                             ScheduleOutput& merged_schedule_output) {
  const size_t outputs_size = schedule_output_group->outputs.size();
  if (outputs_size == 0) {
    return;
  }

  merged_schedule_output.multi_batch_id = schedule_output_group->schedule_id;

  size_t running_reqs_reserve_size = 0;
  size_t worker_running_reqs_reserve_size = 0;
  for (size_t attn_dp_idx = 0; attn_dp_idx < outputs_size; ++attn_dp_idx) {
    ScheduleOutput* schedule_output = schedule_output_group->outputs.at(attn_dp_idx);
    if (schedule_output == nullptr) {
      continue;
    }

    running_reqs_reserve_size += schedule_output->running_reqs.size();
    worker_running_reqs_reserve_size += schedule_output->worker_running_reqs.size();
  }

  merged_schedule_output.running_reqs.reserve(running_reqs_reserve_size);
  merged_schedule_output.worker_running_reqs.reserve(worker_running_reqs_reserve_size);

  for (size_t attn_dp_idx = 0; attn_dp_idx < outputs_size; ++attn_dp_idx) {
    ScheduleOutput* schedule_output = schedule_output_group->outputs.at(attn_dp_idx);
    if (schedule_output == nullptr) {
      continue;
    }

    for (auto& req : schedule_output->running_reqs) {
      req->attn_dp_group_id = attn_dp_idx;
    }
    for (auto& req : schedule_output->worker_running_reqs) {
      req->attn_dp_group_id = attn_dp_idx;
    }

    merged_schedule_output.running_reqs.insert(merged_schedule_output.running_reqs.end(),
                                               schedule_output->running_reqs.begin(),
                                               schedule_output->running_reqs.end());
    merged_schedule_output.worker_running_reqs.insert(merged_schedule_output.worker_running_reqs.end(),
                                                      schedule_output->worker_running_reqs.begin(),
                                                      schedule_output->worker_running_reqs.end());
  }
}

void MergeScheduleInfoForWorkers(const std::vector<ScheduleOutput*>& outputs, ScheduleOutput& merged_schedule_output) {
  const size_t outputs_size = outputs.size();
  if (outputs_size == 0) {
    return;
  }

  merged_schedule_output.finish_req_ids.resize(outputs_size);
  merged_schedule_output.merged_swapout_req_ids.resize(outputs_size);
  merged_schedule_output.merged_swapin_req_ids.resize(outputs_size);
  merged_schedule_output.swapout_req_block_ids.resize(outputs_size);
  merged_schedule_output.swapin_req_block_ids.resize(outputs_size);

  for (size_t attn_dp_idx = 0; attn_dp_idx < outputs_size; ++attn_dp_idx) {
    ScheduleOutput* schedule_output = outputs.at(attn_dp_idx);
    if (schedule_output == nullptr) {
      continue;
    }

    if (!schedule_output->finish_req_ids.empty()) {
      merged_schedule_output.finish_req_ids[attn_dp_idx] = schedule_output->finish_req_ids[0];
    }
    if (!schedule_output->merged_swapout_req_ids.empty()) {
      merged_schedule_output.merged_swapout_req_ids[attn_dp_idx] = schedule_output->merged_swapout_req_ids[0];
    }
    if (!schedule_output->merged_swapin_req_ids.empty()) {
      merged_schedule_output.merged_swapin_req_ids[attn_dp_idx] = schedule_output->merged_swapin_req_ids[0];
    }
    if (!schedule_output->swapout_req_block_ids.empty()) {
      merged_schedule_output.swapout_req_block_ids[attn_dp_idx] = schedule_output->swapout_req_block_ids[0];
    }
    if (!schedule_output->swapin_req_block_ids.empty()) {
      merged_schedule_output.swapin_req_block_ids[attn_dp_idx] = schedule_output->swapin_req_block_ids[0];
    }
    schedule_output->ResetDataForWorkers();
  }
}

}  // namespace ksana_llm
