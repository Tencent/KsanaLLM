/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/continuous_batching.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/transfer/transfer_engine.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

/**
 * @brief 为请求队列中的每个请求添加传输元数据
 *
 * 该方法为队列中的每个推理请求添加传输元数据，包括KV缓存块的物理指针和
 * 已缓存的token数量。如果是prefill节点，则将推理token数设为1。
 *
 * @param queue 需要添加传输元数据的请求队列
 */
void ContinuousBatchingStrategy::AddTransferMeta(std::vector<std::shared_ptr<InferRequest>>& queue) {
  auto transfer_engine = TransferEngine::GetInstance();
  for (auto& req : queue) {
    KLLM_LOG_DEBUG << "try GetTransferMeta req id:" << req->kv_comm_request_id;

    // 如果是prefill节点，将推理token数设为1。
    if (connector_config_.group_role == GroupRole::PREFILL) {
      req->sampling_config.max_new_tokens = 1;
    }

    // 如果该请求尚未添加传输元数据，则添加
    if (!transfer_engine->GetTransferMeta(req->kv_comm_request_id)) {
      std::vector<std::vector<void*>> block_ptrs = req->GetBlockPtrs();
      std::vector<int> kv_occupied_devices = req->GetKVOccupiedDevices();
      // block_token_num应该能整除kv_cached_token_num，在为kv_cached_token_num赋值时应保证这点
      if (req->kv_cached_token_num % req->block_token_num != 0) {
        KLLM_THROW(fmt::format("block_token_num: {} should be able to divide kv_cached_token_num: {}",
                               req->block_token_num, req->kv_cached_token_num));
      }
      size_t shared_block_num = req->kv_cached_token_num / req->block_token_num;
      transfer_engine->AddTransferMeta(req->kv_comm_group_key, req->kv_comm_request_id, shared_block_num, block_ptrs,
                                       kv_occupied_devices);
    }
  }
}

/**
 * @brief 处理decode节点的传输队列
 *
 * 该方法检查传输队列中的每个请求，判断是否已接收完成。
 * 如果接收完成，则将请求从传输队列移至运行队列，并更新相关状态。
 */
void ContinuousBatchingStrategy::ProcessDecodeTransferQueue() {
  KLLM_LOG_DEBUG << "ProcessDecodeTransferQueue invoked, transfer queue size:" << batch_state_->transfer_queue.size();
  scheduler_ticktok_->Barrier();
  if (dp_group_id_ == 0) {
    scheduler_ticktok_->Reset();
    scheduler_shared_counter_->step_batch_size.Reset(0);
  }
  scheduler_ticktok_->Barrier();
  scheduler_shared_counter_->step_batch_size.Increase(batch_state_->schedule_output->running_reqs.size());
  scheduler_ticktok_->Barrier();
  if (batch_state_->transfer_queue.empty()) {
    KLLM_LOG_DEBUG << "transfer queue empty, return";
    scheduler_ticktok_->Skip();
    return;
  }
  auto transfer_engine = TransferEngine::GetInstance();
  // 对transfer_queue排序，kv_comm_request_id小的在前
  std::sort(batch_state_->transfer_queue.begin(), batch_state_->transfer_queue.end(),
            [](const auto& a, const auto& b) { return a->kv_comm_request_id < b->kv_comm_request_id; });
  for (auto it = batch_state_->transfer_queue.begin(); it != batch_state_->transfer_queue.end();) {
    auto req = *it;
    // 检查请求是否接收完成，如果完成则返回第一个token，否则返回-1
    std::vector<int> first_tokens = transfer_engine->IsRecvDone(req->kv_comm_request_id);
    if (first_tokens != std::vector<int>(MAX_TRANSFER_TOKENS, -1)) {
      // 检查是否达到最大的batch
      scheduler_ticktok_->Lock();
      size_t step_batch_size = scheduler_shared_counter_->step_batch_size.Get();
      if (step_batch_size >= dp_max_decode_batch_size_) {
        KLLM_LOG_DEBUG << "max batch size reached, stop processing transfer queue";
        scheduler_ticktok_->Unlock();
        break;
      }
      scheduler_shared_counter_->step_batch_size.Increase(1);
      scheduler_ticktok_->Unlock();
      // 接收完成，更新请求状态
      req->kv_cached_token_num = req->forwarding_tokens.size();
      req->prefix_cache_len = req->kv_cached_token_num;
      req->output_tokens.push_back(first_tokens[0]);
      req->generated_token = first_tokens[0];
      KLLM_LOG_DEBUG << "Received first_tokens: " << Vector2Str(first_tokens);

      req->forwarding_tokens.push_back(first_tokens[0]);
      // TODO(winminkong): PD disaggregation supports mutil draft token and speculative decoding.
      if (first_tokens[1] != -1 && runtime_config_.enable_mtp_module) {
        req->draft_tokens.mtp.push_back(first_tokens[1]);
        req->mtp_kv_cached_token_num = req->kv_cached_token_num;
        req->forwarding_tokens.push_back(first_tokens[1]);
      }
      req->forwarding_tokens_draft_num = req->draft_tokens.size();
      req->sampling_token_num =
          req->logits_custom_length > 0 ? req->logits_custom_length : req->draft_tokens.size() + kStepGenerateTokenNum;

      KLLM_LOG_DEBUG << "Decode running_reqs insert for compute, req id:" << req->kv_comm_request_id;
      batch_state_->schedule_output->running_reqs.push_back(req);
      it = batch_state_->transfer_queue.erase(it);
      transfer_engine->CleanupTransferMeta(req->kv_comm_request_id);
    } else {
      // 接收未完成，继续检查下一个请求
      ++it;
    }
  }
  if (batch_state_->schedule_output->running_reqs.size() == 0) {
    KLLM_LOG_DEBUG << "no req in running queue, return";
  }
  scheduler_ticktok_->Skip();
}

/**
 * @brief 处理prefill节点的传输队列
 *
 * 该方法检查传输队列中的每个请求，判断是否已发送完成。
 * 如果发送完成，则将请求从传输队列中移除。
 */
void ContinuousBatchingStrategy::ProcessPrefillTransferQueue() {
  auto transfer_engine = TransferEngine::GetInstance();
  for (auto it = batch_state_->transfer_queue.begin(); it != batch_state_->transfer_queue.end();) {
    // TODO(zakwang): 检查是否有超时和abort，目前强制要求传输完成再释放req
    // 检查请求是否发送完成
    auto req = *it;
    if (transfer_engine->IsSendDone(req->kv_comm_request_id)) {
      // 发送完成，从传输队列中移除该请求
      KLLM_LOG_DEBUG << "Prefill transfer queue erase reqs has computed, req id:" << req->kv_comm_request_id;
      const auto end_time = ProfileTimer::GetCurrentTimeInMs();
      const size_t output_token_num = req->output_tokens.size() - req->input_tokens.size();
      const uint64_t duration = end_time - req->timestamp_in_ms;
      if (req->finish_status.GetCode() == RET_SUCCESS && output_token_num > 0) {
        REPORT_METRIC("total_latency_ms", duration);
        REPORT_METRIC("output_token_len", output_token_num);
        REPORT_METRIC("input_token_len", req->input_tokens.size());
      } else {
        REPORT_METRIC("forward_req_error_num", req->finish_status.GetCode());
      }

      // for async, the kvblock should be destroyed later to avoid the trash token forward error.
      if (batch_scheduler_config_.enable_async) {
        AsyncStopRequest(req, Status(RET_SUCCESS), false);
      } else {
        StopRequest(req, Status(RET_SUCCESS), RequestState::REQUEST_STATE_RUNNING);
      }
      it = batch_state_->transfer_queue.erase(it);
    } else {
      // 发送未完成，继续检查下一个请求
      ++it;
    }
  }
}

}  // namespace ksana_llm