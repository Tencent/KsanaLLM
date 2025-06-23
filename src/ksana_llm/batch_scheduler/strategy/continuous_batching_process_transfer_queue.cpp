/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/continuous_batching.h"
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
      transfer_engine->AddTransferMeta(req->kv_comm_group_key, req->kv_comm_request_id, req->kv_cached_token_num,
                                       block_ptrs);
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

  if (batch_state_->transfer_queue.empty()) {
    return;
  }

  auto transfer_engine = TransferEngine::GetInstance();

  for (auto it = batch_state_->transfer_queue.begin(); it != batch_state_->transfer_queue.end();) {
    auto req = *it;
    // 检查请求是否接收完成，如果完成则返回第一个token，否则返回-1
    int first_token = transfer_engine->IsRecvDone(req->kv_comm_request_id);
    bool queue_enough = batch_state_->schedule_output->running_reqs.size() < batch_scheduler_config_.max_batch_size;
    if (first_token != -1 && queue_enough) {
      // 接收完成，更新请求状态
      req->kv_cached_token_num = req->forwarding_tokens.size();
      req->prefix_cache_len = req->kv_cached_token_num;
      req->forwarding_tokens.push_back(first_token);
      req->output_tokens.push_back(first_token);
      batch_state_->schedule_output->running_reqs.push_back(req);
      it = batch_state_->transfer_queue.erase(it);
      transfer_engine->CleanupTransferMeta(req->kv_comm_request_id);
    } else {
      // 接收未完成，继续检查下一个请求
      ++it;
    }
  }
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
    // 检查请求是否发送完成
    if (transfer_engine->IsSendDone((*it)->kv_comm_request_id)) {
      // 发送完成，从传输队列中移除该请求
      it = batch_state_->transfer_queue.erase(it);
    } else {
      // 发送未完成，继续检查下一个请求
      ++it;
    }
  }
}

}  // namespace ksana_llm