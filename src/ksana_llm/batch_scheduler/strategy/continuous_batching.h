/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_scheduler/strategy/base_strategy.h"
#include "ksana_llm/runtime/request_state.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The auto-prefix-caching continuous batching implementation.
class ContinuousBatchingStrategy : public BaseScheduleStrategy {
 public:
  explicit ContinuousBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config,
                                      const RuntimeConfig &runtime_config);

  virtual ~ContinuousBatchingStrategy() {}

  // Update cache manager, process finished and timeout requests.
  virtual void UpdateRunningRequests(const std::vector<std::shared_ptr<InferRequest>> &running_reqs) override;

  virtual void Schedule(std::vector<std::shared_ptr<InferRequest>> &waiting_reqs) override;

  virtual void UpdateAsyncState() override;

 private:
  // True if request timeout.
  inline bool CheckRequestTimeout(const std::shared_ptr<InferRequest> req);

  // True if request finished, that is, arrive max output len or encounter eos.
  inline bool CheckRequestFinish(const std::shared_ptr<InferRequest> req);

  // Determine the number of draft_tokens to generate in the current step based on the scheduling status.
  void DetermineDraftNum(std::shared_ptr<InferRequest> req);

  // In asynchronous mode, inflighting request may be in other queues.
  void RemoveRequestFromBatchState(const std::shared_ptr<InferRequest> &req);

  void EstimateRequestPlanningWorkload(const std::shared_ptr<InferRequest> &req);

  // Reset the req and cache status, destroy swap or finish req
  // If terminated is true, the request is terminated, and will notify to stop this request.
  // If is_swap_req is true, the request is a swap request, otherwise it is a normal request.
  void ResetRequest(std::shared_ptr<InferRequest> req, Status ret_status, bool terminated);

  // Destroy the request and add it to the begining of waiting queue to recompute.
  void RecomputeRequest(std::shared_ptr<InferRequest> req);
  void SyncRecomputeRequest(std::shared_ptr<InferRequest> req);
  Status RecomputeMockRequest(std::shared_ptr<InferRequest> &req);
  bool ProcessAsyncRecomputeRequest(const std::shared_ptr<InferRequest> &req);

  void SwapoutRequest(std::shared_ptr<InferRequest> req);
  void SyncSwapoutRequest(std::shared_ptr<InferRequest> req);

  // Set the finish status of the request to finished, timeout or aborted.
  void StopRequest(std::shared_ptr<InferRequest> req, Status ret_status, RequestState req_state);
  void SyncStopRequest(std::shared_ptr<InferRequest> req, Status ret_status, RequestState req_state);

  void RecoverAsyncRecomputedRequests();
  bool ProcessAsyncStoppedRequest(std::shared_ptr<InferRequest> &req);
  void RecoverAsyncSwapoutRequests();
  bool ProcessAsyncSwapoutRequest(const std::shared_ptr<InferRequest> &req);

  // Check the running queue to determine whether it exceeds the
  // max_step_token_num. return [step_token_with_kv_cache,
  // step_token_without_kv_cache]
  std::pair<size_t, size_t> CheckRunningQueueStepTokens(const std::vector<std::shared_ptr<InferRequest>> &checking_reqs,
                                                        std::vector<std::shared_ptr<InferRequest>> &passed_reqs);

  // Schedule the running/swapped/waiting queue.
  void ProcessDecodingQueue();
  void ProcessSwappedQueue();
  void ProcessWaitingQueue();
  void ProcessTransferQueue();

  Status MergePendingSwapinRequests(bool blocking, bool early_stop);
  Status MergePendingSwapoutRequests(bool blocking, bool early_stop);

  /**
   * @brief 处理prefill节点的传输队列
   *
   * 检查传输队列中的每个请求，判断是否已发送完成。
   * 如果发送完成，则将请求从传输队列中移除。
   */
  void ProcessPrefillTransferQueue();

  /**
   * @brief 处理decode节点的传输队列
   *
   * 检查传输队列中的每个请求，判断是否已接收完成。
   * 如果接收完成，则将请求从传输队列移至运行队列，并更新相关状态。
   */
  void ProcessDecodeTransferQueue();

  /**
   * @brief 为请求队列中的每个请求添加传输元数据
   *
   * 为队列中的每个推理请求添加传输元数据，包括KV缓存块的物理指针和
   * 已缓存的token数量。如果是prefill节点，则将推理token数设为1。
   *
   * @param queue 需要添加传输元数据的请求队列
   */
  void AddTransferMeta(std::vector<std::shared_ptr<InferRequest>> &queue);

 private:
  size_t GetMaxRequiredTokenNum(const size_t token_num) const;

  void ReportRequestProgressInfo(const std::shared_ptr<InferRequest> req);

  friend class ContinuousBatchingStrategyTest;  // for test
  ConnectorConfig connector_config_;

  // For one schedule instance.
  size_t dp_max_step_token_num_;
  // 在标准场景用于限制dp分组下最大的batch。
  // 在PD分离场景
  // Deocde节点：增加了预传输的batch大小，使用dp_max_decode_batch_size_参数限制dp分组下最大的batch。
  // Prefill节点：与标准场景保持一致。
  size_t dp_max_batch_size_;
  // 仅在PD分离时作用于Decode节点(不影响标准场景与Prefill节点)
  // 用于限制dp分组下最大的batch，等同于标准场景的dp_max_batch_size_.
  size_t dp_max_decode_batch_size_;
  size_t dp_max_logits_num_;

  // Current active dp group id, only ranks where (active_dp_group_id %
  // world_size
  // == rank % world_size) will be scheduled
  size_t active_dp_group_id_ = 0;
};

}  // namespace ksana_llm
