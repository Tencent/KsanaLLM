/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_scheduler/strategy/base_strategy.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The auto-prefix-caching continuous batching implementation.
class ContinuousBatchingStrategy : public BaseScheduleStrategy {
 public:
  ContinuousBatchingStrategy(const BatchSchedulerConfig &batch_scheduler_config, int tp_num);

  virtual ~ContinuousBatchingStrategy() {}

  // Update cache manager, process finished and timeout requests.
  virtual void UpdateRunningRequests() override;

  virtual void Schedule(std::vector<std::shared_ptr<InferRequest>> &waiting_reqs) override;

 private:
  // True if request timeout.
  inline bool CheckRequestTimeout(const std::shared_ptr<InferRequest> req);

  // True if request finished, that is, arrive max output len or encounter eos.
  inline bool CheckRequestFinish(const std::shared_ptr<InferRequest> req);

  // When the request is in a non-generation state, perform a jump-forward using a constant string.
  void JumpForwardRequest(std::shared_ptr<InferRequest> req);

  // When the request is in a generation state, determine if it can transition to the next state and perform a
  // jump-forward.
  void ProcessStructuredOutput(std::shared_ptr<InferRequest> req);

  // Determine the number of draft_tokens to generate in the current step based on the scheduling status.
  void DetermineDraftNum(std::shared_ptr<InferRequest> req);

  // Expand structured output with a constant string (dependent on the execution of Retokenizationr by Tokenizer).
  void ExtendTokensWithRetokenization(std::shared_ptr<InferRequest> req);

  // Expand structured output with a constant string (independent of Retokenizationr execution by Tokenizer).
  void ExtendTokensWithoutRetokenization(std::shared_ptr<InferRequest> req);

  // Reset the req and cache status, destroy swap or finish req
  // If terminated is true, the request is terminated, and will notify to stop this request.
  // If is_swap_req is true, the request is a swap request, otherwise it is a normal request.
  void ResetRequest(std::shared_ptr<InferRequest> req, Status req_status, bool is_swap_req, bool terminated);

  // Destroy the request and add it to the begining of waiting queue to recompute.
  std::vector<std::shared_ptr<InferRequest>>::iterator RecomputeRequest(
      std::vector<std::shared_ptr<InferRequest>>::iterator &it, bool is_swap_req = false);

  // Set the finish status of the request to finished, timeout or aborted.
  void StopRequest(std::shared_ptr<InferRequest> req, Status req_status, bool is_swap_req);

  // Check the running queue to determine whether it exceeds the max_step_token_num.
  // return [step_token_with_kv_cache, step_token_without_kv_cache]
  std::pair<size_t, size_t> CheckRunningQueueStepTokens();

  // Calculate how many blocks to be allocated.
  size_t GetRunningRequestsBlockNeed();

  // Try to allocate request blocks. If failed, try the allocation again after all swapout finished.
  Status AllocateRequestBlocksWithRetry(std::shared_ptr<InferRequest> req, size_t &total_needed_block_num,
                                        size_t &step_block_num, bool &allocate_block_succ, bool &skip_swapout_check);

  /**
   * Processes a request to determine the appropriate number of tokens to split or fuse based on the current
   * batching strategy configuration. This function adjusts the number of output tokens in the request to match
   * the calculated split or fuse token count, and updates the shared and unique block counts accordingly.
   *
   * The function aims to optimize the processing of requests by dynamically adjusting the number of tokens
   * to be processed together, based on the configured thresholds and the current state of the request and
   * batch scheduler.
   */
  bool ProcessSplitFuseToken(std::shared_ptr<InferRequest> req, size_t &shared_block_num, size_t &unique_block_num,
                             size_t &shared_token_num, const size_t step_token_without_kv_cache_num,
                             const size_t decode_request_num);

  // Schedule the running/swapped/waiting queue.
  void ProcessRunningQueue();
  void ProcessSwappedQueue();
  void ProcessWaitingQueue();
  void ProcessTransferQueue();
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
  // Wait pending swap out/in requests done, and merge these requests.
  // If blocking is false, the function will return immediately even no request finished.
  // If early_stop is false, the function return until all requests finished.
  Status MergePendingSwapinRequests(bool blocking, bool early_stop);
  Status MergePendingSwapoutRequests(bool blocking, bool early_stop);

  // input_ids <= decode_token_num_threshold_ will regard as decode (use page attention), default is 1
  size_t decode_token_num_threshold_ = 1;

  friend class ContinuousBatchingStrategyTest;  // for test
  ConnectorConfig connector_config_;

  // For one schedule instance.
  size_t dp_max_step_token_num_;
  size_t dp_max_batch_size_;
  size_t dp_max_logits_num_;
};

}  // namespace ksana_llm
