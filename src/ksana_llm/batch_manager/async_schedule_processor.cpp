/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/async_schedule_processor.h"
#include "ksana_llm/batch_manager/batch_manager.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

AsyncScheduleProcessor::~AsyncScheduleProcessor() { Stop(); }

void AsyncScheduleProcessor::Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler,
                                        std::shared_ptr<LlmRuntime> llm_runtime,
                                        std::shared_ptr<MultiBatchController> multi_batch_controller) {
  batch_scheduler_ = batch_scheduler;
  llm_runtime_ = llm_runtime;
  multi_batch_controller_ = multi_batch_controller;
  KLLM_LOG_INFO << "AsyncScheduleProcessor initialized";
}

// 异步模式：从队列获取已处理的结果
ScheduleResult AsyncScheduleProcessor::GetNextScheduleResult(size_t multi_batch_id) {
  // 如果还没有为这个multi_batch_id提交任务，先提交一个
  if (pending_results_.size() <= multi_batch_id) {
    pending_results_.resize(multi_batch_id + 1);
  }
  if (!pending_results_[multi_batch_id].valid()) {
    // 创建异步任务并提交
    AsyncScheduleTask task(multi_batch_id);
    pending_results_[multi_batch_id] = task.promise.get_future();
    task_queue_.Put(std::move(task));
  }

  // 获取结果（这会阻塞直到任务完成）
  ScheduleResult result = pending_results_[multi_batch_id].get();

  // 🔧 关键：获取调度结果后立即处理异步调度的fake token修正
  // 这里的result包含当前轮要执行的调度，但其中的token可能是上一轮的fake token
  // 异步调度器在准备当前轮结果时，上一轮的推理已经完成，真实token已经生成
  // 所以需要立即用真实token替换fake token，然后再执行当前轮的推理
  ProcessAsyncPostProcessing(result);
  // 如果结果有效，立即提交下一轮任务（流水线处理）
  if (result.is_valid) {
    AsyncScheduleTask next_task(multi_batch_id);
    pending_results_[multi_batch_id] = next_task.promise.get_future();
    task_queue_.Put(std::move(next_task));
  }

  return result;
}

void AsyncScheduleProcessor::Start() {
  terminated_ = false;

  // 启动工作线程（通常1个线程就够了，因为调度本身是序列化的）
  worker_threads_.push_back(std::make_unique<std::thread>(&AsyncScheduleProcessor::WorkerLoop, this));

  KLLM_LOG_INFO << "AsyncScheduleProcessor started with " << worker_threads_.size() << " worker threads";
}

void AsyncScheduleProcessor::Stop() {
  if (terminated_) {
    return;
  }

  terminated_ = true;
  task_queue_.Stop();

  for (auto &thread : worker_threads_) {
    if (thread && thread->joinable()) {
      thread->join();
    }
  }
  worker_threads_.clear();

  KLLM_LOG_INFO << "AsyncScheduleProcessor stopped";
}

void AsyncScheduleProcessor::WorkerLoop() {
  while (!terminated_) {
    AsyncScheduleTask task = task_queue_.Get();
    if (terminated_) {
      break;
    }
    ProcessAsyncTask(task);
  }
}

// 处理单个调度任务：调度+数据处理（这就是把同步模式的逻辑封装成任务）
void AsyncScheduleProcessor::ProcessAsyncTask(AsyncScheduleTask &task) {
  PROFILE_EVENT_SCOPE(AsyncScheduleTask, fmt::format("AsyncScheduleTask_{}", task.multi_batch_id));

  ScheduleResult result;
  size_t multi_batch_id = task.multi_batch_id;
  while (!terminated_) {
    // 1. 调用Schedule（和同步模式一样）
    std::shared_ptr<ScheduleOutputGroup> schedule_output_group = batch_scheduler_->Schedule(multi_batch_id);

    // 2. 合并调度结果（和同步模式一样）
    result.schedule_output = std::make_shared<ScheduleOutput>();
    MergeScheduleOutputGroup(schedule_output_group, *result.schedule_output);

    // 3. 检查是否有运行中的请求（和同步模式一样）
    if (schedule_output_group->RunningSize() == 0) {
      // 没有运行请求，需要等待
      multi_batch_controller_->NotifyCurrentBatchThreadNotReady(multi_batch_id);
      if (batch_scheduler_->IsIdle(multi_batch_id) && !terminated_) {
        batch_scheduler_->WaitUntilHaveReqs(multi_batch_id);
      } else {
        KLLM_LOG_DEBUG << "multi_batch_id=" << multi_batch_id << " not idle, sleep 100ms";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      continue;  // 继续循环等待
    }

    // 4. 有运行请求，进行数据处理（和同步模式一样，但需要深拷贝）
    result.is_valid = true;
    ProcessScheduleDataInternal(result, multi_batch_id);
    break;
  }

  // 返回结果
  if (terminated_) {
    result.is_valid = false;
  }
  task.promise.set_value(result);
}

void AsyncScheduleProcessor::ProcessScheduleDataInternal(ScheduleResult &result, size_t multi_batch_id) {
  if (!result.schedule_output || !llm_runtime_) {
    KLLM_LOG_ERROR << "Invalid schedule_output or llm_runtime";
    result.is_valid = false;
    return;
  }

  // 设置multi_batch_id
  result.schedule_output->multi_batch_id = multi_batch_id;

  // 重排序请求
  llm_runtime_->ReorderInferRequests(result.schedule_output->running_reqs);

  // result.deep_copy_forwarding_tokens = DeepCopyForwardRequest(result.schedule_output->running_reqs);

  // 构建SamplingRequests
  result.sampling_reqs = std::make_shared<std::vector<SamplingRequest>>();
  llm_runtime_->BuildSamplingRequest(result.schedule_output->multi_batch_id, result.schedule_output->running_reqs,
                                     *result.sampling_reqs);

  for (auto &req : *result.sampling_reqs) {
    DeepCopySamplingRequest(req);
  }

  // 计算hidden_token_num（用于后续推理处理）
  size_t tokens = 0;
  for (size_t i = 0; i < result.schedule_output->running_reqs.size(); ++i) {
    tokens += result.schedule_output->running_reqs[i]->forwarding_tokens.size() -
              result.schedule_output->running_reqs[i]->kv_cached_token_num;
  }
  result.schedule_output->hidden_token_num = tokens;
}

void AsyncScheduleProcessor::ApplyAsyncForwardingTokens(
    const std::unordered_map<int64_t, std::shared_ptr<std::vector<int>>> &deep_copy_forwarding_tokens,
    std::map<ModelInstance *, std::vector<ForwardRequest *>> &grouped_reqs) {
  for (auto &[model_inst, reqs] : grouped_reqs) {
    for (auto &req : reqs) {
      req->forwarding_tokens = deep_copy_forwarding_tokens.at(req->req_id);
    }
  }
}

// 异步后处理：修正fake token，在获取调度结果后立即调用
void AsyncScheduleProcessor::ProcessAsyncPostProcessing(ScheduleResult &result) {
  PROFILE_EVENT_SCOPE(ProcessAsyncPostProcessing, "ProcessAsyncPostProcessing");
  if (!result.is_valid) {
    return;
  }
  KLLM_LOG_DEBUG << "ProcessAsyncPostProcessing: processing " << result.schedule_output->running_reqs.size()
                 << " requests";
  batch_scheduler_->NotifyAsyncRecomputedRequests();
  for (auto &req : result.schedule_output->running_reqs) {
    // 只有在decode阶段才需要修正fake token，prefill阶段不需要修正
    // 因为prefill阶段处理的是输入token序列，不涉及生成的fake token
    // 如果这一轮是prefill 也有可能需要修正，因为有generate token
    // 在什么时候需要修正呢，其实应该看下一轮，如果是做decode，那这一轮应该修正，如果做prefill就不需要修正。
    // 新的请求刚刚加入running队列，step = 0，step > 0说明生成了请求，需要修正。
    if (req->step > 0) {
      // 更新forwarding_tokens中的fake token为真实token
      std::vector<int> draft_tokens = req->draft_tokens.GetDraftTokens();
      req->forwarding_tokens.resize(req->forwarding_tokens.size() - req->forwarding_tokens_draft_num +
                                    req->accepted_tokens.size() - kStepGenerateTokenNum - req->last_step_draft_num);
      // 更新kv cache相关信息
      req->kv_cached_token_num = req->forwarding_tokens.size();
      req->prefix_cache_len = req->kv_cached_token_num;
      req->cache_manager->UpdateRequestTokens(req->req_id, req->forwarding_tokens, req->kv_cached_token_num,
                                              req->kv_cache_blocks);
      // 添加当前生成的真实token
      req->forwarding_tokens.emplace_back(req->generated_token);
      req->last_step_token_num = req->accepted_tokens.size() + kStepGenerateTokenNum;
      req->last_step_draft_num = draft_tokens.size();

      req->output_mutex.lock();
      req->output_tokens.insert(req->output_tokens.end(),
                                req->forwarding_tokens.end() - req->accepted_tokens.size() - kStepGenerateTokenNum,
                                req->forwarding_tokens.end());
      req->output_mutex.unlock();
      req->forwarding_tokens.insert(req->forwarding_tokens.end(), draft_tokens.begin(), draft_tokens.end());
      req->forwarding_tokens_draft_num = req->draft_tokens.size();
      // 设置采样token数量
      req->sampling_token_num =
          req->logits_custom_length > 0 ? req->logits_custom_length : req->draft_tokens.size() + kStepGenerateTokenNum;
    }
  }

  llm_runtime_->ReorderInferRequests(result.schedule_output->running_reqs);

  auto deep_copy_forwarding_tokens = DeepCopyForwardRequest(result.schedule_output->running_reqs);
  // 构建ForwardRequests
  result.grouped_reqs = std::make_shared<std::map<ModelInstance *, std::vector<ForwardRequest *>>>();
  llm_runtime_->BuildForwardRequests(result.schedule_output->multi_batch_id, result.schedule_output->running_reqs,
                                     *result.grouped_reqs);

  ApplyAsyncForwardingTokens(*deep_copy_forwarding_tokens, *result.grouped_reqs);

  // 处理sampling_reqs中的SamplingRequest - 使用origin_tokens获取真实token
  llm_runtime_->DeepCopyAndSyncSamplingRequests(result.schedule_output->running_reqs, *result.sampling_reqs);
}

}  // namespace ksana_llm