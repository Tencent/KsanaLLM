/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <future>
#include <thread>
#include <vector>

#include "ksana_llm/batch_manager/schedule_processor_interface.h"
#include "ksana_llm/utils/blocking_queue.h"
#include "ksana_llm/utils/schedule_output_process.h"

namespace ksana_llm {

// 异步调度任务
struct AsyncScheduleTask {
  size_t multi_batch_id = 0;
  std::promise<ScheduleResult> promise;

  // 默认构造函数（BlockingQueue需要）
  AsyncScheduleTask() = default;

  // 带参数的构造函数
  explicit AsyncScheduleTask(size_t id) : multi_batch_id(id) {}

  // 移动构造函数（promise不能拷贝，只能移动）
  AsyncScheduleTask(AsyncScheduleTask&& other) noexcept
      : multi_batch_id(other.multi_batch_id), promise(std::move(other.promise)) {}

  // 移动赋值操作符
  AsyncScheduleTask& operator=(AsyncScheduleTask&& other) noexcept {
    if (this != &other) {
      multi_batch_id = other.multi_batch_id;
      promise = std::move(other.promise);
    }
    return *this;
  }

  // 禁用拷贝构造和拷贝赋值（因为promise不能拷贝）
  AsyncScheduleTask(const AsyncScheduleTask&) = delete;
  AsyncScheduleTask& operator=(const AsyncScheduleTask&) = delete;
};

// 异步调度处理器：把调度+数据处理封装成任务，丢入生产者消费者队列
class AsyncScheduleProcessor : public ScheduleProcessorInterface {
 public:
  AsyncScheduleProcessor() = default;
  ~AsyncScheduleProcessor() override;

  void Initialize(std::shared_ptr<BatchSchedulerInterface> batch_scheduler, std::shared_ptr<LlmRuntime> llm_runtime,
                  std::shared_ptr<MultiBatchController> multi_batch_controller) override;

  // 异步模式：从队列获取已完全处理的结果（包括后处理）
  ScheduleResult GetNextScheduleResult(size_t multi_batch_id) override;

  void Start() override;
  void Stop() override;

  // 异步后处理：修正fake token，在获取调度结果后立即调用
  void ProcessAsyncPostProcessing(ScheduleResult& result);

  // 应用深拷贝的forwarding_tokens到grouped_reqs
  void ApplyAsyncForwardingTokens(
      const std::unordered_map<int64_t, std::shared_ptr<std::vector<int>>>& deep_copy_forwarding_tokens,
      std::map<ModelInstance*, std::map<InferStage, std::vector<ForwardRequest*>>>& grouped_reqs);

 private:
  // 工作线程循环：处理调度任务
  void WorkerLoop();

  // 处理单个调度任务：调度+数据处理
  void ProcessAsyncTask(AsyncScheduleTask& task);

  // 内部辅助方法：处理调度数据
  void ProcessScheduleDataInternal(ScheduleResult& result, size_t multi_batch_id);
  // 异步处理相关
  std::atomic<bool> terminated_ = false;
  BlockingQueue<AsyncScheduleTask> task_queue_;
  std::vector<std::unique_ptr<std::thread>> worker_threads_;

  // 为每个multi_batch_id维护一个future，实现流水线处理
  std::vector<std::future<ScheduleResult>> pending_results_;
};

}  // namespace ksana_llm