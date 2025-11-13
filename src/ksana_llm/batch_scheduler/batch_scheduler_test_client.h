/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <sstream>
#include <unordered_map>

#include "ksana_llm/batch_scheduler/batch_scheduler_interface.h"
#include "ksana_llm/batch_scheduler/batch_scheduler_test_helper.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/threadpool.h"

namespace ksana_llm {

class ClientSimulator {
 public:
  ClientSimulator(int thread_num, BatchSchedulerInterface* batch_scheduler)
      : scheduler_(batch_scheduler), thread_num_(thread_num), threadpool_(thread_num), is_destroying_(false) {
    threadpool_.Start();
  }

  ~ClientSimulator() {
    KLLM_LOG_INFO << "~ClientSimulator IsAllRequestFinished=" << IsAllRequestFinished();
    is_destroying_ = true;
    if (!IsAllRequestFinished()) {
      for (auto& it : client_req_map_) {
        it.second.req_group[0]->waiter->Notify();
        KLLM_LOG_INFO << "Notify unfinished req: " << it.second.req_group[0]->req_id;
      }
    }
    threadpool_.Stop();
  }

  void AddInferRequests(int req_group_id, std::vector<std::shared_ptr<InferRequest>>& infer_reqs) {
    std::unordered_map<int, ClientRequest>::iterator iter;
    KLLM_CHECK_WITH_INFO(infer_reqs.size() > 0, FormatStr("infer_reqs.size()==%d, must >0.", infer_reqs.size()));
    {
      std::lock_guard<std::mutex> guard(mux_);
      KLLM_CHECK_WITH_INFO(client_req_map_.find(req_group_id) == client_req_map_.end(),
                           FormatStr("req_group_id %d exists.", req_group_id));
      ClientRequest dummy_req;
      client_req_map_[req_group_id] = dummy_req;
      iter = client_req_map_.find(req_group_id);
      iter->second.req_group = infer_reqs;
    }
    threadpool_.Submit([=]() -> int {
      if (is_destroying_) {
        return 0;
      }
      ClientRequest& req = iter->second;
      req.enqueue_status = scheduler_->AddInferRequest(req.req_group);
      // all requests in req_group come from same request.
      req.req_group[0]->waiter->Wait();
      req.is_finished = true;
      return 0;
    });
  }

  void AddAnInferRequest(std::shared_ptr<InferRequest>& infer_req) {
    std::vector<std::shared_ptr<InferRequest>> reqs;
    reqs.push_back(infer_req);
    AddInferRequests(infer_req->req_id, reqs);
  }

  bool IsAllRequestFinished() {
    for (auto& it : client_req_map_) {
      if (!it.second.is_finished) {
        return false;
      }
    }
    return true;
  }

 private:
  struct ClientRequest {
    std::vector<std::shared_ptr<InferRequest>> req_group;
    bool is_finished = false;
    Status enqueue_status;
  };

  BatchSchedulerInterface* scheduler_;
  int thread_num_;
  ThreadPool threadpool_;
  bool is_destroying_;

  std::unordered_map<int, ClientRequest> client_req_map_;
  std::mutex mux_;
};

class ParallelTester {
 public:
  ParallelTester(BatchSchedulerInterface* batch_scheduler, BatchSchedulerEnvironmentSimulator* env_simulator)
      : batch_scheduler_(batch_scheduler), env_simulator_(env_simulator) {}

  struct RequestInfo {
    int req_id;
    int expect_output_token_num;
    int input_token_num;
    std::shared_ptr<Request> req;
    std::vector<std::shared_ptr<InferRequest>> infer_req_group;
  };

  // hooks used during execution.
  class ExeHookInterface {
   public:
    virtual void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) {}

    virtual void CheckRequestsAfterExecution(const std::vector<RequestInfo>& reqs) {}

   public:
    int before_step_num = 0;
    int after_exe_num = 0;
  };

  // This hook checks results when  all requests are finished as expected.
  class DefaultResultCheckHook : public ExeHookInterface {
   public:
    explicit DefaultResultCheckHook(BatchSchedulerEnvironmentSimulator* env_simulator)
        : env_simulator_(env_simulator) {}
    ~DefaultResultCheckHook() {
      KLLM_LOG_INFO << "~DefaultResultCheckHook, after_exe_num=" << after_exe_num;
      EXPECT_GT(after_exe_num,
                0);  // CheckRequestsAfterExecution must be invoked. Maybe this hook is not added to hook list.
    }

    void CheckRequestsAfterExecution(const std::vector<RequestInfo>& reqs) override {
      after_exe_num++;
      for (auto& info : reqs) {
        for (auto& infer_req : info.infer_req_group) {
          env_simulator_->CheckRequestOutput(infer_req);
        }
      }
    }

   private:
    BatchSchedulerEnvironmentSimulator* env_simulator_;
  };

  // Check split fuse
  class SplitFuseCheckHook : public ExeHookInterface {
   public:
    explicit SplitFuseCheckHook(size_t split_fuse_token_num, size_t prefix_token_num = 0)
        : split_fuse_token_num_(split_fuse_token_num), prefix_token_num_(prefix_token_num) {}
    ~SplitFuseCheckHook() {
      KLLM_LOG_INFO << "~SplitFuseCheckHook, before_step_num=" << before_step_num
                    << ", after_exe_num=" << after_exe_num;
      EXPECT_GT(before_step_num, 0);
      EXPECT_GT(after_exe_num, 0);
    }

    void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) override {
      before_step_num++;
      for (auto req : reqs) {
        if (split_fuse_token_num_ > 0 && req->forwarding_tokens.size() < req->output_tokens.size()) {
          EXPECT_EQ((req->forwarding_tokens.size() - req->kv_cached_token_num), split_fuse_token_num_);
          req_split_step_[req->req_id]++;
        }
      }
    }

    void CheckRequestsAfterExecution(const std::vector<RequestInfo>& reqs) override {
      after_exe_num++;
      for (auto req : reqs) {
        if ((split_fuse_token_num_ > 0) && ((req.input_token_num - prefix_token_num_) > split_fuse_token_num_)) {
          EXPECT_GE(req_split_step_[req.infer_req_group[0]->req_id],
                    (req.input_token_num - prefix_token_num_) / split_fuse_token_num_);
        }
      }
    }

   private:
    std::unordered_map<int, int> req_split_step_;
    size_t split_fuse_token_num_;
    size_t prefix_token_num_;
  };

  // Generate RequestInfos
  void GenerateRequests(int request_num, int min_expect_output_num, int max_expect_output_num, int min_input_num,
                        int max_input_num, std::vector<ParallelTester::RequestInfo>& reqs) {
    KLLM_CHECK_WITH_INFO(
        min_expect_output_num < max_expect_output_num,
        FormatStr("min_expect_output_num % should be larger than 0 and less than max_expect_output_num %d.",
                  min_expect_output_num, max_expect_output_num));

    KLLM_CHECK_WITH_INFO(
        min_input_num < max_input_num,
        FormatStr("min_input_num % should be less than max_input_num %d.", min_input_num, max_input_num));

    std::srand(10);
    for (int i = 0; i < request_num; i++) {
      ParallelTester::RequestInfo info;
      info.req_id = i;
      info.expect_output_token_num =
          std::rand() % (max_expect_output_num - min_expect_output_num) + min_expect_output_num;
      info.input_token_num = std::rand() % (max_input_num - min_input_num) + min_input_num;
      reqs.push_back(info);
    }
  }

  // Add RequestInfo by single pair seeds
  void InitRequestInfoListByDefault(std::vector<RequestInfo>& reqs) {
    std::vector<std::pair<int, int>> seeds;
    seeds.resize(1);
    seeds[0].first = 0;
    for (auto& info : reqs) {
      seeds[0].second = info.req_id;
      info.infer_req_group =
          env_simulator_->InitRequest(info.req_id, info.input_token_num, info.expect_output_token_num, info.req, seeds);
    }
  }

  // Run requests in parallel and check
  void DoParallelRequestAndCheck(int client_num, std::vector<RequestInfo>& reqs, std::vector<ExeHookInterface*>& hooks,
                                 int timeout = 10) {
    KLLM_LOG_INFO << "DoParallelRequestAndCheck start.  client_num=" << client_num << ", request_num=" << reqs.size();
    KLLM_CHECK_WITH_INFO(hooks.size() > 0, "There must be some hooks");

    time_t start_time = ProfileTimer::GetCurrentTime();
    ClientSimulator client_simulator(client_num, batch_scheduler_);
    for (auto& info : reqs) {
      client_simulator.AddInferRequests(info.req_id, info.infer_req_group);
    }

    // Wait for request enqueue
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    // schedule and generate tokens
    GenerationOutputGroup generation_output_group;
    size_t multi_batch_idx = 0;
    int step = 0;
    while (true) {
      std::shared_ptr<ScheduleOutputGroup> output_group = batch_scheduler_->Schedule(multi_batch_idx);
      ScheduleOutput* scheduled_out = output_group->outputs.at(0);
      std::vector<std::shared_ptr<InferRequest>> scheduled_reqs = scheduled_out->running_reqs;
      if (scheduled_reqs.empty()) {
        if (client_simulator.IsAllRequestFinished()) {
          KLLM_LOG_INFO << "All requests finished";
          break;
        }
        time_t cur_time = ProfileTimer::GetCurrentTime();
        if ((cur_time - start_time) > timeout) {
          KLLM_LOG_INFO << "Test Timeout. timeout=" << timeout << " seconds";
          break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        continue;
      }
      step++;
      for (auto req : scheduled_reqs) {
        KLLM_LOG_DEBUG << "Before RunAStep step=" << step << ", " << req->ScheduleStateToStr();
      }
      KLLM_CHECK(scheduled_out->IsLaunchable());
      scheduled_out->LaunchScheduleOutput();
      generation_output_group.BuildFromScheduleOutputGroup(*output_group);
      scheduled_out->Clear();
      for (auto hook : hooks) {
        hook->CheckRequestsBeforeAStep(scheduled_reqs);
      }
      env_simulator_->RunAStep(scheduled_reqs);
      batch_scheduler_->UpdateWithGenerationResult(multi_batch_idx, generation_output_group);
    }

    // Check request results
    for (auto hook : hooks) {
      KLLM_LOG_INFO << "CheckRequestsAfterExecution";
      hook->CheckRequestsAfterExecution(reqs);
    }
    KLLM_LOG_INFO << "DoParallelRequestAndCheck finished";
  }

 private:
  BatchSchedulerInterface* batch_scheduler_;
  BatchSchedulerEnvironmentSimulator* env_simulator_;
};

class PrintStepHook : public ParallelTester::ExeHookInterface {
 public:
  explicit PrintStepHook(bool print_all_blocks = false) : print_all_blocks_(print_all_blocks) {}
  ~PrintStepHook() {
    KLLM_LOG_INFO << "~PrintStepHook, before_step_num=" << before_step_num;
    EXPECT_GT(before_step_num, 0);
  }

  void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) override {
    before_step_num++;
    for (auto& req : reqs) {
      std::ostringstream ss;
      ss << "BeforeRunStep " << before_step_num << req << ", output_tokens.size()=" << req->output_tokens.size()
         << ", sampling_result_tokens.size()=" << req->sampling_result_tokens.size();
      ss << ", tokens={ ";
      for (size_t i = 0; i < req->output_tokens.size(); i++) {
        ss << req->output_tokens[i] << ", ";
      }
      ss << "} ";
      KLLM_LOG_INFO << ss.str() << req->PrintKVBlockIds(print_all_blocks_);
    }
  }

 private:
  bool print_all_blocks_;
};

class FixPrefixTestCase {
 public:
  FixPrefixTestCase(int prefix_block_num, int block_token_num, int device_num, size_t splitfuse_token_num,
                    bool init_env_simulator = true)
      : prefix_block_num_(prefix_block_num), device_num_(device_num), splitfuse_token_num_(splitfuse_token_num) {
    // 创建一个 BlockManagerConfig 对象，用于配置 BatchSchedulerEnvironmentSimulator
    block_manager_config_.host_allocator_config.blocks_num = 100;
    block_manager_config_.device_allocator_config.blocks_num = 100;
    block_manager_config_.device_allocator_config.block_token_num = block_token_num;

    init_env_simulator_ = init_env_simulator;

    // 使用配置创建一个 BlockManagerSimulator 对象
    if (init_env_simulator_) {
      std::shared_ptr<FakedBlockAllocatorGroup> block_allocator_group =
          std::make_shared<FakedBlockAllocatorGroup>(block_manager_config_, device_num_);
      env_simulator_ =
          new BatchSchedulerEnvironmentSimulator(block_manager_config_, device_num_, block_allocator_group);
      KLLM_LOG_INFO << "Simulator start";
    }
  }

  ~FixPrefixTestCase() {
    if (init_env_simulator_) {
      delete env_simulator_;
    }
  }

  const BlockManagerConfig& GetBlockManagerConfig() const { return block_manager_config_; }

  void SetBatchScheduler(BatchSchedulerInterface* batch_scheduler) { batch_scheduler_ = batch_scheduler; }

  void SetEnvSimulator(BatchSchedulerEnvironmentSimulator* env_simulator) { env_simulator_ = env_simulator; }

  BatchSchedulerEnvironmentSimulator* GetEnvSimulator() { return env_simulator_; }

  void RunTestNoSwapTriggered() {
    int block_token_num = block_manager_config_.device_allocator_config.block_token_num;
    int block_num = block_manager_config_.device_allocator_config.blocks_num;

    int prefix_token_num = prefix_block_num_ * block_manager_config_.device_allocator_config.block_token_num;

    int request_num = 10;
    int client_num = 3;
    int min_expect_output_num = 3;
    int max_expect_output_num = min_expect_output_num + 3;
    int min_input_num = prefix_token_num;
    int max_input_num = 10 + prefix_token_num;

    // check request token num can not trigger swapout
    ASSERT_LT(client_num * (max_input_num + max_expect_output_num), block_num * block_token_num);

    RunTest(request_num, client_num, min_expect_output_num, max_expect_output_num, min_input_num, max_input_num,
            prefix_token_num);

    auto& stat = env_simulator_->GetBlockManagerStat();
    EXPECT_EQ(stat.swapout_succ_num, 0);
    EXPECT_EQ(stat.swapout_fail_num, 0);
    EXPECT_EQ(stat.swapin_succ_num, 0);
    EXPECT_EQ(stat.swapin_fail_num, 0);
  }

  void RunTestSwapTriggered() {
    int block_token_num = block_manager_config_.device_allocator_config.block_token_num;
    int block_num = block_manager_config_.device_allocator_config.blocks_num;

    int prefix_token_num = prefix_block_num_ * block_manager_config_.device_allocator_config.block_token_num;

    int request_num = 100;
    int client_num = 10;
    int min_expect_output_num = 3;
    int max_expect_output_num = min_expect_output_num + 300;
    int min_input_num = prefix_token_num;
    int max_input_num = 100 + prefix_token_num;

    // check request token num can trigger swapout
    ASSERT_LT((max_input_num + max_expect_output_num), block_num * block_token_num);
    ASSERT_GT(client_num * (min_input_num + min_expect_output_num), block_num * block_token_num);
    ASSERT_GT(request_num, client_num);

    RunTest(request_num, client_num, min_expect_output_num, max_expect_output_num, min_input_num, max_input_num,
            prefix_token_num);

    auto& stat = env_simulator_->GetBlockManagerStat();
    EXPECT_EQ(stat.swapout_succ_num, 0);  // recomputed
    EXPECT_EQ(stat.swapout_fail_num, 0);
    EXPECT_EQ(stat.swapin_succ_num, 0);  // recomputed
    EXPECT_EQ(stat.swapin_fail_num, 0);
  }

 private:
  // In current implementation, requests will not share blocks in first step.
  // Test: all requests have same prefix token
  class SamePrefixCacheNoMergeBeforeFirstBatchCheckHook : public ParallelTester::ExeHookInterface {
   public:
    SamePrefixCacheNoMergeBeforeFirstBatchCheckHook(int prefix_block_num, int block_token_num, int tp_num)
        : prefix_block_num_(prefix_block_num), block_token_num_(block_token_num), tp_num_(tp_num) {
      prefix_blocks_.resize(tp_num_);
      for (int i = 0; i < tp_num_; i++) {
        prefix_blocks_[i].resize(prefix_block_num_);
      }
    }

    ~SamePrefixCacheNoMergeBeforeFirstBatchCheckHook() {
      KLLM_LOG_INFO << "~SamePrefixCacheNoMergeBeforeFirstBatchCheckHook, before_step_num=" << before_step_num;
      EXPECT_GT(before_step_num, 0);
      EXPECT_GT(cache_hit_num_, 0);
    }

    void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) override {
      before_step_num++;
      ASSERT_GT(reqs.size(), 0);
      if (reqs[0]->forwarding_tokens.size() < prefix_block_num_ * block_token_num_) {
        return;
      }

      if (!is_cache_set_) {
        if (reqs_in_first_step_.size() == 0) {
          // record requests in first step. Their kv cache will be reused.
          for (auto& req : reqs) {
            reqs_in_first_step_[req->req_id] = req;
          }

          // Requests should not share same prefix cache blocks
          for (size_t i = 1; i < reqs.size(); i++) {
            EXPECT_FALSE(CheckBlocksSame(reqs[0]->kv_cache_blocks, reqs[i]->kv_cache_blocks, prefix_block_num_));
          }
          return;
        }

        // Blocks should be merged in second step.
        StorePrefixCache(reqs[0]);
        for (size_t i = 1; i < reqs.size(); i++) {
          EXPECT_TRUE(IsRequestPrefixCacheHit(reqs[i]));
        }
        return;
      }

      // All request should use same prefix caching after cache is set
      for (auto& req : reqs) {
        EXPECT_TRUE(IsRequestPrefixCacheHit(req));
        cache_hit_num_++;
      }
    }

   private:
    bool IsRequestPrefixCacheHit(const std::shared_ptr<InferRequest>& req) {
      return CheckBlocksSame(req->kv_cache_blocks, prefix_blocks_, prefix_block_num_);
    }

    bool CheckBlocksSame(const std::vector<std::vector<int>>& this_blocks,
                         const std::vector<std::vector<int>>& that_blocks, int block_num) {
      for (int i = 0; i < tp_num_; i++) {
        for (int j = 0; j < block_num; j++) {
          if (this_blocks[i][j] != that_blocks[i][j]) {
            return false;
          }
        }
      }
      return true;
    }

    void StorePrefixCache(const std::shared_ptr<InferRequest>& req) {
      ASSERT_FALSE(is_cache_set_);
      auto& kv_cache_blocks = req->kv_cache_blocks;
      for (int i = 0; i < tp_num_; i++) {
        std::copy(kv_cache_blocks[i].begin(), kv_cache_blocks[i].begin() + prefix_block_num_,
                  prefix_blocks_[i].begin());
      }
      is_cache_set_ = true;
    }

   private:
    size_t prefix_block_num_;
    size_t block_token_num_;
    int tp_num_;
    std::vector<std::vector<int>> prefix_blocks_;

    bool is_cache_set_ = false;

    std::unordered_map<int, std::shared_ptr<InferRequest>> reqs_in_first_step_;

    int cache_hit_num_ = 0;
  };

  void RunTest(int request_num, int client_num, int min_expect_output_num, int max_expect_output_num, int min_input_num,
               int max_input_num, int prefix_token_num) {
    ParallelTester tester(batch_scheduler_, env_simulator_);

    std::vector<ParallelTester::ExeHookInterface*> hooks;
    ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
    ParallelTester::SplitFuseCheckHook splitfuse_check_hook(splitfuse_token_num_, prefix_token_num);
    PrintStepHook print_hook(true);
    SamePrefixCacheNoMergeBeforeFirstBatchCheckHook prefix_check_hook(
        prefix_token_num / block_manager_config_.device_allocator_config.block_token_num,
        block_manager_config_.device_allocator_config.block_token_num, device_num_);
    hooks.push_back(&default_hook);
    hooks.push_back(&splitfuse_check_hook);
    hooks.push_back(&print_hook);
    hooks.push_back(&prefix_check_hook);

    std::vector<ParallelTester::RequestInfo> req_list;
    tester.GenerateRequests(request_num, min_expect_output_num, max_expect_output_num, prefix_token_num, max_input_num,
                            req_list);

    // Init requests with same prefix tokens.
    std::vector<std::pair<int, int>> seeds;
    seeds.resize(2);
    seeds[0].first = 0;
    seeds[0].second = 1;
    seeds[1].first = prefix_token_num;
    for (auto& info : req_list) {
      seeds[1].second = info.req_id;
      info.infer_req_group =
          env_simulator_->InitRequest(info.req_id, info.input_token_num, info.expect_output_token_num, info.req, seeds);
    }

    tester.DoParallelRequestAndCheck(client_num, req_list, hooks, 60);
  }

 private:
  int prefix_block_num_;
  int device_num_;
  size_t splitfuse_token_num_;

  BatchSchedulerInterface* batch_scheduler_ = nullptr;
  BatchSchedulerEnvironmentSimulator* env_simulator_ = nullptr;
  BlockManagerConfig block_manager_config_;

  bool init_env_simulator_ = true;
};

}  // namespace ksana_llm
