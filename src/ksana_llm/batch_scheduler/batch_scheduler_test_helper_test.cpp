/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/batch_scheduler_test_helper.h"
#include "ksana_llm/batch_scheduler/batch_scheduler_test_client.h"

#include <memory>
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

using namespace ksana_llm;

// 定义一个 BlockSchedulerTest 类，用于测试BatchSchedulerEnvironmentSimulator
class BatchSchedulerEnvironmentSimulatorTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }

  // 在每个测试用例执行之前调用的函数
  void CommonSetUp() {
    // 创建一个 BlockManagerConfig 对象，用于配置 BatchSchedulerEnvironmentSimulator
    block_manager_config.host_allocator_config.blocks_num = 100;
    block_manager_config.device_allocator_config.blocks_num = 100;
    block_manager_config.device_allocator_config.block_token_num = 6;
    device_num = 2;

    std::shared_ptr<FakedBlockAllocatorGroup> block_allocator_group =
        std::make_shared<FakedBlockAllocatorGroup>(block_manager_config, device_num);

    // 使用配置创建一个 BlockManagerSimulator 对象
    env_simulator = new BatchSchedulerEnvironmentSimulator(block_manager_config, device_num, block_allocator_group);
    KLLM_LOG_INFO << "Simulator start";
  }

  // 在每个测试用例执行之后调用的函数
  void TearDown() override {
    // 删除 BatchScheduler 对象
    if (env_simulator) {
      delete env_simulator;
    }
  }

  void InitRequestBlock(std::shared_ptr<InferRequest>& req, int output_token_num) {
    // Allocate all blocks at the beginning;
    int block_token_num = block_manager_config.device_allocator_config.block_token_num;
    int total_block_num = (req->input_tokens.size() + output_token_num + block_token_num - 1) / block_token_num;
    KLLM_LOG_INFO << "Start init req " << req->req_id << ", block num =" << total_block_num;
    KLLM_CHECK_WITH_INFO(req->kv_cache_blocks.size() == (size_t)device_num,
                         FormatStr("req->kv_cache_blocks.size()=%d", req->kv_cache_blocks.size()));
    for (int i = 0; i < device_num; i++) {
      std::vector<int> blocks;
      env_simulator->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(i)->AllocateBlocks(total_block_num, blocks);
      req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
      KLLM_LOG_INFO << "req " << req->req_id << ", kv_cache_blocks[" << i
                    << "].size()=" << req->kv_cache_blocks[i].size();
    }
    KLLM_LOG_INFO << "Init infer request " << req->req_id << " block=" << total_block_num;
  }

 protected:
  // 定义一个 BatchSchedulerEnvironmentSimulator 指针，用于在测试用例中使用
  BatchSchedulerEnvironmentSimulator* env_simulator = nullptr;

  BlockManagerConfig block_manager_config;
  int device_num;
};

TEST_F(BatchSchedulerEnvironmentSimulatorTest, BasicTokenGenerationTest) {
  CommonSetUp();
  int expected_output_token_num1 = 30;
  int input_token_num1 = 20;

  int seed_1_token_num = 20;
  int seed_2_token_num = 30;
  int seed_0 = 20;
  int seed_1 = 30;
  int seed_2 = 40;
  int expected_output_token_num2 = seed_1_token_num + seed_2_token_num;
  int input_token_num2 = 30;

  std::shared_ptr<Request> req1, req2, req2_same_seed, req2_diff_seed, req2_same20_diff30;
  std::shared_ptr<InferRequest> infer_req1, infer_req2, infer_req2_same_seed, infer_req2_diff_seed,
      infer_req2_same20_diff30;

  std::vector<std::pair<int, int>> seeds;
  seeds.push_back(std::make_pair(0, 1));
  // Init req1
  std::vector<std::shared_ptr<InferRequest>> infer_req_list =
      env_simulator->InitRequest(1, input_token_num1, expected_output_token_num1, req1, seeds);
  infer_req1 = infer_req_list[0];
  InitRequestBlock(infer_req1, expected_output_token_num1);

  // Init req2
  seeds[0].second = seed_0;
  infer_req_list = env_simulator->InitRequest(2, input_token_num2, expected_output_token_num2, req2, seeds);
  infer_req2 = infer_req_list[0];
  InitRequestBlock(infer_req2, expected_output_token_num2);

  // Init req2_same_seed
  seeds.push_back(std::make_pair(input_token_num2, seed_0));
  infer_req_list = env_simulator->InitRequest(3, input_token_num2, expected_output_token_num2, req2_same_seed, seeds);
  infer_req2_same_seed = infer_req_list[0];
  InitRequestBlock(infer_req2_same_seed, expected_output_token_num2);

  // Init req2_diff_seed
  seeds[1].second = seed_1;
  infer_req_list = env_simulator->InitRequest(4, input_token_num2, expected_output_token_num2, req2_diff_seed, seeds);
  infer_req2_diff_seed = infer_req_list[0];
  InitRequestBlock(infer_req2_diff_seed, expected_output_token_num2);

  // Init req2_same20_diff30
  seeds.push_back(std::make_pair(input_token_num2 + seed_1_token_num, seed_2));
  infer_req_list =
      env_simulator->InitRequest(5, input_token_num2, expected_output_token_num2, req2_same20_diff30, seeds);
  infer_req2_same20_diff30 = infer_req_list[0];
  InitRequestBlock(infer_req2_same20_diff30, expected_output_token_num2);

  std::vector<std::shared_ptr<InferRequest>> infer_reqs;
  infer_reqs.push_back(infer_req1);
  infer_reqs.push_back(infer_req2);
  infer_reqs.push_back(infer_req2_same_seed);
  infer_reqs.push_back(infer_req2_diff_seed);
  infer_reqs.push_back(infer_req2_same20_diff30);

  int max_output_step = std::max(expected_output_token_num1, expected_output_token_num2) + 1;
  for (int i = 0; i < max_output_step; i++) {
    std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;
    for (auto& req : infer_reqs) {
      if (req->sampling_result_tokens.size() > 0) {
        req->output_tokens.push_back(req->sampling_result_tokens.back());
        req->sampling_result_tokens.clear();
      }
      req->forwarding_tokens = req->output_tokens;
      if (!env_simulator->IsRequestFinished(req)) {
        scheduled_reqs.push_back(req);
      }
    }

    if (scheduled_reqs.empty()) break;
    KLLM_LOG_DEBUG << "Step " << i << ": scheduled_reqs.size(): " << scheduled_reqs.size();
    env_simulator->RunAStep(scheduled_reqs);
    for (auto req : scheduled_reqs) {
      KLLM_LOG_DEBUG << "Step " << i << ": req_id:" << req->req_id
                     << ", output_token.size()=" << req->output_tokens.size()
                     << ", sampling_result_tokens.size()=" << req->sampling_result_tokens.size()
                     << ", sampling_result_tokens token= " << req->sampling_result_tokens.back();
    }
  }

  // Check request results
  for (auto& req : infer_reqs) {
    env_simulator->CheckRequestOutput(req);
  }

  // Check seed generation results
  // input token should be same
  for (int i = 0; i < input_token_num2; i++) {
    int input_token = infer_req2->input_tokens[i];
    EXPECT_EQ(input_token, infer_req2_same_seed->input_tokens[i]);
    EXPECT_EQ(input_token, infer_req2_diff_seed->input_tokens[i]);
    EXPECT_EQ(input_token, infer_req2_same20_diff30->input_tokens[i]);
  }

  for (int i = 0; i < expected_output_token_num2 - 1; i++) {
    int offset = input_token_num2 + i;
    int output2 = infer_req2->output_tokens[offset];
    int output2_same = infer_req2_same_seed->output_tokens[offset];
    int output2_diff = infer_req2_diff_seed->output_tokens[offset];
    int output2_same20_diff30 = infer_req2_same20_diff30->output_tokens[offset];
    if (i < seed_1_token_num) {
      EXPECT_EQ(output2, output2_same);                // same seed_0
      EXPECT_EQ(output2_diff, output2_same20_diff30);  // same seed_1
      EXPECT_NE(output2, output2_diff);                // seed_0 vs seed_1
    } else {
      EXPECT_EQ(output2, output2_same);                // same seed_0
      EXPECT_NE(output2_diff, output2_same20_diff30);  // seed_1 vs seed_2
      EXPECT_NE(output2, output2_diff);                // seed_0 vs seed_1
      EXPECT_NE(output2, output2_same20_diff30);       // seed_0 vs seed_2
    }
  }
}

std::string KvCaches2Str(const std::vector<std::vector<int>>& kv_cache_blocks) {
  std::ostringstream ss;
  ss << "blocks={ ";
  for (size_t i = 0; i < kv_cache_blocks.size(); i++) {
    auto& blocks = kv_cache_blocks[i];
    ss << i << "={ ";
    for (auto blk_id : blocks) {
      ss << blk_id << ", ";
    }
    ss << "} ";
  }
  return ss.str();
}

// All request must have same prefix tokens
// Requests should not trigger swapout or rerun
class FixPrefixBatchScheduler : public BatchSchedulerInterface {
 public:
  FixPrefixBatchScheduler(BatchSchedulerEnvironmentSimulator* env_simulator, int prefix_token_num, int tp_num)
      : env_simulator_(env_simulator), prefix_token_num_(prefix_token_num), tp_num_(tp_num) {
    block_token_num_ = env_simulator_->GetBlockManagerConfig().device_allocator_config.block_token_num;
    KLLM_CHECK_WITH_INFO(
        prefix_token_num % block_token_num_ == 0,
        FormatStr("prefix_token_num(%d) is not aligned with block_token_num(%d)", prefix_token_num, block_token_num_));
    prefix_block_num_ = prefix_token_num_ / block_token_num_;

    PreparePrefixCacheBlocks();

    schedule_output = new ScheduleOutput();
  }

  virtual ~FixPrefixBatchScheduler() {
    if (schedule_output != nullptr) {
      free(schedule_output);
      schedule_output = nullptr;
    }
  }

  std::shared_ptr<CacheManagerInterface>& GetCacheManager(int attn_dp_idx) { return dummy_cache_mgr_; }

  std::shared_ptr<ScheduleOutputGroup> Schedule(size_t pp_batch_idx) override {
    assert(pp_batch_idx == 0);  // only support one batch
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    KLLM_LOG_DEBUG << " ============= Schedule, step " << step_ << ", running_reqs.size=" << running_reqs.size()
                   << ", waiting_reqs.size=" << waiting_reqs.size();

    std::lock_guard<std::mutex> guard(mux_);
    if (running_reqs.size() > 0) {
      for (auto it = running_reqs.begin(); it != running_reqs.end();) {
        // Update request status
        auto& req = *it;
        if (req->sampling_result_tokens.size() > 0) {
          req->output_tokens.push_back(req->sampling_result_tokens.back());
          req->sampling_result_tokens.clear();
        }
        if (env_simulator_->IsRequestFinished(req)) {
          ClearRequestAfterFinished(req);
          it = running_reqs.erase(it);
          continue;
        }
        // Need to add block?
        if (req->output_tokens.size() > (req->kv_cache_blocks[0].size() * block_token_num_)) {
          AdjustRequestKvCacheBlocks(req);
        }
        it++;
      }
      if (step_ == 1) {
        // Merge blocks with req[0]
        for (size_t i = 1; i < running_reqs.size(); i++) {
          auto& req = running_reqs[i];
          KLLM_LOG_DEBUG << "Merging blocks. req_id=" << req->req_id
                         << ", kv_cache_blocks=" << KvCaches2Str(req->kv_cache_blocks);
          for (int device_id = 0; device_id < tp_num_; device_id++) {
            std::vector<int> blocks_to_drop(prefix_block_num_);
            std::copy(req->kv_cache_blocks[device_id].begin(),
                      req->kv_cache_blocks[device_id].begin() + prefix_block_num_, blocks_to_drop.begin());
            env_simulator_->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(device_id)->FreeBlocks(blocks_to_drop);
          }
          FillPrefixCacheBlocks(req);
        }
      }
      step_++;
    }

    // Waiting req starts after running reqs finished.
    if (waiting_reqs.size() > 0 && running_reqs.size() == 0) {
      // Init request and add to running queue;
      if (step_ == 0) {
        // Only first request fill with prefix cache
        for (size_t i = 0; i < waiting_reqs.size(); i++) {
          auto& req = waiting_reqs[i];
          if (i == 0) {
            for (int device_id = 0; device_id < tp_num_; device_id++) {
              req->kv_cache_blocks[device_id].resize(prefix_block_num_);
            }
            FillPrefixCacheBlocks(req);
          }
          AdjustRequestKvCacheBlocks(req);
        }
      } else {
        // All requests fill with prefix cache
        for (auto& req : waiting_reqs) {
          for (int device_id = 0; device_id < tp_num_; device_id++) {
            req->kv_cache_blocks[device_id].resize(prefix_block_num_);
          }
          FillPrefixCacheBlocks(req);
          AdjustRequestKvCacheBlocks(req);
        }
      }
      running_reqs.swap(waiting_reqs);
      step_++;
    }
    KLLM_LOG_DEBUG << " ========= Schedule, running_reqs.size = " << running_reqs.size();
    for (auto& req : running_reqs) {
      req->forwarding_tokens = req->output_tokens;
    }
    schedule_output->running_reqs = running_reqs;

    std::shared_ptr<ScheduleOutputGroup> schedule_output_group = std::make_shared<ScheduleOutputGroup>();
    schedule_output_group->outputs[0] = schedule_output;
    return schedule_output_group;
  }

  virtual Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) override {
    for (auto& req : infer_request_group) {
      AddAnInferRequest(req);
    }
    return Status();
  }

  virtual void SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager, int idx) override {}

  virtual bool IsIdle(size_t pp_batch_idx) override { return false; }

  virtual void WaitUntilHaveReqs(size_t pp_batch_idx) override {}

  virtual void Stop() override {}

 private:
  Status AddAnInferRequest(std::shared_ptr<InferRequest>& infer_req) {
    std::lock_guard<std::mutex> guard(mux_);
    auto& input_tokens = infer_req->input_tokens;
    if (prefix_cache_tokens_.empty()) {
      // init with the first request
      prefix_cache_tokens_.resize(prefix_token_num_, 0);
      std::copy(input_tokens.begin(), input_tokens.begin() + prefix_token_num_, prefix_cache_tokens_.begin());
    }
    // set prefix info
    infer_req->is_use_prefix_cache = true;
    infer_req->prefix_cache_len = prefix_token_num_;
    waiting_reqs.push_back(infer_req);
    return Status();
  }

  void ClearRequestAfterFinished(std::shared_ptr<InferRequest>& infer_req) {
    ASSERT_TRUE(env_simulator_->IsRequestFinished(infer_req));
    KLLM_LOG_DEBUG << "ClearRequestAfterFinished. req_id=" << infer_req->req_id
                   << ", output_tokens.size=" << infer_req->output_tokens.size()
                   << ", kv_cache_blocks=" << KvCaches2Str(infer_req->kv_cache_blocks);
    infer_req->finished = true;
    infer_req->Notify();
    return;
    // Free blocks after prefix blocks
    auto& kv_cache_blocks = infer_req->kv_cache_blocks;
    for (int device_id = 0; device_id < tp_num_; device_id++) {
      std::vector<int> free_blocks(kv_cache_blocks[device_id].size() - prefix_block_num_);
      std::copy(kv_cache_blocks[device_id].begin() + prefix_block_num_, kv_cache_blocks[device_id].end(),
                free_blocks.begin());
      env_simulator_->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(device_id)->FreeBlocks(free_blocks);
    }
  }

  void PreparePrefixCacheBlocks() {
    for (int device_id = 0; device_id < tp_num_; device_id++) {
      std::vector<int> prefix_cache_block_tmp;
      env_simulator_->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(device_id)->AllocateBlocks(
          prefix_block_num_, prefix_cache_block_tmp);
      prefix_cache_blocks_.emplace_back(std::move(prefix_cache_block_tmp));
    }
    KLLM_LOG_INFO << "set prefix_cache_blocks =" << KvCaches2Str(prefix_cache_blocks_);
  }

  bool CheckReqIsValidForPrefixCache(const std::vector<int>& input_tokens) {
    for (int token_idx = 0; token_idx < prefix_token_num_; ++token_idx) {
      if (prefix_cache_tokens_[token_idx] != input_tokens[token_idx]) {
        return false;
      }
    }
    return true;
  }

  void FillPrefixCacheBlocks(std::shared_ptr<InferRequest>& req) {
    auto& kv_cache_blocks = req->kv_cache_blocks;
    for (int device_id = 0; device_id < tp_num_; device_id++) {
      for (int i = 0; i < prefix_block_num_; i++) {
        kv_cache_blocks[device_id][i] = prefix_cache_blocks_[device_id][i];
      }
    }
    KLLM_LOG_DEBUG << " Fill prefix, req_id= " << req->req_id << ", after fill = " << KvCaches2Str(kv_cache_blocks);
  }

  void AdjustRequestKvCacheBlocks(const std::shared_ptr<InferRequest>& req) {
    auto& kv_cache_blocks = req->kv_cache_blocks;
    int adding_block_num =
        (req->output_tokens.size() + block_token_num_ - 1) / block_token_num_ - kv_cache_blocks[0].size();
    for (int i = 0; i < tp_num_; i++) {
      std::vector<int> new_blocks;
      env_simulator_->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(i)->AllocateBlocks(adding_block_num,
                                                                                           new_blocks);
      kv_cache_blocks[i].insert(kv_cache_blocks[i].end(), new_blocks.begin(), new_blocks.end());
    }
    KLLM_LOG_DEBUG << "AdjustRequestKvCache, req_id= " << req->req_id
                   << ", output_tokens.size()=" << req->output_tokens.size() << ", add " << adding_block_num
                   << " blocks, kv_cache=" << KvCaches2Str(req->kv_cache_blocks);
  }

 private:
  BatchSchedulerEnvironmentSimulator* env_simulator_;
  int prefix_token_num_;
  int tp_num_;
  int block_token_num_;

  int step_ = 0;
  std::vector<std::shared_ptr<InferRequest>> waiting_reqs;
  std::vector<std::shared_ptr<InferRequest>> running_reqs;

  int prefix_block_num_;
  std::vector<std::vector<int>> prefix_cache_blocks_;
  std::vector<int> prefix_cache_tokens_;

  std::mutex mux_;
  std::shared_ptr<CacheManagerInterface> dummy_cache_mgr_ = nullptr;

  ScheduleOutput* schedule_output = nullptr;
};

TEST_F(BatchSchedulerEnvironmentSimulatorTest, FixPrefixCacheScheduleTest) {
  int prefix_block_num = 3;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num);
  FixPrefixBatchScheduler batch_scheduler(test_case.GetEnvSimulator(), prefix_block_num * block_token_num, device_num);
  test_case.SetBatchScheduler(&batch_scheduler);
  test_case.RunTestNoSwapTriggered();
}
