/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/continuous_batching.h"

#include <limits>
#include <random>

#include "ksana_llm/cache_manager/block_allocator/block_allocator_manager.h"
#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/transfer/transfer_engine.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tokenizer.h"

#include "test.h"

using namespace ksana_llm;

class ContinuousBatchingStrategyTest : public ContinuousBatchingStrategy {
 public:
  using ContinuousBatchingStrategy::AddTransferMeta;
  using ContinuousBatchingStrategy::ContinuousBatchingStrategy;
  using ContinuousBatchingStrategy::ProcessDecodeTransferQueue;
  using ContinuousBatchingStrategy::ProcessPrefillTransferQueue;

  void SetSplitFuseTokenNum(const size_t val) { batch_scheduler_config_.split_fuse_token_num = val; }

  void SetConnectorRole(GroupRole role) {
    connector_config_.group_role = role;
    TransferEngine::GetInstance()->Initialize(connector_config_.group_role);
    TransferEngine::GetInstance()->SetGroupRole(role);
  }
};

class ContinuousBatchingTest : public testing::Test {
 protected:
  ContinuousBatchingTest() {
    InitializeScheduleOutputPool();
    constexpr int kTpNum = 2;
    constexpr int kAttnDpNum = 2;
    constexpr int kMultiBatchNum = 1;
    BatchSchedulerConfig batch_scheduler_config;
    batch_scheduler_config.split_fuse_token_num = 256;
    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = test_info->name();
    if (test_name.find("ProcessDecodeTransferQueue") != std::string::npos ||
        test_name.find("ProcessMTPDecodeTransferQueueTest") != std::string::npos) {
      auto env = Singleton<Environment>::GetInstance();
      ConnectorConfig connector_config;
      connector_config.group_role = GroupRole::DECODE;
      connector_config.router_addr = "127.0.0.1:13579";
      env->SetConnectorConfigs(connector_config);
    }
    RuntimeConfig runtime_config;
    if (test_name.find("ProcessMTPDecodeTransferQueueTest") != std::string::npos) {
      runtime_config.mtp_step_num = 1;
    }
    continuous_batching_strategy_ =
        std::make_shared<ContinuousBatchingStrategyTest>(batch_scheduler_config, runtime_config);
    size_t multi_batch_id = 0;
    continuous_batching_strategy_->SetBatchState(std::make_shared<BatchState>(multi_batch_id, batch_scheduler_config));

    BlockAllocatorGroupConfig group_1_config;
    group_1_config.devices = {0, 1};
    group_1_config.device_block_num = 100;
    group_1_config.host_block_num = 100;
    group_1_config.block_size = 16 * 1024 * 1024;

    BlockAllocatorManagerConfig block_allocator_manager_config;
    block_allocator_manager_config[1] = group_1_config;

    std::shared_ptr<Context> context = std::make_shared<Context>(kTpNum, kAttnDpNum, kMultiBatchNum);
    std::shared_ptr<MemoryAllocator> memory_allocator_ = std::make_shared<MemoryAllocator>();
    BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context);

    std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group =
        block_allocator_manager.GetBlockAllocatorGroup(1);

    CacheManagerConfig cache_manager_config;
    cache_manager_config.tensor_para_size = kTpNum;
    cache_manager_config.enable_prefix_caching = false;
    auto cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group);
    cache_manager->InitializeCachedBlocks();
    continuous_batching_strategy_->SetCacheManager(cache_manager);

    // 初始化调度同步器与共享计数器，避免 SchedulerTickTok::Barrier 未初始化导致的测试崩溃
    auto scheduler_shared_counter = std::make_shared<SchedulerSharedCounter>(1);
    auto scheduler_ticktok = std::make_shared<SchedulerTickTok>(1);
    continuous_batching_strategy_->SetSharedCounter(scheduler_shared_counter);
    continuous_batching_strategy_->SetSchedulerTickTok(scheduler_ticktok);
    continuous_batching_strategy_->SetDataParaGroupId(0);

    pybind11::initialize_interpreter();
  }

  ~ContinuousBatchingTest() {
    DestroyScheduleOutputPool();
    TransferEngine::GetInstance()->CleanupTransferMeta(123);
    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = test_info->name();
    if (test_name.find("ProcessDecodeTransferQueueTest") != std::string::npos) {
      auto env = Singleton<Environment>::GetInstance();
      ConnectorConfig connector_config;
      connector_config.group_role = GroupRole::NONE;
      connector_config.router_addr = "127.0.0.1:13579";
      env->SetConnectorConfigs(connector_config);
      BlockManagerConfig block_manager_config;
      env->SetBlockManagerConfig(block_manager_config);
    }
    pybind11::finalize_interpreter();
  }

  void SetUp() {}

 protected:
  std::shared_ptr<ContinuousBatchingStrategyTest> continuous_batching_strategy_ = nullptr;
};
/*
// test function: ProcessSplitFuseTokenTest, check the split size
TEST_F(ContinuousBatchingTest, ProcessSplitFuseTokenTest) {
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->block_token_num = 16;

  bool stop_schedule = false;
  size_t shared_block_num, unique_block_num;    // not use in this test
  size_t shared_token_num = 4;                  // random init
  size_t decode_request_num = 1;                // have decode request
  size_t step_token_without_kv_cache_num = 10;  // random init
  size_t forwarding_len;
  size_t split_fuse_token_num;

  constexpr size_t kMinLen = std::numeric_limits<size_t>::min();
  constexpr size_t kMaxLen = 16384;  // avoid overflow
  constexpr size_t kMultiTestTimes = 30;

  auto RandomNum = [](const size_t min, const size_t max) {
    static std::default_random_engine random_engine;
    return std::uniform_int_distribution<size_t>(min, max)(random_engine);
  };

  // disable, forwarding not change
  split_fuse_token_num = 0;  // disable split fuse
  continuous_batching_strategy_->SetSplitFuseTokenNum(split_fuse_token_num);
  forwarding_len = RandomNum(kMinLen, kMaxLen);  // a random number, any len will not change
  req->forwarding_tokens.resize(forwarding_len);
  stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
      req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
  EXPECT_FALSE(stop_schedule);
  EXPECT_EQ(req->forwarding_tokens.size(), forwarding_len);

  // enable split fuse
  split_fuse_token_num = 256;
  continuous_batching_strategy_->SetSplitFuseTokenNum(split_fuse_token_num);

  // Current step already meets quota
  step_token_without_kv_cache_num = RandomNum(split_fuse_token_num, kMaxLen);  // a number >= split_fuse_token_num
  forwarding_len = RandomNum(kMinLen, kMaxLen);  // a random number, any len will not change
  req->forwarding_tokens.resize(forwarding_len);
  stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
      req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
  EXPECT_FALSE(stop_schedule);
  EXPECT_EQ(req->forwarding_tokens.size(), forwarding_len);

  // no decode request
  decode_request_num = 0;                                         // no decode request
  step_token_without_kv_cache_num = RandomNum(kMinLen, kMaxLen);  // a random number, no impact in this case
  for (size_t i = 0; i < kMultiTestTimes; ++i) {
    forwarding_len = RandomNum(kMinLen,
                               kMaxLen);  // a random number, any len will not change
    req->forwarding_tokens.resize(forwarding_len);
    stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
        req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
    EXPECT_FALSE(stop_schedule);
    EXPECT_EQ(req->forwarding_tokens.size(), forwarding_len);
  }
  decode_request_num = 1;  // exist decode request, and number > 0

  // forwading < split_fuse && have decode
  step_token_without_kv_cache_num = RandomNum(1, split_fuse_token_num);  // a number < split_fuse_token_num
  shared_token_num = 0;                                                  // no kv_cache in forwarding_tokens
  // step_token_without_kv_cache_num + forwarding_len <= split_fuse_token_num, len will not change
  forwarding_len = RandomNum(1, split_fuse_token_num - step_token_without_kv_cache_num);
  req->forwarding_tokens.resize(forwarding_len);
  stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
      req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
  EXPECT_FALSE(stop_schedule);
  EXPECT_EQ(req->forwarding_tokens.size(), forwarding_len);

  // forwading < block > split_fuse_remain && have decode
  // reamin split fuse len less than a block
  step_token_without_kv_cache_num = split_fuse_token_num - req->block_token_num / 2;  // reamin less then a block
  shared_token_num = 0;  // no kv_cache in forwarding_tokens
  forwarding_len = RandomNum(split_fuse_token_num - step_token_without_kv_cache_num + 1,
                             kMaxLen);  // large than reamin
  req->forwarding_tokens.resize(forwarding_len);
  stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
      req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
  EXPECT_TRUE(stop_schedule);
  EXPECT_EQ(req->forwarding_tokens.size(), forwarding_len);

  // split fuse with prefix
  step_token_without_kv_cache_num = RandomNum(
      kMinLen, split_fuse_token_num - req->block_token_num);  // a random number < split_fuse_token_num - block_num
  size_t prefix_num = 64;                                     // input have some prefix cache
  shared_token_num = prefix_num;
  forwarding_len =
      shared_token_num + RandomNum(split_fuse_token_num, kMaxLen);  // a large input, exceed split fuse remain len
  req->forwarding_tokens.resize(forwarding_len);
  stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
      req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
  EXPECT_FALSE(stop_schedule);
  EXPECT_EQ(req->forwarding_tokens.size(), prefix_num + (split_fuse_token_num - step_token_without_kv_cache_num) /
                                                            req->block_token_num * req->block_token_num);

  // split fuse
  decode_request_num = 1;  // have decode request, a number > 0
  for (size_t i = 0; i < kMultiTestTimes; ++i) {
    step_token_without_kv_cache_num =
        RandomNum(kMinLen, split_fuse_token_num - req->block_token_num);  // remain >= 1 block
    shared_token_num = RandomNum(kMinLen, split_fuse_token_num / req->block_token_num) * req->block_token_num;
    forwarding_len = shared_token_num + RandomNum(kMinLen, kMaxLen);  // random len
    req->forwarding_tokens.resize(forwarding_len);

    size_t expect_len;
    const size_t remain = split_fuse_token_num - step_token_without_kv_cache_num;
    if (forwarding_len - shared_token_num <= remain) {
      expect_len = forwarding_len;
    } else {
      expect_len = shared_token_num + (remain / req->block_token_num * req->block_token_num);
    }

    stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
        req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
    EXPECT_FALSE(stop_schedule);
    EXPECT_EQ(req->forwarding_tokens.size(), expect_len);
  }
}
*/
// 测试ProcessDecodeTransferQueue函数
TEST_F(ContinuousBatchingTest, ProcessDecodeTransferQueueTest) {
  // 设置为DECODE节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::DECODE);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->kv_comm_request_id = 123;
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();

  // 设置block_token_num，防止添加元数据时报错
  req->block_token_num = 16;

  // 设置request调度状态
  req->input_tokens = {1, 2, 3, 4, 5};
  req->output_tokens = req->input_tokens;
  req->ResetPrefillingTokens();

  // 将请求添加到传输队列
  continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);

  // 调用ProcessDecodeTransferQueue函数
  continuous_batching_strategy_->ProcessDecodeTransferQueue();

  // 验证结果
  // 由于TransferEngine::IsRecvDone返回-1（未完成），请求应该仍在传输队列中
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), 1);
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->decoding_queue.size(), 0);

  // 清理
  continuous_batching_strategy_->batch_state_->transfer_queue.clear();

  for (int i = 0; i < 20; ++i) {
    auto req = std::make_shared<InferRequest>(request, 0);
    req->cache_manager = continuous_batching_strategy_->GetCacheManager();

    // 设置KV缓存块
    req->kv_cache_blocks.resize(2);
    for (size_t i = 0; i < 2; ++i) {
      req->kv_cache_blocks[i].resize(3);
      for (size_t j = 0; j < 3; ++j) {
        req->kv_cache_blocks[i][j] = j + i * 10;
      }
    }
    req->kv_comm_request_id = 123 + i;
    // 设置block_token_num，防止添加元数据时报错
    req->block_token_num = 16;

    // 设置request调度状态
    req->input_tokens = {1, 2, 3, 4, 5};
    req->output_tokens = req->input_tokens;
    req->ResetPrefillingTokens();

    std::vector<std::shared_ptr<InferRequest>> queue;
    queue.push_back(req);

    // 调用AddTransferMeta函数
    continuous_batching_strategy_->AddTransferMeta(queue);
    continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);
  }
  // 调用ProcessDecodeTransferQueue函数
  continuous_batching_strategy_->ProcessDecodeTransferQueue();

  // 验证结果, 计算8个预传输12个
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), 12);
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->decoding_queue.size(), 8);

  for (int i = 0; i < 20; ++i) {
    TransferEngine::GetInstance()->CleanupTransferMeta(123 + i);
  }
}

// 当 step_batch_size >= dp_max_decode_batch_size_ 时不再增加 running_reqs
TEST_F(ContinuousBatchingTest, ProcessDecodeTransferQueueMaxBatchStopTest) {
  // 设置为DECODE节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::DECODE);

  // 预先填充 running_reqs 到达到最大 decode batch size
  const size_t max_decode_bs = continuous_batching_strategy_->batch_state_->batch_scheduler_config_.max_batch_size;

  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);

  for (size_t i = 0; i < max_decode_bs; ++i) {
    auto dummy_req = std::make_shared<InferRequest>(request, 0);
    continuous_batching_strategy_->batch_state_->decoding_queue.push_back(dummy_req);
  }
  size_t transfer_req_num = 20;
  continuous_batching_strategy_->batch_state_->transfer_queue.clear();
  // 构造若干 transfer_queue 请求，若处理应进入 running_reqs，但由于达到上限，应被阻止
  for (size_t i = 0; i < transfer_req_num; ++i) {
    auto req = std::make_shared<InferRequest>(request, 0);
    req->cache_manager = continuous_batching_strategy_->GetCacheManager();

    // 设置KV缓存块
    req->kv_cache_blocks.resize(2);
    for (size_t i = 0; i < 2; ++i) {
      req->kv_cache_blocks[i].resize(3);
      for (size_t j = 0; j < 3; ++j) {
        req->kv_cache_blocks[i][j] = j + i * 10;
      }
    }
    req->kv_comm_request_id = 123 + i;
    // 设置block_token_num，防止添加元数据时报错
    req->block_token_num = 16;

    // 设置request调度状态
    req->input_tokens = {1, 2, 3, 4, 5};
    req->output_tokens = req->input_tokens;
    req->ResetPrefillingTokens();

    std::vector<std::shared_ptr<InferRequest>> queue;
    queue.push_back(req);

    // 调用AddTransferMeta函数
    continuous_batching_strategy_->AddTransferMeta(queue);
    continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);
  }

  // 调用处理函数
  continuous_batching_strategy_->ProcessDecodeTransferQueue();

  // 验证：running_reqs 未超过最大 decode batch size，上限触发后不再增加
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->decoding_queue.size(), max_decode_bs);
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), transfer_req_num);

  // 清理传输元数据
  for (size_t i = 0; i < transfer_req_num; ++i) {
    TransferEngine::GetInstance()->CleanupTransferMeta(123 + i);
  }
}

// 测试MTP开启时的draft token传输
TEST_F(ContinuousBatchingTest, ProcessMTPDecodeTransferQueueTest) {
  // 设置为DECODE节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::DECODE);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);

  for (int i = 0; i < 10; ++i) {
    auto req = std::make_shared<InferRequest>(request, 0);
    req->cache_manager = continuous_batching_strategy_->GetCacheManager();

    // 设置KV缓存块
    req->kv_cache_blocks.resize(2);
    for (size_t i = 0; i < 2; ++i) {
      req->kv_cache_blocks[i].resize(3);
      for (size_t j = 0; j < 3; ++j) {
        req->kv_cache_blocks[i][j] = j + i * 10;
      }
    }
    req->kv_comm_request_id = 123 + i;
    // 设置block_token_num，防止添加元数据时报错
    req->block_token_num = 16;

    // 设置request调度状态
    req->input_tokens = {1, 2, 3};
    req->output_tokens = req->input_tokens;
    req->ResetPrefillingTokens();

    std::vector<std::shared_ptr<InferRequest>> queue;
    queue.push_back(req);
    // 调用AddTransferMeta函数
    continuous_batching_strategy_->AddTransferMeta(queue);
    continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);
  }

  // 调用ProcessDecodeTransferQueue函数
  continuous_batching_strategy_->ProcessDecodeTransferQueue();
  for (auto running_req : continuous_batching_strategy_->batch_state_->decoding_queue) {
    ASSERT_EQ(running_req->forwarding_tokens.size(), 3);
    ASSERT_EQ(running_req->draft_tokens.GetDraftTokens()[0], 1);
    ASSERT_EQ(running_req->generated_tokens[0], 1);
    ScheduleTaskWorkload planning_workload = running_req->GetPlanningWorkload();
    EXPECT_EQ(planning_workload.prefill_token_num, 0);
    EXPECT_EQ(planning_workload.generated_token_num, 1);
    EXPECT_EQ(planning_workload.draft_token_num, 1);
  }

  for (int i = 0; i < 10; ++i) {
    TransferEngine::GetInstance()->CleanupTransferMeta(123 + i);
  }
}

// 测试AddTransferMeta函数
TEST_F(ContinuousBatchingTest, AddTransferMetaTest) {
  // 设置为PREFILL节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::PREFILL);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->kv_comm_request_id = 123;  // 设置请求ID
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();

  // 设置KV缓存块
  req->kv_cache_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    req->kv_cache_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      req->kv_cache_blocks[i][j] = j + i * 10;
    }
  }

  // 设置kv_cached_token_num和block_token_num，防止添加元数据时报错
  req->kv_cached_token_num = 16;
  req->block_token_num = 16;

  // 创建请求队列
  std::vector<std::shared_ptr<InferRequest>> queue;
  queue.push_back(req);

  // 调用AddTransferMeta函数
  continuous_batching_strategy_->AddTransferMeta(queue);

  // 验证结果
  // 对于PREFILL节点，max_new_tokens应该被设置为1
  ASSERT_EQ(req->sampling_config.max_new_tokens, 1);

  // 验证传输元数据是否已添加
  auto transfer_engine = TransferEngine::GetInstance();
  auto meta = transfer_engine->GetTransferMeta(req->kv_comm_request_id);
  ASSERT_NE(meta, nullptr);
}

// 测试ProcessPrefillTransferQueue函数
TEST_F(ContinuousBatchingTest, ProcessPrefillTransferQueueTest) {
  // 设置为PREFILL节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::PREFILL);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->kv_comm_request_id = 123;
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();

  // 设置kv_cached_token_num和block_token_num，防止添加元数据时报错
  req->kv_cached_token_num = 0;
  req->block_token_num = 16;

  // 将请求添加到传输队列
  continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);

  // 调用ProcessPrefillTransferQueue函数
  continuous_batching_strategy_->ProcessPrefillTransferQueue();

  // 验证结果
  // 由于TransferEngine::IsSendDone返回false（未完成），请求应该仍在传输队列中
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), 1);

  // 清理
  continuous_batching_strategy_->batch_state_->transfer_queue.clear();
}

// 测试ProcessTransferQueue函数（DECODE节点）
TEST_F(ContinuousBatchingTest, ProcessTransferQueueDecodeTest) {
  // 设置为DECODE节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::DECODE);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->kv_comm_request_id = 123;
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();

  // 设置block_token_num，防止添加元数据时报错
  req->block_token_num = 16;

  // 设置request调度状态
  req->input_tokens = {1, 2, 3, 4, 5};
  req->output_tokens = req->input_tokens;
  req->ResetPrefillingTokens();

  // 将请求添加到传输队列
  continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);

  // 调用ProcessTransferQueue函数
  continuous_batching_strategy_->ProcessTransferQueue();

  // 验证结果
  // 由于TransferEngine::IsRecvDone返回-1（未完成），请求应该仍在传输队列中
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), 1);
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->schedule_output->running_reqs.size(), 0);

  // 清理
  continuous_batching_strategy_->batch_state_->transfer_queue.clear();
}

// 测试ProcessTransferQueue函数（PREFILL节点）
TEST_F(ContinuousBatchingTest, ProcessTransferQueuePrefillTest) {
  // 设置为PREFILL节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::PREFILL);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->kv_comm_request_id = 123;
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();

  // 设置kv_cached_token_num和block_token_num，防止添加元数据时报错
  req->kv_cached_token_num = 0;
  req->block_token_num = 16;

  // 将请求添加到传输队列
  continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);

  // 调用ProcessTransferQueue函数
  continuous_batching_strategy_->ProcessTransferQueue();

  // 验证结果
  // 由于TransferEngine::IsSendDone返回false（未完成），请求应该仍在传输队列中
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), 1);

  // 清理
  continuous_batching_strategy_->batch_state_->transfer_queue.clear();
}
