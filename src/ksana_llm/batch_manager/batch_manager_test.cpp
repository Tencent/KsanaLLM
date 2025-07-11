/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include "ksana_llm/batch_manager/batch_manager.h"
#include "ksana_llm/batch_scheduler/batch_scheduler_test_helper.h"
#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/helpers/environment_test_helper.h"
#include "ksana_llm/runtime/llm_runtime.h"
#include "ksana_llm/runtime/weight_instance.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/waiter.h"

using namespace ksana_llm;

#define PP_BATCH_NUM 2

class BatchManagerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }

  void SetUp() override {
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
    context_ = std::make_shared<Context>(1, 1, PP_BATCH_NUM);
    batch_manager_ = std::make_unique<BatchManager>(context_, PP_BATCH_NUM);
    if (context_->IsChief()) {
      multi_batch_controller_ = std::make_shared<MultiBatchController>(PP_BATCH_NUM);
    }
    batch_manager_->SetMultiBatchController(multi_batch_controller_);
    memory_allocator_ = std::make_shared<MemoryAllocator>();
  }

  void TearDown() override {
    batch_manager_.reset();
    context_.reset();
  }

 protected:
  // NOTE(karlluo): model instance need shared_ptr as input parameter
  std::shared_ptr<Context> context_ = nullptr;
  std::unique_ptr<BatchManager> batch_manager_ = nullptr;
  std::shared_ptr<MultiBatchController> multi_batch_controller_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;

  BlockManagerConfig block_manager_config_;
  BatchSchedulerConfig batch_scheduler_config_;
  CacheManagerConfig cache_manager_config_;

 protected:
  void InitDefaultConfig(const size_t data_para_size, const size_t tensor_para_size) {
    int device_block_num = 100;
    block_manager_config_.host_allocator_config.blocks_num = device_block_num * tensor_para_size * 2;
    block_manager_config_.device_allocator_config.blocks_num = device_block_num;
    block_manager_config_.device_allocator_config.block_token_num = 6;

    batch_scheduler_config_.schedule_strategy = static_cast<ScheduleStrategy>(0);
    batch_scheduler_config_.waiting_timeout_in_ms = 600000;
    batch_scheduler_config_.max_waiting_queue_len = 256;
    batch_scheduler_config_.max_step_token_num = 4096;
    batch_scheduler_config_.max_batch_size = 1;
    batch_scheduler_config_.max_pp_batch_num = PP_BATCH_NUM;
    batch_scheduler_config_.max_token_len = 1024;
    batch_scheduler_config_.swapout_block_threshold = 1.0;
    batch_scheduler_config_.swapin_block_threshold = 2.0;
    batch_scheduler_config_.launch_block_threshold = 2.0;
    batch_scheduler_config_.preempt_mode = static_cast<PreemptMode>(0);

    cache_manager_config_.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
    cache_manager_config_.tensor_para_size = tensor_para_size;
    cache_manager_config_.swap_threadpool_size = 8;
    cache_manager_config_.enable_prefix_caching = false;
  }

  void PrepareTestCaseMeterial(const size_t data_para_size, const size_t tensor_para_size,
                               std::shared_ptr<BatchSchedulerInterface>& batch_scheduler,
                               std::shared_ptr<CacheManagerInterface>& cache_manager) {
    InitDefaultConfig(data_para_size, tensor_para_size);

    InitializeScheduleOutputPool();
    BlockAllocatorGroupConfig group_1_config;
    group_1_config.devices = {0};
    group_1_config.device_block_num = 100;
    group_1_config.host_block_num = 100;
    group_1_config.block_size = 16 * 1024 * 1024;

    BlockAllocatorManagerConfig block_allocator_manager_config;
    block_allocator_manager_config[1] = group_1_config;

    BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_);

    block_allocator_group_ = block_allocator_manager.GetBlockAllocatorGroup(1);

    batch_scheduler = std::make_shared<BatchScheduler>(batch_scheduler_config_, data_para_size, tensor_para_size);

    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config_, block_allocator_group_);
    cache_manager->InitializeCachedBlocks();
    for (size_t attn_dp_id = 0; attn_dp_id < data_para_size; ++attn_dp_id) {
      batch_scheduler->SetCacheManager(cache_manager, attn_dp_id);
    }
  }
};

TEST_F(BatchManagerTest, HiddenUnitBufferTest) {
  InitializeHiddenUnitBufferPool();
  HiddenUnitDeviceBuffer* hidden_unit_buffer = nullptr;
  const size_t test_multi_batch_id = 123;  // Define a test multi_batch_id

  hidden_unit_buffer = GetHiddenUnitBufferPool()->GetDeviceBuffer();
  EXPECT_TRUE(hidden_unit_buffer != nullptr);

  // Set the multi_batch_id for the buffer
  hidden_unit_buffer->multi_batch_id = test_multi_batch_id;
  SetCurrentHiddenUnitBuffer(hidden_unit_buffer);
  EXPECT_TRUE(GetCurrentHiddenUnitBuffer(test_multi_batch_id) == hidden_unit_buffer);

  GetHiddenUnitBufferPool()->FreeDeviceBuffer(hidden_unit_buffer);
  EXPECT_TRUE(GetHiddenUnitBufferPool()->GetDeviceBuffer() == hidden_unit_buffer);

  GetHiddenUnitBufferPool()->Stop();
  DestroyHiddenUnitBufferPool();
  EXPECT_TRUE(GetHiddenUnitBufferPool() == nullptr);
}

TEST_F(BatchManagerTest, Constructor) { EXPECT_NE(batch_manager_, nullptr); }

TEST_F(BatchManagerTest, RegisterModelInstance) {
  ModelConfig model_config;
  model_config.name = "test_model";
  model_config.end_ids = {1, 2};
  model_config.pad_id = 0;

  std::shared_ptr<WeightInstanceInterface> weight_instance = nullptr;

  auto model_instance = std::make_shared<ModelInstance>(model_config, context_, weight_instance);
  model_instance->name = "test_model";

  Status status = batch_manager_->RegisterModelInstance(model_instance);

  EXPECT_TRUE(status.OK());
}

TEST_F(BatchManagerTest, EnqueueModelNotFound) {
  std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx =
      std::make_shared<std::unordered_map<std::string, std::string>>();
  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input, req_ctx);
  request->req_id = 1;
  request->model_name = "non_existent_model";
  request->waiter = std::make_shared<Waiter>(1);

  Status status = batch_manager_->Enqueue(request);

  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RET_INVALID_ARGUMENT);
}

TEST_F(BatchManagerTest, StartAndStop) {
  int dp_num = 2;
  int tp_num = 1;
  // NOTE(karlluo): need shared_ptr as input parameter
  std::shared_ptr<BatchSchedulerInterface> batch_scheduler;
  std::shared_ptr<CacheManagerInterface> cache_manager;
  PrepareTestCaseMeterial(dp_num, tp_num, batch_scheduler, cache_manager);

  auto llm_runtime = std::make_shared<LlmRuntime>(batch_scheduler_config_, context_);
  llm_runtime->SetMultiBatchController(multi_batch_controller_);

  batch_manager_->SetBatchScheduler(batch_scheduler);
  batch_manager_->SetLlmRuntime(llm_runtime);

  Status start_status = batch_manager_->Start();
  EXPECT_TRUE(start_status.OK());

  Status stop_status = batch_manager_->Stop();
  EXPECT_TRUE(stop_status.OK());
}

TEST_F(BatchManagerTest, MergeScheduleOutputGroup) {
  int dp_num = 2;
  int tp_num = 1;
  // NOTE(karlluo): need shared_ptr as input parameter
  std::shared_ptr<BatchSchedulerInterface> batch_scheduler;
  std::shared_ptr<CacheManagerInterface> cache_manager;
  PrepareTestCaseMeterial(dp_num, tp_num, batch_scheduler, cache_manager);

  batch_manager_->SetBatchScheduler(batch_scheduler);

  batch_manager_->Start();

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  batch_manager_->Stop();
}

TEST_F(BatchManagerTest, WaitAllDone) {
  Status status = batch_manager_->WaitAllDone();
  EXPECT_TRUE(status.OK());
}
