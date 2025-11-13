/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/model_instance.h"

#include "ksana_llm/batch_manager/schedule_processor.h"
#include "ksana_llm/batch_scheduler/batch_scheduler.h"
#include "ksana_llm/batch_scheduler/batch_scheduler_test_helper.h"
#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/helpers/environment_test_helper.h"
#include "ksana_llm/runtime/llm_runtime.h"
#include "ksana_llm/runtime/weight_instance.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/waiter.h"

using namespace ksana_llm;

#define PP_BATCH_NUM 2

class ScheduleProcessorTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }

  void SetUp() override {
    ResetFakedState();
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
    context_ = std::make_shared<Context>(1, 1, PP_BATCH_NUM);
    Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config_);
    runtime_config_.max_pp_batch_num = PP_BATCH_NUM;
    if (context_->IsChief()) {
      multi_batch_controller_ = std::make_shared<MultiBatchController>(PP_BATCH_NUM);
    }
    memory_allocator_ = std::make_shared<MemoryAllocator>();
  }

  void TearDown() override { context_.reset(); }

 protected:
  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<MultiBatchController> multi_batch_controller_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;

  BlockManagerConfig block_manager_config_;
  BatchSchedulerConfig batch_scheduler_config_;
  CacheManagerConfig cache_manager_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::vector<std::shared_ptr<Request>> hold_requests_;
  // 持有 KsanaPythonInput 对象，确保其生命周期长于 Request（因为 Request 中有引用类型成员）
  std::vector<std::shared_ptr<KsanaPythonInput>> hold_python_inputs_;

  std::shared_ptr<BatchSchedulerInterface> batch_scheduler_ = nullptr;
  std::shared_ptr<CacheManagerInterface> cache_manager_ = nullptr;

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

  void PrepareTestCaseMaterial(const size_t data_para_size, const size_t tensor_para_size) {
    InitDefaultConfig(data_para_size, tensor_para_size);

    InitializeScheduleOutputPool();
    size_t tp_num = tensor_para_size;
    block_allocator_group_ = std::make_shared<FakedBlockAllocatorGroup>(block_manager_config_, tp_num);

    // Create ModelInstance
    ModelConfig model_config;
    model_config.name = "test_model";
    model_config.end_ids = {1, 2};
    model_config.pad_id = 0;
    std::shared_ptr<WeightInstanceInterface> weight_instance = nullptr;
    model_instance_ = std::make_shared<ModelInstance>(model_config, runtime_config_, context_, weight_instance);
    model_instance_->name = "test_model";

    std::vector<std::shared_ptr<ModelInstance>> model_instances;
    model_instances.push_back(model_instance_);
    runtime_config_.parallel_basic_config.tensor_parallel_size = tensor_para_size;
    runtime_config_.parallel_basic_config.attn_data_parallel_size = data_para_size;
    batch_scheduler_ =
        std::make_shared<BatchScheduler>(batch_scheduler_config_, runtime_config_, false, model_instances);

    cache_manager_ = std::make_shared<PrefixCacheManager>(cache_manager_config_, block_allocator_group_);
    cache_manager_->InitializeCachedBlocks();
    for (size_t attn_dp_id = 0; attn_dp_id < data_para_size; ++attn_dp_id) {
      batch_scheduler_->SetCacheManager(cache_manager_, attn_dp_id);
    }
  }
};

TEST_F(ScheduleProcessorTest, ProcessorConstructorAndInitialize) {
  int dp_num = 1;
  int tp_num = 1;
  PrepareTestCaseMaterial(dp_num, tp_num);

  auto llm_runtime = std::make_shared<LlmRuntime>(batch_scheduler_config_, runtime_config_, context_);
  llm_runtime->SetMultiBatchController(multi_batch_controller_);

  auto processor = std::make_shared<ScheduleProcessor>(false);
  processor->Initialize(batch_scheduler_, llm_runtime, multi_batch_controller_);

  SUCCEED();
}
