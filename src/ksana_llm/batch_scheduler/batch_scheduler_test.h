/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_scheduler/batch_scheduler.h"

#include "ksana_llm/batch_scheduler/batch_scheduler_test_client.h"
#include "ksana_llm/batch_scheduler/batch_scheduler_test_helper.h"
#include "ksana_llm/cache_manager/cache_manager_factory.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/helpers/environment_test_helper.h"

#include "test.h"

namespace ksana_llm {

// 定义一个 BatchSchedulerTest 类，继承自 testing::Test
class BatchSchedulerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }

  void SetUp() override {
    enable_prefix_cache_ = false;
    enable_flexible_cache_ = false;
    split_fuse_token_num_ = 0;
    enable_async_ = false;
    waiting_timeout_in_ms_ = 600000;
    enable_swap_ = true;
  }

  void CommonSetUp(int dp_num = 1, int tp_num = 4, int ep_world_size = 1) {
    runtime_config_.parallel_basic_config.tensor_parallel_size = tp_num;
    runtime_config_.parallel_basic_config.attn_data_parallel_size = dp_num;
    runtime_config_.parallel_basic_config.attn_tensor_parallel_size = tp_num / dp_num;
    ep_world_size_ = ep_world_size;

    // Init BatchSchedulerEnvironmentSimulator and BatchScheduler
    InitDefaultConfig();
    InitializeScheduleOutputPool();

    block_allocator_group = std::make_shared<FakedBlockAllocatorGroup>(block_manager_config_, tp_num);

    env_simulator_ = new BatchSchedulerEnvironmentSimulator(block_manager_config_, tp_num, block_allocator_group);

    // Create Context
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);

    int pp_batch_num = 1;
    std::shared_ptr<Context> context = std::make_shared<Context>(1, 1, pp_batch_num);

    // Create ModelInstance
    ModelConfig model_config;
    model_config.name = "test_model";
    model_config.end_ids = {1, 2};
    model_config.pad_id = 0;
    RuntimeConfig runtime_config;
    std::shared_ptr<WeightInstanceInterface> weight_instance = nullptr;
    std::shared_ptr<ModelInstance> model_instance =
        std::make_shared<ModelInstance>(model_config, runtime_config, context, weight_instance);
    model_instance->name = "test_model";

    std::vector<std::shared_ptr<ModelInstance>> model_instances;
    model_instances.push_back(model_instance);

    KLLM_LOG_INFO << "enable_prefix_cache=" << cache_manager_config.enable_prefix_caching
                  << ", split_fuse_num=" << batch_scheduler_config_.split_fuse_token_num;
    batch_scheduler_ =
        std::make_shared<BatchScheduler>(batch_scheduler_config_, runtime_config_, ep_world_size_ > 1, model_instances);

    cache_manager = CacheManagerFactory::CreateCacheManager(cache_manager_config, block_allocator_group);
    cache_manager->InitializeCachedBlocks();
    batch_scheduler_->SetCacheManager(cache_manager, 0);
    schedule_processor_ = std::make_shared<TestScheduleProcessor>(batch_scheduler_config_.enable_async,
                                                                  batch_scheduler_config_.max_pp_batch_num);
    schedule_processor_->Initialize(batch_scheduler_, nullptr, nullptr);
  }

  void FixPrefixCacheBlockLimitTriggeredTest() {
    CommonSetUp();

    int prefix_block_num = 30;
    int block_token_num = 6;
    int device_num = 4;
    FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, split_fuse_token_num_, enable_swap_,
                                false);
    test_case.SetBatchScheduler(batch_scheduler_);
    test_case.SetScheduleProcessor(schedule_processor_);
    test_case.SetEnvSimulator(env_simulator_);
    test_case.RunTestSwapTriggered();
  }

  void TearDown() override {
    batch_scheduler_->Stop();
    schedule_processor_->Stop();

    delete env_simulator_;
    DestroyScheduleOutputPool();
  }

 protected:
  void InitDefaultConfig() {
    int device_block_num = 100;
    if (enable_swap_) {
      block_manager_config_.host_allocator_config.blocks_num =
          device_block_num * runtime_config_.parallel_basic_config.tensor_parallel_size * 2;
      batch_scheduler_config_.preempt_mode = SWAP;
    } else {
      block_manager_config_.host_allocator_config.blocks_num = 0;
      batch_scheduler_config_.preempt_mode = RECOMPUTE;
    }
    block_manager_config_.device_allocator_config.blocks_num = device_block_num;
    block_manager_config_.device_allocator_config.block_token_num = 6;

    batch_scheduler_config_.schedule_strategy = static_cast<ScheduleStrategy>(0);
    batch_scheduler_config_.waiting_timeout_in_ms = waiting_timeout_in_ms_;
    batch_scheduler_config_.max_waiting_queue_len = 256;
    batch_scheduler_config_.max_step_token_num = 4096;
    batch_scheduler_config_.max_batch_size = 8;
    batch_scheduler_config_.max_token_len = 1024;
    batch_scheduler_config_.swapout_block_threshold = 1.0;
    batch_scheduler_config_.swapin_block_threshold = 3.0;
    batch_scheduler_config_.launch_block_threshold = 4.0;
    batch_scheduler_config_.split_fuse_token_num = split_fuse_token_num_;
    batch_scheduler_config_.enable_async = enable_async_;

    cache_manager_config.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = runtime_config_.parallel_basic_config.tensor_parallel_size;
    cache_manager_config.swap_threadpool_size = 8;
    cache_manager_config.enable_prefix_caching = enable_prefix_cache_;
    if (enable_flexible_cache_) {
      cache_manager_config.min_flexible_cache_num = cache_manager_config.block_token_num * 4;
    }
  }

 protected:
  // 定义一个 BlockManager 指针，用于在测试用例中使用
  BatchSchedulerEnvironmentSimulator* env_simulator_ = nullptr;
  std::shared_ptr<BatchSchedulerInterface> batch_scheduler_ = nullptr;
  std::shared_ptr<ScheduleProcessorInterface> schedule_processor_ = nullptr;

  std::shared_ptr<FakedBlockAllocatorGroup> block_allocator_group = nullptr;
  std::shared_ptr<CacheManagerInterface> cache_manager = nullptr;

  BlockManagerConfig block_manager_config_;
  BatchSchedulerConfig batch_scheduler_config_;
  CacheManagerConfig cache_manager_config;
  RuntimeConfig runtime_config_;
  int ep_world_size_;

  bool enable_prefix_cache_;
  bool enable_flexible_cache_;
  size_t split_fuse_token_num_;
  bool enable_async_;
  size_t waiting_timeout_in_ms_;
  bool enable_swap_ = true;
};

}  // namespace ksana_llm
