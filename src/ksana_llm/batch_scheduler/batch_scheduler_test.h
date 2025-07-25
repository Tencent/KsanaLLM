/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_scheduler/batch_scheduler.h"

#include "ksana_llm/batch_scheduler/batch_scheduler_test_client.h"
#include "ksana_llm/batch_scheduler/batch_scheduler_test_helper.h"

#include "ksana_llm/cache_manager/direct_cache_manager.h"
#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/helpers/environment_test_helper.h"

#include "test.h"

namespace ksana_llm {

// 定义一个 BatchSchedulerTest 类，继承自 testing::Test
class BatchSchedulerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }

  void CommonSetUp(int dp_num = 1, int tp_num = 4, int ep_world_size = 1) {
    runtime_config_.parallel_basic_config.tensor_parallel_size = tp_num;
    runtime_config_.parallel_basic_config.attn_data_parallel_size = dp_num;
    ep_world_size_ = ep_world_size;

    // Init BatchSchedulerEnvironmentSimulator and BatchScheduler
    InitDefaultConfig();
    InitializeScheduleOutputPool();

    block_allocator_group = std::make_shared<FakedBlockAllocatorGroup>(block_manager_config_, tp_num);

    env_simulator_ = new BatchSchedulerEnvironmentSimulator(block_manager_config_, tp_num, block_allocator_group);

    // Create Context
    int pp_batch_num = 1;
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
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

    // Init expert parallel config.
    ExpertParallelConfig ep_config;
    Singleton<Environment>::GetInstance()->GetExpertParallelConfig(ep_config);
    ep_config.expert_world_size = ep_world_size_;
    Singleton<Environment>::GetInstance()->SetExpertParallelConfig(ep_config);

    batch_scheduler_ = new BatchScheduler(batch_scheduler_config_, runtime_config_, model_instances);

    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group);
    cache_manager->InitializeCachedBlocks();
    batch_scheduler_->SetCacheManager(cache_manager, 0);
  }

  void TearDown() override {
    delete batch_scheduler_;
    delete env_simulator_;
    DestroyScheduleOutputPool();
  }

 protected:
  void InitDefaultConfig() {
    int device_block_num = 100;
    block_manager_config_.host_allocator_config.blocks_num =
        device_block_num * runtime_config_.parallel_basic_config.tensor_parallel_size * 2;
    block_manager_config_.device_allocator_config.blocks_num = device_block_num;
    block_manager_config_.device_allocator_config.block_token_num = 6;

    batch_scheduler_config_.schedule_strategy = static_cast<ScheduleStrategy>(0);
    batch_scheduler_config_.waiting_timeout_in_ms = 600000;
    batch_scheduler_config_.max_waiting_queue_len = 256;
    batch_scheduler_config_.max_step_token_num = 4096;
    batch_scheduler_config_.max_batch_size = 8;
    batch_scheduler_config_.max_token_len = 1024;
    batch_scheduler_config_.swapout_block_threshold = 1.0;
    batch_scheduler_config_.swapin_block_threshold = 2.0;
    batch_scheduler_config_.launch_block_threshold = 2.0;
    batch_scheduler_config_.preempt_mode = static_cast<PreemptMode>(0);

    cache_manager_config.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = runtime_config_.parallel_basic_config.tensor_parallel_size;
    cache_manager_config.swap_threadpool_size = 8;
    cache_manager_config.enable_prefix_caching = false;
  }

 protected:
  // 定义一个 BlockManager 指针，用于在测试用例中使用
  BatchSchedulerEnvironmentSimulator* env_simulator_ = nullptr;
  BatchSchedulerInterface* batch_scheduler_ = nullptr;

  std::shared_ptr<FakedBlockAllocatorGroup> block_allocator_group = nullptr;
  std::shared_ptr<CacheManagerInterface> cache_manager = nullptr;

  BlockManagerConfig block_manager_config_;
  BatchSchedulerConfig batch_scheduler_config_;
  CacheManagerConfig cache_manager_config;
  RuntimeConfig runtime_config_;
  int ep_world_size_;
};

}  // namespace ksana_llm
