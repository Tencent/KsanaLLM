/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/batch_scheduler/batch_scheduler.h"

#include "ksana_llm/batch_scheduler/batch_scheduler_test_client.h"
#include "ksana_llm/batch_scheduler/batch_scheduler_test_helper.h"

#include "ksana_llm/cache_manager/direct_cache_manager.h"
#include "ksana_llm/cache_manager/prefix_cache_manager.h"

#include "ksana_llm/data_hub/data_hub.h"

#include "test.h"

namespace ksana_llm {

// 定义一个 BatchSchedulerTest 类，继承自 testing::Test
class BatchSchedulerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }

  void CommonSetUp(int dp_num = 1, int tp_num = 4) {
    dp_num_ = dp_num;
    tp_num_ = tp_num;
    // Init BatchSchedulerEnvironmentSimulator and BatchScheduler
    InitDefaultConfig();
    InitializeScheduleOutputPool();

    block_allocator_group = std::make_shared<FakedBlockAllocatorGroup>(block_manager_config_, tp_num_);

    env_simulator_ = new BatchSchedulerEnvironmentSimulator(block_manager_config_, tp_num_, block_allocator_group);
    batch_scheduler_ = new BatchScheduler(batch_scheduler_config_, dp_num_, tp_num_);

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
    block_manager_config_.host_allocator_config.blocks_num = device_block_num * tp_num_ * 2;
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
    cache_manager_config.tensor_para_size = tp_num_;
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
  int tp_num_;
  int dp_num_;
};

}  // namespace ksana_llm
