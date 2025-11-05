/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/model_instance.h"

#include "ksana_llm/batch_manager/async_schedule_processor.h"
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
    batch_scheduler_ = std::make_shared<BatchScheduler>(batch_scheduler_config_, runtime_config_, model_instances);

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

  auto processor = std::make_shared<ScheduleProcessor>();
  processor->Initialize(batch_scheduler_, llm_runtime, multi_batch_controller_);

  SUCCEED();
}

TEST_F(ScheduleProcessorTest, AsyncProcessorConstructorAndInitialize) {
  int dp_num = 1;
  int tp_num = 1;
  PrepareTestCaseMaterial(dp_num, tp_num);

  auto llm_runtime = std::make_shared<LlmRuntime>(batch_scheduler_config_, runtime_config_, context_);
  llm_runtime->SetMultiBatchController(multi_batch_controller_);

  auto async_processor = std::make_shared<AsyncScheduleProcessor>();
  async_processor->Initialize(batch_scheduler_, llm_runtime, multi_batch_controller_);

  SUCCEED();
}

TEST_F(ScheduleProcessorTest, ProcessorRun) {
  int dp_num = 1;
  int tp_num = 1;
  PrepareTestCaseMaterial(dp_num, tp_num);

  auto llm_runtime = std::make_shared<LlmRuntime>(batch_scheduler_config_, runtime_config_, context_);
  llm_runtime->SetMultiBatchController(multi_batch_controller_);

  auto processor = std::make_shared<ScheduleProcessor>();
  processor->Initialize(batch_scheduler_, llm_runtime, multi_batch_controller_);

  processor->Start();

  // Add a fake request to avoid blocking.
  std::shared_ptr<Request> req;
  std::shared_ptr<KsanaPythonInput> python_input;
  std::vector<std::shared_ptr<InferRequest>> infer_reqs = InitFakeRequest(
      123, 5, 10, req, {{0, 1}}, tp_num, block_manager_config_.device_allocator_config.block_token_num, &python_input);
  hold_requests_.push_back(req);
  hold_python_inputs_.push_back(python_input);  // 保存 python_input 确保其生命周期
  for (auto& infer_req : infer_reqs) {
    infer_req->model_instance = model_instance_;
  }
  batch_scheduler_->AddInferRequest(infer_reqs);

  // In test mode, there is no real request, so GetNextScheduleResult will not get any result.
  // But it will not block.
  processor->GetNextScheduleResult(0);
  processor->Stop();

  SUCCEED();
}

TEST_F(ScheduleProcessorTest, AsyncProcessorRun) {
  int dp_num = 1;
  int tp_num = 1;
  PrepareTestCaseMaterial(dp_num, tp_num);

  auto llm_runtime = std::make_shared<LlmRuntime>(batch_scheduler_config_, runtime_config_, context_);
  llm_runtime->SetMultiBatchController(multi_batch_controller_);

  auto async_processor = std::make_shared<AsyncScheduleProcessor>();
  async_processor->Initialize(batch_scheduler_, llm_runtime, multi_batch_controller_);

  async_processor->Start();

  // Add a fake request to avoid blocking.
  std::shared_ptr<Request> req;
  std::shared_ptr<KsanaPythonInput> python_input;
  std::vector<std::shared_ptr<InferRequest>> infer_reqs = InitFakeRequest(
      123, 5, 10, req, {{0, 1}}, tp_num, block_manager_config_.device_allocator_config.block_token_num, &python_input);
  hold_requests_.push_back(req);
  hold_python_inputs_.push_back(python_input);  // 保存 python_input 确保其生命周期
  for (auto& infer_req : infer_reqs) {
    infer_req->model_instance = model_instance_;
  }
  batch_scheduler_->AddInferRequest(infer_reqs);

  // In test mode, there is no real request, so GetNextScheduleResult will not get any result.
  // But it will not block.
  async_processor->GetNextScheduleResult(0);
  async_processor->Stop();

  SUCCEED();
}

TEST_F(ScheduleProcessorTest, ApplyAsyncForwardingTokens) {
  auto async_processor = std::make_shared<AsyncScheduleProcessor>();

  auto deep_copy_forwarding_tokens = std::make_shared<std::unordered_map<int64_t, std::shared_ptr<std::vector<int>>>>();
  (*deep_copy_forwarding_tokens)[123] = std::shared_ptr<std::vector<int>>(new std::vector<int>{1, 2, 3});

  auto grouped_reqs = std::make_shared<std::map<ModelInstance*, std::vector<ForwardRequest*>>>();

  ModelInstance* model_inst = reinterpret_cast<ModelInstance*>(0x9876);
  auto forward_req = std::make_unique<ForwardRequest>();
  forward_req->req_id = 123;
  forward_req->forwarding_tokens = std::shared_ptr<std::vector<int>>(new std::vector<int>{4, 5});

  (*grouped_reqs)[model_inst].push_back(forward_req.get());

  async_processor->ApplyAsyncForwardingTokens(*deep_copy_forwarding_tokens, *grouped_reqs);

  EXPECT_EQ(forward_req->forwarding_tokens->size(), 3);
  EXPECT_EQ((*forward_req->forwarding_tokens)[0], 1);
  EXPECT_EQ((*forward_req->forwarding_tokens)[1], 2);
  EXPECT_EQ((*forward_req->forwarding_tokens)[2], 3);
}

TEST_F(ScheduleProcessorTest, ProcessAsyncPostProcessing) {
  int dp_num = 1;
  int tp_num = 1;
  PrepareTestCaseMaterial(dp_num, tp_num);

  auto llm_runtime = std::make_shared<LlmRuntime>(batch_scheduler_config_, runtime_config_, context_);
  llm_runtime->SetMultiBatchController(multi_batch_controller_);

  auto async_processor = std::make_shared<AsyncScheduleProcessor>();
  async_processor->Initialize(batch_scheduler_, llm_runtime, multi_batch_controller_);

  std::shared_ptr<Request> req;
  std::shared_ptr<KsanaPythonInput> python_input;
  auto infer_reqs = InitFakeRequest(123, 5, 10, req, {{42, 1}}, tp_num,
                                    block_manager_config_.device_allocator_config.block_token_num, &python_input);
  hold_requests_.push_back(req);
  hold_python_inputs_.push_back(python_input);  // 保存 python_input 确保其生命周期
  auto infer_req = infer_reqs[0];
  infer_req->step = 2;
  infer_req->infer_stage = InferStage::kDecode;
  infer_req->forwarding_tokens = {10, 11, 12, 13, 14};
  infer_req->forwarding_tokens_draft_num = 2;
  infer_req->last_step_draft_num = 2;
  infer_req->accepted_tokens = {13};
  infer_req->generated_token = 99;
  infer_req->draft_tokens.mtp = {20, 21};
  infer_req->cache_manager = cache_manager_;
  infer_req->kv_cached_token_num = 3;
  infer_req->model_instance = model_instance_;

  auto schedule_output = std::make_shared<ScheduleOutput>();
  schedule_output->running_reqs.push_back(infer_req);

  ScheduleResult result;
  result.is_valid = true;
  result.schedule_output = schedule_output;
  result.sampling_reqs = std::make_shared<std::vector<SamplingRequest>>();

  async_processor->ProcessAsyncPostProcessing(result);

  // According to the implementation, the new size is calculated as:
  // 5 - 2 + 1 - 1 - 2 = 1.
  // Then, generated_token is added, and new draft_tokens are added.
  // Final size = 1 (resized) + 1 (generated) + 2 (new draft) = 4.
  EXPECT_EQ(infer_req->forwarding_tokens.size(), 4);
  EXPECT_EQ(infer_req->kv_cached_token_num, 1);
  EXPECT_EQ(infer_req->forwarding_tokens[0], 10);
  EXPECT_EQ(infer_req->forwarding_tokens[1], 99);
  EXPECT_EQ(infer_req->forwarding_tokens[2], 20);
  EXPECT_EQ(infer_req->forwarding_tokens[3], 21);
}