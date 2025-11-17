/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>
#include <filesystem>

#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/models/qwen/qwen_model.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/layer_progress_tracker.h"
#include "ksana_llm/runtime/weight_instance.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/attention_backend/attention_backend_manager.h"
#include "ksana_llm/utils/dynamic_memory_pool.h"
#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include "ksana_llm/utils/pytorch_file_tensor_loader.h"
#include "ksana_llm/utils/safetensors_file_tensor_loader.h"

using namespace ksana_llm;

class QwenTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    DeviceMemoryPool::Disable();
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = test_info->name();
    std::string model_path = "/model/qwen1.5-hf/0.5B-Chat";
    std::string yaml_path = "../../../../examples/llama7b/ksana_llm.yaml";
    context = std::make_shared<Context>(1, 1, 1);

    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / yaml_path;
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    AttentionBackendManager::GetInstance()->Initialize();
    const auto &env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path, model_path);
    BatchSchedulerConfig batch_scheduler_config;
    env->GetBatchSchedulerConfig(batch_scheduler_config);
    batch_scheduler_config.mtp_step_num = 0;  // Qwen doesn't use MTP
    env->SetBatchSchedulerConfig(batch_scheduler_config);
    env->UpdateModelConfig();
    env->GetModelConfig(model_config);

    KLLM_LOG_INFO << "model_config.quant_config.method: " << model_config.quant_config.method;
    AttnBackendConfig attn_backend_config;
    env->GetAttnBackendConfig(attn_backend_config);
    attn_backend_config.enable_blocked_multi_token_forwarding_kv = false;
    env->SetAttnBackendConfig(attn_backend_config);
    BlockManagerConfig block_manager_config;
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

    block_manager_config.device_allocator_config.blocks_num = 32;  // This test just need a few blocks;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;

    env->GetRuntimeConfig(runtime_config);

    BlockAllocatorGroupConfig group_1_config;
    group_1_config.devices = {0};
    group_1_config.device_block_num = block_manager_config.device_allocator_config.blocks_num;
    group_1_config.host_block_num = block_manager_config.host_allocator_config.blocks_num;
    group_1_config.block_size = block_manager_config.device_allocator_config.block_size;

    BlockAllocatorManagerConfig block_allocator_manager_config;
    block_allocator_manager_config[1] = group_1_config;

    std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = std::make_shared<MemoryAllocator>();
    BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context);
    block_allocator_group = block_allocator_manager.GetBlockAllocatorGroup(1);

    CacheManagerConfig cache_manager_config;
    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.swap_threadpool_size = 2;
    cache_manager_config.enable_prefix_caching = true;
    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group);
  }

  void TearDown() override { std::cout << "TearDown" << std::endl; }

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group;
  std::shared_ptr<PrefixCacheManager> cache_manager = nullptr;
  std::shared_ptr<Context> context{nullptr};
  const size_t multi_batch_id = 0;
  const std::vector<int> input_ids = {
      0,     0,     128803, 2788,  3655,   5979,   3099,  32200, 7624,  7524,   19,     16,     223,   1140,   2056,
      12519, 61320, 58788,  9090,  14721,  625,    303,   8040,  1612,  1049,   410,    31946,  303,   2788,   112467,
      718,   16227, 111162, 303,   1380,   32200,  8955,  7383,  10949, 20,     16,     223,    6094,  42257,  1261,
      40345, 34666, 525,    4385,  7624,   303,    13380, 41495, 718,   111162, 303,    3722,   2056,  422,    8673,
      2032,  54919, 2056,   1380,  6831,   9090,   303,   3722,  6525,  9090,   2032,   112467, 718,   16227,  111162,
      10949, 21,    16,     223,   100260, 2484,   8504,  2541,  34666, 65656,  121504, 654,    917,   2484,   9090,
      525,   19193, 34666,  7804,  303,    2541,   173,   241,   248,   548,    173,    241,    249,   36703,  902,
      34666, 65656, 21066,  4211,  34666,  7804,   303,   883,   1056,  19,     558,    34666,  64043, 173,    241,
      248,   19,    173,    241,   249,    303,    13097, 2032,  6831,  13850,  303,    23305,  19484, 1107,   50292,
      1847,  3722,  1530,   4385,  34666,  121386, 10626, 34666, 7804,  478,    22,     16,     223,   7624,   27095,
      7747,  7919,  16734,  271,   23,     16,     223,   9090,  974,   10209,  1735,   10655,  271,   122641, 7524,
      2556,  17288, 621,    4385,  34666,  271,    2792,  2130,  768,   939,    23,     15,     3425,  15,     3130,
      271,   2056,  768,    12183, 9617,   128804};
  const std::vector<std::vector<int>> expected_tokens = {{115584, 54381}};

  template <typename weight_data_type>
  void TestQwenForward() {
    const int device_id = 0;
    SetDevice(device_id);
    std::filesystem::path model_path(model_config.path);
    if (!std::filesystem::exists(model_path)) {
      KLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
      EXPECT_TRUE(std::filesystem::exists(model_path));
    }
    Event start;
    Event stop;
    float milliseconds = 0;
    constexpr int rounds = 10;
    EventCreate(&start);
    EventCreate(&stop);
    std::unique_ptr<WeightInstance> weight_instance =
        std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    std::shared_ptr<BaseWeight> qwen_weight = weight_instance->GetWeight(/*rank*/ 0);
    std::shared_ptr<QwenModel> qwen =
        std::make_shared<QwenModel>(model_config, runtime_config, device_id, context, qwen_weight);
    qwen->AllocResources(multi_batch_id);

    // ContextDecode
    auto forward = std::make_unique<ForwardRequest>();
    forward->cache_manager = cache_manager;
    forward->attn_dp_group_id = 0;
    forward->forwarding_tokens = std::make_shared<std::vector<int>>(input_ids);
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    forward->flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    EmbeddingSlice embedding_slice;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;
    forward->input_refit_embedding = &embedding_slice;

    std::vector<int> block_ids;
    int use_block_num = (input_ids.size() + runtime_config.attn_backend_config.block_token_num - 1) /
                        runtime_config.attn_backend_config.block_token_num;
    block_allocator_group->GetDeviceBlockAllocator(0)->AllocateBlocks(use_block_num, block_ids);
    forward->kv_cache_ptrs.resize(1);  // rank num = 1
    block_allocator_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward->kv_cache_ptrs[0]);

    forward->atb_kv_cache_base_blk_ids.assign(1, {});
    AppendFlatKVCacheBlkIds(model_config.num_layer, {block_ids}, forward->atb_kv_cache_base_blk_ids, cache_manager);
    for (int block_idx = 0; block_idx < use_block_num; block_idx++) {
      Memset(forward->kv_cache_ptrs[0][block_idx], 0, runtime_config.attn_backend_config.block_size);
      KLLM_LOG_DEBUG << fmt::format(
          "kv_cache_ptrs {} end {}", forward->kv_cache_ptrs[0][block_idx],
          forward->kv_cache_ptrs[0][block_idx] + (runtime_config.attn_backend_config.block_size));
    }

    auto decode_forward = std::make_unique<ForwardRequest>(*forward);
    decode_forward->cache_manager = cache_manager;
    std::vector<int> decode_ids = input_ids;
    decode_forward->forwarding_tokens = std::make_shared<std::vector<int>>(decode_ids);
    decode_forward->infer_stage = InferStage::kDecode;
    decode_forward->kv_cached_token_num = decode_forward->forwarding_tokens->size() - 1;
    std::vector<ForwardRequest *> forward_reqs = {forward.get(), decode_forward.get()};
    Singleton<LayerProgressTracker>::GetInstance()->Initialize(
        runtime_config.parallel_basic_config.tensor_parallel_size, model_config.num_layer);
    Singleton<LayerProgressTracker>::GetInstance()->RegisterCallback([&](int device_id, int layer_index) {
      KLLM_LOG_INFO << "LayerProgressTracker : device_id: " << device_id << " , layer_index: " << layer_index;
    });
    EXPECT_TRUE(qwen->Forward(multi_batch_id, qwen_weight, forward_reqs, false).OK());
    Singleton<LayerProgressTracker>::GetInstance()->Cleanup();

    std::vector<ForwardRequest *> multi_forward_reqs = {forward.get(), forward.get()};
    // warmup
    for (int i = 0; i < rounds; ++i) {
      qwen->Forward(multi_batch_id, qwen_weight, multi_forward_reqs, false);
    }
    // test performance
    EventRecord(start, context->GetComputeStreams()[device_id]);
    for (int i = 0; i < rounds; ++i) {
      qwen->Forward(multi_batch_id, qwen_weight, multi_forward_reqs, false);
    }
    EventRecord(stop, context->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "ContextDecode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

    if (model_config.quant_config.method == QUANT_BLOCK_FP8_E4M3) {
      EXPECT_TRUE((milliseconds / rounds) < 20);
    }

    // Sampling
    SamplingRequest sample_req;
    NgramDict ngram_dict;
    std::vector<std::vector<std::pair<int, float>>> logprobs;
    std::vector<float> prompt_probs;
    std::vector<int> generated_tokens0, generated_tokens1;
    sample_req.input_tokens = std::make_shared<std::vector<int>>(input_ids);
    sample_req.sampling_token_num = 1;
    sample_req.logits_offset = forward_reqs[0]->logits_offset;
    sample_req.sampling_result_tokens = &generated_tokens0;
    sample_req.logprobs = std::make_shared<std::vector<std::vector<std::pair<int, float>>>>(logprobs);
    sample_req.ngram_dict = &ngram_dict;
    sample_req.logits_buf = std::vector<float *>{qwen->GetLogitsPtr(multi_batch_id)};
    SamplingConfig sample_config;
    sample_req.sampling_config = &sample_config;

    SamplingRequest decode_sample_req = sample_req;
    decode_sample_req.sampling_result_tokens = &generated_tokens1;
    decode_sample_req.logits_offset = forward_reqs[1]->logits_offset;
    decode_sample_req.logits_buf = std::vector<float *>{qwen->GetLogitsPtr(multi_batch_id)};

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
    // Adjust the vocab size
    batch_scheduler_config.max_vocab_size = model_config.vocab_size;

    std::vector<SamplingRequest> sample_reqs = {sample_req, decode_sample_req};
    std::shared_ptr<Sampler> sampler = std::make_shared<Sampler>(batch_scheduler_config, device_id, context);
    sampler->Sampling(0, sample_reqs, context->GetComputeStreams()[device_id]);
    std::cout << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0], generated_tokens1[0])
              << std::endl;
    // Check if generated tokens match any group of expected tokens
    bool match0 = false, match1 = false;
    for (size_t i = 0; i < expected_tokens.size(); ++i) {
      if (generated_tokens0[0] == expected_tokens[i][0]) match0 = true;
      if (generated_tokens1[0] == expected_tokens[i][0]) match1 = true;
    }
    EXPECT_TRUE(match0 || expected_tokens.empty());
    EXPECT_TRUE(match1 || expected_tokens.empty());

    // Decode
    (*forward_reqs[0]->forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1]->forwarding_tokens).emplace_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req->infer_stage = InferStage::kDecode;
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }
    EXPECT_TRUE(qwen->Forward(multi_batch_id, qwen_weight, forward_reqs, false).OK());
    sampler->Sampling(0, sample_reqs, context->GetComputeStreams()[device_id]);
    std::cout << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0], generated_tokens1[0])
              << std::endl;
    // Check if generated tokens match any group of expected tokens
    match0 = false;
    match1 = false;
    for (size_t i = 0; i < expected_tokens.size(); ++i) {
      if (generated_tokens0[0] == expected_tokens[i][1]) match0 = true;
      if (generated_tokens1[0] == expected_tokens[i][1]) match1 = true;
    }
    EXPECT_TRUE(match0 || expected_tokens.empty());
    EXPECT_TRUE(match1 || expected_tokens.empty());

    (*forward_reqs[0]->forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1]->forwarding_tokens).emplace_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }

    EventRecord(start, context->GetComputeStreams()[device_id]);
    for (auto &forward_req : multi_forward_reqs) {
      forward_req->infer_stage = InferStage::kDecode;
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }
    for (int i = 0; i < rounds; ++i) {
      qwen->Forward(multi_batch_id, qwen_weight, multi_forward_reqs, false);
    }
    EventRecord(stop, context->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "Decode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;
    EXPECT_TRUE((milliseconds / rounds) < 11);

    qwen.reset();
    qwen_weight.reset();

    StreamSynchronize(context->GetMemoryManageStreams()[device_id]);
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }

  template <typename weight_data_type>
  void TestQwenForwardWithGreedySampler() {
    const int device_id = 0;
    SetDevice(device_id);
    std::filesystem::path model_path(model_config.path);
    if (!std::filesystem::exists(model_path)) {
      KLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
      EXPECT_TRUE(std::filesystem::exists(model_path));
    }
    std::unique_ptr<WeightInstance> weight_instance =
        std::make_unique<WeightInstance>(model_config, runtime_config, context);
    weight_instance->Load();
    std::shared_ptr<BaseWeight> qwen_weight = weight_instance->GetWeight(/*rank*/ 0);
    std::shared_ptr<QwenModel> qwen =
        std::make_shared<QwenModel>(model_config, runtime_config, device_id, context, qwen_weight);
    qwen->AllocResources(multi_batch_id);

    std::vector<int> generated_tokens0(1), generated_tokens1(1);

    // ContextDecode
    auto forward = std::make_unique<ForwardRequest>();
    forward->cache_manager = cache_manager;
    forward->attn_dp_group_id = 0;
    forward->forwarding_tokens = std::make_shared<std::vector<int>>(input_ids);
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    forward->flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    EmbeddingSlice embedding_slice;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;
    forward->input_refit_embedding = &embedding_slice;
    SamplingConfig sample_config;
    sample_config.num_beams = 1;
    sample_config.topk = 1;
    sample_config.topp = 0;
    sample_config.temperature = 0;
    sample_config.repetition_penalty = 1;
    sample_config.no_repeat_ngram_size = 0;
    sample_config.encoder_no_repeat_ngram_size = 0;
    forward->sampling_config = &sample_config;

    std::vector<int> block_ids;
    int use_block_num = (input_ids.size() + runtime_config.attn_backend_config.block_token_num - 1) /
                        runtime_config.attn_backend_config.block_token_num;
    block_allocator_group->GetDeviceBlockAllocator(0)->AllocateBlocks(use_block_num, block_ids);
    forward->kv_cache_ptrs.resize(1);  // rank num = 1
    block_allocator_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward->kv_cache_ptrs[0]);

    forward->atb_kv_cache_base_blk_ids.assign(1, {});
    AppendFlatKVCacheBlkIds(model_config.num_layer, {block_ids}, forward->atb_kv_cache_base_blk_ids, cache_manager);
    for (int block_idx = 0; block_idx < use_block_num; block_idx++) {
      Memset(forward->kv_cache_ptrs[0][block_idx], 0, runtime_config.attn_backend_config.block_size);
      KLLM_LOG_DEBUG << fmt::format(
          "kv_cache_ptrs {} end {}", forward->kv_cache_ptrs[0][block_idx],
          forward->kv_cache_ptrs[0][block_idx] + (runtime_config.attn_backend_config.block_size));
    }

    auto decode_forward = std::make_unique<ForwardRequest>(*forward);
    decode_forward->cache_manager = cache_manager;
    std::vector<int> decode_ids = input_ids;
    decode_forward->forwarding_tokens = std::make_shared<std::vector<int>>(decode_ids);
    decode_forward->infer_stage = InferStage::kDecode;
    decode_forward->kv_cached_token_num = decode_forward->forwarding_tokens->size() - 1;
    std::vector<ForwardRequest *> forward_reqs = {forward.get(), decode_forward.get()};
    Singleton<LayerProgressTracker>::GetInstance()->Initialize(
        runtime_config.parallel_basic_config.tensor_parallel_size, model_config.num_layer);
    Singleton<LayerProgressTracker>::GetInstance()->RegisterCallback([&](int device_id, int layer_index) {
      KLLM_LOG_INFO << "LayerProgressTracker : device_id: " << device_id << " , layer_index: " << layer_index;
    });
    EXPECT_TRUE(qwen->Forward(multi_batch_id, qwen_weight, forward_reqs, false).OK());
    Singleton<LayerProgressTracker>::GetInstance()->Cleanup();
    generated_tokens0[0] = qwen->GetOutputTokensPtr(multi_batch_id)[0];
    generated_tokens1[0] = qwen->GetOutputTokensPtr(multi_batch_id)[1];
    std::cout << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0], generated_tokens1[0])
              << std::endl;
    // Check if generated tokens match any group of expected tokens
    bool match0 = false, match1 = false;
    for (size_t i = 0; i < expected_tokens.size(); ++i) {
      if (generated_tokens0[0] == expected_tokens[i][0]) match0 = true;
      if (generated_tokens1[0] == expected_tokens[i][0]) match1 = true;
    }
    EXPECT_TRUE(match0 || expected_tokens.empty());
    EXPECT_TRUE(match1 || expected_tokens.empty());

    // Decode
    (*forward_reqs[0]->forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1]->forwarding_tokens).emplace_back(generated_tokens1[0]);
    for (auto &forward_req : forward_reqs) {
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }
    EXPECT_TRUE(qwen->Forward(multi_batch_id, qwen_weight, forward_reqs, false).OK());
    generated_tokens0[0] = qwen->GetOutputTokensPtr(multi_batch_id)[0];
    generated_tokens1[0] = qwen->GetOutputTokensPtr(multi_batch_id)[1];
    std::cout << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0], generated_tokens1[0])
              << std::endl;
    // Check if generated tokens match any group of expected tokens
    match0 = false;
    match1 = false;
    for (size_t i = 0; i < expected_tokens.size(); ++i) {
      if (generated_tokens0[0] == expected_tokens[i][1]) match0 = true;
      if (generated_tokens1[0] == expected_tokens[i][1]) match1 = true;
    }
    EXPECT_TRUE(match0 || expected_tokens.empty());
    EXPECT_TRUE(match1 || expected_tokens.empty());

    qwen.reset();
    qwen_weight.reset();

    StreamSynchronize(context->GetMemoryManageStreams()[device_id]);
    DeviceSynchronize();
  }
};

TEST_F(QwenTest, ForwardFP16Test) {
#ifdef ENABLE_CUDA
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  std::cout << "Test FP16 weight_data_type forward." << std::endl;
  TestQwenForward<half>();
#endif
}

TEST_F(QwenTest, ForwardFP16WithGreedySamplerTest) {
#ifdef ENABLE_CUDA
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  std::cout << "Test FP16 weight_data_type forward with greedy sampler." << std::endl;
  TestQwenForwardWithGreedySampler<half>();
#endif
}
