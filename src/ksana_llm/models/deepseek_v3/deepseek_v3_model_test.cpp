/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/models/deepseek_v3/deepseek_v3_model.h"
#include "ksana_llm/runtime/layer_progress_tracker.h"
#include "ksana_llm/runtime/structured_generation/structured_generator_factory.h"
#include "ksana_llm/runtime/structured_generation/xgrammar/xgrammar_structured_generator_creator.h"
#include "ksana_llm/runtime/weight_instance.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/dynamic_memory_pool.h"
#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tokenizer.h"
#include "test.h"

#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include "ksana_llm/utils/pytorch_file_tensor_loader.h"
#include "ksana_llm/utils/safetensors_file_tensor_loader.h"

using namespace ksana_llm;

class DeepSeekV3Test : public testing::Test {
 protected:
  void SetUp() override {
    setenv("W4AFP8_MOE_BACKEND", "0", 1);
    InitLoguru();
    DeviceMemoryPool::Disable();
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = test_info->name();
    std::string model_path = "/model/DeepSeek-R1-17832-fix-mtp";
    std::string yaml_path = "../../../../examples/llama7b/ksana_llm.yaml";
    SetAbsorbWeightsType(AbsorbWeightsType::kAbsorbTypeBMM);
    context = std::make_shared<Context>(1, 1, 1);

    // Skip int4 output token check cause it's not stable.
    if (test_name.find("ForwardGPTQInt4Test") != std::string::npos) {
      model_path = "/model/DeepSeek-R1-17832-fix-mtp-bf16-w4g128-auto-gptq";
    } else if (test_name.find("ForwardMoeInt4Test") != std::string::npos) {
      model_path = "/model/DeepSeek-R1-0528-moe-int4-fix-mtp";
    } else if (test_name.find("SmallExpertsTest") != std::string::npos) {
      model_path = "/model/deepseek_v3";
      expected_tokens = {{3648, 303, 19892}};
    } else {
      if (test_name.find("ForwardW4AFP8Test") != std::string::npos) {
        model_path = "/model/DeepSeek-R1-0528-W4AFP8-mtpbfp8-venus-fix";
      }
      expected_tokens = {{5306, 13245, 15354}, /*without fastmath*/ {5306, 13245, 536}, {28570, 27932, 4180}};
    }

    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / yaml_path;
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    const auto &env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path, model_path);
    // TODO(robertyuan): bad style, remove later
    BatchSchedulerConfig batch_scheduler_config;
    env->GetBatchSchedulerConfig(batch_scheduler_config);
    batch_scheduler_config.enable_mtp_module = true;
    batch_scheduler_config.enable_xgrammar = true;
    env->SetBatchSchedulerConfig(batch_scheduler_config);
    env->UpdateModelConfig();
    env->GetModelConfig(model_config);

    KLLM_LOG_INFO << "model_config.quant_config.method: " << model_config.quant_config.method;
    AttnBackendConfig attn_backend_config;
    env->GetAttnBackendConfig(attn_backend_config);
    attn_backend_config.enable_blocked_multi_token_forwarding_kv = true;
    env->SetAttnBackendConfig(attn_backend_config);
    BlockManagerConfig block_manager_config;
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

    block_manager_config.device_allocator_config.blocks_num = 32;  // This test just need a few blocks;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;

    env->GetRuntimeConfig(runtime_config);
    runtime_config.mtp_step_num = batch_scheduler_config.mtp_step_num;

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

  void TearDown() override {
    unsetenv("W4AFP8_MOE_BACKEND");
    std::cout << "TearDown" << std::endl;
  }

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group;
  std::shared_ptr<PrefixCacheManager> cache_manager = nullptr;
  std::shared_ptr<Context> context{nullptr};
  size_t multi_batch_id = 0;
  std::vector<std::vector<int>> expected_tokens;

  template <typename weight_data_type>
  void TestDeepSeekV3Forward() {
    int device_id = 0;
    SetDevice(device_id);
#ifdef ENABLE_FP8
    // fp8 is not supported
#endif
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
    std::shared_ptr<BaseWeight> deepseek_v3_weight = weight_instance->GetWeight(/*rank*/ 0);
    std::shared_ptr<DeepSeekV3Model> deepseek_v3 =
        std::make_shared<DeepSeekV3Model>(model_config, runtime_config, device_id, context, deepseek_v3_weight);
    deepseek_v3->AllocResources(multi_batch_id);

    // ContextDecode
    ForwardRequest forward;
    forward.cache_manager = cache_manager;
    std::vector<int> input_ids = {
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
    forward->attn_dp_group_id = 0;
    forward->forwarding_tokens = std::make_shared<std::vector<int>>(input_ids);
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    forward->flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    EmbeddingSlice embedding_slice;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;
    forward.input_refit_embedding = &embedding_slice;

    std::vector<int> block_ids;
    int use_block_num = (input_ids.size() + runtime_config.attn_backend_config.block_token_num - 1) /
                        runtime_config.attn_backend_config.block_token_num;
    block_allocator_group->GetDeviceBlockAllocator(0)->AllocateBlocks(use_block_num, block_ids);
    forward.kv_cache_ptrs.resize(1);  // rank num = 1
    block_allocator_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward.kv_cache_ptrs[0]);

    LlmRuntime::BuildFlatKVCacheBlkIds(model_config.num_layer + model_config.num_nextn_predict_layers, {block_ids},
                                       forward.atb_kv_cache_base_blk_ids, cache_manager);
    for (int block_idx = 0; block_idx < use_block_num; block_idx++) {
      Memset(forward.kv_cache_ptrs[0][block_idx], 0, runtime_config.attn_backend_config.block_size);
      KLLM_LOG_DEBUG << fmt::format(
          "kv_cache_ptrs {} end {}", forward.kv_cache_ptrs[0][block_idx],
          forward.kv_cache_ptrs[0][block_idx] + (runtime_config.attn_backend_config.block_size));
    }

    ForwardRequest decode_forward = forward;
    decode_forward.cache_manager = cache_manager;
    std::vector<int> decode_ids = input_ids;
    forward.forwarding_tokens = std::make_shared<std::vector<int>>(decode_ids);
    decode_forward.infer_stage = InferStage::STATE_DECODE;
    decode_forward.kv_cached_token_num = decode_forward.forwarding_tokens->size() - 1;
    std::vector<ForwardRequest> forward_reqs = {forward, decode_forward};
    Singleton<LayerProgressTracker>::GetInstance()->Initialize(
        runtime_config.parallel_basic_config.tensor_parallel_size,
        model_config.num_layer + model_config.num_nextn_predict_layers);
    Singleton<LayerProgressTracker>::GetInstance()->RegisterCallback([&](int device_id, int layer_index) {
      KLLM_LOG_INFO << "LayerProgressTracker : device_id: " << device_id << " , layer_index: " << layer_index;
    });
    EXPECT_TRUE(deepseek_v3->Forward(multi_batch_id, deepseek_v3_weight, forward_reqs, false).OK());
    Singleton<LayerProgressTracker>::GetInstance()->Cleanup();
    std::vector<ForwardRequest> multi_forward_reqs = {forward, forward};
    // warmup
    for (int i = 0; i < rounds; ++i) {
      deepseek_v3->Forward(multi_batch_id, deepseek_v3_weight, multi_forward_reqs, false);
    }
    // test performance
    EventRecord(start, context->GetComputeStreams()[device_id]);
    for (int i = 0; i < rounds; ++i) {
      deepseek_v3->Forward(multi_batch_id, deepseek_v3_weight, multi_forward_reqs, false);
    }
    EventRecord(stop, context->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "ContextDecode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

    if (model_config.quant_config.method == QUANT_BLOCK_FP8_E4M3 && !model_config.quant_config.enable_moe_int4) {
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
    sample_req.logits_offset = forward_reqs[0].logits_offset;
    sample_req.sampling_result_tokens = &generated_tokens0;
    sample_req.logprobs = std::make_shared<std::vector<std::vector<std::pair<int, float>>>>(logprobs);
    sample_req.ngram_dict = &ngram_dict;
    sample_req.logits_buf = std::vector<float*>{deepseek_v3->GetLogitsPtr(multi_batch_id)};
    SamplingConfig sample_config;
    sample_config.num_beams = 1;
    sample_config.topk = 1;
    sample_config.topp = 0;
    sample_config.temperature = 0;
    sample_config.repetition_penalty = 1;
    sample_config.no_repeat_ngram_size = 0;
    sample_config.encoder_no_repeat_ngram_size = 0;
    sample_req.sampling_config = &sample_config;

    SamplingRequest decode_sample_req = sample_req;
    decode_sample_req.sampling_result_tokens = &generated_tokens1;
    decode_sample_req.logits_offset = forward_reqs[1]->logits_offset;
    decode_sample_req.logits_buf = std::vector<float*>{deepseek_v3->GetLogitsPtr(multi_batch_id)};

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

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
    (*forward_reqs[0].forwarding_tokens).push_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).push_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req.infer_stage = InferStage::STATE_DECODE;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    EXPECT_TRUE(deepseek_v3->Forward(multi_batch_id, deepseek_v3_weight, forward_reqs, false).OK());
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

    (*forward_reqs[0].forwarding_tokens).push_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).push_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }

    EventRecord(start, context->GetComputeStreams()[device_id]);
    for (auto &forward_req : multi_forward_reqs) {
      forward_req.infer_stage = InferStage::STATE_DECODE;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    for (int i = 0; i < rounds; ++i) {
      deepseek_v3->Forward(multi_batch_id, deepseek_v3_weight, multi_forward_reqs, false);
    }
    EventRecord(stop, context->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "Decode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;
    EXPECT_TRUE((milliseconds / rounds) < 11);

    // MTP
    for (auto &forward_req : multi_forward_reqs) {
      forward_req.infer_stage = InferStage::STAGE_CONTEXT;
      forward_req.kv_cached_token_num = 0;
    }
    EXPECT_TRUE(deepseek_v3->Forward(multi_batch_id, deepseek_v3_weight, forward_reqs, false, RunMode::kNextN).OK());
    sampler->Sampling(0, sample_reqs, context->GetComputeStreams()[device_id]);
    std::cout << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0], generated_tokens1[0])
              << std::endl;
    // Check if generated tokens match any group of expected tokens
    match0 = false;
    match1 = false;
    for (size_t i = 0; i < expected_tokens.size(); ++i) {
      if (generated_tokens0[0] == expected_tokens[i][2]) match0 = true;
      if (generated_tokens1[0] == expected_tokens[i][2]) match1 = true;
    }
    EXPECT_TRUE(match0 || expected_tokens.empty());
    EXPECT_TRUE(match1 || expected_tokens.empty());

    generated_tokens0.clear();
    generated_tokens1.clear();

    EventRecord(start, context->GetComputeStreams()[device_id]);
    for (int i = 0; i < rounds; ++i) {
      deepseek_v3->Forward(multi_batch_id, deepseek_v3_weight, forward_reqs, false, RunMode::kNextN);
    }
    EventRecord(stop, context->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "prefill mtp milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;
    EXPECT_TRUE((milliseconds / rounds) < 11);

#ifdef ENABLE_CUDA
    // Test Xgrammar
    generated_tokens0.clear();
    generated_tokens1.clear();

    std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory_;
    std::vector<std::string> vocab;
    int vocab_size = static_cast<int>(model_config.vocab_size);
    std::vector<int> stop_token_ids;

    Singleton<Tokenizer>::GetInstance()->InitTokenizer(model_path);
    auto tokenizer = Singleton<Tokenizer>::GetInstance();
    tokenizer->GetVocabInfo(vocab, vocab_size, stop_token_ids);
    structured_generator_factory_ = std::make_shared<StructuredGeneratorFactory>();
    structured_generator_factory_->RegisterCreator(
        StructuredConstraintType::JSON, std::make_unique<GrammarGeneratorCreator>(vocab, vocab_size, stop_token_ids));

    std::string json_schema = R"({
      "type": "object"
    })";

    StructuredGeneratorConfig config(StructuredConstraintType::JSON, json_schema);
    auto structured_generator = structured_generator_factory_->CreateGenerator(config);

    SamplingRequest grammar_sample_req = sample_req;
    grammar_sample_req.structured_generator = structured_generator;
    std::vector<SamplingRequest> grammar_sample_reqs = {grammar_sample_req};

    for (auto &forward_req : forward_reqs) {
      forward_req.infer_stage = InferStage::STATE_DECODE;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }

    EXPECT_TRUE(deepseek_v3->Forward(multi_batch_id, deepseek_v3_weight, forward_reqs, false).OK());
    sampler->Sampling(0, grammar_sample_reqs, context->GetComputeStreams()[device_id]);
    int grammar_token = generated_tokens0[0];

    generated_tokens0.clear();

    SamplingRequest no_grammar_sample_req = sample_req;
    grammar_sample_reqs = {no_grammar_sample_req};
    EXPECT_TRUE(deepseek_v3->Forward(multi_batch_id, deepseek_v3_weight, forward_reqs, false).OK());
    sampler->Sampling(0, grammar_sample_reqs, context->GetComputeStreams()[device_id]);
    int no_grammar_token = generated_tokens0[0];

    EXPECT_NE(grammar_token, no_grammar_token);
    std::cout << fmt::format("grammar_token: {}, no_grammar_token: {}", grammar_token, no_grammar_token) << std::endl;

    Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
#endif
    deepseek_v3.reset();
    deepseek_v3_weight.reset();

    StreamSynchronize(context->GetMemoryManageStreams()[device_id]);
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }
};

TEST_F(DeepSeekV3Test, ForwardFP8BlockWiseTest) {
#ifdef ENABLE_CUDA
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  // deepseek only support fp8 block-wise quantization, don't support fp8 per-tensor quantization
  model_config.quant_config.method = QUANT_BLOCK_FP8_E4M3;
  std::cout << "Test FP8-BlockWise TYPE_BF16 weight_data_type forward." << std::endl;
  TestDeepSeekV3Forward<bfloat16>();
#endif
}

TEST_F(DeepSeekV3Test, ForwardGPTQInt4Test) {
#ifdef ENABLE_CUDA
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_GPTQ;
  std::cout << "Test GPTQ-Quant TYPE_BF16 weight_data_type forward." << std::endl;
  TestDeepSeekV3Forward<bfloat16>();
#endif
}

TEST_F(DeepSeekV3Test, ForwardMoeInt4Test) {
#ifdef ENABLE_CUDA
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_BLOCK_FP8_E4M3;
  model_config.quant_config.enable_moe_int4 = true;
  std::cout << "Test MoeInt4 TYPE_BF16 weight_data_type forward." << std::endl;
  TestDeepSeekV3Forward<bfloat16>();
#endif
}

// Test for `num_experts_per_rank_ <= 224`
TEST_F(DeepSeekV3Test, SmallExpertsTest) {
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_BLOCK_FP8_E4M3;
  TestDeepSeekV3Forward<bfloat16>();
}

TEST_F(DeepSeekV3Test, EnableFullShardExpertTest) {
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_BLOCK_FP8_E4M3;
  runtime_config.enable_full_shared_expert = true;
  TestDeepSeekV3Forward<bfloat16>();
}

TEST_F(DeepSeekV3Test, ForwardW4AFP8Test) {
#ifdef ENABLE_CUDA
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  QuantConfig gptq_quant_config;
  gptq_quant_config.method = QUANT_GPTQ;
  gptq_quant_config.pattern_layers = {".mlp.experts."};
  gptq_quant_config.ignored_layers = {"model.layers.4.mlp.experts."};
  QuantConfig fp8_quant_config;
  fp8_quant_config.method = QUANT_BLOCK_FP8_E4M3;
  fp8_quant_config.is_fp8_blockwise = true;
  fp8_quant_config.weight_block_size = {128, 128};
  fp8_quant_config.pattern_layers = {};
  fp8_quant_config.ignored_layers = {"model.layers.4.eh_proj"};
  fp8_quant_config.enable_moe_int4 = true;
  model_config.quant_config = fp8_quant_config;
  model_config.sub_quant_configs = {gptq_quant_config};
  std::cout << "Test W4AFP8 TYPE_BF16 weight_data_type forward." << std::endl;

  // test EPLB config.json
  runtime_config.enable_load_eplb_weight = true;
  std::string eplb_config_filename = std::filesystem::temp_directory_path() / "DeepSeekV3Test_eplb_config.json";
  {
    std::vector<int> data(256);
    std::iota(data.begin(), data.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
    std::string eplb_expert_map = "{\"layer_3\": [";
    for (size_t i = 0; i < data.size(); ++i) {
      if (i > 0) {
        eplb_expert_map += ", ";
      }
      eplb_expert_map += fmt::format("{}", data[i]);
    }
    eplb_expert_map += "]}";
    {
      std::ofstream ofs(eplb_config_filename);
      ASSERT_TRUE(ofs) << fmt::format("open {}", eplb_config_filename);
      ofs << eplb_expert_map;
      setenv("EPLB_WEIGHT", eplb_config_filename.c_str(), 1);
    }
    std::cout << fmt::format("set env EPLB_WEIGHT={}", eplb_config_filename) << std::endl;
  }

  TestDeepSeekV3Forward<bfloat16>();
  unlink(eplb_config_filename.c_str());
#endif
}
