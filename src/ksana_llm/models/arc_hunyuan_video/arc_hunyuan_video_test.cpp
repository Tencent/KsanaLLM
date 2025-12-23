/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>

#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/models/arc_hunyuan_video/arc_hunyuan_video_model.h"
#include "ksana_llm/runtime/layer_progress_tracker.h"
#include "ksana_llm/runtime/weight_instance.h"
#include "ksana_llm/utils/attention_backend/attention_backend_manager.h"
#include "ksana_llm/utils/dynamic_memory_pool.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

using namespace ksana_llm;

class ArcHunyuanVideoTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    DeviceMemoryPool::Disable();
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = test_info->name();
    std::string model_path = "/model/ARC-Hunyuan-Video-7B";
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
    batch_scheduler_config.mtp_step_num = 0;  // ArcHunyuanVideo doesn't use MTP
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
  // Sample input tokens for ARC-Hunyuan-Video model
  const std::vector<int> input_ids = {127958, 198,  15225, 100765, 101446, 100699, 88240, 198,  5207,  279,  7422,
                                      1920,   304,  366,   27963,  29,     694,    27963, 29,   323,   1620, 4320,
                                      304,    366,  9399,  29,     694,    9399,   29,    9681, 11,    602,  1770,
                                      2637,   366,  27963, 29,     33811,  1920,   1618,  694,  27963, 1822, 9399,
                                      29,     4320, 1618,  694,    9399,   14611,  128000};
  // Expected tokens may vary based on model behavior
  const std::vector<std::vector<int>> expected_tokens = {{14023, 771}};

  template <typename weight_data_type>
  void TestArcHunyuanVideoForwardWithGreedySampler() {
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
    std::shared_ptr<BaseWeight> arc_hunyuan_video_weight = weight_instance->GetWeight(/*rank*/ 0);
    std::shared_ptr<ArcHunyuanVideoModel> arc_hunyuan_video = std::make_shared<ArcHunyuanVideoModel>(
        model_config, runtime_config, device_id, context, arc_hunyuan_video_weight);
    arc_hunyuan_video->AllocResources(multi_batch_id);

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
    // Initialize xdrotary_embedding_pos_offset for arc_hunyuan_video model
    int64_t xdrotary_embedding_pos_offset_value = 0;
    forward->xdrotary_embedding_pos_offset = &xdrotary_embedding_pos_offset_value;
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
    // Initialize separate xdrotary_embedding_pos_offset for decode_forward
    int64_t decode_xdrotary_embedding_pos_offset_value = 0;
    decode_forward->xdrotary_embedding_pos_offset = &decode_xdrotary_embedding_pos_offset_value;
    std::vector<ForwardRequest *> forward_reqs = {forward.get(), decode_forward.get()};
    Singleton<LayerProgressTracker>::GetInstance()->Initialize(
        runtime_config.parallel_basic_config.tensor_parallel_size, model_config.num_layer);
    Singleton<LayerProgressTracker>::GetInstance()->RegisterCallback([&](int device_id, int layer_index) {
      KLLM_LOG_INFO << "LayerProgressTracker : device_id: " << device_id << " , layer_index: " << layer_index;
    });
    EXPECT_TRUE(arc_hunyuan_video->Forward(multi_batch_id, arc_hunyuan_video_weight, forward_reqs, false).OK());
    Singleton<LayerProgressTracker>::GetInstance()->Cleanup();
    generated_tokens0[0] = arc_hunyuan_video->GetOutputTokensPtr(multi_batch_id)[0];
    generated_tokens1[0] = arc_hunyuan_video->GetOutputTokensPtr(multi_batch_id)[1];
    std::cout << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0], generated_tokens1[0])
              << std::endl;
    // Check if generated tokens match expected tokens
    EXPECT_TRUE(generated_tokens0[0] == expected_tokens[0][0]);
    EXPECT_TRUE(generated_tokens1[0] == expected_tokens[0][0]);

    // Decode
    (*forward_reqs[0]->forwarding_tokens).emplace_back(generated_tokens0[0]);
    (*forward_reqs[1]->forwarding_tokens).emplace_back(generated_tokens1[0]);
    for (auto &forward_req : forward_reqs) {
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }
    EXPECT_TRUE(arc_hunyuan_video->Forward(multi_batch_id, arc_hunyuan_video_weight, forward_reqs, false).OK());
    generated_tokens0[0] = arc_hunyuan_video->GetOutputTokensPtr(multi_batch_id)[0];
    generated_tokens1[0] = arc_hunyuan_video->GetOutputTokensPtr(multi_batch_id)[1];
    std::cout << fmt::format("generated_tokens0: {}, generated_tokens1: {}", generated_tokens0[0], generated_tokens1[0])
              << std::endl;
    // Check if generated tokens match expected tokens
    EXPECT_TRUE(generated_tokens0[0] == expected_tokens[0][1]);
    EXPECT_TRUE(generated_tokens1[0] == expected_tokens[0][1]);

    arc_hunyuan_video.reset();
    arc_hunyuan_video_weight.reset();

    StreamSynchronize(context->GetMemoryManageStreams()[device_id]);
    DeviceSynchronize();
  }
};

TEST_F(ArcHunyuanVideoTest, ForwardBF16WithGreedySamplerTest) {
#ifdef ENABLE_CUDA
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  std::cout << "Test BF16 weight_data_type forward with greedy sampler." << std::endl;
  TestArcHunyuanVideoForwardWithGreedySampler<__nv_bfloat16>();
#endif
}