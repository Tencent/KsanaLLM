/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>
#include <filesystem>

#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/models/deepseek_v3/deepseek_v3_model.h"
#include "ksana_llm/models/deepseek_v3/deepseek_v3_weight.h"
#include "ksana_llm/runtime/layer_progress_tracker.h"
#include "ksana_llm/runtime/llm_runtime.h"
#include "ksana_llm/runtime/weight_instance.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"
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

class DeepSeekV3DPTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    DeviceMemoryPool::Disable();

    std::string model_path = "/model/DeepSeek-R1-17832-fix-mtp";
    std::string yaml_path = "../../../../examples/deepseekv2/ksana_llm_deepseek_v2_tp2_dp2.yaml";
    context = std::make_shared<Context>(2, 2, 1);

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / yaml_path;
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    const auto &env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path, model_path);
    BatchSchedulerConfig batch_scheduler_config;
    env->GetBatchSchedulerConfig(batch_scheduler_config);
    batch_scheduler_config.enable_mtp_module = true;
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

    // Initialize cache manager for every dp group.
    BlockAllocatorManagerConfig block_allocator_manager_configs;
    for (int dp_rank : {0, 1}) {
      BlockAllocatorGroupConfig group_config;
      group_config.devices = {dp_rank};
      group_config.device_block_num = block_manager_config.device_allocator_config.blocks_num;
      group_config.host_block_num = block_manager_config.host_allocator_config.blocks_num;
      group_config.block_size = block_manager_config.device_allocator_config.block_size;

      block_allocator_manager_configs[dp_rank] = group_config;
    }

    std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = std::make_shared<MemoryAllocator>();
    BlockAllocatorManager block_allocator_manager(block_allocator_manager_configs, memory_allocator_, context);

    for (int dp_rank : {0, 1}) {
      std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group =
          block_allocator_manager.GetBlockAllocatorGroup(dp_rank);
      block_allocator_groups.push_back(block_allocator_group);

      CacheManagerConfig cache_manager_config;
      cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
      cache_manager_config.tensor_para_size = 1;
      cache_manager_config.swap_threadpool_size = 2;
      cache_manager_config.enable_prefix_caching = true;
      cache_managers.push_back(std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group));
    }
  }

  void TearDown() override {}

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  std::vector<std::shared_ptr<BlockAllocatorGroupInterface>> block_allocator_groups;
  std::vector<std::shared_ptr<PrefixCacheManager>> cache_managers;
  std::shared_ptr<Context> context{nullptr};
  size_t multi_batch_id = 0;
  std::vector<std::shared_ptr<DeepSeekV3Model>> deepseek_v3_dps;
  std::vector<std::shared_ptr<BaseWeight>> deepseek_v3_weight_dps;

  void LoadModel(std::filesystem::path &model_path) {
    for (int device_id : {0, 1}) {
      SetDevice(device_id);
      std::unique_ptr<WeightInstance> weight_instance =
          std::make_unique<WeightInstance>(model_config, runtime_config, context);
      weight_instance->Load();
      std::shared_ptr<BaseWeight> deepseek_v3_weight = weight_instance->GetWeight(device_id);
      std::shared_ptr<DeepSeekV3Model> deepseek_v3 =
          std::make_shared<DeepSeekV3Model>(model_config, runtime_config, device_id, context, deepseek_v3_weight);
      deepseek_v3->AllocResources(multi_batch_id);

      deepseek_v3_dps.push_back(deepseek_v3);
      deepseek_v3_weight_dps.push_back(deepseek_v3_weight);
    }
  }

  ForwardRequest CreateContextForwardRequest(int req_id, int dp_group_id, std::vector<int> &input_ids,
                                             std::vector<FlexibleCachedCopyTask> &flexible_cached_copy_tasks,
                                             EmbeddingSlice &embedding_slice) {
    ForwardRequest forward;
    forward.req_id = req_id;
    forward.cache_manager = cache_managers[dp_group_id];
    forward.attn_dp_group_id = dp_group_id;
    forward.forwarding_tokens = &input_ids;
    forward.draft_token_num = 0;
    forward.flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    forward.logits_buf.resize(2);  // tp 2
    forward.logits_buf[0] = deepseek_v3_dps[0]->GetLogitsPtr(multi_batch_id);
    forward.logits_buf[1] = deepseek_v3_dps[1]->GetLogitsPtr(multi_batch_id);
    forward.logits_offset = 0;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;
    forward.input_refit_embedding = &embedding_slice;

    std::vector<int> block_ids;
    int use_block_num = (input_ids.size() + runtime_config.attn_backend_config.block_token_num - 1) /
                        runtime_config.attn_backend_config.block_token_num;
    block_allocator_groups[dp_group_id]->GetDeviceBlockAllocator(0)->AllocateBlocks(use_block_num, block_ids);
    forward.kv_cache_ptrs.resize(1);  // device num in dp group.
    block_allocator_groups[dp_group_id]->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward.kv_cache_ptrs[0]);

    LlmRuntime::BuildFlatKVCacheBlkIds(model_config.num_layer + model_config.num_nextn_predict_layers, {block_ids},
                                       forward.atb_kv_cache_base_blk_ids, cache_managers[dp_group_id]);
    for (int block_idx = 0; block_idx < use_block_num; block_idx++) {
      Memset(forward.kv_cache_ptrs[0][block_idx], 0, runtime_config.attn_backend_config.block_size);
    }

    return forward;
  }

  void MakeDecodeForwardRequest(ForwardRequest &forward_req, int new_token) {
    forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size();
    forward_req.forwarding_tokens->push_back(new_token);
    forward_req.infer_stage = InferStage::STATE_DECODE;
  }

  SamplingRequest CreateSamplingRequest(ForwardRequest &forward_req,
                                        std::vector<std::vector<std::pair<int, float>>> &logprobs,
                                        std::vector<int> &generated_tokens, NgramDict &ngram_dict,
                                        SamplingConfig &sample_config) {
    SamplingRequest sample_req;
    sample_req.req_id = forward_req.req_id;
    sample_req.input_tokens = forward_req.forwarding_tokens;
    sample_req.sampling_token_num = 1;
    sample_req.logits_offset = forward_req.logits_offset;
    sample_req.sampling_result_tokens = &generated_tokens;
    sample_req.logprobs = &logprobs;
    sample_req.ngram_dict = &ngram_dict;
    sample_req.logits_buf = forward_req.logits_buf;
    sample_req.model_config = &model_config;
    sample_config.num_beams = 1;
    sample_config.topk = 1;
    sample_config.topp = 0;
    sample_config.temperature = 0;
    sample_config.repetition_penalty = 1;
    sample_config.no_repeat_ngram_size = 0;
    sample_config.encoder_no_repeat_ngram_size = 0;
    sample_req.sampling_config = &sample_config;

    return sample_req;
  }

  void ExecuteForward(std::vector<ForwardRequest> &forward_reqs) {
    std::vector<std::thread> forward_threads;
    for (int device_id : {0, 1}) {
      forward_threads.emplace_back([&, this, device_id]() {
        SetDevice(device_id);
        EXPECT_TRUE(deepseek_v3_dps[device_id]
                        ->Forward(multi_batch_id, deepseek_v3_weight_dps[device_id], forward_reqs, false)
                        .OK());
      });
    }
    for (auto &thread : forward_threads) {
      thread.join();
    }
  }

  void TestDeepSeekV3DPForward() {
    std::filesystem::path model_path(model_config.path);
    if (!std::filesystem::exists(model_path)) {
      KLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
      EXPECT_TRUE(std::filesystem::exists(model_path));
    }

    // Load model weights.
    LoadModel(model_path);

    // /////////////////////////////////// Context ///////////////////////////////
    EmbeddingSlice embedding_slice;
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;

    // dp group 0
    std::vector<int> input_ids0 = {100000, 100000, 5726, 25, 207, 31104, 8672, 2224, 185, 185, 77398, 25};
    ForwardRequest forward0 =
        CreateContextForwardRequest(0, 0, input_ids0, flexible_cached_copy_tasks, embedding_slice);

    std::vector<int> input_ids2 = {100000, 100000, 5726, 25, 207, 31104, 8672, 2224, 185, 185, 77398, 25};
    ForwardRequest forward2 =
        CreateContextForwardRequest(2, 0, input_ids2, flexible_cached_copy_tasks, embedding_slice);

    // dp group 1
    std::vector<int> input_ids1 = {100000, 100000, 5726, 25, 207, 31104, 8672, 2224, 185, 185, 77398, 25};
    ForwardRequest forward1 =
        CreateContextForwardRequest(1, 1, input_ids1, flexible_cached_copy_tasks, embedding_slice);

    std::vector<int> input_ids3 = {100000, 100000, 5726, 25, 207, 31104, 8672, 2224, 185, 185, 77398, 25};
    ForwardRequest forward3 =
        CreateContextForwardRequest(3, 1, input_ids3, flexible_cached_copy_tasks, embedding_slice);

    auto hash_fn = [](float *dev_logist_ptr, size_t dim_size) -> size_t {
      float *host_ptr = reinterpret_cast<float *>(malloc(dim_size * sizeof(float)));
      Memcpy(host_ptr, dev_logist_ptr, dim_size * sizeof(float), MEMCPY_DEVICE_TO_HOST);
      std::string str(reinterpret_cast<const char *>(host_ptr), sizeof(float) * dim_size);

      std::hash<std::string> hasher;
      size_t hash = hasher(str);
      free(host_ptr);
      return hash;
    };

    // Execute forward, sorted by (forwarding_token_size, dp_group_id)
    std::vector<ForwardRequest> context_forward_reqs = {forward0, forward2, forward1, forward3};
    ExecuteForward(context_forward_reqs);

    // Chdck data hash.
    float *logist_base_ptr = deepseek_v3_dps[0]->GetLogitsPtr(multi_batch_id);
    size_t context_hash0 = hash_fn(logist_base_ptr, model_config.vocab_size);
    size_t context_hash2 = hash_fn(logist_base_ptr + model_config.vocab_size, model_config.vocab_size);
    size_t context_hash1 = hash_fn(logist_base_ptr + (model_config.vocab_size * 2), model_config.vocab_size);
    size_t context_hash3 = hash_fn(logist_base_ptr + (model_config.vocab_size * 3), model_config.vocab_size);
    EXPECT_EQ(context_hash0, context_hash2);
    EXPECT_EQ(context_hash0, context_hash1);
    EXPECT_EQ(context_hash0, context_hash3);

    // /////////////////////////////////// Decode ///////////////////////////////
    MakeDecodeForwardRequest(forward0, 24588);
    MakeDecodeForwardRequest(forward1, 24588);

    // Execute forward
    std::vector<ForwardRequest> decode_forward_reqs = {forward0, forward1};
    ExecuteForward(decode_forward_reqs);

    // Check data hash.
    size_t decode_hash0 = hash_fn(logist_base_ptr, model_config.vocab_size);
    size_t decode_hash1 = hash_fn(logist_base_ptr + model_config.vocab_size, model_config.vocab_size);
    EXPECT_EQ(decode_hash0, decode_hash1);

    // ///////////////////////// Mixed of context and decode ///////////////////
    // dp group 0
    std::vector<int> input_ids4 = {100000, 100000, 5726, 25, 207, 31104, 8672, 2224, 185, 185, 77398, 25};
    ForwardRequest forward4 =
        CreateContextForwardRequest(4, 0, input_ids4, flexible_cached_copy_tasks, embedding_slice);

    // dp group 1
    std::vector<int> input_ids5 = {100000, 100000, 5726, 25, 207, 31104, 8672, 2224, 185, 185, 77398, 25};
    ForwardRequest forward5 =
        CreateContextForwardRequest(5, 1, input_ids5, flexible_cached_copy_tasks, embedding_slice);

    MakeDecodeForwardRequest(forward2, 24588);
    MakeDecodeForwardRequest(forward3, 24588);

    // Execute forward, sorted by (dp_group_id, forwarding_token_size)
    std::vector<ForwardRequest> mixed_forward_reqs = {forward4, forward5, forward2, forward3};
    ExecuteForward(mixed_forward_reqs);

    // Check data hash.
    size_t mixed_hash4 = hash_fn(logist_base_ptr, model_config.vocab_size);
    size_t mixed_hash5 = hash_fn(logist_base_ptr + model_config.vocab_size, model_config.vocab_size);
    size_t mixed_hash2 = hash_fn(logist_base_ptr + (model_config.vocab_size * 2), model_config.vocab_size);
    size_t mixed_hash3 = hash_fn(logist_base_ptr + (model_config.vocab_size * 3), model_config.vocab_size);
    EXPECT_EQ(mixed_hash4, mixed_hash5);
    EXPECT_EQ(mixed_hash4, context_hash0);
    EXPECT_EQ(mixed_hash2, mixed_hash3);
    EXPECT_EQ(mixed_hash2, decode_hash0);
  }
};

TEST_F(DeepSeekV3DPTest, DataParallelTest) {
#ifdef ENABLE_CUDA
  model_config.is_quant = true;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_BLOCK_FP8_E4M3;
  TestDeepSeekV3DPForward();
#endif
}
