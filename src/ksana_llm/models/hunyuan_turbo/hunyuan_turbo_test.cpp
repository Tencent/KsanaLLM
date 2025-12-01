/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <Python.h>
#include <stdlib.h>
#include <filesystem>

#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/models/hunyuan_turbo/hunyuan_turbo_model.h"
#include "ksana_llm/models/hunyuan_turbo/hunyuan_turbo_weight.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/attention_backend/attention_backend_manager.h"
#include "ksana_llm/utils/calc_intvec_hash.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include "ksana_llm/utils/pytorch_file_tensor_loader.h"
#include "ksana_llm/utils/safetensors_file_tensor_loader.h"

using namespace ksana_llm;

class HunyuanTurboTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1, 1);
    // read config.json,
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    AttentionBackendManager::GetInstance()->Initialize();
    const auto &env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path, "/model/hunyuan_turbo/Hunyuan-Turbo-2B");
    env->GetModelConfig(model_config);

    BlockManagerConfig block_manager_config;
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

    block_manager_config.device_allocator_config.blocks_num = 10;  // This test just need a few blocks;
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
    BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_);
    block_allocator_group = block_allocator_manager.GetBlockAllocatorGroup(1);

    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.swap_threadpool_size = 2;
    cache_manager_config.enable_prefix_caching = false;
    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group);
  }

  void TearDown() override {}

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group;
  std::shared_ptr<Context> context_{nullptr};
  size_t schedule_id = 0;

  CacheManagerConfig cache_manager_config;
  std::shared_ptr<CacheManagerInterface> cache_manager = nullptr;

  template <typename weight_data_type>
  void TestHunyuanTurboForward() {
    int device_id = 0;
    SetDevice(device_id);
    std::filesystem::path model_path(model_config.path);
    if (!std::filesystem::exists(model_path)) {
      KLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
      EXPECT_TRUE(std::filesystem::exists(model_path));
    }
    Event start;
    Event stop;
    float milliseconds = 0;
    int rounds = 10;
    EventCreate(&start);
    EventCreate(&stop);

    std::shared_ptr<BaseWeight> hunyuan_turbo_weight =
        std::make_shared<HunyuanTurboWeight<weight_data_type>>(model_config, runtime_config, 0, context_);
    // Start Loader Weight
    ModelFileFormat model_file_format;
    std::vector<std::string> weights_file_list = SearchLocalPath(model_path, model_file_format);
    for (std::string &file_name : weights_file_list) {
      std::shared_ptr<BaseFileTensorLoader> weights_loader = nullptr;
      if (model_file_format == SAFETENSORS) {
        weights_loader = std::make_shared<SafeTensorsLoader>(file_name, model_config.load_bias);
      } else if (model_file_format == GGUF) {
        weights_loader = std::make_shared<GGUFFileTensorLoader>(file_name, model_config.load_bias);
      } else {
        weights_loader = std::make_shared<PytorchFileTensorLoader>(file_name, model_config.load_bias);
      }
      std::vector<std::string> weight_name_list = weights_loader->GetTensorNameList();
      std::vector<std::string> custom_name_list;

      GetCustomNameList(model_config.path, model_config.type, weight_name_list, custom_name_list, model_file_format);
      hunyuan_turbo_weight->LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
      StreamSynchronize(context_->GetMemoryManageStreams()[device_id]);
    }
    hunyuan_turbo_weight->ProcessWeights();  // End Loader Weight
    std::shared_ptr<HunyuanTurboModel> hunyuan_turbo =
        std::make_shared<HunyuanTurboModel>(model_config, runtime_config, 0, context_, hunyuan_turbo_weight);
    hunyuan_turbo->AllocResources(schedule_id);

    // ContextDecode
    auto forward = std::make_unique<ForwardRequest>();
    forward->cache_manager = cache_manager;
    std::vector<int> input_ids = {233, 1203};
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
    block_allocator_group->GetDeviceBlockAllocator(0)->AllocateBlocks(1, block_ids);
    forward->kv_cache_ptrs.resize(1);
    block_allocator_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward->kv_cache_ptrs[0]);
#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
    forward->atb_kv_cache_base_blk_ids.assign(1, {});
    AppendFlatKVCacheBlkIds(model_config.num_layer, {block_ids}, forward->atb_kv_cache_base_blk_ids, cache_manager);
#endif
    Memset(forward->kv_cache_ptrs[0][0], 0, runtime_config.attn_backend_config.block_size);
    KLLM_LOG_DEBUG << fmt::format("kv_cache_ptrs {} end {}", forward->kv_cache_ptrs[0][0],
                                  forward->kv_cache_ptrs[0][0] + (runtime_config.attn_backend_config.block_size));

    auto decode_forward = std::make_unique<ForwardRequest>(*forward);
    decode_forward->cache_manager = cache_manager;
    std::vector<int> decode_ids = input_ids;
    decode_forward->forwarding_tokens = std::make_shared<std::vector<int>>(decode_ids);
    decode_forward->infer_stage = InferStage::kDecode;
    decode_forward->kv_cached_token_num = decode_forward->forwarding_tokens->size() - 1;
    std::vector<ForwardRequest *> forward_reqs = {forward.get(), decode_forward.get()};
    EXPECT_TRUE(hunyuan_turbo->Forward(schedule_id, hunyuan_turbo_weight, forward_reqs, false).OK());

    std::vector<ForwardRequest *> multi_forward_reqs = {forward.get(), forward.get()};
    EventRecord(start, context_->GetComputeStreams()[device_id]);
    for (int i = 0; i < rounds; ++i) {
      hunyuan_turbo->Forward(schedule_id, hunyuan_turbo_weight, multi_forward_reqs, false);
    }
    EventRecord(stop, context_->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "ContextDecode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

#ifdef ENABLE_CUDA
    EXPECT_TRUE((milliseconds / rounds) < 35);
#else
    // NOTE(karlluo): ACL inference is slower than CUDA
    EXPECT_TRUE((milliseconds / rounds) < 300) << "milliseconds / " << rounds << " is: " << milliseconds / rounds;
#endif

    // Sampling
    SamplingRequest sample_req;
    NgramDict ngram_dict;
    std::vector<std::vector<std::pair<int, float>>> logprobs;
    std::vector<float> prompt_probs;
    std::vector<int> generated_tokens0, generated_tokens1;
    std::map<std::string, TargetDescribe> request_target;
    sample_req.input_tokens = &input_ids;
    sample_req.sampling_token_num = 1;
    sample_req.logits_offset = forward_reqs[0]->logits_offset;
    sample_req.sampling_result_tokens = &generated_tokens0;
    sample_req.logprobs = &logprobs;
    sample_req.ngram_dict = &ngram_dict;
    sample_req.logits_buf = std::vector<float *>{hunyuan_turbo->GetLogitsPtr(schedule_id)};
    SamplingConfig sample_config;
    sample_config.num_beams = 1;
    sample_config.topk = 1;
    sample_config.topp = 0;
    sample_config.temperature = 0;
    sample_config.repetition_penalty = 1;
    sample_config.no_repeat_ngram_size = 0;
    sample_config.encoder_no_repeat_ngram_size = 0;
    sample_req.sampling_config = &sample_config;
    sample_req.request_target = &request_target;

    SamplingRequest decode_sample_req = sample_req;
    decode_sample_req.sampling_result_tokens = &generated_tokens1;
    decode_sample_req.logits_offset = forward_reqs[1]->logits_offset;
    decode_sample_req.logits_buf = std::vector<float *>{hunyuan_turbo->GetLogitsPtr(schedule_id)};

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

    std::vector<SamplingRequest *> sample_reqs = {&sample_req, &decode_sample_req};
    std::shared_ptr<Sampler> sampler = std::make_shared<Sampler>(batch_scheduler_config, device_id, context_);
    sampler->Sampling(0, sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(311, generated_tokens0[0]);
    EXPECT_EQ(311, generated_tokens1[0]);
    (*forward_reqs[0]->forwarding_tokens).push_back(generated_tokens0[0]);
    (*forward_reqs[1]->forwarding_tokens).push_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req->infer_stage = InferStage::kDecode;
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }
    // Decode
    EXPECT_TRUE(hunyuan_turbo->Forward(schedule_id, hunyuan_turbo_weight, forward_reqs, false).OK());
    sampler->Sampling(0, sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(279, generated_tokens0[0]);
    EXPECT_EQ(279, generated_tokens1[0]);
    (*forward_reqs[0]->forwarding_tokens).push_back(generated_tokens0[0]);
    (*forward_reqs[1]->forwarding_tokens).push_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }

#ifdef ENABLE_CUDA
    EXPECT_TRUE(hunyuan_turbo->Forward(schedule_id, hunyuan_turbo_weight, forward_reqs, false).OK());
    sampler->Sampling(0, sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(4113, generated_tokens0[0]);
    EXPECT_EQ(4113, generated_tokens1[0]);
    (*forward_reqs[0]->forwarding_tokens).push_back(generated_tokens0[0]);
    (*forward_reqs[1]->forwarding_tokens).push_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
#endif
    EventRecord(start, context_->GetComputeStreams()[device_id]);
    for (auto &forward_req : multi_forward_reqs) {
      forward_req->infer_stage = InferStage::kDecode;
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }
    for (int i = 0; i < rounds; ++i) {
      hunyuan_turbo->Forward(schedule_id, hunyuan_turbo_weight, multi_forward_reqs, false);
    }
    EventRecord(stop, context_->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "Decode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

    hunyuan_turbo.reset();
    hunyuan_turbo_weight.reset();

    StreamSynchronize(context_->GetMemoryManageStreams()[device_id]);
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }
};

TEST_F(HunyuanTurboTest, ForwardTest) {
#ifdef ENABLE_CUDA
  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_FP16 weight_data_type forward." << std::endl;
  TestHunyuanTurboForward<float16>();
#  ifdef ENABLE_FP8
  // fp8 forward
#  endif

  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_BF16 weight_data_type forward." << std::endl;
  TestHunyuanTurboForward<bfloat16>();
#  ifdef ENABLE_FP8
  // fp8 forward
#  endif
#endif
}
