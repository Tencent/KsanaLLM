/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <Python.h>
#include <stdlib.h>
#include <filesystem>

#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/models/hunyuan_large/hunyuan_large_model.h"
#include "ksana_llm/models/hunyuan_large/hunyuan_large_weight.h"
#include "ksana_llm/samplers/sampler.h"
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

class HunyuanLargeTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1);
    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    const auto &env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path);
    env->batch_scheduler_config_.max_token_len = 256;
    env->ParseModelConfig("/model/hunyuan_large", "/model/hunyuan_large");
    env->GetModelConfig("", model_config);

    BlockManagerConfig block_manager_config;
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

    block_manager_config.device_allocator_config.blocks_num = 10;  // This test just need a few blocks;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;

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
    cache_manager_config.enable_prefix_caching = true;
    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group);
  }

  void TearDown() override {}

 protected:
  ModelConfig model_config;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group;
  std::shared_ptr<Context> context_{nullptr};
  size_t schedule_id = 123;

  CacheManagerConfig cache_manager_config;
  std::shared_ptr<CacheManagerInterface> cache_manager = nullptr;

  template <typename weight_data_type>
  void TestHunyuanLargeForward() {
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
    int rounds = 10;
    EventCreate(&start);
    EventCreate(&stop);

    std::shared_ptr<BaseWeight> hunyuan_large_weight =
        std::make_shared<HunyuanLargeWeight<weight_data_type>>(model_config, 0, context_);
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
      hunyuan_large_weight->LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
      StreamSynchronize(context_->GetMemoryManageStreams()[device_id]);
    }
    hunyuan_large_weight->ProcessWeights();  // End Loader Weight
    std::shared_ptr<HunyuanLargeModel<weight_data_type>> hunyuan_large =
        std::make_shared<HunyuanLargeModel<weight_data_type>>(model_config, 0, context_, hunyuan_large_weight);
    hunyuan_large->AllocResources(schedule_id);

    // ContextDecode
    ForwardRequest forward;
    forward.cache_manager = cache_manager;
    std::vector<int> input_ids = {127958, 127958, 127958, 106360, 102146, 11571, 127966, 127962, 127957};
    forward.forwarding_tokens = &input_ids;
    forward.draft_token_num = 0;
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    forward.flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    forward.logits_buf.resize(1);
    forward.logits_buf[0] = hunyuan_large->GetLogitsPtr(schedule_id);
    forward.logits_offset = 0;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    EmbeddingSlice embedding_slice;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;
    forward.input_refit_embedding = &embedding_slice;
    std::vector<int> block_ids;
    block_allocator_group->GetDeviceBlockAllocator(0)->AllocateBlocks(1, block_ids);
    forward.kv_cache_ptrs.resize(1);
    block_allocator_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward.kv_cache_ptrs[0]);
#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
    // for rank_0
    forward.atb_kv_cache_base_blk_ids.clear();
    forward.atb_kv_cache_base_blk_ids.resize(1);
    // prepare base block ids in rank_0
    int32_t origin_block_id = (uintptr_t(forward.kv_cache_ptrs[0][0]) -
                               uintptr_t(block_allocator_group->GetDeviceBlockAllocator(0)->GetBlocksBasePtr())) /
                              (2 * model_config.num_layer * model_config.block_token_num * model_config.head_num *
                               model_config.size_per_head) /
                              GetTypeSize(Singleton<Environment>::GetInstance()->GetKVCacheType());
    forward.atb_kv_cache_base_blk_ids[0].push_back(
        {static_cast<int32_t>(origin_block_id * 2 * model_config.num_layer)});
#endif
    Memset(forward.kv_cache_ptrs[0][0], 0, Singleton<Environment>::GetInstance()->GetBlockSize());
    KLLM_LOG_DEBUG << fmt::format(
        "kv_cache_ptrs {} end {}", forward.kv_cache_ptrs[0][0],
        forward.kv_cache_ptrs[0][0] + (Singleton<Environment>::GetInstance()->GetBlockSize()));

    ForwardRequest decode_forward = forward;
    decode_forward.cache_manager = cache_manager;
    std::vector<int> decode_ids = input_ids;
    decode_forward.forwarding_tokens = &decode_ids;
    decode_forward.infer_stage = InferStage::STATE_DECODE;
    decode_forward.kv_cached_token_num = decode_forward.forwarding_tokens->size() - 1;
    std::vector<ForwardRequest> forward_reqs = {forward, decode_forward};
    EXPECT_TRUE(hunyuan_large->Forward(schedule_id, hunyuan_large_weight, forward_reqs, false).OK());
    std::vector<ForwardRequest> multi_forward_reqs = {forward, forward};
    EventRecord(start, context_->GetComputeStreams()[device_id]);
    for (int i = 0; i < rounds; ++i) {
      hunyuan_large->Forward(schedule_id, hunyuan_large_weight, multi_forward_reqs, false);
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
    sample_req.input_tokens = &input_ids;
    sample_req.sampling_token_num = 1;
    sample_req.logits_offset = forward_reqs[0].logits_offset;
    sample_req.sampling_result_tokens = &generated_tokens0;
    sample_req.logprobs = &logprobs;
    sample_req.ngram_dict = &ngram_dict;
    sample_req.logits_buf = forward_reqs[0].logits_buf;
    sample_req.model_config = &model_config;
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
    decode_sample_req.logits_offset = forward_reqs[1].logits_offset;
    decode_sample_req.logits_buf = forward_reqs[1].logits_buf;

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

    std::vector<SamplingRequest> sample_reqs = {sample_req, decode_sample_req};
    std::shared_ptr<Sampler> sampler = std::make_shared<Sampler>(batch_scheduler_config, device_id, context_);
    sampler->Sampling(sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(12, generated_tokens0[0]);
    EXPECT_EQ(12, generated_tokens1[0]);
    (*forward_reqs[0].forwarding_tokens).push_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).push_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req.infer_stage = InferStage::STATE_DECODE;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    // Decode
    EXPECT_TRUE(hunyuan_large->Forward(schedule_id, hunyuan_large_weight, forward_reqs, false).OK());
    sampler->Sampling(sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(71, generated_tokens0[0]);
    EXPECT_EQ(71, generated_tokens1[0]);
    (*forward_reqs[0].forwarding_tokens).push_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).push_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }

#ifdef ENABLE_CUDA
    EXPECT_TRUE(hunyuan_large->Forward(schedule_id, hunyuan_large_weight, forward_reqs, false).OK());
    sampler->Sampling(sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(88, generated_tokens0[0]);
    EXPECT_EQ(88, generated_tokens1[0]);
    (*forward_reqs[0].forwarding_tokens).push_back(generated_tokens0[0]);
    (*forward_reqs[1].forwarding_tokens).push_back(generated_tokens1[0]);
    generated_tokens0.clear();
    generated_tokens1.clear();
#endif
    EventRecord(start, context_->GetComputeStreams()[device_id]);
    for (auto &forward_req : multi_forward_reqs) {
      forward_req.infer_stage = InferStage::STATE_DECODE;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    for (int i = 0; i < rounds; ++i) {
      hunyuan_large->Forward(schedule_id, hunyuan_large_weight, multi_forward_reqs, false);
    }
    EventRecord(stop, context_->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "Decode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

    hunyuan_large.reset();
    hunyuan_large_weight.reset();

    StreamSynchronize(context_->GetMemoryManageStreams()[device_id]);
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
  }
};

TEST_F(HunyuanLargeTest, ForwardTest) {
//  Py_Initialize();
#ifdef ENABLE_CUDA
  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_FP16 weight_data_type forward." << std::endl;
  TestHunyuanLargeForward<float16>();
#  ifdef ENABLE_FP8
  // fp8 forward
#  endif

#  ifdef ENABLE_BFLOAT16
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_BF16 weight_data_type forward." << std::endl;
  TestHunyuanLargeForward<bfloat16>();
#    ifdef ENABLE_FP8
  // fp8 forward
#    endif
#  endif
#endif
  //  Py_Finalize();
}
