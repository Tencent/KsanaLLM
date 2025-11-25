/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>
#include <filesystem>

#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/models/common/model_test_helper.h"
#include "ksana_llm/models/llama/llama_model.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/attention_backend/attention_backend_manager.h"
#include "ksana_llm/utils/calc_intvec_hash.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/dynamic_memory_pool.h"
#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/singleton.h"
#include "tests/test.h"

#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include "ksana_llm/utils/pytorch_file_tensor_loader.h"
#include "ksana_llm/utils/safetensors_file_tensor_loader.h"

using namespace ksana_llm;

// 定义一个 LlamaTest 类,继承自 testing::Test
class LlamaTest : public testing::Test {
 protected:
  void SetUp() override {
    DeviceMemoryPool::Disable();

    context_ = std::make_shared<Context>(1, 1, 1);
    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    AttentionBackendManager::GetInstance()->Initialize();
    const auto &env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path);
    env->GetModelConfig(model_config);

    BlockManagerConfig block_manager_config;
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.enable_block_checksum = true;
    context_->SetIsLastLayer(false);
    env->SetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);
    block_manager_config.device_allocator_config.blocks_num = 20;  // This test just need a few blocks;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;

    env->GetRuntimeConfig(runtime_config);

    BlockAllocatorGroupConfig group_1_config;
    group_1_config.devices = {0};
    group_1_config.device_block_num = block_manager_config.device_allocator_config.blocks_num;
    group_1_config.host_block_num = block_manager_config.host_allocator_config.blocks_num;
    group_1_config.block_size = block_manager_config.device_allocator_config.block_size;

    BlockAllocatorManagerConfig block_allocator_manager_config;
    block_allocator_manager_config[1] = group_1_config;

    CacheManagerConfig cache_manager_config;
    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.swap_threadpool_size = 2;
    cache_manager_config.enable_prefix_caching = true;

    memory_allocator_ = std::make_shared<MemoryAllocator>();
    BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_,
                                                  block_allocator_creation_fn_);
    block_allocator_group_ = block_allocator_manager.GetBlockAllocatorGroup(1);
    cache_manager_ = std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group_);
  }

  void TearDown() override {}

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  std::shared_ptr<Context> context_{nullptr};
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;
  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;
  std::shared_ptr<PrefixCacheManager> cache_manager_ = nullptr;
  size_t schedule_id = 0;

  template <typename weight_data_type>
  void TestLlamaForward() {
    int device_id = 0;
    SetDevice(device_id);
#ifdef ENABLE_FP8
    if (model_config.quant_config.method == QUANT_FP8_E4M3 && !context_->IsGemmFp8Supported()) {
      std::cout << "Cublas is insufficient to support FP8, skip test." << std::endl;
      return;
    }
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

    std::shared_ptr<BaseWeight> llama_weight =
        std::make_shared<LlamaWeight<weight_data_type>>(model_config, runtime_config, 0, context_);
    // Start Loader Weight
    ModelFileFormat model_file_format;
    std::vector<std::string> weights_file_list = SearchLocalPath(model_path, model_file_format);
    bool load_bias = true;
    for (std::string &file_name : weights_file_list) {
      std::shared_ptr<BaseFileTensorLoader> weights_loader = nullptr;
      if (model_file_format == SAFETENSORS) {
        weights_loader = std::make_shared<SafeTensorsLoader>(file_name, load_bias);
      } else if (model_file_format == GGUF) {
        weights_loader = std::make_shared<GGUFFileTensorLoader>(file_name, load_bias);
      } else {
        weights_loader = std::make_shared<PytorchFileTensorLoader>(file_name, load_bias);
      }
      std::vector<std::string> weight_name_list = weights_loader->GetTensorNameList();
      std::vector<std::string> custom_name_list;

      GetCustomNameList(model_config.path, model_config.type, weight_name_list, custom_name_list, model_file_format);

      llama_weight->LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
      StreamSynchronize(context_->GetMemoryManageStreams()[device_id]);
    }
    llama_weight->ProcessWeights();  // End Loader Weight
    std::shared_ptr<LlamaModel> llama =
        std::make_shared<LlamaModel>(model_config, runtime_config, 0, context_, llama_weight);
    llama->AllocResources(schedule_id);

    // Weight Name Check
    // 正确的 weight 名称
    std::string weight_name = "lm_head.weight";
    Tensor lm_head = llama_weight->GetModelWeights(weight_name);
    EXPECT_EQ(lm_head.location, MemoryLocation::LOCATION_DEVICE);
#ifdef ENABLE_CUDA
    // TODO(karlluo): GPU weight load implement without trans is inefficient, karl will enhance it someday.
    EXPECT_EQ(std::vector<size_t>(lm_head.shape), std::vector<size_t>({4096, 32000}));
#endif

    // 错误的 weight 名称
    weight_name = "wrong_name";
    Tensor wrong_tensor = llama_weight->GetModelWeights(weight_name);
    EXPECT_EQ(wrong_tensor.location, MemoryLocation::LOCATION_UNKNOWN);
    EXPECT_TRUE(wrong_tensor.shape.empty());

    // ContextDecode
    std::vector<int> input_ids = {233, 1681};
    ForwardRequestBuilderForTest request_builder(model_config, runtime_config, cache_manager_);
    auto forward = request_builder.CreateForwardRequest(1, input_ids);

    KLLM_LOG_DEBUG << fmt::format("kv_cache_ptrs {} end {}", forward->kv_cache_ptrs[0][0],
                                  forward->kv_cache_ptrs[0][0] + runtime_config.attn_backend_config.block_size);
    std::vector<ForwardRequest *> forward_reqs = {forward};
    EXPECT_TRUE(llama->Forward(schedule_id, llama_weight, forward_reqs, false).OK());

    std::vector<ForwardRequest *> multi_forward_reqs = {forward, forward};
    EventRecord(start, context_->GetComputeStreams()[device_id]);
    for (int i = 0; i < rounds; ++i) {
      llama->Forward(schedule_id, llama_weight, multi_forward_reqs, false);
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
    std::vector<int> sampling_result_tokens;
    sample_req.input_tokens = std::make_shared<std::vector<int>>(input_ids);
    sample_req.logits_offset = forward_reqs[0]->logits_offset;
    sample_req.sampling_token_num = 1;
    sample_req.sampling_result_tokens = &sampling_result_tokens;
    sample_req.logprobs = std::make_shared<std::vector<std::vector<std::pair<int, float>>>>(logprobs);
    sample_req.ngram_dict = &ngram_dict;
    sample_req.logits_buf = std::vector<float*>{llama->GetLogitsPtr(schedule_id)};
    SamplingConfig sampling_config;
    sample_req.sampling_config = &sampling_config;
    sample_req.request_target = std::make_shared<const std::map<std::string, TargetDescribe>>();
    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

    std::vector<SamplingRequest> sample_reqs = {sample_req};
    std::shared_ptr<Sampler> sampler = std::make_shared<Sampler>(batch_scheduler_config, device_id, context_);
    sampler->Sampling(0, sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(29871, sampling_result_tokens[0]);
    (*forward_reqs[0]->forwarding_tokens).push_back(sampling_result_tokens[0]);
    sampling_result_tokens.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req->infer_stage = InferStage::kDecode;
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }
    // Decode
    EXPECT_TRUE(llama->Forward(schedule_id, llama_weight, forward_reqs, false).OK());
    sampler->Sampling(0, sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(29896, sampling_result_tokens[0]);
    (*forward_reqs[0]->forwarding_tokens).push_back(sampling_result_tokens[0]);
    sampling_result_tokens.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }

#ifdef ENABLE_CUDA
    EXPECT_TRUE(llama->Forward(schedule_id, llama_weight, forward_reqs, false).OK());
    sampler->Sampling(0, sample_reqs, context_->GetComputeStreams()[device_id]);
    EXPECT_EQ(29929, sampling_result_tokens[0]);
    (*forward_reqs[0]->forwarding_tokens).push_back(sampling_result_tokens[0]);
    sampling_result_tokens.clear();
#endif

    EventRecord(start, context_->GetComputeStreams()[device_id]);
    for (auto &forward_req : multi_forward_reqs) {
      forward_req->infer_stage = InferStage::kDecode;
      forward_req->kv_cached_token_num = forward_req->forwarding_tokens->size() - 1;
    }
    for (int i = 0; i < rounds; ++i) {
      llama->Forward(schedule_id, llama_weight, multi_forward_reqs, false);
    }
    EventRecord(stop, context_->GetComputeStreams()[device_id]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    std::cout << "Decode milliseconds / " << rounds << " is: " << milliseconds / rounds << std::endl;

#ifdef ENABLE_CUDA
    // This maybe cost more than 30 seconds.
    EXPECT_TRUE((milliseconds / rounds) < 60);

    // Test logits_custom_length
    std::vector<int> prompt_probs_input_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    forward->forwarding_tokens = std::make_shared<std::vector<int>>(prompt_probs_input_tokens);
    forward->logits_custom_length = 5;
    forward->infer_stage = InferStage::kContext;
    forward->kv_cached_token_num = 0;
    forward->sampling_token_num = forward->logits_custom_length;
    std::map<std::string, ksana_llm::TargetDescribe> request_target;
    ksana_llm::TargetDescribe target_describe;
    target_describe.slice_pos.push_back({0, 4});
    request_target["logits"] = target_describe;
    forward->request_target = std::make_shared<const std::map<std::string, TargetDescribe>>(request_target);

    std::vector<ForwardRequest *> prompt_probs_forward_reqs = {forward, forward};
    ModelInput model_input(model_config, runtime_config, 0, context_);
    model_input.ParseFromRequests(prompt_probs_forward_reqs);
    std::vector<uint64_t> result(model_input.logits_idx_uint64_tensor.GetElementNumber());
    std::vector<uint64_t> dst = {0, 1, 2, 3, 4, 9, 10, 11, 12, 13};
    Memcpy(result.data(), model_input.logits_idx_uint64_tensor.GetPtr<void>(), result.size() * sizeof(uint64_t),
           MEMCPY_DEVICE_TO_HOST);
    EXPECT_EQ(dst.size(), result.size());
    for (size_t i = 0; i < result.size(); i++) {
      EXPECT_EQ(result[i], dst[i]);
    }
    EXPECT_TRUE(llama->Forward(schedule_id, llama_weight, prompt_probs_forward_reqs, false).OK());
    // 模拟多次请求，计算checksum
    context_->SetIsLastLayer(true);
    auto forward_checksum = request_builder.CreateForwardRequest(2, input_ids);
    forward_checksum->block_checksums->resize(1);
    forward_checksum->checksummed_block_num->resize(1, 0);
    std::vector<ForwardRequest *> forward_reqs_checksum = {forward_checksum};
    for (int i = 0; i < 3; i++) {
      forward_checksum->forwarding_tokens->insert(forward_checksum->forwarding_tokens->end(), 16, 1);
      forward_checksum->infer_stage = InferStage::kDecode;
      forward_checksum->kv_cached_token_num = forward_checksum->forwarding_tokens->size() - 1;
      model_input.ParseFromRequests(forward_reqs_checksum);
      EXPECT_TRUE(llama->Forward(schedule_id, llama_weight, forward_reqs_checksum, false).OK());
    }
#else
    // NOTE(karlluo): ACL inference is slower than CUDA
    EXPECT_TRUE((milliseconds / rounds) < 300) << "milliseconds / " << rounds << " is: " << milliseconds / rounds;
#endif

    llama.reset();
    llama_weight.reset();

    StreamSynchronize(context_->GetMemoryManageStreams()[device_id]);
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
    context_->SetIsLastLayer(false);
  }
};

TEST_F(LlamaTest, ForwardTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_FP16 weight_data_type forward." << std::endl;
  TestLlamaForward<float16>();
#ifdef ENABLE_FP8
  // fp8 forward
  model_config.is_quant = true;
  model_config.quant_config.method = QUANT_FP8_E4M3;
  model_config.quant_config.is_checkpoint_fp8_serialized = false;
  std::cout << "Test TYPE_FP16 weight_data_type with QUANT_FP8_E4M3 forward" << std::endl;
  TestLlamaForward<float16>();
#endif

#ifdef ENABLE_CUDA
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_BF16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_BF16 weight_data_type forward." << std::endl;
  TestLlamaForward<bfloat16>();
#  ifdef ENABLE_FP8
  // fp8 forward
  model_config.is_quant = true;
  model_config.quant_config.method = QUANT_FP8_E4M3;
  model_config.quant_config.is_checkpoint_fp8_serialized = false;
  std::cout << "Test TYPE_BF16 weight_data_type with QUANT_FP8_E4M3 forward" << std::endl;
  TestLlamaForward<bfloat16>();
#  endif
#endif
}

TEST(TorchTensorTest, TorchTensorTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif

#ifdef ENABLE_CUDA
  int device_id = 0;
  SetDevice(device_id);
  // 设定张量的大小
  const int64_t size = 10;

  // 在GPU上分配内存
  void *a_ptr, *b_ptr, *c_ptr;
  Malloc(&a_ptr, size * sizeof(float));
  Malloc(&b_ptr, size * sizeof(float));
  Malloc(&c_ptr, size * sizeof(float));

  // 创建并初始化输入数据
  std::vector<float> a_host(size, 1.0), b_host(size, 2.0);

  // 将数据复制到GPU
  Memcpy(a_ptr, a_host.data(), size * sizeof(float), MEMCPY_HOST_TO_DEVICE);
  Memcpy(b_ptr, b_host.data(), size * sizeof(float), MEMCPY_HOST_TO_DEVICE);

  // 创建torch::Tensor，它们共享GPU内存
  auto options = torch::TensorOptions().device(torch::kCUDA, device_id).dtype(torch::kFloat32);
  torch::Tensor a = torch::from_blob(a_ptr, {size}, options);
  torch::Tensor b = torch::from_blob(b_ptr, {size}, options);
  torch::Tensor c = torch::from_blob(c_ptr, {size}, options);

  // 计算a + b = c
  c.copy_(a.add_(b));
  std::ostringstream oss;
  // 传输到cpu打印
  oss << c.to(torch::kCPU);
  EXPECT_EQ('3', oss.str()[1]);

  // 将结果复制回CPU以进行验证
  std::vector<float> c_host(size);
  Memcpy(c_host.data(), c_ptr, size * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  // 验证结果
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(c_host[i], 3.0);
  }

  // 清理GPU内存
  Free(a_ptr);
  Free(b_ptr);
  Free(c_ptr);
#endif
}
