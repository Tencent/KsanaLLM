/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>
#include <filesystem>

#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/models/common/model_test_helper.h"
#include "ksana_llm/models/llama/llama_model.h"
#include "ksana_llm/samplers/sampler.h"
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

// 比较两个浮点数的相对误差
bool ExpectNearRelative(float expected, float actual, float rel_error) {
  // 处理接近零的情况
  if (std::abs(expected) < 1e-10 && std::abs(actual) < 1e-10) {
    return true;
  }

  // 计算相对误差
  float relative_diff = std::abs((expected - actual) / expected);
  if (relative_diff <= rel_error) {
    return true;
  } else {
    std::cerr << "Relative error too large: expected " << expected << ", actual " << actual << ", relative diff "
              << relative_diff << " > " << rel_error << std::endl;
    return false;
  }
}

// 定义一个 ForwardTest 类，继承自 testing::Test
class ForwardTest : public testing::Test {
 protected:
  void SetUp() override {
    DeviceMemoryPool::Disable();

    context_ = std::make_shared<Context>(1, 1, 1);
    // 解析 config.json，初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    const auto& env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path);
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

    CacheManagerConfig cache_manager_config;
    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.swap_threadpool_size = 2;
    cache_manager_config.enable_prefix_caching = false;

    memory_allocator_ = std::make_shared<MemoryAllocator>();
    BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_,
                                                  block_allocator_creation_fn_);
    block_allocator_group_ = block_allocator_manager.GetBlockAllocatorGroup(1);
    cache_manager_ = std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group_);
  }

  void TearDown() override {}

  // 加载模型权重
  // 加载模型权重，返回模型和权重
  template <typename weight_data_type>
  std::pair<std::shared_ptr<LlamaModel>, std::shared_ptr<BaseWeight>> LoadModel() {
    int device_id = 0;
    SetDevice(device_id);
    std::filesystem::path model_path(model_config.path);
    if (!std::filesystem::exists(model_path)) {
      KLLM_LOG_ERROR << fmt::format("The given model path {} does not exist.", model_config.path);
      return {nullptr, nullptr};
    }

    std::shared_ptr<BaseWeight> llama_weight =
        std::make_shared<LlamaWeight<weight_data_type>>(model_config, runtime_config, 0, context_);
    // Start Loader Weight
    ModelFileFormat model_file_format;
    std::vector<std::string> weights_file_list = SearchLocalPath(model_path, model_file_format);
    bool load_bias = true;
    for (std::string& file_name : weights_file_list) {
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
    llama->AllocResources(multi_batch_id);

    return {llama, llama_weight};
  }

  // 测试 logits GATHER_ALL 功能
  template <typename weight_data_type>
  void TestLogitsGatherAll() {
    std::cout << "Testing logits GATHER_ALL functionality..." << std::endl;

    // 获取模型和权重
    auto [llama, llama_weight] = LoadModel<weight_data_type>();
    if (!llama) {
      std::cout << "Failed to load model, skipping test." << std::endl;
      return;
    }

    // 测试用例1: 使用 GATHER_ALL 模式获取所有 token 的 logits
    std::cout << "Test case 1: GATHER_ALL with slice_pos {{0, 1}}" << std::endl;

    // 创建 ForwardRequest
    ForwardRequest forward;
    std::vector<int> input_ids = {1, 529};  // 示例输入 token
    ForwardRequestBuilderForTest request_builder(model_config, runtime_config, cache_manager_);
    auto forward = request_builder.CreateForwardRequest(1, input_ids);

    // 设置 request_target 参数，使用 GATHER_ALL 模式
    std::map<std::string, ksana_llm::TargetDescribe> request_target;
    ksana_llm::TargetDescribe target_describe;
    target_describe.slice_pos.push_back({0, 1});  // 获取前两个 token 的 logits
    target_describe.token_reduce_mode = GetTokenReduceMode("GATHER_ALL");
    request_target["logits"] = target_describe;
    forward.request_target = std::make_shared<const std::map<std::string, TargetDescribe>>(request_target);

    // 初始化 response 成员变量
    std::map<std::string, PythonTensor> response_map;
    forward.response = &response_map;

    // 执行 Forward
    std::vector<ForwardRequest> forward_reqs = {forward};
    Status status = llama->Forward(multi_batch_id, llama_weight, forward_reqs, false);
    EXPECT_TRUE(status.OK()) << "Forward failed: " << status.GetMessage();

    // 输出 ForwardRequest 的 response
    std::cout << "Checking ForwardRequest response after Forward..." << std::endl;

    // 验证响应
    EXPECT_EQ(forward.response->size(), 1ul) << "Expected exactly one response entry";

    // 检查是否包含 logits 数据
    EXPECT_TRUE(forward.response->find("logits") != forward.response->end()) << "Response should contain logits data";

    const auto& logits_tensor = forward.response->at("logits");

    // 验证返回张量的形状
    EXPECT_EQ(logits_tensor.shape.size(), 2ul) << "Logits tensor should have 2 dimensions";
    EXPECT_EQ(logits_tensor.shape[0], 2) << "First dimension should be 2 (requested slice_pos {0, 1})";
    EXPECT_GT(logits_tensor.shape[1], 0) << "Second dimension (vocab_size) should be greater than 0";

    std::cout << "  Key: logits, Tensor shape: [";
    for (size_t i = 0; i < logits_tensor.shape.size(); ++i) {
      std::cout << logits_tensor.shape[i];
      if (i < logits_tensor.shape.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "], dtype: " << logits_tensor.dtype << std::endl;

    // 输出并验证 logits tensor 的值
    if (logits_tensor.dtype == "float16") {
      const float16* data_ptr = reinterpret_cast<const float16*>(logits_tensor.data.data());
      size_t rows = logits_tensor.shape[0];
      size_t cols = logits_tensor.shape[1];

      std::cout << "  First few logits values for reference:" << std::endl;
      std::vector<float> first_few_logits;

      for (size_t i = 0; i < std::min(rows, size_t(1)); ++i) {
        std::cout << "    Row " << i << ": ";
        for (size_t j = 0; j < std::min(cols, size_t(10)); ++j) {
          float value = static_cast<float>(data_ptr[i * cols + j]);
          first_few_logits.push_back(value);
          std::cout << value << ", ";
        }
        std::cout << std::endl;
      }

#ifdef ENABLE_CUDA
      std::cout << "checking cuda result..." << std::endl;
      // 预期的 logits 值（这些值应该是通过先运行一次测试获取的）
      // 在第一次运行时，可以注释掉下面的比较代码，然后使用上面打印的值来填充这个数组
      std::vector<float> expected_logits = {
          // 这里填入预期的 logits 值，例如：
          -11.3281, -18.7812, -1.0293,  -6.80078, -3.45312,
          -5.90625, -4.62109, -5.72656, -5.08594, -4.875  // 根据之前的运行结果填写
      };

      // 检查前几个 logits 值是否符合预期 - 使用相对误差比较
      float relative_error_threshold = 1e-2;
      if (!expected_logits.empty()) {
        for (size_t i = 0; i < std::min(expected_logits.size(), first_few_logits.size()); ++i) {
          EXPECT_TRUE(ExpectNearRelative(expected_logits[i], first_few_logits[i], relative_error_threshold))
              << "Values differ at index " << i << ": expected " << expected_logits[i] << ", actual "
              << first_few_logits[i];
        }
      }
#endif
    }

    // 清理资源
    if (forward.req_id > 0) {
      // 清理缓存管理器中的请求
      cache_manager_->DestroyFinishedRequest(forward.req_id);
    }

    // 释放所有可重用的缓存块
    size_t freed_blocks = 0;
    cache_manager_->FreeCachedBlocks(std::numeric_limits<size_t>::max(), freed_blocks);
    if (freed_blocks > 0) {
      std::cout << "Freed " << freed_blocks << " cached blocks" << std::endl;
    }

    llama.reset();
    DeviceSynchronize();
  }

  template <typename weight_data_type>
  void TestLayernormGatherAll() {
    std::cout << "Testing layernorm GATHER_ALL functionality..." << std::endl;

    // 获取模型和权重
    auto [llama, llama_weight] = LoadModel<weight_data_type>();
    if (!llama) {
      std::cout << "Failed to load model, skipping test." << std::endl;
      return;
    }

    // 测试用例1: 使用 GATHER_ALL 模式获取所有 token 的 logits
    std::cout << "Test case 1: GATHER_ALL with slice_pos {{0, 1}}" << std::endl;

    // 创建 ForwardRequest
    ForwardRequest forward;
    std::vector<int> input_ids = {1, 529};  // 示例输入 token
    ForwardRequestBuilderForTest request_builder(model_config, runtime_config, cache_manager_);
    request_builder.CreateForwardRequest(1, forward, input_ids);

    // 设置 request_target 参数，使用 GATHER_ALL 模式
    std::map<std::string, ksana_llm::TargetDescribe> request_target;
    ksana_llm::TargetDescribe target_describe;
    target_describe.slice_pos.push_back({0, 1});  // 获取前两个 token 的 layernorm
    target_describe.token_reduce_mode = GetTokenReduceMode("GATHER_ALL");
    request_target["layernorm"] = target_describe;
    forward.request_target = std::make_shared<const std::map<std::string, TargetDescribe>>(request_target);

    // 初始化 response 成员变量
    std::map<std::string, PythonTensor> response_map;
    forward.response = &response_map;

    // 执行 Forward
    std::vector<ForwardRequest> forward_reqs = {forward};
    Status status = llama->Forward(multi_batch_id, llama_weight, forward_reqs, false);
    EXPECT_TRUE(status.OK()) << "Forward failed: " << status.GetMessage();

    // 输出 ForwardRequest 的 response
    std::cout << "Checking ForwardRequest response after Forward..." << std::endl;

    // 验证响应
    EXPECT_EQ(forward.response->size(), 1ul) << "Expected exactly one response entry";

    // 检查是否包含 layernorm 数据
    EXPECT_TRUE(forward.response->find("layernorm") != forward.response->end())
        << "Response should contain layernorm data";

    const auto& layernorm_tensor = forward.response->at("layernorm");

    // 验证返回张量的形状
    EXPECT_EQ(layernorm_tensor.shape.size(), 2ul) << "Layernorm tensor should have 2 dimensions";
    EXPECT_EQ(layernorm_tensor.shape[0], 2) << "First dimension should be 2 (requested slice_pos {0, 1})";
    EXPECT_GT(layernorm_tensor.shape[1], 0) << "Second dimension (vocab_size) should be greater than 0";

    std::cout << "  Key: Layernorm, Tensor shape: [";
    for (size_t i = 0; i < layernorm_tensor.shape.size(); ++i) {
      std::cout << layernorm_tensor.shape[i];
      if (i < layernorm_tensor.shape.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "], dtype: " << layernorm_tensor.dtype << std::endl;

    // 输出并验证 layernorm tensor 的值
    if (layernorm_tensor.dtype == "float16") {
      const float16* data_ptr = reinterpret_cast<const float16*>(layernorm_tensor.data.data());
      size_t rows = layernorm_tensor.shape[0];
      size_t cols = layernorm_tensor.shape[1];

      std::cout << "  First few layernorm values for reference:" << std::endl;
      std::vector<float> first_few_layernorm;

      for (size_t i = 0; i < std::min(rows, size_t(1)); ++i) {
        std::cout << "    Row " << i << ": ";
        for (size_t j = 0; j < std::min(cols, size_t(10)); ++j) {
          float value = static_cast<float>(data_ptr[i * cols + j]);
          first_few_layernorm.push_back(value);
          std::cout << value << ", ";
        }
        std::cout << std::endl;
      }
#ifdef ENABLE_CUDA

#endif
    }

    // 清理资源
    if (forward.req_id > 0) {
      // 清理缓存管理器中的请求
      cache_manager_->DestroyFinishedRequest(forward.req_id);
    }

    // 释放所有可重用的缓存块
    size_t freed_blocks = 0;
    cache_manager_->FreeCachedBlocks(std::numeric_limits<size_t>::max(), freed_blocks);
    if (freed_blocks > 0) {
      std::cout << "Freed " << freed_blocks << " cached blocks" << std::endl;
    }

    llama.reset();
    DeviceSynchronize();
  }

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  std::shared_ptr<Context> context_{nullptr};
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;
  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;
  std::shared_ptr<PrefixCacheManager> cache_manager_ = nullptr;
  size_t multi_batch_id = 0;
};

TEST_F(ForwardTest, LogitsGatherAllTest) {
  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_FP16 weight_data_type with logits GATHER_ALL." << std::endl;
  TestLogitsGatherAll<float16>();
}

TEST_F(ForwardTest, LayernormGatherAllTest) {
  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_FP16 weight_data_type with layernorm GATHER_ALL." << std::endl;
  TestLayernormGatherAll<float16>();
}
