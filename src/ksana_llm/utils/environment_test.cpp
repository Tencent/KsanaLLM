/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/environment.h"
#include <filesystem>
#include <string>
#include "gflags/gflags.h"
#include "test.h"

namespace ksana_llm {

DEFINE_string(config_file_test, "examples/ksana_llm.yaml", "The config file path");

class EnvironmentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    FLAGS_config_file_test = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
  }

  ksana_llm::Environment env_;
};

// 测试配置文件解析
TEST_F(EnvironmentTest, ParseConfig) {
  auto status = env_.ParseConfig(FLAGS_config_file_test);
  EXPECT_TRUE(status.OK()) << status.GetMessage();
}

// 测试获取BatchSchedulerConfig
TEST_F(EnvironmentTest, GetBatchSchedulerConfig) {
  // 先解析配置
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  // 获取配置并验证
  ksana_llm::BatchSchedulerConfig config;
  auto status = env_.GetBatchSchedulerConfig(config);
  EXPECT_TRUE(status.OK());

  // 验证关键配置参数
  EXPECT_GT(config.max_token_len, 0);
  EXPECT_GT(config.max_batch_size, 0);
  EXPECT_GT(config.max_step_token_num, 0);
  EXPECT_GT(config.waiting_timeout_in_ms, 0);
  EXPECT_GT(config.max_waiting_queue_len, 0);
}

// 测试获取CacheManagerConfig
TEST_F(EnvironmentTest, GetCacheManagerConfig) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::CacheManagerConfig config;
  auto status = env_.GetCacheManagerConfig(config);
  EXPECT_TRUE(status.OK());

  EXPECT_GT(config.swap_threadpool_size, 0);
  EXPECT_GE(config.min_flexible_cache_num, 0);
  EXPECT_GT(config.block_token_num, 0);
  EXPECT_GT(config.tensor_para_size, 0);
}

// 测试获取BlockManagerConfig
TEST_F(EnvironmentTest, GetBlockManagerConfig) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());
  ASSERT_TRUE(env_.InitializeBlockManagerConfig().OK());

  ksana_llm::BlockManagerConfig config;
  auto status = env_.GetBlockManagerConfig(config);
  EXPECT_TRUE(status.OK());

  // 验证配置参数
  EXPECT_GT(config.host_allocator_config.block_size, 0);
  EXPECT_GT(config.device_allocator_config.block_size, 0);
  EXPECT_GT(config.host_allocator_config.blocks_num, 0);
  EXPECT_GT(config.device_allocator_config.blocks_num, 0);
  EXPECT_GT(config.reserved_device_memory_ratio, 0);
}

// 测试获取ModelConfig
TEST_F(EnvironmentTest, GetModelConfig) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::ModelConfig config;
  auto status = env_.GetModelConfig(config);
  EXPECT_TRUE(status.OK());

  // 验证模型配置
  EXPECT_FALSE(config.type.empty());
  EXPECT_GT(config.vocab_size, 0);
  EXPECT_GT(config.hidden_units, 0);
  EXPECT_GT(config.num_layer, 0);
  EXPECT_GT(config.head_num, 0);
  EXPECT_GT(config.size_per_head, 0);

  // Test rope scaling factor
  EXPECT_EQ(config.rope_scaling_factor_config.type, "default");
  EXPECT_TRUE(config.rope_scaling_factor_config.mrope_section.empty());

  // 创建一个包含自定义模型配置的临时YAML文件
  auto create_model_config_yaml = [](const std::string& rope_type,
                                     const std::vector<int>& mrope_section) -> std::string {
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::string filename = "ksana_model_config_test.yaml";
    std::filesystem::path temp_file_path = temp_dir / filename;

    std::filesystem::path model_dir = temp_dir / "ksana_test_model_dir";
    try {
      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
      std::filesystem::create_directory(model_dir);

      std::ofstream file(temp_file_path);
      if (!file.is_open()) {
        return "";
      }

      // 添加包含模型配置的YAML内容
      file << "setting:\n";
      file << "  global:\n";
      file << "    tensor_para_size: 1\n";
      file << "    pipeline_para_size: 1\n";
      file << "  batch_scheduler:\n";
      file << "    max_waiting_queue_len: 100\n";
      file << "    max_token_len: 4096\n";
      file << "    max_step_tokens: 4096\n";
      file << "    max_batch_size: 32\n";
      file << "  block_manager:\n";
      file << "    block_token_num: 16\n";
      file << "    reserved_device_memory_ratio: 0.01\n";
      file << "model_spec:\n";
      file << "  base_model:\n";
      file << "    model_dir: \"" << model_dir.string() << "\"\n";

      file.close();

      // 创建模型配置文件
      std::ofstream config_json(model_dir / "config.json");
      if (config_json.is_open()) {
        config_json << "{\n";
        config_json << "  \"architectures\": [\"LlamaForCausalLM\"],\n";
        config_json << "  \"model_type\": \"llama\",\n";
        config_json << "  \"torch_dtype\": \"float16\",\n";
        config_json << "  \"vocab_size\": 32000,\n";
        config_json << "  \"hidden_size\": 4096,\n";
        config_json << "  \"intermediate_size\": 11008,\n";
        config_json << "  \"num_attention_heads\": 32,\n";
        config_json << "  \"num_key_value_heads\": 32,\n";
        config_json << "  \"num_hidden_layers\": 32,\n";
        config_json << "  \"max_position_embeddings\": 4096,\n";
        config_json << "  \"bos_token_id\": 1,\n";
        config_json << "  \"eos_token_id\": 2,\n";
        config_json << "  \"pad_token_id\": 0,\n";
        // 自定义rope_scaling
        config_json << "  \"rope_scaling\": {\n";
        config_json << "    \"type\": \"" << rope_type << '"';
        if (!mrope_section.empty()) {
          EXPECT_EQ(mrope_section.size(), 3);
          config_json << ",\n";
          config_json << "    \"mrope_section\": [\n";
          for (int i = 0; i < 3; i++) {
            config_json << "      " << mrope_section[i];
            if (i < 2) {
              config_json << ',';
            }
            config_json << '\n';
          }
          config_json << "    ]\n";
        } else {
          config_json << '\n';
        }
        config_json << "  }\n";
        config_json << "}\n";
        config_json.close();
      }

      return temp_file_path.string();
    } catch (const std::exception& e) {
      return "";
    }
  };

  // Case 1: type = "default", mropse_section != None
  {
    std::string yaml_path = create_model_config_yaml("default", {16, 24, 24});
    ASSERT_FALSE(yaml_path.empty());

    Environment test_env;
    EXPECT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ksana_llm::ModelConfig config;
    EXPECT_TRUE(test_env.GetModelConfig(config).OK());

    EXPECT_EQ(config.rope_scaling_factor_config.type, "mrope");
    EXPECT_EQ(config.rope_scaling_factor_config.mrope_section.size(), 3);

    // 清理临时文件
    try {
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::filesystem::path model_dir = temp_dir / "ksana_test_model_dir";

      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }
      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }
}

// 测试获取ProfilerConfig
TEST_F(EnvironmentTest, GetProfilerConfig) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::ProfilerConfig config;
  auto status = env_.GetProfilerConfig(config);
  EXPECT_TRUE(status.OK());
  EXPECT_GT(config.export_interval_millis, 0);
  EXPECT_GT(config.export_timeout_millis, 0);
}

// 测试错误的配置文件路径
TEST_F(EnvironmentTest, ParseConfigError) {
  auto status = env_.ParseConfig("non_exist_config.yaml");
  EXPECT_FALSE(status.OK());
}

// 测试前缀缓存功能状态
TEST_F(EnvironmentTest, PrefixCachingStatus) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::BatchSchedulerConfig config;
  auto status = env_.GetBatchSchedulerConfig(config);
  EXPECT_TRUE(status.OK());
}

// 测试设备兼容性检查
TEST_F(EnvironmentTest, DeviceCompatibilityCheck) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::RuntimeConfig config;
  auto status = env_.GetRuntimeConfig(config);
  EXPECT_TRUE(status.OK());

  // 验证设备相关配置
  EXPECT_GT(config.parallel_basic_config.tensor_parallel_size, 0);
}

// 测试配置参数边界条件
TEST_F(EnvironmentTest, ConfigBoundaryConditions) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::BatchSchedulerConfig batch_config;
  auto status = env_.GetBatchSchedulerConfig(batch_config);
  EXPECT_TRUE(status.OK());

  // 验证参数边界
  EXPECT_GT(batch_config.max_token_len, 0);
  EXPECT_LE(batch_config.max_token_len, 32768);  // 合理的上限

  EXPECT_GT(batch_config.max_batch_size, 0);
  EXPECT_LE(batch_config.max_batch_size, 1024);  // 合理的上限

  EXPECT_GT(batch_config.waiting_timeout_in_ms, 0);
  EXPECT_GT(batch_config.max_waiting_queue_len, 0);
}
// 测试配置项依赖关系
TEST_F(EnvironmentTest, ConfigDependencies) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::ModelConfig model_config;
  auto status = env_.GetModelConfig(model_config);
  EXPECT_TRUE(status.OK());

  // 验证量化配置依赖
  if (model_config.quant_config.method == ksana_llm::QUANT_GPTQ) {
    EXPECT_GT(model_config.quant_config.bits, 0);
    EXPECT_GT(model_config.quant_config.group_size, 0);
  }
}

// 测试配置文件完整性
TEST_F(EnvironmentTest, ConfigFileIntegrity) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  // 验证必需的配置部分
  ksana_llm::BatchSchedulerConfig batch_config;
  EXPECT_TRUE(env_.GetBatchSchedulerConfig(batch_config).OK());

  ksana_llm::BlockManagerConfig block_config;
  EXPECT_TRUE(env_.GetBlockManagerConfig(block_config).OK());

  ksana_llm::ProfilerConfig profiler_config;
  EXPECT_TRUE(env_.GetProfilerConfig(profiler_config).OK());
}

// 测试必需参数验证
TEST_F(EnvironmentTest, RequiredParametersValidation) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::ModelConfig model_config;
  auto status = env_.GetModelConfig(model_config);
  EXPECT_TRUE(status.OK());
  RuntimeConfig runtime_config;
  env_.GetRuntimeConfig(runtime_config);

  // 验证必需的模型参数
  EXPECT_FALSE(model_config.type.empty());
  EXPECT_GT(model_config.vocab_size, 0);
  EXPECT_GT(model_config.hidden_units, 0);
  EXPECT_GT(model_config.num_layer, 0);
  EXPECT_GT(model_config.head_num, 0);
  EXPECT_GT(model_config.size_per_head, 0);
  EXPECT_GT(runtime_config.max_seq_len, 0);
}

// 测试性能相关配置
TEST_F(EnvironmentTest, PerformanceConfig) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  // 验证线程池配置
  ksana_llm::CacheManagerConfig cache_config;
  auto status = env_.GetCacheManagerConfig(cache_config);
  EXPECT_TRUE(status.OK());
  EXPECT_GT(cache_config.swap_threadpool_size, 0);

  // 验证批处理配置
  ksana_llm::BatchSchedulerConfig batch_config;
  status = env_.GetBatchSchedulerConfig(batch_config);
  EXPECT_TRUE(status.OK());
  EXPECT_GT(batch_config.max_batch_size, 0);
  EXPECT_GT(batch_config.max_step_token_num, 0);
}

// 测试内存策略配置
TEST_F(EnvironmentTest, MemoryStrategyConfig) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::BlockManagerConfig block_config;
  auto status = env_.GetBlockManagerConfig(block_config);
  EXPECT_TRUE(status.OK());

  // 验证内存分配策略
  EXPECT_EQ(block_config.host_allocator_config.block_size, 0);
  EXPECT_EQ(block_config.device_allocator_config.block_size, 0);
  EXPECT_EQ(block_config.host_allocator_config.blocks_num, 0);
  EXPECT_EQ(block_config.device_allocator_config.blocks_num, 0);

  // 验证内存比例配置
  EXPECT_GT(block_config.reserved_device_memory_ratio, 0.0f);
  EXPECT_LT(block_config.reserved_device_memory_ratio, 1.0f);
}

// 测试连接器配置初始化
TEST_F(EnvironmentTest, ConnectorConfigInitialization) {
  // 辅助函数，创建一个带有connector配置和基本模型配置的临时YAML文件
  auto create_temp_yaml = [](const std::string& role, const std::string& comm_type = "") -> std::string {
    // 使用std::filesystem的临时目录路径，而不是硬编码路径
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::string filename = "ksana_connector_test_" + role + ".yaml";
    std::filesystem::path temp_file_path = temp_dir / filename;

    // 使用绝对路径创建模型目录
    std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);
    try {
      // 确保模型目录存在
      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
      std::filesystem::create_directory(model_dir);

      std::ofstream file(temp_file_path);
      if (!file.is_open()) {
        std::cerr << "Failed to open file: " << temp_file_path.string() << std::endl;
        return "";
      }

      // 添加基本配置，包括必要的模型规格，这样即使不加载外部模型配置文件也能解析
      file << "setting:\n";
      file << "  connector:\n";
      file << "    group_role: " << role << "\n";
      if (role != "none" && role != "invalid_role") {
        file << "    router_addr: \"localhost:9090\"\n";
        file << "    inference_addr: \"localhost:9091\"\n";
        file << "    cluster_name: \"default_cluster\"\n";
        file << "    heartbeat_interval_ms: 2000\n";
        if (!comm_type.empty()) {
          file << "    communication_type: " << comm_type << "\n";
        }
      }
      file << "  global:\n";
      file << "    tensor_para_size: 1\n";
      file << "    pipeline_para_size: 1\n";

      // 添加模型规格部分，这样即使不加载外部模型配置也能初始化
      file << "  batch_scheduler:\n";
      file << "    max_waiting_queue_len: 100\n";
      file << "    max_token_len: 4096\n";
      file << "    max_step_tokens: 4096\n";
      file << "    max_batch_size: 32\n";
      file << "    enable_auto_prefix_cache: true\n";
      file << "    split_fuse_token_num: 256\n";

      file << "  block_manager:\n";
      file << "    block_token_num: 16\n";
      file << "    reserved_device_memory_ratio: 0.01\n";

      file << "model_spec:\n";
      file << "  base_model:\n";
      file << "    model_dir: \"" << model_dir.string() << "\"\n";

      file.close();

      // 创建模型配置目录和config.json文件
      std::ofstream config_json(model_dir / "config.json");
      if (config_json.is_open()) {
        // 创建一个最小化的配置JSON
        config_json << "{\n";
        config_json << "  \"architectures\": [\"LlamaForCausalLM\"],\n";
        config_json << "  \"model_type\": \"llama\",\n";
        config_json << "  \"torch_dtype\": \"float16\",\n";
        config_json << "  \"vocab_size\": 32000,\n";
        config_json << "  \"hidden_size\": 4096,\n";
        config_json << "  \"intermediate_size\": 11008,\n";
        config_json << "  \"num_attention_heads\": 32,\n";
        config_json << "  \"num_key_value_heads\": 32,\n";
        config_json << "  \"num_hidden_layers\": 32,\n";
        config_json << "  \"max_position_embeddings\": 4096,\n";
        config_json << "  \"bos_token_id\": 1,\n";
        config_json << "  \"eos_token_id\": 2,\n";
        config_json << "  \"pad_token_id\": 0\n";
        config_json << "}\n";
        config_json.close();
      } else {
        std::cerr << "Failed to create config.json in: " << (model_dir / "config.json").string() << std::endl;
      }

      return temp_file_path.string();
    } catch (const std::exception& e) {
      std::cerr << "Exception: " << e.what() << std::endl;
      return "";
    }
  };

  // 测试前准备 - 存储原始配置文件路径，以便测试后恢复
  std::string orig_config_file = FLAGS_config_file_test;

  // 测试1：group_role = "none" 的情况
  {
    std::string yaml_path = create_temp_yaml("none");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    auto parse_status = test_env.ParseConfig(yaml_path);
    if (!parse_status.OK()) {
      std::cerr << "Failed to parse config: " << parse_status.GetMessage() << std::endl;
      ASSERT_TRUE(false);
    }

    ConnectorConfig connector_config;
    auto status = test_env.GetConnectorConfigs(connector_config);
    EXPECT_FALSE(status.OK());
    EXPECT_EQ(status.GetCode(), ksana_llm::RET_CONFIG_NOT_FOUND);

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "none";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试2：group_role = "prefill" 的情况
  {
    std::string yaml_path = create_temp_yaml("prefill");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ConnectorConfig connector_config;
    auto status = test_env.GetConnectorConfigs(connector_config);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(connector_config.group_role, ksana_llm::GroupRole::PREFILL);
    EXPECT_EQ(connector_config.router_addr, "http://localhost:9090");
    EXPECT_EQ(connector_config.inference_addr, "localhost:9091");
    EXPECT_EQ(connector_config.cluster_name, "default_cluster");
    EXPECT_EQ(connector_config.heartbeat_interval_ms, 2000);
    EXPECT_EQ(connector_config.communication_type, ksana_llm::CommunicationType::TCP);  // 默认值

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "prefill";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试3：group_role = "decode" 的情况
  {
    std::string yaml_path = create_temp_yaml("decode");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ConnectorConfig connector_config;
    auto status = test_env.GetConnectorConfigs(connector_config);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(connector_config.group_role, ksana_llm::GroupRole::DECODE);
    EXPECT_EQ(connector_config.communication_type, ksana_llm::CommunicationType::TCP);  // 默认值

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "decode";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试4：group_role = "both" 的情况
  {
    std::string yaml_path = create_temp_yaml("both");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ConnectorConfig connector_config;
    auto status = test_env.GetConnectorConfigs(connector_config);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(connector_config.group_role, ksana_llm::GroupRole::BOTH);
    EXPECT_EQ(connector_config.communication_type, ksana_llm::CommunicationType::TCP);  // 默认值

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "both";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试5：无效的 group_role 值
  {
    std::string yaml_path = create_temp_yaml("invalid_role");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ConnectorConfig connector_config;
    auto status = test_env.GetConnectorConfigs(connector_config);
    EXPECT_FALSE(status.OK());  // 无效的角色会被设置为NONE，所以获取配置应该失败
    EXPECT_EQ(status.GetCode(), ksana_llm::RET_CONFIG_NOT_FOUND);

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "invalid_role";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试6：communication_type = "nccl" 的情况
  {
    std::string yaml_path = create_temp_yaml("prefill", "nccl");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ConnectorConfig connector_config;
    auto status = test_env.GetConnectorConfigs(connector_config);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(connector_config.communication_type, ksana_llm::CommunicationType::NCCL);

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "prefill";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试7：其他 communication_type 值 (应默认为TCP)
  {
    std::string yaml_path = create_temp_yaml("prefill", "other_type");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ConnectorConfig connector_config;
    auto status = test_env.GetConnectorConfigs(connector_config);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(connector_config.communication_type, ksana_llm::CommunicationType::TCP);  // 默认为TCP

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "prefill";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试8：测试大小写不敏感
  {
    std::string yaml_path = create_temp_yaml("PREFILL", "NCCL");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ConnectorConfig connector_config;
    auto status = test_env.GetConnectorConfigs(connector_config);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(connector_config.group_role, ksana_llm::GroupRole::PREFILL);               // 应能识别大写
    EXPECT_EQ(connector_config.communication_type, ksana_llm::CommunicationType::NCCL);  // 应能识别大写

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "PREFILL";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试9：group_role = "prefill" 但配置了 $DECODE_NODE_BENCHMARK 的情况
  {
    // 设置环境变量
    if (setenv("DECODE_NODE_BENCHMARK", "1", 1) != 0) {
      std::cerr << "Error setting environment variable" << std::endl;
    }

    std::string yaml_path = create_temp_yaml("prefill");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ConnectorConfig connector_config;
    auto status = test_env.GetConnectorConfigs(connector_config);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(connector_config.group_role, ksana_llm::GroupRole::DECODE);

    // 取消环境变量
    if (unsetenv("DECODE_NODE_BENCHMARK") != 0) {
      std::cerr << "Error unsetting environment variable" << std::endl;
    }

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "decode";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试10：communication_type = "ZMQ" 的情况
  {
    std::string yaml_path = create_temp_yaml("prefill", "ZMQ");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ConnectorConfig connector_config;
    auto status = test_env.GetConnectorConfigs(connector_config);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(connector_config.communication_type, ksana_llm::CommunicationType::ZMQ);

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "prefill";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试11：自动解决split fuse参数冲突的情况。支持split fuse后移除此单测
  {
    std::string yaml_path = create_temp_yaml("prefill");
    ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

    // 设置测试配置文件路径
    FLAGS_config_file_test = yaml_path;

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    ConnectorConfig connector_config;
    EXPECT_TRUE(test_env.GetConnectorConfigs(connector_config).OK());
    BatchSchedulerConfig batch_scheduluer_config;
    EXPECT_TRUE(test_env.GetBatchSchedulerConfig(batch_scheduluer_config).OK());
    EXPECT_EQ(batch_scheduluer_config.split_fuse_token_num, 0);

    try {
      // 获取YAML文件的基本路径信息
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::string role = "prefill";
      std::filesystem::path model_dir = temp_dir / ("ksana_test_model_dir_" + role);

      // 清理临时文件和目录
      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }

      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }
  // 恢复原始配置文件路径
  FLAGS_config_file_test = orig_config_file;
}

// 测试参数边界条件
TEST_F(EnvironmentTest, ParameterBoundaries) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::ModelConfig model_config;
  auto status = env_.GetModelConfig(model_config);
  EXPECT_TRUE(status.OK());

  // 验证参数范围
  EXPECT_GT(model_config.vocab_size, 0);
  EXPECT_LT(model_config.vocab_size, 1000000);  // 合理的词表大小上限

  EXPECT_GT(model_config.hidden_units, 0);
  EXPECT_LT(model_config.hidden_units, 100000);  // 合理的隐藏层大小上限

  EXPECT_GT(model_config.num_layer, 0);
  EXPECT_LT(model_config.num_layer, 1000);  // 合理的层数上限
}

// 测试无效配置组合
TEST_F(EnvironmentTest, InvalidConfigCombinations) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::ModelConfig model_config;
  auto status = env_.GetModelConfig(model_config);
  EXPECT_TRUE(status.OK());

  // 验证量化配置组合
  if (model_config.quant_config.method == ksana_llm::QUANT_GPTQ) {
    EXPECT_NE(model_config.quant_config.bits, 0);
    EXPECT_NE(model_config.quant_config.group_size, 0);
  }
}

// 测试特殊字符处理
TEST_F(EnvironmentTest, SpecialCharacterHandling) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::ModelConfig model_config;
  auto status = env_.GetModelConfig(model_config);
  EXPECT_TRUE(status.OK());
}

// 测试运行时配置
TEST_F(EnvironmentTest, RuntimeConfig) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::BatchSchedulerConfig batch_config;
  auto status = env_.GetBatchSchedulerConfig(batch_config);
  EXPECT_TRUE(status.OK());
}

// 测试Expert-Parallel配置读取
TEST_F(EnvironmentTest, InitExpertParaConfig) {
  setenv("EXPERT_NODE_RANK", "5", 1);
  setenv("EXPERT_MASTER_HOST", "localhost", 1);
  setenv("EXPERT_MASTER_PORT", "12345", 1);
  ExpertParallelConfig expert_parallel_config;
  env_.GetExpertParallelConfig(expert_parallel_config);
  expert_parallel_config.expert_world_size = 8;
  env_.SetExpertParallelConfig(expert_parallel_config);

  env_.SetTensorParallelSize(12);
  env_.SetExpertParallelSize(6);
  env_.InitializeExpertParallelConfig();
  env_.GetExpertParallelConfig(expert_parallel_config);

  EXPECT_EQ(expert_parallel_config.expert_para_size, 6);
  EXPECT_EQ(expert_parallel_config.expert_tensor_para_size, 2);
  EXPECT_EQ(expert_parallel_config.expert_world_size, 8);
  EXPECT_EQ(expert_parallel_config.expert_node_rank, 5);
  EXPECT_EQ(expert_parallel_config.expert_master_host, "localhost");
  EXPECT_EQ(expert_parallel_config.expert_master_port, 12345);

  // 验证计算的值
  EXPECT_EQ(expert_parallel_config.global_expert_para_size,
            expert_parallel_config.expert_world_size * expert_parallel_config.expert_para_size);

  unsetenv("EXPERT_MASTER_HOST");
  unsetenv("EXPERT_MASTER_PORT");
  unsetenv("EXPERT_WORLD_SIZE");
  unsetenv("EXPERT_NODE_RANK");
}

// 测试Environment的InitializeKVCacheConfigs方法
TEST_F(EnvironmentTest, InitializeKVCacheConfigs) {
  // 创建测试用的PipelineConfig
  PipelineConfig pipeline_config;
  pipeline_config.lower_layer_idx = 0;
  pipeline_config.upper_layer_idx = 99;
  pipeline_config.lower_nextn_layer_idx = 100;
  pipeline_config.upper_nextn_layer_idx = 101;

  auto& attn_backend_config = env_.schedule_config_parser_.runtime_config_.attn_backend_config;
  auto& block_manager_config = env_.schedule_config_parser_.block_manager_config_;
  auto& batch_scheduler_config = env_.schedule_config_parser_.batch_scheduler_config_;
  // 设置每个block包含的token数量
  attn_backend_config.block_token_num = 16;
  block_manager_config.device_allocator_config.block_token_num = 16;

  // 创建测试用的ModelConfig
  ModelConfig model_config;
  model_config.num_layer = pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1;
  model_config.weight_data_type = DataType::TYPE_FP16;

  // 约束 max_step_tokens
  batch_scheduler_config.launch_block_threshold = 2;
  batch_scheduler_config.max_step_token_num = 1e9;  // 极大值

  // Case 1: Models using MLA with auto KV cache
  model_config.type = "deepseek_v3";
  attn_backend_config.kv_cache_dtype_str = "auto";
  attn_backend_config.block_token_num = 16;
  model_config.use_mla = true;
  model_config.mla_config.kv_lora_rank = 32;
  model_config.mla_config.qk_rope_head_dim = 64;
  // Call InitializeKVCacheConfigs
  env_.schedule_config_parser_.InitializeKVCacheConfigs(model_config, pipeline_config);
  // Check Result
  env_.GetAttnBackendConfig(attn_backend_config);
  EXPECT_EQ(attn_backend_config.kv_cache_dtype, model_config.weight_data_type);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.size(), 1);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").block_bytes, 313344);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").k_offset, 0);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").v_offset, 0);
  EXPECT_EQ(attn_backend_config.block_size, 313344);
  block_manager_config.device_allocator_config.block_size = attn_backend_config.block_size;
  block_manager_config.host_allocator_config.block_size = attn_backend_config.block_size;
  env_.CalculateBlockNumber();
  // Expect at least one block
  EXPECT_GT(env_.schedule_config_parser_.block_manager_config_.device_allocator_config.blocks_num, 0);
  // 由于 block 不足, max_step_token_num 也应对应下降
  EXPECT_LT(batch_scheduler_config.max_step_token_num, 1e9);
  if (env_.schedule_config_parser_.block_manager_config_.device_allocator_config.blocks_num >
      batch_scheduler_config.launch_block_threshold) {
    size_t usable_block_num = env_.schedule_config_parser_.block_manager_config_.device_allocator_config.blocks_num -
                              batch_scheduler_config.launch_block_threshold;
    EXPECT_EQ(batch_scheduler_config.max_step_token_num,
              usable_block_num * block_manager_config.device_allocator_config.block_token_num);
  }

  // Case 2: Models using MLA with fp8_e4m3 KV cache
  model_config.type = "deepseek_v3";
  attn_backend_config.kv_cache_dtype_str = "fp8_e4m3";
  // Call InitializeKVCacheConfigs
  env_.schedule_config_parser_.InitializeKVCacheConfigs(model_config, pipeline_config);
  // Check Result
  env_.GetAttnBackendConfig(attn_backend_config);
  EXPECT_EQ(attn_backend_config.kv_cache_dtype, TYPE_FP8_E4M3);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.size(), 1);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").block_bytes, 156672);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").k_offset, 0);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").v_offset, 0);
  EXPECT_EQ(attn_backend_config.block_size, 156672);
  block_manager_config.device_allocator_config.block_size = attn_backend_config.block_size;
  block_manager_config.host_allocator_config.block_size = attn_backend_config.block_size;
  env_.CalculateBlockNumber();
  // Expect at least one block
  EXPECT_GT(env_.schedule_config_parser_.block_manager_config_.device_allocator_config.blocks_num, 0);

  // Case 3: Models using DeepSeek Sparse MLA
  model_config.type = "deepseea_v32";
  attn_backend_config.kv_cache_dtype_str = "fp8_ds_mla";
  model_config.use_dsa = true;
  model_config.mla_config.kv_lora_rank = 128;
  model_config.dsa_config.index_head_dim = 128;
  // Call InitializeKVCacheConfigs
  env_.schedule_config_parser_.InitializeKVCacheConfigs(model_config, pipeline_config);
  // Check Result
  env_.GetAttnBackendConfig(attn_backend_config);
  EXPECT_EQ(attn_backend_config.kv_cache_dtype, TYPE_FP8_DS_MLA);
  // There are two kv caches
  EXPECT_EQ(attn_backend_config.kv_cache_configs.size(), 2);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("indexer").block_bytes, 215424);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("indexer").k_offset, 0);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("indexer").v_offset, 2048);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").block_bytes, 424320);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").k_offset, 2112);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").v_offset, 2112);
  EXPECT_EQ(attn_backend_config.block_size, 639744);  // 215424 + 424320
  block_manager_config.device_allocator_config.block_size = attn_backend_config.block_size;
  block_manager_config.host_allocator_config.block_size = attn_backend_config.block_size;
  env_.CalculateBlockNumber();
  // Expect at least one block
  EXPECT_GT(env_.schedule_config_parser_.block_manager_config_.device_allocator_config.blocks_num, 0);

  // Case 4: Common models
  model_config.type = "qwen";
  attn_backend_config.kv_cache_dtype_str = "auto";
  model_config.use_mla = false;
  model_config.use_dsa = false;
  model_config.size_per_head = 128;
  // local_num_key_value_heads = 8
  model_config.num_key_value_heads = 16;
  env_.schedule_config_parser_.runtime_config_.parallel_basic_config.attn_tensor_parallel_size = 2;
  // Call InitializeKVCacheConfigs
  env_.schedule_config_parser_.InitializeKVCacheConfigs(model_config, pipeline_config);
  // Check Result
  env_.GetAttnBackendConfig(attn_backend_config);
  EXPECT_EQ(attn_backend_config.kv_cache_dtype, model_config.weight_data_type);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.size(), 1);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").block_bytes, 6684672);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").k_offset, 0);
  EXPECT_EQ(attn_backend_config.kv_cache_configs.at("attention").v_offset, 32768);
  EXPECT_EQ(attn_backend_config.block_size, 6684672);
  block_manager_config.device_allocator_config.block_size = attn_backend_config.block_size;
  block_manager_config.host_allocator_config.block_size = attn_backend_config.block_size;
  env_.CalculateBlockNumber();
  // Expect at least one block
  EXPECT_GT(env_.schedule_config_parser_.block_manager_config_.device_allocator_config.blocks_num, 0);
}

// 测试Environment的ParseModelQuantConfig方法
TEST_F(EnvironmentTest, ParseModelQuantConfig) {
  // 测试混合量化的配置加载
  {
    nlohmann::json mixed_config = {
        {"quantization_config",
         {{"quant_method", "mixed"},
          {"configs",
           {{"fp8",
             {{"method", "fp8"}, {"activation_scheme", "dynamic"}, {"fmt", "e4m3"}, {"weight_block_size", {128, 128}}}},
            {"gptq",
             {{"method", "gptq"},
              {"bits", 4},
              {"damp_percent", 0.01},
              {"desc_act", false},
              {"group_size", 128},
              {"sym", true}}}}},
          {"layer_mapping",
           {{"gptq",
             {{"default_config", false},
              {"pattern_layers", {".mlp.experts."}},
              {"ignored_layers", nlohmann::json::array()}}},
            {"fp8",
             {{"default_config", true},
              {"pattern_layers", nlohmann::json::array()},
              {"ignored_layers", {"model.layers.61.eh_proj"}}}}}}}}};
    ModelConfig model_config;
    std::string yaml_weight_quant_method = "auto";
    std::string yaml_gptq_backend = "cutlass";
    EnvModelConfigParser model_config_parser("", "");
    model_config_parser.ParseModelQuantConfig(mixed_config, model_config, yaml_weight_quant_method, yaml_gptq_backend);
    EXPECT_TRUE(model_config.is_quant);
    EXPECT_TRUE(model_config.quant_config.is_checkpoint_fp8_serialized);
    EXPECT_TRUE(!model_config.quant_config.is_activation_scheme_static);
    EXPECT_TRUE(model_config.quant_config.is_fp8_blockwise);
    EXPECT_EQ(model_config.quant_config.method, QUANT_BLOCK_FP8_E4M3);
    EXPECT_EQ(model_config.quant_config.weight_block_size.size(), 2);
    EXPECT_EQ(model_config.quant_config.weight_block_size[0], 128);
    EXPECT_EQ(model_config.quant_config.weight_block_size[1], 128);
    EXPECT_EQ(model_config.sub_quant_configs.size(), 1);
    EXPECT_EQ(model_config.sub_quant_configs[0].method, QUANT_GPTQ);
    EXPECT_EQ(model_config.sub_quant_configs[0].bits, 4);
    EXPECT_EQ(model_config.sub_quant_configs[0].group_size, 128);
    EXPECT_EQ(model_config.sub_quant_configs[0].desc_act, false);
  }

  // 测试GPTQ量化的配置加载
  {
    nlohmann::json gptq_config = {{"quantization_config",
                                   {{"quant_method", "gptq"},
                                    {"bits", 4},
                                    {"damp_percent", 0.01},
                                    {"desc_act", false},
                                    {"group_size", 128},
                                    {"sym", true}}}};
    ModelConfig model_config;
    std::string yaml_weight_quant_method = "auto";
    std::string yaml_gptq_backend = "cutlass";
    EnvModelConfigParser model_config_parser("", "");
    model_config_parser.ParseModelQuantConfig(gptq_config, model_config, yaml_weight_quant_method, yaml_gptq_backend);
    EXPECT_TRUE(model_config.is_quant);
    EXPECT_EQ(model_config.quant_config.method, QUANT_GPTQ);
    EXPECT_EQ(model_config.quant_config.bits, 4);
    EXPECT_EQ(model_config.quant_config.group_size, 128);
    EXPECT_EQ(model_config.quant_config.desc_act, false);
  }

  // 测试AWQ量化的配置加载
  {
    nlohmann::json awq_config = {{"quantization_config", {{"quant_method", "awq"}, {"bits", 4}, {"group_size", 128}}}};
    ModelConfig model_config;
    std::string yaml_weight_quant_method = "auto";
    std::string yaml_gptq_backend = "cutlass";
    EnvModelConfigParser model_config_parser("", "");
    model_config_parser.ParseModelQuantConfig(awq_config, model_config, yaml_weight_quant_method, yaml_gptq_backend);
    EXPECT_TRUE(model_config.is_quant);
    EXPECT_EQ(model_config.quant_config.method, QUANT_AWQ);
    EXPECT_EQ(model_config.quant_config.bits, 4);
    EXPECT_EQ(model_config.quant_config.group_size, 128);
  }
}

// 测试GetTransferLayerChunkSize方法
TEST_F(EnvironmentTest, GetTransferLayerChunkSize) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  // 测试默认值
  size_t chunk_size = env_.GetTransferLayerChunkSize();
  EXPECT_GT(chunk_size, 0);  // 应该大于0

  // 默认值应该是1（如果配置文件中没有设置）
  EXPECT_EQ(chunk_size, 1);
}

// 测试BatchSchedulerConfig中的transfer_layer_chunk_size配置
TEST_F(EnvironmentTest, BatchSchedulerConfigTransferLayerChunkSize) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::BatchSchedulerConfig config;
  auto status = env_.GetBatchSchedulerConfig(config);
  EXPECT_TRUE(status.OK());

  // 验证transfer_layer_chunk_size字段存在且有效
  EXPECT_GT(config.transfer_layer_chunk_size, 0);

  // 验证默认值
  EXPECT_EQ(config.transfer_layer_chunk_size, 1);
}

// 测试BatchSchedulerConfig中的max_pretransfer_batch_size配置
TEST_F(EnvironmentTest, BatchSchedulerConfigMaxPretransferBatchSize) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::BatchSchedulerConfig config;
  auto status = env_.GetBatchSchedulerConfig(config);
  EXPECT_TRUE(status.OK());

  // 验证max_pretransfer_batch_size字段存在且有效
  EXPECT_GT(config.max_pretransfer_batch_size, 0);

  // 验证默认值
  EXPECT_EQ(config.max_pretransfer_batch_size, 64);
}

// 测试自定义chunk传输配置
TEST_F(EnvironmentTest, CustomChunkTransferConfig) {
  // 创建一个包含自定义chunk传输配置的临时YAML文件
  auto create_chunk_config_yaml = []() -> std::string {
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::string filename = "ksana_chunk_config_test.yaml";
    std::filesystem::path temp_file_path = temp_dir / filename;

    std::filesystem::path model_dir = temp_dir / "ksana_test_chunk_model_dir";
    try {
      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
      std::filesystem::create_directory(model_dir);

      std::ofstream file(temp_file_path);
      if (!file.is_open()) {
        return "";
      }

      // 添加包含chunk传输配置的YAML内容
      file << "setting:\n";
      file << "  global:\n";
      file << "    tensor_para_size: 1\n";
      file << "    pipeline_para_size: 1\n";
      file << "  batch_scheduler:\n";
      file << "    max_waiting_queue_len: 100\n";
      file << "    max_token_len: 4096\n";
      file << "    max_step_tokens: 4096\n";
      file << "    max_batch_size: 32\n";
      file << "    max_pretransfer_batch_size: 128\n";  // 自定义值
      file << "    transfer_layer_chunk_size: 4\n";     // 自定义值
      file << "  block_manager:\n";
      file << "    block_token_num: 16\n";
      file << "    reserved_device_memory_ratio: 0.01\n";
      file << "model_spec:\n";
      file << "  base_model:\n";
      file << "    model_dir: \"" << model_dir.string() << "\"\n";

      file.close();

      // 创建模型配置文件
      std::ofstream config_json(model_dir / "config.json");
      if (config_json.is_open()) {
        config_json << "{\n";
        config_json << "  \"architectures\": [\"LlamaForCausalLM\"],\n";
        config_json << "  \"model_type\": \"llama\",\n";
        config_json << "  \"torch_dtype\": \"float16\",\n";
        config_json << "  \"vocab_size\": 32000,\n";
        config_json << "  \"hidden_size\": 4096,\n";
        config_json << "  \"intermediate_size\": 11008,\n";
        config_json << "  \"num_attention_heads\": 32,\n";
        config_json << "  \"num_key_value_heads\": 32,\n";
        config_json << "  \"num_hidden_layers\": 32,\n";
        config_json << "  \"max_position_embeddings\": 4096,\n";
        config_json << "  \"bos_token_id\": 1,\n";
        config_json << "  \"eos_token_id\": 2,\n";
        config_json << "  \"pad_token_id\": 0\n";
        config_json << "}\n";
        config_json.close();
      }

      return temp_file_path.string();
    } catch (const std::exception& e) {
      return "";
    }
  };

  std::string yaml_path = create_chunk_config_yaml();
  ASSERT_FALSE(yaml_path.empty()) << "Failed to create temporary YAML file";

  Environment test_env;
  ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

  // 测试自定义的chunk大小
  size_t chunk_size = test_env.GetTransferLayerChunkSize();
  EXPECT_EQ(chunk_size, 4);

  // 测试BatchSchedulerConfig中的自定义值
  ksana_llm::BatchSchedulerConfig config;
  auto status = test_env.GetBatchSchedulerConfig(config);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(config.transfer_layer_chunk_size, 4);
  EXPECT_EQ(config.max_pretransfer_batch_size, 128);

  // 清理临时文件
  try {
    std::filesystem::path yaml_file_path(yaml_path);
    std::filesystem::path temp_dir = yaml_file_path.parent_path();
    std::filesystem::path model_dir = temp_dir / "ksana_test_chunk_model_dir";

    if (std::filesystem::exists(yaml_path)) {
      std::filesystem::remove(yaml_path);
    }
    if (std::filesystem::exists(model_dir)) {
      std::filesystem::remove_all(model_dir);
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception during cleanup: " << e.what() << std::endl;
  }
}

// 测试chunk传输配置的边界值
TEST_F(EnvironmentTest, ChunkTransferConfigBoundaryValues) {
  // 测试chunk_size为1的情况（最小值）
  auto create_boundary_yaml = [](size_t chunk_size, size_t max_pretransfer_batch_size) -> std::string {
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::string filename = "ksana_boundary_test_" + std::to_string(chunk_size) + ".yaml";
    std::filesystem::path temp_file_path = temp_dir / filename;

    std::filesystem::path model_dir = temp_dir / ("ksana_test_boundary_model_dir_" + std::to_string(chunk_size));
    try {
      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
      std::filesystem::create_directory(model_dir);

      std::ofstream file(temp_file_path);
      if (!file.is_open()) {
        return "";
      }

      file << "setting:\n";
      file << "  global:\n";
      file << "    tensor_para_size: 1\n";
      file << "    pipeline_para_size: 1\n";
      file << "  batch_scheduler:\n";
      file << "    max_waiting_queue_len: 100\n";
      file << "    max_token_len: 4096\n";
      file << "    max_step_tokens: 4096\n";
      file << "    max_batch_size: 32\n";
      file << "    max_pretransfer_batch_size: " << max_pretransfer_batch_size << "\n";
      file << "    transfer_layer_chunk_size: " << chunk_size << "\n";
      file << "  block_manager:\n";
      file << "    block_token_num: 16\n";
      file << "    reserved_device_memory_ratio: 0.01\n";
      file << "model_spec:\n";
      file << "  base_model:\n";
      file << "    model_dir: \"" << model_dir.string() << "\"\n";

      file.close();

      // 创建模型配置文件
      std::ofstream config_json(model_dir / "config.json");
      if (config_json.is_open()) {
        config_json << "{\n";
        config_json << "  \"architectures\": [\"LlamaForCausalLM\"],\n";
        config_json << "  \"model_type\": \"llama\",\n";
        config_json << "  \"torch_dtype\": \"float16\",\n";
        config_json << "  \"vocab_size\": 32000,\n";
        config_json << "  \"hidden_size\": 4096,\n";
        config_json << "  \"intermediate_size\": 11008,\n";
        config_json << "  \"num_attention_heads\": 32,\n";
        config_json << "  \"num_key_value_heads\": 32,\n";
        config_json << "  \"num_hidden_layers\": 32,\n";
        config_json << "  \"max_position_embeddings\": 4096,\n";
        config_json << "  \"bos_token_id\": 1,\n";
        config_json << "  \"eos_token_id\": 2,\n";
        config_json << "  \"pad_token_id\": 0\n";
        config_json << "}\n";
        config_json.close();
      }

      return temp_file_path.string();
    } catch (const std::exception& e) {
      return "";
    }
  };

  // 测试最小值
  {
    std::string yaml_path = create_boundary_yaml(1, 1);
    ASSERT_FALSE(yaml_path.empty());

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    EXPECT_EQ(test_env.GetTransferLayerChunkSize(), 1);

    ksana_llm::BatchSchedulerConfig config;
    auto status = test_env.GetBatchSchedulerConfig(config);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(config.transfer_layer_chunk_size, 1);
    EXPECT_EQ(config.max_pretransfer_batch_size, 1);

    // 清理
    try {
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::filesystem::path model_dir = temp_dir / "ksana_test_boundary_model_dir_1";

      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }
      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }

  // 测试较大值
  {
    std::string yaml_path = create_boundary_yaml(16, 512);
    ASSERT_FALSE(yaml_path.empty());

    Environment test_env;
    ASSERT_TRUE(test_env.ParseConfig(yaml_path).OK());

    EXPECT_EQ(test_env.GetTransferLayerChunkSize(), 16);

    ksana_llm::BatchSchedulerConfig config;
    auto status = test_env.GetBatchSchedulerConfig(config);
    EXPECT_TRUE(status.OK());
    EXPECT_EQ(config.transfer_layer_chunk_size, 16);
    EXPECT_EQ(config.max_pretransfer_batch_size, 512);

    // 清理
    try {
      std::filesystem::path yaml_file_path(yaml_path);
      std::filesystem::path temp_dir = yaml_file_path.parent_path();
      std::filesystem::path model_dir = temp_dir / "ksana_test_boundary_model_dir_16";

      if (std::filesystem::exists(yaml_path)) {
        std::filesystem::remove(yaml_path);
      }
      if (std::filesystem::exists(model_dir)) {
        std::filesystem::remove_all(model_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Exception during cleanup: " << e.what() << std::endl;
    }
  }
}

// 测试chunk传输配置的合理性验证
TEST_F(EnvironmentTest, ChunkTransferConfigValidation) {
  ASSERT_TRUE(env_.ParseConfig(FLAGS_config_file_test).OK());

  ksana_llm::BatchSchedulerConfig config;
  auto status = env_.GetBatchSchedulerConfig(config);
  EXPECT_TRUE(status.OK());

  // 验证配置的合理性
  EXPECT_GT(config.transfer_layer_chunk_size, 0);
  EXPECT_LE(config.transfer_layer_chunk_size, 1024);  // 合理的上限

  EXPECT_GT(config.max_pretransfer_batch_size, 0);
  EXPECT_LE(config.max_pretransfer_batch_size, 10240);  // 合理的上限

  // 验证chunk大小不应该超过max_batch_size的某个倍数（合理性检查）
  EXPECT_LE(config.transfer_layer_chunk_size, config.max_batch_size * 10);
}

}  // namespace ksana_llm
