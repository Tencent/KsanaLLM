/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <pybind11/embed.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fmt/core.h"
#include "ksana_llm/model_loader/config_parser/model_config_parser.h"
#include "ksana_llm/model_loader/file_loader/model_file_loader.h"
#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/model_loader/weight_loader/model_weight_loader.h"

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"

#include "include/gtest/gtest.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

#include "ksana_llm/helpers/environment_test_helper.h"

using namespace ksana_llm;

class NewDeepSeekV3LoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    pybind11::initialize_interpreter();

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/ksana_llm_deepseekv2.yaml";
    // Initialize block manager.
    Singleton<Environment>::GetInstance()->ParseConfig(config_path_relate);
    RuntimeConfig runtime_config;
    Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
    parallel_basic_config_ = runtime_config.parallel_basic_config;

    // Initialize context, set tensor paralle to 2 and attn data parallel to 1.
    context_ = std::make_shared<Context>(2, 1, 1);
    KLLM_LOG_INFO << "Initialize context success." << std::endl;
  }

  void TearDown() override { pybind11::finalize_interpreter(); }

 protected:
  std::shared_ptr<Context> context_ = nullptr;
  ParallelismBasicConfig parallel_basic_config_;
};

TEST_F(NewDeepSeekV3LoaderTest, TestNewDeepSeekV3ConfigParser) {
  std::vector<std::string> model_dirs = {"/model/DeepSeek-V2-Lite-Chat-17868/", "/model/DeepSeek-R1-17832-fix-bf16/"};

  std::vector<int> head_nums = {16, 128};
  std::vector<int> hidden_units = {2048, 7168};
  std::vector<int> inter_sizes = {10944, 18432};
  std::vector<int> num_layers = {27, 4};
  std::vector<int> vocab_sizes = {102400, 129280};
  std::vector<int> q_lora_ranks = {0, 1536};
  std::vector<int> kv_lora_ranks = {512, 512};
  std::vector<int> num_key_value_heads = {16, 128};
  std::vector<int> max_position_embeddings = {163840, 163840};
  std::vector<ModelFormat> model_formats = {ModelFormat::PYTORCH_SAFETENSOR, ModelFormat::PYTORCH_SAFETENSOR};

  for (size_t i = 0; i < model_dirs.size(); ++i) {
    const std::string& model_dir = model_dirs[i];
    if (!std::filesystem::is_directory(model_dir)) {
      KLLM_LOG_WARNING << "Model dir " << model_dir << " not exsit, skip." << std::endl;
      EXPECT_TRUE(false);
    }

    ModelConfigParser model_config_parser;

    std::shared_ptr<BaseModelConfig> model_config;
    Status status = model_config_parser.ParseModelConfig(model_dir, parallel_basic_config_, model_config);
    EXPECT_TRUE(status.OK());
    EXPECT_TRUE(model_config);

    // Check real type.
    EXPECT_TRUE(typeid(NewDeepSeekV3Config) == typeid(*model_config));

    // check config type.
    std::shared_ptr<NewDeepSeekV3Config> new_deepseek_v3_config =
        std::dynamic_pointer_cast<NewDeepSeekV3Config>(model_config);

    // Check common info.
    EXPECT_EQ(new_deepseek_v3_config->model_dir, model_dir);
    EXPECT_EQ(new_deepseek_v3_config->model_format, model_formats[i]);
    EXPECT_EQ(new_deepseek_v3_config->model_arch, ModelArchitecture::ARCH_DEEPSEEK);

    // Check value.
    EXPECT_EQ(new_deepseek_v3_config->head_num, head_nums[i]);
    EXPECT_EQ(new_deepseek_v3_config->hidden_units, hidden_units[i]);
    EXPECT_EQ(new_deepseek_v3_config->inter_size, inter_sizes[i]);
    EXPECT_EQ(new_deepseek_v3_config->num_layer, num_layers[i]);
    EXPECT_EQ(new_deepseek_v3_config->vocab_size, vocab_sizes[i]);
    EXPECT_EQ(new_deepseek_v3_config->mla_config.q_lora_rank, q_lora_ranks[i]);
    EXPECT_EQ(new_deepseek_v3_config->mla_config.kv_lora_rank, kv_lora_ranks[i]);
    EXPECT_EQ(new_deepseek_v3_config->num_key_value_heads, num_key_value_heads[i]);
    EXPECT_EQ(new_deepseek_v3_config->max_position_embeddings, max_position_embeddings[i]);
  }
}

TEST_F(NewDeepSeekV3LoaderTest, TestNewDeepSeekV3WeightLoader) {
  std::vector<std::string> model_dirs = {"/model/DeepSeek-R1-17832-fix-bf16/"};

  for (size_t i = 0; i < model_dirs.size(); ++i) {
    const std::string& model_dir = model_dirs[i];
    if (!std::filesystem::is_directory(model_dir)) {
      KLLM_LOG_WARNING << "Model dir " << model_dir << " not exist, skip." << std::endl;
      EXPECT_TRUE(false);
    }

    ModelWeightLoader weight_loader(Singleton<Environment>::GetInstance(), context_);
    std::vector<std::shared_ptr<ModelWeight>> dev_weights;

    // Get config first.
    ModelConfigParser model_config_parser;

    std::shared_ptr<BaseModelConfig> model_config;
    Status status = model_config_parser.ParseModelConfig(model_dir, parallel_basic_config_, model_config);
    EXPECT_TRUE(status.OK());
    EXPECT_TRUE(model_config);

    status = weight_loader.LoadWeights(model_config, dev_weights);
    EXPECT_TRUE(status.OK());
    EXPECT_TRUE(dev_weights.size() > 0);

    // Write every tensor to file.
    std::filesystem::path dump_path = std::filesystem::current_path();
    dump_path /= std::filesystem::path(Str2Vector(model_dir, "/").back()).filename();

    // Remove dump path and create it again.
    std::filesystem::remove_all(dump_path);
    std::filesystem::create_directories(dump_path);

    for (size_t rank = 0; rank < dev_weights.size(); ++rank) {
      SetDevice(rank);

      std::filesystem::path rank_dump_path = dump_path;
      rank_dump_path /= std::to_string(rank);
      std::filesystem::create_directories(rank_dump_path);

      for (const std::string& weight_name : dev_weights[rank]->GetWeightNames()) {
        Tensor weight_tensor = dev_weights[rank]->GetWeightTensor(weight_name);

        std::filesystem::path tensor_data_file = rank_dump_path;
        tensor_data_file /= weight_name + ".npy";
        weight_tensor.SaveToNpyFile(tensor_data_file);
      }
    }

    std::string command = fmt::format(
        "python {}/../src/ksana_llm/model_loader/"
        "check_deepseek_weight_tensor.py {} {}",
        std::filesystem::current_path().string(), model_dir, dev_weights.size());
    // check ret code
    int ret_code = std::system(command.c_str());
    EXPECT_TRUE(WEXITSTATUS(ret_code) == 0);

    // check succ file
    std::filesystem::path succ_file_path = dump_path;
    succ_file_path /= "SUCCESS";
    EXPECT_TRUE(std::filesystem::is_regular_file(succ_file_path));
  }
}
