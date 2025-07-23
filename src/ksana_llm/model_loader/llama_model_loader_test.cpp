/* Copyright 2024 Tencent Inc.  All rights reserved.

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

#include "ksana_llm/models/llama/llama_model_config.h"

#include "include/gtest/gtest.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

#include "ksana_llm/helpers/environment_test_helper.h"

using namespace ksana_llm;

class LlamaModelLoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    pybind11::initialize_interpreter();

    // Initialize block manager.
    std::string config_file = GetTestTPConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
    RuntimeConfig runtime_config;
    Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
    parallel_basic_config_ = runtime_config.parallel_basic_config;

    // Initialize context, set tensor paralle to 2 and attn data parallel to 1.
    context_ = std::make_shared<Context>(2, 1, 1);
  }

  void TearDown() override { pybind11::finalize_interpreter(); }

 protected:
  std::shared_ptr<Context> context_ = nullptr;
  ParallelismBasicConfig parallel_basic_config_;
};

TEST_F(LlamaModelLoaderTest, TestLlamaModelConfigParser) {
  std::vector<std::string> model_dirs = {"/model/llama-hf/7B", "/model/llama3-gguf/8B"};

  std::vector<int> head_nums = {32, 32};
  std::vector<int> hidden_units = {4096, 4096};
  std::vector<int> inter_sizes = {11008, 14336};
  std::vector<int> num_layers = {32, 32};
  std::vector<int> vocab_sizes = {32000, 128256};
  std::vector<int> size_per_heads = {128, 128};
  std::vector<int> num_key_value_heads = {32, 8};
  std::vector<int> max_position_embeddings = {2048, 8192};
  std::vector<ModelFormat> model_formats = {ModelFormat::PYTORCH_BIN, ModelFormat::GGUF};

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
    EXPECT_TRUE(typeid(LlamaModelConfig) == typeid(*model_config));

    // check config type.
    std::shared_ptr<LlamaModelConfig> llama_model_config = std::dynamic_pointer_cast<LlamaModelConfig>(model_config);

    // Check common info.
    EXPECT_EQ(llama_model_config->model_dir, model_dir);
    EXPECT_EQ(llama_model_config->model_format, model_formats[i]);
    EXPECT_EQ(llama_model_config->model_arch, ModelArchitecture::ARCH_LLAMA);

    // Check value.
    EXPECT_EQ(llama_model_config->head_num, head_nums[i]);
    EXPECT_EQ(llama_model_config->hidden_units, hidden_units[i]);
    EXPECT_EQ(llama_model_config->inter_size, inter_sizes[i]);
    EXPECT_EQ(llama_model_config->num_layer, num_layers[i]);
    EXPECT_EQ(llama_model_config->vocab_size, vocab_sizes[i]);
    EXPECT_EQ(llama_model_config->size_per_head, size_per_heads[i]);
    EXPECT_EQ(llama_model_config->num_key_value_heads, num_key_value_heads[i]);
    EXPECT_EQ(llama_model_config->max_position_embeddings, max_position_embeddings[i]);
  }
}

TEST_F(LlamaModelLoaderTest, TestLlamaModelWeightLoader) {
  std::vector<std::string> model_dirs = {"/model/llama-hf/7B"};

  for (size_t i = 0; i < model_dirs.size(); ++i) {
    const std::string& model_dir = model_dirs[i];
    if (!std::filesystem::is_directory(model_dir)) {
      KLLM_LOG_WARNING << "Model dir " << model_dir << " not exsit, skip." << std::endl;
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

    for (int rank = 0; rank < dev_weights.size(); ++rank) {
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

    std::string command = fmt::format("python {}/../src/ksana_llm/model_loader/check_llama_weight_tensor.py {} {}",
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
