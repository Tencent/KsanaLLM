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

class ModelLoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    pybind11::initialize_interpreter();

    // Initialize block manager.
    std::string config_file = GetTestTPConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);

    // Initialize context, set tensor paralle to 2 and attn data parallel to 1.
    context_ = std::make_shared<Context>(2, 1, 1);
  }

  void TearDown() override { pybind11::finalize_interpreter(); }

 protected:
  std::shared_ptr<Context> context_ = nullptr;
};

TEST_F(ModelLoaderTest, TestModelFileLoader) {
  std::vector<std::string> model_dirs = {"/model/llama-hf/7B", "/model/qwen2-hf/0.5B", "/model/llama3-gguf/8B"};

  std::vector<ModelFormat> model_formats = {
      ModelFormat::PYTORCH_BIN,
      ModelFormat::PYTORCH_SAFETENSOR,
      ModelFormat::GGUF,
  };

  std::vector<std::string> py_scripts = {"check_pytorch_bin_tensor.py", "check_pytorch_safe_tensor.py",
                                         "check_gguf_tensor.py"};

  for (size_t i = 0; i < model_dirs.size(); ++i) {
    const std::string& model_dir = model_dirs[i];
    if (!std::filesystem::is_directory(model_dir)) {
      KLLM_LOG_WARNING << "Model dir " << model_dir << " not exsit, skip." << std::endl;
      EXPECT_TRUE(false);
    }

    std::vector<std::string> model_file_list;
    Status status = GetModelFileList(model_dir, model_file_list);
    EXPECT_TRUE(status.OK());
    EXPECT_TRUE(model_file_list.size() > 0);

    ModelFormat model_format;
    status = GetModelFormat(model_dir, model_format);
    EXPECT_EQ(model_format, model_formats[i]);

    status = FilterModelFormatFiles(model_format, model_file_list);
    EXPECT_TRUE(status.OK());
    EXPECT_TRUE(model_file_list.size() > 0);

    // Use last file, because first gguf part maybe contains meta data only.
    std::string model_file = model_file_list.back();
    FileLoader file_loader(model_file);

    std::vector<std::string> weight_names;
    status = file_loader.LoadWeightNames(model_format, weight_names);
    std::cout << status.GetMessage() << std::endl;
    EXPECT_TRUE(status.OK());

    std::unordered_map<std::string, Tensor> result;
    status = file_loader.LoadModelWeights(model_format, weight_names, result);
    EXPECT_TRUE(status.OK());
    EXPECT_TRUE(result.size() > 0);

    // Write every tensor to file.
    std::filesystem::path dump_path = std::filesystem::current_path();
    dump_path /= std::filesystem::path(model_file).filename();

    // Remove dump path and create it again.
    std::filesystem::remove_all(dump_path);
    std::filesystem::create_directories(dump_path);

    for (auto& [weight_name, weight_tensor] : result) {
      std::filesystem::path tensor_data_file = dump_path;
      tensor_data_file /= weight_name + ".npy";
      weight_tensor.SaveToNpyFile(tensor_data_file);
    }

    std::string command = fmt::format("python {}/../src/ksana_llm/model_loader/{} {}",
                                      std::filesystem::current_path().string(), py_scripts[i], model_file);

    // check ret code
    int ret_code = std::system(command.c_str());
    EXPECT_TRUE(WEXITSTATUS(ret_code) == 0);

    // check succ file
    std::filesystem::path succ_file_path = dump_path;
    succ_file_path /= "SUCCESS";
    EXPECT_TRUE(std::filesystem::is_regular_file(succ_file_path));
  }
}

TEST_F(ModelLoaderTest, TestModelConfigParser) {
  std::vector<std::string> model_dirs = {"/model/llama-hf/7B", "/model/llama3-gguf/8B"};

  std::vector<int> head_nums = {32, 32};
  std::vector<int> hidden_units = {4096, 4096};
  std::vector<int> inter_sizes = {11008, 14336};
  std::vector<ModelFormat> model_formats = {ModelFormat::PYTORCH_BIN, ModelFormat::GGUF};

  for (size_t i = 0; i < model_dirs.size(); ++i) {
    const std::string& model_dir = model_dirs[i];
    if (!std::filesystem::is_directory(model_dir)) {
      KLLM_LOG_WARNING << "Model dir " << model_dir << " not exsit, skip." << std::endl;
      EXPECT_TRUE(false);
    }

    ModelConfigParser model_config_parser;

    std::shared_ptr<BaseModelConfig> model_config;
    Status status = model_config_parser.ParseModelConfig(model_dir, model_config);
    EXPECT_TRUE(status.OK());
    EXPECT_TRUE(model_config);

    // Check real type.
    EXPECT_TRUE(typeid(LlamaModelConfig) == typeid(*model_config));

    // check config.
    std::shared_ptr<LlamaModelConfig> llama_model_config = std::dynamic_pointer_cast<LlamaModelConfig>(model_config);

    EXPECT_EQ(llama_model_config->model_dir, model_dir);
    EXPECT_EQ(llama_model_config->model_format, model_formats[i]);
    EXPECT_EQ(llama_model_config->model_arch, ModelArchitecture::ARCH_LLAMA);

    // Check value.
    EXPECT_EQ(llama_model_config->head_num, head_nums[i]);
    EXPECT_EQ(llama_model_config->hidden_units, hidden_units[i]);
    EXPECT_EQ(llama_model_config->inter_size, inter_sizes[i]);
  }
}
