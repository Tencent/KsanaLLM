/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/runtime/weight_instance.h"
#include "tests/test.h"

namespace ksana_llm {

class WeightInstanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Py_Initialize();
    InitLoguru();
    setenv("ENABLE_MODEL_CACHE", "1", 1);
    setenv("ENABLE_OLD_LOADER", "1", 1);
    context_ = std::make_shared<Context>(1, 1, 1);
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();
    tmp_cache_model_path_ =
        std::filesystem::absolute(parent_path / "../../../build/tmp" / std::to_string(std::time(nullptr)));
    setenv("MODEL_CACHE_PATH", tmp_cache_model_path_.c_str(), 1);

    const auto &env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path, "/model/deepseek_v3");
    env->GetModelConfig(model_config_);

    AttnBackendConfig attn_backend_config;
    env->GetAttnBackendConfig(attn_backend_config);
    attn_backend_config.enable_blocked_multi_token_forwarding_kv = true;
    env->SetAttnBackendConfig(attn_backend_config);

    BlockManagerConfig block_manager_config;
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.device_allocator_config.blocks_num = 10;  // This test just need a few blocks;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;

    env->GetRuntimeConfig(runtime_config_);
  }

  void TearDown() override {
    unsetenv("ENABLE_MODEL_CACHE");
    unsetenv("MODEL_CACHE_PATH");
    unsetenv("ENABLE_OLD_LOADER");
    uintmax_t n = std::filesystem::remove_all(tmp_cache_model_path_);
    Py_Finalize();
  }

  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<Context> context_{nullptr};
  std::filesystem::path tmp_cache_model_path_;
};

}  // namespace ksana_llm
