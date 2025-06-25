/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/service/inference_engine.h"
#include "tests/test.h"

namespace ksana_llm {

class InferenceEngineTest : public ::testing::Test {
 protected:
  void SetUp() override { Py_Initialize(); }

  void TearDown() override {
    if (inference_engine_) {
      inference_engine_->Stop();
    }

    Py_Finalize();
    KLLM_LOG_INFO << "InferenceEngineTest TearDown";
  }

  void InitInferenceEngine() {
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();
    auto env = Singleton<Environment>::GetInstance();
    if (!env) {
      KLLM_THROW("The Environment is nullptr.");
    }
    STATUS_CHECK_FAILURE(env->ParseConfig(config_path));

    PipelineConfig pipeline_config;
    env->GetPipelineConfig(pipeline_config);
    pipeline_config.SetDistributeRelatedConfig();
    env->SetPipelineConfig(pipeline_config);

    Singleton<Environment>::GetInstance()->InitializeExpertParallelConfig();

    // Init inference engine.
    inference_engine_ = std::make_shared<InferenceEngine>(request_queue_);
  }

  std::shared_ptr<InferenceEngine> inference_engine_;

  Channel<std::pair<Status, std::shared_ptr<Request>>> request_queue_;
};

TEST_F(InferenceEngineTest, LoadOperatorOptimization) {
#ifdef ENABLE_CUDA
  std::filesystem::path current_path = __FILE__;
  std::filesystem::path parent_path = current_path.parent_path();
  std::filesystem::path tmp_path =
      std::filesystem::absolute(parent_path / "../../../build/tmp" / std::to_string(std::time(nullptr)));
  setenv("KSANA_GEMM_ALGO_MAP_DIR", tmp_path.c_str(), 1);

  size_t max_m = 4;
  std::string cmd = fmt::format(
      "mkdir -p {} && cd {} && "
      "../../bin/search_best_gemm_algo --batch_size 1 --max_m {} --n 16160 --k 7168 --input_dtype bf16 --output_dtype "
      "bf16 "
      "--inner_compute_dtype bf16 --input_a_transop false --input_b_transop false --op_type 0",
      tmp_path.c_str(), tmp_path.c_str(), max_m);

  // Execute the command
  int ret_code = std::system(cmd.c_str());
  EXPECT_EQ(ret_code, 0) << "Command execution failed: " << cmd;

  InitInferenceEngine();

  auto& algo_map = inference_engine_->context_->ext->GetGPUGemmAlgoHelper().algo_map_;
  EXPECT_EQ(algo_map.begin()->second.begin()->second.size(), max_m);

  std::filesystem::remove_all(tmp_path);
#endif
}

}  // namespace ksana_llm
