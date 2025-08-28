/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/llm_runtime.h"

namespace ksana_llm {

struct PerfProfileConfig {
  struct PerfDpReqConfig {
    size_t single_token_request_num = 0;
    size_t single_token_request_cached_token_num = 0;
    size_t multi_token_request_num = 0;
    size_t multi_token_request_token_num = 0;
    size_t multi_token_forwarding_token_num = 0;
  };

  uint32_t config_id = 0;
  std::vector<PerfDpReqConfig> req_configs;
  size_t warmup_round = 0;
  size_t profile_round = 1;
  size_t layer_forward_round = 1;  // used to simulate multiple layer forward

  std::string ToStr() const {
    std::ostringstream oss;
    oss << "config_id=" << config_id << ", warmup_round=" << warmup_round << ", profile_round=" << profile_round
        << ", layer_forward_round=" << layer_forward_round << ", req_configs.size=" << req_configs.size()
        << ", req_configs={ ";
    for (size_t dp_idx = 0; dp_idx < req_configs.size(); dp_idx++) {
      auto& config = req_configs[dp_idx];
      oss << "\n  [" << dp_idx << "]={ st_req_num=" << config.single_token_request_num
          << ", st_req_token_num=" << config.single_token_request_cached_token_num
          << ", mt_req_num=" << config.multi_token_request_num
          << ", mt_req_token_num=" << config.multi_token_request_token_num
          << ", mt_f_token_num=" << config.multi_token_forwarding_token_num << " } ";
    }
    oss << "}";
    return oss.str();
  }
};

struct PerfProfileResult {
  uint32_t config_id = 0;
  float time_cost_ms;
};

class ModelPerformanceRunner {
 public:
  explicit ModelPerformanceRunner(const std::string& config_path, const PerfProfileConfig& max_config,
                                  int16_t lower_layer_idx = -1, int16_t upper_layer_idx = -1);

  ~ModelPerformanceRunner();

  Status RunPerformanceForward(const PerfProfileConfig& profile_config, PerfProfileResult& result);

  uint32_t GetAttnDpNum() { return attn_dp_worker_num_; }

 private:
  void InitEnvs(const std::string& config_path, const PerfProfileConfig& max_config, int16_t lower_layer_idx,
                int16_t upper_layer_idx);

  void OptimizeBlockManagerConfig(BlockManagerConfig& block_manager_config, const PerfProfileConfig& max_config);

  size_t GetNeededBlockNum(size_t block_token_num, const PerfProfileConfig& max_config) const;

  void LoadModel();

  void ResetInferRequests();

  void InitInferRequests(const PerfProfileConfig& profile_config);

  void CheckRequests() const;

  Status ParsePerformanceRunnerConfig(const std::string& config_file);

  size_t GetBlockNum(std::shared_ptr<InferRequest> req) const;

  uint32_t GetAttnDpGroupId(int64_t req_id) const;

 private:
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<ModelInstance> model_instance_ = nullptr;
  std::shared_ptr<WorkerGroup> worker_group_ = nullptr;
  std::vector<std::shared_ptr<CacheManagerInterface>> cache_managers_;
  std::shared_ptr<LlmRuntime> llm_runtime_ = nullptr;
  uint32_t attn_dp_worker_num_ = 0;

  size_t multi_batch_id_ = DEFAULT_MULTI_BATCH_ID;

  // requests
  std::shared_ptr<KsanaPythonInput> ksana_python_input_;
  std::vector<std::shared_ptr<InferRequest>> infer_reqs_;
  std::unordered_map<size_t, std::vector<int>> input_ids_map_;
  SamplingConfig sampling_config_;
  std::vector<int> input_refit_pos_;
  std::vector<std::vector<float>> embeddings_;
  EmbeddingSlice embedding_slice_;
  std::vector<py::object> embedding_tensors_;
};

class PerfProfileConfigBuilderInterface {
 public:
  ~PerfProfileConfigBuilderInterface() {}

  virtual PerfProfileConfig GetMaxPerfProfileConfig() = 0;
  void SetAttnDpNum(size_t dp_num) { dp_num_ = dp_num; }

  virtual void GetPerfProfileConfigs(std::vector<PerfProfileConfig>& configs) = 0;

 protected:
  size_t dp_num_ = 0;
};

}  // namespace ksana_llm
