/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/llm_runtime.h"

namespace ksana_llm {

struct PerfProfileConfig {
  struct PerfDpReqConfig {
    struct RequestInfo {
      size_t forwarding_token_num;
      size_t sequence_len;
      size_t request_num = 1;  // Default to 1 if not specified
    };

    std::vector<RequestInfo> reqs;
  };

  uint32_t config_id = 0;
  std::vector<PerfDpReqConfig> req_configs;
  size_t warmup_round = 0;
  size_t profile_round = 1;
  size_t layer_forward_round = 1;  // used to simulate multiple layer forward

  std::string ToStr() const {
    std::ostringstream oss;
    oss << "config_id=" << config_id << ", warmup_round=" << warmup_round << ", profile_round=" << profile_round
        << ", layer_forward_round=" << layer_forward_round;
    for (size_t dp_idx = 0; dp_idx < req_configs.size(); dp_idx++) {
      auto& config = req_configs[dp_idx];
      oss << "\n  batch " << dp_idx << ": [f_token_num,seq_len,req_num]={ ";
      for (size_t req_idx = 0; req_idx < config.reqs.size(); req_idx++) {
        if (req_idx > 0) oss << ", ";
        oss << "[" << config.reqs[req_idx].forwarding_token_num << "," << config.reqs[req_idx].sequence_len << ","
            << config.reqs[req_idx].request_num << "]";
      }
      oss << " }";
    }
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

  size_t GetDecodeTokenNumThreshold() const { return decode_token_num_threshold_; }

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

  // Threshold for determining decode vs prefill requests
  // input_ids <= decode_token_num_threshold_ will be regarded as decode (use page attention), default is 1
  size_t decode_token_num_threshold_ = 1;

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

  virtual void GetPerfProfileConfigs(std::vector<PerfProfileConfig>& configs, size_t attn_dp_num) = 0;
};

}  // namespace ksana_llm
