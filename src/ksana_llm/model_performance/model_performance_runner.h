/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/llm_runtime.h"

namespace ksana_llm {

class ModelPerformanceRunner {
 public:
  explicit ModelPerformanceRunner(const std::string& config_path);

  ~ModelPerformanceRunner();

  Status RunPerformanceForward();

 private:
  void InitEnvs(const std::string& config_path);

  void OptimizeBlockManagerConfig(BlockManagerConfig& block_manager_config);

  size_t GetNeededBlockNum(size_t block_token_num) const;

  void LoadModel();

  void InitInferRequests();

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

  // input_config
  size_t single_token_request_num_ = 0;
  size_t single_token_request_cached_token_num_ = 0;
  size_t multi_token_request_num_ = 0;
  size_t multi_token_cached_token_num_ = 0;
  size_t multi_token_request_token_num_ = 0;

  // runner_config
  size_t rounds_ = 0;
  size_t warmp_up_rounds_ = 0;
  size_t multi_batch_id_ = DEFAULT_MULTI_BATCH_ID;

  // requests
  std::shared_ptr<KsanaPythonInput> ksana_python_input_;
  std::vector<std::shared_ptr<InferRequest>> infer_reqs_;
  std::vector<std::vector<int>> input_ids_vec_;
  SamplingConfig sampling_config_;
  std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks_;
  std::vector<int> input_refit_pos_;
  std::vector<std::vector<float>> embeddings_;
  EmbeddingSlice embedding_slice_;
  std::vector<py::object> embedding_tensors_;
};
}  // namespace ksana_llm
