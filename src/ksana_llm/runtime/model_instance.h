/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <memory>
#include <string>

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/runtime/weight_instance_inferface.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {
class Worker;
class WorkerGroup;

class ModelInstance {
 public:
  ModelInstance(const ModelConfig& model_config, const RuntimeConfig& runtime_config, std::shared_ptr<Context> context,
                std::shared_ptr<WeightInstanceInterface>& weight_instance)
      : model_config_(model_config),
        runtime_config_(runtime_config),
        context_(context),
        weight_instance_(weight_instance) {
    loader_models_threadpool_ = std::make_shared<ThreadPool>(context->GetTensorParallelSize());
    loader_models_threadpool_->Start();
  }

  ~ModelInstance() {
    models_.clear();
    loader_models_threadpool_->Stop();
  }
  // Load model with specified model config.
  void Load();

  // The instance name.
  std::string name;
  std::string type;

  std::vector<Status> Forward(size_t multi_batch_id, std::shared_ptr<WorkerGroup> worker_group, InferStage stage,
                              std::vector<ForwardRequest*>& forward_reqs, bool epilogue);

  std::vector<std::future<Status>> ForwardAsync(size_t multi_batch_id, std::shared_ptr<WorkerGroup> worker_group,
                                                InferStage stage,
                                                std::vector<ForwardRequest*>& forward_reqs,
                                                bool epilogue, RunMode run_mode = RunMode::kMain);

  // Get  the data type.
  DataType GetWeightDataType() { return model_config_.weight_data_type; }

  // Get the base ptr of model's logits buf.
  std::vector<float*> GetLogitsPtr(size_t multi_batch_id);

  // Get the base ptr of model's output tokens host buf.
  std::vector<int*> GetOutputTokensPtr(size_t multi_batch_id);

  const ModelConfig& GetModelConfig() { return model_config_; }

  size_t GetMaxTokenNum() { return runtime_config_.max_seq_len; }

  uint32_t GetLayerNum() {
    if (pipeline_config_.upper_layer_idx < 0 || pipeline_config_.lower_layer_idx < 0) {
      Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config_);
    }
    uint32_t layer_num = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
    if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(model_config_.num_layer)) {
      layer_num += pipeline_config_.upper_nextn_layer_idx - pipeline_config_.lower_nextn_layer_idx + 1;
    }
    return layer_num;
  }

  void Reset() {
    KLLM_LOG_INFO << "ModelInstance::Reset clear models and weights.";
    models_.clear();
    weight_instance_ = nullptr;
  }

  // Allocate resources for a specific multi_batch_id
  Status AllocResources(size_t multi_batch_id);

  // Free resources for a specific multi_batch_id
  Status FreeResources(size_t multi_batch_id);

 private:
  // The model config.
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;

  PipelineConfig pipeline_config_;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<WeightInstanceInterface> weight_instance_;

  // The base modes
  std::vector<std::shared_ptr<BaseModel>> models_;
  std::shared_ptr<ThreadPool> loader_models_threadpool_ = nullptr;
};

}  // namespace ksana_llm
