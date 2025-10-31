/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/model_instance.h"

#include <future>
#include <memory>
#include <vector>

#include "ksana_llm/runtime/worker.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/status.h"
#include "nlohmann/json.hpp"

#include "ksana_llm/models/baichuan/baichuan_model.h"
#include "ksana_llm/models/bge_reranker_minicpm/bge_reranker_minicpm_model.h"
#include "ksana_llm/models/chatglm/chatglm_model.h"
#include "ksana_llm/models/deepseek_v3/deepseek_v3_model.h"
#include "ksana_llm/models/gpt/gpt_model.h"
#include "ksana_llm/models/hunyuan_large/hunyuan_large_model.h"
#include "ksana_llm/models/internlm2/internlm_model.h"
#include "ksana_llm/models/internlmxcomposer2/internlmxcomposer2_model.h"
#include "ksana_llm/models/llama/llama_model.h"
#include "ksana_llm/models/llama4/llama4_model.h"
#include "ksana_llm/models/mixtral/mixtral_model.h"
#include "ksana_llm/models/qwen/qwen_model.h"
#include "ksana_llm/models/qwen2_moe/qwen2_moe_model.h"
#include "ksana_llm/models/qwen3_moe/qwen3_moe_model.h"

namespace ksana_llm {

template <class ModelType>
void CreateModelInstance(const std::string model_name, ModelConfig& model_config, RuntimeConfig& runtime_config,
                         std::shared_ptr<Context>& context, std::vector<std::shared_ptr<BaseModel>>& models,
                         std::shared_ptr<WeightInstanceInterface>& weight_instance) {
  KLLM_LOG_INFO << "Start to init model instance " << model_name;
  for (size_t worker_id = 0; worker_id < context->GetTensorParallelSize(); ++worker_id) {
    KLLM_LOG_INFO << "Start to create model on device " << worker_id;
    models.push_back(std::make_shared<ModelType>(model_config, runtime_config, worker_id, context,
                                                 weight_instance->GetWeight(worker_id)));
  }
}

void ModelInstance::Load() {
  std::string unified_model_type = model_config_.type;
  // unify it to lower case
  std::transform(unified_model_type.begin(), unified_model_type.end(), unified_model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (unified_model_type.find("llama4") != std::string::npos) {
    type = "llama4";
    CreateModelInstance<Llama4Model>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                     weight_instance_);
  } else if (unified_model_type.find("llama") != std::string::npos) {
    type = "llama";
    CreateModelInstance<LlamaModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                    weight_instance_);
  } else if (unified_model_type.find("qwen3_moe") != std::string::npos) {
    type = "qwen3_moe";
    CreateModelInstance<Qwen3MoeModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                       weight_instance_);
  } else if (unified_model_type.find("qwen3") != std::string::npos) {
    type = "qwen3";
    model_config_.enable_qk_pre_norm_before_rotary_pos = true;
    CreateModelInstance<QwenModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                   weight_instance_);
  } else if (unified_model_type.find("qwen2_moe") != std::string::npos) {
    type = "qwen2_moe";
    CreateModelInstance<Qwen2MoeModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                       weight_instance_);
  } else if (unified_model_type.find("qwen") != std::string::npos) {
    type = "qwen";
    // or qwen2_vl
    model_config_.enable_add_qkv_bias = true;
    CreateModelInstance<QwenModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                   weight_instance_);
  } else if (unified_model_type.find("baichuan") != std::string::npos) {
    type = "baichuan";
    CreateModelInstance<BaichuanModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                       weight_instance_);
  } else if (unified_model_type.find("chatglm") != std::string::npos) {
    type = "chatglm";
    CreateModelInstance<ChatglmModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                      weight_instance_);
  } else if (unified_model_type.find("gpt") != std::string::npos ||
             unified_model_type.find("fairseq-transformer") != std::string::npos) {
    CreateModelInstance<GptModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                  weight_instance_);
  } else if (unified_model_type.find("internlm2") != std::string::npos ||
             unified_model_type.find("internvl_chat") != std::string::npos) {
    type = "internlm2";
    CreateModelInstance<Internlm2Model>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                        weight_instance_);
  } else if (unified_model_type.find("internlmxcomposer2") != std::string::npos) {
    type = "internlm2";
    CreateModelInstance<InternlmxComposer2Model>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                                 weight_instance_);
  } else if (unified_model_type.find("hunyuan") != std::string::npos && model_config_.is_moe) {
    type = "hunyuan";
    CreateModelInstance<HunyuanLargeModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                           weight_instance_);
  } else if (unified_model_type.find("mixtral") != std::string::npos) {
    type = "mixtral";
    CreateModelInstance<MixtralModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                      weight_instance_);
  } else if (unified_model_type.find("deepseek_v3") != std::string::npos ||
             unified_model_type.find("deepseek_v32") != std::string::npos ||
             unified_model_type.find("deepseek_v2") != std::string::npos ||
             unified_model_type.find("kimi_k2") != std::string::npos) {
    type = "deepseek_v3";
    // deepseek v2 and v3 share a weight and model build process
    CreateModelInstance<DeepSeekV3Model>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                         weight_instance_);
  } else if (unified_model_type.find("minicpm") != std::string::npos) {
    type = "minicpm";
    CreateModelInstance<BgeRerankerMinicpmModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                                 weight_instance_);
  } else {
    // Optional weights map
    auto optional_file = Singleton<OptionalFile>::GetInstance();
    std::string& weight_map =
        optional_file->GetOptionalFile(model_config_.path, "weight_map", unified_model_type + "_weight_map.json");
    if (weight_map != "") {
      type = "llama";
      CreateModelInstance<LlamaModel>(unified_model_type, model_config_, runtime_config_, context_, models_,
                                      weight_instance_);
    } else {
      KLLM_THROW(fmt::format("Model type {} is not supported.", unified_model_type));
    }
  }
}

std::vector<float*> ModelInstance::GetLogitsPtr(size_t multi_batch_id) {
  std::vector<float*> results;
  for (auto& model : models_) {
    results.push_back(model->GetLogitsPtr(multi_batch_id));
  }
  return results;
}

std::vector<Status> ModelInstance::Forward(size_t multi_batch_id, std::shared_ptr<WorkerGroup> worker_group,
                                           InferStage stage, std::vector<ForwardRequest>& forward_reqs, bool epilogue) {
  std::vector<Status> results;
  for (size_t worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    results.push_back(worker_group->GetWorker(worker_id)->Forward(
        multi_batch_id, models_[worker_id], weight_instance_->GetWeight(worker_id), stage, forward_reqs, epilogue));
  }
  return results;
}

std::vector<std::future<Status>> ModelInstance::ForwardAsync(size_t multi_batch_id,
                                                             std::shared_ptr<WorkerGroup> worker_group,
                                                             InferStage stage,
                                                             std::vector<ForwardRequest*>& forward_reqs, bool epilogue,
                                                             RunMode run_mode) {
  std::vector<std::future<Status>> results(context_->GetTensorParallelSize());
  for (size_t worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    results.push_back(worker_group->GetWorker(worker_id)->ForwardAsync(multi_batch_id, models_[worker_id],
                                                                       weight_instance_->GetWeight(worker_id), stage,
                                                                       forward_reqs, epilogue, run_mode));
  }
  return results;
}

Status ModelInstance::AllocResources(size_t multi_batch_id) {
  std::vector<Status> results;
  for (auto& model : models_) {
    Status status = model->AllocResources(multi_batch_id);
    if (!status.OK()) {
      return status;
    }
  }
  return Status();
}

Status ModelInstance::FreeResources(size_t multi_batch_id) {
  std::vector<Status> results;
  for (auto& model : models_) {
    Status status = model->FreeResources(multi_batch_id);
    if (!status.OK()) {
      return status;
    }
  }
  return Status();
}

}  // namespace ksana_llm
