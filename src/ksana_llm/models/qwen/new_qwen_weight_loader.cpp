/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <string>

#include "ksana_llm/kernels/trans_layout.h"
#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/models/base/model_arch.h"
#include "ksana_llm/models/base/model_format.h"
#include "ksana_llm/models/qwen/new_qwen_config.h"
#include "ksana_llm/models/qwen/new_qwen_weight_loader.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"
#include "ksana_llm/utils/tensor.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {

NewQwenWeightLoader::NewQwenWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                         std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_shared<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  weight_method_ = std::make_shared<WeightMethod>(common_weight_loader_, tp_);
}

NewQwenWeightLoader::~NewQwenWeightLoader() {}

Status NewQwenWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq"};
  std::vector<std::string> master_only_list = {"model.embed_tokens.weight", "lm_head.weight"};

  int lower_layer_idx = pipeline_config_.lower_layer_idx;
  int upper_layer_idx = pipeline_config_.upper_layer_idx;
  int lower_nextn_layer_idx = pipeline_config_.lower_nextn_layer_idx;
  int upper_nextn_layer_idx = pipeline_config_.upper_nextn_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, MatchMode::PartialMatch)) {
      weight_names.erase(it);
      continue;
    }

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && ((layer_idx < lower_layer_idx || layer_idx > upper_layer_idx) &&
                             (layer_idx < lower_nextn_layer_idx || layer_idx > upper_nextn_layer_idx))) {
        weight_names.erase(it);
        continue;
      }
    }

    // Skip embedding and lm_head on worker node in distributed mode.
    if (!context_->IsStandalone() && !context_->IsChief()) {
      if (CheckWeightNameMatched(*it, master_only_list, MatchMode::PartialMatch)) {
        weight_names.erase(it);
        continue;
      }
    }
    ++it;
  }
  return Status();
}

Status NewQwenWeightLoader::PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights) {
  std::shared_ptr<NewQwenConfig> new_qwen_config = std::dynamic_pointer_cast<NewQwenConfig>(model_config_);

#define REGISTER_COMMON(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetCommonMethod().get(), \
                                                        &CommonMethod::key##_##weight_name)
#define REGISTER_W4A8AWQ(key, weight_name)                                                        \
  weight_method_->GetRegistry()[#key][#weight_name].Add(weight_method_->GetW4A8AWQMethod().get(), \
                                                        &W4A8AWQMethod::key##_##weight_name)

  if (new_qwen_config->quant_config.method == QUANT_W4A8_AWQ) {
    REGISTER_W4A8AWQ(load, attn_q_k_v_proj);
    REGISTER_W4A8AWQ(load, attn_o_proj);
    REGISTER_W4A8AWQ(load, mlp_gate_up_proj);
    REGISTER_W4A8AWQ(load, mlp_down_proj);
    REGISTER_W4A8AWQ(process, attn_qkv_proj);
    REGISTER_W4A8AWQ(process, attn_o_proj);
    REGISTER_W4A8AWQ(process, mlp_gate_up_proj);
    REGISTER_W4A8AWQ(process, mlp_down_proj);
  } else {
    REGISTER_COMMON(load, attn_q_k_v_proj);
    REGISTER_COMMON(load, attn_o_proj);
    REGISTER_COMMON(load, mlp_gate_up_proj);
    REGISTER_COMMON(load, mlp_down_proj);
    REGISTER_COMMON(process, attn_qkv_proj);
    REGISTER_COMMON(process, attn_o_proj);
    REGISTER_COMMON(process, mlp_gate_up_proj);
    REGISTER_COMMON(process, mlp_down_proj);
  }

  REGISTER_COMMON(load, attn_norm);
  REGISTER_COMMON(load, norm);
  REGISTER_COMMON(load, embed_tokens);
  REGISTER_COMMON(load, lm_head);

  REGISTER_COMMON(process, lm_head);

#undef REGISTER_COMMON
#undef REGISTER_W4A8AWQ

  return Status();
}

Status NewQwenWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                int dev_rank,
                                                std::unordered_map<std::string, Tensor>& device_model_weights,
                                                std::unordered_map<std::string, Tensor>& left_host_weights) {
  auto registry = weight_method_->GetRegistry()["load"];

  for (auto& [host_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, host_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    // 1. attn weights
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"},
                               MatchMode::PartialMatch)) {
      registry["attn_q_k_v_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.o_proj"}, MatchMode::PartialMatch)) {
      registry["attn_o_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".self_attn.key_layernorm", ".self_attn.query_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["attn_norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 2. mlp weights
    if (CheckWeightNameMatched(host_weight_name, {".mlp.gate_proj", ".mlp.up_proj"}, MatchMode::PartialMatch)) {
      registry["mlp_gate_up_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {".mlp.down_proj"}, MatchMode::PartialMatch)) {
      registry["mlp_down_proj"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 3. norm weights
    if (CheckWeightNameMatched(host_weight_name, {"model.norm", ".input_layernorm", ".post_attention_layernorm"},
                               MatchMode::PartialMatch)) {
      registry["norm"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // 4. embedding weights && lm_head weights
    if (CheckWeightNameMatched(host_weight_name, {"model.embed_tokens"}, MatchMode::PartialMatch)) {
      registry["embed_tokens"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(host_weight_name, {"lm_head"}, MatchMode::PartialMatch)) {
      registry["lm_head"].Run<Status>(device_model_weights, host_weight_name, host_weight_tensor, dev_rank);
      continue;
    }

    // TODO(jinxcwu) 后续找到具体模型再优化这一部份
    if (CheckWeightNameMatched(host_weight_name,
                               {".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"},
                               MatchMode::PartialMatch)) {
      const std::string query_key_value_bias_name =
          host_weight_name.substr(0, host_weight_name.find(".self_attn.")) + ".self_attn.query_key_value.bias";
      size_t host_shape0_split = DivRoundUp(host_weight_tensor.shape[0], tp_);
      if (device_model_weights.find(query_key_value_bias_name) == device_model_weights.end()) {
        Tensor query_key_value_bias =
            Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, {1, 3, host_shape0_split}, dev_rank);
        device_model_weights[query_key_value_bias_name] = query_key_value_bias;
      }

      Tensor& query_key_value_bias_tensor = device_model_weights.at(query_key_value_bias_name);
      size_t host_offset = host_shape0_split * GetTypeSize(host_weight_tensor.dtype) * dev_rank;
      size_t dev_offset = 0;
      if (host_weight_name.find("q_proj.bias") != std::string::npos) {
        dev_offset = 0;
      } else if (host_weight_name.find("k_proj.bias") != std::string::npos) {
        dev_offset = host_shape0_split * GetTypeSize(host_weight_tensor.dtype);
      } else if (host_weight_name.find("v_proj.bias") != std::string::npos) {
        dev_offset = host_shape0_split * GetTypeSize(host_weight_tensor.dtype) * 2;
      }
      MemcpyAsync(query_key_value_bias_tensor.GetPtr<void>() + dev_offset,
                  host_weight_tensor.GetPtr<void>() + host_offset,
                  host_shape0_split * GetTypeSize(host_weight_tensor.dtype), MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[dev_rank]);
      continue;
    }
  }
  return Status();
}

Status NewQwenWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                    int dev_rank) {
  auto registry = weight_method_->GetRegistry()["process"];

  for (int layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx; layer_idx++) {
    // 1. attn weights
    const std::string q_proj_name = fmt::format("model.layers.{}.self_attn.q_proj.", layer_idx);
    registry["attn_qkv_proj"].Run<Status>(dev_weights_map, q_proj_name, dev_rank);

    const std::string o_proj_name = fmt::format("model.layers.{}.self_attn.o_proj.", layer_idx);
    registry["attn_o_proj"].Run<Status>(dev_weights_map, o_proj_name, dev_rank);

    // 2. mlp weights
    const std::string gate_proj_name = fmt::format("model.layers.{}.mlp.gate_proj.", layer_idx);
    registry["mlp_gate_up_proj"].Run<Status>(dev_weights_map, gate_proj_name, dev_rank);

    const std::string down_proj_name = fmt::format("model.layers.{}.mlp.down_proj.", layer_idx);
    registry["mlp_down_proj"].Run<Status>(dev_weights_map, down_proj_name, dev_rank);
  }

  // 3. lm_head weights
  const std::string lm_head_name = "lm_head.";
  registry["lm_head"].Run<Status>(dev_weights_map, lm_head_name, dev_rank);

  return Status();
}

}  // namespace ksana_llm
