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
#include "ksana_llm/utils/absorb_weights_type.h"
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
      common_weight_loader_(std::make_unique<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  RuntimeConfig runtime_config;
  env->GetRuntimeConfig(runtime_config);
  weights_to_permute_.resize(runtime_config.parallel_basic_config.tensor_parallel_size);
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
    if (CheckWeightNameMatched(*it, skip_list, false)) {
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
      if (CheckWeightNameMatched(*it, master_only_list, false)) {
        weight_names.erase(it);
        continue;
      }
    }
    ++it;
  }
  return Status();
}

Status NewQwenWeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                    int dev_rank) {
  std::shared_ptr<NewQwenConfig> new_qwen_config = std::dynamic_pointer_cast<NewQwenConfig>(model_config_);
  for (auto& weight_name : weights_to_permute_.at(dev_rank)) {
    auto itr = dev_weights_map.find(weight_name);
    if (itr == dev_weights_map.end()) {
      KLLM_THROW(fmt::format("Can't find weight: {} in device model weights map.", weight_name));
    } else {
      Tensor& weight_tensor = itr->second;
      STATUS_CHECK_RETURN(common_weight_loader_->PermuteWeight(weight_tensor, {1, 0}, dev_rank));
    }
  }

  for (auto& [name, tensor] : dev_weights_map) {
    if (tensor.dtype != new_qwen_config->weight_data_type) {
      CastDeviceTensorType(tensor, new_qwen_config->weight_data_type, dev_rank);
    }
  }
  return Status();
}

Status NewQwenWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                int dev_rank,
                                                std::unordered_map<std::string, Tensor>& device_model_weights,
                                                std::unordered_map<std::string, Tensor>& left_host_weights) {
  std::shared_ptr<NewQwenConfig> new_qwen_config = std::dynamic_pointer_cast<NewQwenConfig>(model_config_);

  for (auto& [host_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, host_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    // 1. attn weights
    // q_proj, k_proj, v_proj, split along axis = 0, cat together, transpose
    // o_proj, transpose and split along axis = 0
    if (host_weight_name.find(".self_attn.") != std::string::npos &&
        host_weight_name.find("norm.") == std::string::npos) {
      if (CheckWeightNameMatched(host_weight_name,
                                 {".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"},
                                 false)) {
        // Init `size_per_head` during first iteration of attn weights
        if (new_qwen_config->size_per_head == 0) {
          if (host_weight_name.find(".q_proj.") != std::string::npos &&
              new_qwen_config->size_per_head * new_qwen_config->head_num != host_weight_tensor.shape[0]) {
            new_qwen_config->size_per_head = host_weight_tensor.shape[0] / new_qwen_config->head_num;
          } else if ((host_weight_name.find(".v_proj.") != std::string::npos ||
                      host_weight_name.find(".k_proj.") != std::string::npos) &&
                     new_qwen_config->size_per_head * new_qwen_config->head_num != host_weight_tensor.shape[1]) {
            new_qwen_config->size_per_head = host_weight_tensor.shape[1] / new_qwen_config->num_key_value_heads;
          }
        }

        STATUS_CHECK_RETURN(common_weight_loader_->LoadMhaWeights(
            host_weight_name, host_weight_tensor, device_model_weights, dev_rank, new_qwen_config->head_num,
            new_qwen_config->num_key_value_heads, new_qwen_config->size_per_head));
        const std::string query_key_value_name =
            host_weight_name.substr(0, host_weight_name.find(".self_attn.")) + ".self_attn.query_key_value.weight";
        weights_to_permute_.at(dev_rank).insert(query_key_value_name);
        continue;
      }

      if (CheckWeightNameMatched(host_weight_name,
                                 {"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"}, false)) {
        const std::string query_key_value_bias_name =
            host_weight_name.substr(0, host_weight_name.find(".self_attn.")) + ".self_attn.query_key_value.bias";
        size_t host_shape0_split = DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize());
        if (device_model_weights.find(query_key_value_bias_name) == device_model_weights.end()) {
          Tensor query_key_value_bias =
              Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, {1, 3, host_shape0_split}, dev_rank);
          device_model_weights[query_key_value_bias_name] = query_key_value_bias;
        }

        Tensor& query_key_value_bias_tensor = device_model_weights.at(query_key_value_bias_name);
        size_t host_offset = host_shape0_split * GetTypeSize(host_weight_tensor.dtype) * dev_rank;
        size_t dev_offset = 0;
        if (host_weight_name.find(".self_attn.q_proj.bias") != std::string::npos) {
          dev_offset = 0;
        } else if (host_weight_name.find(".self_attn.k_proj.bias") != std::string::npos) {
          dev_offset = host_shape0_split * GetTypeSize(host_weight_tensor.dtype);
        } else if (host_weight_name.find(".self_attn.v_proj.bias") != std::string::npos) {
          dev_offset = host_shape0_split * GetTypeSize(host_weight_tensor.dtype) * 2;
        }
        MemcpyAsync(query_key_value_bias_tensor.GetPtr<void>() + dev_offset,
                    host_weight_tensor.GetPtr<void>() + host_offset,
                    host_shape0_split * GetTypeSize(host_weight_tensor.dtype), MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[dev_rank]);
        continue;
      }

      if (CheckWeightNameMatched(host_weight_name, {".self_attn.o_proj.weight"}, false)) {
        Tensor dev_tensor;
        common_weight_loader_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank,
                                                  context_->GetTensorParallelSize(), false);
        device_model_weights[host_weight_name] = dev_tensor;
        continue;
      }
    }

    // 2. mlp weights
    // gate && up _proj, split along axis = 0, cat together, transpose
    if (CheckWeightNameMatched(host_weight_name, {".mlp.gate_proj.", ".mlp.up_proj."}, false)) {
      Tensor dev_tensor;
      common_weight_loader_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, context_->GetTensorParallelSize(),
                                           false);
      const std::string gate_up_proj_name =
          host_weight_name.substr(0, host_weight_name.find(".mlp.")) + ".mlp.gate_up_proj.weight";
      if (device_model_weights.find(gate_up_proj_name) == device_model_weights.end()) {
        Tensor gate_up_proj = Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype,
                                     {dev_tensor.shape[0] * 2, dev_tensor.shape[1]}, dev_rank);
        device_model_weights[gate_up_proj_name] = gate_up_proj;
        weights_to_permute_.at(dev_rank).insert(gate_up_proj_name);
      }
      Tensor& gate_up_proj_tensor = device_model_weights.at(gate_up_proj_name);
      if (host_weight_name.find(".gate_proj.") != std::string::npos) {
        MemcpyAsync(gate_up_proj_tensor.GetPtr<void>(), dev_tensor.GetPtr<void>(), dev_tensor.GetTotalBytes(),
                    MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      } else if (host_weight_name.find(".up_proj.") != std::string::npos) {
        size_t offset = dev_tensor.GetTotalBytes();
        MemcpyAsync(gate_up_proj_tensor.GetPtr<void>() + offset, dev_tensor.GetPtr<void>(), dev_tensor.GetTotalBytes(),
                    MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      }
      continue;
    }
    // down_proj, transpose, split along axis = 0
    if (CheckWeightNameMatched(host_weight_name, {".mlp.down_proj."}, false)) {
      Tensor dev_tensor;
      common_weight_loader_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank,
                                                context_->GetTensorParallelSize(), false);
      device_model_weights[host_weight_name] = dev_tensor;
      continue;
    }

    // 3. norm weights
    if (host_weight_name.find("norm.") != std::string::npos) {
      Tensor dev_tensor =
          Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);
      MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      device_model_weights[host_weight_name] = dev_tensor;
      continue;
    }

    // 4. embedding weights && lm_head weights
    if (host_weight_name.find("model.embed_tokens.weight") != std::string::npos) {
      Tensor dev_tensor;
      common_weight_loader_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank,
                                                context_->GetTensorParallelSize(), true);

      device_model_weights[host_weight_name] = dev_tensor;
      continue;
    }
    if (host_weight_name.find("lm_head.weight") != std::string::npos) {
      Tensor permute_dev_tensor;
      common_weight_loader_->SplitOptTrans(host_weight_tensor, permute_dev_tensor, dev_rank,
                                           context_->GetTensorParallelSize(), true);
      device_model_weights[host_weight_name] = permute_dev_tensor;
      continue;
    }
  }
  return Status();
}

}  // namespace ksana_llm
