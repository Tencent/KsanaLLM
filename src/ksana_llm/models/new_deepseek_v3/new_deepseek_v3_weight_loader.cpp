/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <regex>
#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>

#include "ksana_llm/models/base/model_arch.h"
#include "ksana_llm/models/base/model_format.h"
#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"
#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_weight_loader.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/kernels/trans_layout.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/string_utils.h"
#include "ksana_llm/utils/absorb_weights_type.h"

#ifdef ENABLE_CUDA
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {
NewDeepSeekV3WeightLoader::NewDeepSeekV3WeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                                     std::shared_ptr<Environment> env,
                                                     std::shared_ptr<Context> context) :
                                                     BaseModelWeightLoader(model_config, env, context),
                                                     weight_impl_(nullptr) {
  // Initialize pipeline config, for distributed mode.
  env->GetPipelineConfig(pipeline_config_);
}

NewDeepSeekV3WeightLoader::~NewDeepSeekV3WeightLoader() {}

// local helper function
std::string replace_first(std::string str, const std::string& key, const std::string& replacement) {
  size_t pos = str.find(key);
  if (pos != std::string::npos) {
      str.replace(pos, key.length(), replacement);
  }
  return str;
}

Status NewDeepSeekV3WeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq"};
  std::vector<std::string> master_only_list = {"model.embed_tokens.weight", "lm_head.weight"};

  int lower_layer_idx = pipeline_config_.lower_layer_idx;
  int upper_layer_idx = pipeline_config_.upper_layer_idx;
  int lower_nextn_layer_idx = pipeline_config_.lower_nextn_layer_idx;
  int upper_nextn_layer_idx = pipeline_config_.upper_nextn_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end(); ) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      weight_names.erase(it);
      continue;
    }

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 &&
          ((layer_idx < lower_layer_idx || layer_idx > upper_layer_idx) &&
          (layer_idx < lower_nextn_layer_idx || layer_idx > upper_nextn_layer_idx))) {
        weight_names.erase(it);
        continue;
      }
      // Skip nextn layer in distributed non-master node
      if (layer_idx >= 0 && !context_->IsChief() &&
          (layer_idx >= lower_nextn_layer_idx && layer_idx <= upper_nextn_layer_idx)) {
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

Status NewDeepSeekV3WeightLoader::PostProcessModelWeights(
                                        std::unordered_map<std::string, Tensor>& dev_weights_map,
                                        int dev_rank) {
  std::shared_ptr<NewDeepSeekV3Config> new_deepseek_v3_config =
      std::dynamic_pointer_cast<NewDeepSeekV3Config> (model_config_);
  if (GetAbsorbWeightsType() == AbsorbWeightsType::kAbsorbTypeUKV) {
#ifdef ENABLE_CUDA
    weight_impl_->ProcessAbsorbWeightsTypeUKV(dev_weights_map, dev_rank,
              new_deepseek_v3_config);
#endif
  }
  if (new_deepseek_v3_config->quant_config.is_fp8_blockwise) {
    for (auto & [weight_name, weight_tensor] : dev_weights_map) {
      if (weight_tensor.dtype == DataType::TYPE_FP8_E4M3 ||
          weight_tensor.dtype == DataType::TYPE_BLOCK_FP8_E4M3) {
        std::string weight_scale_name = weight_name + "_scale_inv";
        auto scale_iter = dev_weights_map.find(weight_scale_name);
        if (scale_iter != dev_weights_map.end()) {
          weight_tensor.weight_scales = &(scale_iter->second);
          KLLM_LOG_INFO << fmt::format("bind {}, shape: {} to {}, shape: {}\n",
                                        weight_scale_name,
                                        Vector2Str(std::vector<size_t>(weight_tensor.weight_scales->shape)),
                                        weight_name,
                                        Vector2Str(std::vector<size_t>(weight_tensor.shape)));
        } else {
          KLLM_LOG_INFO << fmt::format("weight scale not found: {}", weight_scale_name);
        }
      }
    }
  }

  return Status();
}

// TODO(huicongyao): support attn data parallel and expert parallel for deepseek v3
Status NewDeepSeekV3WeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                       int dev_rank,
                                       std::unordered_map<std::string, Tensor>& device_model_weights,
                                       std::unordered_map<std::string, Tensor>& left_host_weights) {
  std::shared_ptr<NewDeepSeekV3Config> new_deepseek_v3_config =
      std::dynamic_pointer_cast<NewDeepSeekV3Config> (model_config_);

  if (weight_impl_ == nullptr) {
    STATUS_CHECK_RETURN(InitQuantWeightLoader(new_deepseek_v3_config));
  }

  size_t layer_idx, expert_idx;
  int num_experts = new_deepseek_v3_config->moe_config.num_experts;
  bool use_vllm_moe = new_deepseek_v3_config->moe_config.use_vllm_moe;

  // Moe TP directly distributes the expert dimensions across multiple cards.
  size_t moe_inter_size_per_rank = DivRoundUp(new_deepseek_v3_config->moe_config.moe_inter_size,
    context_->GetTensorParallelSize());
  size_t hidden_units = new_deepseek_v3_config->hidden_units;
  std::vector<size_t> up_gate_experts_shape = {size_t(num_experts),
    /* up & gate*/moe_inter_size_per_rank * 2, hidden_units};
  std::vector<size_t> down_experts_shape = {size_t(num_experts), hidden_units, moe_inter_size_per_rank};

  size_t kv_lora_rank = new_deepseek_v3_config->mla_config.kv_lora_rank;
  size_t qk_rope_head_dim = new_deepseek_v3_config->mla_config.qk_rope_head_dim;
  size_t qk_nope_head_dim = new_deepseek_v3_config->mla_config.qk_nope_head_dim;
  size_t v_head_dim = new_deepseek_v3_config->mla_config.v_head_dim;
  size_t head_num = new_deepseek_v3_config->head_num;
  size_t head_num_tp = static_cast<size_t>(DivRoundUp(new_deepseek_v3_config->head_num,
      context_->GetTensorParallelSize()));

  if (new_deepseek_v3_config->model_format == ModelFormat::GGUF) {
    return Status(RET_INVALID_ARGUMENT, "Not support GGUF format yet.");
  }

  // Record processed model weights
  std::unordered_set<std::string> processed_weights;
  // Record the weights that need to dequant
  std::unordered_set<std::string> dequant_weights;
  for (auto & [file_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_INFO << fmt::format("Processing weight: {}, shape: {}",
      file_weight_name, Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    // 1. model.embed_tokens.weight;
    // ::FilterWeightNames() filtered this for non master nodes
    // Embedding TP needs to be transposed first, then split, and then transposed back.
    if (CheckWeightNameMatched(file_weight_name, {"model.embed_tokens.weight"}, true)) {
      Tensor dev_tensor;
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, new_deepseek_v3_config, true);

      device_model_weights[file_weight_name] = dev_tensor;
      processed_weights.insert(file_weight_name);
      continue;
    }

    // lm_head.weight
    // lm_head need to split along axis = 0, and then transpose
    if (CheckWeightNameMatched(file_weight_name, {
        "lm_head.weight"
      }, true)) {
      Tensor permute_dev_tensor;
      weight_impl_->SplitOptTrans(host_weight_tensor, permute_dev_tensor,
            dev_rank, new_deepseek_v3_config, false);
      device_model_weights[file_weight_name] = permute_dev_tensor;
      processed_weights.insert(file_weight_name);
      continue;
    }

    // 2. dense MLP layer and shared expert layer
    // The parameters of both the mlp (dense) layer and the shared expert layer need to be transposed.
    if (CheckWeightNameMatched(file_weight_name, {
        ".mlp.gate_proj.",
        ".mlp.up_proj.",
        ".mlp.shared_experts.gate_proj",
        ".mlp.shared_experts.up_proj"
      }, false)) {
      // "up && gate proj(bf16/fp16): First split along axis = 0, then transpose."
      // "up && gate proj(fp8 weight/fp32 weight_scale_inv): split along axis = 0."
      Tensor dev_tensor;
      weight_impl_->SplitOptTrans(host_weight_tensor,
        dev_tensor, dev_rank, new_deepseek_v3_config, new_deepseek_v3_config->is_quant);
      std::string file_weight_name_;
      if (file_weight_name.find(".shared_experts.") != std::string::npos) {
        file_weight_name_ = replace_first(file_weight_name, "shared_experts", "shared_expert");
      } else {
        file_weight_name_ = file_weight_name;
      }
      if (new_deepseek_v3_config->type == "deepseek_v2") {
        device_model_weights[file_weight_name_] = dev_tensor;
        processed_weights.insert(file_weight_name_);
      } else {
        // deepseek v3 need to combine gate & up proj
        weight_impl_->ProcessGateUpProjWeight(file_weight_name_, dev_tensor, device_model_weights,
            dev_rank, new_deepseek_v3_config->is_quant);
        if (file_weight_name_.find("_scale_inv") == std::string::npos) {
          processed_weights.insert(file_weight_name_);
        }
      }
      continue;
    }

    if (CheckWeightNameMatched(file_weight_name, {
      ".mlp.down_proj.",
      ".mlp.shared_experts.down_proj",
    }, false)) {
      // down proj(bf16/fp16): transpose first, then split along axis = 0
      // down proj(fp8/fp32): transpose first, then split along axis = 0, and then transpose back.
      Tensor dev_tensor;
      weight_impl_->TransSplitOptTrans(host_weight_tensor,
        dev_tensor, dev_rank, new_deepseek_v3_config, new_deepseek_v3_config->is_quant);

      if (file_weight_name.find(".shared_experts.") != std::string::npos) {
        std::string file_weight_name_ = replace_first(file_weight_name, "shared_experts", "shared_expert");
        device_model_weights[file_weight_name_] = dev_tensor;
        if (file_weight_name.find("_scale_inv") == std::string::npos) {
          processed_weights.insert(file_weight_name_);
        }
      } else {
        device_model_weights[file_weight_name] = dev_tensor;
        if (file_weight_name.find("_scale_inv") == std::string::npos) {
          processed_weights.insert(file_weight_name);
        }
      }
      continue;
    }

    // 3. MOE layer
    // Instructions for loading MoE model weights:
    // For each layer of the model, the experts at the same positions
    // of up and gate need to be concatenated and named as up_gate.weight.
    // All experts corresponding to up_gate and down in each layer need to be stacked into one expert weight.
    if (file_weight_name.find(".experts.") != std::string::npos) {
#ifdef ENABLE_FP8
      if (new_deepseek_v3_config->quant_config.is_fp8_blockwise &&
          weight_impl_->LoadMoeFp8E4m3BlockWiseScale(file_weight_name,
                                       host_weight_tensor,
                                       dev_rank,
                                       new_deepseek_v3_config,
                                       device_model_weights)) {
        continue;
      }
#endif
      if (new_deepseek_v3_config->quant_config.method == QUANT_FP8_E4M3 &&
          (file_weight_name.find("input_scale") != std::string::npos ||
          file_weight_name.find("weight_scale") != std::string::npos)) {
        continue;
      }

      STATUS_CHECK_RETURN(weight_impl_->GetExpertsIdx(file_weight_name, layer_idx, expert_idx));
      if (file_weight_name.find(".up_proj.") != std::string::npos ||
          file_weight_name.find(".gate_proj.") != std::string::npos) {
        std::string up_gate_experts_name =
          "model.layers." + std::to_string(layer_idx) + ".mlp.experts.up_gate_proj.weight";
        // if this is quant model, keep the weight same as host weights
        DataType up_gate_experts_dtype = new_deepseek_v3_config->is_quant ?
          static_cast<DataType>(host_weight_tensor.dtype) : new_deepseek_v3_config->weight_data_type;
        if (device_model_weights.find(up_gate_experts_name) == device_model_weights.end()) {
          device_model_weights[up_gate_experts_name] = Tensor(MemoryLocation::LOCATION_DEVICE,
            up_gate_experts_dtype, up_gate_experts_shape, dev_rank);
          processed_weights.insert(up_gate_experts_name);
        }

        Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
          host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);
        MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(),
          host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[dev_rank]);

        size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(up_gate_experts_dtype);
        size_t double_expert_pitch = expert_pitch * 2;
        size_t src_upgate_offset = dev_rank * expert_pitch;

        Tensor& up_gate_experts_tensor = device_model_weights.at(up_gate_experts_name);
        if (up_gate_experts_tensor.dtype != dev_tensor.dtype) {
          CastDeviceTensorType(dev_tensor, new_deepseek_v3_config->weight_data_type, dev_rank);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
        }
        if (file_weight_name.find(".up_proj.") != std::string::npos) {
          MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx) * double_expert_pitch +
                      (use_vllm_moe ? expert_pitch : 0),
                      dev_tensor.GetPtr<void>() + src_upgate_offset, expert_pitch, MEMCPY_DEVICE_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);
        } else if (file_weight_name.find(".gate_proj.") != std::string::npos) {
          MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx) * double_expert_pitch +
                      (use_vllm_moe ? 0 : expert_pitch),
                      dev_tensor.GetPtr<void>() + src_upgate_offset, expert_pitch, MEMCPY_DEVICE_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);
        }
      }

      if (file_weight_name.find(".down_proj.") != std::string::npos) {
        std::string down_experts_name = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.down_proj.weight";

        // if this is quant model, keep the weight same as host weights
        DataType down_experts_dtype = new_deepseek_v3_config->is_quant ?
          static_cast<DataType>(host_weight_tensor.dtype) : new_deepseek_v3_config->weight_data_type;
        if (device_model_weights.find(down_experts_name) == device_model_weights.end()) {
          device_model_weights[down_experts_name] = Tensor(MemoryLocation::LOCATION_DEVICE,
            down_experts_dtype, down_experts_shape, dev_rank);
          processed_weights.insert(down_experts_name);
        }

        Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
          host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);
        MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(),
          host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[dev_rank]);

        size_t dst_pitch = moe_inter_size_per_rank * GetTypeSize(down_experts_dtype);
        size_t src_pitch = moe_inter_size_per_rank *
          context_->GetTensorParallelSize() * GetTypeSize(down_experts_dtype);
        size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(down_experts_dtype);
        size_t src_down_offset = dev_rank * dst_pitch;
        Tensor& down_expert_tensor = device_model_weights.at(down_experts_name);
        if (down_expert_tensor.dtype != dev_tensor.dtype) {
          CastDeviceTensorType(dev_tensor, down_experts_dtype, dev_rank);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
        }
        Memcpy2DAsync(down_expert_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx) * expert_pitch, dst_pitch,
                      dev_tensor.GetPtr<void>() + src_down_offset, src_pitch,
                      dst_pitch, hidden_units, MEMCPY_DEVICE_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);
        StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
      }
      continue;
    }

    // 4. MLA layer
    if (file_weight_name.find("self_attn") != std::string::npos &&
        file_weight_name.find("norm.") == std::string::npos) {
#ifdef ENABLE_FP8
      if (new_deepseek_v3_config->quant_config.is_fp8_blockwise &&
          weight_impl_->LoadMlaFp8E4m3BlockWiseScale(file_weight_name,
                                       host_weight_tensor,
                                       dev_rank,
                                       new_deepseek_v3_config,
                                       device_model_weights)) {
        continue;
      }
#endif
      // q_proj is for deepseek v2, q_b_proj is for deepseek v3
      if (file_weight_name.find(".q_proj.weight") != std::string::npos ||
          file_weight_name.find(".q_b_proj.weight") != std::string::npos) {
        // 3072 is deepseek v2
        if (host_weight_tensor.shape[0] != 3072 &&
            (qk_nope_head_dim + qk_rope_head_dim) * head_num != host_weight_tensor.shape[0]) {
          KLLM_THROW(fmt::format(
            "The shape of the 0th dim of the weight named '{} ({})' is not equal to the sum of qk_nope_head_dim {} "
            "and qk_rope_head_dim {}.",
            file_weight_name, host_weight_tensor.shape[0], qk_nope_head_dim, qk_rope_head_dim));
        }

        if (!new_deepseek_v3_config->quant_config.is_fp8_blockwise) {
          // For q_b_nope_proj weight load
          std::string q_b_nope_name =
            file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.q_b_nope_proj.weight";
          std::vector<size_t> q_b_nope_shape = {
            static_cast<size_t>(DivRoundUp(head_num * qk_nope_head_dim, context_->GetTensorParallelSize())),
            host_weight_tensor.shape[1]
          };

          size_t para_pitch = DivRoundUp(head_num, context_->GetTensorParallelSize()) *
                              (qk_nope_head_dim + qk_rope_head_dim) *
                              host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t tensor_para_offset = dev_rank * para_pitch;

          Tensor q_b_nope_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, q_b_nope_shape, dev_rank);
          size_t nope_dst_pitch = qk_nope_head_dim * host_weight_tensor.shape[1] *
            GetTypeSize(host_weight_tensor.dtype);
          size_t nope_src_pitch = (qk_nope_head_dim + qk_rope_head_dim) *
            host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          Memcpy2DAsync(q_b_nope_tensor.GetPtr<void>(), nope_dst_pitch,
                        host_weight_tensor.GetPtr<void>() + tensor_para_offset, nope_src_pitch,
                        nope_dst_pitch, DivRoundUp(head_num, context_->GetTensorParallelSize()), MEMCPY_HOST_TO_DEVICE,
                        context_->GetMemoryManageStreams()[dev_rank]);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

          Tensor q_b_nope_permute = Tensor(MemoryLocation::LOCATION_DEVICE,
                        host_weight_tensor.dtype, q_b_nope_shape, dev_rank);
          PermuteDeviceTensor(q_b_nope_tensor, {1, 0}, dev_rank, q_b_nope_permute);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

          device_model_weights[q_b_nope_name] = q_b_nope_permute;
          processed_weights.insert(q_b_nope_name);

          // For q_b_rope_proj weight load
          std::string q_b_rope_name =
            file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.q_b_rope_proj.weight";
          std::vector<size_t> q_b_rope_shape = {
            static_cast<size_t> (DivRoundUp(head_num * qk_rope_head_dim, context_->GetTensorParallelSize())),
            host_weight_tensor.shape[1]
          };

          Tensor q_b_rope_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, q_b_rope_shape, dev_rank);
          size_t rope_dst_pitch = qk_rope_head_dim *
            host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          Memcpy2DAsync(q_b_rope_tensor.GetPtr<void>(),
            rope_dst_pitch, host_weight_tensor.GetPtr<void>() + nope_dst_pitch + tensor_para_offset,
            nope_src_pitch, rope_dst_pitch,
            DivRoundUp(head_num, context_->GetTensorParallelSize()), MEMCPY_HOST_TO_DEVICE,
            context_->GetMemoryManageStreams()[dev_rank]);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

          Tensor q_b_rope_permute = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, q_b_rope_shape, dev_rank);
          PermuteDeviceTensor(q_b_rope_tensor, {1, 0}, dev_rank, q_b_rope_permute);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

          device_model_weights[q_b_rope_name] = q_b_rope_permute;
          processed_weights.insert(q_b_rope_name);
        } else {
          // For fp8 blockwise quant, do not split the weights initially, split them after dequantization later.
          std::string q_b_proj_name =
            file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.q_b_proj.weight";
          std::vector<size_t> q_b_proj_shape = {
            static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize())),
            host_weight_tensor.shape[1]
          };
          Tensor q_b_proj_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, q_b_proj_shape, dev_rank);

          size_t para_pitch = DivRoundUp(head_num,
            context_->GetTensorParallelSize()) * (qk_nope_head_dim + qk_rope_head_dim) *
            host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t tensor_para_offset = dev_rank * para_pitch;
          MemcpyAsync(q_b_proj_tensor.GetPtr<void>(),
            host_weight_tensor.GetPtr<void>() + tensor_para_offset, para_pitch, MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);

          device_model_weights[q_b_proj_name] = q_b_proj_tensor;
          // processed_weights.insert(q_b_proj_name);
          dequant_weights.insert(q_b_proj_name);
        }
      }
      // q_a_proj is for deepseek v3
      if (file_weight_name.find(".q_a_proj.weight") != std::string::npos) {
        // Weights are not split and are copied to each GPU.
        // fp16/bf16: weights needs to transpose
        Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
          host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

        MemcpyAsync(dev_tensor.GetPtr<void>(),
          host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[dev_rank]);
        StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

        if (host_weight_tensor.dtype == DataType::TYPE_FP16 ||
            host_weight_tensor.dtype == DataType::TYPE_BF16) {
          Tensor q_a_proj_permute = Tensor(MemoryLocation::LOCATION_DEVICE,
            dev_tensor.dtype, dev_tensor.shape, dev_rank);
          PermuteDeviceTensor(dev_tensor, {1, 0}, dev_rank, q_a_proj_permute);
          device_model_weights[file_weight_name] = q_a_proj_permute;
        } else {
          device_model_weights[file_weight_name] = dev_tensor;
          processed_weights.insert(file_weight_name);
        }
        continue;
      }
      if (file_weight_name.find(".kv_a_proj_with_mqa.weight") != std::string::npos) {
        if ((kv_lora_rank + qk_rope_head_dim) != host_weight_tensor.shape[0]) {
          KLLM_THROW(
            fmt::format("The shape of the 0th dim of the weight named `{}` is not equal to the sum of kv_lora_rank {} "
            "and qk_rope_head_dim {}.",
            file_weight_name, kv_lora_rank, qk_rope_head_dim));
        }

        // For kv_a_lora_proj weight load
        std::string kv_a_lora_name =
          file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.kv_a_lora_proj.weight";
        std::vector<size_t> kv_a_lora_shape = {kv_lora_rank, host_weight_tensor.shape[1]};
        Tensor kv_a_lora_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
          host_weight_tensor.dtype, kv_a_lora_shape, dev_rank);
        size_t kv_a_lora_size = kv_lora_rank *
          host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
        MemcpyAsync(kv_a_lora_tensor.GetPtr<void>(),
          host_weight_tensor.GetPtr<void>(), kv_a_lora_size, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[dev_rank]);
        StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

        if (new_deepseek_v3_config->is_quant) {
          device_model_weights[kv_a_lora_name] = kv_a_lora_tensor;
        } else {
          Tensor kv_a_lora_permute = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, kv_a_lora_shape, dev_rank);
          PermuteDeviceTensor(kv_a_lora_tensor, {1, 0}, dev_rank, kv_a_lora_permute);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
          device_model_weights[kv_a_lora_name] = kv_a_lora_permute;
        }
        processed_weights.insert(kv_a_lora_name);

        // For kv_a_rope_proj weight load
        std::string kv_a_rope_name =
          file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.kv_a_rope_proj.weight";
        std::vector<size_t> kv_a_rope_shape = {qk_rope_head_dim, host_weight_tensor.shape[1]};
        size_t kv_a_rope_size = qk_rope_head_dim * host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
        Tensor kv_a_rope_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
          host_weight_tensor.dtype, kv_a_rope_shape, dev_rank);
        MemcpyAsync(kv_a_rope_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + kv_a_lora_size,
                    kv_a_rope_size, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[dev_rank]);
        StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

        // do not permute for fp8 weights
        if (new_deepseek_v3_config->is_quant) {
          device_model_weights[kv_a_rope_name] = kv_a_rope_tensor;
        } else {
          Tensor kv_a_rope_permute = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, kv_a_rope_shape, dev_rank);
          PermuteDeviceTensor(kv_a_rope_tensor, {1, 0}, dev_rank, kv_a_rope_permute);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
          device_model_weights[kv_a_rope_name] = kv_a_rope_permute;
        }
        processed_weights.insert(kv_a_rope_name);
      }
      if (file_weight_name.find(".kv_b_proj.weight") != std::string::npos) {
        if (head_num * (qk_nope_head_dim + v_head_dim) != host_weight_tensor.shape[0]) {
          KLLM_THROW(fmt::format(
            "The shape of the 0th dim of the weight named '{}' is not equal to the sum of qk_nope_head_dim {} "
            "and v_head_dim {}.",
            file_weight_name, kv_lora_rank, qk_rope_head_dim));
        }

        if (!new_deepseek_v3_config->quant_config.is_fp8_blockwise) {
          // For kv_b_nope_proj weight load
          std::string kv_b_nope_name =
            file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.kv_b_nope_proj.weight";
          std::vector<size_t> kv_b_nope_shape = {
            static_cast<size_t>(DivRoundUp(head_num * qk_nope_head_dim, context_->GetTensorParallelSize())),
            host_weight_tensor.shape[1]
          };

          size_t para_pitch = DivRoundUp(head_num, context_->GetTensorParallelSize()) *
                              (qk_nope_head_dim + v_head_dim) *
                              host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t tensor_para_offset = dev_rank * para_pitch;

          Tensor kv_b_nope_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, kv_b_nope_shape, dev_rank);
          size_t nope_dst_pitch = qk_nope_head_dim * host_weight_tensor.shape[1] *
            GetTypeSize(host_weight_tensor.dtype);
          size_t nope_src_pitch = (qk_nope_head_dim + v_head_dim) *
            host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          Memcpy2DAsync(kv_b_nope_tensor.GetPtr<void>(), nope_dst_pitch,
                        host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                        nope_src_pitch, nope_dst_pitch,
                        DivRoundUp(head_num, context_->GetTensorParallelSize()), MEMCPY_HOST_TO_DEVICE,
                        context_->GetMemoryManageStreams()[dev_rank]);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

          Tensor kv_b_nope_permute = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, kv_b_nope_shape, dev_rank);
          PermuteDeviceTensor(kv_b_nope_tensor, {1, 0}, dev_rank, kv_b_nope_permute);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

          device_model_weights[kv_b_nope_name] = kv_b_nope_permute;
          processed_weights.insert(kv_b_nope_name);

          // For v_head_proj weight load
          std::string v_head_name = file_weight_name.substr(0,
              file_weight_name.find_first_of('_')) + "_attn.v_head_proj.weight";
          std::vector<size_t> v_head_shape = {
            static_cast<size_t>(DivRoundUp(head_num * v_head_dim, context_->GetTensorParallelSize())),
            host_weight_tensor.shape[1]
          };

          Tensor v_head_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, v_head_shape, dev_rank);
          size_t v_head_dst_pitch = v_head_dim * host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          Memcpy2DAsync(v_head_tensor.GetPtr<void>(),
            v_head_dst_pitch, host_weight_tensor.GetPtr<void>() + nope_dst_pitch + tensor_para_offset,
            nope_src_pitch, v_head_dst_pitch,
            DivRoundUp(head_num, context_->GetTensorParallelSize()), MEMCPY_HOST_TO_DEVICE,
            context_->GetMemoryManageStreams()[dev_rank]);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

          Tensor v_head_permute = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, v_head_shape, dev_rank);
          PermuteDeviceTensor(v_head_tensor, {1, 0}, dev_rank, v_head_permute);
          StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

          device_model_weights[v_head_name] = v_head_permute;
          processed_weights.insert(v_head_name);

          if (GetAbsorbWeightsType() == AbsorbWeightsType::kAbsorbTypeBMM) {
            // Copy kv_b_nope_proj to w_uk_t
            Tensor w_uk_t_tensor = kv_b_nope_tensor;
            w_uk_t_tensor.shape = {head_num_tp, qk_nope_head_dim, kv_b_nope_tensor.shape[1]};
            std::string w_uk_t_name = kv_b_nope_name.substr(0, kv_b_nope_name.find_first_of('_')) +
                "_attn.w_uk_t.weight";
            device_model_weights[w_uk_t_name] = w_uk_t_tensor;
            processed_weights.insert(w_uk_t_name);

            // Permute vhead_weight_name to w_uv
            Tensor w_uv_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, w_uk_t_tensor.dtype,
                {head_num_tp, v_head_shape[1], v_head_dim}, dev_rank);
            v_head_tensor.shape = {head_num_tp, v_head_dim, v_head_shape[1]};
            PermuteDeviceTensor(v_head_tensor, {0, 2, 1}, dev_rank, w_uv_tensor);
            std::string w_uv_name = v_head_name.substr(0, v_head_name.find_first_of('_')) +
                "_attn.w_uv.weight";
            v_head_tensor.shape = v_head_shape;
            device_model_weights[w_uv_name] = w_uv_tensor;
            processed_weights.insert(w_uv_name);
          }
        } else {
          // For fp8 blockwise quant, do not split the weights initially, split them after dequantization later.
          std::vector<size_t> kv_b_proj_shape  = {
            static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize())),
            host_weight_tensor.shape[1]
          };

          Tensor kv_b_proj_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
            host_weight_tensor.dtype, kv_b_proj_shape, dev_rank);
          size_t para_pitch = DivRoundUp(head_num, context_->GetTensorParallelSize()) *
            (qk_nope_head_dim + v_head_dim) *
                              host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t tensor_para_offset = dev_rank * para_pitch;
          MemcpyAsync(kv_b_proj_tensor.GetPtr<void>(),
                      host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                      para_pitch, MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);

          device_model_weights[file_weight_name] = kv_b_proj_tensor;
          dequant_weights.insert(file_weight_name);
        }
      }
      if (file_weight_name.find(".o_proj.weight") != std::string::npos) {
        // bf16/fp16: Transpose, then split along axis = 0
        // fp8: Transpose, then split along axis = 0, then transpose
        Tensor dev_tensor;
        weight_impl_->TransSplitOptTrans(host_weight_tensor,
          dev_tensor, dev_rank, new_deepseek_v3_config, new_deepseek_v3_config->is_quant);

        device_model_weights[file_weight_name] = dev_tensor;
        processed_weights.insert(file_weight_name);
        continue;
      }
    }

    // 5. norm layer or `gate.e_score_correction_bias`
    // Directly load to each device.
    if (file_weight_name.find("norm.") != std::string::npos ||
        file_weight_name.find("gate.e_score_correction_bias") != std::string::npos) {
      Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
        host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

      MemcpyAsync(dev_tensor.GetPtr<void>(),
                  host_weight_tensor.GetPtr<void>(),
                  host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[dev_rank]);

      device_model_weights[file_weight_name] = dev_tensor;
      processed_weights.insert(file_weight_name);
      continue;
    }

    // 6. gate weight
    // Copy to each device and transpose
    if (file_weight_name.find(".mlp.gate.weight") != std::string::npos) {
      Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
        host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

      MemcpyAsync(dev_tensor.GetPtr<void>(),
        host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[dev_rank]);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      Tensor permute_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
        host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);
      PermuteDeviceTensor(dev_tensor, {1, 0}, dev_rank, permute_dev_tensor);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      device_model_weights[file_weight_name] = permute_dev_tensor;
      processed_weights.insert(file_weight_name);
      continue;
    }

    // 7. eh_proj.weight (for mtp layer)
    // Split along axis = 0
    if (file_weight_name.find(".eh_proj.weight") != std::string::npos) {
      Tensor dev_tensor;
      weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, new_deepseek_v3_config, false);
      CastDeviceTensorType(dev_tensor, new_deepseek_v3_config->weight_data_type, dev_rank);
      device_model_weights[file_weight_name] = dev_tensor;
      processed_weights.insert(file_weight_name);
    }
  }
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

#ifdef ENABLE_FP8
#ifdef ENABLE_FP8_TORCH
  // handle q_b_proj and kv_b_proj
  weight_impl_->ProcessMlaFp8E4m3BlockWiseScaleOfWeight(processed_weights,
                                                         dequant_weights,
                                                         dev_rank,
                                                         new_deepseek_v3_config,
                                                         device_model_weights);
#endif
#endif
  // cast all data type to desired data type (skip quant weights)
  for (const std::string& weight_name : processed_weights) {
    if (device_model_weights.at(weight_name).dtype != new_deepseek_v3_config->weight_data_type &&
        !(weight_name.find("proj.weight") != std::string::npos && new_deepseek_v3_config->is_quant) &&
        (weight_name.find("gate.e_score_correction_bias") == std::string::npos)) {
      CastDeviceTensorType(device_model_weights.at(weight_name), new_deepseek_v3_config->weight_data_type, dev_rank);
    }
  }
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

  KLLM_LOG_INFO << fmt::format("Process model weights on device-{} finished.", dev_rank);
  return Status();
}

Status NewDeepSeekV3WeightLoader::InitQuantWeightLoader(
        std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config) {
  switch (new_deepseek_v3_config->weight_data_type) {
    case DataType::TYPE_FP32:
      weight_impl_ = std::make_unique<NewDeepSeekV3WeightImpl<float>>(context_);
      break;
#ifdef ENABLE_BFLOAT16
    case DataType::TYPE_BF16:
      weight_impl_ = std::make_unique<NewDeepSeekV3WeightImpl<bfloat16>>(context_);
      break;
#endif
    case DataType::TYPE_FP16:
      weight_impl_ = std::make_unique<NewDeepSeekV3WeightImpl<float16>>(context_);
      break;
    default:
      // Handle unexpected data type if needed
      KLLM_THROW(fmt::format("Unexpected data type: {}", new_deepseek_v3_config->weight_data_type));
      break;
  }
  return Status();
}
}  // namespace ksana_llm
