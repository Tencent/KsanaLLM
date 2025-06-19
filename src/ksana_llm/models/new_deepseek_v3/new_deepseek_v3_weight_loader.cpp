/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <regex>
#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>

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
                                                     quant_weight_(nullptr) {
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

  for (auto it = weight_names.begin(); it != weight_names.end(); ) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      weight_names.erase(it);
      continue;
    }

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && (layer_idx < lower_layer_idx || layer_idx > upper_layer_idx)) {
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

Status NewDeepSeekV3WeightLoader::TransposeSplitWithFp8Adjustment(const Tensor & host_weight_tensor,
                                      Tensor & output_tensor,
                                      int dev_rank,
                                      std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                      bool is_quant_weight) {
  Tensor full_dev_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

  MemcpyAsync(full_dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
              MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

  Tensor permute_dev_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);
  PermuteDeviceTensor(full_dev_tensor, {1, 0}, dev_rank, permute_dev_tensor);
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

  std::vector<size_t> slice_shape = {
      static_cast<size_t>(DivRoundUp(permute_dev_tensor.shape[0], context_->GetTensorParallelSize())),
      permute_dev_tensor.shape[1]};

  Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, slice_shape, dev_rank);

  size_t slice_offset = dev_tensor.GetTotalBytes() * dev_rank;
  size_t slice_bytes = dev_tensor.GetTotalBytes();
  if (static_cast<size_t>(dev_rank) == context_->GetTensorParallelSize() - 1) {
    slice_bytes = host_weight_tensor.GetTotalBytes() - slice_offset;
  }

  MemcpyAsync(dev_tensor.GetPtr<void>(), permute_dev_tensor.GetPtr<void>() + slice_offset, slice_bytes,
              MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
  if (is_quant_weight) {
    permute_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype, dev_tensor.shape, dev_rank);
    PermuteDeviceTensor(dev_tensor, {1, 0}, dev_rank, permute_dev_tensor);
    output_tensor = permute_dev_tensor;
  } else {
    output_tensor = dev_tensor;
  }

  return Status();
}

Status NewDeepSeekV3WeightLoader::SplitTransposeWithFp8Adjustment(const Tensor & host_weight_tensor,
                                      Tensor & output_tensor,
                                      int dev_rank,
                                      std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                      bool is_quant_weight) {
  std::vector<size_t> slice_shape = {
    static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize())),
    host_weight_tensor.shape[1]};
  Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, slice_shape, dev_rank);

  size_t slice_offset = dev_tensor.GetTotalBytes() * dev_rank;
  size_t slice_bytes = dev_tensor.GetTotalBytes();
  if (static_cast<size_t>(dev_rank) == context_->GetTensorParallelSize() - 1) {
    slice_bytes = host_weight_tensor.GetTotalBytes() - slice_offset;
  }

  MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + slice_offset, slice_bytes,
              MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

  if (is_quant_weight) {
    output_tensor = dev_tensor;
  } else {
    Tensor permute_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype, dev_tensor.shape, dev_rank);
    PermuteDeviceTensor(dev_tensor, {1, 0}, dev_rank, permute_dev_tensor);
    output_tensor = permute_dev_tensor;
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

  STATUS_CHECK_RETURN(InitQuantWeightLoader(new_deepseek_v3_config));

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

    // 1. lm_head.weight or model.embed_tokens.weight;
    // ::FilterWeightNames() filtered this for non master nodes
    // Embedding TP needs to be transposed first, then split, and then transposed back.
    if (CheckWeightNameMatched(file_weight_name, {"model.embed_tokens.weight"}, true)) {
      Tensor dev_tensor;
      TransposeSplitWithFp8Adjustment(host_weight_tensor, dev_tensor, dev_rank, new_deepseek_v3_config);

      Tensor permute_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype, dev_tensor.shape, dev_rank);
      PermuteDeviceTensor(dev_tensor, {1, 0}, dev_rank, permute_dev_tensor);
      device_model_weights[file_weight_name] = permute_dev_tensor;
      processed_weights.insert(file_weight_name);
      continue;
    }

    // The lm_head needs to be split first and then transposed.
    if (CheckWeightNameMatched(file_weight_name, {"lm_head.weight"}, true)) {
      Tensor permute_dev_tensor;
      SplitTransposeWithFp8Adjustment(host_weight_tensor, permute_dev_tensor, dev_rank, new_deepseek_v3_config);
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
      // "up && gate proj(fp8/fp32): split along axis = 0."
      Tensor dev_tensor;
      SplitTransposeWithFp8Adjustment(host_weight_tensor,
        dev_tensor, dev_rank, new_deepseek_v3_config, new_deepseek_v3_config->is_quant);
      std::string file_weight_name_;
      if (file_weight_name.find(".shared_experts.") != std::string::npos) {
        file_weight_name_ = replace_first(file_weight_name, "shared_experts", "shared_expert");
      } else {
        file_weight_name_ = file_weight_name;
      }
      if (!new_deepseek_v3_config->is_quant) {
        device_model_weights[file_weight_name_] = dev_tensor;
        processed_weights.insert(file_weight_name_);
      } else {
        ProcessGateUpProjWeight(file_weight_name_, dev_tensor, device_model_weights, dev_rank, processed_weights);
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
      TransposeSplitWithFp8Adjustment(host_weight_tensor,
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
          LoadMoeFp8E4m3BlockWiseScale(file_weight_name,
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

      STATUS_CHECK_RETURN(GetExpertsIdx(file_weight_name, layer_idx, expert_idx));
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
          LoadMlaFp8E4m3BlockWiseScale(file_weight_name,
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
      // q_a_proj is for deepseek v3 (quant model)
      if (file_weight_name.find(".q_a_proj.weight") != std::string::npos) {
        // Weights are not split and are copied to each GPU.
        Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
          host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

        MemcpyAsync(dev_tensor.GetPtr<void>(),
          host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[dev_rank]);
        StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
        device_model_weights[file_weight_name] = dev_tensor;
        processed_weights.insert(file_weight_name);
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
        TransposeSplitWithFp8Adjustment(host_weight_tensor,
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

    // 6. gate layer
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
  }
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

#ifdef ENABLE_FP8
#  ifdef ENABLE_FP8_TORCH
  // handle q_b_proj and kv_b_proj
  quant_weight_->ProcessMlaFp8E4m3BlockWiseScaleOfWeight(processed_weights,
                                                         dequant_weights,
                                                         dev_rank,
                                                         new_deepseek_v3_config,
                                                         device_model_weights,
                                                         context_);
#  endif
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

  return Status();
}

#ifdef ENABLE_FP8
bool NewDeepSeekV3WeightLoader::LoadMoeFp8E4m3BlockWiseScale(const std::string & host_weight_name,
                                                             const Tensor & host_weight_tensor,
                                                             int dev_rank,
                                                             std::shared_ptr<NewDeepSeekV3Config>
                                                                & new_deepseek_v3_config,
                                                             std::unordered_map<std::string, Tensor>
                                                                & device_model_weights) {
  if (new_deepseek_v3_config->quant_config.method != QUANT_FP8_E4M3 &&
    !new_deepseek_v3_config->quant_config.is_fp8_blockwise) {
    return false;
  }
  if (host_weight_name.find(".experts.") == std::string::npos ||
      (host_weight_name.find(".weight_scale") == std::string::npos &&
       host_weight_name.find(".input_scale") == std::string::npos)) {
    return false;
  }
  if (host_weight_tensor.dtype != DataType::TYPE_FP32) {
    KLLM_THROW("Not support data type of scale: " + host_weight_name);
  }

  size_t layer_idx = -1, expert_idx = -1;
  GetExpertsIdx(host_weight_name, layer_idx, expert_idx);
  if (layer_idx == -1 || expert_idx == -1) {
    return false;
  }

  size_t block_n = new_deepseek_v3_config->quant_config.weight_block_size[0];
  size_t block_k = new_deepseek_v3_config->quant_config.weight_block_size[1];
  size_t moe_inter_size_per_rank = DivRoundUp(
    new_deepseek_v3_config->moe_config.moe_inter_size,
    new_deepseek_v3_config->moe_tensor_para_size);
  if (moe_inter_size_per_rank % block_n != 0) {
    KLLM_THROW(fmt::format(
        "The moe_inter_size_per_rank of gate's and up's weight = {}, is not divisible by weight quant block_n = {}",
        moe_inter_size_per_rank, block_n));
  }
  if (context_->GetTensorParallelSize() > 1 && moe_inter_size_per_rank % block_k != 0) {
    KLLM_THROW(fmt::format(
        "The moe_inter_size_per_rank of down's weight = {}, is not divisible by weight quant block_k = {}",
        moe_inter_size_per_rank, block_k));
  }
  size_t hidden_units = new_deepseek_v3_config->hidden_units;
  std::vector<size_t> up_gate_experts_scale_shape = {
    new_deepseek_v3_config->moe_config.num_experts,
    static_cast<size_t>(DivRoundUp(moe_inter_size_per_rank, block_n) * 2),
    static_cast<size_t>(DivRoundUp(hidden_units, block_k))};
  std::vector<size_t> down_experts_scale_shape = {
    new_deepseek_v3_config->moe_config.num_experts,
    static_cast<size_t>(DivRoundUp(hidden_units, block_n)),
    static_cast<size_t>(DivRoundUp(moe_inter_size_per_rank, block_k))};
  // For up_gate proj scale
  if (host_weight_name.find("up_proj.weight_scale") != std::string::npos ||
      host_weight_name.find("gate_proj.weight_scale") != std::string::npos) {
    if (host_weight_tensor.shape[0] != DivRoundUp(new_deepseek_v3_config->moe_config.moe_inter_size, block_n)) {
      KLLM_THROW(fmt::format("Not support shape of scale: {}", host_weight_name));
    }
    std::string up_gate_experts_scale_name =
        "model.layers." + std::to_string(layer_idx) + ".mlp.experts.up_gate_proj.weight_scale_inv";
    if (device_model_weights.find(up_gate_experts_scale_name) == device_model_weights.end()) {
      device_model_weights[up_gate_experts_scale_name] = Tensor(MemoryLocation::LOCATION_DEVICE,
        DataType::TYPE_FP32, up_gate_experts_scale_shape, dev_rank);
    }

    size_t expert_scale_pitch = up_gate_experts_scale_shape[1] / 2 *
      up_gate_experts_scale_shape[2] * GetTypeSize(host_weight_tensor.dtype);
    size_t double_expert_scale_pitch = expert_scale_pitch * 2;
    size_t src_upgate_offset =
      new_deepseek_v3_config->moe_tensor_para_size > 1 ? dev_rank * expert_scale_pitch : 0;
    if (host_weight_name.find(".gate_proj.") != std::string::npos) {
      MemcpyAsync(device_model_weights.at(up_gate_experts_scale_name).GetPtr<void>() +
                  static_cast<size_t>(expert_idx) * double_expert_scale_pitch,
                  host_weight_tensor.GetPtr<void>() + src_upgate_offset, expert_scale_pitch, MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[dev_rank]);
    } else if (host_weight_name.find(".up_proj.") != std::string::npos) {
      MemcpyAsync(device_model_weights.at(up_gate_experts_scale_name).GetPtr<void>() +
                  static_cast<size_t>(expert_idx) * double_expert_scale_pitch + expert_scale_pitch,
                  host_weight_tensor.GetPtr<void>() + src_upgate_offset, expert_scale_pitch, MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[dev_rank]);
    }
  }
  if (host_weight_name.find(".down_proj.weight_scale") != std::string::npos) {
    std::string down_experts_scale_name =
      "model.layers." + std::to_string(layer_idx) + ".mlp.experts.down_proj.weight_scale_inv";
    if (device_model_weights.find(down_experts_scale_name) == device_model_weights.end()) {
      device_model_weights[down_experts_scale_name] = Tensor(MemoryLocation::LOCATION_DEVICE,
          DataType::TYPE_FP32, down_experts_scale_shape, dev_rank);
    }

    size_t dst_pitch = down_experts_scale_shape[2] * GetTypeSize(host_weight_tensor.dtype);
    size_t src_pitch = down_experts_scale_shape[2] *
      new_deepseek_v3_config->moe_tensor_para_size * GetTypeSize(host_weight_tensor.dtype);
    size_t expert_scale_pitch =
      down_experts_scale_shape[2] * down_experts_scale_shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t src_down_offset = dev_rank * dst_pitch;

    Memcpy2DAsync(device_model_weights.at(down_experts_scale_name).GetPtr<void>() +
                  static_cast<size_t>(expert_idx) * expert_scale_pitch,
                  dst_pitch, host_weight_tensor.GetPtr<void>() + src_down_offset,
                  src_pitch, dst_pitch, down_experts_scale_shape[1],
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  }
  return true;
}

bool NewDeepSeekV3WeightLoader::LoadMlaFp8E4m3BlockWiseScale(const std::string & host_weight_name,
                                                             const Tensor & host_weight_tensor,
                                                             int dev_rank,
                                                             std::shared_ptr<NewDeepSeekV3Config> &
                                                                new_deepseek_v3_config,
                                                             std::unordered_map<std::string, Tensor> &
                                                                device_model_weights) {
  // q_a_proj：权重不拆分，复制到各个卡上
  // q_b_proj: 权重拆分，需先反量化，再拆分，再进行量化， 再分卡到每张卡上
  // kv_a_proj：复制到各个卡上，各卡上直接拆分，不需要反量化
  // kv_b_proj: 权重拆分，需先反量化，再拆分，再进行量化， 再分卡到每张卡上
  if (new_deepseek_v3_config->quant_config.method != QUANT_FP8_E4M3 &&
      !new_deepseek_v3_config->quant_config.is_fp8_blockwise) {
    return false;
  }
  if (host_weight_name.find(".self_attn.") == std::string::npos ||
      (host_weight_name.find(".weight_scale") == std::string::npos &&
      host_weight_name.find(".input_scale") == std::string::npos)) {
    return false;
  }
  // scale is float scalar
  if (host_weight_tensor.dtype != TYPE_FP32) {
    KLLM_THROW("Not support data type of scale:" + host_weight_name);
  }

  if (host_weight_name.find(".q_a_proj.weight_scale_inv") != std::string::npos) {
    // Weights are not split and are copied to each GPU.
    Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
      host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

    MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(),
                host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[host_weight_name] = dev_tensor;
  }
  // For q_b_proj scale
  if (host_weight_name.find(".q_b_proj.weight_scale_inv") != std::string::npos) {
    size_t para_pitch =
        static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize())) *
        host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t tensor_para_offset = dev_rank * para_pitch;
    std::vector<size_t> q_b_proj_scale_shape = {
      static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0],
        context_->GetTensorParallelSize())), host_weight_tensor.shape[1]
    };

    Tensor weight_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
      DataType::TYPE_FP32, q_b_proj_scale_shape, dev_rank);
    MemcpyAsync(weight_scale_tensor.GetPtr<void>(),
                host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                weight_scale_tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[host_weight_name] = weight_scale_tensor;
  }

  // For kv_a_proj scale
  if (host_weight_name.find(".kv_a_proj_with_mqa.weight_scale_inv") != std::string::npos) {
    // For kv_a_lora_proj scale
    size_t kv_lora_rank = new_deepseek_v3_config->mla_config.kv_lora_rank;
    if (kv_lora_rank % new_deepseek_v3_config->quant_config.weight_block_size[0] != 0) {
      KLLM_THROW("Not support shape of scale:" + host_weight_name);
    }
    std::string kv_a_lora_scale_name =
        host_weight_name.substr(0, host_weight_name.find_first_of('_')) + "_attn.kv_a_lora_proj.weight_scale_inv";
    size_t kv_a_lora_scale_shape_0 = kv_lora_rank / new_deepseek_v3_config->quant_config.weight_block_size[0];
    std::vector<size_t> kv_a_lora_scale_shape = { kv_a_lora_scale_shape_0, host_weight_tensor.shape[1]};
    Tensor kv_a_lora_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
      DataType::TYPE_FP32, kv_a_lora_scale_shape, dev_rank);
    MemcpyAsync(kv_a_lora_scale_tensor.GetPtr<void>(),
                host_weight_tensor.GetPtr<void>(),
                kv_a_lora_scale_tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[kv_a_lora_scale_name] = kv_a_lora_scale_tensor;

    // For kv_a_rope_proj scale
    std::string kv_a_rope_scale_name =
        host_weight_name.substr(0, host_weight_name.find_first_of('_')) + "_attn.kv_a_rope_proj.weight_scale_inv";
    size_t qk_rope_head_dim = new_deepseek_v3_config->mla_config.qk_rope_head_dim;
    size_t kv_a_rope_scale_shape_0 = DivRoundUp(qk_rope_head_dim,
      new_deepseek_v3_config->quant_config.weight_block_size[0]);
    if (kv_a_rope_scale_shape_0 + kv_a_lora_scale_shape_0 != host_weight_tensor.shape[0]) {
      KLLM_THROW("Not support shape of scale:" + host_weight_name);
    }
    std::vector<size_t> kv_a_rope_scale_shape = {kv_a_rope_scale_shape_0, host_weight_tensor.shape[1]};
    Tensor kv_a_rope_scale_tesnor = Tensor(MemoryLocation::LOCATION_DEVICE,
      DataType::TYPE_FP32, kv_a_rope_scale_shape, dev_rank);
    MemcpyAsync(kv_a_rope_scale_tesnor.GetPtr<void>(),
                host_weight_tensor.GetPtr<void>() + kv_a_lora_scale_tensor.GetTotalBytes(),
                kv_a_rope_scale_tesnor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[kv_a_rope_scale_name] = kv_a_rope_scale_tesnor;
  }

  // for kv_b_proj_scale
  if (host_weight_name.find(".kv_b_proj.weight_scale_inv") != std::string::npos) {
    size_t para_pitch =
        DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize()) *
        host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t tensor_para_offset = dev_rank * para_pitch;
    std::vector<size_t> kv_b_proj_scale_shape = {
        static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize())),
        host_weight_tensor.shape[1]
    };

    Tensor weight_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
      DataType::TYPE_FP32, kv_b_proj_scale_shape, dev_rank);
    MemcpyAsync(weight_scale_tensor.GetPtr<void>(),
                host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                weight_scale_tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[host_weight_name] = weight_scale_tensor;
  }

  if (host_weight_name.find(".o_proj.weight_scale_inv") != std::string::npos) {
    // fp8: Transpose, then split along axis = 0, then transpose
    Tensor dev_tensor;
    TransposeSplitWithFp8Adjustment(host_weight_tensor,
      dev_tensor, dev_rank, new_deepseek_v3_config, new_deepseek_v3_config->is_quant);

    device_model_weights[host_weight_name] = dev_tensor;
  }
  return true;
}

#endif

Status NewDeepSeekV3WeightLoader::GetExpertsIdx(const std::string & expert_name,
                                                size_t & layer_idx_,
                                                size_t & expert_idx_) {
  // Get the index of the moe layer and the index of each expert
  std::regex re(R"(\d+)");
  std::sregex_iterator next(expert_name.begin(), expert_name.end(), re);
  std::sregex_iterator end;
  if (next != end) {
    std::smatch match = *next;
    layer_idx_ = std::stoi(match.str());
    next++;
    match = *next;
    expert_idx_ = std::stoi(match.str());
  }
  return Status();
}

Status NewDeepSeekV3WeightLoader::ProcessGateUpProjWeight(std::string& file_weight_name_,
                                                              const Tensor& dev_tensor,
                                                              std::unordered_map<std::string, Tensor>&
                                                                device_model_weights,
                                                              int dev_rank,
                                                              std::unordered_set<std::string>&
                                                                processed_weights) {
  int concat_offset = 0;
  std::string replacement = "gate_up_proj";
  if (file_weight_name_.find("gate_proj") != std::string::npos) {
      concat_offset = 0;
      std::regex pattern("gate_proj");
      file_weight_name_ = std::regex_replace(file_weight_name_, pattern, replacement);
  } else {
      concat_offset = 1;
      std::regex pattern("up_proj");
      file_weight_name_ = std::regex_replace(file_weight_name_, pattern, replacement);
  }

  if (device_model_weights.find(file_weight_name_) == device_model_weights.end()) {
      device_model_weights[file_weight_name_] = Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype,
          {dev_tensor.shape[0] * 2, dev_tensor.shape[1]}, dev_rank);
  }
  Tensor& gate_up_proj_tensor = device_model_weights[file_weight_name_];
  size_t total_bytes = gate_up_proj_tensor.GetTotalBytes() / 2;
  MemcpyAsync(gate_up_proj_tensor.GetPtr<void>() + concat_offset * total_bytes,
      dev_tensor.GetPtr<void>(), total_bytes, MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);

  if (file_weight_name_.find("_scale_inv") == std::string::npos) {
      processed_weights.insert(file_weight_name_);
  }
  return Status();
}

Status NewDeepSeekV3WeightLoader::InitQuantWeightLoader(
        std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config) {
  switch (new_deepseek_v3_config->weight_data_type) {
    case DataType::TYPE_FP32:
      quant_weight_ = std::make_unique<NewDeepSeekV3QuantWeight<float>>();
      break;
#ifdef ENABLE_BFLOAT16
    case DataType::TYPE_BF16:
      quant_weight_ = std::make_unique<NewDeepSeekV3QuantWeight<bfloat16>>();
      break;
#endif
    case DataType::TYPE_FP16:
      quant_weight_ = std::make_unique<NewDeepSeekV3QuantWeight<float16>>();
      break;
    default:
      // Handle unexpected data type if needed
      KLLM_THROW(fmt::format("Unexpected data type: {}", new_deepseek_v3_config->weight_data_type));
      break;
  }
  return Status();
}
}  // namespace ksana_llm
