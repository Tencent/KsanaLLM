/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM
 * Copyright (c) 2024, Tencent Inc.  All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ksana_llm/models/quant/quant_weight.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/nn/functional/normalization.h>

#include <regex>

#include "nlohmann/json.hpp"

#include "ksana_llm/utils/common_device.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/kernels/permute.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

#define NameReplace(ss, src, dst) std::regex_replace((ss), std::regex((src)), (dst))

template <typename T>
QuantWeight<T>::QuantWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config_, int rank,
                            std::shared_ptr<Context> context, std::unordered_map<std::string, Tensor>& weights_map,
                            std::unordered_map<std::string, DataType>& weights_data_type_map)
    : weights_map_(weights_map),
      weights_data_type_map_(weights_data_type_map),
      rank_(rank),
      context_(context),
      model_config_(model_config),
      runtime_config_(runtime_config_) {
  enable_ = CheckQuantModel();
  tensor_manager_ = std::make_shared<TensorManager>(rank, weights_map_);
  tensor_para_size_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  expert_world_size_ = runtime_config_.parallel_basic_config.expert_world_size;
  expert_para_size_ = runtime_config_.parallel_basic_config.expert_parallel_size;
  global_expert_para_size_ = expert_world_size_ * expert_para_size_;
  weight_data_type_ = model_config.weight_data_type;
  enable_full_shared_expert_ = runtime_config_.enable_full_shared_expert;

  cutlass_helper_ = std::make_shared<CutlassUtils>(context_, rank, model_config_.quant_config.bits);
  marlin_helper_ = std::make_shared<MarlinUtils>(context_, rank, model_config_.quant_config.bits,
                                                 model_config_.quant_config.group_size);
  machete_helper_ = std::make_shared<MacheteUtils>(context_, rank, model_config_.quant_config.bits);

  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config_);

  // extract required layer idx
  for (auto idx = pipeline_config_.lower_layer_idx; idx <= pipeline_config_.upper_layer_idx; ++idx) {
    required_layer_idx_.all.emplace(idx);
  }
  KLLM_LOG_INFO << fmt::format("QuantWeight IsChief:{}, layer:[{}, {}].", context_->IsChief(),
                               pipeline_config_.lower_layer_idx, pipeline_config_.upper_layer_idx);

  // add nextn predict layers
  if (static_cast<int>(pipeline_config_.lower_nextn_layer_idx) >= static_cast<int>(model_config_.num_layer)) {
    for (auto idx = pipeline_config_.lower_nextn_layer_idx; idx <= pipeline_config_.upper_nextn_layer_idx; ++idx) {
      required_layer_idx_.all.emplace(idx);
    }
    KLLM_LOG_INFO << fmt::format("QuantWeight IsChief:{}, nextn layer:[{}, {}].", context_->IsChief(),
                                 pipeline_config_.lower_nextn_layer_idx, pipeline_config_.upper_nextn_layer_idx);
  }

  // extract dense and moe layer idx
  std::vector<size_t>& moe_layers = model_config_.moe_config.moe_layers;
  for (const auto idx : required_layer_idx_.all) {
    if (model_config_.is_moe && idx >= model_config_.moe_config.first_k_dense_replace &&
        (moe_layers.empty() || std::find(moe_layers.begin(), moe_layers.end(), idx) != moe_layers.end())) {
      required_layer_idx_.moe.emplace(idx);
    } else {
      required_layer_idx_.dense.emplace(idx);
    }
  }

  // init expert map
  // TODO(zezhao) 与 common_moe_weight.cpp 复用一份
  ExpertParallelConfig expert_parallel_config;
  Singleton<Environment>::GetInstance()->GetExpertParallelConfig(expert_parallel_config);
  size_t expert_node_rank = expert_parallel_config.expert_node_rank;
  size_t num_experts = model_config_.moe_config.num_experts;
  num_experts_per_rank_ = (num_experts + global_expert_para_size_ - 1) / global_expert_para_size_;
  expert_map_ = std::vector<int>(num_experts, -1);
  size_t rank_expert_offset = expert_node_rank * expert_para_size_ * num_experts_per_rank_;
  size_t expert_offset = (global_expert_para_size_ > 1) ? ((rank_ % expert_para_size_) * num_experts_per_rank_) : 0;
  size_t expert_start_id = rank_expert_offset + expert_offset;
  size_t expert_end_id = std::min(num_experts, expert_start_id + num_experts_per_rank_);
  for (size_t i = expert_start_id; i < expert_end_id; ++i) {
    expert_map_[i] = static_cast<int>(i - expert_start_id);
  }
  KLLM_LOG_DEBUG << fmt::format("In quant_weight.cpp Rank {} valid expert range is from {} to {}", rank_,
                                expert_start_id, expert_end_id - 1);
}

template <typename T>
QuantWeight<T>::~QuantWeight() {}

template <typename T>
bool QuantWeight<T>::IsEnable() {
  return enable_;
}

template <typename T>
bool QuantWeight<T>::FilterOutQuantWeight(const std::string& tensor_name) {
  if (!enable_) {
    return false;
  }

  std::vector<std::string> skip_lists = {".o_proj.bias", ".gate_proj.bias", ".up_proj.bias", ".down_proj.bias"};
  if (model_config_.quant_config.method == QUANT_GPTQ) {
    skip_lists.push_back(".qzeros");
  }
  if (model_config_.quant_config.desc_act != true || model_config_.quant_config.method == QUANT_AWQ) {
    skip_lists.push_back(".g_idx");
  }
  for (const std::string& skip : skip_lists) {
    if (tensor_name.find(skip) != std::string::npos) {
      return true;
    }
  }
  return false;
}

template <typename T>
bool QuantWeight<T>::CheckQuantModel() {
  // TODO(jinxcwu): make a struct to store different quant type: gptq, awq, ...
  if (model_config_.is_quant) {
    if (model_config_.quant_config.method == QUANT_GPTQ) {
      return true;
    }
    if (model_config_.quant_config.method == QUANT_AWQ) {
      return true;
    }
    if (model_config_.quant_config.method == QUANT_FP8_E4M3) {
      if (context_->IsGemmFp8Supported()) {
        KLLM_LOG_INFO << "Device is sufficient to support FP8 GEMM.";
      } else {
        KLLM_THROW("Device is insufficient to support FP8 GEMM.");
      }
    }
  }
  return false;
}

template <typename T>
bool QuantWeight<T>::IsClaLayer(const int layer_idx) {
  if (model_config_.use_cla && model_config_.cla_share_factor != 0 &&
      (layer_idx % model_config_.cla_share_factor != 0)) {
    return true;
  }
  return false;
}

// Only for fuse_moe_gptq_awq triton op and marlin_moe op
template <typename T>
void QuantWeight<T>::LoadMoeIntQuantWeight(const std::string& tensor_name, std::vector<size_t>& weight_shape,
                                           DataType& weight_data_type, void* weight_ptr) {
  SetDevice(rank_);
  size_t moe_inter_size = model_config_.moe_config.moe_inter_size;
  size_t moe_inter_size_per_rank =
      DivRoundUp(moe_inter_size, runtime_config_.parallel_basic_config.moe_tensor_para_size);
  size_t hidden_units = model_config_.hidden_units;
  size_t pack_factor = 32 / model_config_.quant_config.bits;
  size_t group_size = model_config_.quant_config.group_size;
  int layer_idx = -1, expert_idx = -1;
  GetExpertsScaleIdx(tensor_name, layer_idx, expert_idx);
  if (expert_map_[expert_idx] < 0) {
    // Skip load weight when the expert_id will be not used in current rank.
    return;
  }
  expert_idx = expert_map_[expert_idx];
  int tp_rank = (runtime_config_.parallel_basic_config.moe_tensor_para_size > 1 ? rank_ / expert_para_size_ : 0);
  if (!model_config_.use_mla && !model_config_.moe_config.use_vllm_moe &&
      tensor_name.find(".g_idx") != std::string::npos) {
    if (tensor_name.find("up_proj") != std::string::npos || tensor_name.find("gate_proj") != std::string::npos) {
      // model.layers.layer_idx.mlp.experts.expert_idx.up/gate_proj.g_idx shape is {hidden_units}
      // model.layers.mlp.experts.up_gate_proj.g_idx shape is {num_experts_per_rank_, hidden_units}
      std::string name = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.up_gate_proj.g_idx";
      std::vector<size_t> shape = {static_cast<size_t>(num_experts_per_rank_), weight_shape[0]};
      KLLM_LOG_DEBUG << "copying weight to:" << name;
      if (weights_map_.find(name) == weights_map_.end()) {
        tensor_manager_->AddWeightTensor(name, shape, weight_data_type);
      }
      size_t size = shape[1] * GetTypeSize(weight_data_type);
      void* dst_ptr = weights_map_[name].GetPtr<void>() + expert_idx * size;
      // up and gate use the same g_idx for all ranks
      MemcpyAsync(dst_ptr, weight_ptr, size, MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else if (tensor_name.find("down_proj") != std::string::npos) {
      // model.layers.layer_idx.mlp.experts.expert_idx.down_proj.g_idx shape is {moe_inter_size}
      // model.layers.mlp.experts.down_proj.g_idx shape is {num_experts_per_rank_, moe_inter_size_per_rank}
      std::string name = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.down_proj.g_idx";
      std::vector<size_t> shape = {static_cast<size_t>(num_experts_per_rank_), moe_inter_size_per_rank};
      if (weights_map_.find(name) == weights_map_.end()) {
        tensor_manager_->AddWeightTensor(name, shape, weight_data_type);
      }
      size_t size = shape[1] * GetTypeSize(weight_data_type);
      void* src_ptr = weight_ptr + tp_rank * size;
      void* dst_ptr = weights_map_[name].GetPtr<void>() + expert_idx * size;
      MemcpyAsync(dst_ptr, src_ptr, size, MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else {
      KLLM_THROW(fmt::format("Not support weight : {}", tensor_name));
    }
    StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
    return;
  }

  // TODO(winminkong): support awq moe
  if (tensor_name.find(".qweight") != std::string::npos || tensor_name.find(".scales") != std::string::npos) {
    bool is_qweight = false;
#ifdef ENABLE_CUDA
    if (weight_ptr == nullptr) {
      KLLM_LOG_ERROR << fmt::format("The {}'s weight_ptr is null", tensor_name);
    }
    // Preprocess, transpose and cast qweight tensor type from int32 to uint8
    torch::Tensor tensor = GetTorchTensorFromWeightPtr(weight_shape, weight_data_type, weight_ptr, false);

    tensor = tensor.t().contiguous();
    if (tensor_name.find(".qweight") != std::string::npos) {
      is_qweight = true;
      tensor = tensor.view(torch::kUInt8);
    }

    DataType processed_tensor_type = GetDataTypeFromTorchType(tensor.scalar_type());
    // Postprocess, split tensor for TP and copy to cuda device
    if (tensor_name.find(".gate_proj.") != std::string::npos || tensor_name.find(".up_proj.") != std::string::npos) {
      if (tensor.sizes()[0] != moe_inter_size || tensor.strides()[1] != 1) {
        KLLM_THROW(fmt::format("The weight named {} transpose failed.", tensor_name));
      }
      if (is_qweight &&
          (tensor.sizes()[1] != (hidden_units / pack_factor * 4) || processed_tensor_type != TYPE_UINT8)) {
        KLLM_THROW(fmt::format("The weight named {} cast data type failed.", tensor_name));
      }
      std::vector<size_t> up_gate_experts_shape = {num_experts_per_rank_, moe_inter_size_per_rank * 2,
                                                   static_cast<size_t>(tensor.sizes()[1])};
      std::string up_gate_experts_name =
          fmt::format("model.layers.{}.mlp.experts.up_gate_proj.{}", layer_idx, is_qweight ? "weight" : "scales");
      if (weights_map_.find(up_gate_experts_name) == weights_map_.end()) {
        tensor_manager_->AddWeightTensor(up_gate_experts_name, up_gate_experts_shape, processed_tensor_type);
        weights_data_type_map_[up_gate_experts_name] = processed_tensor_type;
      }

      size_t expert_pitch =
          moe_inter_size_per_rank * static_cast<size_t>(tensor.sizes()[1]) * GetTypeSize(processed_tensor_type);
      size_t double_expert_pitch = expert_pitch * 2;
      size_t src_upgate_offset = runtime_config_.parallel_basic_config.moe_tensor_para_size > 1
                                     ? (rank_ / expert_para_size_) * expert_pitch
                                     : 0;
      Tensor& up_gate_experts_tensor = weights_map_[up_gate_experts_name];
      if (tensor_name.find(".gate_proj.") != std::string::npos) {
        MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + expert_idx * double_expert_pitch,
                    tensor.data_ptr() + src_upgate_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      } else if (tensor_name.find(".up_proj.") != std::string::npos) {
        MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + expert_idx * double_expert_pitch + expert_pitch,
                    tensor.data_ptr() + src_upgate_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      }
    }

    if (tensor_name.find(".down_proj.") != std::string::npos) {
      if (tensor.sizes()[0] != hidden_units || tensor.strides()[1] != 1) {
        KLLM_THROW(fmt::format("The weight named {} transpose failed.", tensor_name));
      }
      if (is_qweight && (tensor.sizes()[1] != (moe_inter_size / pack_factor * 4))) {
        KLLM_THROW(fmt::format("The weight named {} cast data type failed.", tensor_name));
      }

      size_t down_inter_size_per_rank = DivRoundUp(static_cast<size_t>(tensor.sizes()[1]),
                                                   runtime_config_.parallel_basic_config.moe_tensor_para_size);
      std::vector<size_t> down_experts_shape = {num_experts_per_rank_, hidden_units, down_inter_size_per_rank};
      std::string down_experts_name =
          fmt::format("model.layers.{}.mlp.experts.down_proj.{}", layer_idx, is_qweight ? "weight" : "scales");
      if (weights_map_.find(down_experts_name) == weights_map_.end()) {
        tensor_manager_->AddWeightTensor(down_experts_name, down_experts_shape, processed_tensor_type);
        weights_data_type_map_[down_experts_name] = processed_tensor_type;
      }

      size_t dst_pitch = down_inter_size_per_rank * GetTypeSize(processed_tensor_type);
      size_t src_pitch = static_cast<size_t>(tensor.sizes()[1]) * GetTypeSize(processed_tensor_type);
      size_t expert_pitch = down_inter_size_per_rank * hidden_units * GetTypeSize(processed_tensor_type);
      size_t src_down_offset =
          runtime_config_.parallel_basic_config.moe_tensor_para_size > 1 ? (rank_ / expert_para_size_) * dst_pitch : 0;
      Tensor& down_experts_tensor = weights_map_[down_experts_name];
      Memcpy2DAsync(down_experts_tensor.GetPtr<void>() + expert_idx * expert_pitch, dst_pitch,
                    tensor.data_ptr() + src_down_offset, src_pitch, dst_pitch, hidden_units, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
    }
    KLLM_LOG_DEBUG << fmt::format("Success load weight:{} on rank {}", tensor_name, rank_);
#else
    KLLM_LOG_DEBUG << fmt::format("Not support load moe weight:{}", tensor_name);
#endif
  }
}

template <typename T>
bool QuantWeight<T>::LoadQuantWeight(const std::string& tensor_name, std::vector<size_t>& weight_shape,
                                     DataType& weight_data_type, void* weight_ptr) {
  if (!enable_) {
    return false;
  }

#ifdef ENABLE_CUDA
  int tp = tensor_para_size_;
  int slice_pos = rank_;
  size_t attn_data_parallel_size = runtime_config_.parallel_basic_config.attn_data_parallel_size;
  if (model_config_.quant_config.desc_act == true && tensor_name.find(".scales") != std::string::npos) {
    tp = 1;
    slice_pos = 0;
  }
  if (tensor_name.find(".shared_expert.") != std::string::npos && enable_full_shared_expert_) {
    auto options = torch::TensorOptions().device(torch::kCPU).dtype(GetTorchTypeFromDataType(weight_data_type));
    torch::Tensor tensor =
        torch::from_blob(weight_ptr, std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), options);
    AddWeightFromTorchTensor(tensor_name, tensor);
  } else if (tensor_name.find(".g_idx") != std::string::npos) {
    torch::Tensor tensor = GetTorchTensorFromWeightPtr(weight_shape, weight_data_type, weight_ptr, true);
    if (tensor_name.find(".experts.") != std::string::npos && !model_config_.use_mla &&
        !model_config_.moe_config.use_vllm_moe) {
      LoadMoeIntQuantWeight(tensor_name, weight_shape, weight_data_type, weight_ptr);
    } else if (tensor_name.find("W_pack") != std::string::npos) {
      AddWeightFromTorchTensor(NameReplace(tensor_name, "W_pack", "q_proj"), tensor);
      AddWeightFromTorchTensor(NameReplace(tensor_name, "W_pack", "k_proj"), tensor);
      AddWeightFromTorchTensor(NameReplace(tensor_name, "W_pack", "v_proj"), tensor);
    } else if (model_config_.type == "chatglm" && tensor_name.find("gate_proj") != std::string::npos) {
      AddWeightFromTorchTensor(tensor_name, tensor);
      AddWeightFromTorchTensor(NameReplace(tensor_name, "gate", "up"), tensor);
    } else if (tensor_name.find("o_proj") != std::string::npos || tensor_name.find("down_proj") != std::string::npos) {
      size_t single_size = tensor.size(0) / tensor_para_size_;
      int inner_group_rank = rank_;
      if (attn_data_parallel_size > 1) {
        // NOTE(karlluo): for tp + attn_dp, all gpus consist tensor parallel group, attn_data_parallel_size is the
        // number of attn dp groups and conduct tp in each dp groups. For example, if tp = 4, then gpus = 4 and attn_dp
        // = 2, then each attn dp group size is 2.
        single_size = tensor.size(0) / runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
        inner_group_rank = rank_ % runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
      }
      tensor = TpSplitTensor(tensor, 0, inner_group_rank, single_size);
      AddWeightFromTorchTensor(tensor_name, tensor);
    } else {
      AddWeightFromTorchTensor(tensor_name, tensor);
    }
    return true;
  } else if (tensor_name.find(".qweight") != std::string::npos || tensor_name.find(".scales") != std::string::npos ||
             tensor_name.find(".qzeros") != std::string::npos) {
    if (tensor_name.find("W_pack") != std::string::npos) {
      size_t q_proj_size = model_config_.size_per_head * model_config_.head_num;
      size_t kv_proj_size = model_config_.size_per_head * model_config_.num_key_value_heads;

      size_t inner_tensor_para_size = tensor_para_size_;
      size_t attn_data_parallel_size = runtime_config_.parallel_basic_config.attn_data_parallel_size;
      if (attn_data_parallel_size > 1) {
        // NOTE(karlluo): for tp + attn_dp, all gpus consist tensor parallel group, attn_data_parallel_size is the
        // number of attn dp groups and conduct tp in each dp groups. For example, if tp = 4, then gpus = 4 and attn_dp
        // = 2, then each attn dp group size is 2.
        inner_tensor_para_size = runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
      }

      if (q_proj_size % inner_tensor_para_size != 0 || kv_proj_size % inner_tensor_para_size != 0) {
        KLLM_THROW(
            fmt::format("Model can't run with tensor_para_size == {}. "
                        "The size of q_proj_size {} or kv_proj_size {} cannot be evenly divided by the size of "
                        "tensor_parallel_size_.",
                        inner_tensor_para_size, q_proj_size, kv_proj_size));
      }
      torch::Tensor tensor = GetTorchTensorFromWeightPtr(weight_shape, weight_data_type, weight_ptr, true);
      tensor = TrySmartAutoUnpack(tensor_name, tensor);

      size_t s = (q_proj_size + kv_proj_size + kv_proj_size) / tensor.size(1);
      q_proj_size /= s;
      kv_proj_size /= s;
      auto tensors = torch::split(
          tensor,
          {static_cast<int64_t>(q_proj_size), static_cast<int64_t>(kv_proj_size), static_cast<int64_t>(kv_proj_size)},
          1);
      int inner_group_rank = rank_;
      if (attn_data_parallel_size > 1) {
        // NOTE(karlluo): for tp + attn_dp, all gpus consist tensor parallel group, attn_data_parallel_size is the
        // number of attn dp groups and conduct tp in each dp groups. For example, if tp = 4, then gpus = 4 and attn_dp
        // = 2, then each attn dp group size is 2.
        q_proj_size /= runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
        q_proj_size /= runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
        inner_group_rank = rank_ % runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
      } else {
        q_proj_size /= tensor_para_size_;
        kv_proj_size /= tensor_para_size_;
      }
      tensors[0] = TpSplitTensor(tensors[0], 1, inner_group_rank, q_proj_size);
      tensors[1] = TpSplitTensor(tensors[1], 1, inner_group_rank, kv_proj_size);
      tensors[2] = TpSplitTensor(tensors[2], 1, inner_group_rank, kv_proj_size);
      AddWeightFromTorchTensor(NameReplace(tensor_name, "W_pack", "q_proj"), tensors[0], weight_data_type);
      AddWeightFromTorchTensor(NameReplace(tensor_name, "W_pack", "k_proj"), tensors[1], weight_data_type);
      AddWeightFromTorchTensor(NameReplace(tensor_name, "W_pack", "v_proj"), tensors[2], weight_data_type);
    } else if (tensor_name.find(".experts.") != std::string::npos) {
      LoadMoeIntQuantWeight(tensor_name, weight_shape, weight_data_type, weight_ptr);
    } else if (tensor_name.find("o_proj") != std::string::npos) {
      size_t inner_tensor_para_size = tensor_para_size_;
      int inner_group_rank = rank_;
      if (attn_data_parallel_size > 1) {
        if (model_config_.quant_config.desc_act == true && tensor_name.find(".scales") != std::string::npos) {
          inner_tensor_para_size = 1;
          inner_group_rank = 0;
        } else {
          // NOTE(karlluo): for tp + attn_dp, all gpus consist tensor parallel group, attn_data_parallel_size is the
          // number of attn dp groups and conduct tp in each dp groups. For example, if tp = 4, then gpus = 4 and
          // attn_dp = 2, then each attn dp group size is 2.
          inner_tensor_para_size = runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
          inner_group_rank = rank_ % runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
        }
      }

      if (weight_shape[0] % inner_tensor_para_size != 0) {
        KLLM_THROW(
            fmt::format("Model can't run with tensor_para_size == {}."
                        "The size of weight_shape[0] {} cannot be evenly divided by the size of tensor_para_size_",
                        inner_tensor_para_size, weight_shape[0]));
      }
      torch::Tensor tensor = GetTorchTensorFromWeightPtr(weight_shape, weight_data_type, weight_ptr, true);
      tensor = TrySmartAutoUnpack(tensor_name, tensor);

      size_t single_size = tensor.size(0) / inner_tensor_para_size;
      tensor = TpSplitTensor(tensor, 0, inner_group_rank, single_size);
      AddWeightFromTorchTensor(tensor_name, tensor, weight_data_type);
    } else if (tensor_name.find("down_proj") != std::string::npos) {
      if (weight_shape[0] % tp != 0) {
        KLLM_THROW(
            fmt::format("Model can't run with tensor_para_size == {}."
                        "The size of weight_shape[0] {} cannot be evenly divided by the size of tensor_para_size_",
                        tp, weight_shape[0]));
      }
      torch::Tensor tensor = GetTorchTensorFromWeightPtr(weight_shape, weight_data_type, weight_ptr, true);
      tensor = TrySmartAutoUnpack(tensor_name, tensor);

      size_t single_size = tensor.size(0) / tp;
      tensor = TpSplitTensor(tensor, 0, slice_pos, single_size);
      AddWeightFromTorchTensor(tensor_name, tensor, weight_data_type);
    } else {
      if (weight_shape[1] % tensor_para_size_ != 0) {
        KLLM_THROW(
            fmt::format("Model can't run with tensor_para_size == {}."
                        "The size of weight_shape[1] {} cannot be evenly divided by the size of tensor_para_size_",
                        tensor_para_size_, weight_shape[1]));
      }
      torch::Tensor tensor = GetTorchTensorFromWeightPtr(weight_shape, weight_data_type, weight_ptr, true);
      tensor = TrySmartAutoUnpack(tensor_name, tensor);

      if (model_config_.type == "chatglm" && tensor_name.find("gate_proj") != std::string::npos) {
        auto tensors = torch::chunk(tensor, 2, -1);
        const size_t single_size = tensors[0].size(1) / tensor_para_size_;
        torch::Tensor gate_tensor = TpSplitTensor(tensors[0], 1, rank_, single_size);
        torch::Tensor up_tensor = TpSplitTensor(tensors[1], 1, rank_, single_size);
        AddWeightFromTorchTensor(tensor_name, gate_tensor, weight_data_type);
        AddWeightFromTorchTensor(NameReplace(tensor_name, "gate", "up"), up_tensor, weight_data_type);
      } else {
        size_t single_size = tensor.size(1) / tensor_para_size_;
        tensor = TpSplitTensor(tensor, 1, rank_, single_size);
        AddWeightFromTorchTensor(tensor_name, tensor, weight_data_type);
      }
      KLLM_LOG_DEBUG << fmt::format("success load mlp weight : {}", tensor_name);
    }
    return true;
  }
#endif
  return false;
}

#ifdef ENABLE_CUDA
template <typename T>
torch::Tensor QuantWeight<T>::TrySmartAutoUnpack(const std::string& tensor_name, torch::Tensor& tensor) {
  if (model_config_.quant_config.backend != CUTLASS_LINEAR_BACKEND) {
    return tensor;
  }
  if (tensor_name.find(".qweight") != std::string::npos) {
    if (model_config_.quant_config.method == QUANT_GPTQ) {
      tensor = cutlass_helper_->CutlassUnpackGPTQ(tensor);
    } else if (model_config_.quant_config.method == QUANT_AWQ) {
      tensor = cutlass_helper_->CutlassUnpackAWQ(tensor);
    }
    int8_t zero = std::pow(2, model_config_.quant_config.bits - 1);
    tensor = (tensor - zero).contiguous();
    tensor = cutlass_helper_->CutlassPackInt8ToPackedInt4(tensor);
  }
  if (tensor_name.find(".qzeros") != std::string::npos && model_config_.quant_config.method == QUANT_AWQ) {
    tensor = cutlass_helper_->CutlassUnpackAWQ(tensor);
    tensor = tensor.to(torch::kHalf);
  }
  return tensor;
}
#endif

template <typename T>
Status QuantWeight<T>::PackAndBindGroupTensor(int layer_idx, const std::string& needed_slove_weight_name) {
  std::string qweight_name = fmt::format("model.layers.{}.{}.qweight", layer_idx, needed_slove_weight_name);
  std::string scales_name = fmt::format("model.layers.{}.{}.scales", layer_idx, needed_slove_weight_name);
  std::string zeros_name = fmt::format("model.layers.{}.{}.zeros", layer_idx, needed_slove_weight_name);
  std::string gidx_name = fmt::format("model.layers.{}.{}.g_idx", layer_idx, needed_slove_weight_name);
  std::string perm_name = fmt::format("model.layers.{}.{}.perm", layer_idx, needed_slove_weight_name);
  std::string weight_name = fmt::format("model.layers.{}.{}.weight", layer_idx, needed_slove_weight_name);
  if (weights_map_.find(qweight_name) == weights_map_.end() && weights_map_.find(weight_name) == weights_map_.end()) {
    KLLM_LOG_WARNING << fmt::format("Process quant weight: {}, but not found", qweight_name);
    return Status();
  }
#ifdef ENABLE_CUDA
  SetDevice(rank_);
  KLLM_LOG_DEBUG << fmt::format("Starting bind quant weight {} on rank: {}", qweight_name, rank_);
  bool is_experts = (needed_slove_weight_name.find("mlp.experts") != std::string::npos);
  if (!model_config_.use_mla && !model_config_.moe_config.use_vllm_moe || !is_experts) {
    if (model_config_.quant_config.desc_act == true) {
      torch::Tensor gidx_gpu = GetTorchTensorFromWeight(gidx_name).clone();
      torch::Tensor perm_gpu = marlin_helper_->MarlinSortGIdx(gidx_gpu);
      weights_map_.erase(gidx_name);
      AddWeightFromTorchTensor(gidx_name, gidx_gpu);
      AddWeightFromTorchTensor(perm_name, perm_gpu);
    }
    torch::Tensor qweight_gpu = GetTorchTensorFromWeight(qweight_name);
    if (model_config_.quant_config.backend == CUTLASS_LINEAR_BACKEND && !is_experts) {
      torch::Tensor processed_tensor_gpu = cutlass_helper_->CutlassPreprocessWeightsForMixedGemmWarpper(
          qweight_gpu, llm_kernels::nvidia::QuantType::W4_A16);
      AddWeightFromTorchTensor(weight_name, processed_tensor_gpu);
    } else if (model_config_.quant_config.backend == MARLIN_LINEAR_BACKEND || is_experts) {
      if (model_config_.quant_config.method == QUANT_GPTQ) {
        std::optional<torch::Tensor> perm_gpu = std::nullopt;
        if (model_config_.quant_config.desc_act == true) {
          perm_gpu.emplace(GetTorchTensorFromWeight(perm_name));
        }
        torch::Tensor processed_tensor_gpu = marlin_helper_->PackGptqWeight(qweight_gpu, perm_gpu);
        AddWeightFromTorchTensor(weight_name, processed_tensor_gpu);
      } else if (model_config_.quant_config.method == QUANT_AWQ) {
        torch::Tensor processed_tensor_gpu = marlin_helper_->PackAwqWeight(qweight_gpu);
        AddWeightFromTorchTensor(weight_name, processed_tensor_gpu);
      } else {
        KLLM_THROW("Unsupported group quant method, only support GPTQ and AWQ.");
      }
    } else if (model_config_.quant_config.backend == MACHETE_LINEAR_BACKEND) {
      torch::Tensor processed_tensor_gpu =
          machete_helper_->PackWeight<T>(qweight_gpu, model_config_.quant_config.method);
      AddWeightFromTorchTensor(weight_name, processed_tensor_gpu);
    } else {
      KLLM_THROW("Unsupported backend for group quant, only support CUTLASS, MARLIN and MACHETE(sm 90).");
    }

    weights_map_.erase(qweight_name);

    // In Marlin, GPTQ and AWQ share the same scale layout
    if (model_config_.quant_config.backend == MARLIN_LINEAR_BACKEND || is_experts) {
      torch::Tensor scales_gpu = GetTorchTensorFromWeight(scales_name);
      scales_gpu = marlin_helper_->MarlinPermuteScales<T>(
          scales_gpu, model_config_.quant_config.group_size * scales_gpu.size(-2), scales_gpu.size(-1));
      weights_map_.erase(scales_name);
      AddWeightFromTorchTensor(scales_name, scales_gpu);
    }
  }
  // binding scales
  weights_map_[weight_name].scales = &weights_map_[scales_name];
  // binding zeros
  if (model_config_.quant_config.method == QUANT_AWQ) {
    weights_map_[weight_name].zeros = &weights_map_[zeros_name];
  }
  // binding g_idx and perm
  if (model_config_.quant_config.desc_act == true) {
    weights_map_[weight_name].g_idx = &weights_map_[gidx_name];
    weights_map_[weight_name].perm = &weights_map_[perm_name];
  }
  KLLM_LOG_DEBUG << fmt::format("Successfully process quant weight name: {} on rank: {}", weight_name, rank_);
  return Status();
#endif
  return Status(RetCode::RET_MODEL_QUANT_FAILED, "Not supported Ascend.");
}

template <typename T>
Status QuantWeight<T>::AutoPackAndBindGroupTensor(std::vector<std::string> needed_slove_weights_name) {
  for (std::string& needed_slove_weight_name : needed_slove_weights_name) {
    int max_layer_idx = pipeline_config_.upper_layer_idx;
    int min_layer_idx = pipeline_config_.lower_layer_idx;
    if (needed_slove_weight_name.find("mlp.") != std::string::npos &&
        needed_slove_weight_name.find("expert") == std::string::npos &&
        model_config_.moe_config.first_k_dense_replace != 0) {
      if ((model_config_.moe_config.first_k_dense_replace - 1) >= pipeline_config_.lower_layer_idx &&
          (model_config_.moe_config.first_k_dense_replace - 1) <= pipeline_config_.upper_layer_idx) {
        max_layer_idx = (model_config_.moe_config.first_k_dense_replace - 1);
      }
      if ((model_config_.moe_config.first_k_dense_replace - 1) < pipeline_config_.lower_layer_idx) {
        continue;
      }
    }
    if (needed_slove_weight_name.find("expert") != std::string::npos &&
        model_config_.moe_config.first_k_dense_replace != 0) {
      if (model_config_.moe_config.first_k_dense_replace >= pipeline_config_.lower_layer_idx &&
          model_config_.moe_config.first_k_dense_replace <= pipeline_config_.upper_layer_idx) {
        min_layer_idx = model_config_.moe_config.first_k_dense_replace;
      }
      if ((model_config_.moe_config.first_k_dense_replace - 1) > pipeline_config_.upper_layer_idx) {
        continue;
      }
    }
    // TODO(winminkong) : 后期改为required_layer_idx_并解耦三个int4后端以及mla和moe权重的处理
    for (int layer_idx = min_layer_idx; layer_idx <= max_layer_idx; ++layer_idx) {
      std::string name = needed_slove_weight_name;
      if (name == "self_attn.query_key_value" && IsClaLayer(layer_idx)) {
        name = "self_attn.q_proj";
      }
      PackAndBindGroupTensor(layer_idx, name);
    }
    if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(model_config_.num_layer)) {
      if (needed_slove_weight_name.find("mlp.") != std::string::npos &&
          needed_slove_weight_name.find("expert") == std::string::npos &&
          model_config_.moe_config.first_k_dense_replace != 0) {
        continue;
      }
      for (int layer_idx = pipeline_config_.lower_nextn_layer_idx; layer_idx <= pipeline_config_.upper_nextn_layer_idx;
           ++layer_idx) {
        std::string name = needed_slove_weight_name;
        if (name == "self_attn.query_key_value" && IsClaLayer(layer_idx)) {
          name = "self_attn.q_proj";
        }
        PackAndBindGroupTensor(layer_idx, name);
      }
    }
  }
  return Status();
}

template <typename T>
Status QuantWeight<T>::ConvertGroupTensor() {
  if (!enable_) {
    return Status();
  }

#ifdef ENABLE_CUDA
  SetDevice(rank_);

  bool use_mla = model_config_.use_mla;
  // pack q, k, v to qkv
  std::vector<std::string> needed_slove_weights_name = {"qweight", "scales"};
  if (model_config_.quant_config.method == QUANT_AWQ) {
    needed_slove_weights_name.push_back("qzeros");
  }
  if (model_config_.quant_config.desc_act == true) {
    needed_slove_weights_name.push_back("g_idx");
  }
  if (!use_mla) {
    for (std::string& needed_slove_weight_name : needed_slove_weights_name) {
      for (const auto layer_idx : required_layer_idx_.all) {
        if (IsClaLayer(layer_idx)) continue;
        std::string q_name = fmt::format("model.layers.{}.self_attn.q_proj.{}", layer_idx, needed_slove_weight_name);
        std::string k_name = fmt::format("model.layers.{}.self_attn.k_proj.{}", layer_idx, needed_slove_weight_name);
        std::string v_name = fmt::format("model.layers.{}.self_attn.v_proj.{}", layer_idx, needed_slove_weight_name);
        std::string qkv_name =
            fmt::format("model.layers.{}.self_attn.query_key_value.{}", layer_idx, needed_slove_weight_name);

        torch::Tensor q_tensor_gpu = GetTorchTensorFromWeight(q_name);
        torch::Tensor k_tensor_gpu = GetTorchTensorFromWeight(k_name);
        torch::Tensor v_tensor_gpu = GetTorchTensorFromWeight(v_name);
        torch::Tensor qkv_tensor_gpu = torch::cat({q_tensor_gpu, k_tensor_gpu, v_tensor_gpu}, -1);
        if (model_config_.quant_config.desc_act == true && needed_slove_weight_name == "g_idx") {
          qkv_tensor_gpu = q_tensor_gpu;
        }

        AddWeightFromTorchTensor(qkv_name, qkv_tensor_gpu);
        weights_map_.erase(q_name);
        weights_map_.erase(k_name);
        weights_map_.erase(v_name);
      }
    }
  }

  if (model_config_.is_moe && !model_config_.use_mla && !model_config_.moe_config.use_vllm_moe) {
    std::vector<std::string> needed_slove_weights_name = {"up_gate_proj.weight", "up_gate_proj.scales",
                                                          "down_proj.weight", "down_proj.scales"};
    for (std::string& needed_slove_weight_name : needed_slove_weights_name) {
      for (const auto layer_idx : required_layer_idx_.all) {
        std::string tensor_name = fmt::format("model.layers.{}.mlp.experts.{}", layer_idx, needed_slove_weight_name);
        KLLM_LOG_DEBUG << "transpose(1,2) of " << tensor_name;
        std::string new_name = tensor_name;
        torch::Tensor torch_tensor_gpu = GetTorchTensorFromWeight(tensor_name);
        if (tensor_name.find(".weight") != std::string::npos) {
          torch_tensor_gpu = torch_tensor_gpu.view(torch::kInt32);
          new_name.replace(tensor_name.length() - 6, 6, "qweight");
        }
        torch_tensor_gpu = torch_tensor_gpu.transpose(1, 2).contiguous();
        std::string trans_name = new_name + ".trans";
        AddWeightFromTorchTensor(trans_name, torch_tensor_gpu);
        weights_map_.erase(tensor_name);
        weights_map_[new_name] = weights_map_[trans_name];
        weights_map_.erase(trans_name);
      }
    }
  }

  // convert qzeros
  if (model_config_.quant_config.method == QUANT_AWQ) {
    // TODO(winminkong) : MLA does not currently support AWQ, will be added.
    needed_slove_weights_name = {"self_attn.query_key_value", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                                 "mlp.down_proj"};
    for (int name_idx = 0; name_idx < needed_slove_weights_name.size(); ++name_idx) {
      for (const auto layer_idx : required_layer_idx_.all) {
        std::string needed_slove_weight_name = needed_slove_weights_name[name_idx];
        if (needed_slove_weight_name == "self_attn.query_key_value" && IsClaLayer(layer_idx)) {
          needed_slove_weight_name = "self_attn.q_proj";
        }
        std::string scales_name = fmt::format("model.layers.{}.{}.scales", layer_idx, needed_slove_weight_name);
        std::string qzeros_name = fmt::format("model.layers.{}.{}.qzeros", layer_idx, needed_slove_weight_name);
        std::string zeros_name = fmt::format("model.layers.{}.{}.zeros", layer_idx, needed_slove_weight_name);

        torch::Tensor scales_gpu = GetTorchTensorFromWeight(scales_name);
        torch::Tensor qzeros_gpu = GetTorchTensorFromWeight(qzeros_name);

        torch::Tensor zeros_cpu;
        if (model_config_.quant_config.backend == CUTLASS_LINEAR_BACKEND) {
          // In AWQ: weight@fp16 = scale@fp16 * (qweight@uint4 - zeros@uint4)
          // In cutlass kernel: weight@fp16 = scale@fp16 * qweight@int4 + zeros@fp16
          // So: weight = scale * (qweight - zeros)
          //            = scale * (qweight - 8 + 8 - zeros)
          //            = scale * (qweight - 8) + scale * (8 - zeros)
          int8_t zero = std::pow(2, model_config_.quant_config.bits - 1);
          zeros_cpu = (scales_gpu * (zero - qzeros_gpu)).to(torch::kCPU).contiguous();
        } else if (model_config_.quant_config.backend == MARLIN_LINEAR_BACKEND) {
          torch::Tensor qzeros_cpu = qzeros_gpu.to(torch::kCPU);
          zeros_cpu = marlin_helper_->MarlinAwqToMarlinZeroPoints(qzeros_cpu, scales_gpu.size(0), scales_gpu.size(1));
        } else {
          KLLM_THROW("Unsupported backend for group quant, only support CUTLASS and MARLIN.");
        }

        AddWeightFromTorchTensor(zeros_name, zeros_cpu);
        weights_map_.erase(qzeros_name);
      }
    }
  }

  // convert qweight layout and binding scales
  needed_slove_weights_name = {"self_attn.o_proj", "self_attn.query_key_value"};

  if (model_config_.is_moe) {
    needed_slove_weights_name.push_back("mlp.experts.down_proj");
    needed_slove_weights_name.push_back("mlp.experts.up_gate_proj");
  }
  if (!model_config_.is_moe || model_config_.moe_config.first_k_dense_replace > 0) {
    std::vector<std::string> mlp_needed_slove_weights_name = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
    needed_slove_weights_name.insert(needed_slove_weights_name.end(), mlp_needed_slove_weights_name.begin(),
                                     mlp_needed_slove_weights_name.end());
  }
  if (model_config_.has_shared_experts) {
    std::vector<std::string> share_mlp_needed_slove_weights_name = {
        "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"};
    needed_slove_weights_name.insert(needed_slove_weights_name.end(), share_mlp_needed_slove_weights_name.begin(),
                                     share_mlp_needed_slove_weights_name.end());
  }
  AutoPackAndBindGroupTensor(needed_slove_weights_name);

  // permute lm_head: permute(1, 0)
  if (weights_map_.find("lm_head.weight") != weights_map_.end()) {
    tensor_manager_->CreateTensorWithSameShape("lm_head.weight", "empty_lm_head_tensor");
    Tensor& lm_head_tensor = weights_map_["lm_head.weight"];
    Tensor& lm_head_transpose_tensor = weights_map_["empty_lm_head_tensor"];
    Permute(lm_head_tensor, lm_head_transpose_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
    Tensor t = lm_head_transpose_tensor;
    lm_head_transpose_tensor = lm_head_tensor;
    t.shape = {size_t(lm_head_tensor.shape[1]), size_t(lm_head_tensor.shape[0])};
    weights_map_["lm_head.weight"] = t;
    weights_map_.erase("empty_lm_head_tensor");
  } else {
    KLLM_LOG_ERROR << "Process quant weight failed, lm_head.weight not found.";
  }

  return Status();
#endif
  return Status(RetCode::RET_MODEL_QUANT_FAILED, "Not supported Ascend.");
}

#ifdef ENABLE_FP8
template <typename T>
Status QuantWeight<T>::ConvertFp8E4m3() {
  if (!context_->IsGemmFp8Supported()) {
    KLLM_THROW("Cublas is insufficient to support FP8.");
  }
  DataType quant_type = TYPE_FP8_E4M3;
  KLLM_LOG_DEBUG << "Converting weight to fp8_e4m3";
  SetDevice(rank_);
  std::vector<std::string> names = {".mlp.gate_proj.weight",    ".mlp.up_proj.weight",
                                    ".mlp.gate_up_proj.weight", ".mlp.down_proj.weight",
                                    ".self_attn.o_proj.weight", ".self_attn.query_key_value.weight"};
  for (const auto layer_idx : required_layer_idx_.all) {
    for (auto name : names) {
      std::string weight_name = "model.layers." + std::to_string(layer_idx) + name;
      STATUS_CHECK_RETURN(ConvertFp8E4m3Tensor(weight_name, quant_type));
    }
  }
  return Status();
}

template <typename T>
Status QuantWeight<T>::ConvertFp8E4m3Tensor(const std::string& weight_name, DataType quant_type) {
  // replace weight tensor with quantized tensor in weights_map_
  // and add scale tensor to weights_map_
  if (weights_map_.find(weight_name) == weights_map_.end()) {  // some weight is optional
    return Status();
  }

  std::string trans_name = weight_name + "_trans";
  std::string quant_name = weight_name + "_quant";
  std::string scale_name = weight_name + "_scale";

  Tensor& weight_tensor = weights_map_[weight_name];
  auto weight_shape = std::vector<size_t>(weight_tensor.shape);
  if (weight_shape.back() % 2 != 0) {
    KLLM_LOG_INFO << "The last dim of weight is " << weight_shape.back() << " % 2 != 0 "
                  << ", therefore the weight cannot be calculated after quantization. "
                  << "Tensor of weight will not be quantized.";
    return Status();
  }

  // transpose weight from [k, n] to [n, k]
  std::vector<size_t> trans_shape{weight_shape[1], weight_shape[0]};
  tensor_manager_->AddWeightTensor(trans_name, trans_shape, weight_tensor.dtype);
  Tensor& trans_tensor = weights_map_[trans_name];
  weight_tensor.shape.insert(weight_tensor.shape.begin(), 1);
  trans_tensor.shape.insert(trans_tensor.shape.begin(), 1);
  // Permute only support 3D trans
  Permute(weight_tensor, trans_tensor, {0, 2, 1}, context_->GetMemoryManageStreams()[rank_]);
  weight_tensor.shape.erase(weight_tensor.shape.begin());
  trans_tensor.shape.erase(trans_tensor.shape.begin());

  tensor_manager_->AddWeightTensor(quant_name, std::vector<size_t>(trans_tensor.shape), quant_type);
  tensor_manager_->AddWeightTensor(scale_name, {1}, TYPE_FP32);
  Tensor& quant_tensor = weights_map_[quant_name];
  Tensor& scale_tensor = weights_map_[scale_name];
  Fp8E4m3Quantize(1, size_t(trans_tensor.shape[0]) * size_t(trans_tensor.shape[1]),
                  static_cast<const T*>(trans_tensor.GetPtr<void>()), quant_tensor.GetPtr<void>(),
                  static_cast<float*>(scale_tensor.GetPtr<void>()), false,
                  context_->GetMemoryManageStreams()[rank_].Get());
  quant_tensor.weight_scales = &scale_tensor;
  weights_map_[weight_name] = weights_map_[quant_name];
  weights_map_.erase(quant_name);
  weights_map_.erase(trans_name);
  return Status();
}

template <typename T>
bool QuantWeight<T>::LoadFp8E4m3Scale(const std::string& tensor_name, std::vector<size_t>& weight_shape,
                                      DataType& weight_data_type, void* weight_ptr) {
  SetDevice(rank_);
  if (model_config_.quant_config.method != QUANT_FP8_E4M3 || model_config_.quant_config.is_fp8_blockwise) {
    return false;
  }
  if (tensor_name.find(".weight_scale") == std::string::npos && tensor_name.find(".input_scale") == std::string::npos) {
    return false;
  }
  // scale is float scalar
  if (weight_data_type != TYPE_FP32) {
    KLLM_THROW("Not support data type of scale:" + tensor_name);
  }
  // shape is empty or [1]
  if (!weight_shape.empty() || (weight_shape.size() == 1 && weight_shape[0] == 1)) {
    KLLM_THROW("Not support shape of scale:" + tensor_name);
  }
  weight_shape = {static_cast<size_t>(1)};
  std::string weight_name;
  if (tensor_name.find("self_attn.W_pack") != std::string::npos) {
    // .weight_scale or .input_scale
    std::string suffix = tensor_name.substr(tensor_name.find_last_of("."), tensor_name.length());
    weight_name = tensor_name.substr(0, tensor_name.rfind("W_pack")) + "query_key_value" + suffix;
    tensor_manager_->AddWeightTensor(weight_name, weight_shape, weight_data_type);
    MemcpyAsync(weights_map_[weight_name].GetPtr<void>(), weight_ptr, sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[rank_]);
  } else if (tensor_name.find("_proj")) {
    weight_name = tensor_name;
    tensor_manager_->AddWeightTensor(weight_name, weight_shape, weight_data_type);
    MemcpyAsync(weights_map_[weight_name].GetPtr<void>(), weight_ptr, sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[rank_]);
  } else {
    KLLM_THROW("Not support scale:" + tensor_name);
  }
  return true;
}

template <typename T>
bool QuantWeight<T>::LoadMoeFp8E4m3BlockWiseScale(const std::string& tensor_name, std::vector<size_t>& weight_shape,
                                                  DataType& weight_data_type, void* weight_ptr) {
  SetDevice(rank_);
  if (model_config_.quant_config.method != QUANT_FP8_E4M3 && !model_config_.quant_config.is_fp8_blockwise) {
    return false;
  }
  if (tensor_name.find(".weight_scale") == std::string::npos && tensor_name.find(".input_scale") == std::string::npos) {
    return false;
  }
  // scale is float scalar
  if (weight_data_type != TYPE_FP32) {
    KLLM_THROW("Not support data type of scale:" + tensor_name);
  }
  if (tensor_name.find(".experts.") != std::string::npos) {
    int layer_idx = -1, expert_idx = -1;
    GetExpertsScaleIdx(tensor_name, layer_idx, expert_idx);
    if (layer_idx == -1) {
      return false;
    }
    if (expert_map_[expert_idx] < 0) {
      // Skip load weight when the expert_id will be not used in current rank.
      return false;
    }
    expert_idx = expert_map_[expert_idx];

    size_t block_n = model_config_.quant_config.weight_block_size[0];
    size_t block_k = model_config_.quant_config.weight_block_size[1];

    size_t moe_inter_size_per_rank =
        DivRoundUp(model_config_.moe_config.moe_inter_size, runtime_config_.parallel_basic_config.moe_tensor_para_size);
    if (moe_inter_size_per_rank % block_n != 0) {
      KLLM_THROW(fmt::format(
          "The moe_inter_size_per_rank of gate's and up's weight = {}, is not divisible by weight quant block_n = {}",
          moe_inter_size_per_rank, block_n));
    }
    if (tensor_para_size_ > 1 && moe_inter_size_per_rank % block_k != 0) {
      KLLM_THROW(fmt::format(
          "The moe_inter_size_per_rank of down's weight = {}, is not divisible by weight quant block_k = {}",
          moe_inter_size_per_rank, block_k));
    }
    size_t hidden_units = model_config_.hidden_units;
    std::vector<size_t> up_gate_experts_scale_shape = {size_t(num_experts_per_rank_),
                                                       DivRoundUp(moe_inter_size_per_rank, block_n) * 2,
                                                       DivRoundUp(hidden_units, block_k)};
    std::vector<size_t> down_experts_scale_shape = {size_t(num_experts_per_rank_), DivRoundUp(hidden_units, block_n),
                                                    DivRoundUp(moe_inter_size_per_rank, block_k)};
    // For up_gate proj scale
    if (tensor_name.find(".up_proj.weight_scale") != std::string::npos ||
        tensor_name.find(".gate_proj.weight_scale") != std::string::npos) {
      if (weight_shape[0] != DivRoundUp(model_config_.moe_config.moe_inter_size, block_n)) {
        KLLM_THROW("Not support shape of scale:" + tensor_name);
      }
      std::string up_gate_experts_scale_name =
          "model.layers." + std::to_string(layer_idx) + ".mlp.experts.up_gate_proj.weight_scale_inv";
      if (weights_map_.find(up_gate_experts_scale_name) == weights_map_.end()) {
        tensor_manager_->AddWeightTensor(up_gate_experts_scale_name, up_gate_experts_scale_shape, weight_data_type);
      }
      size_t expert_scale_pitch =
          up_gate_experts_scale_shape[1] / 2 * up_gate_experts_scale_shape[2] * GetTypeSize(weight_data_type);
      size_t double_expert_scale_pitch = expert_scale_pitch * 2;
      size_t src_upgate_offset = runtime_config_.parallel_basic_config.moe_tensor_para_size > 1
                                     ? (rank_ / expert_para_size_) * expert_scale_pitch
                                     : 0;
      if (tensor_name.find(".gate_proj.") != std::string::npos) {
        MemcpyAsync(weights_map_[up_gate_experts_scale_name].GetPtr<void>() +
                        static_cast<size_t>(expert_idx) * double_expert_scale_pitch,
                    weight_ptr + src_upgate_offset, expert_scale_pitch, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      } else if (tensor_name.find(".up_proj.") != std::string::npos) {
        MemcpyAsync(weights_map_[up_gate_experts_scale_name].GetPtr<void>() +
                        static_cast<size_t>(expert_idx) * double_expert_scale_pitch + expert_scale_pitch,
                    weight_ptr + src_upgate_offset, expert_scale_pitch, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      }
    }
    // For down_proj scale
    if (tensor_name.find(".down_proj.weight_scale") != std::string::npos) {
      std::string down_experts_scale_name =
          "model.layers." + std::to_string(layer_idx) + ".mlp.experts.down_proj.weight_scale_inv";
      if (weights_map_.find(down_experts_scale_name) == weights_map_.end()) {
        tensor_manager_->AddWeightTensor(down_experts_scale_name, down_experts_scale_shape, weight_data_type);
      }

      size_t dst_pitch = down_experts_scale_shape[2] * GetTypeSize(weight_data_type);
      size_t src_pitch = down_experts_scale_shape[2] * runtime_config_.parallel_basic_config.moe_tensor_para_size *
                         GetTypeSize(weight_data_type);
      size_t expert_scale_pitch =
          down_experts_scale_shape[2] * down_experts_scale_shape[1] * GetTypeSize(weight_data_type);
      size_t src_down_offset =
          runtime_config_.parallel_basic_config.moe_tensor_para_size > 1 ? (rank_ / expert_para_size_) * dst_pitch : 0;
      Memcpy2DAsync(
          weights_map_[down_experts_scale_name].GetPtr<void>() + static_cast<size_t>(expert_idx) * expert_scale_pitch,
          dst_pitch, weight_ptr + src_down_offset, src_pitch, dst_pitch, down_experts_scale_shape[1],
          MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    }
  }
  return true;
}

template <typename T>
Status QuantWeight<T>::BindFp8E4m3Scale(const int num_heads, const int num_kv_heads) {
  // KLLM_LOG_INFO << "Start binding scale";
  SetDevice(rank_);
  std::vector<std::string> names = {".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.down_proj.", ".self_attn.o_proj."};
  for (auto name : names) {
    BindFp8E4m3ScaleOfProjWeight(name);
  }

  std::string name = ".self_attn.query_key_value.";
  BindFp8E4m3ScaleOfQkvWeight(name, num_heads, num_kv_heads);

  return Status();
}

template <typename T>
Status QuantWeight<T>::BindFp8E4m3ScaleOfProjWeight(const std::string& name) {
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string weight_name = "model.layers." + std::to_string(layer_idx) + name + "weight";
    KLLM_LOG_DEBUG << "Try to bind scales to weight " << weight_name;
    if (weights_map_.find(weight_name) == weights_map_.end()) {
      KLLM_LOG_DEBUG << "weight " << weight_name << " not found, continue.";
      continue;
    }
    std::string weight_scale_name = "model.layers." + std::to_string(layer_idx) + name + "weight_scale";
    std::string input_scale_name = "model.layers." + std::to_string(layer_idx) + name + "input_scale";
    if (weights_map_.find(weight_scale_name) != weights_map_.end()) {
      weights_map_[weight_name].weight_scales = &(weights_map_[weight_scale_name]);
    }
    if (weights_map_.find(input_scale_name) != weights_map_.end()) {
      weights_map_[weight_name].input_scales = &(weights_map_[input_scale_name]);
    }
    if ((name == ".mlp.up_proj.") && (weights_map_.find(weight_scale_name) == weights_map_.end())) {
      std::string gate_name = ".mlp.gate_proj.";
      std::string gate_weight_scale_name = "model.layers." + std::to_string(layer_idx) + gate_name + "weight_scale";
      std::string gate_input_scale_name = "model.layers." + std::to_string(layer_idx) + gate_name + "input_scale";
      if (weights_map_.find(gate_weight_scale_name) != weights_map_.end()) {
        weights_map_[weight_name].weight_scales = &(weights_map_[gate_weight_scale_name]);
      }
      if (weights_map_.find(gate_input_scale_name) != weights_map_.end()) {
        weights_map_[weight_name].input_scales = &(weights_map_[gate_input_scale_name]);
      }
    }
  }
  return Status();
}

template <typename T>
Status QuantWeight<T>::BindMoeFp8E4m3BlockWiseScaleOfWeight() {
  SetDevice(rank_);
  std::vector<std::string> names = {".mlp.shared_expert.gate_up_proj.", ".mlp.shared_expert.gate_proj.",
                                    ".mlp.shared_expert.up_proj.", ".mlp.shared_expert.down_proj."};
  if (!model_config_.quant_config.enable_moe_int4) {
    names.push_back(".mlp.experts.down_proj.");
    names.push_back(".mlp.experts.up_gate_proj.");
  }
  const std::unordered_set<std::string> optional_names = {
      ".mlp.shared_expert.gate_up_proj.", ".mlp.shared_expert.gate_proj.", ".mlp.shared_expert.up_proj."};
  for (auto name : names) {
    for (const auto layer_idx : required_layer_idx_.moe) {
      const std::string weight_name = "model.layers." + std::to_string(layer_idx) + name + "weight";
      const std::string weight_scale_name = "model.layers." + std::to_string(layer_idx) + name + "weight_scale_inv";
      if (weights_map_.find(weight_scale_name) != weights_map_.end()) {
        KLLM_LOG_DEBUG << "Binding " << weight_scale_name << " to " << weight_name;
        weights_map_[weight_name].weight_scales = &(weights_map_[weight_scale_name]);
      } else if (optional_names.find(name) == optional_names.end()) {
        KLLM_THROW("Bind error: Not find scale: " + weight_scale_name);
      }
    }
  }
  return Status();
}

template <typename T>
Status QuantWeight<T>::BindFp8E4m3ScaleOfQkvWeight(const std::string& name, const int num_heads,
                                                   const int num_kv_heads) {
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string weight_name = "model.layers." + std::to_string(layer_idx) + name + "weight";
    std::string weight_scale_name = "model.layers." + std::to_string(layer_idx) + name + "weight_scale";
    std::string input_scale_name = "model.layers." + std::to_string(layer_idx) + name + "input_scale";

    std::string q_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.";
    std::string k_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.";
    std::string v_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.";

    std::string q_input_scale_name = q_name + "input_scale";
    std::string k_input_scale_name = k_name + "input_scale";
    std::string v_input_scale_name = v_name + "input_scale";
    // If weights of q,k,v are saved independently,
    // input_scale of qkv is max of q/k/v's input_scale
    if (weights_map_.find(q_input_scale_name) != weights_map_.end() &&
        weights_map_.find(k_input_scale_name) != weights_map_.end() &&
        weights_map_.find(v_input_scale_name) != weights_map_.end() &&
        weights_map_.find(input_scale_name) == weights_map_.end()) {
      tensor_manager_->AddWeightTensor(input_scale_name, {1}, weights_map_[q_input_scale_name].dtype);
      float* q_scale = static_cast<float*>(weights_map_[q_input_scale_name].GetPtr<void>());
      float* k_scale = static_cast<float*>(weights_map_[k_input_scale_name].GetPtr<void>());
      float* v_scale = static_cast<float*>(weights_map_[v_input_scale_name].GetPtr<void>());
      float* qkv_scale = static_cast<float*>(weights_map_[input_scale_name].GetPtr<void>());
      GetMaxScaleOfQkv(q_scale, k_scale, v_scale, qkv_scale);
    }

    std::string q_weight_scale_name = q_name + "weight_scale";
    std::string k_weight_scale_name = k_name + "weight_scale";
    std::string v_weight_scale_name = v_name + "weight_scale";
    // If weights of q,k,v are saved independently,
    // weight_scale of qkv is max of q/k/v's weight_scale,
    // weight of qkv need to be Rescale.
    if (weights_map_.find(q_weight_scale_name) != weights_map_.end() &&
        weights_map_.find(k_weight_scale_name) != weights_map_.end() &&
        weights_map_.find(v_weight_scale_name) != weights_map_.end() &&
        weights_map_.find(weight_scale_name) == weights_map_.end()) {
      tensor_manager_->AddWeightTensor(weight_scale_name, {1}, weights_map_[q_weight_scale_name].dtype);
      float* q_scale = static_cast<float*>(weights_map_[q_weight_scale_name].GetPtr<void>());
      float* k_scale = static_cast<float*>(weights_map_[k_weight_scale_name].GetPtr<void>());
      float* v_scale = static_cast<float*>(weights_map_[v_weight_scale_name].GetPtr<void>());
      float* qkv_scale = static_cast<float*>(weights_map_[weight_scale_name].GetPtr<void>());
      GetMaxScaleOfQkv(q_scale, k_scale, v_scale, qkv_scale);

      // q,k,v_weight * (q,k,v_scale / qkv_scale)
      Tensor& weight = weights_map_[weight_name];
      size_t n = weight.GetElementNumber() / (num_heads / num_kv_heads + 2);
      size_t size = weight.GetTotalBytes() / (num_heads / num_kv_heads + 2);
      void* q_weight = weight.GetPtr<void>();
      void* k_weight = q_weight + size * num_heads / num_kv_heads;
      void* v_weight = k_weight + size;
      RescaleFp8E4m3(q_weight, q_weight, n * num_heads / num_kv_heads, q_scale, qkv_scale,
                     context_->GetMemoryManageStreams()[rank_].Get());
      RescaleFp8E4m3(k_weight, k_weight, n, k_scale, qkv_scale, context_->GetMemoryManageStreams()[rank_].Get());
      RescaleFp8E4m3(v_weight, v_weight, n, v_scale, qkv_scale, context_->GetMemoryManageStreams()[rank_].Get());
    }

    if (weights_map_.find(weight_scale_name) != weights_map_.end()) {
      weights_map_[weight_name].weight_scales = &(weights_map_[weight_scale_name]);
    }
    if (weights_map_.find(input_scale_name) != weights_map_.end()) {
      weights_map_[weight_name].input_scales = &(weights_map_[input_scale_name]);
    }
  }
  return Status();
}

template <typename T>
Status QuantWeight<T>::GetMaxScaleOfQkv(float* q_scale, float* k_scale, float* v_scale, float* qkv_scale) {
  auto options = torch::TensorOptions().device(torch::kCUDA, rank_).dtype(torch::kFloat32);
  torch::Tensor q_scale_tensor = torch::from_blob(q_scale, {1}, options);
  torch::Tensor k_scale_tensor = torch::from_blob(k_scale, {1}, options);
  torch::Tensor v_scale_tensor = torch::from_blob(v_scale, {1}, options);
  torch::Tensor qkv_scale_tensor = torch::from_blob(qkv_scale, {1}, options);
  torch::max_out(qkv_scale_tensor, q_scale_tensor, k_scale_tensor);
  torch::max_out(qkv_scale_tensor, qkv_scale_tensor, v_scale_tensor);
  return Status();
}

// origin weight matrix @ identity matrix = dequantized weight matrix
// NOTE(winminkong): 目前只支持GPTQ量化和后端为Machete算子的反量化，如有其他需要可继续增加
template <typename T>
Tensor QuantWeight<T>::CommonDequantTensor(const std::string& weight_name, bool remove_weight) {
  SetDevice(rank_);
  torch::Tensor eye_matrix;
  auto options = torch::TensorOptions().device(torch::kCUDA, rank_).dtype(GetTorchTypeFromDataType(weight_data_type_));
  std::string dequant_weight_name = weight_name + "_dequant";
  size_t input_size_per_tp = weights_map_[weight_name].shape[0];
  size_t pack_factor = 32 / model_config_.quant_config.bits;

  if (model_config_.quant_config.method == QUANT_GPTQ) {
    eye_matrix = torch::eye(static_cast<int64_t>(input_size_per_tp * pack_factor), options);
    tensor_manager_->AddWeightTensor(
        dequant_weight_name, {input_size_per_tp * pack_factor, weights_map_[weight_name].shape[1]}, weight_data_type_);
  }
  if (model_config_.quant_config.backend == MACHETE_LINEAR_BACKEND && model_config_.quant_config.method == QUANT_GPTQ) {
    size_t m = input_size_per_tp * pack_factor;
    size_t n = weights_map_[weight_name].shape[1];
    // 获取 workspace
    int64_t current_workspace_size = -1;
    InvokeMacheteGemm(current_workspace_size, nullptr, context_->GetMemoryManageStreams()[rank_].Get(), m, n, m,
                      eye_matrix.data_ptr(), weights_map_[weight_name].GetPtr<void>(),
                      weights_map_[dequant_weight_name].GetPtr<void>(), GetMacheteDataType<T>(),
                      llm_kernels::nvidia::vllm_dtype::kU4B8, weights_map_[weight_name].scales->GetPtr<void>(),
                      weights_map_[weight_name].scales->shape, GetMacheteDataType<T>(), std::nullopt, std::nullopt,
                      std::nullopt, model_config_.quant_config.group_size, std::nullopt);
    if (current_workspace_size > -1) {
      std::string dequant_ws = "temp_dequant_workspace";
      tensor_manager_->AddWeightTensor(dequant_ws, {current_workspace_size}, DataType::TYPE_INT8);
      InvokeMacheteGemm(current_workspace_size, weights_map_[dequant_ws].GetPtr<void>(),
                        context_->GetMemoryManageStreams()[rank_].Get(), m, n, m, eye_matrix.data_ptr(),
                        weights_map_[weight_name].GetPtr<void>(), weights_map_[dequant_weight_name].GetPtr<void>(),
                        GetMacheteDataType<T>(), llm_kernels::nvidia::vllm_dtype::kU4B8,
                        weights_map_[weight_name].scales->GetPtr<void>(), weights_map_[weight_name].scales->shape,
                        GetMacheteDataType<T>(), std::nullopt, std::nullopt, std::nullopt,
                        model_config_.quant_config.group_size, std::nullopt);
      weights_map_.erase(dequant_ws);
    } else {
      KLLM_LOG_ERROR << "Machete GEMM failed for Dequant";
    }
  }
  return weights_map_[dequant_weight_name];
}

template <typename T>
Status QuantWeight<T>::BindFp8E4m3ScaleOfMoeWeight() {
  SetDevice(rank_);
  std::vector<std::string> names = {".self_attn.q_proj.", ".mlp.shared_expert.gate_proj.",
                                    ".mlp.shared_expert.up_proj.", ".mlp.shared_expert.down_proj."};
  size_t num_experts = model_config_.moe_config.num_experts;
  for (size_t idx = 0; idx < num_experts; idx++) {
    if (expert_map_[idx] < 0) {
      continue;
    }
    int expert_idx = expert_map_[idx];
    names.push_back(fmt::format(".mlp.experts.{}.gate_proj", expert_idx));
    names.push_back(fmt::format(".mlp.experts.{}.up_proj", expert_idx));
    names.push_back(fmt::format(".mlp.experts.{}.down_proj", expert_idx));
  }
  for (auto name : names) {
    BindFp8E4m3ScaleOfProjWeight(name);
  }
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.";
    std::string name = prefix + "down_proj.";
    std::string weight_name = name + "weight";
    std::string input_scale_name = name + "input_scale";
    float* down_input_scale = nullptr;
    for (size_t idx = 0; idx < num_experts_per_rank_; ++idx) {
      KLLM_LOG_DEBUG << "Build " << input_scale_name << " layer " << layer_idx << " expert " << idx;
      std::string down_idx_input_scale_name = prefix + std::to_string(idx) + ".down_proj.input_scale";
      if (weights_map_.find(down_idx_input_scale_name) == weights_map_.end()) {
        // model is dynamic quantization, no input_scale
        break;
      }
      float* down_idx_input_scale = static_cast<float*>(weights_map_[down_idx_input_scale_name].GetPtr<void>());
      if (idx == 0) {
        tensor_manager_->AddWeightTensor(input_scale_name, {1}, weights_map_[down_idx_input_scale_name].dtype);
        down_input_scale = static_cast<float*>(weights_map_[input_scale_name].GetPtr<void>());
        MemcpyAsync(down_input_scale, down_idx_input_scale, sizeof(float), MEMCPY_DEVICE_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      } else {
        Max(down_input_scale, down_input_scale, down_idx_input_scale, 1, rank_);
      }
    }
    if (weights_map_.find(input_scale_name) != weights_map_.end()) {
      KLLM_LOG_DEBUG << "Binding " << input_scale_name << " to " << weight_name;
      weights_map_[weight_name].input_scales = &(weights_map_[input_scale_name]);
    }
  }
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.";
    std::string name = prefix + "down_proj.";
    std::string weight_name = name + "weight";
    std::string weight_scale_name = name + "weight_scale";
    std::string input_scale_name = name + "input_scale";
    float* down_input_scale = nullptr;
    float* down_weight_scale = nullptr;
    for (size_t idx = 0; idx < num_experts_per_rank_; ++idx) {
      KLLM_LOG_DEBUG << "Build " << weight_scale_name << " layer " << layer_idx << " expert " << idx;
      std::string down_idx_weight_scale_name = prefix + std::to_string(idx) + ".down_proj.weight_scale";
      float* down_idx_weight_scale = static_cast<float*>(weights_map_[down_idx_weight_scale_name].GetPtr<void>());
      if (idx == 0) {
        tensor_manager_->AddWeightTensor(weight_scale_name, {static_cast<size_t>(num_experts_per_rank_)},
                                         weights_map_[down_idx_weight_scale_name].dtype);
        down_weight_scale = static_cast<float*>(weights_map_[weight_scale_name].GetPtr<void>());
        if (weights_map_.find(input_scale_name) != weights_map_.end()) {
          down_input_scale = static_cast<float*>(weights_map_[input_scale_name].GetPtr<void>());
        }
      }
      if (down_input_scale != nullptr) {
        InvokeMul(down_input_scale, down_idx_weight_scale, down_weight_scale + idx, 1, rank_);
      }
    }
    if (weights_map_.find(weight_scale_name) != weights_map_.end()) {
      KLLM_LOG_DEBUG << "Binding " << weight_scale_name << " to " << weight_name;
      weights_map_[weight_name].weight_scales = &(weights_map_[weight_scale_name]);
    }
    // down_input_scale = 1 / down_input_scale
    if (down_input_scale != nullptr) {
      Reciprocal(down_input_scale, down_input_scale, 1, rank_);
    }
  }
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.";
    std::string name = prefix + "up_gate_proj.";
    std::string weight_name = name + "weight";
    std::string input_scale_name = name + "input_scale";
    float* up_gate_input_scale = nullptr;
    for (size_t idx = 0; idx < num_experts_per_rank_; ++idx) {
      KLLM_LOG_DEBUG << "Build " << input_scale_name << " layer " << layer_idx << " expert " << idx;
      std::string up_idx_input_scale_name = prefix + std::to_string(idx) + ".up_proj.input_scale";
      std::string gate_idx_input_scale_name = prefix + std::to_string(idx) + ".gate_proj.input_scale";
      if (weights_map_.find(up_idx_input_scale_name) == weights_map_.end()) {
        // model is dynamic quantization, no input_scale
        break;
      }
      float* up_idx_input_scale = static_cast<float*>(weights_map_[up_idx_input_scale_name].GetPtr<void>());
      float* gate_idx_input_scale = static_cast<float*>(weights_map_[gate_idx_input_scale_name].GetPtr<void>());
      if (idx == 0) {
        tensor_manager_->AddWeightTensor(input_scale_name, {1}, weights_map_[up_idx_input_scale_name].dtype);
        up_gate_input_scale = static_cast<float*>(weights_map_[input_scale_name].GetPtr<void>());
        Max(up_gate_input_scale, up_idx_input_scale, gate_idx_input_scale, 1, rank_);
      } else {
        Max(up_gate_input_scale, up_gate_input_scale, up_idx_input_scale, 1, rank_);
        Max(up_gate_input_scale, up_gate_input_scale, gate_idx_input_scale, 1, rank_);
      }
    }
    if (weights_map_.find(input_scale_name) != weights_map_.end()) {
      KLLM_LOG_DEBUG << "Binding " << input_scale_name << " to " << weight_name;
      weights_map_[weight_name].input_scales = &(weights_map_[input_scale_name]);
    }
  }
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.";
    std::string name = prefix + "up_gate_proj.";
    std::string weight_name = name + "weight";
    std::string weight_scale_name = name + "weight_scale";
    float* up_gate_weight_scale = nullptr;
    for (size_t idx = 0; idx < num_experts_per_rank_; ++idx) {
      KLLM_LOG_DEBUG << "Build " << weight_scale_name << " layer " << layer_idx << " expert " << idx;
      std::string up_idx_weight_scale_name = prefix + std::to_string(idx) + ".up_proj.weight_scale";
      std::string gate_idx_weight_scale_name = prefix + std::to_string(idx) + ".gate_proj.weight_scale";
      float* up_idx_weight_scale = static_cast<float*>(weights_map_[up_idx_weight_scale_name].GetPtr<void>());
      float* gate_idx_weight_scale = static_cast<float*>(weights_map_[gate_idx_weight_scale_name].GetPtr<void>());
      if (idx == 0) {
        tensor_manager_->AddWeightTensor(weight_scale_name, {static_cast<size_t>(num_experts_per_rank_)},
                                         weights_map_[up_idx_weight_scale_name].dtype);
        up_gate_weight_scale = static_cast<float*>(weights_map_[weight_scale_name].GetPtr<void>());
      }
      Max(up_gate_weight_scale + idx, up_idx_weight_scale, gate_idx_weight_scale, 1, rank_);
    }
  }
  // rescale up_gate_weight with up_gate_proj.weight_scale
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.";
    std::string name = prefix + "up_gate_proj.";
    std::string weight_name = name + "weight";
    std::string weight_scale_name = name + "weight_scale";
    // [num_experts_per_rank_, moe_inter_size_per_rank * 2, hidden_units]
    std::vector<size_t> up_gate_weight_shape = std::vector<size_t>(weights_map_[weight_name].shape);
    void* up_gate_weight = weights_map_[weight_name].GetPtr<void>();
    float* up_gate_weight_scale = static_cast<float*>(weights_map_[weight_scale_name].GetPtr<void>());
    for (size_t idx = 0; idx < num_experts_per_rank_; ++idx) {
      KLLM_LOG_DEBUG << name << " layer " << layer_idx << " expert " << idx;
      std::string up_idx_weight_scale_name = prefix + std::to_string(idx) + ".up_proj.weight_scale";
      std::string gate_idx_weight_scale_name = prefix + std::to_string(idx) + ".gate_proj.weight_scale";
      float* up_idx_weight_scale = static_cast<float*>(weights_map_[up_idx_weight_scale_name].GetPtr<void>());
      float* gate_idx_weight_scale = static_cast<float*>(weights_map_[gate_idx_weight_scale_name].GetPtr<void>());
      int n = up_gate_weight_shape[1] / 2 * up_gate_weight_shape[2];
      // up
      void* up_idx_weight = up_gate_weight + idx * up_gate_weight_shape[1] * up_gate_weight_shape[2];
      RescaleFp8E4m3(up_idx_weight, up_idx_weight, n, up_idx_weight_scale, up_gate_weight_scale + idx,
                     context_->GetMemoryManageStreams()[rank_].Get());
      // gate
      void* gate_idx_weight = up_gate_weight + idx * up_gate_weight_shape[1] * up_gate_weight_shape[2] + n;
      RescaleFp8E4m3(gate_idx_weight, gate_idx_weight, n, gate_idx_weight_scale, up_gate_weight_scale + idx,
                     context_->GetMemoryManageStreams()[rank_].Get());
    }
  }
  // up_gate_proj.weight_scale = [(up_gate_proj.input_scale * up_gate_proj.idx.weight_scale)
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.";
    std::string name = prefix + "up_gate_proj.";
    std::string weight_name = name + "weight";
    std::string weight_scale_name = name + "weight_scale";
    std::string input_scale_name = name + "input_scale";
    float* up_gate_weight_scale = static_cast<float*>(weights_map_[weight_scale_name].GetPtr<void>());
    if (weights_map_.find(input_scale_name) != weights_map_.end()) {
      float* up_gate_input_scale = static_cast<float*>(weights_map_[input_scale_name].GetPtr<void>());
      for (size_t idx = 0; idx < num_experts_per_rank_; ++idx) {
        KLLM_LOG_DEBUG << "scale " << weight_scale_name << " layer " << layer_idx << " expert " << idx;
        InvokeMul(up_gate_weight_scale + idx, up_gate_input_scale, up_gate_weight_scale + idx, 1, rank_);
      }
    }
    if (weights_map_.find(weight_scale_name) != weights_map_.end()) {
      KLLM_LOG_DEBUG << "Binding " << weight_scale_name << " to " << weight_name;
      weights_map_[weight_name].weight_scales = &(weights_map_[weight_scale_name]);
    }
  }

  return Status();
}

#endif

#ifdef ENABLE_CUDA
template <typename T>
Status QuantWeight<T>::AddWeightFromTorchTensor(const std::string& name, torch::Tensor& tensor) {
  tensor_manager_->AddWeightTensor(name, std::vector<size_t>(tensor.sizes().begin(), tensor.sizes().end()),
                                   GetDataTypeFromTorchType(tensor.scalar_type()));
  if (tensor.device().type() == torch::kCPU) {
    MemcpyAsync(weights_map_[name].GetPtr<void>(), tensor.contiguous().data_ptr(), weights_map_[name].GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
  } else {
    MemcpyAsync(weights_map_[name].GetPtr<void>(), tensor.contiguous().data_ptr(), weights_map_[name].GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
  }
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
  return Status();
}

template <typename T>
Status QuantWeight<T>::AddWeightFromTorchTensor(const std::string& name, torch::Tensor& tensor,
                                                DataType& weight_data_type) {
  AddWeightFromTorchTensor(name, tensor);
  weights_data_type_map_[name] = weight_data_type;
  return Status();
}

template <typename T>
torch::Tensor QuantWeight<T>::GetTorchTensorFromWeightPtr(std::vector<size_t> weight_shape, DataType weight_data_type,
                                                          void* weight_ptr, bool to_gpu) {
  auto options = torch::TensorOptions().device(torch::kCPU).dtype(GetTorchTypeFromDataType(weight_data_type));
  torch::Tensor tensor =
      torch::from_blob(weight_ptr, std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), options);
  if (to_gpu) {
    tensor.to(torch::Device(torch::kCUDA, rank_));
  }
  return tensor;
}

template <typename T>
torch::Tensor QuantWeight<T>::GetTorchTensorFromWeight(const std::string& name) {
  auto options =
      torch::TensorOptions().device(torch::kCUDA, rank_).dtype(GetTorchTypeFromDataType(weights_map_[name].dtype));
  std::vector<size_t> tensor_shape = weights_map_[name].shape;
  torch::Tensor tensor_gpu = torch::from_blob(weights_map_[name].GetPtr<void>(),
                                              std::vector<int64_t>(tensor_shape.begin(), tensor_shape.end()), options);
  return tensor_gpu;
}

template <typename T>
torch::Tensor QuantWeight<T>::TpSplitTensor(torch::Tensor& tensor, int split_dim, int split_pos, int single_size) {
  return tensor.slice(split_dim, split_pos * single_size, (split_pos + 1) * single_size).contiguous();
}

#endif

// TODO(winminkong): delete this function
template <typename T>
void QuantWeight<T>::GetExpertsScaleIdx(const std::string& expert_scale_name, int& layer_idx, int& expert_idx) {
  // Get the index of the moe layer and the index of each expert
  static const std::regex re(R"(\d+)");
  std::sregex_iterator next(expert_scale_name.begin(), expert_scale_name.end(), re);
  std::sregex_iterator end;
  if (next != end) {
    std::smatch match = *next;
    layer_idx = std::stoi(match.str());
    next++;
    match = *next;
    expert_idx = std::stoi(match.str());
  } else {
    KLLM_LOG_ERROR << fmt::format("Failed to get Expert ID from tensor name {}", expert_scale_name);
  }
}

template class QuantWeight<float>;
template class QuantWeight<float16>;
template class QuantWeight<bfloat16>;

}  // namespace ksana_llm
