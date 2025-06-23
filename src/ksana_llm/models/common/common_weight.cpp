/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/common/common_weight.h"

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/nn/functional/normalization.h>
#include <torch/torch.h>
#include <cstdlib>
#include <filesystem>
#include <regex>

#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"
#include "nlohmann/json.hpp"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#ifdef ENABLE_ACL
#  include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#endif

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/kernels/permute.h"
#include "ksana_llm/kernels/trans_layout.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/safetensors_file_saver.h"

namespace ksana_llm {

// weight_name 与存储的模型文件的关联映射关系
// gather_embedding                   : model.wte.weight.bin
// input_layernorm          + layer x : model.layers.x.input_layernorm.weight.bin
// post_attention_layernorm + layer x : model.layers.x.post_attention_layernorm.weight.bin
// attention.dense + rank r + layer x : model.layers.x.attention.dense.weight.r.bin
// attention.query_key_value + r +  x : model.layers.x.attention.query_key_value.weight.r.bin
// mlp.gate_proj +  rank r  + layer x : model.layers.x.mlp.gate_proj.weight.r.bin
// mlp.up_proj   +  rank r  + layer x : model.layers.x.mlp.up_proj.weight.r.bin
// mlp.down_proj +  rank r  + layer x : model.layers.x.mlp.down_proj.weight.r.bin
// norm                               : model.final_layernorm.weight
// lm_head                            : model.lm_head.weight
template <typename T>
CommonWeight<T>::~CommonWeight() {}

template <typename T>
CommonWeight<T>::CommonWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : BaseWeight(model_config, rank, context) {
  model_path_ = model_config.path;
  rank_ = rank;
  if (!GetModelInfo(model_config).OK()) {
    KLLM_THROW(fmt::format("Load model config file error."));
  }
  tensor_manager_ = std::make_shared<TensorManager>(rank, weights_map_);
  quant_weight_solver_ =
      std::make_shared<QuantWeight<T>>(model_config, rank, context, weights_map_, weights_data_type_map_);
}

int CheckQKVWeight(const std::string& str, const int head_num, const int num_kv_heads) {
  std::string suffix = "_proj.weight";
  if (str.find("_proj.bias") != std::string::npos) {
    suffix = "_proj.bias";
  }
  if (str.length() < suffix.length() + 1 || str.compare(str.length() - suffix.length(), suffix.length(), suffix)) {
    return -1;
  }
  std::vector<char> qkv_list = {'q', 'k', 'v'};
  std::vector<int> qkv_offset = {0, head_num / num_kv_heads, head_num / num_kv_heads + 1};
  for (int i = 0; i < 3; ++i) {
    if (str[str.length() - suffix.length() - 1] == qkv_list[i]) {
      return qkv_offset[i];
    }
  }
  return -1;
}

template <typename T>
void CommonWeight<T>::SetEmbeddingsConfig() {
  model_config_.tie_word_embeddings = true;
}

template <typename T>
bool CommonWeight<T>::ShouldUseFusedGateUpWeights() {
#ifdef ENABLE_CUDA
  // TODO(rockcao): support other models like llama, qwen, baichuan
  const std::vector<std::string> enabled_type_list = {"deepseek_v3"};
  std::string unified_model_type = model_config_.type;
  // unify type to lower case
  std::transform(unified_model_type.begin(), unified_model_type.end(), unified_model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  for (const auto& type : enabled_type_list) {
    if (unified_model_type.find(type) != std::string::npos) {
      KLLM_LOG_INFO << "using fused gate up weights for " << unified_model_type;
      return true;
    }
  }
#endif
  return false;
}

template <typename T>
Status CommonWeight<T>::PrepareLoadOpMeta(size_t& tensor_para_offset, std::vector<size_t>& weight_shape,
                                          bool& transpose_first, const std::string& tensor_name) {
  // per tensor fp8 scale does not require slice
  if (tensor_name.find("_scale") != std::string::npos && !model_config_.quant_config.is_fp8_blockwise) {
    return Status();
  }
  // EmbedTokensUseCpu does not require slicing
  if (Singleton<Environment>::GetInstance()->EmbedTokensUseCpu() &&
      tensor_name.find("embed_tokens") != std::string::npos) {
    return Status();
  }
  // Layernorm, o_proj bias and down_proj bias do not require slicing
  if (tensor_name.find("norm.") != std::string::npos || tensor_name.find("o_proj.bias") != std::string::npos ||
      tensor_name.find("down_proj.bias") != std::string::npos || tensor_name.find("a_proj") != std::string::npos) {
    return Status();
  }
  // Quant Weight, Scales do not slicing here
  if (tensor_name.find(".qweight") != std::string::npos || tensor_name.find(".scales") != std::string::npos ||
      tensor_name.find(".qzeros") != std::string::npos || tensor_name.find(".g_idx") != std::string::npos) {
    return Status();
  }

  // Moe Gating weight does not require slicing include share_gating
  if (tensor_name.find("gate.") != std::string::npos) {
    return Status();
  }

  // Moe Shared Expert do not slicing when enable_whole_shared_expert
  if (enable_full_shared_expert_ && tensor_name.find(".shared_expert") != std::string::npos) {
    KLLM_LOG_DEBUG << fmt::format("Using enable_full_shared_expert, tensor {} no need to slicing.", tensor_name);
    return Status();
  }

  tensor_para_offset = rank_;
  if (tensor_name.find(".bias") != std::string::npos || tensor_name.find("o_proj") != std::string::npos ||
      tensor_name.find("down_proj") != std::string::npos || tensor_name.find("embed_") != std::string::npos) {
    transpose_first = true;
  }
  if (transpose_first) {
    if (tensor_name.find("o_proj") != std::string::npos) {
      int attn_group_tp_size = Singleton<Environment>::GetInstance()->GetAttentionTensorParallel();
      tensor_para_offset = rank_ % attn_group_tp_size;
      weight_shape[1] /= attn_group_tp_size;
    } else {
      weight_shape[1] /= tensor_para_size_;
    }
  } else {
    weight_shape[0] = DivRoundUp(weight_shape[0], tensor_para_size_);
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                            const std::vector<std::string>& weight_name_list,
                                            const std::vector<std::string>& custom_name_list) {
  SetDevice(rank_);
  const bool use_fused_gate_up_weights = ShouldUseFusedGateUpWeights();
  KLLM_LOG_DEBUG << "use_fused_gate_up_weights: " << use_fused_gate_up_weights;
  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    // tensor_para_offset 用于标记读取 weights_data 时是否做分卡处理:
    //     input_layernorm:          不做分卡处理
    //     post_attention_layernorm: 不做分卡处理
    //     self_attn.o_proj:         先转置,再按 axis=0 切分
    //     self_attn.qkv_proj:       先按 axis=0 切分, 再 permute((2, 0, 1))
    //     mlp.down_proj:            先转置,再按 axis=0 切分
    //     mlp.up_proj:              先按 axis=0 切分, 再转置
    //     mlp.gate_proj:            先按 axis=0 切分, 再转置
    //     lm_head:                  不做分卡处理, 需转置
    //     norm:                     不做分卡处理
    //     embedding:                不做分卡处理
    const std::string& tensor_name = custom_name_list[idx];
    const std::string& weight_name = weight_name_list[idx];

    if (!BaseWeight::IsPipelineNodeWeight(tensor_name)) {
      continue;
    }

    KLLM_LOG_DEBUG << "Start load weight:" << weight_name;
    // filter out some weight
    if (quant_weight_solver_->FilterOutQuantWeight(tensor_name)) {
      continue;
    }

    if (context_->IsChief()) {
      if (Singleton<Environment>::GetInstance()->EmbedTokensUseCpu() &&
          tensor_name.find("embed_positions") != std::string::npos) {
        KLLM_THROW("CPU embedding lookup does not support learned absolute position encoding, please turn it off.");
      }
    }

    bool transpose_first = false;  // 使用 transpose_first 表明转置(若存在)是否在分卡(若存在)之前
    size_t tensor_para_offset = 0;
    std::vector<size_t> weight_shape = weights_loader->GetTensorShape(weight_name);

    // get weight's data ptr
    auto [weight_ptr, weight_size] = weights_loader->GetTensor(weight_name);
    if (weight_ptr == nullptr) {
      KLLM_LOG_DEBUG << fmt::format("The {}'s weight_ptr is null", weight_name);
      continue;
    }
    DataType weight_data_type = weights_loader->GetTensorDataType(weight_name);

#ifdef ENABLE_FP8
    if (quant_weight_solver_->LoadFp8E4m3Scale(tensor_name, weight_shape, weight_data_type, weight_ptr)) {
      continue;
    }
#endif

    if (model_config_.quant_config.enable_moe_int4 &&
        (tensor_name.find(".qweight") != std::string::npos || tensor_name.find(".scales") != std::string::npos ||
         tensor_name.find(".qzeros") != std::string::npos) &&
        tensor_name.find(".experts.") != std::string::npos) {
      quant_weight_solver_->LoadMoeIntQuantWeight(tensor_name, weight_shape, weight_data_type, weight_ptr);
      continue;
    }

    // filter out moe or mla model weight
    // TODO(winminkong): suport mla and moe weight type from fp32 to fp16/bf16 cast
    if (!quant_weight_solver_->IsEnable() &&
        (tensor_name.find(".experts.") != std::string::npos || tensor_name.find("q_b_proj") != std::string::npos ||
         tensor_name.find("kv_a_proj") != std::string::npos || tensor_name.find("kv_b_proj.") != std::string::npos)) {
      moe_weight_data_type_ = weight_data_type;
      continue;
    }

    torch::Tensor weight_cpu_tensor;
    if (tensor_name.find(".weight_scale") == std::string::npos &&
        tensor_name.find(".input_scale") == std::string::npos &&
        tensor_name.find(".e_score_correction_bias") == std::string::npos && weight_data_type == TYPE_FP32) {
      // cast TYPE_FP32 to weight_data_type_.
      auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
      torch::Tensor in = torch::from_blob(weight_ptr, {(int64_t)(weight_size / sizeof(float))}, options);
      weight_size /= sizeof(float) / GetTypeSize(weight_data_type_);
      if (weight_data_type_ == TYPE_FP16) {
        weight_cpu_tensor = in.to(torch::kFloat16);
        weight_ptr = weight_cpu_tensor.data_ptr();
        weight_data_type = weight_data_type_;
      } else if (weight_data_type_ == TYPE_BF16) {
        weight_cpu_tensor = in.to(torch::kBFloat16);
        weight_ptr = weight_cpu_tensor.data_ptr();
        weight_data_type = weight_data_type_;
      } else {
        KLLM_LOG_WARNING << "Weight " << tensor_name << " data type " << GetTypeString(weight_data_type)
                         << " can't cast to type " << GetTypeString(weight_data_type_);
      }
    } else if (weight_data_type != TYPE_FP16 && weight_data_type != TYPE_BF16 &&
               (model_config_.quant_config.method != QUANT_GPTQ || weight_data_type != TYPE_INT32) &&
               weight_data_type != TYPE_FP8_E4M3) {
      KLLM_LOG_WARNING << "Weight " << tensor_name << " data type is " << GetTypeString(weight_data_type);
    }

    // GPT-1 and GPT-2 use Conv1D instead of Linear, we need to do an extra transpose.
    if ((model_config_.type == "openai-gpt" || model_config_.type == "gpt2") &&
        (tensor_name.find("self_attn.W_pack.weight") != std::string::npos ||
         tensor_name.find("o_proj.weight") != std::string::npos ||
         tensor_name.find("gate_proj.weight") != std::string::npos ||
         tensor_name.find("down_proj.weight") != std::string::npos)) {
      weight_cpu_tensor =
          weight_cpu_tensor.view({static_cast<int64_t>(weight_shape[0]), static_cast<int64_t>(weight_shape[1])})
              .t()
              .reshape(-1);
      std::swap(weight_shape[0], weight_shape[1]);
      weight_ptr = weight_cpu_tensor.data_ptr();
    }

    bool is_bias = tensor_name.find(".bias") != std::string::npos;
    // The Add-Bias-Residual Kernel uses the shape[0] of the input tensor to determine whether
    // broadcasting is required.
    if (is_bias && tensor_name.find("norm.bias") == std::string::npos) {
      weight_shape.insert(weight_shape.begin(), 1);
    }

    STATUS_CHECK_RETURN(PrepareLoadOpMeta(tensor_para_offset, weight_shape, transpose_first, tensor_name));

    if (quant_weight_solver_->LoadQuantWeight(tensor_name, weight_shape, weight_data_type, weight_ptr)) {
      continue;
    }

    const int head_num = model_config_.head_num;
    const int num_kv_heads = model_config_.num_key_value_heads;
    // copy host data to device
    const int qkv_offset = CheckQKVWeight(tensor_name, head_num, num_kv_heads);
    // Load qkv weights, name like: "model.layers.(\\d+).self_attn.q_proj.weight",
    // "model.layers.(\\d+).self_attn.k_proj.weight", "model.layers.(\\d+).self_attn.v_proj.weight".
    if (qkv_offset >= 0) {
      // Start get layer_idx
      int layer_idx = 0;
      static const std::regex re("\\d+");
      std::smatch match;
      if (std::regex_search(tensor_name, match, re)) {
        std::string layer_idx_str = match.str(0);
        layer_idx = std::stoi(layer_idx_str);
      }  // End get layer_idx
      // For cla, cla layer only has q_proj weight
      if (qkv_offset == 0 && model_config_.use_cla && model_config_.cla_share_factor != 0 &&
          (layer_idx % model_config_.cla_share_factor != 0) &&
          tensor_name.find("self_attn.q_proj.weight") != std::string::npos) {
        LoadRegularTensor(weight_ptr, tensor_name, weight_shape, weight_data_type, transpose_first, tensor_para_offset,
                          weight_size);
      } else {
        const std::string qkv_name = tensor_name.substr(0, tensor_name.find_last_of('_') - 1) + "query_key_value" +
                                     (is_bias ? ".bias" : ".weight");
        if (!weights_map_.count(qkv_name)) {
          // Bias has been prepended with 1.
          const size_t first_dim = is_bias ? 1 : 0;
          if (qkv_offset == 0) {
            // For q_proj in the GQA scenario, the weight_shape is first transformed into k_proj.
            weight_shape[first_dim] /= head_num / num_kv_heads;
          }
          weight_shape.insert(weight_shape.begin() + first_dim, ((head_num / num_kv_heads) + 2));
          tensor_manager_->AddWeightTensor(qkv_name, weight_shape, weight_data_type);
        }
        weights_data_type_map_[qkv_name] = weight_data_type;
        Tensor& qkv_weight_tensor = weights_map_[qkv_name];
        size_t single_proj_size = qkv_weight_tensor.GetTotalBytes() / (head_num / num_kv_heads + 2);
        const size_t saved_offset = qkv_offset * single_proj_size;
        if (qkv_offset == 0) {
          single_proj_size *= head_num / num_kv_heads;
        }
        tensor_para_offset *= single_proj_size;
        MemcpyAsync(qkv_weight_tensor.GetPtr<void>() + saved_offset, weight_ptr + tensor_para_offset, single_proj_size,
                    MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
      }
    } else if (tensor_name.find("_proj.") != std::string::npos || tensor_name.find("norm.") != std::string::npos ||
               tensor_name == "model.embed_positions.weight" || tensor_name.find("gate.") != std::string::npos ||
               tensor_name.find("gate_proj_bias") != std::string::npos ||
               tensor_name.find("up_proj_bias") != std::string::npos ||
               (tensor_name == "lm_head.weight" && !model_config_.tie_word_embeddings)) {
      if (tensor_name == "model.embed_positions.weight" ||
          (tensor_name == "lm_head.weight" && !model_config_.tie_word_embeddings)) {
        // Load embd only for standalone or distributed master node.
        if (context_->IsChief()) {
          LoadRegularTensor(weight_ptr, tensor_name, weight_shape, weight_data_type, transpose_first,
                            tensor_para_offset, weight_size);
        }
      } else if (use_fused_gate_up_weights &&
                 (tensor_name.find("mlp.up_proj.weight") != std::string::npos ||
                  tensor_name.find("mlp.gate_proj.weight") != std::string::npos ||
                  tensor_name.find("mlp.shared_expert.up_proj.weight") != std::string::npos ||
                  tensor_name.find("mlp.shared_expert.gate_proj.weight") != std::string::npos)) {
        LoadMlpUpGateTensor(weight_ptr, tensor_name, weight_shape, weight_data_type, transpose_first,
                            tensor_para_offset);
      } else {
        LoadRegularTensor(weight_ptr, tensor_name, weight_shape, weight_data_type, transpose_first, tensor_para_offset,
                          weight_size);
      }

    } else if (tensor_name == "model.embed_tokens.weight") {
      // Load lm head only for standalone or distributed master node.
      if (context_->IsChief()) {
        LoadRegularTensor(weight_ptr, tensor_name, weight_shape, weight_data_type, transpose_first, tensor_para_offset,
                          weight_size);
        if (model_config_.tie_word_embeddings) {
          /* When the "tie-word-embeddings" is set to True in the model's config.json, the model's
           * "model.embed_tokens.weight" and "lm_head.weight" share the same data space. Therefore, it is necessary
           * to load the data from "weight_ptr" twice and store it in the corresponding device spaces of the two
           * weights.
           */
          KLLM_LOG_DEBUG << "tie_word_embeddings = true, lm_head.weight = model.embed_tokens.weight";
          if (Singleton<Environment>::GetInstance()->EmbedTokensUseCpu()) {
            tensor_para_offset = rank_;
          } else {
            weight_shape[1] *= tensor_para_size_;
          }
          std::vector<size_t> lm_head_shape = {DivRoundUp(weight_shape[0], tensor_para_size_), weight_shape[1]};
          LoadRegularTensor(weight_ptr, "lm_head.weight", lm_head_shape, weight_data_type, /*transpose_first*/ false,
                            tensor_para_offset, weight_size);
        }
      }
    } else if (tensor_name.find("self_attn.W_pack.weight") != std::string::npos && model_config_.type == "hunyuan" &&
               model_config_.is_moe == false) {
      // dealing  QKV Tensor in hunyuanturbo model
      std::string qkv_name = tensor_name.substr(0, tensor_name.find_last_of('_') - 1) + "query_key_value.weight";
      weights_data_type_map_[qkv_name] = weight_data_type;
      int local_head_num = head_num / tensor_para_size_;
      int groups_per_rank = num_kv_heads / tensor_para_size_;
      int q_heads_per_group = head_num / num_kv_heads;
      size_t head_size = (weight_shape[0] * tensor_para_size_) / ((q_heads_per_group + 2) * num_kv_heads);
      size_t hidden_size = weight_shape[1];

      if (!weights_map_.count(qkv_name)) {
        weight_shape.insert(weight_shape.begin(), (q_heads_per_group + 2));
        weight_shape[1] /= (q_heads_per_group + 2);
        tensor_manager_->AddWeightTensor(qkv_name, weight_shape, weight_data_type);
      }
      Tensor& qkv_weight_tensor = weights_map_[qkv_name];
      size_t type_size = GetTypeSize(weight_data_type);

      int start_group = rank_ * groups_per_rank;
      int end_group = start_group + groups_per_rank;

      size_t k_dst_start = local_head_num * head_size * hidden_size * type_size;
      size_t v_dst_start = k_dst_start + groups_per_rank * head_size * hidden_size * type_size;

      size_t q_head_size = head_size * hidden_size * type_size;
      size_t kv_head_size = head_size * hidden_size * type_size;
      size_t group_size = q_heads_per_group * q_head_size + 2 * kv_head_size;

      for (int group = start_group; group < end_group; group++) {
        size_t src_group_offset = group * group_size;
        int local_group = group - start_group;

        size_t total_group_q_size = q_heads_per_group * q_head_size;
        size_t dst_q_offset = local_group * total_group_q_size;

        MemcpyAsync(qkv_weight_tensor.GetPtr<void>() + dst_q_offset, weight_ptr + src_group_offset, total_group_q_size,
                    MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);

        size_t src_k_offset = src_group_offset + total_group_q_size;
        size_t dst_k_offset = k_dst_start + local_group * kv_head_size;
        MemcpyAsync(qkv_weight_tensor.GetPtr<void>() + dst_k_offset, weight_ptr + src_k_offset, kv_head_size,
                    MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);

        size_t src_v_offset = src_k_offset + kv_head_size;
        size_t dst_v_offset = v_dst_start + local_group * kv_head_size;
        MemcpyAsync(qkv_weight_tensor.GetPtr<void>() + dst_v_offset, weight_ptr + src_v_offset, kv_head_size,
                    MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
      }
    } else if (tensor_name.find("self_attn.W_pack.weight") != std::string::npos) {
      std::string qkv_name = tensor_name.substr(0, tensor_name.find_last_of('_') - 1) + "query_key_value.weight";
      weights_data_type_map_[qkv_name] = weight_data_type;
      if (!weights_map_.count(qkv_name)) {
        weight_shape.insert(weight_shape.begin(), ((head_num / num_kv_heads) + 2));
        weight_shape[1] /= ((head_num / num_kv_heads) + 2);
        tensor_manager_->AddWeightTensor(qkv_name, weight_shape, weight_data_type);
      }

      Tensor& qkv_weight_tensor = weights_map_[qkv_name];
      size_t q_para_offset = rank_;
      size_t kv_para_offset = rank_;
      size_t qkv_pitch = weight_shape[1] * weight_shape[2] * GetTypeSize(weight_data_type);
      size_t q_size = (head_num / num_kv_heads) * qkv_pitch;

      q_para_offset *= q_size;
      kv_para_offset *= qkv_pitch;

      MemcpyAsync(qkv_weight_tensor.GetPtr<void>(), weight_ptr + q_para_offset, q_size, MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);
      MemcpyAsync(qkv_weight_tensor.GetPtr<void>() + q_size, weight_ptr + q_size * tensor_para_size_ + kv_para_offset,
                  qkv_pitch, MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
      MemcpyAsync(qkv_weight_tensor.GetPtr<void>() + q_size + qkv_pitch,
                  weight_ptr + kv_para_offset + (q_size + qkv_pitch) * tensor_para_size_, qkv_pitch,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);

    } else if (tensor_name.find("query_key_value.bias") != std::string::npos) {
      weights_data_type_map_[tensor_name] = weight_data_type;
      tensor_manager_->AddWeightTensor(tensor_name, weight_shape, weight_data_type);

      Tensor& qkv_bias_tensor = weights_map_[tensor_name];
      size_t q_para_offset = rank_;
      size_t kv_para_offset = rank_;
      size_t qkv_pitch = weight_shape[1] / ((head_num / num_kv_heads) + 2) * GetTypeSize(weight_data_type);
      size_t q_size = (head_num / num_kv_heads) * qkv_pitch;

      q_para_offset *= q_size;
      kv_para_offset *= qkv_pitch;

      MemcpyAsync(qkv_bias_tensor.GetPtr<void>(), weight_ptr + q_para_offset, q_size, MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);
      MemcpyAsync(qkv_bias_tensor.GetPtr<void>() + q_size, weight_ptr + kv_para_offset + q_size * tensor_para_size_,
                  qkv_pitch, MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
      MemcpyAsync(qkv_bias_tensor.GetPtr<void>() + q_size + qkv_pitch,
                  weight_ptr + kv_para_offset + (q_size + qkv_pitch) * tensor_para_size_, qkv_pitch,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else {
      KLLM_LOG_DEBUG << "state_dict[" << tensor_name << "] will not be used";
    }
    KLLM_LOG_DEBUG << "Success load weight:" << tensor_name << " on rank " << rank_;
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::PermuteSingleTensorOfQKVWeight(void* qkv_src, void* qkv_dst, Tensor& q_in_tensor,
                                                       Tensor& q_out_tensor, std::vector<size_t>& data_shape,
                                                       std::vector<size_t>& qkv_dst_shape) {
  q_in_tensor.shape = data_shape;
  q_out_tensor.shape = data_shape;
  MemcpyAsync(q_in_tensor.GetPtr<void>(), qkv_src, q_in_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
              context_->GetMemoryManageStreams()[rank_]);
  Permute(q_in_tensor, q_out_tensor, {2, 0, 1}, context_->GetMemoryManageStreams()[rank_]);

#ifdef ENALBE_CUDA
  Memcpy2DAsync(qkv_dst, qkv_dst_shape[1] * sizeof(T), q_out_tensor.GetPtr<void>(), data_shape[1] * sizeof(T),
                data_shape[1] * sizeof(T), data_shape[2], MEMCPY_DEVICE_TO_DEVICE,
                context_->GetMemoryManageStreams()[rank_]);
#else
  // NOTE(karlluo): for ascend, there is a issue that can not use Memcpy2DAsync.
  // will fix it when it work.
  for (size_t row_idx = 0; row_idx < data_shape[2]; ++row_idx) {
    MemcpyAsync(qkv_dst + row_idx * qkv_dst_shape[1] * sizeof(T),
                q_out_tensor.GetPtr<void>() + row_idx * data_shape[1] * sizeof(T), data_shape[1] * sizeof(T),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
  }
#endif

  return Status();
}

template <typename T>
Status CommonWeight<T>::PermuteQKVWeight(Tensor& last_qkv_tensor, Tensor& q_in_tensor, Tensor& q_out_tensor) {
  SetDevice(rank_);

  // src tensor: qkv_weight_tensor[head_num / num_kv_heads + 2, d1, d2]
  // after split: q[head_num / num_kv_heads, d1, d2], k[1, d1, d2], v[1, d1, d2]
  // after permute: q[d2, head_num / num_kv_heads, d1], k[d2, 1, d1], v[d2, 1, d1]
  // dst tensor: last_kv_tensor[d2, head_num / num_kv_heads * d1 + d1 + d1]
  int head_num = model_config_.head_num;
  int num_kv_heads = model_config_.num_key_value_heads;
  for (const auto layer_idx : required_layer_idx_.all) {
    if (model_config_.use_cla && model_config_.cla_share_factor != 0 &&
        (layer_idx % model_config_.cla_share_factor != 0))
      continue;
    std::string qkv_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.query_key_value.weight";
    Tensor& qkv_weight_tensor = weights_map_[qkv_name];
    auto qkv_shape = qkv_weight_tensor.shape;
    std::vector<size_t> q_shape = {1, head_num / num_kv_heads * qkv_shape[1], qkv_shape[2]};
    std::vector<size_t> kv_shape = {1, qkv_shape[1], qkv_shape[2]};
    size_t q_size = q_shape[1] * q_shape[2] * sizeof(T);
    size_t kv_size = kv_shape[1] * kv_shape[2] * sizeof(T);
    std::vector<size_t> qkv_dst_shape = {qkv_shape[2], qkv_shape[0] * qkv_shape[1]};
    last_qkv_tensor.shape = qkv_dst_shape;

    void* qkv_src = qkv_weight_tensor.GetPtr<void>();
    void* qkv_dst = last_qkv_tensor.GetPtr<void>();
    PermuteSingleTensorOfQKVWeight(qkv_src, qkv_dst, q_in_tensor, q_out_tensor, q_shape, qkv_dst_shape);

    qkv_src = qkv_src + q_size;
    qkv_dst = qkv_dst + q_shape[1] * sizeof(T);
    PermuteSingleTensorOfQKVWeight(qkv_src, qkv_dst, q_in_tensor, q_out_tensor, kv_shape, qkv_dst_shape);

    qkv_src = qkv_src + kv_size;
    qkv_dst = qkv_dst + kv_shape[1] * sizeof(T);
    PermuteSingleTensorOfQKVWeight(qkv_src, qkv_dst, q_in_tensor, q_out_tensor, kv_shape, qkv_dst_shape);

    Tensor t = last_qkv_tensor;
    last_qkv_tensor = qkv_weight_tensor;
    TransLayout(t, context_->GetMemoryManageStreams()[rank_]);
    weights_map_[qkv_name] = t;
    KLLM_LOG_INFO << fmt::format("weights_map[{}] = new Tensor({})", qkv_name, Vector2Str<size_t>(t.shape));
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::CommonPermuteWeight(const std::string& origin_tensor_name, Tensor& swap_tensor) {
  Tensor& origin_mlp_tensor = weights_map_[origin_tensor_name];
  Permute(origin_mlp_tensor, swap_tensor, {1, 0}, context_->GetMemoryManageStreams()[rank_]);
  Tensor t = swap_tensor;
  swap_tensor = origin_mlp_tensor;
  t.shape = {origin_mlp_tensor.shape[1], origin_mlp_tensor.shape[0]};
  TransLayout(t, context_->GetMemoryManageStreams()[rank_]);
  weights_map_[origin_tensor_name] = t;
  return Status();
}

template <typename T>
Status CommonWeight<T>::PermuteMLPWeight(Tensor& last_down_tensor, Tensor& last_gate_tensor,
                                         const std::string& weight_name_format,
                                         const std::unordered_set<int16_t>& layers, const bool is_weight_scale) {
  SetDevice(rank_);
  const std::string weight_type = is_weight_scale ? "weight_scale_inv" : "weight";
  for (const auto layer_idx : layers) {
    std::string down_proj_name = fmt::format(weight_name_format, layer_idx, "down_proj", weight_type);
    CommonPermuteWeight(down_proj_name, last_down_tensor);
    std::string gate_up_proj_name = fmt::format(weight_name_format, layer_idx, "gate_up_proj", weight_type);
    if (weights_map_.find(gate_up_proj_name) != weights_map_.end()) {  // gate_up proj is optional
      CommonPermuteWeight(gate_up_proj_name, last_gate_tensor);
    } else {
      std::string gate_proj_name = fmt::format(weight_name_format, layer_idx, "gate_proj", weight_type);
      CommonPermuteWeight(gate_proj_name, last_gate_tensor);

      std::string up_proj_name = fmt::format(weight_name_format, layer_idx, "up_proj", weight_type);
      // up_proj is optional
      if (weights_map_.find(up_proj_name) != weights_map_.end()) {
        CommonPermuteWeight(up_proj_name, last_down_tensor);
      }
    }
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::PermuteOutputProjectWeight(Tensor& last_o_proj_tensor) {
  SetDevice(rank_);
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string o_proj_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight";
    CommonPermuteWeight(o_proj_name, last_o_proj_tensor);
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::LoadRegularTensor(void* weight_ptr, std::string tensor_name, std::vector<size_t>& weight_shape,
                                          DataType& weight_data_type, bool transpose_first, size_t tensor_para_offset,
                                          size_t& weight_size) {
  int tensor_para_size = tensor_para_size_;
  if (tensor_name.find("o_proj") != std::string::npos) {
    tensor_para_size = Singleton<Environment>::GetInstance()->GetAttentionTensorParallel();
  }

  tensor_manager_->AddWeightTensor(tensor_name, weight_shape, weight_data_type);
  weights_data_type_map_[tensor_name] = weight_data_type;
  if (transpose_first) {
    size_t src_pitch = weights_map_[tensor_name].shape[1] * tensor_para_size * GetTypeSize(weight_data_type);
    size_t dst_pitch = weights_map_[tensor_name].shape[1] * GetTypeSize(weight_data_type);
    tensor_para_offset *= dst_pitch;
    Memcpy2DAsync(weights_map_[tensor_name].GetPtr<void>(), dst_pitch, weight_ptr + tensor_para_offset, src_pitch,
                  dst_pitch, weights_map_[tensor_name].shape[0], MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);
  } else {
    tensor_para_offset *= weights_map_[tensor_name].GetTotalBytes();
    SetDevice(rank_);
    size_t sub_bytes = 0;
    if (rank_ == (tensor_para_size - 1) && tensor_name == "lm_head.weight") {
      sub_bytes = weights_map_[tensor_name].GetTotalBytes() * tensor_para_size - weight_size;
    }
    if (model_config_.type == "chatglm" && tensor_name.find("mlp.gate_proj.weight") != std::string::npos) {
      size_t gate_para_offset = rank_;
      size_t src_pitch = weight_shape[0] * weight_shape[1] / 2 * tensor_para_size * GetTypeSize(weight_data_type);
      size_t dst_pitch = weight_shape[0] * weight_shape[1] / 2 * GetTypeSize(weight_data_type);
      gate_para_offset *= dst_pitch;
      Memcpy2DAsync(weights_map_[tensor_name].GetPtr<void>(), dst_pitch, weight_ptr + gate_para_offset, src_pitch,
                    dst_pitch, 2, MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else if (tensor_name.find("gate_up_proj.weight") != std::string::npos && model_config_.type == "hunyuan" &&
               model_config_.is_moe == false) {
      KLLM_LOG_DEBUG << "dealing with reverse gate up";
      size_t half_len = weights_map_[tensor_name].GetTotalBytes() / 2;
      size_t all_half = half_len * tensor_para_size;

      MemcpyAsync(weights_map_[tensor_name].GetPtr<void>(), weight_ptr + all_half + half_len * rank_, half_len,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);

      MemcpyAsync(weights_map_[tensor_name].GetPtr<void>() + half_len, weight_ptr + half_len * rank_, half_len,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);

    } else {
      MemcpyAsync(weights_map_[tensor_name].GetPtr<void>(), weight_ptr + tensor_para_offset,
                  weights_map_[tensor_name].GetTotalBytes() - sub_bytes, MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);
    }
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::LoadMlpUpGateTensor(void* weight_ptr, std::string tensor_name,
                                            std::vector<size_t>& weight_shape, DataType& weight_data_type,
                                            bool transpose_first, size_t tensor_para_offset) {
  if (transpose_first) {
    KLLM_THROW("not implemented");
  }
  std::string another_weights_name;
  std::string up_gate_weights_name;
  int concat_offset = 0;
  std::string replacement = "gate_up_proj";
  if (tensor_name.find("gate_proj") != std::string::npos) {
    concat_offset = 0;
    static const std::regex pattern("gate_proj");
    up_gate_weights_name = std::regex_replace(tensor_name, pattern, replacement);
  } else if (tensor_name.find("up_proj") != std::string::npos) {
    concat_offset = 1;
    static const std::regex pattern("up_proj");
    up_gate_weights_name = std::regex_replace(tensor_name, pattern, replacement);
  } else {
    KLLM_THROW("tensor name should contain gate_proj or up_proj");
  }

  // concat gate and up weights
  weight_shape[0] *= 2;
  tensor_manager_->AddWeightTensor(up_gate_weights_name, weight_shape, weight_data_type);
  weights_data_type_map_[up_gate_weights_name] = weight_data_type;

  size_t total_bytes = weights_map_[up_gate_weights_name].GetTotalBytes() / 2;
  tensor_para_offset *= total_bytes;
  SetDevice(rank_);
  MemcpyAsync(weights_map_[up_gate_weights_name].GetPtr<void>() + concat_offset * total_bytes,
              weight_ptr + tensor_para_offset, total_bytes, MEMCPY_HOST_TO_DEVICE,
              context_->GetMemoryManageStreams()[rank_]);
  return Status();
}

template <typename T>
Status CommonWeight<T>::ConvertMLPWeight(bool is_weight_scale) {
  return ConvertMLPWeight("model.layers.{}.mlp.{}.{}", required_layer_idx_.dense, is_weight_scale);
}

template <typename T>
Status CommonWeight<T>::ConvertMLPWeight(const std::string& weight_name_format,
                                         const std::unordered_set<int16_t>& layers, bool is_weight_scale) {
  if (layers.empty()) {
    return Status();
  }
  const std::string weight_type = is_weight_scale ? "weight_scale_inv" : "weight";
  const auto permute_idx = *(layers.begin());  // a random idx, just for shape
  tensor_manager_->CreateTensorWithSameShape(fmt::format(weight_name_format, permute_idx, "down_proj", weight_type),
                                             "empty_down_tensor");

  const std::string gate_up_proj_name = fmt::format(weight_name_format, permute_idx, "gate_up_proj", weight_type);
  if (weights_map_.find(gate_up_proj_name) != weights_map_.end()) {
    tensor_manager_->CreateTensorWithSameShape(gate_up_proj_name, "empty_gate_tensor");
  } else {
    tensor_manager_->CreateTensorWithSameShape(fmt::format(weight_name_format, permute_idx, "gate_proj", weight_type),
                                               "empty_gate_tensor");
  }
  Tensor& last_down_tensor = weights_map_["empty_down_tensor"];
  Tensor& last_gate_tensor = weights_map_["empty_gate_tensor"];
  STATUS_CHECK_RETURN(
      PermuteMLPWeight(last_down_tensor, last_gate_tensor, weight_name_format, layers, is_weight_scale));

  weights_map_.erase("empty_down_tensor");
  weights_map_.erase("empty_gate_tensor");
  return Status();
}

template <typename T>
Status CommonWeight<T>::ConvertLmheadTensor() {
  // Worker node do not need lm head.
  if (context_->IsChief()) {
    // permute lm_head: permute(1, 0)
    tensor_manager_->CreateTensorWithSameShape("lm_head.weight", "empty_lm_head_tensor");
    Tensor& lm_head_transpose_tensor = weights_map_["empty_lm_head_tensor"];
    CommonPermuteWeight("lm_head.weight", lm_head_transpose_tensor);
    weights_map_.erase("empty_lm_head_tensor");
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::ConvertNextnProjTensor() {
  if (pipeline_config_.lower_nextn_layer_idx < static_cast<int>(model_config_.num_layer)) {
    return Status();
  }

  for (auto layer_idx = pipeline_config_.lower_nextn_layer_idx; layer_idx <= pipeline_config_.upper_nextn_layer_idx;
       ++layer_idx) {
    const std::string tensor_name = fmt::format("model.layers.{}.eh_proj.weight", layer_idx);
    const std::string empty_tensor_name = fmt::format("empty_{}", tensor_name);
    tensor_manager_->CreateTensorWithSameShape(tensor_name, empty_tensor_name);
    Tensor& transpose_tensor = weights_map_[empty_tensor_name];
    CommonPermuteWeight(tensor_name, transpose_tensor);
    weights_map_.erase(empty_tensor_name);
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::ReshapeQkvTensor() {
  if (!model_config_.use_mla) {
    for (const auto layer_idx : required_layer_idx_.all) {
      std::string qkv_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.query_key_value.weight";
      Tensor& tensor = weights_map_[qkv_name];
      if (tensor.shape.size() == 3) {
        tensor.shape = {tensor.shape[0] * tensor.shape[1], tensor.shape[2]};
      }
    }
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::ConvertQkvTensor() {
  SetDevice(rank_);
  // deepseek use MLA, it doesn't have qkv_tensor
  if (!model_config_.use_mla) {
    // permute qkv_tensor: permute((2, 0, 1))
    tensor_manager_->CreateTensorWithSameShape(
        "model.layers." + std::to_string(pipeline_config_.lower_layer_idx) + ".self_attn.query_key_value.weight",
        "empty_qkv_tensor");
    std::string qkv_name =
        "model.layers." + std::to_string(pipeline_config_.lower_layer_idx) + ".self_attn.query_key_value.weight";
    auto shape = weights_map_[qkv_name].shape;
    auto dtype = weights_map_[qkv_name].dtype;
    shape[0] = shape[0] * model_config_.head_num / (model_config_.head_num + 2 * model_config_.num_key_value_heads);
    tensor_manager_->AddWeightTensor("empty_q_in_tensor", shape, dtype);
    tensor_manager_->AddWeightTensor("empty_q_out_tensor", shape, dtype);
    Tensor& last_qkv_tensor = weights_map_["empty_qkv_tensor"];
    Tensor& q_in_tensor = weights_map_["empty_q_in_tensor"];
    Tensor& q_out_tensor = weights_map_["empty_q_out_tensor"];
    STATUS_CHECK_RETURN(PermuteQKVWeight(last_qkv_tensor, q_in_tensor, q_out_tensor));
    weights_map_.erase("empty_qkv_tensor");
    weights_map_.erase("empty_q_in_tensor");
    weights_map_.erase("empty_q_out_tensor");
  }
  return Status();
}

template <typename T>
Status CommonWeight<T>::ConvertOprojTensor() {
  // permute o_proj: permute(1, 0)
  tensor_manager_->CreateTensorWithSameShape(
      "model.layers." + std::to_string(pipeline_config_.lower_layer_idx) + ".self_attn.o_proj.weight",
      "empty_o_proj_tensor");
  Tensor& last_o_proj_tensor = weights_map_["empty_o_proj_tensor"];
  STATUS_CHECK_RETURN(PermuteOutputProjectWeight(last_o_proj_tensor));
  weights_map_.erase("empty_o_proj_tensor");
  return Status();
}

template <typename T>
Status CommonWeight<T>::ConvertCommonTensor(int hidden_units, int inter_size, int vocab_size) {
  SetDevice(rank_);

  // permute qkv_tensor: permute((2, 0, 1))
  ConvertQkvTensor();

  // permute gate_proj, up_proj, down_proj: permute(1, 0)
  if (!model_config_.is_moe || !required_layer_idx_.dense.empty()) {
    ConvertMLPWeight(false);
  }

  ConvertOprojTensor();

  ConvertLmheadTensor();
  return Status();
}

template <typename T>
bool CommonWeight<T>::IsLoaded() {
  return weights_had_loaded_;
}

template <typename T>
std::string CommonWeight<T>::ConcatLayerName(std::string layer_flag, int& layer_index, bool is_bias) {
  std::string layer_name =
      "model.layers." + std::to_string(layer_index) + "." + layer_flag + (is_bias ? ".bias" : ".weight");
  return layer_name;
}

template <typename T>
Tensor CommonWeight<T>::GetModelWeights(const std::string& weight_name) {
  if (!weights_map_.count(weight_name)) {
    KLLM_LOG_WARNING << fmt::format("weight name {} not in weights map", weight_name);
    return Tensor();
  }
  return weights_map_[weight_name];
}

template <typename T>
Status CommonWeight<T>::GetModelInfo(const ModelConfig& model_config) {
  weight_data_type_ = model_config.weight_data_type;
  model_name_ = model_config.name;
  tensor_para_size_ = model_config.tensor_para_size;
  expert_world_size_ = model_config.expert_world_size;
  expert_para_size_ = model_config.expert_para_size;
  global_expert_para_size_ = expert_world_size_ * expert_para_size_;
  enable_full_shared_expert_ = model_config.enable_full_shared_expert;
  return Status();
}

template <typename T>
void CommonWeight<T>::ProcessWeights() {
  const int hidden_units = model_config_.hidden_units;
  const int inter_size = model_config_.inter_size;
  const int vocab_size = model_config_.vocab_size;
  // Convert of BFP16 and FP16
  if (model_config_.weight_data_type == TYPE_FP16 || model_config_.weight_data_type == TYPE_BF16) {
    for (auto& data_type_iter : weights_data_type_map_) {
      if (data_type_iter.second == TYPE_FP16 || data_type_iter.second == TYPE_BF16) {
        Tensor& tensor = weights_map_[data_type_iter.first];
        tensor.dtype = data_type_iter.second;
        SetDevice(rank_);
        CastInplace(tensor, model_config_.weight_data_type, context_->GetMemoryManageStreams()[rank_]);
        tensor.dtype = model_config_.weight_data_type;
      }
    }
  }

  // Load embed only for standalone or distributed master node.
  if (context_->IsChief()) {
    if (Singleton<Environment>::GetInstance()->EmbedTokensUseCpu()) {
      KLLM_LOG_INFO << "Enable EmbedTokensUseCpu";
      auto weight_name = "model.embed_tokens.weight";
      Tensor& tensor = weights_map_[weight_name];
      Tensor cpu_tensor(MemoryLocation::LOCATION_HOST, tensor.dtype, tensor.shape);
      MemcpyAsync(cpu_tensor.GetPtr<void>(), tensor.GetPtr<void>(), cpu_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                  context_->GetMemoryManageStreams()[rank_]);
      weights_map_.insert_or_assign(weight_name, cpu_tensor);
      StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
    }
  }

  if (model_config_.quant_config.enable_moe_int4) {
    std::vector<std::string> needed_slove_weights_name = {"mlp.experts.down_proj", "mlp.experts.up_gate_proj"};
    quant_weight_solver_->AutoPackAndBindGroupTensor(needed_slove_weights_name);
  }

  if (quant_weight_solver_->IsEnable()) {
    quant_weight_solver_->ConvertGroupTensor();
    ConvertNextnProjTensor();
  } else if (model_config_.is_quant &&
             (model_config_.quant_config.method == QUANT_FP8_E4M3 ||
              model_config_.quant_config.method == QUANT_BLOCK_FP8_E4M3) &&
             model_config_.quant_config.is_checkpoint_fp8_serialized) {
    ConvertLmheadTensor();
    ConvertNextnProjTensor();
    if (weights_map_.find("model.layers.0.self_attn.o_proj.weight") != weights_map_.end() &&
        weights_map_["model.layers.0.self_attn.o_proj.weight"].dtype != TYPE_FP8_E4M3) {
      ConvertOprojTensor();
    }
    if (weights_map_.find("model.layers.0.self_attn.query_key_value.weight") != weights_map_.end() &&
        weights_map_["model.layers.0.self_attn.query_key_value.weight"].dtype != TYPE_FP8_E4M3) {
      ConvertQkvTensor();
    } else {
      ReshapeQkvTensor();
    }
  } else {  // roll back to common weight slover
    ConvertCommonTensor(hidden_units, inter_size, vocab_size);
  }

  // We use vocab_size to determine whether it is the Baichuan2 model.
  // If vocab_size is equal to 125,696, then it is the Baichuan2 model.
  // And Unlike Baichuan1, the Baichuan2 model requires normalizing the head
  // weights. Refer to repo:
  // https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat commit:
  // 84603cde5ebffb6084e476cfaeceaf0b8b91fe54 file: modeling_baichuan.py#L508
  if (model_config_.type == "baichuan" && vocab_size == 125696) {
    if (weights_data_type_map_.find("lm_head.weight") != weights_data_type_map_.end()) {
      Tensor& tensor = weights_map_["lm_head.weight"];
      SetDevice(rank_);
      StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
      torch::ScalarType torch_dtype;
      if (tensor.dtype == DataType::TYPE_FP32) {
        torch_dtype = torch::kFloat32;
      } else if (tensor.dtype == DataType::TYPE_FP16) {
        torch_dtype = torch::kFloat16;
#ifdef ENABLE_BFLOAT16
      } else if (tensor.dtype == DataType::TYPE_BF16) {
        torch_dtype = torch::kBFloat16;
#endif
      } else {
        KLLM_THROW(fmt::format("Unsupported Tensor type {}.", tensor.dtype));
      }
      auto options = torch::TensorOptions().device(torch::kCUDA, rank_).dtype(torch_dtype);
      torch::Tensor in =
          torch::from_blob(tensor.GetPtr<void>(), {(int64_t)tensor.shape[0], (int64_t)tensor.shape[1]}, options);
      auto out = torch::nn::functional::normalize(in, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
      MemcpyAsync(tensor.GetPtr<void>(), out.data_ptr(), sizeof(T) * tensor.shape[0] * tensor.shape[1],
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    }
  }
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);

  // The chatglm model does not contain up_proj.weight, but its gate_proj.weight
  // includes up_proj.weight. We need chunk them for the convenience of later
  // decoding.
  if (!quant_weight_solver_->IsEnable() && model_config_.type == "chatglm") {
    ChunkGateWeight();
  }
  if (model_config_.is_quant && model_config_.quant_config.method == QUANT_FP8_E4M3) {
    if (model_config_.quant_config.is_checkpoint_fp8_serialized == false) {
#ifdef ENABLE_FP8
      quant_weight_solver_->ConvertFp8E4m3();
#else
      KLLM_THROW("Device not support Fp8");
#endif
    } else {
      int head_num = model_config_.head_num;
      int num_kv_heads = model_config_.num_key_value_heads;
#ifdef ENABLE_FP8
      if (!model_config_.quant_config.is_fp8_blockwise) {
        quant_weight_solver_->BindFp8E4m3Scale(head_num, num_kv_heads);
      }
#else
      KLLM_THROW("Device not support Fp8");
#endif
    }
  }
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
}

template <typename T>
void CommonWeight<T>::ChunkGateWeight() {
  SetDevice(rank_);
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string gate_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight";
    std::string up_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight";
    Tensor& gate_weight = weights_map_[gate_proj_name];
    std::string gate_proj_name_bk = gate_proj_name + "_bk";

    if (model_config_.is_quant && model_config_.quant_config.method == QUANT_FP8_E4M3 &&
        model_config_.quant_config.is_checkpoint_fp8_serialized == true) {
      std::vector<size_t> shape = {gate_weight.shape[0] / 2, gate_weight.shape[1]};
      size_t size = gate_weight.GetTotalBytes() / 2;
      tensor_manager_->AddWeightTensor(gate_proj_name_bk, shape, gate_weight.dtype);
      tensor_manager_->AddWeightTensor(up_proj_name, shape, gate_weight.dtype);
      MemcpyAsync(weights_map_[gate_proj_name_bk].GetPtr<void>(), gate_weight.GetPtr<void>(), size,
                  MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
      MemcpyAsync(weights_map_[up_proj_name].GetPtr<void>(), gate_weight.GetPtr<void>() + size, size,
                  MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    } else {
      tensor_manager_->AddWeightTensor(gate_proj_name_bk, {gate_weight.shape[0], gate_weight.shape[1] / 2},
                                       gate_weight.dtype);
      tensor_manager_->AddWeightTensor(up_proj_name, {gate_weight.shape[0], gate_weight.shape[1] / 2},
                                       gate_weight.dtype);
      size_t spitch = gate_weight.shape[1] * GetTypeSize(gate_weight.dtype);
      size_t dpitch = (gate_weight.shape[1] / 2) * GetTypeSize(gate_weight.dtype);
      Memcpy2DAsync(weights_map_[gate_proj_name_bk].GetPtr<void>(), dpitch, gate_weight.GetPtr<void>(), spitch, dpitch,
                    gate_weight.shape[0], MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
      Memcpy2DAsync(weights_map_[up_proj_name].GetPtr<void>(), dpitch, gate_weight.GetPtr<void>() + dpitch, spitch,
                    dpitch, gate_weight.shape[0], MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    }
    weights_map_[gate_proj_name] = weights_map_[gate_proj_name_bk];
    weights_map_.erase(gate_proj_name_bk);
  }
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
}

template <typename T>
void CommonWeight<T>::PrintDebugMessage() {
  std::string debug_message = "CommonWeight weights:";
  for (const auto& [k, v] : weights_map_) {
    debug_message += fmt::format(" [{}, shape: {}, dtype: {}]", k, Vector2Str(std::vector<size_t>(v.shape)), v.dtype);
  }
  KLLM_LOG_INFO << debug_message;
}
template class CommonWeight<float>;
template class CommonWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class CommonWeight<bfloat16>;
#endif

}  // namespace ksana_llm
