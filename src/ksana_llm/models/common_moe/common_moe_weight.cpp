/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/common_moe/common_moe_weight.h"

#include <numeric>
#include <regex>

#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
CommonMoeWeight<T>::CommonMoeWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                                    std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, runtime_config, rank, context) {
  SetDevice(rank_);
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
  KLLM_LOG_INFO << fmt::format(
      "node number = {}, node rank = {}, expert_para_size on each node = {}. experts number on each gpu rank = {}",
      expert_world_size_, expert_node_rank, expert_para_size_, num_experts_per_rank_);

  for (size_t i = expert_start_id; i < expert_end_id; ++i) {
    expert_map_[i] = i - expert_start_id;
  }
  KLLM_LOG_INFO << fmt::format("In Rank {}, valid expert range is from {} to {}", rank_, expert_start_id,
                               expert_end_id - 1);
}

template <typename T>
Status CommonMoeWeight<T>::GetExpertsIdx(const std::string& expert_name, int& layer_idx, int& expert_idx) {
  // Get the index of the moe layer and the index of each expert
  static const std::regex re(R"(\d+)");
  std::sregex_iterator next(expert_name.begin(), expert_name.end(), re);
  std::sregex_iterator end;
  if (next != end) {
    std::smatch match = *next;
    layer_idx = std::stoi(match.str());
    if (++next != end) {
      match = *next;
      expert_idx = std::stoi(match.str());
    } else {
      expert_idx = -1;
    }
  } else {
    layer_idx = -1;
    return Status(RET_INIT_FAILED);
  }
  return Status();
}

template <typename T>
Status CommonMoeWeight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                               const std::vector<std::string>& weight_name_list,
                                               const std::vector<std::string>& custom_name_list) {
  CommonWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  SetDevice(rank_);

  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    // For each layer of the model, experts at the same position in 'up' and 'gate' need to be concatenated together
    // and named 'up_gate.weight. And for each layer, all corresponding experts from 'up_gate' and 'down' need to be
    // stacked to form one expert weight.
    const std::string& tensor_name = custom_name_list[idx];
    const std::string& weight_name = weight_name_list[idx];

    if (!BaseWeight::IsPipelineNodeWeight(tensor_name)) {
      continue;
    }

    if (quant_weight_solver_->IsEnable() || model_config_.quant_config.enable_moe_int4) {
      break;
    }
    auto [weight_ptr, weight_size] = weights_loader->GetTensor(weight_name);
    DataType weight_data_type = weights_loader->GetTensorDataType(weight_name);
#ifdef ENABLE_FP8
    std::vector<size_t> weight_shape = weights_loader->GetTensorShape(weight_name);
    if (model_config_.quant_config.is_fp8_blockwise &&
        quant_weight_solver_->LoadMoeFp8E4m3BlockWiseScale(tensor_name, weight_shape, weight_data_type, weight_ptr)) {
      continue;
    }
#endif
    if (model_config_.quant_config.method == QUANT_FP8_E4M3 && tensor_name.find("input_scale") != std::string::npos ||
        tensor_name.find("weight_scale") != std::string::npos) {
      continue;
    }

    if (tensor_name.find(".experts.") == std::string::npos) {
      continue;
    }

    int layer_idx = -1, expert_idx = -1;
    STATUS_CHECK_RETURN(GetExpertsIdx(tensor_name, layer_idx, expert_idx));
    if (expert_idx >= 0 && expert_map_[expert_idx] < 0) {
      // Skip load weight when the expert_id will be not used in current rank.
      continue;
    }

    // cast TYPE_FP32 to weight_data_type_.
    if (weight_data_type == TYPE_FP32) {
      torch::Tensor weight_cpu_tensor;
      const auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
      torch::Tensor in = torch::from_blob(weight_ptr, {static_cast<int64_t>(weight_size / sizeof(float))}, options);
      weight_size /= sizeof(float) / GetTypeSize(weight_data_type_);
      if (weight_data_type_ == TYPE_FP16) {
        weight_cpu_tensor = in.to(torch::kFloat16);
      } else if (weight_data_type_ == TYPE_BF16) {
        weight_cpu_tensor = in.to(torch::kBFloat16);
      } else {
        KLLM_LOG_WARNING << "Weight " << tensor_name << " data type " << weight_data_type << " can't cast to type "
                         << weight_data_type_;
      }
      weight_ptr = weight_cpu_tensor.data_ptr();
      weight_data_type = weight_data_type_;
    } else if (weight_data_type != TYPE_FP16 && weight_data_type != TYPE_BF16 &&
               (model_config_.quant_config.method != QUANT_GPTQ || weight_data_type != TYPE_INT32) &&
               weight_data_type != TYPE_FP8_E4M3) {
      KLLM_LOG_WARNING << "Weight " << tensor_name << " data type is " << weight_data_type;
    }

    // determine weight shape
    const size_t hidden_units = model_config_.hidden_units;
    const bool use_vllm_moe = model_config_.moe_config.use_vllm_moe;

    const size_t moe_inter_size = model_config_.moe_config.moe_inter_size;
    const size_t moe_inter_size_per_rank =
        DivRoundUp(moe_inter_size, runtime_config_.parallel_basic_config.moe_tensor_para_size);
    if (tensor_name.find(".experts.gate_up_proj.") != std::string::npos) {
      // cpu weight is [config.num_experts, hidden_units, 2 * moe_inter_size]
      // Create gpu up_gate_proj Tensor
      std::vector<size_t> up_gate_experts_shape = {num_experts_per_rank_, hidden_units, 2 * moe_inter_size_per_rank};
      std::string up_gate_experts_name =
          "model.layers." + std::to_string(layer_idx) + ".mlp.experts.up_gate_proj.weight";
      if (weights_map_.find(tensor_name) == weights_map_.end()) {
        KLLM_LOG_INFO << fmt::format("will create {} in rank {}", up_gate_experts_name, rank_);
        tensor_manager_->AddWeightTensor(up_gate_experts_name, up_gate_experts_shape, weight_data_type);
        weights_data_type_map_[up_gate_experts_name] = weight_data_type;
      }
      Tensor& tensor = weights_map_[up_gate_experts_name];
      // view as Memcpy2D
      // from cpu[config.num_experts * hidden_units, moe_inter_size + moe_inter_size]
      // to gpu [num_experts * hidden_units, moe_inter_size_per_rank + moe_inter_size_per_rank]
      const size_t start_expert =
          std::distance(expert_map_.begin(), std::find(expert_map_.begin(), expert_map_.end(), 0));
      const size_t height = num_experts_per_rank_ * hidden_units;
      const size_t width = moe_inter_size_per_rank * GetTypeSize(weight_data_type);
      const size_t src_width = moe_inter_size * GetTypeSize(weight_data_type);
      const size_t src_pitch = 2 * src_width;
      const size_t dst_pitch = 2 * width;
      // gate
      void* src_ptr = weight_ptr + start_expert * hidden_units * src_pitch + 0 * src_width + rank_ * width;
      void* dst_ptr = tensor.GetPtr<void>() + 1 * width;
      Memcpy2DAsync(dst_ptr, dst_pitch, src_ptr, src_pitch, width, height, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      // up
      src_ptr = weight_ptr + start_expert * hidden_units * src_pitch + 1 * src_width + rank_ * width;
      dst_ptr = tensor.GetPtr<void>() + 0 * width;
      Memcpy2DAsync(dst_ptr, dst_pitch, src_ptr, src_pitch, width, height, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
    } else if (tensor_name.find(".experts.down_proj.") != std::string::npos) {
      // cpu weight is [config.num_experts, moe_inter_size, hidden_units]
      // Create gpu down_proj Tensor
      if (weights_map_.find(tensor_name) == weights_map_.end()) {
        KLLM_LOG_INFO << fmt::format("will create {} in rank {}", tensor_name, rank_);
        tensor_manager_->AddWeightTensor(tensor_name, {num_experts_per_rank_, moe_inter_size_per_rank, hidden_units},
                                         weight_data_type);
        weights_data_type_map_[tensor_name] = weight_data_type;
      }
      const Tensor& tensor = weights_map_[tensor_name];

      // view as Memcpy2D
      // from cpu [config.num_experts, moe_inter_size * hidden_units]
      // to gpu [num_experts, moe_inter_size_per_rank * hidden_units]
      const size_t start_expert =
          std::distance(expert_map_.begin(), std::find(expert_map_.begin(), expert_map_.end(), 0));
      const size_t height = num_experts_per_rank_;
      const size_t width = moe_inter_size_per_rank * hidden_units * GetTypeSize(weight_data_type);
      const size_t src_pitch = moe_inter_size * hidden_units * GetTypeSize(weight_data_type);
      const size_t dst_pitch = width;
      void* const src_ptr = weight_ptr + start_expert * src_pitch + rank_ * dst_pitch;
      void* const dst_ptr = tensor.GetPtr<void>();
      Memcpy2DAsync(dst_ptr, dst_pitch, src_ptr, src_pitch, width, height, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
    } else if (tensor_name.find(".up_proj.") != std::string::npos ||
               tensor_name.find(".gate_proj.") != std::string::npos) {
      if (expert_idx < 0) {
        continue;
      }
      expert_idx = expert_map_[expert_idx];
      const std::string up_gate_experts_name =
          "model.layers." + std::to_string(layer_idx) + ".mlp.experts.up_gate_proj.weight";
      // Create up_gate_proj Tensor
      if (weights_map_.find(up_gate_experts_name) == weights_map_.end()) {
        KLLM_LOG_INFO << fmt::format("will create {} in rank {}", up_gate_experts_name, rank_);
        tensor_manager_->AddWeightTensor(
            up_gate_experts_name, {num_experts_per_rank_, moe_inter_size_per_rank * 2, hidden_units}, weight_data_type);
        weights_data_type_map_[up_gate_experts_name] = weight_data_type;
      }

      const size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(weight_data_type);
      const size_t double_expert_pitch = expert_pitch * 2;
      const size_t src_upgate_offset = runtime_config_.parallel_basic_config.moe_tensor_para_size > 1
                                           ? (rank_ / expert_para_size_) * expert_pitch
                                           : 0;
      const Tensor& up_gate_experts_tensor = weights_map_[up_gate_experts_name];
      if (tensor_name.find(".up_proj.") != std::string::npos) {
        MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx) * double_expert_pitch +
                        (use_vllm_moe ? expert_pitch : 0),
                    weight_ptr + src_upgate_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      } else if (tensor_name.find(".gate_proj.") != std::string::npos) {
        MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx) * double_expert_pitch +
                        (use_vllm_moe ? 0 : expert_pitch),
                    weight_ptr + src_upgate_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      }
    } else if (tensor_name.find(".down_proj.") != std::string::npos) {
      if (expert_idx < 0) {
        continue;
      }
      expert_idx = expert_map_[expert_idx];
      const std::string down_experts_name =
          "model.layers." + std::to_string(layer_idx) + ".mlp.experts.down_proj.weight";
      if (weights_map_.find(down_experts_name) == weights_map_.end()) {
        tensor_manager_->AddWeightTensor(
            down_experts_name, {num_experts_per_rank_, hidden_units, moe_inter_size_per_rank}, weight_data_type);
        weights_data_type_map_[down_experts_name] = weight_data_type;
      }

      const size_t dst_pitch = moe_inter_size_per_rank * GetTypeSize(weight_data_type);
      const size_t src_pitch = moe_inter_size_per_rank * runtime_config_.parallel_basic_config.moe_tensor_para_size *
                               GetTypeSize(weight_data_type);
      const size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(weight_data_type);
      const size_t src_down_offset =
          runtime_config_.parallel_basic_config.moe_tensor_para_size > 1 ? (rank_ / expert_para_size_) * dst_pitch : 0;
      const Tensor& down_expert_tensor = weights_map_[down_experts_name];
      Memcpy2DAsync(down_expert_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx) * expert_pitch, dst_pitch,
                    weight_ptr + src_down_offset, src_pitch, dst_pitch, hidden_units, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
    }
    KLLM_LOG_DEBUG << "Success load weight:" << tensor_name << " on rank " << rank_;
  }  // end for loop

  return Status();
}

template <typename T>
Status CommonMoeWeight<T>::PermuteGatingWeight(Tensor& last_gating_tensor) {
  SetDevice(rank_);
  for (const auto layer_idx : required_layer_idx_.moe) {
    std::string gating_name = "model.layers." + std::to_string(layer_idx) + ".mlp.gate.weight";
    CommonWeight<T>::CommonPermuteWeight(gating_name, last_gating_tensor);
  }
  return Status();
}

template <typename T>
Status CommonMoeWeight<T>::ConvertShareMLPWeight(bool is_weight_scale) {
  return CommonWeight<T>::ConvertMLPWeight("model.layers.{}.mlp.shared_expert.{}.{}", required_layer_idx_.moe,
                                           is_weight_scale);
}

template <typename T>
void CommonMoeWeight<T>::ProcessWeights() {
  // Permute Gating Weight
  if (!required_layer_idx_.moe.empty()) {
    const auto permute_idx = *(required_layer_idx_.moe.begin());  // a random idx, just for shape
    tensor_manager_->CreateTensorWithSameShape("model.layers." + std::to_string(permute_idx) + ".mlp.gate.weight",
                                               "empty_gating_tensor");
    Tensor& last_gating_tensor = weights_map_["empty_gating_tensor"];
    PermuteGatingWeight(last_gating_tensor);
    weights_map_.erase("empty_gating_tensor");
  }

  if (model_config_.has_shared_experts) {
    // Permute Share MLP Weight
    if (model_config_.quant_config.is_fp8_blockwise) {
#ifdef ENABLE_FP8
      quant_weight_solver_->BindMoeFp8E4m3BlockWiseScaleOfWeight();
#else
      KLLM_THROW("Device not support Fp8");
#endif
    } else if (model_config_.quant_config.method == QUANT_FP8_E4M3 &&
               model_config_.quant_config.is_checkpoint_fp8_serialized) {
#ifdef ENABLE_FP8
      quant_weight_solver_->BindFp8E4m3ScaleOfMoeWeight();
#else
      KLLM_THROW("Device not support Fp8");
#endif
    } else if (!quant_weight_solver_->IsEnable()) {
      ConvertShareMLPWeight(false);
    }
  }
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
}

template class CommonMoeWeight<float>;
template class CommonMoeWeight<float16>;
template class CommonMoeWeight<bfloat16>;

}  // namespace ksana_llm
