/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/internlm2/internlm2_weight.h"

namespace ksana_llm {

template <typename T>
Internlm2Weight<T>::Internlm2Weight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, rank, context) {}

template <typename T>
Status Internlm2Weight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                               const std::vector<std::string>& weight_name_list,
                                               const std::vector<std::string>& custom_name_list) {
  CommonWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);

  SetDevice(rank_);

  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    const std::string& tensor_name = custom_name_list[idx];
    const std::string& weight_name = weight_name_list[idx];
    size_t tensor_para_offset = 0;
    bool transpose_first = false;

    int head_num = model_config_.head_num;
    int hidden_units = model_config_.hidden_units;
    int num_kv_heads = model_config_.num_key_value_heads;
    int head_dim = hidden_units / head_num;
    int num_key_value_groups = head_num / num_kv_heads;

    // InternLM2 use Interleaving, we need to de-Interleaving.
    if ((tensor_name.find("self_attn.W_rqkv_pack.weight") != std::string::npos) ||
        (tensor_name.find("self_attn.qkv_rproj.Plora_B.weight") != std::string::npos)) {
      auto [weight_ptr, weight_size] = weights_loader->GetTensor(weight_name);
      DataType weight_data_type = weights_loader->GetTensorDataType(weight_name);
      std::vector<size_t> weight_shape = weights_loader->GetTensorShape(weight_name);

      torch::ScalarType torch_dtype;
      if (weight_data_type_ == TYPE_FP32) {
        torch_dtype = torch::kFloat32;
      } else if (weight_data_type_ == TYPE_FP16) {
        torch_dtype = torch::kFloat16;
      } else if (weight_data_type_ == TYPE_BF16) {
        torch_dtype = torch::kBFloat16;
      }
      auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch_dtype);
      torch::Tensor in =
          torch::from_blob(weight_ptr, std::vector<int64_t>(weight_shape.begin(), weight_shape.end()), options);
      in = in.to(options);
      in = in.view({num_kv_heads, 2 + num_key_value_groups, head_dim, -1});
      auto splits = torch::split(in, {num_key_value_groups, 1, 1}, 1);
      auto query_states =
          splits[0].contiguous().view({num_kv_heads * num_key_value_groups * head_dim, weight_shape[1]});
      auto key_states = splits[1].contiguous().view({num_kv_heads * head_dim, weight_shape[1]});
      auto value_states = splits[2].contiguous().view({num_kv_heads * head_dim, weight_shape[1]});
      in = torch::cat({query_states, key_states, value_states}, 0);
      weight_ptr = in.data_ptr();

      CommonWeight<T>::PrepareLoadOpMeta(tensor_para_offset, weight_shape, transpose_first, tensor_name);
      if (tensor_name.find("qkv_rproj.Plora_B") != std::string::npos) {
        CommonWeight<T>::LoadRegularTensor(weight_ptr, tensor_name, weight_shape, weight_data_type, transpose_first,
                                           tensor_para_offset, weight_size);
      } else if (tensor_name.find("self_attn.W_rqkv_pack.weight") != std::string::npos) {
        size_t last_underscore = tensor_name.find_last_of('_');
        size_t second_last_underscore = tensor_name.find_last_of('_', last_underscore - 1);
        std::string qkv_name = tensor_name.substr(0, second_last_underscore - 1) + "query_key_value.weight";
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
      } else {
        KLLM_LOG_DEBUG << "state_dict[" << tensor_name << "] will not be used";
      }
    }
  }
  return Status();
}

template <typename T>
Status Internlm2Weight<T>::PermuteLoraWeight(Tensor& last_lora_a_tensor, Tensor& last_lora_b_tensor,
                                             const int num_layer, const std::string& last_lora_a_proj_name,
                                             const std::string& last_lora_b_proj_name) {
  SetDevice(rank_);
  for (size_t layer_idx = 0; layer_idx < (size_t)num_layer; ++layer_idx) {
    std::string lora_a_proj_name = "model.layers." + std::to_string(layer_idx) + last_lora_a_proj_name;
    CommonWeight<T>::CommonPermuteWeight(lora_a_proj_name, last_lora_a_tensor);
    std::string lora_b_proj_name = "model.layers." + std::to_string(layer_idx) + last_lora_b_proj_name;
    CommonWeight<T>::CommonPermuteWeight(lora_b_proj_name, last_lora_b_tensor);
  }
  return Status();
}

template <typename T>
void Internlm2Weight<T>::ProcessWeights() {
  CommonWeight<T>::ProcessWeights();

  if (model_config_.type == "internlmxcomposer2") {
    int num_layer = model_config_.num_layer;

    tensor_manager_->CreateTensorWithSameShape("model.layers.0.self_attn.qkv_proj.Plora_A.weight",
                                               "empty_qkv_lora_a_tensor");
    tensor_manager_->CreateTensorWithSameShape("model.layers.0.self_attn.qkv_rproj.Plora_B.weight",
                                               "empty_qkv_lora_b_tensor");
    Tensor& last_qkv_lora_a_tensor = weights_map_["empty_qkv_lora_a_tensor"];
    Tensor& last_qkv_lora_b_tensor = weights_map_["empty_qkv_lora_b_tensor"];
    PermuteLoraWeight(last_qkv_lora_a_tensor, last_qkv_lora_b_tensor, num_layer, ".self_attn.qkv_proj.Plora_A.weight",
                      ".self_attn.qkv_rproj.Plora_B.weight");

    tensor_manager_->CreateTensorWithSameShape("model.layers.0.self_attn.o_proj.Plora_A.weight",
                                               "empty_o_lora_a_tensor");
    tensor_manager_->CreateTensorWithSameShape("model.layers.0.self_attn.o_proj.Plora_B.weight",
                                               "empty_o_lora_b_tensor");
    Tensor& last_o_lora_a_tensor = weights_map_["empty_o_lora_a_tensor"];
    Tensor& last_o_lora_b_tensor = weights_map_["empty_o_lora_b_tensor"];
    PermuteLoraWeight(last_o_lora_a_tensor, last_o_lora_b_tensor, num_layer, ".self_attn.o_proj.Plora_A.weight",
                      ".self_attn.o_proj.Plora_B.weight");

    tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.down_proj.Plora_A.weight",
                                               "empty_down_lora_a_tensor");
    tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.down_proj.Plora_B.weight",
                                               "empty_down_lora_b_tensor");
    Tensor& last_down_lora_a_tensor = weights_map_["empty_down_lora_a_tensor"];
    Tensor& last_down_lora_b_tensor = weights_map_["empty_down_lora_b_tensor"];
    PermuteLoraWeight(last_down_lora_a_tensor, last_down_lora_b_tensor, num_layer, ".mlp.down_proj.Plora_A.weight",
                      ".mlp.down_proj.Plora_B.weight");

    tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.up_proj.Plora_A.weight", "empty_up_lora_a_tensor");
    tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.up_proj.Plora_B.weight", "empty_up_lora_b_tensor");
    Tensor& last_up_lora_a_tensor = weights_map_["empty_up_lora_a_tensor"];
    Tensor& last_up_lora_b_tensor = weights_map_["empty_up_lora_b_tensor"];
    PermuteLoraWeight(last_up_lora_a_tensor, last_up_lora_b_tensor, num_layer, ".mlp.up_proj.Plora_A.weight",
                      ".mlp.up_proj.Plora_B.weight");

    tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.gate_proj.Plora_A.weight",
                                               "empty_gate_lora_a_tensor");
    tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.gate_proj.Plora_B.weight",
                                               "empty_gate_lora_b_tensor");
    Tensor& last_gate_lora_a_tensor = weights_map_["empty_gate_lora_a_tensor"];
    Tensor& last_gate_lora_b_tensor = weights_map_["empty_gate_lora_b_tensor"];
    PermuteLoraWeight(last_gate_lora_a_tensor, last_gate_lora_b_tensor, num_layer, ".mlp.gate_proj.Plora_A.weight",
                      ".mlp.gate_proj.Plora_B.weight");

    weights_map_.erase("empty_qkv_lora_a_tensor");
    weights_map_.erase("empty_qkv_lora_b_tensor");
    weights_map_.erase("empty_o_lora_a_tensor");
    weights_map_.erase("empty_o_lora_b_tensor");
    weights_map_.erase("empty_down_lora_a_tensor");
    weights_map_.erase("empty_down_lora_b_tensor");
    weights_map_.erase("empty_up_lora_a_tensor");
    weights_map_.erase("empty_up_lora_b_tensor");
    weights_map_.erase("empty_gate_lora_a_tensor");
    weights_map_.erase("empty_gate_lora_b_tensor");
  }
}

template class Internlm2Weight<float>;
template class Internlm2Weight<float16>;
template class Internlm2Weight<bfloat16>;

}  // namespace ksana_llm
