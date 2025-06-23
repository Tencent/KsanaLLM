/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/llama/llama_weight.h"

namespace ksana_llm {

template <typename T>
LlamaWeight<T>::LlamaWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context) {
  common_weight_ = std::make_shared<CommonWeight<T>>(model_config, rank, context);
  model_config_ = model_config;
}

template <typename T>
Tensor LlamaWeight<T>::GetModelWeights(const std::string& weight_name) {
  return common_weight_->GetModelWeights(weight_name);
}
template <typename T>
void LlamaWeight<T>::PermuteQKWeight(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                     const std::vector<std::string>& weight_name_list,
                                     const std::vector<std::string>& custom_name_list) {
  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    const std::string& tensor_name = custom_name_list[idx];
    const std::string& weight_name = weight_name_list[idx];
    if (tensor_name.find("k_proj.weight") != std::string_view::npos ||
        tensor_name.find("k_proj.bias") != std::string_view::npos ||
        tensor_name.find("q_proj.weight") != std::string_view::npos ||
        tensor_name.find("q_proj.bias") != std::string_view::npos) {
      std::vector<size_t> weight_shape = weights_loader->GetTensorShape(weight_name);
      // get weight's data ptr
      auto [weight_ptr, weight_size] = weights_loader->GetTensor(weight_name);
      if (weight_ptr == nullptr) {
        KLLM_LOG_DEBUG << fmt::format("The {}'s weight_ptr is null", weight_name);
        continue;
      }
      DataType weight_data_type = weights_loader->GetTensorDataType(weight_name);

      torch::Tensor weight_cpu_tensor;
      int64_t head_num = 0;
      if (tensor_name.find("k_proj") != std::string_view::npos) {
        head_num = model_config_.num_key_value_heads;
      } else if (tensor_name.find("q_proj") != std::string_view::npos) {
        head_num = model_config_.head_num;
      }

      if (head_num > 0) {
        torch::TensorOptions options = torch::TensorOptions().device(torch::kCPU);
        switch (weight_data_type) {
          case TYPE_FP16:
            options = options.dtype(torch::kFloat16);
            break;
          case TYPE_FP32:
            options = options.dtype(torch::kFloat32);
            break;
          case TYPE_BF16:
            options = options.dtype(torch::kBFloat16);
            break;
          default:  // TYPE_INVALID
            throw std::runtime_error("Invalid data type");
        }
        std::vector<int64_t> shape = {head_num, weight_shape[0] / head_num / 2, 2, weight_shape[1]};
        int64_t num_elements = weight_size / GetTypeSize(weight_data_type);

        torch::Tensor in = torch::from_blob(weight_ptr, {num_elements}, options);
        weight_cpu_tensor = in.reshape(shape).swapaxes(1, 2).contiguous();
        weights_loader->SetTensor(weight_name, std::move(weight_cpu_tensor));
      }
    }
  }
}

template <typename T>
Status LlamaWeight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                           const std::vector<std::string>& weight_name_list,
                                           const std::vector<std::string>& custom_name_list) {
  if (model_config_.model_file_format == GGUF) {
    PermuteQKWeight(weights_loader, weight_name_list, custom_name_list);
  }
  if (!common_weight_->LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list).OK()) {
    KLLM_THROW(fmt::format("Load weight file {} error.", weights_loader->GetTensorFileName()));
  }
  return Status();
}

template <typename T>
void LlamaWeight<T>::ProcessWeights() {
  common_weight_->ProcessWeights();
}

template <typename T>
void LlamaWeight<T>::SetEmbeddingsConfig() {
  common_weight_->SetEmbeddingsConfig();
}

template class LlamaWeight<float>;
template class LlamaWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class LlamaWeight<bfloat16>;
#endif

}  // namespace ksana_llm
