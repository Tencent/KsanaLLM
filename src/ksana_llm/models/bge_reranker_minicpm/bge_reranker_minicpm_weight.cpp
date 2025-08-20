/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/bge_reranker_minicpm/bge_reranker_minicpm_weight.h"
#include <numeric>
#include <regex>

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
BgeRerankerMinicpmWeight<T>::BgeRerankerMinicpmWeight(const ModelConfig& model_config,
                                                      const RuntimeConfig& runtime_config, int rank,
                                                      std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, runtime_config, rank, context) {}

template <typename T>
Status BgeRerankerMinicpmWeight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                                        const std::vector<std::string>& weight_name_list,
                                                        const std::vector<std::string>& custom_name_list) {
  Status status = CommonWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  if (!status.OK()) return status;

  SetDevice(this->rank_);
  if (this->rank_ != 0) {
    return Status();
  }

  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    const std::string& tensor_name = custom_name_list[idx];
    if (tensor_name.find(".linear_head.weight") == std::string::npos) {
      continue;
    }

    const std::string& weight_name = weight_name_list[idx];
    std::vector<size_t> weight_shape = weights_loader->GetTensorShape(weight_name);
    DataType weight_data_type = weights_loader->GetTensorDataType(weight_name);
    void* weight_ptr;
    size_t weight_size;
    std::tie(weight_ptr, weight_size) = weights_loader->GetTensor(weight_name);

    torch::Tensor weight_cpu_tensor;
    bool should_cast = (tensor_name.find(".weight_scale") == std::string::npos &&
                        tensor_name.find(".input_scale") == std::string::npos && weight_data_type == TYPE_FP32);

    if (should_cast) {
      auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
      torch::Tensor in = torch::from_blob(weight_ptr, {(int64_t)(weight_size / sizeof(float))}, options);
      weight_size /= sizeof(float) / GetTypeSize(this->weight_data_type_);

      if (this->weight_data_type_ == TYPE_FP16) {
        weight_cpu_tensor = in.to(torch::kFloat16);
        weight_ptr = weight_cpu_tensor.data_ptr();
        weight_data_type = this->weight_data_type_;
      } else if (this->weight_data_type_ == TYPE_BF16) {
        weight_cpu_tensor = in.to(torch::kBFloat16);
        weight_ptr = weight_cpu_tensor.data_ptr();
        weight_data_type = this->weight_data_type_;
      }
    }

    CommonWeight<T>::LoadRegularTensor(weight_ptr, tensor_name, weight_shape, weight_data_type, false, 0, weight_size);
  }

  return Status();
}

template <typename T>
Status BgeRerankerMinicpmWeight<T>::PermuteLinearHeadWeight(Tensor& last_linear_head_tensor, const int num_layer) {
  SetDevice(this->rank_);
  if (this->rank_ != 0) {
    return Status();
  }

  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string linear_head_name = "lm_head." + std::to_string(layer_idx) + ".linear_head.weight";
    std::string temp_tensor_name = "temp_linear_head_tensor_" + std::to_string(layer_idx);

    this->tensor_manager_->CreateTensorWithSameShape(linear_head_name, temp_tensor_name);
    Tensor& temp_tensor = this->weights_map_[temp_tensor_name];

    CommonWeight<T>::CommonPermuteWeight(linear_head_name, temp_tensor);
    this->weights_map_.erase(temp_tensor_name);
  }

  return Status();
}

template <typename T>
Status BgeRerankerMinicpmWeight<T>::ConvertCommonTensor(int hidden_units, int inter_size, int vocab_size) {
  SetDevice(rank_);

  CommonWeight<T>::ConvertQkvTensor();

  if (!model_config_.is_moe || !CommonWeight<T>::required_layer_idx_.dense.empty()) {
    CommonWeight<T>::ConvertMLPWeight(false);
  }

  CommonWeight<T>::ConvertOprojTensor();
  return Status();
}

template <typename T>
void BgeRerankerMinicpmWeight<T>::ProcessWeights() {
  const int hidden_units = model_config_.hidden_units;
  const int inter_size = model_config_.inter_size;
  const int vocab_size = model_config_.vocab_size;

  // Convert FP16/BF16 weights
  if (model_config_.weight_data_type == TYPE_FP16 || model_config_.weight_data_type == TYPE_BF16) {
    SetDevice(rank_);
    for (auto& data_type_iter : weights_data_type_map_) {
      if (data_type_iter.second == TYPE_FP16 || data_type_iter.second == TYPE_BF16) {
        Tensor& tensor = weights_map_[data_type_iter.first];
        tensor.dtype = data_type_iter.second;
        CastInplace(tensor, model_config_.weight_data_type, context_->GetMemoryManageStreams()[rank_]);
        tensor.dtype = model_config_.weight_data_type;
      }
    }
  }

  ConvertCommonTensor(hidden_units, inter_size, vocab_size);
  CommonWeight<T>::ConvertNextnProjTensor();

  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);

  // BGE reranker specific processing
  if (rank_ == 0) {
    int num_layers = model_config_.num_layer - model_config_.start_layer + 1;
    Tensor dummy_tensor;
    PermuteLinearHeadWeight(dummy_tensor, num_layers);
  }
}

template class BgeRerankerMinicpmWeight<float>;
template class BgeRerankerMinicpmWeight<float16>;
template class BgeRerankerMinicpmWeight<bfloat16>;

}  // namespace ksana_llm