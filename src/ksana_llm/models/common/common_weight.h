/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/quant/quant_weight.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/tensor_manager.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T>
class CommonWeight : public BaseWeight {
 public:
  CommonWeight() {}
  ~CommonWeight() override;
  explicit CommonWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Tensor GetModelWeights(const std::string& weight_name) override;

  void ProcessWeights() override;

  Status LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                             std::vector<std::string>& weight_name_list,
                             std::vector<std::string>& custom_name_list) override;

  void SetEmbeddingsConfig() override;

  void PrintDebugMessage() override;

 protected:
  Status ConvertCommonTensor(int hidden_units, int inter_size, int vocab_size);

  Status ReshapeQkvTensor();
  Status ConvertLmheadTensor();
  Status ConvertNextnProjTensor();

  Status ConvertOprojTensor();
  Status ConvertQkvTensor();

  Status GetModelInfo(const ModelConfig& model_config);

  std::string ConcatLayerName(std::string layer_flag, int& layer_index, bool is_bias = false);

  Status LoadMlpUpGateTensor(void* weight_ptr, std::string tensor_name, std::vector<size_t>& weight_shape,
                             DataType& weight_data_type, bool transpose_first, size_t tensor_para_offset);

  Status LoadRegularTensor(void* weight_ptr, std::string tensor_name, std::vector<size_t>& weight_shape,
                           DataType& weight_data_type, bool transpose_first, size_t tensor_para_offset,
                           size_t& weight_size);

  Status PermuteSingleTensorOfQKVWeight(void* src, void* dst, Tensor& q_in_tensor, Tensor& q_out_tensor,
                                        std::vector<size_t>& data_shape, std::vector<size_t>& qkv_dst_shape);
  Status PermuteQKVWeight(Tensor& last_qkv_tensor, Tensor& q_in_tensor, Tensor& q_out_tensor);

  Status CommonPermuteWeight(const std::string& origin_tensor_name, Tensor& swap_tensor);

  Status ConvertMLPWeight(bool is_weight_scale);

  Status ConvertMLPWeight(const std::string& weight_name_format, const std::unordered_set<int16_t>& layers,
                          bool is_weight_scale);

  Status PermuteMLPWeight(Tensor& last_down_up_tensor, Tensor& last_gate_tensor, const std::string& weight_name_format,
                          const std::unordered_set<int16_t>& layers, const bool is_weight_scale);

  Status PermuteOutputProjectWeight(Tensor& last_o_proj_tensor);

  Status PrepareLoadOpMeta(size_t& tensor_para_offset, std::vector<size_t>& weight_shape, bool& transpose_first,
                           const std::string& tensor_name);

  bool ShouldUseFusedGateUpWeights();

  void ChunkGateWeight();

  bool IsLoaded();
  bool weights_had_loaded_ = false;

  using BaseWeight::weights_data_type_map_;
  using BaseWeight::weights_map_;

  std::string model_path_ = "";
  size_t tensor_para_size_ = 1;
  size_t expert_world_size_ = 1;
  size_t expert_para_size_ = 1;
  size_t global_expert_para_size_ = 1;
  bool enable_full_shared_expert_ = false;

  std::string model_name_ = "";
  DataType weight_data_type_ = TYPE_FP16;
  DataType moe_weight_data_type_ = TYPE_FP16;

  std::shared_ptr<QuantWeight<T>> quant_weight_solver_;
};

}  // namespace ksana_llm
