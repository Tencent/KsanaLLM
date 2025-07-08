/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/quant/cutlass_utils.h"
#include "ksana_llm/models/quant/machete_utils.h"
#include "ksana_llm/models/quant/marlin_utils.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/tensor_manager.h"
#include "ksana_llm/utils/utils.h"

#include "ksana_llm/kernels/permute.h"

namespace ksana_llm {

// Load quantized weights, used together with CommonWeight
template <typename T>
class QuantWeight {
 public:
  QuantWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context,
              std::unordered_map<std::string, Tensor>& weights_map,
              std::unordered_map<std::string, DataType>& weights_data_type_map);
  ~QuantWeight();

  // Enable quantized loading if the model is a quantized model
  bool IsEnable();

  // Determine if weights need to be filtered, e.g. in gptq model, "*.g_idx" is unnecessary and should be filtered out.
  bool FilterOutQuantWeight(const std::string& tensor_name);

  // Group weight conversion for weight transformation, partitioning, transposition, etc.
  Status PackAndBindGroupTensor(int layer_idx, const std::string& needed_slove_weight_name);
  Status AutoPackAndBindGroupTensor(std::vector<std::string> needed_slove_weights_name);
  Status ConvertGroupTensor();

  void LoadMoeIntQuantWeight(const std::string& tensor_name, std::vector<size_t>& weight_shape,
                             DataType& weight_data_type, void* weight_ptr);
  // Load the weight if it is a quantized weight.
  // Currently, for q/k/v, they are loaded separately first,
  // then merged into qkv in ConvertGroupTensor, and finally q/k/v are deleted.
  // This is because weight layout conversion requires individual weight processing.
  bool LoadQuantWeight(const std::string& tensor_name, std::vector<size_t>& weight_shape, DataType& weight_data_type,
                       void* weight_ptr);

  Tensor CommonDequantTensor(const std::string& weight_name, bool remove_weight = false);

#ifdef ENABLE_FP8
  // Copy scale from weights_loader_ to weights_map_
  bool LoadFp8E4m3Scale(const std::string& tensor_name, std::vector<size_t>& weight_shape, DataType& weight_data_type,
                        void* weight_ptr);
  bool LoadMlaFp8E4m3BlockWiseScale(const std::string& tensor_name, std::vector<size_t>& weight_shape,
                                    DataType& weight_data_type, void* weight_ptr);
  bool LoadMoeFp8E4m3BlockWiseScale(const std::string& tensor_name, std::vector<size_t>& weight_shape,
                                    DataType& weight_data_type, void* weight_ptr);

#  ifdef ENABLE_FP8_TORCH
  Status ProcessMlaFp8E4m3BlockWiseScaleOfWeight();
#  endif
  // Bind scale to weight
  Status BindFp8E4m3Scale(const int num_heads, const int num_kv_heads);
  Status BindFp8E4m3ScaleOfProjWeight(const std::string& name);
  Status BindFp8E4m3ScaleOfQkvWeight(const std::string& name, const int num_heads, const int num_kv_heads);
  // Bind fp8-blockwise scale to weight
  Status BindMlaFp8E4m3BlockWiseScaleOfWeight();
  Status BindMoeFp8E4m3BlockWiseScaleOfWeight();

  Status BindFp8E4m3ScaleOfMoeWeight();

  Status GetMaxScaleOfQkv(float* q_scale, float* k_scale, float* v_scale, float* qkv_scale);

  Status ConvertFp8E4m3Tensor(const std::string& weight_name, DataType quant_type);

  Status ConvertFp8E4m3();
  Tensor DequantMlaFp8E4m3BlockWiseTensor(const std::string& weight_name, bool remove_weight = false);
  Tensor QuantMlaFp8E4m3BlockWiseTensor(const std::string& weight_name, bool remove_weight = false);
#endif

 private:
#ifdef ENABLE_CUDA

  torch::Tensor TrySmartAutoUnpack(const std::string& tensor_name, torch::Tensor& tensor);

  // tools
  torch::Tensor TpSplitTensor(torch::Tensor& tensor, int split_dim, int split_pos, int single_size);

  Status AddWeightFromTorchTensor(const std::string& name, torch::Tensor& tensor);

  Status AddWeightFromTorchTensor(const std::string& name, torch::Tensor& tensor, DataType& weight_data_type);

  torch::Tensor GetTorchTensorFromWeightPtr(std::vector<size_t> weight_shape, DataType weight_data_type,
                                            void* weight_ptr, bool to_gpu);

  torch::Tensor GetTorchTensorFromWeight(const std::string& name);
#endif

  // Check if the model is a quantized model
  bool CheckQuantModel();

  bool IsClaLayer(const int layer_idx);

  void GetExpertsScaleIdx(const std::string& expert_scale_name, int& layer_idx, int& expert_idx);

  std::shared_ptr<CutlassUtils> cutlass_helper_{nullptr};
  std::shared_ptr<MarlinUtils> marlin_helper_{nullptr};
  std::shared_ptr<MacheteUtils> machete_helper_{nullptr};

  // Weigth list for storing model weights, it needs to come from CommonWeight
  std::unordered_map<std::string, Tensor>& weights_map_;
  std::unordered_map<std::string, DataType>& weights_data_type_map_;

  // TensorManager for adding weights, it needs to come from CommonWeight
  std::shared_ptr<TensorManager> tensor_manager_;

  size_t tensor_para_size_ = 1;
  size_t expert_world_size_ = 1;
  size_t expert_para_size_ = 1;
  size_t global_expert_para_size_ = 1;
  size_t num_experts_per_rank_ = 1;
  bool enable_full_shared_expert_ = false;

  struct {
    std::unordered_set<int16_t> all, moe, dense;
  } required_layer_idx_;

  // weight is quantized in checkpoint
  bool enable_ = false;

  int rank_ = 0;
  std::shared_ptr<Context> context_{nullptr};
  ModelConfig model_config_;
  PipelineConfig pipeline_config_;

  DataType weight_data_type_ = TYPE_FP16;

  // TODO(zezhao) 与 common_moe_weight.h 复用一份
  std::vector<int> expert_map_;
};

}  // namespace ksana_llm
