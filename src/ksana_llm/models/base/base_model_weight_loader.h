/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <unordered_map>

#include "ksana_llm/models/base/base_model_config.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// The base class of all custom weight loader, any implemented weight loader could override it.
class BaseModelWeightLoader {
 public:
  BaseModelWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                        std::shared_ptr<Context> context);
  virtual ~BaseModelWeightLoader();

  // Do some filter on model file list.
  virtual Status FilterModelFiles(std::vector<std::string>& model_files);

  // Do some filter on model weight names.
  virtual Status FilterWeightNames(std::vector<std::string>& weight_names);

  // Process weights, such as rename, split, merge, type convert, quantization, etc.
  // The unprocessed weights will be put to left_host_weights, and merge to host_model_weights with next file.
  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                     std::unordered_map<std::string, Tensor>& left_host_weights);

  // Invoked only once before ProcessModelWeights.
  virtual Status PreProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights);

  // Invoked only once after ProcessModelWeights.
  virtual Status PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map, int dev_rank);

 protected:
  // Permute tensor by specific permutation.
  Status PermuteDeviceTensor(const Tensor& input_tensor, const std::vector<size_t>& permutation, int dev_rank,
                             Tensor& output_tensor);

  // cast device tensor type.
  Status CastDeviceTensorType(Tensor& input_tensor, DataType new_dtype, int dev_rank);

  // Copy host tensor to device.
  Status CopyHostTensorToDevice(const Tensor host_tensor, int dev_rank, Tensor& dev_tensor);

  // Check whether the weight_name is matched.
  bool CheckWeightNameMatched(const std::string& weight_name, const std::vector<std::string>& name_list,
                              bool full_match = true);

  // Check whether all the weight names is exists.
  bool CheckAllWeightsExist(const std::unordered_map<std::string, Tensor>& host_model_weights,
                            const std::vector<std::string>& name_list);

 protected:
  std::shared_ptr<BaseModelConfig> model_config_ = nullptr;
  std::shared_ptr<Environment> env_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;
};

}  // namespace ksana_llm
