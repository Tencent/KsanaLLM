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

class CommonModelWeightLoader {
 public:
  CommonModelWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                          std::shared_ptr<Context> context);

  ~CommonModelWeightLoader() = default;

  Status LoadMhaWeights(const std::string& host_weight_name, const Tensor& host_weight_tensor,
                        std::unordered_map<std::string, Tensor>& device_model_weights, int dev_rank, size_t num_heads,
                        size_t num_kv_heads, size_t size_per_head);

  // Reuse temporary created tensor while processing weights
  // these tensors should not be inserted into device_model_weights
  // Careful with this function because it may cause memory issue
  Tensor& GetTempTensor(const std::vector<size_t>& shape, DataType data_type, int dev_rank);

  // Permutation with buffer
  Status PermuteWeight(Tensor& input_tensor, const std::vector<size_t>& permutation, int dev_rank);

  // Split weight along axis = 0, then with param `skip_transpose` to decide whether skip transpose
  Status SplitOptTrans(const Tensor& host_weight_tensor, Tensor& output_tensor, int dev_rank, size_t para_size,
                       bool transpose);

  // Transpose and split weight along axis = 0, then with param `transpose` to decide whether to transpose back
  Status TransSplitOptTrans(const Tensor& host_weight_tensor, Tensor& output_tensor, int dev_rank, size_t para_size,
                            bool transpose);

 private:
  std::shared_ptr<BaseModelConfig> model_config_;
  std::shared_ptr<Environment> env_;
  std::shared_ptr<Context> context_;

  std::vector<std::unordered_map<size_t, Tensor>> permute_buffers_;
  std::vector<std::unordered_map<size_t, Tensor>> tensor_buffers_;
};

}  // namespace ksana_llm
