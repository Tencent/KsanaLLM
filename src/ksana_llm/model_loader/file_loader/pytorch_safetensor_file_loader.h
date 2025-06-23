/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/base_file_tensor_loader.h"
#include "nlohmann/json.hpp"

#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

#include "ksana_llm/model_loader/file_loader/base_file_loader.h"

using json = nlohmann::json;

namespace ksana_llm {

// Used to load pytorch safetensor files.
class PytorchSafetensorFileLoader : public BaseFileLoader {
 public:
  explicit PytorchSafetensorFileLoader(const std::string& filename);
  virtual ~PytorchSafetensorFileLoader();

  // Load weight names from file, but not load it.
  virtual Status LoadWeightNames(std::vector<std::string>& weight_names) override;

  // Load weights in weight_names.
  virtual Status LoadModelWeights(const std::vector<std::string>& weight_names,
                                  std::unordered_map<std::string, Tensor>& result) override;

 private:
  Status LoadSafetensorTensorDict();

 private:
  const std::string filename_;
  json tensor_dict_;

  void* mmap_ptr_ = nullptr;
  size_t file_size_;
};

}  // namespace ksana_llm
