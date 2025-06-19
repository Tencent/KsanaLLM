/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// The base class of all file loader.
class BaseFileLoader {
 public:
  virtual ~BaseFileLoader() {}

  // Load weight names from file, but not load it.
  virtual Status LoadWeightNames(std::vector<std::string>& weight_names) = 0;

  // Load weights in weight_names.
  virtual Status LoadModelWeights(const std::vector<std::string>& weight_names,
                                  std::unordered_map<std::string, Tensor>& result) = 0;
};

}  // namespace ksana_llm
