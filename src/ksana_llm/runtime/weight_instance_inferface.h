/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"

namespace ksana_llm {

class WeightInstanceInterface {
 public:
  WeightInstanceInterface() {}
  virtual ~WeightInstanceInterface() = default;

  virtual void Load() = 0;
  virtual std::shared_ptr<BaseWeight> GetWeight(int rank) = 0;
};

}  // namespace ksana_llm
