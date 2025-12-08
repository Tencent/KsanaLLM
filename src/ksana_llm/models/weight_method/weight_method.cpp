/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/weight_method/weight_method.h"

namespace ksana_llm {

WeightMethod::WeightMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp)
    : common_method_(std::make_shared<CommonMethod>(common_weight_loader, tp)),
      w4a8_awq_method_(std::make_shared<W4A8AWQMethod>(common_weight_loader, tp)) {}

}  // namespace ksana_llm