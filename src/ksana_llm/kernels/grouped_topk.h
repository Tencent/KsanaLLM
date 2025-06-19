/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

template <typename T>
void InvokeGroupedTopk(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows, int num_experts,
                       int topk, bool renormalize, int num_expert_group, int topk_group, std::string scoring_func,
                       void* e_bias, float routed_scaling_factor, int rank, cudaStream_t stream);

}  // namespace ksana_llm