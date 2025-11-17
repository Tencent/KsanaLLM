/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/expert_parallel_deepep_wrapper.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

Status InitializeExpertParallelDeepepWrapper(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                             const std::shared_ptr<Context>& context);

const std::shared_ptr<ExpertParallelDeepepWrapper>& GetExpertParallelDeepepWrapper();
void SetExpertParallelDeepepWrapper(const std::shared_ptr<ExpertParallelDeepepWrapper>& deepep_wrapper);

}  // namespace ksana_llm