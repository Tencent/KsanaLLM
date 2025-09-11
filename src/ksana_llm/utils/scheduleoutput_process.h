/* Copyright 2025 Tencent Inc.
==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/sampling_request.h"

namespace ksana_llm {

void DeepCopySamplingRequest(SamplingRequest& original);

void DeepCopyForwardRequest(ForwardRequest& original);

void MergeScheduleOutputGroup(std::shared_ptr<ScheduleOutputGroup>& schedule_output_group,
                              ScheduleOutput& merged_schedule_output);

}  // namespace ksana_llm
