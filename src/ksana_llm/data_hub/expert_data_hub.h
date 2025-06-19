/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/hidden_unit_buffer.h"

#include "ksana_llm/data_hub/expert_parallel_hidden_unit_buffer.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/distributed/packet_type.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

void InitializeExpertHiddenUnitBufferPool();
ExpertParallelHiddenUnitBufferPool* GetExpertHiddenUnitBufferPool();

Status ResetExpertReceiveWaiter();
Status ResetExpertWaiter();

Status ExpertWait();

Status ExpertNotify();

HiddenUnitDeviceBuffer* RecvExpertHiddenUnits(int rank);
HiddenUnitDeviceBuffer* AsyncRecvExpertHiddenUnits(int rank);

Status SendExpertHiddenUnits(HiddenUnitDeviceBuffer* hidden_unit_buffer, bool is_sync);

Status InitExpertHiddenUnits();

Status FreeExpertRecvHiddenUnits(HiddenUnitDeviceBuffer* hidden_unit_buffer);

void DestroyExpertHiddenUnitBufferPool();

void SetCurrentExpertRecvHiddenUnitBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer);
HiddenUnitDeviceBuffer* GetCurrentExpertRecvHiddenUnitBuffer();

void SetCurrentExpertSendHiddenUnitBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer);
HiddenUnitDeviceBuffer* GetCurrentExpertSendHiddenUnitBuffer();

void SetCurrentExpertRecvCommMetaHiddenUnitBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer);
HiddenUnitDeviceBuffer* GetCurrentExpertRecvCommMetaHiddenUnitBuffer();

void PrintExpertHiddenUnitBufferPoolInfo(std::string tag = "PrintExpertHiddenUnitBufferPoolInfo");

}  // namespace ksana_llm