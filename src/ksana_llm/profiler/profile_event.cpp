/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/utils/logger.h"

#ifdef ENABLE_CUDA
#include "nvToolsExt.h"
#endif

namespace ksana_llm {

size_t g_profile_layer_forwarding_round = 1;

static bool ReadEnableProfileEventFlag() {
  if (std::getenv("ENABLE_PROFILE_EVENT") != nullptr) {
    KLLM_LOG_INFO << "PROFILE_EVENT is enabled";
    return true;
  } else {
    return false;
  }
}

static bool g_enable_profile_event = ReadEnableProfileEventFlag();


void ProfileEvent::PushEvent(const std::string& profile_event_name, int rank) {
#ifdef ENABLE_CUDA
  if (g_enable_profile_event) {
    nvtxStringHandle_t nvtx_string_handle = nvtxDomainRegisterStringA(NULL, profile_event_name.c_str());
    nvtxEventAttributes_t event_attr = {0};
    event_attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    event_attr.message.registered = nvtx_string_handle;
    event_attr.payloadType = NVTX_PAYLOAD_TYPE_INT32;
    event_attr.payload.iValue = rank;
    nvtxRangePushEx(&event_attr);
  }
#endif
}

void ProfileEvent::PopEvent() {
#ifdef ENABLE_CUDA
  if (g_enable_profile_event) {
    nvtxRangePop();
  }
#endif
}

}  // namespace ksana_llm
