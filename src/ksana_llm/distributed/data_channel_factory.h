/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/distributed/data_channel_interface.h"
#include "ksana_llm/distributed/packet_util.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/distributed/nvidia/nccl_data_channel.h"
#endif

namespace ksana_llm {

// Used to create different kind of data channels.
class DataChannelFactory {
 public:
  static Status CreateDataChannel(PacketCreationFunc packet_creation_fn, HiddenUnitBufferPool* hidden_unit_buffer_pool,
                                  std::shared_ptr<Environment> env, std::shared_ptr<Context> context,
                                  std::shared_ptr<DataChannelInterface>& data_channel);
};

}  // namespace ksana_llm
