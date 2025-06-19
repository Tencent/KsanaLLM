/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/distributed/data_channel_factory.h"
#include "ksana_llm/distributed/data_channel.h"
#include "ksana_llm/distributed/expert_parallel_data_channel.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/distributed/nvidia/expert_parallel_nccl_data_channel.h"
#  include "ksana_llm/distributed/nvidia/nccl_data_channel.h"
#endif
#include "ksana_llm/utils/status.h"

namespace ksana_llm {
Status DataChannelFactory::CreateDataChannel(PacketCreationFunc packet_creation_fn,
                                             HiddenUnitBufferPool* hidden_unit_buffer_pool,
                                             std::shared_ptr<Environment> env, std::shared_ptr<Context> context,
                                             std::shared_ptr<DataChannelInterface>& data_channel) {
  const char* use_tcp_data_channel = std::getenv("USE_TCP_DATA_CHANNEL");
  if (use_tcp_data_channel && strcmp(use_tcp_data_channel, "1") == 0) {
    data_channel = std::make_shared<DataChannel>(packet_creation_fn, hidden_unit_buffer_pool, env);
    return Status();
  }

#ifdef ENABLE_CUDA
  data_channel = std::make_shared<NcclDataChannel>(hidden_unit_buffer_pool, env, context);
  return Status();
#endif

#ifdef ENABLE_ACL
  data_channel = std::make_shared<DataChannel>(packet_creation_fn, hidden_unit_buffer_pool, env);
  return Status();
#endif
}

template <typename DataChannelType>
Status DataChannelFactory::CreateExpertDataChannel(PacketCreationFunc packet_creation_fn,
                                                   HiddenUnitBufferPool* hidden_unit_buffer_pool,
                                                   ExpertParallelHiddenUnitBufferPool* expert_hidden_unit_buffer_pool,
                                                   std::shared_ptr<Environment> env, std::shared_ptr<Context> context,
                                                   std::shared_ptr<DataChannelInterface>& data_channel) {
  const char* use_tcp_data_channel = std::getenv("USE_TCP_DATA_CHANNEL");
  if (use_tcp_data_channel && strcmp(use_tcp_data_channel, "1") == 0) {
    KLLM_LOG_INFO << "CreateExpertDataChannel, USE_TCP_DATA_CHANNEL";
    data_channel = std::make_shared<ExpertParallelDataChannel>(packet_creation_fn, hidden_unit_buffer_pool,
                                                               expert_hidden_unit_buffer_pool, env);
    return Status();
  }

#ifdef ENABLE_CUDA
  data_channel = std::make_shared<DataChannelType>(expert_hidden_unit_buffer_pool, env, context);
  return Status();
#endif

#ifdef ENABLE_ACL
  data_channel = std::make_shared<ExpertParallelDataChannel>(packet_creation_fn, expert_hidden_unit_buffer_pool, env);
  return Status();
#endif

  return Status();
}

#ifdef ENABLE_CUDA

template Status DataChannelFactory::CreateExpertDataChannel<ExpertParallelNcclDataChannel>(
    PacketCreationFunc packet_creation_fn, HiddenUnitBufferPool* hidden_unit_buffer_pool,
    ExpertParallelHiddenUnitBufferPool* expert_hidden_unit_buffer_pool, std::shared_ptr<Environment> env,
    std::shared_ptr<Context> context, std::shared_ptr<DataChannelInterface>& data_channel);
#endif

}  // namespace ksana_llm
