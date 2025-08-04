/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <vector>
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/hidden_unit_buffer.h"

#include "ksana_llm/models/base/forwarding_context.h"
#include "ksana_llm/utils/barrier.h"
#include "ksana_llm/utils/expert_parallel_utils.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

template <typename T>
class ExpertParallelDataTransfer {
 public:
  ExpertParallelDataTransfer();
  ~ExpertParallelDataTransfer() {}
  // Get a reference for hidden buffer.
  std::vector<Tensor>& GetHiddenUnitBufferRef(ForwardingContext& forwarding_context);
  std::vector<Tensor>& GetExpertRecvHiddenUnitBufferRef(HiddenUnitDeviceBuffer* hidden_unit,
                                                        ForwardingContext& forwarding_context);
  // Broadcast hidden state to other expert parallel nodes.
  void SendHiddenUnitBufferForEP(const std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context,
                                 bool is_sync);
  std::vector<Tensor>& RecvHiddenUnitBufferForEP(ForwardingContext& forwarding_context);
  std::vector<Tensor>& AsyncRecvHiddenUnitBufferForEP(ForwardingContext& forwarding_context);
  // Combine hidden state from other ep-worker after moe.
  void CombineHiddenUnitBufferForEP(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context);
  // Set hidden state,
  HiddenUnitDeviceBuffer* SetHiddenUnitBufferForEP(const std::vector<Tensor>& residual_buffer,
                                                   ForwardingContext& forwarding_context);
  // Set hidden state,
  HiddenUnitDeviceBuffer* SetCommMetaHiddenUnitBufferForEP(expert_parallel_comm_meta& meta, DataType dtype,
                                                           ForwardingContext& forwarding_context);
  void FreeHiddenUnitDeviceBuffer(ForwardingContext& forwarding_context);

 private:
  std::vector<Tensor> distributed_device_buffer_;
  std::vector<Tensor> distributed_device_buffer_prefill_;
  std::vector<Tensor> local_residual_buffer_;
  // Only add and free by the rank 0 device.
  std::vector<HiddenUnitDeviceBuffer*> hidden_device_buffer_;
};

}  // namespace ksana_llm
