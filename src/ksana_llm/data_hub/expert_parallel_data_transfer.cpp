/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/data_hub/expert_parallel_data_transfer.h"

#include <thread>
#include <vector>
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/expert_data_hub.h"

#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

ExpertParallelDataTransfer::ExpertParallelDataTransfer() {}

void ExpertParallelDataTransfer::SendHiddenUnitBufferForEP(const std::vector<Tensor>& residual_buffer,
                                                           ForwardingContext& forwarding_context, bool is_sync) {
  if (forwarding_context.GetContext()->IsExpertParallelStandalone()) return;

  // Send comm meta.
  expert_parallel_comm_meta meta;
  assert(residual_buffer[0].shape.size() == 2);
  meta.shape_0 = residual_buffer[0].shape[0];
  meta.shape_1 = residual_buffer[0].shape[1];

  // Every rank copy data to hidde_unit_buffer[tp_size], not need to send
  // h[tp_size] indeed. Split tokens into groups binded to specific experts set.

  GetExpertHiddenUnitBufferPool()->ExpertBarrier();

  HiddenUnitDeviceBuffer* hidden_unit_meta =
      SetCommMetaHiddenUnitBufferForEP(meta, DataType::TYPE_UINT32, forwarding_context);

  HiddenUnitDeviceBuffer* hidden_unit = SetHiddenUnitBufferForEP(residual_buffer, forwarding_context);

  // Memcpy use the default stream.
  StreamSynchronize(forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);

  KLLM_LOG_DEBUG << fmt::format("SendHiddenUnitBufferForEP hidden_unit_meta rank_: {},  shape: {} {}, dtype: {}",
                                forwarding_context.GetCurrentRank(),
                                hidden_unit_meta->tensors[forwarding_context.GetCurrentRank()].shape[0],
                                hidden_unit_meta->tensors[forwarding_context.GetCurrentRank()].shape[1],
                                hidden_unit_meta->tensors[forwarding_context.GetCurrentRank()].ToString());

  // Need, to avoid potential data race when using wait-notify.
  // rank 0 ---> ResetExpertWaiter ---> ExpertNotify().notify
  // rank 1 ----------------------------------------------------> ResetExpertWaiter --> ExpertWait();
  GetExpertHiddenUnitBufferPool()->ExpertBarrier();

  if (forwarding_context.GetCurrentRank() == 0) {
    SendExpertHiddenUnits(hidden_unit_meta, false /*is_sync*/);
    KLLM_LOG_DEBUG << "SendHiddenUnitBufferForEP send comm meta finished";

    // Send: ep worker send hidden_state to other ep workers, sync operator.
    // only expert-worker0 send.
    SendExpertHiddenUnits(hidden_unit, is_sync);

    KLLM_LOG_DEBUG << "SendHiddenUnitBufferForEP send real data finished";
  }
  GetExpertHiddenUnitBufferPool()->ExpertBarrier();
}

std::vector<Tensor>& ExpertParallelDataTransfer::RecvHiddenUnitBufferForEP(ForwardingContext& forwarding_context) {
  static std::vector<Tensor> empty;
  if (forwarding_context.GetContext()->IsExpertParallelStandalone()) {
    return empty;
  }
  // Receive: collect hidden_state from other nodes. not need?

  // rank = 0, receive hidden_states from other nodes.
  HiddenUnitDeviceBuffer* hidden_unit = RecvExpertHiddenUnits(forwarding_context.GetCurrentRank());
  if (hidden_unit != nullptr) {
    hidden_device_buffer_.push_back(hidden_unit);
    size_t rank = forwarding_context.GetCurrentRank();
    KLLM_LOG_DEBUG << fmt::format(
        "RecvHiddenUnitBufferForEP rank: {}, shape: {} {}, dtype: {} ptr: {}, tensors.size: {}",
        forwarding_context.GetCurrentRank(), hidden_unit->tensors[rank].shape[0], hidden_unit->tensors[rank].shape[1],
        hidden_unit->tensors[rank].ToString(), hidden_unit->tensors[rank].GetPtr<void>(), hidden_unit->tensors.size());

    std::vector<Tensor>& buffer = GetExpertRecvHiddenUnitBufferRef(hidden_unit, forwarding_context);

    return buffer;
  } else {
    KLLM_LOG_INFO << "RecvHiddenUnitBufferForEP not recv data";
    return empty;
  }
}

std::vector<Tensor>& ExpertParallelDataTransfer::AsyncRecvHiddenUnitBufferForEP(ForwardingContext& forwarding_context) {
  static std::vector<Tensor> empty;
  if (forwarding_context.GetContext()->IsExpertParallelStandalone()) {
    return empty;
  }
  // Receive: collect hidden_state from other nodes. not need?
  ResetExpertReceiveWaiter();

  // rank = 0, receive hidden_states from other nodes.
  HiddenUnitDeviceBuffer* hidden_unit = AsyncRecvExpertHiddenUnits(forwarding_context.GetCurrentRank());
  if (hidden_unit != nullptr) {
    hidden_device_buffer_.push_back(hidden_unit);

    std::vector<Tensor>& buffer = GetExpertRecvHiddenUnitBufferRef(hidden_unit, forwarding_context);
    KLLM_LOG_DEBUG << "Recv buffer shape: " << buffer[0].shape[0] << ", dtype: " << buffer[0].ToString()
                   << ", refer_ptr " << buffer[0].GetPtr<void>();

    return buffer;
  } else {
    KLLM_LOG_INFO << "AsyncRecvHiddenUnitBufferForEP not recv data";
    return empty;
  }
}

// TODO(xingjinglu): To support async mode of expert parallel, every expert node may send and receive token-hidden
// states asynchronously.
void ExpertParallelDataTransfer::CombineHiddenUnitBufferForEP(std::vector<Tensor>& residual_buffer,
                                                              ForwardingContext& forwarding_context) {
  return;
}

std::vector<Tensor>& ExpertParallelDataTransfer::GetExpertRecvHiddenUnitBufferRef(
    HiddenUnitDeviceBuffer* hidden_unit, ForwardingContext& forwarding_context) {
  if (forwarding_context.GetContext()->IsExpertParallelStandalone()) {
    // Should not execute here.
    KLLM_LOG_INFO << "Not expert parallel mode, return empty result.";
    return local_residual_buffer_;
  }

#ifdef ENABLE_ACL
  if (forwarding_context.GetModelInput()->infer_stage == InferStage::STAGE_CONTEXT) {
    if (distributed_device_buffer_prefill_.empty()) {
      HiddenUnitDeviceBuffer* device_buffer = hidden_unit;
      if (!distributed_device_buffer_.empty()) distributed_device_buffer_.clear();

      distributed_device_buffer_prefill_.push_back(device_buffer->prefill_tensors[forwarding_context.GetCurrentRank()]);
    }

    return distributed_device_buffer_prefill_;
  } else {
#endif
    if (!distributed_device_buffer_.empty()) distributed_device_buffer_.clear();

    distributed_device_buffer_.push_back(hidden_unit->tensors[forwarding_context.GetCurrentRank()]);
    KLLM_LOG_DEBUG << "GetExpertRecvHiddenUnitBufferRef rank: " << forwarding_context.GetCurrentRank()
                   << ", dtype: " << distributed_device_buffer_[0].ToString() << ", refer_ptr "
                   << distributed_device_buffer_[0].GetPtr<void>();

    return distributed_device_buffer_;
#ifdef ENABLE_ACL
  }
#endif
}

// Now used for expert parallel.
HiddenUnitDeviceBuffer* ExpertParallelDataTransfer::SetHiddenUnitBufferForEP(const std::vector<Tensor>& residual_buffer,
                                                                             ForwardingContext& forwarding_context) {
  // Copy to hidden_unit_buffer if not standalone.
  if (!forwarding_context.GetContext()->IsExpertParallelStandalone()) {
    bool is_prefill = forwarding_context.GetModelInput()->infer_stage == InferStage::STAGE_CONTEXT;
    auto working_stream = forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()];
    StreamSynchronize(working_stream);

    HiddenUnitDeviceBuffer* hidden_unit =
        GetExpertHiddenUnitBufferPool()->GetDeviceBuffer(forwarding_context.GetCurrentRank());
    if (!hidden_unit) {
      KLLM_LOG_WARNING << "SetHiddenUnitBufferForEP no device buffer got for sending data.";
      return nullptr;
    }
    CopyToHiddenUnitBuffer(hidden_unit, const_cast<Tensor&>(residual_buffer[0]), forwarding_context.GetCurrentRank(),
                           is_prefill, working_stream);

    return hidden_unit;
  }
  return nullptr;
}

HiddenUnitDeviceBuffer* ExpertParallelDataTransfer::SetCommMetaHiddenUnitBufferForEP(
    expert_parallel_comm_meta& meta_data, DataType dtype, ForwardingContext& forwarding_context) {
  // Copy to hidden_unit_buffer if not standalone.
  if (!forwarding_context.GetContext()->IsExpertParallelStandalone()) {
    bool is_prefill = forwarding_context.GetModelInput()->infer_stage == InferStage::STAGE_CONTEXT;

    // Not need.
    StreamSynchronize(forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
    KLLM_LOG_DEBUG << fmt::format("SetCommMetaHiddenUnitBufferForEP meta shape: {} {}, rank: {}", meta_data.shape_0,
                                  meta_data.shape_1, forwarding_context.GetCurrentRank());
    HiddenUnitDeviceBuffer* hidden_unit =
        GetExpertHiddenUnitBufferPool()->GetCommMetaDeviceBuffer(forwarding_context.GetCurrentRank());
    if (!hidden_unit) {
      KLLM_LOG_WARNING << "SetCommMetaHiddenUnitBufferForEP no device buffer got for sending data.";
      return nullptr;
    }
    CopyHostMemToHiddenUnitBuffer(hidden_unit, reinterpret_cast<void*>(&meta_data),
                                  {sizeof(expert_parallel_comm_meta), 1}, DataType::TYPE_UINT8,
                                  forwarding_context.GetCurrentRank(), is_prefill);

    return hidden_unit;
  }
  return nullptr;
}

void ExpertParallelDataTransfer::FreeHiddenUnitDeviceBuffer(ForwardingContext& forwarding_context) {
  if (forwarding_context.GetCurrentRank() == 0) {
    while (!hidden_device_buffer_.empty()) {
      HiddenUnitDeviceBuffer* hidden_unit_buffer = hidden_device_buffer_.back();
      FreeExpertRecvHiddenUnits(hidden_unit_buffer);
      hidden_device_buffer_.pop_back();
    }
  }

  GetExpertHiddenUnitBufferPool()->ExpertBarrier();
  return;
}

}  // namespace ksana_llm
