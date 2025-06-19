/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/communicator/tp_communicator.h"

#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

template <typename T>
Status TpCommunicator<T>::AllReduce(std::vector<Tensor>& reduce_buffer_tensors,
                                    std::vector<Tensor>& hidden_buffer_tensors_0, const bool is_multi_token_forward,
                                    ForwardingContext<T>& forwarding_context) {
  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!forwarding_context.context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(forwarding_context.model_output_->compute_ready_event,
                forwarding_context.context_->GetComputeStreams()[forwarding_context.rank_]);
    StreamWaitEvent(forwarding_context.context_->GetCommStreams()[forwarding_context.rank_],
                    forwarding_context.model_output_->compute_ready_event);
  }

  // AllReduceSum
  if (forwarding_context.model_communicator_) {
    forwarding_context.model_communicator_->ReduceSum(reduce_buffer_tensors, hidden_buffer_tensors_0,
                                                      is_multi_token_forward, /*use_custom*/ true);
  }
  return Status();
}

template <typename T>
Status TpCommunicator<T>::AllGather(Tensor& gather_tensor, Tensor& buffer, ForwardingContext<T>& forwarding_context) {
  if (!forwarding_context.model_communicator_) {
    return Status();
  }

  std::vector<Tensor> input{gather_tensor, buffer};
  std::vector<Tensor> output{gather_tensor};
  forwarding_context.model_communicator_->AllGather(input, output);
  gather_tensor = std::move(output[0]);
  return Status();
}

template class TpCommunicator<float>;
template class TpCommunicator<float16>;
#ifdef ENABLE_BFLOAT16
template class TpCommunicator<bfloat16>;
#endif

}  // namespace ksana_llm
