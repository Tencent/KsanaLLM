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
  if (!forwarding_context.GetContext()->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(forwarding_context.GetModelOutput()->compute_ready_event,
                forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
    StreamWaitEvent(forwarding_context.GetContext()->GetCommStreams()[forwarding_context.GetCurrentRank()],
                    forwarding_context.GetModelOutput()->compute_ready_event);
  }

  // AllReduceSum
  if (forwarding_context.GetModelCommunicator()) {
    forwarding_context.GetModelCommunicator()->ReduceSum(reduce_buffer_tensors, hidden_buffer_tensors_0,
                                                         is_multi_token_forward, /*use_custom*/ true);
  }
  return Status();
}

template <typename T>
Status TpCommunicator<T>::AllGather(Tensor& gather_tensor, Tensor& buffer, ForwardingContext<T>& forwarding_context) {
  if (!forwarding_context.GetModelCommunicator()) {
    return Status();
  }

  std::vector<Tensor> input{gather_tensor, buffer};
  std::vector<Tensor> output{gather_tensor};
  forwarding_context.GetModelCommunicator()->AllGather(input, output);
  gather_tensor = std::move(output[0]);
  return Status();
}

template class TpCommunicator<float>;
template class TpCommunicator<float16>;
#ifdef ENABLE_BFLOAT16
template class TpCommunicator<bfloat16>;
#endif

}  // namespace ksana_llm
