/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/forwarding_context.h"

namespace ksana_llm {

class TpCommunicator {
 public:
  TpCommunicator() {}
  ~TpCommunicator() = default;

  Status AllReduce(std::vector<Tensor>& reduce_buffer_tensors, std::vector<Tensor>& hidden_buffer_tensors_0,
                   const bool is_multi_token_forward, ForwardingContext& forwarding_context);

  Status AllGather(Tensor& gather_tensor, Tensor& buffer, ForwardingContext& forwarding_context);
};  // namespace ksana_llm

}  // namespace ksana_llm
