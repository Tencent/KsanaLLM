/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_output.h"

namespace ksana_llm {

ModelOutput::ModelOutput(const size_t max_logits_num, const size_t vocab_size, const int rank,
                         const std::shared_ptr<Context> context, const size_t max_hidden_buffer_size,
                         const DataType hidden_dtype)
    : rank_(rank), context_(context) {
  logits_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {max_logits_num, vocab_size}, rank_);
  output_tokens_host_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {max_logits_num}, rank_);

  EventCreateWithFlags(&compute_ready_event, EVENT_DISABLE_TIMING);
}

ModelOutput::~ModelOutput() { EventDestroy(compute_ready_event); }

}  // namespace ksana_llm
