/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <vector>
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// Copy model output to samplers.
class ModelOutput {
 public:
  ModelOutput(const size_t max_logits_num, const size_t vocab_size, const int rank,
              const std::shared_ptr<Context> context, const size_t max_hidden_buffer_size, const DataType hidden_dtype);
  ~ModelOutput();

 public:
  // Whether the compute is ready for output.
  Event compute_ready_event;

  // Store logist result, shape: [max_logits_num, vocab_size], dtype: float
  Tensor logits_tensor;
  // Store output tokens on the host, shape: [max_logits_num], dtype: int
  Tensor output_tokens_host_tensor;

 private:
  int rank_;
  std::shared_ptr<Context> context_;
};

}  // namespace ksana_llm
