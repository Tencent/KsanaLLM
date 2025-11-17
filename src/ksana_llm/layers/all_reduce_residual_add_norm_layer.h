/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/forwarding_context.h"
#include "ksana_llm/layers/add_norm_layer.h"
#include "ksana_llm/models/communicator/tp_communicator.h"

namespace ksana_llm {

/**
 * This layer is used to fuse the allreduce and residual add norm operation
 * into one kernel operation for dtype: fp16, bf16
 * Step 1: tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0)
 * Step 2: adds->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer)
 * Step 3: layernorm_layer_->Forward(residual_buffer, hidden_buffer_tensors_0)
 */
class AllReduceResidualAddNormLayer : public BaseLayer {
 public:
  virtual ~AllReduceResidualAddNormLayer() = default;

  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                        std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const bool is_multi_token_forward, ForwardingContext& forwarding_context);

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  bool is_init_ = false;
  float rms_norm_eps_;
  Tensor rms_norm_weight_;
  std::shared_ptr<TpCommunicator> tp_comm_;
  std::shared_ptr<AddNormLayer> add_norm_layer_;
};

}  // namespace ksana_llm
