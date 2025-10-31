/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/layers/add_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

class AddNormLayer : public BaseLayer {
 public:
  virtual ~AddNormLayer() = default;

  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  float rms_norm_eps_;
  std::shared_ptr<LayernormLayer> layernorm_layer_;
  std::shared_ptr<AddLayer> add_layer_;
  bool enable_pdl_{false};
};

}  // namespace ksana_llm
