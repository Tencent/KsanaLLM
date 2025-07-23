/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

template <typename T>
class EmbLookupLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 protected:
  // The scale for word embedding.
  T emb_scale_{static_cast<T>(1.f)};

  // The position encoding weight.
  void* pos_weight_{nullptr};

#ifdef ENABLE_ACL
  llm_kernels::utils::ATBOperationExecutor atb_op_executor_;
#endif  // ENABLE_ACL
};

template <typename T>
class CpuEmbLookupLayer : public BaseLayer {
 public:
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;
};

}  // namespace ksana_llm
