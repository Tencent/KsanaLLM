/* Copyright 2024 Tencent Inc.  All rights reserved.
 *  *  * ==============================================================================*/
#pragma once
#include "csrc/utils/nvidia/workspace.h"
#include "ksana_llm/layers/moe_layer.h"
#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/common_moe/moe_config.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA

class Fp8MoeLayer : public MoeLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status SetWorkspaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) override;

  virtual Status Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  template <typename T>
  size_t GetWorkspaceSizeT();

  template <typename T>
  Status PreprocessT(const ModelConfig& model_config_, const RuntimeConfig& runtime_config);

  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  size_t quant_buffer_size_;
  void* quant_buffer_ptr_;
  size_t best_config_index_;
};
#endif
}  // namespace ksana_llm
