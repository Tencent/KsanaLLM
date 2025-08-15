/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

class BatchedMatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
#ifdef ENABLE_CUDA
  void* cublas_workspace_ptr_{nullptr};
  int cublas_workspace_size_{-1};
  cublasLtMatmulAlgo_t* cublaslt_algo_ptr_{nullptr};
#endif

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);
};

}  // namespace ksana_llm
