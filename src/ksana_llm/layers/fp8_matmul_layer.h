/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_FP8
#  include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

class Fp8MatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

  size_t GetWorkSpaceSize(const int m, const int k);

  virtual size_t GetWorkSpaceSize() override;

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  int max_m_;
  int max_k_;

#  ifdef ENABLE_CUDA
  void* cublas_workspace_ptr_{nullptr};
  size_t cublas_workspace_size_{0};
  int cublas_workspace_block_id_{-1};
  cublasLtMatmulAlgo_t* cublaslt_algo_ptr_{nullptr};
#  endif
};

}  // namespace ksana_llm
#endif
