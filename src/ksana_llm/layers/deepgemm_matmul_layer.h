/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_FP8
#  include "ksana_llm/layers/base_layer.h"
#  ifdef ENABLE_CUDA
#    include "csrc/kernels/nvidia/gemm/deepgemm/deepgemm_wrapper.h"
#  endif
namespace ksana_llm {

class DeepGemmMatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

  virtual size_t GetWorkSpaceSize() override;

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  size_t max_m_;
  size_t n_;
  size_t k_;
  size_t block_size_;
#  ifdef ENABLE_CUDA
  std::shared_ptr<llm_kernels::nvidia::DeepGEMMWrapper> deepgemm_wrapper_;
#  endif
};

}  // namespace ksana_llm
#endif
