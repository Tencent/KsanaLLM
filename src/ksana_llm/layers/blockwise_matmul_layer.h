/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_FP8
#  include "ksana_llm/layers/base_layer.h"
#  include "ksana_llm/layers/deepgemm_matmul_layer.h"
#  ifdef ENABLE_CUDA
#    include "csrc/kernels/nvidia/deepgemm_aot_wrapper/deepgemm_aot_wrapper.h"
#  endif
namespace ksana_llm {

class BlockwiseMatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

  virtual size_t GetWorkSpaceSize() override;

  virtual Status SetWorkSpaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) override;

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  size_t max_m_;
  size_t n_;
  size_t k_;
  size_t block_size_;
  size_t workspace_size_;
  size_t kDeepGemmMaxMThreshold_ = 0;
  size_t input_buffer_size_ = 0;
  size_t gemm_workspace_size_ = 0;

  // key: "DataType_max_m_k_n", value: cutlass_buffer_size
  static inline std::unordered_map<std::string, size_t> cutlass_buffer_size_cache_;
  static inline std::mutex cache_mutex_;

#  ifdef ENABLE_CUDA
  DeepGemmMatMulLayer deepgemm_matmul_layer_;
#  endif
};

}  // namespace ksana_llm
#endif
