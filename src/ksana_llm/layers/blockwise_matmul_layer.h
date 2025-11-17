/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_FP8
#  include <mutex>
#  include "ksana_llm/layers/base_layer.h"
#  ifdef ENABLE_CUDA
#    include "csrc/kernels/nvidia/gemm/deepgemm/deepgemm_wrapper.h"
#  endif

namespace ksana_llm {

class BlockwiseMatMulLayer : public BaseLayer {
 public:
  enum class Fp8GemmType : int32_t { Dynamic, Cutlass, DeepGemm, DeepGemmSwapAB };

  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) override;

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  template <typename T>
  size_t GetWorkspaceSizeT();

  template <typename T>
  size_t GetCachedCutlassBufferSize();

  void SetGemmType(BlockwiseMatMulLayer::Fp8GemmType gemm_type);

  BlockwiseMatMulLayer::Fp8GemmType PickGemmType(size_t m);

  void BuildDeepGemmKernels();

 private:
  size_t max_m_;
  size_t n_;
  size_t k_;
  size_t block_size_;
  Tensor weight_;

  size_t workspace_size_;
  size_t input_buffer_size_ = 0;
  size_t cutlass_gemm_workspace_size_ = 0;

  // 使用deepgemm时，m的最大值，必须是kAlignSize_的整数倍
  // 超过该值时，使用cutlass
  size_t deepgemm_max_m_threshold_ = 0;
  // 使用deepgemm_swap_ab时，m的最大值，必须是kAlignSize_的整数倍
  // 超过该值时，使用deepgemm
  size_t swap_ab_max_m_threshold_ = 0;
  bool deepgemm_enabled_ = false;
  BlockwiseMatMulLayer::Fp8GemmType gemm_type_ = BlockwiseMatMulLayer::Fp8GemmType::Dynamic;

  static constexpr size_t kAlignSize_ = 4;

  // When true, it indicates that quantization has already been performed elsewhere and stored in the workspace buffer
  bool skip_quant_ = false;

#  ifdef ENABLE_CUDA
  std::shared_ptr<llm_kernels::nvidia::DeepGEMMWrapper> deepgemm_wrapper_;
#  endif
};

}  // namespace ksana_llm
#endif
