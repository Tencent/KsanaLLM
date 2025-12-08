/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/base_weight.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/gemm/finegrained_mixed_dtype_gemm/finegrained_mixed_dtype_gemm_wrapper.h"
#endif

namespace ksana_llm {

struct FinegrainedMixedDtypeGemmLayerParameters : public BaseLayerParameters {
  size_t m{0};
  size_t n{0};
  size_t k{0};
  size_t group_size{0};
  bool has_zero{false};
  DataType activation_type{TYPE_INVALID};
  DataType output_type{TYPE_INVALID};

  std::string ToString() const {
    return fmt::format("m:{}, n:{}, k:{}, group_size:{}, has_zero:{}, activation_type:{}, output_type:{}", m, n, k,
                       group_size, has_zero, GetTypeString(activation_type), GetTypeString(output_type));
  }
};

#ifdef ENABLE_CUDA
class FinegrainedMixedDtypeGemmLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  size_t GetKernelWorkspaceSize();

  int64_t GetBestTactic(const size_t& m);

 private:
  struct WorkspaceInfo {
    bool set_workspace_buffer_info_{true};

    size_t kernel_workspace_size_{0};
    size_t quantized_size_{0};

    void* kernel_workspace_ptr_{nullptr};
    void* quantized_input_ptr_{nullptr};

    size_t workspace_size_{0};
    inline size_t GetTotalSize() {
      workspace_size_ = kernel_workspace_size_ + quantized_size_;
      return workspace_size_;
    }

    inline void SetPtr(void* ptr) {
      if (set_workspace_buffer_info_) {
        set_workspace_buffer_info_ = false;
        kernel_workspace_ptr_ = ptr;
        quantized_input_ptr_ = kernel_workspace_ptr_ + kernel_workspace_size_;
      }
    }
  } workspace_info_;
  struct FinegrainedMixedDtypeGemmLayerParameters params_;
  std::vector<std::vector<int64_t>> tactics_;
  std::shared_ptr<llm_kernels::nvidia::FinegrainedMixedDtypeGemmWrapper> wrapper_;
};
#endif
}  // namespace ksana_llm
