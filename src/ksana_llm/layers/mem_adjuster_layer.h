/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

class MemAdjusterLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

  Status GatherSubmatrix(const Tensor& input, Tensor& output_tensor, size_t dp_group_id,
                         const std::vector<int>& dp_token_offset, size_t max_seq_len, size_t tp_size,
                         Tensor& workspace_tensor);

  Status DpMapCopy(const Tensor& input, Tensor& output_tensor, const std::vector<int>& dp_token_offset,
                   Tensor& workspace_tensor);

 private:
  // Gather submatrices from input tensor [M, N] according to data parallel (DP) group mapping.
  // Each DP group contributes a submatrix with its own sequence length.
  // Output shape: [m1 + m2 + ... + mk, N / tp_size]
  //   - m1, m2, ..., mk: sequence lengths from each DP group
  //   - N / tp_size: feature dimension after tensor parallel partitioning
  template <typename T>
  Status GatherSubmatrixT(const Tensor& input, Tensor& output_tensor, size_t dp_group_id,
                          const std::vector<int>& dp_token_offset, size_t max_seq_len, size_t tp_size,
                          Tensor& workspace_tensor);

  /**
   * Redistributes input tensor by reorganizing prefill and decode tokens across DP groups.
   *
   * Transforms the interleaved token layout into a grouped layout where all prefill
   * tokens appear before all decode tokens.
   *
   * Input:  [mp0+md0+mp1+md1+...+mpK+mdK, N] - interleaved layout
   * Output: [mp0+mp1+...+mpK + md0+md1+...+mdK, N] - grouped layout
   *
   * Where:
   *   - mpk: prefill token count for k-th DP group
   *   - mdk: decode token count for k-th DP group
   *   - K: number of DP groups (0 to K)
   *   - N: feature dimension
   */
  template <typename T>
  Status DpMapCopyT(const Tensor& input, Tensor& output_tensor, const std::vector<int>& dp_token_offset,
                    Tensor& workspace_tensor);
};

}  // namespace ksana_llm
