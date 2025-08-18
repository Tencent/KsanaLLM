/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/mem_adjuster_layer.h"

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status MemAdjusterLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                              std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  return Status();
}

Status MemAdjusterLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  KLLM_THROW("MemAdjusterLayer::Forward is not supported, please use other specific functions instead");
  return Status(RET_UNDEFINED_REFERENCE, "MemAdjusterLayer::Forward not supported.");
}

Status MemAdjusterLayer::GatherSubmatrix(const Tensor& input, Tensor& output_tensor, size_t dp_group_id,
                                         const std::vector<int>& dp_token_offset, size_t max_seq_len, size_t tp_size,
                                         Tensor& workspace_tensor) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, GatherSubmatrixT, input, output_tensor, dp_group_id, dp_token_offset,
                      max_seq_len, tp_size, workspace_tensor);
}

Status MemAdjusterLayer::DpMapCopy(const Tensor& input, Tensor& output_tensor, const std::vector<int>& dp_token_offset,
                                   Tensor& workspace_tensor) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, DpMapCopyT, input, output_tensor, dp_token_offset, workspace_tensor);
}

template <typename T>
Status MemAdjusterLayer::GatherSubmatrixT(const Tensor& input, Tensor& output_tensor, size_t dp_group_id,
                                          const std::vector<int>& dp_token_offset, size_t max_seq_len, size_t tp_size,
                                          Tensor& workspace_tensor) {
  std::vector<size_t> m_num_per_group;
  size_t output_m = 0;
  for (size_t i = 0; i < dp_token_offset.size(); i += 4) {
    size_t seq_len = dp_token_offset[i + 1] - dp_token_offset[i] + dp_token_offset[i + 3] - dp_token_offset[i + 2];
    m_num_per_group.push_back(seq_len);
    output_m += seq_len;
  }
  size_t n = input.shape[1];
  size_t col_num = n / tp_size;
  size_t n_start = dp_group_id * col_num;
  InvokeGatherSubmatrix(input.GetPtr<T>(), output_tensor.GetPtr<T>(), m_num_per_group, max_seq_len, tp_size, n_start,
                        n_start + col_num, max_seq_len * tp_size, n, workspace_tensor.GetPtr<void>(),
                        context_->GetComputeStreams()[rank_].Get());
  output_tensor.shape[0] = output_m;
  output_tensor.shape[1] = col_num;
  return Status();
}

template <typename T>
Status MemAdjusterLayer::DpMapCopyT(const Tensor& input, Tensor& output_tensor, const std::vector<int>& dp_token_offset,
                                    Tensor& workspace_tensor) {
  std::vector<size_t> group_info(dp_token_offset.size());
  size_t last_group_row_offset = 0;
  size_t prefill_start_offset = 0;
  if (dp_token_offset.size() % 4 != 0) {
    KLLM_THROW("invalid dp_token_offset");
  }
  for (size_t i = 0; i < dp_token_offset.size(); i += 4) {
    size_t prefill_num = dp_token_offset[i + 1] - dp_token_offset[i];
    size_t decode_num = dp_token_offset[i + 3] - dp_token_offset[i + 2];
    last_group_row_offset += prefill_num + decode_num;
    group_info[i] = last_group_row_offset;
    group_info[i + 1] = prefill_num;
    group_info[i + 2] = prefill_start_offset;
    prefill_start_offset += prefill_num;
  }
  size_t decode_start_offset = prefill_start_offset;
  for (size_t i = 0; i < dp_token_offset.size(); i += 4) {
    size_t decode_num = dp_token_offset[i + 3] - dp_token_offset[i + 2];
    group_info[i + 3] = decode_start_offset;
    decode_start_offset += decode_num;
  }
  if (decode_start_offset != last_group_row_offset || input.shape[0] != last_group_row_offset) {
    KLLM_THROW("invalid input");
  }

  InvokeDpMapCopy(input.GetPtr<T>(), output_tensor.GetPtr<T>(), group_info, input.shape[0], input.shape[1],
                  workspace_tensor.GetPtr<void>(), context_->GetComputeStreams()[rank_].Get());
  output_tensor.shape[0] = input.shape[0];
  output_tensor.shape[1] = input.shape[1];
  return Status();
}

}  // namespace ksana_llm
