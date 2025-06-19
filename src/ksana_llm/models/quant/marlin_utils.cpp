/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/quant/marlin_utils.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {

MarlinUtils::MarlinUtils(std::shared_ptr<Context> context, int rank, int bits, int groupsize)
    : context_(context), rank_(rank), bits_(bits), groupsize_(groupsize) {
  pack_factor_ = 32 / bits_;
}

#ifdef ENABLE_CUDA

// https://github.com/vllm-project/vllm/blob/v0.5.3/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L162
template <typename T>
torch::Tensor MarlinUtils::MarlinPermuteScales(torch::Tensor s, int k, int n) {
  torch::Tensor permute_s = torch::empty_like(s);
  InvokeMarlinPermuteScales<T>(context_->GetMemoryManageStreams()[rank_].Get(), s.data_ptr(), permute_s.data_ptr(), k,
                               n, groupsize_);
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
  return permute_s;
}

torch::Tensor MarlinUtils::MarlinUnpackCols(const torch::Tensor& packed_q_w, int k, int n) {
  torch::Tensor q_res = torch::zeros({k, n}, torch::kInt32);
  torch::Tensor packed_w = packed_q_w.clone();
  int mask = (1 << bits_) - 1;
  for (int i = 0; i < pack_factor_; ++i) {
    torch::Tensor vals = packed_w.bitwise_and(mask);
    packed_w = packed_w.bitwise_right_shift(bits_);
    q_res.index_put_({torch::indexing::Slice(), torch::indexing::Slice(i, torch::indexing::None, pack_factor_)}, vals);
  }
  return q_res.contiguous();
}

torch::Tensor MarlinUtils::MarlinPackCols(const torch::Tensor& q_w, int k, int n) {
  torch::Tensor q_res = torch::zeros({k, n / pack_factor_}, torch::kInt32);
  for (int i = 0; i < pack_factor_; ++i) {
    torch::Tensor val =
        q_w.index({torch::indexing::Slice(), torch::indexing::Slice(i, torch::indexing::None, pack_factor_)});
    val = val.bitwise_left_shift(bits_ * i);
    q_res = q_res.bitwise_or(val);
  }
  return q_res.contiguous();
}

// https://github.com/vllm-project/vllm/blob/v0.5.3/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L188
torch::Tensor MarlinUtils::MarlinZeroPoints(const torch::Tensor& zp_, int k, int n) {
  std::vector<int64_t> scale_perm_vec;
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      scale_perm_vec.push_back(i + 8 * j);
    }
  }
  torch::Tensor scale_perm = torch::tensor(scale_perm_vec, torch::kInt64);

  torch::Tensor zp = zp_.clone();
  zp = zp.reshape({-1, scale_perm.size(0)});
  zp = zp.index_select(1, scale_perm);

  torch::Tensor interleave;
  if (bits_ == 4) {
    interleave = torch::tensor({0, 2, 4, 6, 1, 3, 5, 7}, torch::kInt64);
  } else if (bits_ == 8) {
    interleave = torch::tensor({0, 2, 1, 3}, torch::kInt64);
  }

  zp = zp.reshape({-1, interleave.size(0)});
  zp = zp.index_select(1, interleave);
  zp = zp.reshape({-1, n}).contiguous();

  zp = MarlinPackCols(zp, k, n);

  return zp;
}

// https://github.com/vllm-project/vllm/blob/v0.5.3/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L201
torch::Tensor MarlinUtils::MarlinAwqToMarlinZeroPoints(const torch::Tensor& q_zp_packed, int k, int n) {
  torch::Tensor q_zp = MarlinUnpackCols(q_zp_packed, k, n);
  torch::Tensor undo_interleave;
  if (bits_ == 4) {
    undo_interleave = torch::argsort(torch::tensor({0, 2, 4, 6, 1, 3, 5, 7}));
  } else if (bits_ == 8) {
    undo_interleave = torch::argsort(torch::tensor({0, 2, 1, 3}));
  }
  q_zp = q_zp.reshape({-1, undo_interleave.size(0)});
  q_zp = q_zp.index_select(1, undo_interleave);
  q_zp = q_zp.reshape({-1, n}).contiguous();

  torch::Tensor marlin_zp = MarlinZeroPoints(q_zp, k, n);
  return marlin_zp;
}

torch::Tensor MarlinUtils::MarlinSortGIdx(torch::Tensor& g_idx) {
  torch::Tensor g_idx_sort_indices = torch::argsort(g_idx, true, 0, false);
  g_idx = g_idx.index_select(0, g_idx_sort_indices).contiguous();
  return g_idx_sort_indices.to(torch::kInt32);
}

torch::Tensor MarlinUtils::PackGptqWeight(torch::Tensor& qweight, std::optional<torch::Tensor> perm) {
  int k = qweight.size(0) * pack_factor_;
  int n = qweight.size(1);
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank_);
  torch::Tensor processed_tensor = torch::empty(GetMarlinGptqRepackMeta(k, n, bits_), options);
  InvokeMarlinGptqRepack(qweight.data_ptr(), perm.has_value() ? perm->data_ptr() : nullptr, processed_tensor.data_ptr(),
                         k, n, bits_, perm.has_value(), rank_, context_->GetMemoryManageStreams()[rank_].Get());
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
  return processed_tensor;
}

torch::Tensor MarlinUtils::PackAwqWeight(torch::Tensor& qweight) {
  int k = qweight.size(0);
  int n = qweight.size(1) * pack_factor_;
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank_);
  torch::Tensor processed_tensor = torch::empty(GetMarlinAwqRepackMeta(k, n, bits_), options);
  InvokeMarlinAwqRepack(qweight.data_ptr(), processed_tensor.data_ptr(), k, n, bits_, rank_,
                        context_->GetMemoryManageStreams()[rank_].Get());
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
  return processed_tensor;
}

template torch::Tensor MarlinUtils::MarlinPermuteScales<float>(torch::Tensor s, int k, int n);
template torch::Tensor MarlinUtils::MarlinPermuteScales<float16>(torch::Tensor s, int k, int n);
#  ifdef ENABLE_BFLOAT16
template torch::Tensor MarlinUtils::MarlinPermuteScales<bfloat16>(torch::Tensor s, int k, int n);
#  endif

#endif

}  // namespace ksana_llm