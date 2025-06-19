/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/quant/cutlass_utils.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA

// Follow the logic from
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.20.0rc3/tensorrt_llm/quantization/functional.py#L971
torch::Tensor CutlassUtils::CutlassGetReverseOrder(const torch::Tensor& iweights) {
  torch::Tensor index_select_tensor =
      torch::tensor({0, 4, 1, 5, 2, 6, 3, 7}, torch::kInt64).to(torch::Device(iweights.device()));
  torch::Tensor reverse_order_tensor =
      torch::arange(iweights.size(-1), torch::kInt64).to(torch::Device(iweights.device()));
  reverse_order_tensor = reverse_order_tensor.view({-1, 32 / bits_});
  reverse_order_tensor = reverse_order_tensor.index_select(1, index_select_tensor).contiguous();
  reverse_order_tensor = reverse_order_tensor.view({-1});
  return reverse_order_tensor;
}

torch::Tensor CutlassUtils::CutlassUnpackQWeight(const torch::Tensor& qtensor) {
  torch::Tensor shifts = torch::arange(0, 32, bits_).to(torch::Device(qtensor.device())).unsqueeze(0).unsqueeze(0);
  torch::Tensor itensor = torch::bitwise_right_shift(qtensor.unsqueeze(-1), shifts).to(torch::kInt8);
  itensor = itensor.view({itensor.size(0), -1});
  return itensor;
}

// Follow the logic from https://github.com/casper-hansen/AutoAWQ/blob/v0.2.6/awq/utils/packing_utils.py
torch::Tensor CutlassUtils::CutlassUnpackAWQ(const torch::Tensor& qweight) {
  torch::Tensor iweight = CutlassUnpackQWeight(qweight);
  torch::Tensor reverse_order_tensor = CutlassGetReverseOrder(iweight);
  iweight = iweight.index_select(1, reverse_order_tensor).contiguous();
  auto mask = (1 << bits_) - 1;
  iweight = torch::bitwise_and(iweight, mask);
  return iweight.contiguous();
}

// Unpack [k/groupsize,n]int32 to [k,n]int4
torch::Tensor CutlassUtils::CutlassUnpackGPTQ(const torch::Tensor& w_packed) {
  auto w_packed_int4x2 = w_packed.t().contiguous().view(torch::kUInt8);
  auto w_unpacked = torch::zeros({w_packed_int4x2.size(0), w_packed_int4x2.size(1) * 2}, torch::kInt8)
                        .to(torch::Device(w_packed.device()));
  w_unpacked.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                        w_packed_int4x2 % 16);
  w_unpacked.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                        w_packed_int4x2 / 16);
  return w_unpacked.t().contiguous();
}

// Pack [k,n]int4 to [k,n/2]int8, each byte save 2 int4 weight
torch::Tensor CutlassUtils::CutlassPackInt8ToPackedInt4(torch::Tensor weight) {
  torch::Tensor reshaped_weight = weight.reshape({weight.size(0), -1, 2});
  torch::Tensor high = torch::bitwise_left_shift(torch::bitwise_and(reshaped_weight.select(2, 1), 0x0F), 4);
  torch::Tensor low = torch::bitwise_and(reshaped_weight.select(2, 0), 0x0F);
  torch::Tensor packed_weight = torch::bitwise_or(high, low);
  return packed_weight.to(torch::kInt8);
}

torch::Tensor CutlassUtils::CutlassPreprocessWeightsForMixedGemmWarpper(torch::Tensor row_major_quantized_weight,
                                                                        llm_kernels::nvidia::QuantType quant_type) {
  std::vector<int32_t> row_permutation = get_permutation_map(quant_type);
  auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
  torch::Tensor row_permutation_tensor = torch::from_blob(row_permutation.data(), {row_permutation.size()}, options);
  row_permutation_tensor = row_permutation_tensor.to(row_major_quantized_weight.device());

  const size_t bits_in_quant_type = get_weight_quant_bits(quant_type);
  const size_t num_experts = row_major_quantized_weight.dim() == 2 ? 1 : row_major_quantized_weight.size(0);
  const size_t num_rows = row_major_quantized_weight.size(-2);
  const size_t num_cols = (8 / bits_in_quant_type) * row_major_quantized_weight.size(-1);

  torch::Tensor processed_tensor = torch::zeros_like(row_major_quantized_weight);
  fast_preprocess_weights_for_mixed_gemm(reinterpret_cast<int8_t*>(processed_tensor.data_ptr()),
                                         reinterpret_cast<int8_t*>(row_major_quantized_weight.data_ptr()),
                                         reinterpret_cast<const int32_t*>(row_permutation_tensor.data_ptr()),
                                         row_permutation.size(), {num_experts, num_rows, num_cols}, quant_type, false,
                                         context_->GetMemoryManageStreams()[rank_].Get());
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
  return processed_tensor;
}

#endif

}  // namespace ksana_llm