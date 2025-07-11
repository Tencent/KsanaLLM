/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_attention_layer.h"
#include <cstdlib>

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Init(const std::vector<std::any>& parameters,
                                                              std::shared_ptr<Context> context, int rank) {
  AttentionLayer<SCALAR_T>::Init(parameters, context, rank);
  return Status(RET_UNDEFINED_REFERENCE, "FlashAttentionLayer not supported.");
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Forward(const std::vector<Tensor>& input_tensors,
                                                                 std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "FlashAttentionLayer not supported.");
}

using llm_kernels::utils::KVCacheType;
template class FlashAttentionLayer<float, float, KVCacheType::kAuto>;
template class FlashAttentionLayer<float, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashAttentionLayer<float, uint8_t, KVCacheType::kFp8E5M2>;
template class FlashAttentionLayer<float16, float16, KVCacheType::kAuto>;
template class FlashAttentionLayer<float16, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashAttentionLayer<float16, uint8_t, KVCacheType::kFp8E5M2>;
template class FlashAttentionLayer<bfloat16, bfloat16, KVCacheType::kAuto>;
#if defined(ENABLE_FP8)
template class FlashAttentionLayer<bfloat16, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashAttentionLayer<bfloat16, uint8_t, KVCacheType::kFp8E5M2>;
#endif

}  // namespace ksana_llm
