/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_attention_layer.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Init(const std::vector<std::any>& parameters,
                                                              const RuntimeConfig& runtime_config,
                                                              std::shared_ptr<Context> context, int rank) {
  return Status(RET_UNDEFINED_REFERENCE, "PagedAttentionLayer not supported.");
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Forward(const std::vector<Tensor>& input_tensors,
                                                                 std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "PagedAttentionLayer not supported.");
}

using llm_kernels::utils::KVCacheType;
template class PagedAttentionLayer<float, float, KVCacheType::kAuto>;
template class PagedAttentionLayer<float, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedAttentionLayer<float, uint8_t, KVCacheType::kFp8E5M2>;
template class PagedAttentionLayer<float16, float16, KVCacheType::kAuto>;
template class PagedAttentionLayer<float16, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedAttentionLayer<float16, uint8_t, KVCacheType::kFp8E5M2>;
template class PagedAttentionLayer<bfloat16, bfloat16, KVCacheType::kAuto>;
#if defined(ENABLE_FP8)
template class PagedAttentionLayer<bfloat16, uint8_t, KVCacheType::kFp8E4M3>;
template class PagedAttentionLayer<bfloat16, uint8_t, KVCacheType::kFp8E5M2>;
#endif

}  // namespace ksana_llm
