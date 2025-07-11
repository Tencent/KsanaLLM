/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_mla_attention_layer.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashMlaAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Init(const std::vector<std::any>& parameters,
                                                                 std::shared_ptr<Context> context, int rank) {
  return AttentionLayer<SCALAR_T>::Init(parameters, context, rank);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashMlaAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Forward(const std::vector<Tensor>& input_tensors,
                                                                    std::vector<Tensor>& output_tensors) {
  KLLM_THROW("FlashMlaAttentionLayer not implement in Ascend.");
  return Status();
}

using llm_kernels::utils::KVCacheType;
template class FlashMlaAttentionLayer<float, float, KVCacheType::kAuto>;
template class FlashMlaAttentionLayer<float, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashMlaAttentionLayer<float, uint8_t, KVCacheType::kFp8E5M2>;
template class FlashMlaAttentionLayer<float16, float16, KVCacheType::kAuto>;
template class FlashMlaAttentionLayer<float16, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashMlaAttentionLayer<float16, uint8_t, KVCacheType::kFp8E5M2>;
template class FlashMlaAttentionLayer<bfloat16, bfloat16, KVCacheType::kAuto>;
#if defined(ENABLE_FP8)
template class FlashMlaAttentionLayer<bfloat16, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashMlaAttentionLayer<bfloat16, uint8_t, KVCacheType::kFp8E5M2>;
#endif

}  // namespace ksana_llm
