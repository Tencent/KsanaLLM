/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unordered_map>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

template <typename T>
struct InvokeInputIdsEmbeddingLookupPosEncodingParam {
  // Batch number of ptrs, each ptr is the ptr of the specific p/prompt tuning weights for this sequence
  const T** p_prompt_tuning_batch_weights = nullptr;
  // The start id of p_prompt_tuning token ids (based on the tokenizer)
  // PROMPT_0 --> p_prompt_tuning_id_start; PROMPT_1 --> p_prompt_tuning_id_start + 1; ...
  const int32_t p_prompt_tuning_id_start = 0;
  // Request prompt embeddding's max length
  const int32_t request_prompt_max_length = 0;
  // Whether or not use the request prompt embeddings
  const bool use_request_p_prompt_embedding = false;
  // Request prompt embeddings
  const T* request_prompt_embedding = nullptr;
};

template <typename T, bool DO_POSITION_ENCODING, bool USE_EMB_SCALE>
void LookupFusedEmbeddingWithCSRInputs(T* output_hidden_units, const T* embedding_table, const T* pos_table,
                                        const T emb_scale,
                                        InvokeInputIdsEmbeddingLookupPosEncodingParam<T> prompt_param,
                                        const int32_t* input_ids, const size_t* steps, const size_t* ids_offsets,
                                        const size_t* prefix_offsets, const int32_t batch_size,
                                       const uint32_t hidden_units, const size_t vocab_size, const size_t vocab_id,
                                       cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
