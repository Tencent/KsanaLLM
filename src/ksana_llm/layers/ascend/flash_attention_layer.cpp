/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_attention_layer.h"
#include <cstdlib>
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/utils/ascend/common.h"

namespace ksana_llm {

Status FlashAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                 std::shared_ptr<Context> context, int rank) {
  AttentionLayer::Init(parameters, runtime_config, context, rank);
  if (atb_flash_attn_ == nullptr) {
    atb_flash_attn_ = std::make_shared<llm_kernels::ascend::ATBAttention>();
    atb_flash_attn_->Initialize(static_cast<aclDataType>(this->inter_data_type_),
                                static_cast<uint32_t>(this->max_batch_size_), this->num_heads_, this->num_kv_heads_,
                                this->head_size_, this->layer_num_, this->layer_index_, this->block_token_num_,
                                context->GetComputeStreams()[rank].Get(), rank,
                                /*is_multi_token_forward*/ true, this->max_position_embeddings_, this->base_);

    KLLM_LOG_DEBUG << "FlashAttentionLayer Init, layer_num:" << this->layer_num_;
  }
  return Status();
}

Status FlashAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // for ATB input_tensors:
  //   0: qkv_tensor shape [max_token_num, hidden_units * 3], type same as weight
  //   1: rotary_embedding_pos shape [max_token_num], type int64_t
  //   2: slot_mapping shape [max_token_num], type int32_t
  //   3: k_cache shape [max_block_num, block_token_num, head_size, head_dim], type same as weight
  //   4: v_cache shape [max_block_num, block_token_num, head_size, head_dim], type same as weight
  //   5: seq_len_host shape [batch_size]
  //   6: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back()]
  //   7: atb_attention_attr shape: [2], content: 0: layers_slot_mapping_dim_1; 1: max_num_blocks_per_query
  // output_tensors:
  //   0: flash_attention_output shape: [std::max(max_batch_size * vocab_size, max_token_num * hidden_units * 3)]
  void* output = output_tensors[0].GetPtr<void>();
  output_tensors[0].shape[0] = input_tensors[0].shape[0];
  output_tensors[0].shape[1] = this->num_heads_ * this->head_size_;
  output_tensors[0].dtype = input_tensors[0].dtype;
  void* qkv_tensor = input_tensors[0].GetPtr<void>();
  int32_t total_token_num = input_tensors[0].shape[0];
  reinterpret_cast<atb::Context*>(GetRuntimeContext(this->rank_))
      ->SetExecuteStream(this->context_->GetComputeStreams()[this->rank_].Get());
  void* k_cache = input_tensors[3].GetPtr<void>();
  void* v_cache = input_tensors[4].GetPtr<void>();
  int32_t* seq_len_host_ptr = input_tensors[5].GetPtr<int32_t>();
  uint64_t* atb_attention_attr_ptr = input_tensors[7].GetPtr<uint64_t>();
  uint64_t slot_mapping_dim_1 = atb_attention_attr_ptr[0];
  int64_t batch_size = input_tensors[6].shape[0];
  int32_t total_block_num = input_tensors[3].shape[0];
  int32_t* slot_mapping = input_tensors[2].GetPtr<int32_t>() + this->layer_index_ * slot_mapping_dim_1;
  void* rotary_embedding_pos = input_tensors[1].GetPtr<void>();

  atb_flash_attn_->Forward(output, qkv_tensor, rotary_embedding_pos, reinterpret_cast<void*>(slot_mapping), k_cache,
                           v_cache,
                           /*block_tables*/ nullptr,
                           /*max_num_blocks_per_query*/ 0, static_cast<uint32_t>(batch_size),
                           static_cast<uint32_t>(total_token_num), static_cast<uint32_t>(total_block_num),
                           this->block_token_num_, static_cast<uint32_t>(this->layer_index_), seq_len_host_ptr,
                           /*is_multi_token_forward*/ true,
                           reinterpret_cast<atb::Context*>(GetRuntimeContext(this->rank_)), GetWorkSpaceFunc());
  return Status();
}

}  // namespace ksana_llm
