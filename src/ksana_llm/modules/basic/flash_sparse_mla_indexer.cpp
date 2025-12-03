/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/flash_sparse_mla_indexer.h"
#include "ksana_llm/layers/flash_sparse_mla_indexer_layer.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

FlashSparseMlaIndexer::FlashSparseMlaIndexer(const size_t layer_idx, const LayerCreationContext& creation_context,
                                             const AttentionCreationConfig& attn_config, int block_size)
    : context_(creation_context.context), rank_(creation_context.rank) {
  uint32_t index_n_heads = attn_config.model_config.dsa_config.index_n_heads;
  uint32_t index_head_dim = attn_config.model_config.dsa_config.index_head_dim;
  uint32_t qk_rope_head_dim = attn_config.model_config.mla_config.qk_rope_head_dim;
  uint32_t index_topk = attn_config.model_config.dsa_config.index_topk;

  flash_sparse_mla_indexer_layer_ = std::make_shared<FlashSparseMlaIndexerLayer>();

  // NOTE(karlluo): acsends's image g++ is 9.4.0, it do not support convert
  // from '<brace-enclosed initializer list>' to 'std::vector<std::any>' so
  // we use push back to make it work.
  std::vector<std::any> indexer_param;
  indexer_param.push_back(static_cast<int>(attn_config.model_config.hidden_units));  // 0: dim
  indexer_param.push_back(static_cast<int>(index_n_heads));                          // 1: n_heads
  indexer_param.push_back(static_cast<int>(index_head_dim));                         // 2: head_dim
  indexer_param.push_back(static_cast<int>(qk_rope_head_dim));                       // 3: rope_head_dim
  indexer_param.push_back(static_cast<int>(index_topk));                             // 4: index_topk
  // Pass block_size from SparseMlaIndexer constructor
  indexer_param.push_back(block_size);                     // 5: block_size (kv block size)
  indexer_param.push_back(128);                            // 7: quant_block_size (quantization block size)
  indexer_param.push_back(attn_config.position_encoding);  // 8: rotary_embedding type
  indexer_param.push_back(attn_config.rope_theta);         // 9: rope_theta
  indexer_param.push_back(attn_config.cos_sin_cache_ptr);  // 10: cos/sin cache
  indexer_param.push_back(attn_config.model_config.rope_scaling_factor_config);  // 11: rope scaling
  indexer_param.push_back(static_cast<int>(attn_config.max_batch_size));         // 12: max_batch_size
  indexer_param.push_back(attn_config.max_position_embeddings);                  // 13: max_seq_len
  indexer_param.push_back(static_cast<int>(layer_idx));                          // 14: layer_idx
  flash_sparse_mla_indexer_layer_->Init(indexer_param, creation_context.runtime_config, context_, rank_);
  flash_sparse_mla_indexer_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(flash_sparse_mla_indexer_layer_->GetWorkspaceSize()));
}

Status FlashSparseMlaIndexer::Forward(const std::shared_ptr<ModelInput>& model_input,
                                      const AttentionForwardContext& attn_ctx,
                                      Tensor& q_indexer_tensor, Tensor& k_indexer_tensor, Tensor& weights_tensor,
                                      std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(flash_sparse_mla_indexer_layer_->Forward({q_indexer_tensor,
                                                                k_indexer_tensor,
                                                                weights_tensor,
                                                                model_input->flash_input.rotary_embedding_pos,
                                                                model_input->flash_input.rotary_embedding_mask,
                                                                model_input->flash_input.indexer_kv_list,
                                                                model_input->flash_input.kv_cache_offset,
                                                                model_input->dp_prefill_q_offset_uint64_tensor,
                                                                model_input->dp_input_prefix_uint64_tensor,
                                                                model_input->dp_input_offset_uint64_tensor,
                                                                model_input->flash_input.block_table,
                                                                model_input->flash_input.cur_seq_len_start,
                                                                model_input->flash_input.cur_seq_len_end,
                                                                model_input->layer_indexer_kv_cache_ptr,
                                                                attn_ctx.forward_shape},
                                                               output_tensors));
  return Status();
}

}  // namespace ksana_llm
