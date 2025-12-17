/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/paged_sparse_mla_indexer.h"
#include "ksana_llm/layers/paged_sparse_mla_indexer_layer.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

PagedSparseMlaIndexer::PagedSparseMlaIndexer(const size_t layer_idx, const LayerCreationContext& creation_context,
                                             const AttentionCreationConfig& attn_config, int block_size) {
  const uint32_t index_n_heads = attn_config.model_config.dsa_config.index_n_heads;
  const uint32_t index_head_dim = attn_config.model_config.dsa_config.index_head_dim;
  const uint32_t qk_rope_head_dim = attn_config.model_config.mla_config.qk_rope_head_dim;
  const uint32_t index_topk = attn_config.model_config.dsa_config.index_topk;

  paged_sparse_mla_indexer_layer_ = std::make_shared<PagedSparseMlaIndexerLayer>();

  // NOTE: acsends's image g++ is 9.4.0, it do not support convert
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
  indexer_param.push_back(static_cast<int>(attn_config.max_batch_size));         // 12: max_batch_size int
  indexer_param.push_back(attn_config.max_position_embeddings);                  // 13: max_seq_len
  indexer_param.push_back(static_cast<int>(layer_idx));                          // 14: layer_idx
  indexer_param.push_back(static_cast<int>(attn_config.layer_num_on_node));      // 15: layer_num_on_node

  paged_sparse_mla_indexer_layer_->Init(indexer_param, creation_context.runtime_config, creation_context.context,
                                        creation_context.rank);
  paged_sparse_mla_indexer_layer_->SetWorkspaceBuffer(
      creation_context.workspace_mgr->GetWorkspace(paged_sparse_mla_indexer_layer_->GetWorkspaceSize()));
}

Status PagedSparseMlaIndexer::Forward(const std::shared_ptr<ModelInput>& model_input,
                                      const ModelInput::input_info& page_input, const AttentionForwardContext& attn_ctx,
                                      Tensor& q_indexer_tensor, Tensor& k_indexer_tensor, Tensor& weights_tensor,
                                      Tensor& quant_workspace_tensor, std::vector<Tensor>& output_tensors) {
  return paged_sparse_mla_indexer_layer_->Forward(
      {q_indexer_tensor, k_indexer_tensor, weights_tensor, quant_workspace_tensor, page_input.rotary_embedding_pos,
       page_input.rotary_embedding_mask, page_input.input_length, page_input.indexer_kv_list,
       page_input.kv_cache_offset, page_input.block_table, page_input.cur_seq_len_start, page_input.cur_seq_len_end,
       model_input->layer_indexer_kv_cache_ptr, page_input.paged_schedule_meta},
      output_tensors);
}

}  // namespace ksana_llm
