/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/attention/sparse_mla_indexer.h"

#include "ksana_llm/modules/basic/flash_sparse_mla_indexer.h"
#include "ksana_llm/modules/basic/paged_sparse_mla_indexer.h"
#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

SparseMlaIndexer::SparseMlaIndexer(int layer_idx, LayerCreationContext& creation_context,
                                   ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx) {
  auto& attn_config = model_creation_config.attn_config;
  auto& model_config = attn_config.model_config;
  auto& runtime_config = creation_context.runtime_config;

  // Initialize dimensions from model config
  n_heads_ = model_config.dsa_config.index_n_heads;
  head_dim_ = model_config.dsa_config.index_head_dim;
  index_topk_ = model_config.dsa_config.index_topk;

  // block_size指的是一个block 里面有多少个token，区别cache block size：表示一个block的字节个数
  block_size_ = runtime_config.attn_backend_config.block_token_num;

  const std::string layer_prefix = fmt::format("model.layers.{}.self_attn", layer_idx);
  const auto& linear_compute_backend = model_config.quant_config.backend;

  // Initialize linear layers
  wq_b_ = std::make_shared<Linear>(layer_prefix + ".indexer.wq_b.weight", creation_context, linear_compute_backend);

  wk_ = std::make_shared<Linear>(layer_prefix + ".indexer.wk.weight", creation_context, linear_compute_backend);

  weights_proj_ =
      std::make_shared<Linear>(layer_prefix + ".indexer.weights_proj.weight", creation_context, linear_compute_backend);

  // Initialize layer norm
  k_norm_ = std::make_shared<Layernorm>(layer_prefix + ".indexer.k_norm.weight",
                                        model_creation_config.layernorm_config.layernorm_eps, creation_context,
                                        layer_prefix + ".indexer.k_norm.bias");

  // Calculate local layer index for Pipeline Parallelism
  // In PP mode, KV cache is organized by local layer index (0 to local_num_layers-1)
  // But weights are accessed by global layer index
  const int local_layer_idx = layer_idx - creation_context.pipeline_config.lower_layer_idx;
  // Initialize flash and paged indexer layers with local layer index
  flash_sparse_mla_indexer_ =
      std::make_shared<FlashSparseMlaIndexer>(local_layer_idx, creation_context, attn_config, block_size_);
  paged_sparse_mla_indexer_ =
      std::make_shared<PagedSparseMlaIndexer>(local_layer_idx, creation_context, attn_config, block_size_);

  KLLM_LOG_DEBUG << fmt::format(
      "SparseMlaIndexer layer {} (local_idx={}) initialized: n_heads={}, "
      "head_dim={}, index_topk={}, block_size={}, lower_layer_idx={}",
      layer_idx_, local_layer_idx, n_heads_, head_dim_, index_topk_, block_size_,
      creation_context.pipeline_config.lower_layer_idx);
}

Status SparseMlaIndexer::Forward(const Tensor& x, const Tensor& qr, Tensor& workspace, Tensor& topk_indices,
                                 ForwardingContext& forwarding_context) {
  const int rank = forwarding_context.GetCurrentRank();

  PROFILE_EVENT_SCOPE(mla_forward, "sparse_mla_index_forward", rank);
  // x.shape = [total_tokens, hidden_dim]
  const size_t total_tokens = x.shape[0];
  const size_t hidden_dim = x.shape[1];

  KLLM_LOG_DEBUG << fmt::format("SparseMlaIndexer Forward: layer={}, total_tokens={}, hidden_dim={}", layer_idx_,
                                total_tokens, hidden_dim);

  // Step 1: Key projection k = wk(x)
  // Input: x [total_tokens, dim]
  // Output: k [total_tokens, head_dim]
  std::vector<Tensor> k_indexer_tensors{workspace.GetView({total_tokens, head_dim_})};
  {
    PROFILE_EVENT_SCOPE(wk_forward, "indexer_wk", rank);
    STATUS_CHECK_RETURN(wk_->Forward(x, k_indexer_tensors));
  }
  // Step 2: Key normalization
  {
    PROFILE_EVENT_SCOPE(k_norm_forward, "indexer_k_norm", rank);
    STATUS_CHECK_RETURN(k_norm_->Forward(k_indexer_tensors, k_indexer_tensors));
  }

  // Step 3: Compute weights = weights_proj(x)
  // Input: x [total_tokens, dim]
  // Output: weight [total_tokens, n_local_heads]
  std::vector<Tensor> weights_tensors{workspace.GetView({total_tokens, n_heads_}, total_tokens * head_dim_)};
  {
    PROFILE_EVENT_SCOPE(weights_proj_forward, "indexer_weights_proj", rank);
    STATUS_CHECK_RETURN(weights_proj_->Forward(x, weights_tensors));
  }

  // Step 4: Query projection q = wq_b(qr)
  // Input: qr [total_tokens, q_lora_rank]
  // Output: q [total_tokens, n_local_heads * head_dim]
  std::vector<Tensor> q_indexer_tensors{x.GetView({total_tokens, n_heads_, head_dim_})};
  {
    PROFILE_EVENT_SCOPE(wq_b_forward, "indexer_wq_b", rank);
    STATUS_CHECK_RETURN(wq_b_->Forward(qr, q_indexer_tensors));
  }

  // Step 5: Process flash and paged attention separately
  Tensor quant_workspace = workspace.GetView({}, total_tokens * (n_heads_ + head_dim_));
  auto model_input = forwarding_context.GetModelInput();
  auto& attn_ctx = forwarding_context.GetAttentionForwardContext();

  // Calculate token counts
  const size_t dp_context_tokens = model_input->dp_context_tokens;
  const size_t dp_decode_tokens = model_input->dp_decode_tokens;
  if (dp_context_tokens > 0) {
    // Flash attention for prefill tokens (first dp_context_tokens)
    Tensor context_q_indexer = q_indexer_tensors[0].GetView({dp_context_tokens, n_heads_, head_dim_});
    Tensor context_k_indexer = k_indexer_tensors[0].GetView({dp_context_tokens, head_dim_});
    Tensor context_weights = weights_tensors[0].GetView({dp_context_tokens, n_heads_});

    std::vector<Tensor> context_output_tensors = {topk_indices.GetView({dp_context_tokens, index_topk_})};
    STATUS_CHECK_RETURN(flash_sparse_mla_indexer_->Forward(model_input, attn_ctx, context_q_indexer, context_k_indexer,
                                                           context_weights, quant_workspace, context_output_tensors));
  }

  if (dp_decode_tokens > 0) {
    // Paged attention for decode tokens (after dp_context_tokens)
    // Process page_dual and page_single sequentially
    // Page_dual is placed before page_single, since requests are sorted by token_num in descending order
    size_t skip_tokens = 0;
    for (const auto& page_input : model_input->page_inputs) {
      const size_t current_tokens = page_input.total_dp_input_ids_len;
      // Offset tensors by `dp_context_tokens + skip_tokens`
      const size_t total_offset = dp_context_tokens + skip_tokens;

      Tensor current_q_indexer =
          q_indexer_tensors[0].GetView({current_tokens, n_heads_, head_dim_}, total_offset * n_heads_ * head_dim_);
      Tensor current_k_indexer = k_indexer_tensors[0].GetView({current_tokens, head_dim_}, total_offset * head_dim_);
      Tensor current_weights = weights_tensors[0].GetView({current_tokens, n_heads_}, total_offset * n_heads_);

      // Offset output tensor for current page
      std::vector<Tensor> current_output_tensors = {
          topk_indices.GetView({current_tokens, index_topk_}, total_offset * index_topk_)};

      STATUS_CHECK_RETURN(paged_sparse_mla_indexer_->Forward(model_input, page_input, attn_ctx, current_q_indexer,
                                                             current_k_indexer, current_weights, quant_workspace,
                                                             current_output_tensors));
      skip_tokens += current_tokens;
    }
  }

  // Correctly set the output shape
  topk_indices.shape = {total_tokens, index_topk_};

  KLLM_LOG_DEBUG << fmt::format("SparseMlaIndexer Forward completed: layer={}", layer_idx_);

  return Status();
}

}  // namespace ksana_llm
