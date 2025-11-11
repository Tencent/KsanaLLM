/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/models/base/forwarding_context.h"
#include "ksana_llm/models/base/layer_creation_context.h"
#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/basic/layernorm.h"
#include "ksana_llm/modules/basic/linear.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// Forward declarations
class FlashSparseMlaIndexer;
class PagedSparseMlaIndexer;

// Buffers used in sparse mla indexer.
struct IndexerBuffers {
  TensorBuffer* q_indexer_buffer;     // for wq_b output
  TensorBuffer* k_indexer_buffer;     // for wk output
  TensorBuffer* weights_buffer;       // for weights_proj output
  TensorBuffer* topk_indices_buffer;  // for topk indices
};

// Sparse MLA Indexer module
// Implements the indexer logic for sparse multi-head latent attention
class SparseMlaIndexer {
 public:
  SparseMlaIndexer(int layer_idx, LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config,
                   IndexerBuffers& indexer_buffers);

  ~SparseMlaIndexer() = default;

  // Forward pass
  // Inputs:
  //   - x: hidden states [batch, seq_len, hidden_dim]
  //   - qr: query lora output [batch, seq_len, q_lora_rank]
  // Outputs:
  //   - topk_indices: top-k indices [batch, seq_len, n_heads, index_topk]
  Status Forward(const Tensor& x, const Tensor& qr, Tensor& topk_indices, ForwardingContext& forwarding_context);

  // Create buffers for indexer
  static Status CreateBuffers(BufferManager* buffer_mgr, const AttentionCreationConfig& attn_config,
                              const RuntimeConfig& runtime_config, IndexerBuffers& indexer_buffers);

 private:
  const int layer_idx_;
  const int tensor_parallel_size_;
  IndexerBuffers& indexer_buffers_;

  // Model parameters (actively used)
  int n_heads_;     // number of index heads
  int head_dim_;    // index head dimension
  int index_topk_;  // top-k value
  int block_size_;  // block size for quantization

  // Layers
  std::shared_ptr<Linear> wq_b_;          // Query projection (q_lora_rank -> n_heads * head_dim)
  std::shared_ptr<Linear> wk_;            // Key projection (dim -> head_dim)
  std::shared_ptr<Layernorm> k_norm_;     // Key normalization
  std::shared_ptr<Linear> weights_proj_;  // Weights projection (dim -> n_heads)

  // Flash and paged indexer layers
  std::shared_ptr<FlashSparseMlaIndexer> flash_sparse_mla_indexer_;
  std::shared_ptr<PagedSparseMlaIndexer> paged_sparse_mla_indexer_;

  // Context and rank for layer operations
  std::shared_ptr<Context> context_;
  int rank_;
};

}  // namespace ksana_llm
