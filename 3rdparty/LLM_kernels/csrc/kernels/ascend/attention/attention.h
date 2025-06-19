/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <stdint.h>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"

#include "csrc/utils/ascend/atb_executor.h"

namespace llm_kernels {
namespace ascend {

enum RotaryEmbeddingType { DEFAULT, LINEAR_SCALING, DYNAMIC_NTK_SCALING };

template <typename DTYPE>
class ATBAttention {
 public:
  ~ATBAttention();

  // Invoke paged attention.
  void Forward(void* output, void* qkv_tensor, void* pos_ids, void* slot_mapping, void* k_cache, void* v_cache,
               void* block_tables, const uint32_t max_num_blocks_per_query, const uint32_t batch_size,
               const uint32_t total_token_num, const uint32_t total_block_num, const uint32_t block_token_num,
               const uint32_t layer_index, void* seq_len, const bool is_multi_token_forward, atb::Context* atb_context,
               void (*ws_func)(size_t, void**));

  // Initialize some necessary information.
  void Initialize(uint32_t max_batch_size, uint32_t head_size, uint32_t kv_head_size, uint32_t head_dim,
                  uint32_t layer_num, uint32_t layer_idx, uint32_t block_token_num, aclrtStream& stream, const int rank,
                  const bool is_multi_token_forward, const size_t max_position_embeddings, const float rope_base,
                  const RotaryEmbeddingType scaling_type = RotaryEmbeddingType::DEFAULT,
                  const float scaling_factor = 1.0f);

  bool IsMultiTokenForwardOp() { return is_multi_token_forward_; }

 private:
  bool is_multi_token_forward_{true};
  llm_kernels::utils::ATBOperationExecutor atb_op_executor_;

  size_t max_position_embeddings_;
  size_t head_size_;
  size_t kv_head_size_;
  size_t head_dim_;
  size_t layer_num_;
  size_t block_token_num_;
  uint32_t rank_;
  uint32_t max_batch_size_;
  std::vector<int32_t> batch_status_;

  void* rope_cos_workspace_ptr_{nullptr};
  void* rope_sin_workspace_ptr_{nullptr};
  void* attn_mask_ptr_{nullptr};

 private:
  void InitRopeCosSinWorkspace(const size_t max_position_embeddings, const float rope_base, const uint32_t head_dim,
                               const float scaling_factor, const RotaryEmbeddingType scaling_type, aclrtStream& stream);

  void InitAttnMask();

  // Create Split QKV subgraph
  void CreateSplitQKVOperation(uint32_t head_size, uint32_t kv_head_size, uint32_t head_dim,
                               atb::Operation** operation);
};

}  // namespace ascend
}  // namespace llm_kernels
