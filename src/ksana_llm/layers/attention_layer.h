/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include <optional>

#  include "csrc/kernels/nvidia/alibi/alibi.h"
#  include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#endif

#ifdef ENABLE_ACL
#  include "csrc/kernels/ascend/attention/attention.h"
#endif

#include "csrc/utils/quant_type.h"
#include "ksana_llm/layers/base_layer.h"

namespace ksana_llm {

// The positional encoding.
enum PositionEncoding { LEARNED_ABSOLUTE = 0, ROPE = 1, ALIBI = 2, NO_ROPE = 3 };

template <typename T, template <typename, typename, llm_kernels::utils::KVCacheType> class ATTENTION_LAYER>
std::shared_ptr<BaseLayer> CreateAttentionLayer(DataType kv_cache_dtype) {
  switch (kv_cache_dtype) {
#ifdef ENABLE_CUDA
    case TYPE_FP8_E5M2:
      return std::make_shared<ATTENTION_LAYER<T, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2>>();
    case TYPE_FP8_E4M3:
      return std::make_shared<ATTENTION_LAYER<T, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3>>();
#endif
    default:
      return std::make_shared<ATTENTION_LAYER<T, T, llm_kernels::utils::KVCacheType::kAuto>>();
  }
}

template <typename T>
class AttentionLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) = 0;

 protected:
  int layer_num_;
  int layer_index_;
  int block_size_;
  int block_token_num_;
  int num_heads_;
  int num_kv_heads_;
  int head_size_;
  int stride_size_;
  size_t tensor_para_size_;
  int max_position_embeddings_;
  float base_;
  bool use_qk_norm_;                           // Check if normlize the attention out q and k.
  float layernorm_eps_;                        // The epsilon value used in layer normalization layers.
  bool enable_qk_pre_norm_before_rotary_pos_;  // Whether to normalize q and k before rotary position
  // kv_cache storage type and kv scale
  DataType kv_cache_dtype_;
  float k_scale_;
  float v_scale_;

  bool is_causal_{true};
  // mla config params
  uint32_t q_lora_rank_;
  uint32_t kv_lora_rank_;
  uint32_t qk_nope_head_dim_;
  uint32_t qk_rope_head_dim_;
  uint32_t v_head_dim_;
  float attn_scale_;
  // TODO(winminkong): the matmul op will be removed from mla attn in the subsequent steps.
  QuantMode mm_quant_mode_;

#ifdef ENABLE_CUDA
  // for deepseek yarn
  float deepseek_yarn_get_mscale(const float scale, const float mscale);

  float common_yarn_get_mscale(const float scale);

  std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<T>> rotary_embedding_cuda_;
  std::optional<void*> alibi_slopes_;
#endif

#ifdef ENABLE_ACL
  size_t max_batch_size_;
  bool is_multi_token_forward_;
#endif

  bool no_rope_;
  size_t attn_temperature_tuning_;
  size_t floor_scale_;
};

}  // namespace ksana_llm
