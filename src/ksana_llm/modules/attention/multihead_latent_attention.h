/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_model.h"
#include "ksana_llm/models/common/common_weight.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/utils.h"

#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/basic/bmm.h"
#include "ksana_llm/modules/basic/flash_mla_attention.h"
#include "ksana_llm/modules/basic/linear.h"
#include "ksana_llm/modules/basic/paged_mla_attention.h"

namespace ksana_llm {

// Buffers used in mla.
// TODO(robertyuan): Some maybe reused with other modules
struct MlaBuffers {
  TensorBuffer* q_buffer;
  TensorBuffer* q_rope_buffer;
  TensorBuffer* kv_buffer;
  TensorBuffer* k_rope_buffer;

  // The tensor buffer used for flash attn, used to store qkv data with prefix part.
  TensorBuffer* prefix_o_buffer;
  TensorBuffer* prefix_k_up_buffer;
  TensorBuffer* prefix_v_up_buffer;

  // shared
  TensorBuffer* shared_prefix_k_v_kv_buffer;
  size_t prefix_k_buffer_size;
  size_t prefix_v_buffer_size;
};

template <typename T>
class MultiHeadLatentAttention {
 public:
  MultiHeadLatentAttention(int layer_idx, bool is_neox, LayerCreationContext<T>& creation_context,
                           ModelCreationConfig& model_creation_config, MlaBuffers& mla_buffers);

  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                 std::vector<Tensor>& paged_buffer_tensors, ForwardingContext<T>& forwarding_context);

  // do forward.
  Status DataParallelForward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                             std::vector<Tensor>& extra_buffer_tensors, ForwardingContext<T>& forwarding_context);

  static Status CreateBuffers(BufferManager* buffer_mgr, const AttentionCreationConfig& attn_config,
                              MlaBuffers& mla_buffers);

  // Used for 2 stage dp forward.
  Status ContextForward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& hidden_buffer_tensors_1,
                        std::vector<Tensor>& reduce_buffer_tensors, std::vector<Tensor>& prefill_buffer_tensors,
                        ForwardingContext<T>& forwarding_context);

  Status DecodeForward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& hidden_buffer_tensors_1,
                       std::vector<Tensor>& reduce_buffer_tensors, std::vector<Tensor>& paged_buffer_tensors,
                       ForwardingContext<T>& forwarding_context);

 private:
  Status FlashAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                               std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& reduce_buffer_tensors,
                               std::vector<Tensor>& prefill_buffer_tensors, Tensor& prefill_q_buffer_tensor,
                               Tensor& q_rope_buffer_tensor, Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor,
                               ForwardingContext<T>& forwarding_context);

  Status PagedAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                               std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& reduce_buffer_tensors,
                               std::vector<Tensor>& paged_buffer_tensors, Tensor& prefill_q_buffer_tensor,
                               Tensor& q_rope_buffer_tensor, Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor,
                               ForwardingContext<T>& forwarding_context);

 private:
  const int layer_idx_;
  MlaBuffers& mla_buffers_;

 protected:
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  std::shared_ptr<BaseLayer> set_torch_stream_layers_;
#endif

  bool use_q_lora_ = false;

  std::shared_ptr<Linear<T>> attn_q_a_projs_;
  std::shared_ptr<Linear<T>> attn_kv_a_lora_projs_;
  std::shared_ptr<Linear<T>> attn_kv_a_ropes_;
  std::shared_ptr<Linear<T>> attn_q_b_lora_projs_;
  std::shared_ptr<Linear<T>> attn_q_b_rope_projs_;
  std::shared_ptr<Linear<T>> attn_kv_b_nope_projs_;
  std::shared_ptr<Linear<T>> attn_v_head_projs_;
  std::shared_ptr<Linear<T>> attn_w_q_uks_;
  std::shared_ptr<Bmm<T>> attn_w_uk_t_bmm_;
  std::shared_ptr<FlashMlaAttention<T>> flash_mla_attention_layers_;
  std::shared_ptr<PagedMlaAttention<T>> paged_mla_attention_layers_;
  inline static uint32_t qk_nope_head_dim_ = 0;
  inline static uint32_t kv_lora_rank_ = 0;
  inline static int head_num_per_tp_ = 0;

  std::shared_ptr<Layernorm<T>> kv_a_layernorms_;
  std::shared_ptr<Layernorm<T>> q_a_layernorms_;

  AbsorbWeightsType absorb_type_ = AbsorbWeightsType::kAbsorbDisabled;
};

}  // namespace ksana_llm
