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
#include "ksana_llm/modules/basic/mem_adjuster.h"
#include "ksana_llm/modules/basic/paged_mla_attention.h"
#include "ksana_llm/modules/basic/split.h"

namespace ksana_llm {

// Buffers used in mla.
// TODO(robertyuan): Some maybe reused with other modules
struct MlaBuffers {
  TensorBuffer* q_buffer;
  TensorBuffer* kv_lora_or_q_rope_buffer;
  TensorBuffer* kv_buffer;
  TensorBuffer* k_rope_buffer;
  TensorBuffer* mem_adjuster_buffer;

  // shared
  TensorBuffer* shared_prefix_kv_buffer;
};

class MultiHeadLatentAttention {
 public:
  MultiHeadLatentAttention(int layer_idx, bool is_neox, LayerCreationContext& creation_context,
                           ModelCreationConfig& model_creation_config, MlaBuffers& mla_buffers);

  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                 ForwardingContext& forwarding_context);

  // do forward.
  Status DataParallelForward(std::vector<Tensor>& dp_input_tensors, std::vector<Tensor>& hidden_buffer_tensors_0,
                             std::vector<Tensor>& reduce_buffer_tensors, ForwardingContext& forwarding_context);

  static Status CreateBuffers(BufferManager* buffer_mgr, const AttentionCreationConfig& attn_config,
                              const RuntimeConfig& runtime_config, MlaBuffers& mla_buffers);

  Status AcquireBuffers(ForwardingContext& forwarding_context);
  Status ReleaseBuffers();

  // Used for 2 stage dp forward.
  Status ContextForward(std::vector<Tensor>& input_tensors, std::vector<Tensor>& hidden_buffer_tensors_0,
                        std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& reduce_buffer_tensors,
                        ForwardingContext& forwarding_context);

  Status DecodeForward(std::vector<Tensor>& input_tensors, std::vector<Tensor>& hidden_buffer_tensors_0,
                       std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& reduce_buffer_tensors,
                       ForwardingContext& forwarding_context);

 private:
  Status FlashAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& workspace_buffer,
                               std::vector<Tensor>& output_tensors, Tensor& q_nope_tensor, Tensor& q_rope_buffer_tensor,
                               Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor,
                               ForwardingContext& forwarding_context);

  Status PagedAttentionForward(std::vector<Tensor>& output_tensor, std::vector<Tensor>& hidden_buffer_tensors_1,
                               std::vector<Tensor>& paged_buffer_tensors, Tensor& prefill_q_buffer_tensor,
                               Tensor& q_rope_buffer_tensor, Tensor& kv_buffer_tensor, Tensor& k_rope_buffer_tensor,
                               ForwardingContext& forwarding_context);

 private:
  int rank_;
  const int layer_idx_;
  const int tensor_parallel_size_;
  MlaBuffers& mla_buffers_;

 protected:
#ifdef ENABLE_CUDA
  std::shared_ptr<BaseLayer> set_torch_stream_layers_;
#endif

  bool use_q_lora_ = false;
  // TODO(huicongyao, jinxcwu): suppport INT4 model to keep use_fused_lora_a_ always true
  bool use_fused_lora_a_ = false;
  // compute o_proj out of dp group
  bool o_proj_out_of_dp_ = false;
  bool use_q_b_nope_rope_ = false;

  std::shared_ptr<Linear> attn_fused_lora_a_projs_;
  std::shared_ptr<Linear> attn_q_a_projs_;
  std::shared_ptr<Linear> attn_kv_a_lora_projs_;
  std::shared_ptr<Linear> attn_kv_a_ropes_;
  std::shared_ptr<Linear> attn_q_b_projs_;
  std::shared_ptr<Linear> attn_q_b_lora_projs_;
  std::shared_ptr<Linear> attn_q_b_rope_projs_;
  std::shared_ptr<Linear> attn_o_proj_;
  std::shared_ptr<Split> split_;
  std::shared_ptr<Bmm> attn_w_uk_t_bmm_;
  std::shared_ptr<FlashMlaAttention> flash_mla_attention_layers_;
  std::shared_ptr<PagedMlaAttention> paged_mla_attention_layers_;
  std::shared_ptr<MemAdjuster> mem_adjuster_;
  inline static uint32_t qk_nope_head_dim_ = 0;
  inline static uint32_t qk_rope_head_dim_ = 0;
  inline static uint32_t kv_lora_rank_ = 0;
  inline static uint32_t q_lora_rank_ = 0;
  inline static int head_num_per_tp_ = 0;
  size_t o_proj_k_dim_ = 0;

  std::shared_ptr<Layernorm> kv_a_layernorms_;
  std::shared_ptr<Layernorm> q_a_layernorms_;

  AbsorbWeightsType absorb_type_ = AbsorbWeightsType::kAbsorbDisabled;
};

}  // namespace ksana_llm
