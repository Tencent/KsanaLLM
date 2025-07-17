/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_model.h"

#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/attention/common_attention.h"

#include "ksana_llm/modules/basic/moe.h"

namespace ksana_llm {

struct ClaBuffers {
  std::vector<Tensor> cla_k_buffer_{1};
  std::vector<Tensor> cla_v_buffer_{1};
};

template <typename T>
class CrossLayerAttention {
 public:
  CrossLayerAttention(int layer_idx, int cla_share_factor, ClaBuffers& cla_buffers,
                      LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config);
  ~CrossLayerAttention() = default;

  // Input tensors: hidden_buffer_tensors_0
  // Output tensors: hidden_buffer_tensors_0
  //                 or reduce_buffer_tensors if have forwarding_context.GetModelCommunicator()
  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                 const bool is_multi_token_forward, ForwardingContext<T>& forwarding_context);

  static Status CreateBuffers(BufferManager* buffer_mgr, const RuntimeConfig& runtime_config,
                              const AttentionCreationConfig& attn_config, ClaBuffers& cla_buffers);

 private:
  Status QKVClaBufferCopy(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& hidden_buffer_tensors_1,
                          ForwardingContext<T>& forwarding_context);

 private:
  int layer_idx_;
  std::shared_ptr<Linear<T>> attn_qkv_projs_;
  std::shared_ptr<CommonAttention<T>> attentions_;

#ifdef ENABLE_VLLM_FLASH_ATTN_2
  std::shared_ptr<BaseLayer> set_torch_stream_layer_;
#endif

  // cla related variables
  int cla_share_factor_;
  ClaBuffers& cla_buffers_;
  size_t qkv_pitch_ = 0;
  size_t q_pitch_ = 0;
  size_t kv_pitch_ = 0;
};
}  // namespace ksana_llm
