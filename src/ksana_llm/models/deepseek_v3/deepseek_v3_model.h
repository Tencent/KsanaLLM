/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/data_hub/expert_parallel_data_transfer.h"
#include "ksana_llm/data_hub/expert_parallel_hidden_unit_buffer.h"
#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/layers/activation_layer.h"
#include "ksana_llm/layers/mul_layer.h"

#include "ksana_llm/models/common/common_model.h"
#include "ksana_llm/modules/attention/multihead_latent_attention.h"

#include "ksana_llm/modules/basic/add_norm.h"
#include "ksana_llm/modules/basic/layernorm.h"

#include "ksana_llm/modules/basic/linear.h"
#include "ksana_llm/modules/basic/moe.h"
#include "ksana_llm/modules/ffn/two_layered_ffn.h"

namespace ksana_llm {
template <typename T>
class DeepSeekV3DecoderLayer {
 public:
  DeepSeekV3DecoderLayer(int layer_idx, bool is_moe, LayerCreationContext<T>& creation_context,
                         ModelCreationConfig& model_creation_config, MlaBuffers& mla_buffers, TensorBuffer* moe_buffer);

  ~DeepSeekV3DecoderLayer() = default;
  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext<T>& forwarding_context, bool need_add_residual_before_attn,
                 bool need_add_residual_after_mlp);

 private:
  Status CommonMlp(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                   const bool is_multi_token_forward, ForwardingContext<T>& forwarding_context);

 private:
  bool is_moe_;

  std::shared_ptr<Layernorm<T>> input_layernorm_;
  std::shared_ptr<AddNorm<T>> pre_attention_add_norm_;
  std::shared_ptr<AddNorm<T>> post_attention_add_norm_;
  std::shared_ptr<Add<T>> add_;
  std::shared_ptr<TpCommunicator<T>> tp_comm_;

  std::shared_ptr<MultiHeadLatentAttention<T>> mla_;

  bool enable_full_shared_expert_;
  int layer_idx_;
  int rank_;

  std::shared_ptr<TwoLayeredFFN<T>> mlp_;
  std::shared_ptr<TwoLayeredFFN<T>> shared_mlp_;
  std::shared_ptr<Linear<T>> expert_gate_;
  std::shared_ptr<MoE<T>> moe_;

  MlaBuffers& mla_buffers_;
  TensorBuffer* moe_buffer_;

  // Be a replacement of residual_buffer_, for distributed mode only.
  std::vector<Tensor> local_residual_buffer_{1};
  std::vector<Tensor> distributed_device_buffer_;
  std::vector<Tensor> distributed_device_buffer_prefill_;

  // Used to send and recive moe input/output among expert parallel nodes.
  std::shared_ptr<ExpertParallelDataTransfer<T>> ep_data_transfer_;
  // Store the moe-computing-tasks from remote expert parallel nodes.
  std::vector<std::vector<Tensor>> moe_queue_in_;
};

template <typename T>
class DeepSeekV3MtpLayer {
 public:
  DeepSeekV3MtpLayer(const int layer_idx, LayerCreationContext<T>& creation_context,
                     ModelCreationConfig& model_creation_config,
                     std::shared_ptr<DeepSeekV3DecoderLayer<T>> decoder_layer);

  ~DeepSeekV3MtpLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context);

 private:
  std::shared_ptr<Layernorm<T>> enorm_;
  std::shared_ptr<Layernorm<T>> hnorm_;
  std::shared_ptr<BaseLayer> concat_layer_;
  std::shared_ptr<Linear<T>> eh_proj_;
  std::shared_ptr<BaseLayer> gather_layer_;
  std::shared_ptr<BaseLayer> emb_lookup_layer_;
  std::shared_ptr<DeepSeekV3DecoderLayer<T>> decoder_layer_;

  std::shared_ptr<TpCommunicator<T>> tp_comm_;
};

template <typename T>
class DeepSeekV3Model : public CommonModel<T> {
 public:
  DeepSeekV3Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                  std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight);

  Status CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) override;

  Status LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel<T>::GetHiddenUnitBuffer;
  using CommonModel<T>::SetHiddenUnitBuffer;

  std::map<int, std::shared_ptr<DeepSeekV3DecoderLayer<T>>> layers_;
  std::map<int, std::shared_ptr<DeepSeekV3MtpLayer<T>>> nextn_layers_;

  int first_k_dense_replace_;
  MlaBuffers mla_buffers_;
  TensorBuffer* moe_buffer_;
};
}  // namespace ksana_llm
