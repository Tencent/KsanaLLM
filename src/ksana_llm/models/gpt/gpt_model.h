/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_model.h"
#include "ksana_llm/models/common/common_weight.h"
#include "ksana_llm/models/common/model_interface.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/utils.h"

#include "ksana_llm/modules/basic/activation.h"
#include "ksana_llm/modules/basic/add.h"
#include "ksana_llm/modules/basic/layernorm.h"
#include "ksana_llm/modules/basic/linear.h"

#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/attention/multihead_attention.h"
#include "ksana_llm/modules/ffn/two_layered_ffn.h"

namespace ksana_llm {

template <typename T>
class GPTDecoderLayer {
 public:
  GPTDecoderLayer(int layer_idx, LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config,
                  TensorBuffer* mlp_temp_buffer_);
  ~GPTDecoderLayer() {}
  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext<T>& forwarding_context);

 private:
  Status ForwardMlp(std::vector<Tensor>& mlp_temp_buffer_tensors, std::vector<Tensor>& hidden_buffer_tensors_0,
                    std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                    ForwardingContext<T>& forwarding_context);

  Status ForwardMha(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                    const bool is_multi_token_forward, ForwardingContext<T>& forwarding_context);

  Status FlashAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                               std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& reduce_buffer_tensors,
                               ForwardingContext<T>& forwarding_context);

  Status PagedAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                               std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& reduce_buffer_tensors,
                               ForwardingContext<T>& forwarding_context);

 private:
  int layer_idx_;
  std::shared_ptr<TpCommunicator<T>> tp_comm_;
  std::shared_ptr<Layernorm<T>> input_layernorms_;
  std::shared_ptr<Layernorm<T>> post_attention_layernorms_;
  std::shared_ptr<Add<T>> adds_;
  std::shared_ptr<Add<T>> attn_proj_bias_add_;
  std::shared_ptr<Add<T>> mlp_gate_bias_add_;
  std::shared_ptr<Add<T>> mlp_down_proj_bias_add_;

  // ffn related
  std::shared_ptr<Linear<T>> mlp_gate_proj_;
  std::shared_ptr<Linear<T>> mlp_down_proj_;
  std::shared_ptr<Activation<T>> activation_layer_;

  // buffer
  TensorBuffer* mlp_temp_buffer_;

  // attention
  std::shared_ptr<CommonAttention<T>> attentions_;
  Tensor qkv_bias_;
  std::shared_ptr<Linear<T>> attn_qkv_projs_;
};

template <typename T>
class Gpt : public ModelInterface<T> {
 public:
  Gpt() {}
  ~Gpt() = default;
  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) override;

 private:
  std::map<int, std::shared_ptr<GPTDecoderLayer<T>>> decoder_layers_;
  std::shared_ptr<Activation<T>> activation_layer_;
  TensorBuffer* mlp_temp_buffer_;
};

template <typename T>
class GptModel : public CommonModel<T> {
 public:
  GptModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
           std::shared_ptr<BaseWeight> base_weight);
  ~GptModel() {}

 private:
  Status CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) override;

  // Execute the forward of specific layers.
  Status LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

  void InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight);

 protected:
  using CommonModel<T>::GetHiddenUnitBuffer;
  using CommonModel<T>::SetHiddenUnitBuffer;

 private:
  ModelConfig model_config_;

 private:
  Gpt<T> gpt_;
};

}  // namespace ksana_llm
