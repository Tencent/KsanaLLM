/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/model_interface.h"

#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/attention/multihead_attention.h"
#include "ksana_llm/modules/basic/moe.h"
#include "ksana_llm/modules/basic/mul.h"
#include "ksana_llm/modules/basic/sigmoid.h"
#include "ksana_llm/modules/ffn/two_layered_ffn.h"

namespace ksana_llm {

template <typename T>
class Llama4DecoderLayer {
 public:
  Llama4DecoderLayer(int layer_idx, TensorBuffer* moe_buffer, bool is_moe_layer_,
                     LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config);
  ~Llama4DecoderLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext<T>& forwarding_context);

 private:
  Status ForwardMlp(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                    const bool is_multi_token_forward, ForwardingContext<T>& forwarding_context);

 private:
  int layer_idx_;
  std::shared_ptr<Add<T>> adds_;
  std::shared_ptr<Layernorm<T>> input_layernorms_;
  std::shared_ptr<Layernorm<T>> post_attention_layernorms_;
  std::shared_ptr<TpCommunicator<T>> tp_comm_;

  std::shared_ptr<MultiHeadAttention<T>> mha_;
  std::shared_ptr<MoE<T>> moes_;
  std::shared_ptr<Linear<T>> expert_gates_;
  std::shared_ptr<TwoLayeredFFN<T>> mlps_;

  TensorBuffer* moe_buffer_;

  bool is_moe_layer_;
};

template <typename T>
class Llama4 : public ModelInterface<T> {
 public:
  Llama4() {}
  ~Llama4() = default;

  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) override;

 private:
  TensorBuffer* moe_buffer_;

  std::map<int, std::shared_ptr<Llama4DecoderLayer<T>>> decoder_layers_;
};

template <typename T>
class Llama4Model : public CommonModel<T> {
 public:
  Llama4Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
              std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight);
  ~Llama4Model() = default;

 private:
  Status CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config);
  Status LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel<T>::GetHiddenUnitBuffer;
  using CommonModel<T>::SetHiddenUnitBuffer;

 private:
  Llama4<T> Llama4_;
};

}  // namespace ksana_llm
