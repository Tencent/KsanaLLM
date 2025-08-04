/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/model_interface.h"

#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/attention/multihead_attention.h"
#include "ksana_llm/modules/basic/moe.h"

namespace ksana_llm {

template <typename T>
class MixtralDecoderLayer {
 public:
  MixtralDecoderLayer(int layer_idx, LayerCreationContext& creation_context,
                      ModelCreationConfig& model_creation_config);
  ~MixtralDecoderLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext& forwarding_context);

 private:
  int layer_idx_;
  std::shared_ptr<Add<T>> adds_;
  std::shared_ptr<Layernorm<T>> input_layernorms_;
  std::shared_ptr<Layernorm<T>> post_attention_layernorms_;
  std::shared_ptr<TpCommunicator<T>> tp_comm_;

  std::shared_ptr<MultiHeadAttention<T>> mha_;
  std::shared_ptr<MoE<T>> moes_;
  std::shared_ptr<Linear<T>> expert_gates_;
};

template <typename T>
class Mixtral : public ModelInterface<T> {
 public:
  Mixtral() {}
  ~Mixtral() = default;

  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;

 private:
  std::map<int, std::shared_ptr<MixtralDecoderLayer<T>>> decoder_layers_;
};

template <typename T>
class MixtralModel : public CommonModel<T> {
 public:
  MixtralModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
               std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight);
  ~MixtralModel() = default;

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config);
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel<T>::GetHiddenUnitBuffer;
  using CommonModel<T>::SetHiddenUnitBuffer;

 private:
  Mixtral<T> mixtral_;
};

}  // namespace ksana_llm
