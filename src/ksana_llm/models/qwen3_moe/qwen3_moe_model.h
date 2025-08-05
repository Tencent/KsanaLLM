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
class Qwen3MoeDecoderLayer {
 public:
  Qwen3MoeDecoderLayer(int layer_idx, TensorBuffer* moe_buffer, LayerCreationContext& creation_context,
                       ModelCreationConfig& model_creation_config);
  ~Qwen3MoeDecoderLayer() = default;

  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext& forwarding_context);

 private:
  int layer_idx_;
  std::shared_ptr<Add> adds_;
  std::shared_ptr<Layernorm> input_layernorms_;
  std::shared_ptr<Layernorm> post_attention_layernorms_;
  std::shared_ptr<TpCommunicator<T>> tp_comm_;

  std::shared_ptr<MultiHeadAttention<T>> mha_;
  std::shared_ptr<MoE> moes_;
  std::shared_ptr<Linear> expert_gates_;

  TensorBuffer* moe_buffer_;
};

template <typename T>
class Qwen3Moe : public ModelInterface<T> {
 public:
  Qwen3Moe() {}
  ~Qwen3Moe() = default;

  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;

 private:
  TensorBuffer* moe_buffer_;

  std::map<int, std::shared_ptr<Qwen3MoeDecoderLayer<T>>> decoder_layers_;
};

template <typename T>
class Qwen3MoeModel : public CommonModel {
 public:
  Qwen3MoeModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight);
  ~Qwen3MoeModel() = default;

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config);
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel::GetHiddenUnitBuffer;
  using CommonModel::SetHiddenUnitBuffer;

 private:
  Qwen3Moe<T> qwen3moe_;
};

}  // namespace ksana_llm
