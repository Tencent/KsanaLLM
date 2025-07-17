/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include "ksana_llm/models/common/common_model.h"
#include "ksana_llm/models/common/model_interface.h"

#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/attention/cross_layer_attention.h"
#include "ksana_llm/modules/ffn/two_layered_ffn.h"

#include "ksana_llm/modules/basic/moe.h"

namespace ksana_llm {

template <typename T>
class HunyuanDecoderLayer {
 public:
  HunyuanDecoderLayer(int layer_idx, TensorBuffer* moe_buffer, int cla_share_factor, ClaBuffers& cla_buffers,
                      LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config);
  ~HunyuanDecoderLayer() = default;

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

  std::shared_ptr<CrossLayerAttention<T>> cla_;

  std::shared_ptr<TwoLayeredFFN<T>> shared_mlps_;
  std::shared_ptr<MoE<T>> moes_;
  std::shared_ptr<Linear<T>> expert_gates_;
  std::shared_ptr<TpCommunicator<T>> tp_comm_;

  TensorBuffer* moe_buffer_;
};

template <typename T>
class HunyuanLarge : public ModelInterface<T> {
 public:
  HunyuanLarge() {}
  ~HunyuanLarge() = default;

  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) override;

 private:
  // for cla (if the model not use cross of attention, default nullptr)
  int cla_share_factor_;
  ClaBuffers cla_buffers_;
  TensorBuffer* moe_buffer_;

  std::map<int, std::shared_ptr<HunyuanDecoderLayer<T>>> decoder_layers_;
};

template <typename T>
class HunyuanLargeModel : public CommonModel<T> {
 public:
  HunyuanLargeModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                    std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight);

 private:
  Status CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) override;

  // Execute the forward of specific layers.
  Status LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel<T>::GetHiddenUnitBuffer;
  using CommonModel<T>::SetHiddenUnitBuffer;

 private:
  HunyuanLarge<T> hunyuan_large_;
};

}  // namespace ksana_llm
