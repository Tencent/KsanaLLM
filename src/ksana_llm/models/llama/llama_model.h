/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/model_interface.h"
#include "ksana_llm/models/common/simple_decoder_layer.h"

namespace ksana_llm {

template <typename T>
class Llama : public ModelInterface<T> {
 public:
  Llama() {}
  ~Llama() = default;

  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) override;

 private:
  std::map<int, std::shared_ptr<SimpleDecoderLayer<T>>> decoder_layers_;
};

template <typename T>
class LlamaModel : public CommonModel<T> {
 public:
  LlamaModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
             std::shared_ptr<BaseWeight> base_weight);
  ~LlamaModel() = default;

 private:
  Status CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) override;
  Status LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel<T>::GetHiddenUnitBuffer;
  using CommonModel<T>::SetHiddenUnitBuffer;

 private:
  Llama<T> llama_;
};

}  // namespace ksana_llm
