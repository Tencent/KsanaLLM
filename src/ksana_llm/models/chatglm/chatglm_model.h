/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_model.h"
#include "ksana_llm/models/common/simple_decoder_layer.h"

namespace ksana_llm {
template <typename T>
class ChatglmModel : public CommonModel<T> {
 public:
  ChatglmModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
               std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight);
  ~ChatglmModel() {}

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;

  // Execute the forward of specific layers.
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel<T>::GetHiddenUnitBuffer;
  using CommonModel<T>::SetHiddenUnitBuffer;

 private:
  std::map<int, std::shared_ptr<SimpleDecoderLayer<T>>> decoder_layers_;
};

}  // namespace ksana_llm
