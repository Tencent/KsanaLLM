/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/model_interface.h"
#include "ksana_llm/models/common/simple_decoder_layer.h"

namespace ksana_llm {

class Baichuan : public ModelInterface {
 public:
  Baichuan() {}
  ~Baichuan() = default;

  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;

 private:
  std::map<int, std::shared_ptr<SimpleDecoderLayer>> decoder_layers_;
};

class BaichuanModel : public CommonModel {
 public:
  BaichuanModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight);
  ~BaichuanModel() = default;

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel::GetHiddenUnitBuffer;
  using CommonModel::SetHiddenUnitBuffer;

 private:
  Baichuan baichuan_;
};

}  // namespace ksana_llm
