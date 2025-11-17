/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_model.h"

#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/attention/cross_layer_attention.h"
#include "ksana_llm/modules/ffn/two_layered_ffn.h"

#include "ksana_llm/models/common/simple_decoder_layer.h"

#include "ksana_llm/modules/basic/moe.h"

namespace ksana_llm {

class HunyuanTurboModel : public CommonModel {
 public:
  HunyuanTurboModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                    std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight);

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;

  // Execute the forward of specific layers.
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel::GetHiddenUnitBuffer;
  using CommonModel::SetHiddenUnitBuffer;

 private:
  // for cla (if the model not use cross of attention, default nullptr)
  int cla_share_factor_;
  ClaBuffers cla_buffers_;
  TensorBuffer* moe_buffer_;

  std::map<int, std::shared_ptr<SimpleDecoderLayer>> decoder_layers_;
};

}  // namespace ksana_llm
