/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_model.h"
#include "ksana_llm/models/common/common_weight.h"
#include "ksana_llm/models/common/model_interface.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/config/model_config_parser.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/utils.h"

#include "ksana_llm/models/common/simple_decoder_layer.h"
#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/attention/multihead_attention.h"
#include "ksana_llm/modules/basic/add.h"
#include "ksana_llm/modules/basic/add_mul.h"
#include "ksana_llm/modules/ffn/two_layered_ffn.h"
namespace ksana_llm {

// Custom decoder layer for BGE reranker that supports scale_depth
class BgeScaledDecoderLayer {
 public:
  BgeScaledDecoderLayer(int layer_idx, bool is_neox, bool add_qkv_bias, float scale_depth,
                        LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config);
  ~BgeScaledDecoderLayer() {}
  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext& forwarding_context);

 private:
  int layer_idx_;
  std::shared_ptr<MultiHeadAttention> mha_;
  std::shared_ptr<TwoLayeredFFN> mlps_;
  std::shared_ptr<TpCommunicator> tp_comm_;

  std::shared_ptr<AddMul> attn_add_layer_;
  std::shared_ptr<AddMul> mlp_add_layer_;

  std::shared_ptr<Layernorm> input_layernorms_;
  std::shared_ptr<Layernorm> post_attention_layernorms_;

  float scale_depth_;
};

class BgeRerankerMinicpm : public ModelInterface {
 public:
  BgeRerankerMinicpm() {}
  ~BgeRerankerMinicpm() = default;

  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;

 private:
  std::map<int, std::shared_ptr<BgeScaledDecoderLayer>> decoder_layers_;
};

class BgeRerankerMinicpmModel : public CommonModel {
 public:
  BgeRerankerMinicpmModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                          std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight);
  ~BgeRerankerMinicpmModel() {}

 private:
  void InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight);

  // Implement pure virtual functions from CommonModel
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

  bool BgeRerankerUpdateResponse(std::vector<ForwardRequest>& forward_reqs, Tensor& output, const std::string& stage);

  Status LmHead(ForwardingContext& forwarding_context, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                std::vector<ForwardRequest>& forward_reqs, RunMode run_mode) override;

 protected:
  using CommonModel::context_;
  using CommonModel::rank_;

  using CommonModel::GetHiddenUnitBuffer;
  using CommonModel::model_config_;
  using CommonModel::model_run_config_;
  using CommonModel::SetHiddenUnitBuffer;

 private:
  std::shared_ptr<BaseLayer> add_layer_;
  std::shared_ptr<BaseLayer> lm_head_proj_layer_;
  BgeRerankerMinicpm bge_reranker_minicpm_;
};

}  // namespace ksana_llm