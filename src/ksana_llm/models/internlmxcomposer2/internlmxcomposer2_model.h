/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_weight.h"
#include "ksana_llm/models/common/model_interface.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/utils.h"

#include "ksana_llm/modules/basic/add.h"

#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/attention/multihead_attention.h"
#include "ksana_llm/modules/basic/mul.h"
#include "ksana_llm/modules/ffn/two_layered_ffn.h"

namespace ksana_llm {

class InternlmxComposer2DecoderLayer {
 public:
  InternlmxComposer2DecoderLayer(int layer_idx, LayerCreationContext& creation_context,
                                 ModelCreationConfig& model_creation_config, TensorBuffer* plora_a_buffer_,
                                 TensorBuffer* plora_b_buffer_);
  ~InternlmxComposer2DecoderLayer() {}
  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext& forwarding_context);

 private:
  Status FlashAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                               std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& reduce_buffer_tensors,
                               ForwardingContext& forwarding_context);

  Status PagedAttentionForward(std::vector<Tensor>& hidden_buffer_tensors_0,
                               std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& reduce_buffer_tensors,
                               ForwardingContext& forwarding_context);

  Status ForwardMlp(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& hidden_buffer_tensors_1,
                    std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                    ForwardingContext& forwarding_context);

  Status ForwardMha(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                    std::vector<Tensor>& hidden_buffer_tensors_1, const bool is_multi_token_forward,
                    ForwardingContext& forwarding_context);

 private:
  int layer_idx_;
  std::shared_ptr<TpCommunicator> tp_comm_;
  std::shared_ptr<Layernorm> input_layernorms_;
  std::shared_ptr<Layernorm> post_attention_layernorms_;

  // PLora
  std::shared_ptr<Add> adds_;
  std::shared_ptr<Linear> gate_proj_plora_a_;
  std::shared_ptr<Linear> gate_proj_plora_b_;
  std::shared_ptr<Linear> up_proj_plora_a_;
  std::shared_ptr<Linear> up_proj_plora_b_;
  std::shared_ptr<Linear> down_proj_plora_a_;
  std::shared_ptr<Linear> down_proj_plora_b_;
  std::shared_ptr<Linear> qkv_lora_a_proj_;
  std::shared_ptr<Linear> qkv_lora_b_proj_;
  std::shared_ptr<Linear> o_lora_a_proj_;
  std::shared_ptr<Linear> o_lora_b_proj_;
  std::shared_ptr<Mul> mask_muls_;

  // buffer
  TensorBuffer* plora_a_buffer_;
  TensorBuffer* plora_b_buffer_;

  // attention
  std::shared_ptr<Linear> attn_qkv_projs_;
  std::shared_ptr<Linear> attn_o_projs_;
  std::shared_ptr<FlashAttention> flash_attentions_;
  std::shared_ptr<PagedAttention> paged_attentions_;

  // ffn related
  std::shared_ptr<Linear> mlp_gate_projs_;
  std::shared_ptr<Linear> mlp_up_projs_;
  std::shared_ptr<Linear> mlp_down_projs_;
  std::shared_ptr<SiluMul> silu_muls_;

  Tensor query_layernorm_weight_;
  Tensor key_layernorm_weight_;
};

class InternlmxComposer2 : public ModelInterface {
 public:
  InternlmxComposer2() {}
  ~InternlmxComposer2() = default;
  Status GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) override;
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;
  Status Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;

 private:
  std::map<int, std::shared_ptr<InternlmxComposer2DecoderLayer>> decoder_layers_;
  TensorBuffer* plora_a_buffer_;
  TensorBuffer* plora_b_buffer_;
};

class InternlmxComposer2Model : public CommonModel {
 public:
  InternlmxComposer2Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                          std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight);
  ~InternlmxComposer2Model() {}

 private:
  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override;

  // Execute the forward of specific layers.
  Status LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode = RunMode::kMain) override;

 protected:
  using CommonModel::GetHiddenUnitBuffer;
  using CommonModel::SetHiddenUnitBuffer;

 private:
  ModelConfig model_config_;

 private:
  InternlmxComposer2 internlmx_composer2_;
};

}  // namespace ksana_llm
