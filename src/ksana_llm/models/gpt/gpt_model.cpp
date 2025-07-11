/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/gpt/gpt_model.h"

#include <memory>
#include <vector>

#include "fmt/core.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

template <typename T>
GPTDecoderLayer<T>::GPTDecoderLayer(int layer_idx, LayerCreationContext<T>& creation_context,
                                    ModelCreationConfig& model_creation_config, TensorBuffer* mlp_temp_buffer_)
    : layer_idx_(layer_idx), mlp_temp_buffer_(mlp_temp_buffer_) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
  if (model_creation_config.layernorm_config.activation_function == "gelu" ||
      model_creation_config.layernorm_config.activation_function == "gelu_new") {
    activation_layer_ = std::make_shared<Activation<T>>("gelu", creation_context);
  } else {
    activation_layer_ = std::make_shared<Activation<T>>("relu", creation_context);
  }

  std::string input_layernorm_name = layer_prefix + ".input_layernorm.weight";
  std::string input_layernorm_bias_name =
      input_layernorm_name.substr(0, input_layernorm_name.size() - strlen("weight")) + "bias";
  input_layernorms_ =
      std::make_shared<Layernorm<T>>(input_layernorm_name, model_creation_config.layernorm_config.layernorm_eps,
                                     creation_context, input_layernorm_bias_name);

  std::string post_attention_layernorm_name = layer_prefix + ".post_attention_layernorm.weight";
  std::string post_attention_layernorm_bias_name =
      post_attention_layernorm_name.substr(0, post_attention_layernorm_name.size() - strlen("weight")) + "bias";
  post_attention_layernorms_ = std::make_shared<Layernorm<T>>(post_attention_layernorm_name,
                                                              model_creation_config.layernorm_config.layernorm_eps,
                                                              creation_context, post_attention_layernorm_bias_name);

  adds_ = std::make_shared<Add<T>>(creation_context);
  attn_proj_bias_add_ = std::make_shared<Add<T>>(creation_context, layer_prefix + ".self_attn.o_proj.bias");
  mlp_gate_bias_add_ = std::make_shared<Add<T>>(creation_context, layer_prefix + ".mlp.gate_proj.bias");
  mlp_down_proj_bias_add_ = std::make_shared<Add<T>>(creation_context, layer_prefix + ".mlp.down_proj.bias");

  mlp_gate_proj_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.gate_proj.weight", creation_context,
                                               model_creation_config.attn_config.model_config.quant_config.backend);
  mlp_down_proj_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.down_proj.weight", creation_context,
                                               model_creation_config.attn_config.model_config.quant_config.backend);

  attn_qkv_projs_ = std::make_shared<Linear<T>>(layer_prefix + ".self_attn.query_key_value.weight", creation_context,
                                                model_creation_config.attn_config.model_config.quant_config.backend);

  bool is_neox = true;
  bool use_qk_norm = false;
  qkv_bias_ = creation_context.base_weight->GetModelWeights(layer_prefix + ".self_attn.query_key_value.bias");
  attentions_ =
      std::make_shared<CommonAttention<T>>(layer_idx, is_neox, use_qk_norm, creation_context, model_creation_config);

  tp_comm_ = std::make_shared<TpCommunicator<T>>();
}

template <typename T>
Status GPTDecoderLayer<T>::ForwardMlp(std::vector<Tensor>& mlp_temp_buffer_tensors,
                                      std::vector<Tensor>& hidden_buffer_tensors_0,
                                      std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                      ForwardingContext<T>& forwarding_context) {
  STATUS_CHECK_RETURN(mlp_gate_bias_add_->Forward(mlp_temp_buffer_tensors[0], mlp_temp_buffer_tensors));
  std::swap(mlp_temp_buffer_tensors, hidden_buffer_tensors_0);
  STATUS_CHECK_RETURN(activation_layer_->Forward({hidden_buffer_tensors_0[0]}, hidden_buffer_tensors_0));
  // Mlp down_proj MatMul
  if (forwarding_context.GetModelCommunicator()) {
    STATUS_CHECK_RETURN(mlp_down_proj_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors));
  } else {
    STATUS_CHECK_RETURN(mlp_down_proj_->Forward(hidden_buffer_tensors_0, mlp_temp_buffer_tensors));
    std::swap(mlp_temp_buffer_tensors, hidden_buffer_tensors_0);
  }
  // Only add down_proj bias for rank 0 to avoid duplication.
  if (forwarding_context.GetCurrentRank() == 0) {
    STATUS_CHECK_RETURN(mlp_down_proj_bias_add_->Forward(hidden_buffer_tensors_0[0], hidden_buffer_tensors_0));
  }
  return Status();
}

template <typename T>
Status GPTDecoderLayer<T>::ForwardMha(std::vector<Tensor>& hidden_buffer_tensors_0,
                                      std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                      ForwardingContext<T>& forwarding_context) {
  {
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
    STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], qkv_bias_, hidden_buffer_tensors_1));
    std::swap(hidden_buffer_tensors_0, hidden_buffer_tensors_1);
  }

  // Common attention
  STATUS_CHECK_RETURN(
      attentions_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  return Status();
}

template <typename T>
Status GPTDecoderLayer<T>::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                   ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);
  CREATE_BUFFER_SCOPE(mlp_temp_buffer_tensors, mlp_temp_buffer_);

  attn_qkv_projs_->Forward(residual_buffer, hidden_buffer_tensors_0);

  STATUS_CHECK_RETURN(
      ForwardMha(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  // Only Add o_proj bias on rank 0 to avoid duplication.
  if (forwarding_context.GetCurrentRank() == 0) {
    STATUS_CHECK_RETURN(attn_proj_bias_add_->Forward(hidden_buffer_tensors_0[0], hidden_buffer_tensors_0));
  }

  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Attn residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], hidden_buffer_tensors_0));

  // Post layernorm
  input_layernorms_->Forward(hidden_buffer_tensors_0, residual_buffer);

  // Mlp gate_proj MatMul
  STATUS_CHECK_RETURN(mlp_gate_proj_->Forward(residual_buffer, mlp_temp_buffer_tensors));

  // Common mlp
  STATUS_CHECK_RETURN(ForwardMlp(mlp_temp_buffer_tensors, hidden_buffer_tensors_0, reduce_buffer_tensors,
                                 is_multi_token_forward, forwarding_context));

  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Mlp residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], hidden_buffer_tensors_0));
  // Post layernorm
  post_attention_layernorms_->Forward(hidden_buffer_tensors_0, residual_buffer);
  return Status();
}

template <typename T>
Status Gpt<T>::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  model_run_config.position_encoding = PositionEncoding::LEARNED_ABSOLUTE;
  model_run_config.emb_lookup_use_rotary_embedding_pos = true;
  // Use the vocab size to distinguish each model
  if (model_config.vocab_size == 40478) {  // GPT-1
    model_run_config.layernorm_position = LayerNormPosition::POST_NORM;
  } else if (model_config.vocab_size == 7000) {  // Fairseq transformerp
    model_run_config.layernorm_position = LayerNormPosition::POST_NORM;
    model_run_config.emb_scale = std::sqrt(model_config.hidden_units);
  }
  return Status();
}

template <typename T>
Status Gpt<T>::CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) {
  auto& model_config = model_creation_config.attn_config.model_config;
  int hidden_units = model_config.size_per_head * model_config.head_num;
  int inter_size_per_tp = model_config.inter_size / model_config.tensor_para_size;
  size_t shared_buffer_size = model_config.max_step_token_num * std::max(inter_size_per_tp, hidden_units * 2);
  mlp_temp_buffer_ = creation_context.buffer_mgr_->CreateBufferTensor("mlp_temp_buffer_", {shared_buffer_size},
                                                                      model_config.weight_data_type);
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] =
        std::make_shared<GPTDecoderLayer<T>>(layer_idx, creation_context, model_creation_config, mlp_temp_buffer_);
  }
  return Status();
}

template <typename T>
Status Gpt<T>::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

template class Gpt<float>;
template class Gpt<float16>;
template class Gpt<bfloat16>;

/**********************************************************
 * GptModel
 ***********************************************************/
template <typename T>
GptModel<T>::GptModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                      std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, rank, context) {
  ModelRunConfig model_run_config;
  gpt_.GetModelRunConfig(model_run_config, model_config);
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status GptModel<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                 ModelCreationConfig& model_creation_config) {
  return gpt_.CreateLayers(creation_context, model_creation_config);
}

template <typename T>
Status GptModel<T>::LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  STATUS_CHECK_RETURN(gpt_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);
  return Status();
}

template class GptModel<float>;
template class GptModel<float16>;
template class GptModel<bfloat16>;

}  // namespace ksana_llm
