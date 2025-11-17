/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/bge_reranker_minicpm/bge_reranker_minicpm_model.h"
#include "ksana_llm/layers/add_layer.h"
#include "ksana_llm/modules/basic/linear.h"

namespace ksana_llm {

// Implementation of BgeScaledDecoderLayer
BgeScaledDecoderLayer::BgeScaledDecoderLayer(int layer_idx, bool is_neox, bool add_qkv_bias, float scale_depth,
                                             LayerCreationContext& creation_context,
                                             ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx), scale_depth_(scale_depth) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  input_layernorms_ = std::make_shared<Layernorm>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);
  post_attention_layernorms_ =
      std::make_shared<Layernorm>(layer_prefix + ".post_attention_layernorm.weight",
                                  model_creation_config.layernorm_config.layernorm_eps, creation_context);

  // Create AddMul instances with scale_depth parameters using MUL_THEN_ADD operation
  // Original logic: output = (input * scale_depth) + residual
  // MUL_THEN_ADD: output = input1 * scale1 + input2 * scale2
  // input1=hidden, input2=residual, scale1=scale_depth, scale2=1.0
  attn_add_layer_ = std::make_shared<AddMul>(scale_depth, 1.0f, creation_context);
  mlp_add_layer_ = std::make_shared<AddMul>(scale_depth, 1.0f, creation_context);

  bool use_qk_norm = model_creation_config.attn_config.use_qk_norm;
  mha_ = std::make_shared<MultiHeadAttention>(layer_idx, is_neox, add_qkv_bias, use_qk_norm, creation_context,
                                              model_creation_config);
  mlps_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config);
  tp_comm_ = std::make_shared<TpCommunicator>();
}

Status BgeScaledDecoderLayer::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                      ForwardingContext& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);

  input_layernorms_->Forward(residual_buffer, hidden_buffer_tensors_0);

  // MultiHeadAttention
  STATUS_CHECK_RETURN(
      mha_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Attn residual add with scale_depth using MUL_THEN_ADD: hidden * scale_depth + residual * 1.0
  STATUS_CHECK_RETURN(attn_add_layer_->Forward({hidden_buffer_tensors_0[0], residual_buffer[0]}, residual_buffer));

  // Pre mlp layernorm
  post_attention_layernorms_->Forward(residual_buffer, hidden_buffer_tensors_0);

  // Common mlp
  STATUS_CHECK_RETURN(
      mlps_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Mlp residual add with scale_depth using MUL_THEN_ADD: hidden * scale_depth + residual * 1.0
  STATUS_CHECK_RETURN(mlp_add_layer_->Forward({hidden_buffer_tensors_0[0], residual_buffer[0]}, residual_buffer));

  return Status();
}

// Implementation of BgeRerankerMinicpm (ModelInterface layer)
Status BgeRerankerMinicpm::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.use_emb_scale = true;
  model_run_config.emb_scale = model_config.emb_scale;
  model_run_config.scale_depth = model_config.scale_depth / std::sqrt(model_config.num_layer);
  return Status();
}

Status BgeRerankerMinicpm::CreateLayers(LayerCreationContext& creation_context,
                                        ModelCreationConfig& model_creation_config) {
  bool is_neox = true;
  bool add_qkv_bias = false;
  auto& model_config = model_creation_config.attn_config.model_config;
  float scale_depth = model_config.scale_depth / std::sqrt(model_config.num_layer);

  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] = std::make_shared<BgeScaledDecoderLayer>(layer_idx, is_neox, add_qkv_bias, scale_depth,
                                                                         creation_context, model_creation_config);
  }
  return Status();
}

Status BgeRerankerMinicpm::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  int cutoff_layer = forwarding_context.GetModelInput()->cutoff_layer;

  int max_layer_to_execute = forwarding_context.GetPipelineConfig().upper_layer_idx;
  if (cutoff_layer != 0) {
    max_layer_to_execute = std::min(static_cast<int>(cutoff_layer - 1),
                                    static_cast<int>(forwarding_context.GetPipelineConfig().upper_layer_idx));
  }

  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx; layer_idx <= max_layer_to_execute;
       ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }

  return Status();
}

// Implementation of BgeRerankerMinicpmModel (CommonModel layer)
BgeRerankerMinicpmModel::BgeRerankerMinicpmModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                                 const int rank, std::shared_ptr<Context> context,
                                                 std::shared_ptr<BaseWeight> base_weight)
    : CommonModel(model_config, runtime_config, rank, context) {
  ModelRunConfig model_run_config;
  bge_reranker_minicpm_.GetModelRunConfig(model_run_config, model_config);
  InitRunConfig(model_run_config, base_weight);
  KLLM_LOG_DEBUG << "BgeRerankerMinicpmModel constructor called";
}

void BgeRerankerMinicpmModel::InitRunConfig(const ModelRunConfig& model_run_config,
                                            std::shared_ptr<BaseWeight> base_weight) {
  CommonModel::InitRunConfig(model_run_config, base_weight);

  KLLM_LOG_DEBUG << "BgeRerankerMinicpmModel InitRunConfig called";
#ifdef ENABLE_CUDA
  if (context_->IsChief() && rank_ == 0) {
    DataType weight_type = model_config_.weight_data_type;
    DataType input_type = weight_type;
    DataType output_type = weight_type;

    lm_head_proj_layer_ = this->layer_creation_context_.matmul_layer_factory->AutoCreateLayer(
        base_weight, "lm_head.0.linear_head.weight", weight_type, input_type, output_type, DEFAULT_LINEAR_BACKEND, {});
  }
#endif
}

Status BgeRerankerMinicpmModel::CreateLayers(LayerCreationContext& creation_context,
                                             ModelCreationConfig& model_creation_config) {
  return bge_reranker_minicpm_.CreateLayers(creation_context, model_creation_config);
}

Status BgeRerankerMinicpmModel::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  STATUS_CHECK_RETURN(bge_reranker_minicpm_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);
  return Status();
}

bool BgeRerankerMinicpmModel::BgeRerankerUpdateResponse(std::vector<ForwardRequest*>& forward_reqs,
                                                        Tensor& output, const std::string& stage) {
  bool ret = true;
  int req_offset = 0;
  for (auto& req : forward_reqs) {
    int output_token_num = req->forwarding_tokens->size();
    // Validate request_target, stage existence, and token_reduce_mode in one check
    if (!req->request_target) {
      ret = false;
      continue;
    }
    auto it = req->request_target->find(stage);
    if (it == req->request_target->end() || it->second.token_reduce_mode != TokenReduceMode::GATHER_ALL) {
      ret = false;
      continue;
    }
    // Determine whether to exit early
    ret &= req->request_target->size() == req->response->size();
    if (rank_ != 0) continue;

    if (stage == "lm_head") {
      size_t chunk_size = GetTypeSize(output.dtype) * output.shape[1];
      // Update the response tensor with the sliced data.
      PythonTensor& ret_tensor = (*req->response)[stage];
      ret_tensor.shape = {static_cast<size_t>(output_token_num), output.shape[1]};
      ret_tensor.dtype = GetTypeString(output.dtype);
      ret_tensor.data.resize(output_token_num * chunk_size);
      MemcpyAsync(ret_tensor.data.data(), output.GetPtr<void>() + req_offset * chunk_size,
                  output_token_num * chunk_size, MEMCPY_DEVICE_TO_HOST, context_->GetComputeStreams()[rank_]);
      StreamSynchronize(context_->GetComputeStreams()[rank_]);
      req_offset += output_token_num;
      continue;
    }
    return false;
  }
  return ret;
}

Status BgeRerankerMinicpmModel::LmHead(ForwardingContext& forwarding_context,
                                       std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                       std::vector<ForwardRequest*>& forward_reqs, RunMode run_mode) {
  if (rank_ != 0) return Status();

  int cutoff_layer = forwarding_context.GetModelInput()->cutoff_layer;
  if (cutoff_layer == 0 || cutoff_layer < model_config_.start_layer) return Status();

  int layer_idx = cutoff_layer - model_config_.start_layer;
  Tensor lm_head_weight = base_weight->GetModelWeights(fmt::format("lm_head.{}.linear_head.weight", layer_idx));
  std::vector<Tensor>& cutoff_hidden_states = this->GetHiddenUnitBuffer(forwarding_context, false);

  this->lm_head_prenorm_->Forward(cutoff_hidden_states, cutoff_hidden_states);

  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  STATUS_CHECK_RETURN(lm_head_proj_layer_->Forward({cutoff_hidden_states[0], lm_head_weight}, hidden_buffer_tensors_0));

  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  if (is_multi_token_forward && this->BgeRerankerUpdateResponse(forward_reqs, hidden_buffer_tensors_0[0], "lm_head")) {
    StreamSynchronize(context_->GetComputeStreams()[rank_]);
    this->input_refit_layer_->Clear();
  }

  return Status();
}

}  // namespace ksana_llm