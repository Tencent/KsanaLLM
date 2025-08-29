/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/deepseek_v3/deepseek_v3_model.h"

#include <vector>

#include "ksana_llm/layers/assemble_tokens_hidden_layer.h"
#include "ksana_llm/layers/concat_layer.h"
#include "ksana_llm/layers/emb_lookup_layer.h"
#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/profiler/sched_event_tracer.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

DeepSeekV3DecoderLayer::DeepSeekV3DecoderLayer(int layer_idx, bool is_moe, LayerCreationContext& creation_context,
                                               ModelCreationConfig& model_creation_config, MlaBuffers& mla_buffers,
                                               TensorBuffer* moe_buffer)
    : is_moe_(is_moe), mla_buffers_(mla_buffers), moe_buffer_(moe_buffer) {
  bool is_neox = false;
  layer_idx_ = layer_idx;
  rank_ = creation_context.rank;
  enable_full_shared_expert_ = creation_context.runtime_config.enable_full_shared_expert;

  MoeScaleNormMode moe_scale_norm_mode;
  if (model_creation_config.attn_config.model_config.mla_config.q_lora_rank != 0) {
    moe_scale_norm_mode = MoeScaleNormMode::RE_NORM;
  } else {
    moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;
  }

  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  pre_attention_add_norm_ = std::make_shared<AddNorm>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);

  input_layernorm_ = std::make_shared<Layernorm>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);

  add_ = std::make_shared<Add>(creation_context);

  post_attention_add_norm_ =
      std::make_shared<AddNorm>(layer_prefix + ".post_attention_layernorm.weight",
                                model_creation_config.layernorm_config.layernorm_eps, creation_context);

  if (is_moe) {
    shared_mlp_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config,
                                                  ".mlp.shared_expert.{}.weight");
    expert_gate_ = std::make_shared<Linear>(layer_prefix + ".mlp.gate.weight", creation_context,
                                            model_creation_config.attn_config.model_config.quant_config.backend);
    if (model_creation_config.attn_config.model_config.moe_config.use_e_score_correction_bias) {
      moe_ = std::make_shared<MoE>(
          layer_prefix + ".mlp.experts.up_gate_proj.weight", layer_prefix + ".mlp.experts.down_proj.weight",
          layer_prefix + ".mlp.gate.e_score_correction_bias", creation_context, moe_scale_norm_mode);
    } else {
      moe_ =
          std::make_shared<MoE>(layer_prefix + ".mlp.experts.up_gate_proj.weight",
                                layer_prefix + ".mlp.experts.down_proj.weight", creation_context, moe_scale_norm_mode);
    }
    // Expert parallel.
    ep_data_transfer_ = std::make_shared<ExpertParallelDataTransfer>();
    if (ep_data_transfer_ == nullptr)
      KLLM_LOG_ERROR << "Create ExpertParallelDataTransfer object( ep_data_transfer_ ) failed ";
    else
      KLLM_LOG_INFO << "Create ExpertParallelDataTransfer object( ep_data_transfer_ ) succeed ";

  } else {
    mlp_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config);
  }

  // mla should be init after linear, because mla will reuse workspace buffer which is created by linear layers
  mla_ = std::make_shared<MultiHeadLatentAttention>(layer_idx, is_neox, creation_context, model_creation_config,
                                                    mla_buffers_);

  tp_comm_ = std::make_shared<TpCommunicator>();
}

Status DeepSeekV3DecoderLayer::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                       ForwardingContext& forwarding_context, bool need_add_residual_before_attn,
                                       bool need_add_residual_after_mlp) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);

  if (need_add_residual_before_attn) {  // Adding the residual should have been done after mlp in the previous layer for
                                        // better performance.
    pre_attention_add_norm_->Forward({hidden_buffer_tensors_0[0], residual_buffer[0]}, hidden_buffer_tensors_0);
  } else {
    input_layernorm_->Forward(residual_buffer, hidden_buffer_tensors_0);
  }

  // Mla
  mla_->AcquireBuffers(forwarding_context);
  mla_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, tp_comm_, is_multi_token_forward, forwarding_context);
  mla_->ReleaseBuffers();
  // Mla all reduce
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  post_attention_add_norm_->Forward({hidden_buffer_tensors_0[0], residual_buffer[0]}, hidden_buffer_tensors_0);

  // Common mlp/moe
  AcquireMoeBuffers(forwarding_context);
  STATUS_CHECK_RETURN(
      CommonMlp(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  ReleaseMoeBuffers();
  // Mlp/moe all reduce
  if (!enable_full_shared_expert_ || !is_moe_) {
    PROFILE_EVENT_SCOPE(DS_CommonAllReduce, "DS_CommonAllReduce", forwarding_context.GetCurrentRank());
    tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);
  }

  if (is_moe_ && forwarding_context.GetContext()->GetExpertParallelWorldSize() > 1) {
    ep_data_transfer_->FreeHiddenUnitDeviceBuffer(forwarding_context);
  }

  // Mlp/moe residual add
  // need_add_residual_after_mlp==false: residual is expected to be added before attn in the next layer for better
  // performance.
  if (need_add_residual_after_mlp) {
    STATUS_CHECK_RETURN(add_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));
  }
  return Status();
}

Status DeepSeekV3DecoderLayer::CommonMlp(std::vector<Tensor>& hidden_buffer_tensors_0,
                                         std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                         ForwardingContext& forwarding_context) {
  size_t seq_len = hidden_buffer_tensors_0[0].shape[0];
  size_t hidden_units = hidden_buffer_tensors_0[0].shape[1];

  PROFILE_EVENT_SCOPE(DS_CommonMlp_seq_len_,
                      fmt::format("DS_CommonMlp_seq_len_{}_hidden_units_{}", seq_len, hidden_units),
                      forwarding_context.GetCurrentRank());

  if (!is_moe_) {
    PROFILE_EVENT_SCOPE(CommonMlp, "CommonMlp", forwarding_context.GetCurrentRank());
    mlp_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context);
  } else {
    // CREATE_BUFFER_SCOPE(moe_buffer_tensors, moe_buffer_);

    // Stage 1. Compute moe for mlp input from local nodes.
    CREATE_BUFFER_SCOPE(common_mlp_buffer_tensors, moe_buffer_);
    auto& gated_buffer_ = common_mlp_buffer_tensors;

    // Expert gating MatMul
    {
      PROFILE_EVENT_SCOPE(expert_gate, "expert_gate", forwarding_context.GetCurrentRank());
      STATUS_CHECK_RETURN(expert_gate_->Forward(hidden_buffer_tensors_0, gated_buffer_));
    }
    {
      PROFILE_EVENT_SCOPE(moe, "MOE", forwarding_context.GetCurrentRank());
      STATUS_CHECK_RETURN(moe_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], reduce_buffer_tensors));
    }

    size_t expert_node_rank = forwarding_context.GetContext()->GetExpertParallelExpertNodeRank();
    size_t world_size = forwarding_context.GetContext()->GetExpertParallelWorldSize();

    KLLM_LOG_DEBUG << fmt::format("Send moe_buffer_tensors rank_: {}, shape: {} {}, dtype: {}, refer_ptr: {}",
                                  forwarding_context.GetCurrentRank(), reduce_buffer_tensors[0].shape[0],
                                  reduce_buffer_tensors[0].shape[1], reduce_buffer_tensors[0].ToString(),
                                  reduce_buffer_tensors[0].GetPtr<void>());

    // Expert parallel.
    if (world_size > 1) {
      // Stage 2. Send mlp input to other expert-parallel nodes or
      // Receive mlp input from other expert-parallel nodes for computing moe.

      std::vector<Tensor> remote_mlp_input_compute;
      for (size_t node_rank = 0; node_rank < world_size; node_rank++) {
        if (node_rank == expert_node_rank) {
          // Stage 2.1 Send mlp input to other expert-parallel nodes.

          ep_data_transfer_->SendHiddenUnitBufferForEP(hidden_buffer_tensors_0, forwarding_context, true);
          KLLM_LOG_DEBUG << "Stage 2. Send mlp input for moe computing finished";

        } else {
          // Stage 2.2 Receive mlp input from other expert-parallel nodes.
          std::vector<Tensor>& remote_mlp_input_compute =
              ep_data_transfer_->RecvHiddenUnitBufferForEP(forwarding_context);

          if (!remote_mlp_input_compute.empty()) {
            moe_queue_in_.push_back(remote_mlp_input_compute);

          } else {
            KLLM_LOG_DEBUG << "Stage 2.2. not recv mlp input of remote expert parallel nodes";
          }

          KLLM_LOG_DEBUG << "Stage 2. Receive mlp input for moe computing finished";
        }
      }
      KLLM_LOG_DEBUG << "Stage 2. Send and receive mlp input for moe computing finished";

      // Stage 3. Compute moe for mlp input of other nodes and send results back. Only support two ep nodes now.
      for (std::vector<Tensor>& remote_mlp_input : moe_queue_in_) {
        // remote_mlp_input_temp.push_back(remote_mlp_input[0]);
        KLLM_LOG_DEBUG << "Recv remote_mlp_input shape: " << remote_mlp_input[0].shape[0]
                       << remote_mlp_input[0].shape[1] << ", dtype: " << remote_mlp_input[0].ToString()
                       << ", refer_ptr " << remote_mlp_input[0].GetPtr<void>()
                       << ", rank: " << forwarding_context.GetCurrentRank();
        // Expert gating MatMul
        {
          PROFILE_EVENT_SCOPE(gate, "gate", forwarding_context.GetCurrentRank());
          STATUS_CHECK_RETURN(expert_gate_->Forward(remote_mlp_input, gated_buffer_));
        }
        {
          PROFILE_EVENT_SCOPE(MOE, "MOE", forwarding_context.GetCurrentRank());
          STATUS_CHECK_RETURN(moe_->Forward(remote_mlp_input[0], gated_buffer_[0], common_mlp_buffer_tensors));
        }
      }

      // Clear finished tasks.
      moe_queue_in_.clear();

      // Stage 4 Send and Receive moe results.
      std::vector<Tensor> remote_residual_buffer_result;
      for (size_t node_rank = 0; node_rank < world_size; node_rank++) {
        if (node_rank == expert_node_rank) {
          // Stage 4.1 Send moe results to other nodes.
          ep_data_transfer_->SendHiddenUnitBufferForEP(common_mlp_buffer_tensors, forwarding_context, true);
          KLLM_LOG_DEBUG << fmt::format("Send moe_buffer_tensors rank_: {},  shape: {} {}, dtype: {}, refer_ptr: {}",
                                        forwarding_context.GetCurrentRank(), common_mlp_buffer_tensors[0].shape[0],
                                        common_mlp_buffer_tensors[0].shape[1], common_mlp_buffer_tensors[0].ToString(),
                                        common_mlp_buffer_tensors[0].GetPtr<void>());

        } else {
          // Stage 4.2 Revc moe results from other expert-parallel nodes.
          remote_residual_buffer_result = ep_data_transfer_->RecvHiddenUnitBufferForEP(forwarding_context);

          KLLM_LOG_DEBUG << fmt::format("Recv moe_buffer_tensors rank_: {}, shape: {} {}, dtype: {}, refer_ptr: {}",
                                        forwarding_context.GetCurrentRank(), remote_residual_buffer_result[0].shape[0],
                                        remote_residual_buffer_result[0].shape[1],
                                        remote_residual_buffer_result[0].ToString(),
                                        remote_residual_buffer_result[0].GetPtr<void>());
        }
      }

      // Stage5. Combine local and remote moe results.
      STATUS_CHECK_RETURN(
          add_->Forward(reduce_buffer_tensors[0], remote_residual_buffer_result[0], reduce_buffer_tensors));
    }

    {
      PROFILE_EVENT_SCOPE(CommonShareMlp, "CommonShareMlp", forwarding_context.GetCurrentRank());
      STATUS_CHECK_RETURN(shared_mlp_->Forward(hidden_buffer_tensors_0, common_mlp_buffer_tensors,
                                               is_multi_token_forward, forwarding_context));
    }

    if (enable_full_shared_expert_) {
      PROFILE_EVENT_SCOPE(DS_CommonAllReduce, "DS_CommonAllReduce", forwarding_context.GetCurrentRank());
      tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);
    }

    /*
     * Add moe output and share_expert output:
     *
     * When model_communicator is False, which means device_num = 1:
     *     MoeOutput          saved in reduce_buffer_tensors
     *     SharedExpertOutput saved in hidden_buffer_tensors_0
     *     CommonMlp Output = reduce_buffer_tensors + hidden_buffer_tensors_0
     *
     * When model_communicator is True, and enable_full_shared_expert_ is False:
     *     MoeOutput          saved in reduce_buffer_tensors
     *     SharedExpertOutput saved in common_mlp_buffer_tensors
     *     CommonMlp Output = reduce_buffer_tensors + common_mlp_buffer_tensors
     *
     * When model_communicator is True, and enable_full_shared_expert_ is True:
     *     MoeOutput          saved in hidden_buffer_tensors_0
     *     SharedExpertOutput saved in common_mlp_buffer_tensors
     *     CommonMlp Output = hidden_buffer_tensors_0 + common_mlp_buffer_tensors
     */
    if (forwarding_context.GetModelCommunicator()) {
      PROFILE_EVENT_SCOPE(add_layer, "add_layer_with_comm", forwarding_context.GetCurrentRank());
      if (enable_full_shared_expert_) {
        STATUS_CHECK_RETURN(
            add_->Forward(hidden_buffer_tensors_0[0], common_mlp_buffer_tensors[0], hidden_buffer_tensors_0));
      } else {
        STATUS_CHECK_RETURN(
            add_->Forward(reduce_buffer_tensors[0], common_mlp_buffer_tensors[0], reduce_buffer_tensors));
      }
    } else {
      PROFILE_EVENT_SCOPE(add_layer, "add_layer", forwarding_context.GetCurrentRank());
      STATUS_CHECK_RETURN(add_->Forward(hidden_buffer_tensors_0[0], reduce_buffer_tensors[0], hidden_buffer_tensors_0));
    }
  }

  return Status();
}

void DeepSeekV3DecoderLayer::AcquireMoeBuffers(ForwardingContext& forwarding_context) {
  // TODO(yancyliu): Get tensor from moe_buffer_
  // Reset its shape from batch_size and token_num, and then allocate tensor memory.
}

void DeepSeekV3DecoderLayer::ReleaseMoeBuffers() {
  // TODO(yancyliu): Get tensor from moe_buffer_, and then release its memory.
}

DeepSeekV3MtpLayer::DeepSeekV3MtpLayer(const int layer_idx, LayerCreationContext& creation_context,
                                       ModelCreationConfig& model_creation_config,
                                       std::shared_ptr<DeepSeekV3DecoderLayer> decoder_layer) {
  enorm_ = std::make_shared<Layernorm>(fmt::format("model.layers.{}.enorm.weight", layer_idx),
                                       model_creation_config.attn_config.model_config.layernorm_eps, creation_context);
  hnorm_ = std::make_shared<Layernorm>(fmt::format("model.layers.{}.hnorm.weight", layer_idx),
                                       model_creation_config.attn_config.model_config.layernorm_eps, creation_context);

  concat_layer_ = std::make_shared<ConcatLayer>();
  concat_layer_->Init({size_t{1}}, creation_context.runtime_config, creation_context.context, creation_context.rank);

  gather_layer_ = std::make_shared<AssembleTokensHiddenLayer>();
  gather_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);

  emb_lookup_layer_ = std::make_shared<EmbLookupLayer>();
  emb_lookup_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);

  eh_proj_ = std::make_shared<Linear>(fmt::format("model.layers.{}.eh_proj.weight", layer_idx), creation_context,
                                      model_creation_config.attn_config.model_config.quant_config.backend);

  decoder_layer_ = decoder_layer;

  tp_comm_ = std::make_shared<TpCommunicator>();
}

Status DeepSeekV3MtpLayer::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, "MTP", RequestEventPhase::Begin);
  auto& mtp_hidden_buffer = forwarding_context.GetForwardingBuffers()->mtp_hidden_buffer_tensors;
  const auto& model_input = forwarding_context.GetModelInput();
  {
    CREATE_BUFFER_SCOPE(shared_buffer, forwarding_context.GetForwardingBuffers()->shared_buffer);

    // Embedding Norm
    enorm_->Forward(residual_buffer, residual_buffer);

    // gather last token hidden
    STATUS_CHECK_RETURN(gather_layer_->Forward(
        {mtp_hidden_buffer[0], forwarding_context.GetModelInput()->nextn_hidden_idx_uint64_tensor}, shared_buffer));

    // last token hidden norm
    hnorm_->Forward(shared_buffer, mtp_hidden_buffer);

    // concat embedding_norm and hidden norm
    concat_layer_->Forward({residual_buffer[0], mtp_hidden_buffer[0]}, shared_buffer);

    // linear, no bias. hidden_units * 2 -> hidden_units
    STATUS_CHECK_RETURN(eh_proj_->Forward(shared_buffer[0], residual_buffer));
    tp_comm_->AllGather(residual_buffer[0], mtp_hidden_buffer[0], forwarding_context);
  }
  const bool is_multi_token_forward = model_input->multi_token_request_num > 0;
  STATUS_CHECK_RETURN(decoder_layer_->Forward(residual_buffer, is_multi_token_forward, forwarding_context,
                                              /* need_add_residual_before_attn */ false,
                                              /* need_add_residual_after_mlp */ true));

  StreamSynchronize(forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()]);
  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, "MTP", RequestEventPhase::End);
  return Status();
}

/**********************************************************
 * DeepSeekV3Model
 ***********************************************************/

DeepSeekV3Model::DeepSeekV3Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                                 std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight)
    : CommonModel(model_config, runtime_config, rank, context),
      first_k_dense_replace_(model_config.moe_config.first_k_dense_replace) {
  ModelRunConfig model_run_config;
  model_run_config.position_encoding = PositionEncoding::ROPE;

  CommonModel::InitRunConfig(model_run_config, base_weight);
}

Status DeepSeekV3Model::CreateLayers(LayerCreationContext& creation_context,
                                     ModelCreationConfig& model_creation_config) {
  MultiHeadLatentAttention::CreateBuffers(CommonModel::GetBufferManager(), model_creation_config.attn_config,
                                          creation_context.runtime_config, mla_buffers_);
  const DataType weight_type = model_creation_config.attn_config.model_config.weight_data_type;
  const size_t max_token_num = creation_context.runtime_config.max_step_token_num;
  size_t moe_buffer_size = max_token_num * model_creation_config.attn_config.model_config.hidden_units;
  // Used for TwoLayeredFFN
  moe_buffer_size = std::max(
      moe_buffer_size, max_token_num * model_creation_config.attn_config.model_config.moe_config.moe_inter_size * 2);
  if (creation_context.runtime_config.enable_full_shared_expert) {
    moe_buffer_size = std::max(
        moe_buffer_size, max_token_num * model_creation_config.attn_config.model_config.moe_config.moe_inter_size *
                             model_creation_config.attn_config.model_config.moe_config.num_shared_experts);
    KLLM_LOG_DEBUG << fmt::format("when using enable_full_shared_expert, moe_buffer_size = {} * max({}, {} * {})",
                                  max_token_num, model_creation_config.attn_config.model_config.hidden_units,
                                  model_creation_config.attn_config.model_config.moe_config.moe_inter_size,
                                  model_creation_config.attn_config.model_config.moe_config.num_shared_experts);
  }
  moe_buffer_ = CommonModel::GetBufferManager()->CreateBufferTensor("moe_buffer_", {moe_buffer_size}, weight_type);

  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; ++layer_idx) {
    const bool is_moe = layer_idx >= first_k_dense_replace_;
    layers_[layer_idx] = std::make_shared<DeepSeekV3DecoderLayer>(layer_idx, is_moe, creation_context,
                                                                  model_creation_config, mla_buffers_, moe_buffer_);
  }

  if (creation_context.pipeline_config.lower_nextn_layer_idx >=
      static_cast<int>(model_creation_config.attn_config.model_config.num_layer)) {
    for (int layer_idx = creation_context.pipeline_config.lower_nextn_layer_idx;
         layer_idx <= creation_context.pipeline_config.upper_nextn_layer_idx; ++layer_idx) {
      const bool is_moe = layer_idx >= first_k_dense_replace_;
      layers_[layer_idx] = std::make_shared<DeepSeekV3DecoderLayer>(layer_idx, is_moe, creation_context,
                                                                    model_creation_config, mla_buffers_, moe_buffer_);

      // create nextn layer, give decoder layer
      nextn_layers_[layer_idx] =
          std::make_shared<DeepSeekV3MtpLayer>(layer_idx, creation_context, model_creation_config, layers_[layer_idx]);
    }
  }
  return Status();
}

Status DeepSeekV3Model::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  PROFILE_EVENT_SCOPE(DS_LayerForward_, fmt::format("DS_LayerForward_{}", forwarding_context.GetMultiBatchId()),
                      forwarding_context.GetCurrentRank());
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  const bool need_recv = !forwarding_context.GetContext()->IsChief() && run_mode == RunMode::kMain;

  if (run_mode == RunMode::kMain) {
    std::vector<Tensor>& residual_buffer = GetHiddenUnitBuffer(forwarding_context, need_recv);
    for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
         layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
      STATUS_CHECK_RETURN(layers_[layer_idx]->Forward(
          residual_buffer, is_multi_token_forward, forwarding_context,
          /* need_add_residual_before_attn */ layer_idx != forwarding_context.GetPipelineConfig().lower_layer_idx,
          /* need_add_residual_after_mlp */ layer_idx == forwarding_context.GetPipelineConfig().upper_layer_idx));
    }
    SetHiddenUnitBuffer(residual_buffer, forwarding_context);
  } else if (run_mode == RunMode::kNextN && !nextn_layers_.empty()) {
    forwarding_context.SetIsForwardingLayers(false);  // Don't record ForwardingLayers event
    std::vector<Tensor>& residual_buffer = GetHiddenUnitBuffer(forwarding_context, need_recv);
    for (int layer_idx = forwarding_context.GetPipelineConfig().lower_nextn_layer_idx;
         layer_idx <= forwarding_context.GetPipelineConfig().upper_nextn_layer_idx; ++layer_idx) {
      STATUS_CHECK_RETURN(nextn_layers_[layer_idx]->Forward(residual_buffer, forwarding_context));
    }
  }
  return Status();
}

}  // namespace ksana_llm
