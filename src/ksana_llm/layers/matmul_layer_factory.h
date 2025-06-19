/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/batched_matmul_layer.h"
#include "ksana_llm/layers/blockwise_matmul_layer.h"
#include "ksana_llm/layers/cutlass_matmul_layer.h"
#include "ksana_llm/layers/fp8_matmul_layer.h"
#include "ksana_llm/layers/fp8_moe_layer.h"
#include "ksana_llm/layers/machete_matmul_layer.h"
#include "ksana_llm/layers/marlin_matmul_layer.h"
#include "ksana_llm/layers/matmul_layer.h"
#include "ksana_llm/layers/moe_layer.h"
#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

template <typename T>
class MatMulLayerFactory {
 public:
  typedef std::shared_ptr<BaseLayer> (MatMulLayerFactory<T>::*BuildLayerFunc)();
  MatMulLayerFactory(std::shared_ptr<Tensor>& workspace_buffer, const ModelConfig& model_config, const int rank,
                     std::shared_ptr<Context> context) {
    context_ = context;
    rank_ = rank;
    model_config_ = model_config;
    workspace_buffer_ = workspace_buffer;

    builder_map_[{TYPE_FP32, TYPE_FP32, TYPE_FP32, QUANT_NONE, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<MatMulLayer<T>>;
    builder_map_[{TYPE_FP16, TYPE_FP16, TYPE_FP16, QUANT_NONE, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<MatMulLayer<T>>;
    builder_map_[{TYPE_BF16, TYPE_BF16, TYPE_BF16, QUANT_NONE, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<MatMulLayer<T>>;

    builder_map_[{TYPE_VOID, TYPE_FP32, TYPE_FP32, QUANT_NONE, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<BatchedMatMulLayer<T>>;
    builder_map_[{TYPE_VOID, TYPE_FP16, TYPE_FP16, QUANT_NONE, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<BatchedMatMulLayer<T>>;
    builder_map_[{TYPE_VOID, TYPE_BF16, TYPE_BF16, QUANT_NONE, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<BatchedMatMulLayer<T>>;

#ifdef ENABLE_CUDA
    builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_GPTQ, CUTLASS_BACKEND}] =
        &MatMulLayerFactory<T>::BuildLayer<CutlassMatMulLayer<T, TYPE_I4_GROUP>>;
    builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_AWQ, CUTLASS_BACKEND}] =
        &MatMulLayerFactory<T>::BuildLayer<CutlassMatMulLayer<T, TYPE_I4_GROUP>>;

    builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_GPTQ, MARLIN_BACKEND}] =
        &MatMulLayerFactory<T>::BuildLayer<MarlinMatMulLayer<T, TYPE_I4_GROUP>>;
    builder_map_[{TYPE_I4_GROUP, TYPE_BF16, TYPE_BF16, QUANT_GPTQ, MARLIN_BACKEND}] =
        &MatMulLayerFactory<T>::BuildLayer<MarlinMatMulLayer<T, TYPE_I4_GROUP>>;
    builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_AWQ, MARLIN_BACKEND}] =
        &MatMulLayerFactory<T>::BuildLayer<MarlinMatMulLayer<T, TYPE_I4_GROUP>>;

    builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_GPTQ, MACHETE_BACKEND}] =
        &MatMulLayerFactory<T>::BuildLayer<MacheteMatMulLayer<T, TYPE_I4_GROUP>>;
    builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_AWQ, MACHETE_BACKEND}] =
        &MatMulLayerFactory<T>::BuildLayer<MacheteMatMulLayer<T, TYPE_I4_GROUP>>;
    builder_map_[{TYPE_I4_GROUP, TYPE_BF16, TYPE_BF16, QUANT_GPTQ, MACHETE_BACKEND}] =
        &MatMulLayerFactory<T>::BuildLayer<MacheteMatMulLayer<T, TYPE_I4_GROUP>>;
    builder_map_[{TYPE_I4_GROUP, TYPE_BF16, TYPE_BF16, QUANT_AWQ, MACHETE_BACKEND}] =
        &MatMulLayerFactory<T>::BuildLayer<MacheteMatMulLayer<T, TYPE_I4_GROUP>>;

    if (model_config_.is_moe) {
      // TODO(winminkong): Organize the quantization backend and quantization types of the MoE layer.
      builder_map_[{TYPE_FP32, TYPE_FP32, TYPE_FP32, MOE_QUANT_NONE, NONE_QUANT}] =
          &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;
      builder_map_[{TYPE_FP16, TYPE_FP16, TYPE_FP16, MOE_QUANT_NONE, NONE_QUANT}] =
          &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;
      builder_map_[{TYPE_BF16, TYPE_BF16, TYPE_BF16, MOE_QUANT_NONE, NONE_QUANT}] =
          &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;
      // for fused_moe_gtpq_triton
      builder_map_[{TYPE_UINT8, TYPE_FP16, TYPE_FP16, MOE_QUANT_GTPQ, NONE_QUANT}] =
          &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;
      builder_map_[{TYPE_UINT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_GTPQ, NONE_QUANT}] =
          &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;

#  ifdef ENABLE_FP8
      builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, MOE_QUANT_BLOCK_FP8_E4M3, NONE_QUANT}] =
          &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;
      builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, MOE_QUANT_BLOCK_FP8_E4M3, NONE_QUANT}] =
          &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;
      builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, MOE_QUANT_BLOCK_FP8_E4M3, NONE_QUANT}] =
          &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;
#  endif
    }
#endif
#ifdef ENABLE_FP8
    builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, QUANT_FP8_E4M3, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<Fp8MatMulLayer<T>>;
    builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, QUANT_FP8_E4M3, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<Fp8MatMulLayer<T>>;
    builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, QUANT_FP8_E4M3, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<Fp8MatMulLayer<T>>;

    builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, QUANT_BLOCK_FP8_E4M3, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<BlockwiseMatMulLayer<T>>;
    builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, QUANT_BLOCK_FP8_E4M3, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<BlockwiseMatMulLayer<T>>;
    builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, QUANT_BLOCK_FP8_E4M3, NONE_QUANT}] =
        &MatMulLayerFactory<T>::BuildLayer<BlockwiseMatMulLayer<T>>;
    if (model_config_.is_moe) {
      if constexpr (!std::is_same_v<T, float>) {
        builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, MOE_QUANT_FP8_E4M3, NONE_QUANT}] =
            &MatMulLayerFactory<T>::BuildLayer<Fp8MoeLayer<T, fp8e4m3>>;
        builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, MOE_QUANT_FP8_E4M3, NONE_QUANT}] =
            &MatMulLayerFactory<T>::BuildLayer<Fp8MoeLayer<T, fp8e4m3>>;
        builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, MOE_QUANT_FP8_E4M3, NONE_QUANT}] =
            &MatMulLayerFactory<T>::BuildLayer<Fp8MoeLayer<T, fp8e4m3>>;
      }
    }
#endif
  }

  ~MatMulLayerFactory() {}

  template <typename ClassT>
  std::shared_ptr<BaseLayer> BuildLayer() {
    return std::make_shared<ClassT>();
  }
  std::shared_ptr<BaseLayer> AutoCreateLayer(std::shared_ptr<BaseWeight> base_weight, std::string weight_name,
                                             DataType weight_type, DataType input_type, DataType output_type,
                                             GroupQuantBackend backend, const std::vector<std::any>& init_params) {
    // gptq layer
    if (model_config_.is_quant &&
        (model_config_.quant_config.method == QUANT_GPTQ || model_config_.quant_config.method == QUANT_AWQ)) {
      size_t tp = model_config_.tensor_para_size;
      size_t hidden_size = model_config_.hidden_units;
      size_t inter_size = model_config_.inter_size;
      size_t shared_expert_inter_size_per_rank = model_config_.enable_full_shared_expert
                                                     ? model_config_.moe_config.shared_expert_inter_size
                                                     : model_config_.moe_config.shared_expert_inter_size / tp;
      uint32_t qk_rope_head_dim = model_config_.mla_config.qk_rope_head_dim;
      uint32_t qk_nope_head_dim = model_config_.mla_config.qk_nope_head_dim;
      uint32_t q_lora_rank = model_config_.mla_config.q_lora_rank;
      uint32_t kv_lora_rank = model_config_.mla_config.kv_lora_rank;
      uint32_t v_head_dim = model_config_.mla_config.v_head_dim;
      size_t head_num = model_config_.head_num;
      // The inter size in config.json for the qwen1 model is twice the true inter size.
      if (model_config_.type == "qwen") {
        inter_size /= 2;
      }
      size_t qkv_size = model_config_.size_per_head * (model_config_.head_num + 2 * model_config_.num_key_value_heads);
      // Because the layout convertion, we can't get n/k from weight shape, and have to calculate it.
      std::map<std::string, std::tuple<size_t, size_t, bool>> kn_pairs;
      kn_pairs["query_key_value"] = std::make_tuple(hidden_size, qkv_size / tp, true);
      kn_pairs["o_proj"] = std::make_tuple(hidden_size / tp, hidden_size, false);
      kn_pairs["mlp.gate_proj"] = std::make_tuple(hidden_size, inter_size / tp, true);
      kn_pairs["mlp.up_proj"] = kn_pairs["mlp.gate_proj"];
      kn_pairs["mlp.down_proj"] = std::make_tuple(inter_size / tp, hidden_size, false);
      kn_pairs["mlp.shared_expert.gate_proj"] = std::make_tuple(hidden_size, shared_expert_inter_size_per_rank, true);
      kn_pairs["mlp.shared_expert.up_proj"] = kn_pairs["mlp.shared_expert.gate_proj"];
      kn_pairs["mlp.shared_expert.down_proj"] = std::make_tuple(shared_expert_inter_size_per_rank, hidden_size, false);
      kn_pairs["q_a_proj"] = std::make_tuple(hidden_size, q_lora_rank, false);
      kn_pairs["kv_a_lora_proj"] = std::make_tuple(hidden_size, kv_lora_rank, false);
      kn_pairs["kv_a_rope_proj"] = std::make_tuple(hidden_size, qk_rope_head_dim, false);
      kn_pairs["q_b_nope_proj"] = std::make_tuple(q_lora_rank, head_num / tp * qk_nope_head_dim, true);
      kn_pairs["q_b_rope_proj"] = std::make_tuple(q_lora_rank, head_num / tp * qk_rope_head_dim, true);
      kn_pairs["v_head_proj"] = std::make_tuple(kv_lora_rank, head_num / tp * v_head_dim, true);
      for (const auto& kn : kn_pairs) {
        if (weight_name.find(kn.first) != std::string::npos) {
          std::vector<std::any> group_matmul_param;
          group_matmul_param.push_back(model_config_.max_step_token_num);                                   // m
          group_matmul_param.push_back(std::get<1>(kn.second));                                             // n
          group_matmul_param.push_back(std::get<0>(kn.second));                                             // k
          group_matmul_param.push_back(model_config_.quant_config.group_size);                              // groupsize
          group_matmul_param.push_back(static_cast<bool>(model_config_.quant_config.method == QUANT_AWQ));  // awq
          group_matmul_param.push_back(static_cast<bool>(model_config_.quant_config.desc_act));             // gptq desc
          group_matmul_param.push_back(static_cast<bool>(std::get<2>(kn.second)));                          // k full
          group_matmul_param.push_back(true);                                                               // cuda gemv
          if (weight_name.find("kv_a_rope_proj") != std::string::npos) {
            return CreateLayer(TYPE_I4_GROUP, input_type, output_type, group_matmul_param, QUANT_GPTQ, MARLIN_BACKEND);
          }
          return CreateLayer(TYPE_I4_GROUP, input_type, output_type, group_matmul_param, QUANT_GPTQ, backend);
        }
      }
    }
    // fp8 layer
    if (base_weight->GetModelWeights(weight_name).dtype == TYPE_FP8_E4M3) {
      if (model_config_.quant_config.method == QUANT_BLOCK_FP8_E4M3) {
        std::vector<std::any> fp8_blockwise_matmul_params;
        fp8_blockwise_matmul_params.push_back(model_config_.max_step_token_num);  // m
        // weight is [n， k]
        fp8_blockwise_matmul_params.push_back(size_t(base_weight->GetModelWeights(weight_name).shape[0]));  // n
        fp8_blockwise_matmul_params.push_back(size_t(base_weight->GetModelWeights(weight_name).shape[1]));  // k
        // block_k size
        fp8_blockwise_matmul_params.push_back(model_config_.quant_config.weight_block_size[1]);
        fp8_blockwise_matmul_params.push_back(model_config_.tensor_para_size);

        // weight is [n， k], k is shape[1],
        if (rank_ == 0) {
          KLLM_LOG_INFO << fmt::format("rockcao weight_name: {}, weight_shape: {}", weight_name,
                                       base_weight->GetModelWeights(weight_name).ToString());
        }

        return CreateLayer(TYPE_FP8_E4M3, input_type, output_type, fp8_blockwise_matmul_params, QUANT_BLOCK_FP8_E4M3,
                           NONE_QUANT);
      } else {
        std::vector<std::any> fp8_matmul_params;
        // max_m_
        fp8_matmul_params.push_back(model_config_.max_step_token_num);
        // weight is [n, k], k is shape[1]
        fp8_matmul_params.push_back(size_t(base_weight->GetModelWeights(weight_name).shape[1]));
        return CreateLayer(TYPE_FP8_E4M3, input_type, output_type, fp8_matmul_params, QUANT_FP8_E4M3, NONE_QUANT);
      }
    }
    // batched matmul has no weight
    if (weight_name == "" && weight_type == TYPE_VOID) {
      return CreateLayer(weight_type, input_type, output_type, init_params, QUANT_NONE, NONE_QUANT);
    }
    // default layer
    return CreateLayer(base_weight, weight_name, input_type, output_type, init_params, QUANT_NONE, NONE_QUANT);
  }

  std::shared_ptr<BaseLayer> AutoCreateMoeLayer(std::shared_ptr<BaseWeight> base_weight,
                                                std::vector<std::string> weight_names, DataType weight_type,
                                                DataType input_type, DataType output_type,
                                                const std::vector<std::any>& init_params) {
    // moe layer   (weight_names[0]: up_gate_experts, weight_names[1]: down_experts)
    std::vector<std::any> moe_matmul_param = init_params;
    moe_matmul_param.push_back(model_config_.max_step_token_num);
    size_t up_gate_experts_num = base_weight->GetModelWeights(weight_names[0]).shape[0];
    size_t down_experts_num = base_weight->GetModelWeights(weight_names[1]).shape[0];
    if (up_gate_experts_num != down_experts_num) {
      KLLM_THROW(fmt::format("Moe Weights Load Error: up_gate experts {} and down_experts {} should should be equal",
                             up_gate_experts_num, down_experts_num));
    }
    moe_matmul_param.push_back(model_config_.moe_config.num_experts /
                               (model_config_.expert_para_size * model_config_.expert_world_size));  // num_experts
    size_t up_gate_hidden_size = base_weight->GetModelWeights(weight_names[0]).shape[2];
    size_t down_hidden_size = base_weight->GetModelWeights(weight_names[1]).shape[1];
    if (model_config_.quant_config.method == QUANT_GPTQ) {
      up_gate_hidden_size = up_gate_hidden_size / 4 * (32 / model_config_.quant_config.bits);
    }
    if (up_gate_hidden_size != down_hidden_size) {
      KLLM_THROW(
          fmt::format("Moe Weights Load Error: up_gate_experts hidden_size {} and down_experts hidden_size {} should "
                      "should be equal",
                      up_gate_hidden_size, down_hidden_size));
    }

    size_t hidden_size = static_cast<size_t>(model_config_.hidden_units);
    size_t moe_inter_size_per_rank =
        static_cast<size_t>(DivRoundUp(model_config_.moe_config.moe_inter_size, model_config_.moe_tensor_para_size));
    moe_matmul_param.push_back(hidden_size);                                           // hidden_size
    moe_matmul_param.push_back(moe_inter_size_per_rank);                               // Inter_size
    moe_matmul_param.push_back(model_config_.moe_config.experts_topk);                 // experts topk
    moe_matmul_param.push_back(model_config_.tensor_para_size);                        // TP_size
    moe_matmul_param.push_back(model_config_.moe_config.use_vllm_moe);                 // use_vllm_moe
    moe_matmul_param.push_back(model_config_.moe_config.num_expert_group);             // num_expert_group
    moe_matmul_param.push_back(model_config_.moe_config.expert_groups_topk);           // expert_groups_topk
    moe_matmul_param.push_back(model_config_.moe_config.scoring_func);                 // scoring_func
    moe_matmul_param.push_back(model_config_.moe_config.topk_method);                  // topk_method
    moe_matmul_param.push_back(model_config_.moe_config.norm_topk_prob);               // norm_topk_prob
    moe_matmul_param.push_back(model_config_.moe_config.routed_scaling_factor);        // routed_scaling_factor
    moe_matmul_param.push_back(model_config_.moe_config.use_e_score_correction_bias);  // use_e_score_correction_bias
    moe_matmul_param.push_back(model_config_.quant_config.is_fp8_blockwise ? DataType::TYPE_BLOCK_FP8_E4M3
                                                                           : DataType::TYPE_INVALID);
    if (model_config_.quant_config.method == QUANT_GPTQ && model_config_.quant_config.bits == 4) {
      moe_matmul_param.push_back(DataType::TYPE_I4_GROUP);
    } else {
      moe_matmul_param.push_back(DataType::TYPE_INVALID);
    }
    moe_matmul_param.push_back(model_config_.moe_config.apply_weight);

    weight_type = base_weight->GetModelWeights(weight_names[0]).dtype;
    DataType down_weight_type = base_weight->GetModelWeights(weight_names[1]).dtype;
    if (down_weight_type != weight_type) {
      KLLM_THROW(fmt::format(
          "Moe Weights Load Error: down_experts dtype {} and up_gate_experts dtype {} should have same dtype",
          down_weight_type, weight_type));
    }
    if (weight_type == TYPE_FP8_E4M3) {
      if (model_config_.quant_config.is_fp8_blockwise) {
        return CreateLayer(TYPE_FP8_E4M3, input_type, output_type, moe_matmul_param, MOE_QUANT_BLOCK_FP8_E4M3,
                           NONE_QUANT);
      } else {
        return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_FP8_E4M3, NONE_QUANT);
      }
    }
    if (weight_type == TYPE_UINT8 && model_config_.quant_config.method == QUANT_GPTQ) {
      return CreateLayer(TYPE_UINT8, input_type, output_type, moe_matmul_param, MOE_QUANT_GTPQ, NONE_QUANT);
    }
    return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_NONE, NONE_QUANT);
  }

  std::shared_ptr<BaseLayer> CreateLayer(std::shared_ptr<BaseWeight> base_weight, std::string weight_name,
                                         DataType input_type, DataType output_type,
                                         const std::vector<std::any>& init_params, QuantMode quant_mode = QUANT_NONE,
                                         GroupQuantBackend backend = NONE_QUANT) {
    KLLM_LOG_DEBUG << fmt::format("weight_name: {}", weight_name);
    DataType weight_type = base_weight->GetModelWeights(weight_name).dtype;
    // deepseek v3
    if (weight_type == TYPE_INVALID) {
      weight_type = input_type;
    }
    return CreateLayer(weight_type, input_type, output_type, init_params, quant_mode, backend);
  }
  std::shared_ptr<BaseLayer> CreateLayer(DataType weight_type, DataType input_type, DataType output_type,
                                         const std::vector<std::any>& init_params, QuantMode quant_mode = QUANT_NONE,
                                         GroupQuantBackend backend = NONE_QUANT) {
    auto it = builder_map_.find({weight_type, input_type, output_type, quant_mode, backend});
    if (it != builder_map_.end()) {
      std::shared_ptr<BaseLayer> layer = (this->*(it->second))();
      layer->Init(init_params, context_, rank_);
      size_t workspace_size = layer->GetWorkSpaceSize();
      if (workspace_buffer_ == nullptr) {
        if (workspace_size > 0) {
          KLLM_LOG_DEBUG << fmt::format("Rank[{}] Create WorkSpace Buffer: {}", rank_, workspace_size);
          workspace_buffer_ = std::shared_ptr<Tensor>(
              new Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT8, {workspace_size}, rank_));
        } else {
          KLLM_LOG_DEBUG << fmt::format("Rank[{}] No need any WorkSpace Buffer", rank_);
        }
      } else {
        if (workspace_buffer_->GetTotalBytes() < workspace_size) {
          KLLM_LOG_DEBUG << fmt::format("Rank[{}] Increase WorkSpace Buffer from: {} to: {}", rank_,
                                        workspace_buffer_->GetTotalBytes(), workspace_size);
          workspace_buffer_.reset();
          workspace_buffer_ = std::shared_ptr<Tensor>(
              new Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT8, {workspace_size}, rank_));
        } else {
          KLLM_LOG_DEBUG << fmt::format("Rank[{}] WorkSpace Buffer {} is big enough", rank_,
                                        workspace_buffer_->GetTotalBytes());
        }
      }
      layer->SetWorkSpaceBuffer(workspace_buffer_);

      layer->Preprocess(model_config_);
      KLLM_LOG_DEBUG << "return layer";
      return layer;
    } else {
      KLLM_THROW(fmt::format("Not support weight_type {}, input_type {}, output_type {}, quant_mode {}, backend {}.",
                             weight_type, input_type, output_type, quant_mode, backend));
    }
  }

  std::shared_ptr<Tensor> GetWorkspaceBuffer() const { return workspace_buffer_; }

 private:
  std::shared_ptr<Context> context_;
  int rank_;
  std::shared_ptr<Tensor> workspace_buffer_ = nullptr;
  ModelConfig model_config_;

  // std::map<std::tuple<weight_type, input_type, output_type, quant_mode, backend>, BuildLayerFunc>
  std::map<std::tuple<DataType, DataType, DataType, QuantMode, GroupQuantBackend>, BuildLayerFunc> builder_map_;
};

}  // namespace ksana_llm
