/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/moe_layer_factory.h"

#include "ksana_llm/layers/batched_matmul_layer.h"
#include "ksana_llm/layers/blockwise_matmul_layer.h"
#include "ksana_llm/layers/cutlass_matmul_layer.h"
#include "ksana_llm/layers/cutlass_moe_layer.h"
#include "ksana_llm/layers/fp8_matmul_layer.h"
#include "ksana_llm/layers/fp8_moe_layer.h"
#include "ksana_llm/layers/machete_matmul_layer.h"
#include "ksana_llm/layers/marlin_matmul_layer.h"
#include "ksana_llm/layers/marlin_moe_layer.h"
#include "ksana_llm/layers/matmul_layer.h"
#include "ksana_llm/layers/moe_layer.h"

namespace ksana_llm {

MoeLayerFactory::MoeLayerFactory(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                                 std::shared_ptr<Context> context) {
  context_ = context;
  rank_ = rank;
  model_config_ = model_config;
  runtime_config_ = runtime_config;

#ifdef ENABLE_CUDA
  // TODO(winminkong): Organize the quantization backend and quantization types of the MoE layer.
  builder_map_[{TYPE_FP32, TYPE_FP32, TYPE_FP32, MOE_QUANT_NONE, NONE_QUANT}] = &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_FP16, TYPE_FP16, TYPE_FP16, MOE_QUANT_NONE, NONE_QUANT}] = &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_BF16, TYPE_BF16, TYPE_BF16, MOE_QUANT_NONE, NONE_QUANT}] = &MoeLayerFactory::BuildLayer<MoeLayer>;

  builder_map_[{TYPE_UINT8, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, NONE_QUANT}] = &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_UINT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, NONE_QUANT}] = &MoeLayerFactory::BuildLayer<MoeLayer>;

  builder_map_[{TYPE_INT8, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, NONE_QUANT}] =
      &MoeLayerFactory::BuildLayer<CutlassMoeLayer>;
  builder_map_[{TYPE_INT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, NONE_QUANT}] =
      &MoeLayerFactory::BuildLayer<CutlassMoeLayer>;

  // for marlin gptq moe
  builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, MARLIN_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MarlinMoeLayer>;
#endif

#ifdef ENABLE_FP8
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, MOE_QUANT_BLOCK_FP8_E4M3, NONE_QUANT}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, MOE_QUANT_BLOCK_FP8_E4M3, NONE_QUANT}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, MOE_QUANT_BLOCK_FP8_E4M3, NONE_QUANT}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  if (runtime_config.inter_data_type != TYPE_FP32) {  // TODO(robertyuan): weird condition
    // NOTE: Fp8MoeLayer only suport fp8e4m3 rightnow
    builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, MOE_QUANT_FP8_E4M3, NONE_QUANT}] =
        &MoeLayerFactory::BuildLayer<Fp8MoeLayer>;
    builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, MOE_QUANT_FP8_E4M3, NONE_QUANT}] =
        &MoeLayerFactory::BuildLayer<Fp8MoeLayer>;
    builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, MOE_QUANT_FP8_E4M3, NONE_QUANT}] =
        &MoeLayerFactory::BuildLayer<Fp8MoeLayer>;
  }
#endif
}

std::shared_ptr<BaseLayer> MoeLayerFactory::AutoCreateMoeLayer(std::shared_ptr<BaseWeight> base_weight,
                                                               std::vector<std::string> weight_names,
                                                               DataType weight_type, DataType input_type,
                                                               DataType output_type,
                                                               const std::vector<std::any>& init_params) {
  // moe layer   (weight_names[0]: up_gate_experts, weight_names[1]: down_experts)
  bool use_vllm_moe = model_config_.moe_config.use_vllm_moe;
  std::vector<std::any> moe_matmul_param = init_params;
  moe_matmul_param.push_back(runtime_config_.max_step_token_num);
  size_t up_gate_experts_num = base_weight->GetModelWeights(weight_names[0]).shape[0];
  size_t down_experts_num = base_weight->GetModelWeights(weight_names[1]).shape[0];
  if (up_gate_experts_num != down_experts_num) {
    KLLM_THROW(fmt::format("Moe Weights Load Error: up_gate experts {} and down_experts {} should should be equal",
                           up_gate_experts_num, down_experts_num));
  }
  moe_matmul_param.push_back(model_config_.moe_config.num_experts /
                             (runtime_config_.parallel_basic_config.expert_parallel_size *
                              runtime_config_.parallel_basic_config.expert_world_size));  // num_experts
  size_t up_gate_hidden_size = base_weight->GetModelWeights(weight_names[0]).shape[2];
  size_t down_hidden_size = base_weight->GetModelWeights(weight_names[1]).shape[1];
  bool enable_moe_int4 = false;
  if (model_config_.quant_config.method != QUANT_GPTQ && model_config_.sub_quant_configs.size() > 0 &&
      model_config_.sub_quant_configs[0].method == QUANT_GPTQ) {
    for (std::string& pattern_layer : model_config_.sub_quant_configs[0].pattern_layers) {
      if (weight_names[0].find(pattern_layer) != std::string::npos) {
        enable_moe_int4 = true;
        break;
      }
    }
    for (std::string& ignored_layer : model_config_.sub_quant_configs[0].ignored_layers) {
      if (weight_names[0].find(ignored_layer) != std::string::npos) {
        enable_moe_int4 = false;
        break;
      }
    }
  }
  if (model_config_.quant_config.method == QUANT_GPTQ || enable_moe_int4) {
    if (use_vllm_moe) {
      up_gate_hidden_size = up_gate_hidden_size / 4 * (32 / model_config_.quant_config.bits);
    } else {  // marlin gptq
      up_gate_hidden_size =
          base_weight->GetModelWeights(weight_names[0]).shape[1] / (sizeof(int) / model_config_.quant_config.bits) * 16;
      down_hidden_size = base_weight->GetModelWeights(weight_names[1]).shape[2] / model_config_.quant_config.bits * 2;
    }
  }
  if (up_gate_hidden_size != down_hidden_size) {
    KLLM_THROW(
        fmt::format("Moe Weights Load Error: up_gate_experts hidden_size {} and down_experts hidden_size {} should "
                    "should be equal",
                    up_gate_hidden_size, down_hidden_size));
  }

  size_t hidden_size = static_cast<size_t>(model_config_.hidden_units);
  size_t moe_inter_size_per_rank = static_cast<size_t>(
      DivRoundUp(model_config_.moe_config.moe_inter_size, runtime_config_.parallel_basic_config.moe_tensor_para_size));
  moe_matmul_param.push_back(hidden_size);                                                 // hidden_size
  moe_matmul_param.push_back(moe_inter_size_per_rank);                                     // Inter_size
  moe_matmul_param.push_back(model_config_.moe_config.experts_topk);                       // experts topk
  moe_matmul_param.push_back(runtime_config_.parallel_basic_config.tensor_parallel_size);  // TP_size
  moe_matmul_param.push_back(use_vllm_moe);                                                // use_vllm_moe
  moe_matmul_param.push_back(model_config_.moe_config.num_expert_group);                   // num_expert_group
  moe_matmul_param.push_back(model_config_.moe_config.expert_groups_topk);                 // expert_groups_topk
  moe_matmul_param.push_back(model_config_.moe_config.scoring_func);                       // scoring_func
  moe_matmul_param.push_back(model_config_.moe_config.topk_method);                        // topk_method
  moe_matmul_param.push_back(model_config_.moe_config.norm_topk_prob);                     // norm_topk_prob
  moe_matmul_param.push_back(model_config_.moe_config.routed_scaling_factor);              // routed_scaling_factor
  moe_matmul_param.push_back(model_config_.moe_config.use_e_score_correction_bias);  // use_e_score_correction_bias
  if (enable_moe_int4) {
    moe_matmul_param.push_back(DataType::TYPE_INVALID);
  } else {
    moe_matmul_param.push_back(model_config_.quant_config.is_fp8_blockwise ? DataType::TYPE_BLOCK_FP8_E4M3
                                                                           : DataType::TYPE_INVALID);
  }
  if ((model_config_.quant_config.method == QUANT_GPTQ && model_config_.quant_config.bits == 4) || enable_moe_int4) {
    moe_matmul_param.push_back(DataType::TYPE_I4_GROUP);
    // group_size
    moe_matmul_param.push_back(static_cast<int>(model_config_.quant_config.group_size));
  } else {
    moe_matmul_param.push_back(DataType::TYPE_INVALID);
    // group_size
    moe_matmul_param.push_back(0);
  }
  moe_matmul_param.push_back(model_config_.moe_config.apply_weight);

  weight_type = base_weight->GetModelWeights(weight_names[0]).dtype;
  DataType down_weight_type = base_weight->GetModelWeights(weight_names[1]).dtype;
  if (down_weight_type != weight_type) {
    KLLM_THROW(
        fmt::format("Moe Weights Load Error: down_experts dtype {} and up_gate_experts dtype {} should have same dtype",
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
  if (model_config_.quant_config.method == QUANT_GPTQ || enable_moe_int4) {
    if (weight_type == TYPE_UINT8 || weight_type == TYPE_INT8) {
      return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ, NONE_QUANT);
    }
  }

  if (!use_vllm_moe && (model_config_.quant_config.method == QUANT_GPTQ && model_config_.quant_config.bits == 4) ||
      enable_moe_int4) {
    return CreateLayer(TYPE_I4_GROUP, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ, MARLIN_BACKEND);
  }
  return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_NONE, NONE_QUANT);
}

std::shared_ptr<BaseLayer> MoeLayerFactory::CreateLayer(DataType weight_type, DataType input_type, DataType output_type,
                                                        const std::vector<std::any>& init_params, QuantMode quant_mode,
                                                        GroupQuantBackend backend) {
  auto it = builder_map_.find({weight_type, input_type, output_type, quant_mode, backend});
  if (it != builder_map_.end()) {
    std::shared_ptr<BaseLayer> layer = (this->*(it->second))();
    layer->Init(init_params, runtime_config_, context_, rank_);
    return layer;
  } else {
    KLLM_THROW(
        fmt::format("MatMul Not support weight_type {}, input_type {}, output_type {}, quant_mode {}, backend {}.",
                    GetTypeString(weight_type), GetTypeString(input_type), GetTypeString(output_type),
                    GetQuantModeString(quant_mode), GetGroupQuantBackendString(backend)));
  }
}

}  // namespace ksana_llm
