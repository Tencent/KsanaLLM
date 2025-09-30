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
                                 std::shared_ptr<Context> context)
    : model_config_(model_config), runtime_config_(runtime_config), rank_(rank), context_(context) {
#ifdef ENABLE_CUDA
  // TODO(winminkong): Organize the quantization backend and quantization types of the MoE layer.
  // for common moe, qwen and llama moe
  builder_map_[{TYPE_FP32, TYPE_FP32, TYPE_FP32, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_FP16, TYPE_FP16, TYPE_FP16, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_BF16, TYPE_BF16, TYPE_BF16, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;

  // for hunyuan-large int4
  builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, MARLIN_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MarlinMoeLayer>;

#  ifdef ENABLE_FP8
  // NOTE: Fp8MoeLayer only suport fp8e4m3 rightnow
  // for hunyuan-large fp8
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, MOE_QUANT_FP8_E4M3, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<Fp8MoeLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, MOE_QUANT_FP8_E4M3, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<Fp8MoeLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, MOE_QUANT_FP8_E4M3, DEFAULT_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<Fp8MoeLayer>;

  // for deepseek fp8 blockwise
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, MOE_QUANT_BLOCK_FP8_E4M3, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, MOE_QUANT_BLOCK_FP8_E4M3, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, MOE_QUANT_BLOCK_FP8_E4M3, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;

  // for deepseek moe-int4
  builder_map_[{TYPE_UINT8, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;
  builder_map_[{TYPE_UINT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<MoeLayer>;

  // for deepseek w4afp8
  builder_map_[{TYPE_INT8, TYPE_FP16, TYPE_FP16, MOE_QUANT_GPTQ, CUTLASS_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<CutlassMoeLayer>;
  builder_map_[{TYPE_INT8, TYPE_BF16, TYPE_BF16, MOE_QUANT_GPTQ, CUTLASS_MOE_BACKEND}] =
      &MoeLayerFactory::BuildLayer<CutlassMoeLayer>;
#  endif
#endif
}

std::shared_ptr<BaseLayer> MoeLayerFactory::AutoCreateMoeLayer(std::shared_ptr<BaseWeight> base_weight,
                                                               std::vector<std::string> weight_names,
                                                               DataType weight_type, DataType input_type,
                                                               DataType output_type, int layer_idx,
                                                               const std::vector<std::any>& init_params) {
  // moe layer   (weight_names[0]: up_gate_experts, weight_names[1]: down_experts)
  bool use_vllm_moe = model_config_.moe_config.use_vllm_moe;
  size_t hidden_size = static_cast<size_t>(model_config_.hidden_units);
  size_t moe_inter_size_per_rank = static_cast<size_t>(
      DivRoundUp(model_config_.moe_config.moe_inter_size, runtime_config_.parallel_basic_config.moe_tensor_para_size));
  size_t expert_num_per_node =
      model_config_.moe_config.num_experts / (runtime_config_.parallel_basic_config.expert_parallel_size *
                                              runtime_config_.parallel_basic_config.expert_world_size);
  // 专家数量检查
  size_t up_gate_experts_num = base_weight->GetModelWeights(weight_names[0]).shape[0];
  size_t down_experts_num = base_weight->GetModelWeights(weight_names[1]).shape[0];
  KLLM_CHECK_WITH_INFO(up_gate_experts_num == down_experts_num,
                       fmt::format("up_gate experts {} != down_experts {}", up_gate_experts_num, down_experts_num));
  // 解析enable_moe_int4
  bool enable_moe_int4 = false;
  if (model_config_.quant_config.method == QUANT_GPTQ && model_config_.quant_config.bits == 4) {
    enable_moe_int4 = true;
  } else if (model_config_.quant_config.method != QUANT_GPTQ && model_config_.sub_quant_configs.size() > 0 &&
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
  // moe层size检查
  size_t up_gate_hidden_size = base_weight->GetModelWeights(weight_names[0]).shape[2];
  size_t down_hidden_size = base_weight->GetModelWeights(weight_names[1]).shape[1];
  if (enable_moe_int4) {
    if (use_vllm_moe) {
      up_gate_hidden_size = up_gate_hidden_size / 4 * (32 / model_config_.quant_config.bits);
    } else {  // marlin gptq
      up_gate_hidden_size =
          base_weight->GetModelWeights(weight_names[0]).shape[1] / (sizeof(int) / model_config_.quant_config.bits) * 16;
      down_hidden_size = base_weight->GetModelWeights(weight_names[1]).shape[2] / model_config_.quant_config.bits * 2;
    }
  }
  KLLM_CHECK_WITH_INFO(
      up_gate_hidden_size == down_hidden_size,
      fmt::format("up_gate_experts hidden_size != down_experts hidden_size {}", up_gate_hidden_size, down_hidden_size));
  // moe层权重类型检查
  weight_type = base_weight->GetModelWeights(weight_names[0]).dtype;
  DataType down_weight_type = base_weight->GetModelWeights(weight_names[1]).dtype;
  KLLM_CHECK_WITH_INFO(down_weight_type == weight_type,
                       fmt::format("down_experts dtype {} != up_gate_experts dtype {}", down_weight_type, weight_type));

  std::vector<std::any> moe_matmul_param = init_params;                                    // moe_scale_norm_mode
  moe_matmul_param.push_back(runtime_config_.max_step_token_num);                          // max_token_num
  moe_matmul_param.push_back(layer_idx);                                                   // layer_idx
  moe_matmul_param.push_back(expert_num_per_node);                                         // expert_num_per_node
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
  moe_matmul_param.push_back(runtime_config_.enable_full_shared_expert);             // dp = ep
  if (!enable_moe_int4 && model_config_.quant_config.is_fp8_blockwise) {
    moe_matmul_param.push_back(DataType::TYPE_BLOCK_FP8_E4M3);  // fp8_weight_dtype
  } else {
    moe_matmul_param.push_back(DataType::TYPE_INVALID);  // fp8_weight_dtype
  }
  if (enable_moe_int4) {
    moe_matmul_param.push_back(DataType::TYPE_I4_GROUP);                                  // int_weight_dtype
    moe_matmul_param.push_back(static_cast<int>(model_config_.quant_config.group_size));  // group_size
  } else {
    moe_matmul_param.push_back(DataType::TYPE_INVALID);  // int_weight_dtype
    moe_matmul_param.push_back(0);                       // group_size
  }
  moe_matmul_param.push_back(model_config_.moe_config.apply_weight);

  // fp8 moe创建
  if (weight_type == TYPE_FP8_E4M3) {
    QuantMode quant_mode = model_config_.quant_config.is_fp8_blockwise ? MOE_QUANT_BLOCK_FP8_E4M3 : MOE_QUANT_FP8_E4M3;
    MoeComputeBackend backend = model_config_.quant_config.is_fp8_blockwise ? TRITON_MOE_BACKEND : DEFAULT_MOE_BACKEND;
    return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, quant_mode, backend);
  }
  // int4 moe创建
  if (enable_moe_int4) {
    if (weight_type == TYPE_INT8) {
      return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ, CUTLASS_MOE_BACKEND);
    }
    if (weight_type == TYPE_UINT8) {
      return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ, TRITON_MOE_BACKEND);
    }
    if (!use_vllm_moe) {
      return CreateLayer(TYPE_I4_GROUP, input_type, output_type, moe_matmul_param, MOE_QUANT_GPTQ, MARLIN_MOE_BACKEND);
    }
  }
  // 默认moe创建
  return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_NONE, DEFAULT_MOE_BACKEND);
}

std::shared_ptr<BaseLayer> MoeLayerFactory::CreateLayer(DataType weight_type, DataType input_type, DataType output_type,
                                                        const std::vector<std::any>& init_params, QuantMode quant_mode,
                                                        MoeComputeBackend backend) {
  auto it = builder_map_.find({weight_type, input_type, output_type, quant_mode, backend});
  KLLM_CHECK_WITH_INFO(
      it != builder_map_.end(),
      fmt::format("Moe Not support weight_type {}, input_type {}, output_type {}, quant_mode {}, backend {}.",
                  GetTypeString(weight_type), GetTypeString(input_type), GetTypeString(output_type),
                  GetQuantModeString(quant_mode), GetMoeComputeBackendString(backend)));
  std::shared_ptr<BaseLayer> layer = (this->*(it->second))();
  layer->Init(init_params, runtime_config_, context_, rank_);
  return layer;
}

}  // namespace ksana_llm
