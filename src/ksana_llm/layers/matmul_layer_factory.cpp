/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/layers/matmul_layer_factory.h"

#include "ksana_llm/layers/batched_matmul_layer.h"
#include "ksana_llm/layers/blockwise_matmul_layer.h"
#include "ksana_llm/layers/cutlass_matmul_layer.h"
#include "ksana_llm/layers/fp8_matmul_layer.h"
#include "ksana_llm/layers/fp8_moe_layer.h"
#include "ksana_llm/layers/machete_matmul_layer.h"
#include "ksana_llm/layers/marlin_matmul_layer.h"
#include "ksana_llm/layers/marlin_moe_layer.h"
#include "ksana_llm/layers/matmul_layer.h"
#include "ksana_llm/layers/moe_layer.h"

namespace ksana_llm {

MatMulLayerFactory::MatMulLayerFactory(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                       const int rank, std::shared_ptr<Context> context) {
  context_ = context;
  rank_ = rank;
  model_config_ = model_config;
  runtime_config_ = runtime_config;

  builder_map_[{TYPE_FP32, TYPE_FP32, TYPE_FP32, QUANT_NONE, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<MatMulLayer>;
  builder_map_[{TYPE_FP16, TYPE_FP16, TYPE_FP16, QUANT_NONE, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<MatMulLayer>;
  builder_map_[{TYPE_BF16, TYPE_BF16, TYPE_BF16, QUANT_NONE, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<MatMulLayer>;

  builder_map_[{TYPE_VOID, TYPE_FP32, TYPE_FP32, QUANT_NONE, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<BatchedMatMulLayer>;
  builder_map_[{TYPE_VOID, TYPE_FP16, TYPE_FP16, QUANT_NONE, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<BatchedMatMulLayer>;
  builder_map_[{TYPE_VOID, TYPE_BF16, TYPE_BF16, QUANT_NONE, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<BatchedMatMulLayer>;

#ifdef ENABLE_CUDA
  builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_GPTQ, CUTLASS_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<CutlassMatMulLayer>;
  builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_AWQ, CUTLASS_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<CutlassMatMulLayer>;

  builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_GPTQ, MARLIN_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<MarlinMatMulLayer>;
  builder_map_[{TYPE_I4_GROUP, TYPE_BF16, TYPE_BF16, QUANT_GPTQ, MARLIN_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<MarlinMatMulLayer>;
  builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_AWQ, MARLIN_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<MarlinMatMulLayer>;

  builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_GPTQ, MACHETE_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<MacheteMatMulLayer>;
  builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_AWQ, MACHETE_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<MacheteMatMulLayer>;
  builder_map_[{TYPE_I4_GROUP, TYPE_BF16, TYPE_BF16, QUANT_GPTQ, MACHETE_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<MacheteMatMulLayer>;
  builder_map_[{TYPE_I4_GROUP, TYPE_BF16, TYPE_BF16, QUANT_AWQ, MACHETE_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<MacheteMatMulLayer>;

#endif
#ifdef ENABLE_FP8
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, QUANT_FP8_E4M3, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<Fp8MatMulLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, QUANT_FP8_E4M3, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<Fp8MatMulLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, QUANT_FP8_E4M3, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<Fp8MatMulLayer>;

  builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, QUANT_BLOCK_FP8_E4M3, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<BlockwiseMatMulLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, QUANT_BLOCK_FP8_E4M3, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<BlockwiseMatMulLayer>;
  builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, QUANT_BLOCK_FP8_E4M3, DEFAULT_LINEAR_BACKEND}] =
      &MatMulLayerFactory::BuildLayer<BlockwiseMatMulLayer>;

#endif
}

std::shared_ptr<BaseLayer> MatMulLayerFactory::AutoCreateLayer(std::shared_ptr<BaseWeight> base_weight,
                                                               std::string weight_name, DataType weight_type,
                                                               DataType input_type, DataType output_type,
                                                               LinearComputeBackend backend,
                                                               const std::vector<std::any>& init_params) {
  // gptq layer
  if (model_config_.is_quant &&
      (model_config_.quant_config.method == QUANT_GPTQ || model_config_.quant_config.method == QUANT_AWQ)) {
    bool enable_full_shared_expert = runtime_config_.enable_full_shared_expert;
    size_t tp = enable_full_shared_expert ? 1 : runtime_config_.parallel_basic_config.tensor_parallel_size;
    size_t attn_tp = runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
    size_t hidden_size = model_config_.hidden_units;
    size_t inter_size_per_rank = model_config_.inter_size;
    size_t shared_expert_inter_size_per_rank = model_config_.moe_config.shared_expert_inter_size;
    if (!enable_full_shared_expert) {
      inter_size_per_rank /= tp;
      shared_expert_inter_size_per_rank /= tp;
    }
    uint32_t qk_rope_head_dim = model_config_.mla_config.qk_rope_head_dim;
    uint32_t qk_nope_head_dim = model_config_.mla_config.qk_nope_head_dim;
    uint32_t q_lora_rank = model_config_.mla_config.q_lora_rank;
    uint32_t kv_lora_rank = model_config_.mla_config.kv_lora_rank;
    uint32_t v_head_dim = model_config_.mla_config.v_head_dim;
    size_t head_num = model_config_.head_num;
    // The inter size in config.json for the qwen1 model is twice the true inter size.
    if (model_config_.type == "qwen") {
      inter_size_per_rank /= 2;
    }
    size_t qkv_size = model_config_.size_per_head * (model_config_.head_num + 2 * model_config_.num_key_value_heads);
    // Because the layout convertion, we can't get n/k from weight shape, and have to calculate it.
    std::map<std::string, std::tuple<size_t, size_t, bool>> kn_pairs;
    // mlp
    kn_pairs["mlp.gate_proj"] = std::make_tuple(hidden_size, inter_size_per_rank, true);
    kn_pairs["mlp.up_proj"] = kn_pairs["mlp.gate_proj"];
    kn_pairs["mlp.down_proj"] = std::make_tuple(inter_size_per_rank, hidden_size, false);
    kn_pairs["mlp.shared_expert.gate_proj"] = std::make_tuple(hidden_size, shared_expert_inter_size_per_rank, true);
    kn_pairs["mlp.shared_expert.up_proj"] = kn_pairs["mlp.shared_expert.gate_proj"];
    kn_pairs["mlp.shared_expert.down_proj"] = std::make_tuple(shared_expert_inter_size_per_rank, hidden_size, false);
    kn_pairs["query_key_value"] = std::make_tuple(hidden_size, qkv_size / attn_tp, true);
    kn_pairs["q_proj"] = std::make_tuple(hidden_size, hidden_size / attn_tp, true);

    // attn
    kn_pairs["q_a_proj"] = std::make_tuple(hidden_size, q_lora_rank, false);
    kn_pairs["kv_a_lora_proj"] = std::make_tuple(hidden_size, kv_lora_rank, false);
    kn_pairs["kv_a_rope_proj"] = std::make_tuple(hidden_size, qk_rope_head_dim, false);
    kn_pairs["q_b_nope_proj"] = std::make_tuple(q_lora_rank, head_num / attn_tp * qk_nope_head_dim, true);
    kn_pairs["q_b_rope_proj"] = std::make_tuple(q_lora_rank, head_num / attn_tp * qk_rope_head_dim, true);
    kn_pairs["q_b_nope_rope_proj"] =
        std::make_tuple(q_lora_rank, head_num / attn_tp * (qk_nope_head_dim + qk_rope_head_dim), true);
    kn_pairs["v_head_proj"] = std::make_tuple(kv_lora_rank, head_num / attn_tp * v_head_dim, true);
    if (v_head_dim > 0) {  // mla
      kn_pairs["o_proj"] = std::make_tuple(head_num / attn_tp * v_head_dim, hidden_size, false);
    } else {
      kn_pairs["o_proj"] = std::make_tuple(head_num / attn_tp * model_config_.size_per_head, hidden_size, false);
    }

    for (const auto& kn : kn_pairs) {
      if (weight_name.find(kn.first) != std::string::npos) {
        std::vector<std::any> group_matmul_param;
        group_matmul_param.push_back(runtime_config_.max_step_token_num);                                 // m
        group_matmul_param.push_back(std::get<1>(kn.second));                                             // n
        group_matmul_param.push_back(std::get<0>(kn.second));                                             // k
        group_matmul_param.push_back(model_config_.quant_config.group_size);                              // groupsize
        group_matmul_param.push_back(static_cast<bool>(model_config_.quant_config.method == QUANT_AWQ));  // awq
        group_matmul_param.push_back(static_cast<bool>(model_config_.quant_config.desc_act));             // gptq desc
        group_matmul_param.push_back(static_cast<bool>(std::get<2>(kn.second)));                          // k full
        group_matmul_param.push_back(true);                                                               // cuda gemv
        group_matmul_param.push_back(TYPE_I4_GROUP);  // weight data type
        if (weight_name.find("kv_a_rope_proj") != std::string::npos) {
          return CreateLayer(TYPE_I4_GROUP, input_type, output_type, group_matmul_param, QUANT_GPTQ,
                             MARLIN_LINEAR_BACKEND);
        }
        return CreateLayer(TYPE_I4_GROUP, input_type, output_type, group_matmul_param, QUANT_GPTQ, backend);
      }
    }
  }
  // fp8 layer
  if (base_weight->GetModelWeights(weight_name).dtype == TYPE_FP8_E4M3) {
    if (model_config_.quant_config.method == QUANT_BLOCK_FP8_E4M3) {
      std::vector<std::any> fp8_blockwise_matmul_params;
      fp8_blockwise_matmul_params.push_back(runtime_config_.max_step_token_num);  // m
      // weight is [n， k]
      fp8_blockwise_matmul_params.push_back(size_t(base_weight->GetModelWeights(weight_name).shape[0]));  // n
      fp8_blockwise_matmul_params.push_back(size_t(base_weight->GetModelWeights(weight_name).shape[1]));  // k
      // block_k size
      fp8_blockwise_matmul_params.push_back(model_config_.quant_config.weight_block_size[1]);
      // weight
      fp8_blockwise_matmul_params.push_back(base_weight->GetModelWeights(weight_name));
      return CreateLayer(TYPE_FP8_E4M3, input_type, output_type, fp8_blockwise_matmul_params, QUANT_BLOCK_FP8_E4M3,
                         DEFAULT_LINEAR_BACKEND);
    } else {
      std::vector<std::any> fp8_matmul_params;
      // max_m_
      fp8_matmul_params.push_back(runtime_config_.max_step_token_num);
      // weight is [n, k], k is shape[1]
      fp8_matmul_params.push_back(size_t(base_weight->GetModelWeights(weight_name).shape[1]));
      return CreateLayer(TYPE_FP8_E4M3, input_type, output_type, fp8_matmul_params, QUANT_FP8_E4M3,
                         DEFAULT_LINEAR_BACKEND);
    }
  }
  // batched matmul has no weight
  if (weight_name == "" && weight_type == TYPE_VOID) {
    return CreateLayer(weight_type, input_type, output_type, init_params, QUANT_NONE, DEFAULT_LINEAR_BACKEND);
  }
  // default layer
  return CreateLayer(base_weight, weight_name, input_type, output_type, init_params, QUANT_NONE,
                     DEFAULT_LINEAR_BACKEND);
}

std::shared_ptr<BaseLayer> MatMulLayerFactory::CreateLayer(std::shared_ptr<BaseWeight> base_weight,
                                                           std::string weight_name, DataType input_type,
                                                           DataType output_type,
                                                           const std::vector<std::any>& init_params,
                                                           QuantMode quant_mode, LinearComputeBackend backend) {
  DataType weight_type = base_weight->GetModelWeights(weight_name).dtype;
  // deepseek v3
  if (weight_type == TYPE_INVALID) {
    weight_type = input_type;
  }
  KLLM_LOG_DEBUG << fmt::format(
      "MatMul Creating weight_name: {}, weight_type {}, input_type {}, output_type {}, quant_mode {}, backend {}.",
      weight_name, GetTypeString(weight_type), GetTypeString(input_type), GetTypeString(output_type),
      GetQuantModeString(quant_mode), GetLinearComputeBackendString(backend));
  return CreateLayer(weight_type, input_type, output_type, init_params, quant_mode, backend);
}

std::shared_ptr<BaseLayer> MatMulLayerFactory::CreateLayer(DataType weight_type, DataType input_type,
                                                           DataType output_type,
                                                           const std::vector<std::any>& init_params,
                                                           QuantMode quant_mode, LinearComputeBackend backend) {
  auto it = builder_map_.find({weight_type, input_type, output_type, quant_mode, backend});
  if (it != builder_map_.end()) {
    std::shared_ptr<BaseLayer> layer = (this->*(it->second))();
    layer->Init(init_params, runtime_config_, context_, rank_);
    return layer;
  } else {
    KLLM_THROW(
        fmt::format("MatMul Not support weight_type {}, input_type {}, output_type {}, quant_mode {}, backend {}.",
                    GetTypeString(weight_type), GetTypeString(input_type), GetTypeString(output_type),
                    GetQuantModeString(quant_mode), GetLinearComputeBackendString(backend)));
  }
}

}  // namespace ksana_llm
