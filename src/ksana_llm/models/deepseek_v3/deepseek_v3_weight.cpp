/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/deepseek_v3/deepseek_v3_weight.h"
#include <numeric>

#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/singleton.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {

template <typename T>
DeepSeekV3Weight<T>::DeepSeekV3Weight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                                      std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, runtime_config, rank, context),
      CommonMlaWeight<T>(model_config, runtime_config, rank, context),
      CommonMoeWeight<T>(model_config, runtime_config, rank, context) {}

template <typename T>
Status DeepSeekV3Weight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                                const std::vector<std::string>& weight_name_list,
                                                const std::vector<std::string>& custom_name_list) {
  SetDevice(rank_);
  CommonMoeWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  CommonMlaWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  return Status();
}

template <typename T>
void DeepSeekV3Weight<T>::ProcessWeights() {
#ifdef ENABLE_CUDA
  CommonMoeWeight<T>::ProcessWeights();
  CommonMlaWeight<T>::ProcessWeights();
  CommonWeight<T>::ProcessWeights();
  CommonWeight<T>::PrintDebugMessage();
  // Cast some quant model's e_score_correction_bias to float32 type
  for (auto it = weights_map_.begin(); it != weights_map_.end(); ++it) {
    if ((it->first.find("e_score_correction_bias") != std::string::npos) && (it->second.dtype != DataType::TYPE_FP32)) {
      tensor_manager_->AddWeightTensor("empty_score_bias", it->second.shape, DataType::TYPE_FP32);

      if (it->second.dtype == DataType::TYPE_BF16) {
        DataToFloat<__nv_bfloat16>(it->second.template GetPtr<void>(), it->second.shape[0], 1, 1,
                                   weights_map_["empty_score_bias"].template GetPtr<void>(),
                                   context_->GetMemoryManageStreams()[rank_].Get());
      } else if (it->second.dtype == DataType::TYPE_FP16) {
        DataToFloat<half>(it->second.template GetPtr<void>(), it->second.shape[0], 1, 1,
                          weights_map_["empty_score_bias"].template GetPtr<void>(),
                          context_->GetMemoryManageStreams()[rank_].Get());
      } else {
        KLLM_LOG_ERROR << "Unsupported e_score_correction_bias data type: " << it->second.dtype;
      }
      it->second = weights_map_["empty_score_bias"];
      weights_map_.erase("empty_score_bias");
    }
  }

  // Absorb V2 for GPTQ
  if ((GetAbsorbWeightsType() == AbsorbWeightsType::kAbsorbTypeBMM) &&
      (model_config_.quant_config.method == QUANT_GPTQ)) {
    int head_num = model_config_.head_num;
    size_t head_num_per_tp = head_num / Singleton<Environment>::GetInstance()->GetAttentionTensorParallel();
    size_t kv_lora_rank = model_config_.mla_config.kv_lora_rank;
    size_t qk_nope_head_dim = model_config_.mla_config.qk_nope_head_dim;
    size_t v_head_dim = model_config_.mla_config.v_head_dim;
    for (const auto layer_idx : required_layer_idx_.all) {
      //  For W_UK_T
      std::string kv_b_nope_proj_name = fmt::format("model.layers.{}.self_attn.kv_b_nope_proj.weight", layer_idx);
      Tensor kv_b_nope_proj_weight =
          weights_map_[kv_b_nope_proj_name];  // (512, 2048) （shape : kv_lora_rank, num_attn_heads, qk_nope_head_dim)
      // Dequant
      kv_b_nope_proj_weight = quant_weight_solver_->CommonDequantTensor(kv_b_nope_proj_name);

      // Creat W_UK_T
      std::string w_uk_t_name = fmt::format("model.layers.{}.self_attn.w_uk_t.weight", layer_idx);
      tensor_manager_->AddWeightTensor(w_uk_t_name, {head_num_per_tp, qk_nope_head_dim, kv_lora_rank},
                                       kv_b_nope_proj_weight.dtype);
      kv_b_nope_proj_weight.shape = {kv_lora_rank, head_num_per_tp, qk_nope_head_dim};
      Permute(kv_b_nope_proj_weight, weights_map_[w_uk_t_name], {1, 2, 0}, context_->GetMemoryManageStreams()[rank_]);

      // For W_UV
      std::string v_head_proj_name = fmt::format("model.layers.{}.self_attn.v_head_proj.weight", layer_idx);
      Tensor v_head_proj_weight =
          weights_map_[v_head_proj_name];  // (512, 2048) (shape : kv_lora_rank, num_attn_heads, v_head_dim)

      // Dequant
      v_head_proj_weight = quant_weight_solver_->CommonDequantTensor(v_head_proj_name);
      std::string w_uv_name = fmt::format("model.layers.{}.self_attn.w_uv.weight", layer_idx);
      v_head_proj_weight.shape = {kv_lora_rank, head_num_per_tp, v_head_dim};
      tensor_manager_->AddWeightTensor(w_uv_name, {head_num_per_tp, kv_lora_rank, v_head_dim},
                                       v_head_proj_weight.dtype);
      Permute(v_head_proj_weight, weights_map_[w_uv_name], {1, 0, 2}, context_->GetMemoryManageStreams()[rank_]);

      // Free unused dequant weight
      for (auto it = weights_map_.begin(); it != weights_map_.end();) {
        if (it->first.find("dequant") != std::string::npos) {
          it = weights_map_.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  CommonWeight<T>::PrintDebugMessage();
#endif
}

template class DeepSeekV3Weight<float>;
template class DeepSeekV3Weight<float16>;
template class DeepSeekV3Weight<bfloat16>;

}  // namespace ksana_llm
