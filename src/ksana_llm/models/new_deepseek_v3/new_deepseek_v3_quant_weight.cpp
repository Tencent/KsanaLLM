/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_quant_weight.h"

namespace ksana_llm {

#ifdef ENABLE_FP8
#ifdef ENABLE_FP8_TORCH
template <typename T>
Status NewDeepSeekV3QuantWeight<T>::ProcessMlaFp8E4m3BlockWiseScaleOfWeight(
                                          std::unordered_set<std::string> & processed_weights,
                                          std::unordered_set<std::string> & dequant_weights,
                                          int dev_rank,
                                          std::shared_ptr<NewDeepSeekV3Config> &
                                            new_deepseek_v3_config,
                                          std::unordered_map<std::string, Tensor> &
                                            device_model_weights,
                                          std::shared_ptr<Context> & context_) {
  size_t tp_size = context_->GetTensorParallelSize();
  size_t qk_rope_head_dim = new_deepseek_v3_config->mla_config.qk_rope_head_dim;
  size_t qk_nope_head_dim = new_deepseek_v3_config->mla_config.qk_nope_head_dim;
  size_t v_head_dim = new_deepseek_v3_config->mla_config.v_head_dim;
  size_t head_num_tp = static_cast<size_t>(DivRoundUp(new_deepseek_v3_config->head_num, tp_size));
  for (auto & weight_name : dequant_weights) {
    if (weight_name.find(".q_b_proj.weight") != std::string::npos) {
      std::string weight_scale_name = weight_name + "_scale_inv";
      if (device_model_weights.find(weight_name) == device_model_weights.end() ||
          device_model_weights.find(weight_scale_name) == device_model_weights.end()) {
        KLLM_THROW(fmt::format("Not found weight: {} or weight scale: {}", weight_name, weight_scale_name));
      }
      Tensor& quant_weight = device_model_weights.at(weight_name);
      Tensor& quant_weight_scale = device_model_weights.at(weight_scale_name);
      // Dequant q_b_proj (set dequant dtype tp bf16 for compatible)
      Tensor dequant_q_b_proj = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_BF16,
                                      quant_weight.shape, dev_rank);
      DequantFp8E4m3BlockWise<T>(
        quant_weight.GetPtr<void>(),
        quant_weight_scale.GetPtr<void>(),
        dequant_q_b_proj.GetPtr<void>(),
        quant_weight.shape[0],
        quant_weight.shape[1],
        new_deepseek_v3_config->quant_config.weight_block_size[1],
        context_->GetMemoryManageStreams()[dev_rank].Get());
      // split dequant q_b_proj
      if (dequant_q_b_proj.shape[0] != (head_num_tp * (qk_nope_head_dim + qk_rope_head_dim))) {
        KLLM_THROW(fmt::format("Not support shape of dequant weight: {}", weight_name));
      }
      Tensor dequant_q_b_nope_proj = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_BF16,
                                            {head_num_tp * qk_nope_head_dim, dequant_q_b_proj.shape[1]}, dev_rank);
      size_t nope_dst_pitch =
          qk_nope_head_dim * dequant_q_b_proj.shape[1] * GetTypeSize(DataType::TYPE_BF16);
      size_t src_pitch = (qk_nope_head_dim + qk_rope_head_dim) * dequant_q_b_proj.shape[1] *
                          GetTypeSize(DataType::TYPE_BF16);
      Memcpy2DAsync(dequant_q_b_nope_proj.GetPtr<void>(), nope_dst_pitch,
                    dequant_q_b_proj.GetPtr<void>(), src_pitch, nope_dst_pitch, head_num_tp,
                    MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);

      Tensor dequant_q_b_rope_proj = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_BF16,
                                            {head_num_tp * qk_rope_head_dim, dequant_q_b_proj.shape[1]}, dev_rank);
      size_t rope_dst_pitch = qk_rope_head_dim * dequant_q_b_rope_proj.shape[1]*
                              GetTypeSize(DataType::TYPE_BF16);
      Memcpy2DAsync(dequant_q_b_rope_proj.GetPtr<void>(), rope_dst_pitch,
                    dequant_q_b_proj.GetPtr<void>() + nope_dst_pitch, src_pitch, rope_dst_pitch, head_num_tp,
                    MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);

      // Quant q_b_nope_proj and q_b_rope_proj
      std::string quant_nope_weight_name =
        weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.q_b_nope_proj.weight";
      std::string quant_nope_weight_scale =
        weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.q_b_nope_proj.weight_scale_inv";
      Tensor quant_q_b_nope_weight = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP8_E4M3,
                                            dequant_q_b_nope_proj.shape, dev_rank);
      size_t weight_scale_shape_0 = static_cast<size_t>(DivRoundUp(quant_q_b_nope_weight.shape[0],
                                    new_deepseek_v3_config->quant_config.weight_block_size[0]));
      size_t weight_scale_shape_1 = static_cast<size_t>(DivRoundUp(quant_q_b_nope_weight.shape[1],
                                    new_deepseek_v3_config->quant_config.weight_block_size[1]));
      Tensor quant_q_b_nope_weight_scale = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                                            {weight_scale_shape_0, weight_scale_shape_1}, dev_rank);
      ScaledQuantizeFp8E4m3<T>(
            static_cast<T*>(dequant_q_b_nope_proj.GetPtr<void>()),
            quant_q_b_nope_weight.GetPtr<void>(),
            static_cast<float*>(quant_q_b_nope_weight_scale.GetPtr<void>()),
            new_deepseek_v3_config->quant_config.weight_block_size,
            dequant_q_b_nope_proj.shape[0],
            dequant_q_b_nope_proj.shape[1],
            dev_rank);
      device_model_weights[quant_nope_weight_name] = quant_q_b_nope_weight;
      device_model_weights[quant_nope_weight_scale] = quant_q_b_nope_weight_scale;
      processed_weights.insert(quant_nope_weight_name);

      std::string quant_rope_weight_name =
        weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.q_b_rope_proj.weight";
      std::string quant_rope_weight_scale =
        weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.q_b_rope_proj.weight_scale_inv";
      Tensor quant_q_b_rope_weight = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP8_E4M3,
                                            dequant_q_b_rope_proj.shape, dev_rank);
      weight_scale_shape_0 = static_cast<size_t>(DivRoundUp(quant_q_b_rope_weight.shape[0],
                                    new_deepseek_v3_config->quant_config.weight_block_size[0]));
      weight_scale_shape_1 = static_cast<size_t>(DivRoundUp(quant_q_b_rope_weight.shape[1],
                                    new_deepseek_v3_config->quant_config.weight_block_size[1]));
      Tensor quant_q_b_rope_weight_scale = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                                            {weight_scale_shape_0, weight_scale_shape_1}, dev_rank);
      ScaledQuantizeFp8E4m3<T>(
            static_cast<T*>(dequant_q_b_rope_proj.GetPtr<void>()),
            quant_q_b_rope_weight.GetPtr<void>(),
            static_cast<float*>(quant_q_b_rope_weight_scale.GetPtr<void>()),
            new_deepseek_v3_config->quant_config.weight_block_size,
            dequant_q_b_rope_proj.shape[0],
            dequant_q_b_rope_proj.shape[1],
            dev_rank);
      device_model_weights[quant_rope_weight_name] = quant_q_b_rope_weight;
      device_model_weights[quant_rope_weight_scale] = quant_q_b_rope_weight_scale;
      processed_weights.insert(quant_rope_weight_name);

      device_model_weights.erase(weight_name);
      device_model_weights.erase(weight_scale_name);
      continue;
    }
    if (weight_name.find(".kv_b_proj.weight") != std::string::npos) {
      std::string weight_scale_name = weight_name + "_scale_inv";
      if (device_model_weights.find(weight_name) == device_model_weights.end() ||
          device_model_weights.find(weight_scale_name) == device_model_weights.end()) {
        KLLM_THROW(fmt::format("Not found weight: {} or weight scale: {}", weight_name, weight_scale_name));
      }
      Tensor& quant_weight = device_model_weights.at(weight_name);
      Tensor& quant_weight_scale = device_model_weights.at(weight_scale_name);
      // Dequant kv_b_proj
      Tensor dequant_kv_b_proj = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_BF16,
                                        quant_weight.shape, dev_rank);
      DequantFp8E4m3BlockWise<T>(
        quant_weight.GetPtr<void>(),
        quant_weight_scale.GetPtr<void>(),
        dequant_kv_b_proj.GetPtr<void>(),
        quant_weight.shape[0],
        quant_weight.shape[1],
        new_deepseek_v3_config->quant_config.weight_block_size[1],
        context_->GetMemoryManageStreams()[dev_rank].Get());

      // split dequant kv_b_proj
      if (dequant_kv_b_proj.shape[0] != (head_num_tp * (qk_nope_head_dim + v_head_dim))) {
        KLLM_THROW(fmt::format("Not support shape of dequant weight: {}", weight_name));
      }
      Tensor dequant_kv_b_nope_proj = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_BF16,
                                            {head_num_tp * qk_nope_head_dim, dequant_kv_b_proj.shape[1]}, dev_rank);
      size_t nope_dst_pitch =
          qk_nope_head_dim * dequant_kv_b_proj.shape[1] * GetTypeSize(DataType::TYPE_BF16);
      size_t src_pitch = (qk_nope_head_dim + v_head_dim) * dequant_kv_b_proj.shape[1] *
                          GetTypeSize(DataType::TYPE_BF16);
      Memcpy2DAsync(dequant_kv_b_nope_proj.GetPtr<void>(), nope_dst_pitch,
                    dequant_kv_b_proj.GetPtr<void>(), src_pitch, nope_dst_pitch, head_num_tp,
                    MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);

      Tensor dequant_v_head_proj = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_BF16,
                                          {head_num_tp * v_head_dim, dequant_kv_b_proj.shape[1]}, dev_rank);
      size_t v_head_dst_pitch =
          v_head_dim * dequant_kv_b_proj.shape[1] * GetTypeSize(DataType::TYPE_BF16);
      Memcpy2DAsync(dequant_v_head_proj.GetPtr<void>(), v_head_dst_pitch,
                    dequant_kv_b_proj.GetPtr<void>() + nope_dst_pitch, src_pitch,
                    v_head_dst_pitch, head_num_tp,
                    MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      // For the latest weight absorption version process
      if (GetAbsorbWeightsType() == AbsorbWeightsType::kAbsorbTypeBMM) {
        // Copy dequant kv_b_proj to w_uk_t
        Tensor w_uk_t_tensor = dequant_kv_b_nope_proj;
        w_uk_t_tensor.shape = {head_num_tp, qk_nope_head_dim, dequant_kv_b_nope_proj.shape[1]};
        std::string w_uk_t_name = weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.w_uk_t.weight";
        device_model_weights[w_uk_t_name] = w_uk_t_tensor;
        processed_weights.insert(w_uk_t_name);

        // Permute dequant_nope_weight to w_uv
        Tensor w_uv_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, w_uk_t_tensor.dtype,
                                    {head_num_tp, dequant_v_head_proj.shape[1], v_head_dim}, dev_rank);
        dequant_v_head_proj.shape = {head_num_tp, v_head_dim, dequant_v_head_proj.shape[1]};
        Permute(dequant_v_head_proj, w_uv_tensor, {0, 2, 1}, context_->GetMemoryManageStreams()[dev_rank]);
        w_uv_tensor.shape = dequant_v_head_proj.shape;
        for (size_t i = 0; i < dequant_v_head_proj.shape.size(); i++) {
          w_uv_tensor.shape[i] = dequant_v_head_proj.shape[i];
        }
        std::string w_uv_name = weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.w_uv.weight";
        device_model_weights[w_uv_name] = w_uv_tensor;
        dequant_v_head_proj.shape = {head_num_tp * v_head_dim, dequant_kv_b_proj.shape[1]};
        processed_weights.insert(w_uv_name);
      }

      // Quant kv_b_nope_proj and v_head_proj
      std::string quant_nope_weight_name =
        weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.kv_b_nope_proj.weight";
      std::string quant_nope_weight_scale_name =
        weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.kv_b_nope_proj.weight_scale_inv";
      size_t weight_scale_shape_0 = DivRoundUp(dequant_kv_b_nope_proj.shape[0],
                                              new_deepseek_v3_config->quant_config.weight_block_size[0]);
      size_t weight_scale_shape_1 = DivRoundUp(dequant_kv_b_nope_proj.shape[1],
                                              new_deepseek_v3_config->quant_config.weight_block_size[1]);
      Tensor quant_nope_weight = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP8_E4M3,
                                        dequant_kv_b_nope_proj.shape, dev_rank);
      Tensor quant_nope_weight_scale = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                                              {weight_scale_shape_0, weight_scale_shape_1}, dev_rank);
      ScaledQuantizeFp8E4m3<T>(
        static_cast<T*>(dequant_kv_b_nope_proj.GetPtr<void>()),
        quant_nope_weight.GetPtr<void>(),
        static_cast<float*>(quant_nope_weight_scale.GetPtr<void>()),
        new_deepseek_v3_config->quant_config.weight_block_size,
        dequant_kv_b_nope_proj.shape[0],
        dequant_kv_b_nope_proj.shape[1],
        dev_rank);
      device_model_weights[quant_nope_weight_name] = quant_nope_weight;
      device_model_weights[quant_nope_weight_scale_name] = quant_nope_weight_scale;
      processed_weights.insert(quant_nope_weight_name);

      std::string quant_v_head_weight_name =
        weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.v_head_proj.weight";
      std::string quant_v_head_weight_scale_name =
        weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.v_head_proj.weight_scale_inv";
      Tensor quant_v_head_weight = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP8_E4M3,
                                          dequant_v_head_proj.shape, dev_rank);
      weight_scale_shape_0 = DivRoundUp(dequant_v_head_proj.shape[0],
                                        new_deepseek_v3_config->quant_config.weight_block_size[0]);
      weight_scale_shape_1 = DivRoundUp(dequant_v_head_proj.shape[1],
                                        new_deepseek_v3_config->quant_config.weight_block_size[1]);
      Tensor quant_v_head_weight_scale = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                                                {weight_scale_shape_0, weight_scale_shape_1}, dev_rank);
      ScaledQuantizeFp8E4m3<T>(
        static_cast<T*>(dequant_v_head_proj.GetPtr<void>()),
        quant_v_head_weight.GetPtr<void>(),
        static_cast<float*>(quant_v_head_weight_scale.GetPtr<void>()),
        new_deepseek_v3_config->quant_config.weight_block_size,
        dequant_v_head_proj.shape[0],
        dequant_v_head_proj.shape[1],
        dev_rank);
      device_model_weights[quant_v_head_weight_name] = quant_v_head_weight;
      device_model_weights[quant_v_head_weight_scale_name] = quant_v_head_weight_scale;
      processed_weights.insert(quant_v_head_weight_name);

      device_model_weights.erase(weight_name);
      device_model_weights.erase(weight_scale_name);
      continue;
    }
  }
  return Status();
}
#endif
#endif


template class NewDeepSeekV3QuantWeight<float>;
template class NewDeepSeekV3QuantWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class NewDeepSeekV3QuantWeight<bfloat16>;
#endif
}  // namespace ksana_llm
