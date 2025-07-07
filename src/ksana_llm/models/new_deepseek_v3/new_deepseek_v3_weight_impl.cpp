/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_weight_impl.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {
template <typename T>
Status NewDeepSeekV3WeightImpl<T>::TransSplitOptTrans(const Tensor & host_weight_tensor,
                                      Tensor & output_tensor,
                                      int dev_rank,
                                      std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                      bool transpose) {
  Tensor full_dev_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

  MemcpyAsync(full_dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
              MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

  Tensor permute_dev_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);
  Permute(full_dev_tensor, permute_dev_tensor, {1, 0}, context_->GetMemoryManageStreams()[dev_rank]);
  std::swap(permute_dev_tensor.shape[0], permute_dev_tensor.shape[1]);
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

  std::vector<size_t> slice_shape = {
      static_cast<size_t>(DivRoundUp(permute_dev_tensor.shape[0], context_->GetTensorParallelSize())),
      permute_dev_tensor.shape[1]};

  Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, slice_shape, dev_rank);

  size_t slice_offset = dev_tensor.GetTotalBytes() * dev_rank;
  size_t slice_bytes = dev_tensor.GetTotalBytes();
  if (static_cast<size_t>(dev_rank) == context_->GetTensorParallelSize() - 1) {
    slice_bytes = host_weight_tensor.GetTotalBytes() - slice_offset;
  }

  MemcpyAsync(dev_tensor.GetPtr<void>(), permute_dev_tensor.GetPtr<void>() + slice_offset, slice_bytes,
              MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

  if (transpose) {
    permute_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype, dev_tensor.shape, dev_rank);
    Permute(dev_tensor, permute_dev_tensor, {1, 0}, context_->GetMemoryManageStreams()[dev_rank]);
    std::swap(permute_dev_tensor.shape[0], permute_dev_tensor.shape[1]);
    output_tensor = permute_dev_tensor;
  } else {
    output_tensor = dev_tensor;
  }

  return Status();
}

template <typename T>
Status NewDeepSeekV3WeightImpl<T>::SplitOptTrans(const Tensor & host_weight_tensor,
                      Tensor & output_tensor,
                      int dev_rank,
                      std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                      bool skip_transpose) {
  std::vector<size_t> slice_shape = {
    static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize())),
    host_weight_tensor.shape[1]};
  Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, slice_shape, dev_rank);

  size_t slice_offset = dev_tensor.GetTotalBytes() * dev_rank;
  size_t slice_bytes = dev_tensor.GetTotalBytes();
  if (static_cast<size_t>(dev_rank) == context_->GetTensorParallelSize() - 1) {
    slice_bytes = host_weight_tensor.GetTotalBytes() - slice_offset;
  }

  MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + slice_offset, slice_bytes,
              MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

  if (skip_transpose) {
    output_tensor = dev_tensor;
  } else {
    Tensor permute_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype, dev_tensor.shape, dev_rank);
    Permute(dev_tensor, permute_dev_tensor, {1, 0}, context_->GetMemoryManageStreams()[dev_rank]);
    std::swap(permute_dev_tensor.shape[0], permute_dev_tensor.shape[1]);
    output_tensor = permute_dev_tensor;
  }

  return Status();
}

template <typename T>
Status NewDeepSeekV3WeightImpl<T>::GetExpertsIdx(const std::string& expert_name,
                                                  size_t & layer_idx_,
                                                  size_t & expert_idx_) {
  // Get the index of the moe layer and the index of each expert
  static const std::regex re(R"(\d+)");
  std::sregex_iterator next(expert_name.begin(), expert_name.end(), re);
  std::sregex_iterator end;
  if (next != end) {
    std::smatch match = *next;
    layer_idx_ = std::stoi(match.str());
    next++;
    match = *next;
    expert_idx_ = std::stoi(match.str());
  }
  return Status();
}

template <typename T>
Status NewDeepSeekV3WeightImpl<T>::ProcessGateUpProjWeight(std::string& file_weight_name_,
                                              const Tensor& dev_tensor,
                                              std::unordered_map<std::string, Tensor>& device_model_weights,
                                              int dev_rank,
                                              bool is_quant_weight) {
  int concat_offset = 0;
  std::string replacement = "gate_up_proj";
  if (file_weight_name_.find("gate_proj") != std::string::npos) {
    concat_offset = 0;
    static const std::regex pattern("gate_proj");
    file_weight_name_ = std::regex_replace(file_weight_name_, pattern, replacement);
  } else {
    concat_offset = 1;
    static const std::regex pattern("up_proj");
    file_weight_name_ = std::regex_replace(file_weight_name_, pattern, replacement);
  }

  if (device_model_weights.find(file_weight_name_) == device_model_weights.end()) {
    device_model_weights[file_weight_name_] = Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype,
        {dev_tensor.shape[0] * 2, dev_tensor.shape[1]}, dev_rank);
  }
  Tensor& gate_up_proj_tensor = device_model_weights[file_weight_name_];
  size_t total_bytes = gate_up_proj_tensor.GetTotalBytes() / 2;
  if (is_quant_weight) {
    MemcpyAsync(gate_up_proj_tensor.GetPtr<void>() + concat_offset * total_bytes,
      dev_tensor.GetPtr<void>(), total_bytes, MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  } else {
    gate_up_proj_tensor.shape = {dev_tensor.shape[0], dev_tensor.shape[1] * 2};
    size_t dst_pitch = gate_up_proj_tensor.shape[1] * GetTypeSize(gate_up_proj_tensor.dtype);
    size_t src_pitch = dev_tensor.shape[1] * GetTypeSize(dev_tensor.dtype);
    Memcpy2DAsync(gate_up_proj_tensor.GetPtr<void>() + concat_offset * src_pitch,
      dst_pitch, dev_tensor.GetPtr<void>(), src_pitch, src_pitch, dev_tensor.shape[0],
      MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  }

  return Status();
}

#ifdef ENABLE_CUDA
template <typename T>
Status NewDeepSeekV3WeightImpl<T>::ProcessAbsorbWeightsTypeUKV(
                                      std::unordered_map<std::string, Tensor>& dev_weights_map_,
                                      int dev_rank,
                                      const std::shared_ptr<NewDeepSeekV3Config> &
                                          new_deepseek_v3_config) {
  // local helper function
  auto replace_first = [](std::string str, const std::string& key, const std::string& replacement) {
    size_t pos = str.find(key);
    if (pos != std::string::npos) {
        str.replace(pos, key.length(), replacement);
    }
    return str;
  };

  std::unordered_set<std::string> q_b_nope_names;
  for (const auto & [weight_name, weight_tensor] : dev_weights_map_) {
    if (weight_name.find("self_attn.q_b_nope_proj.weight") != std::string::npos &&
        weight_name.find("self_attn.q_b_nope_proj.weight_scale_inv") == std::string::npos) {
      q_b_nope_names.insert(weight_name);
    }
  }

  auto env_ = Singleton<Environment>::GetInstance();
  size_t q = new_deepseek_v3_config->mla_config.q_lora_rank;
  size_t head_num_per_tp = new_deepseek_v3_config->head_num / env_->GetAttentionTensorParallel();
  size_t qk_nope_head_dim = new_deepseek_v3_config->mla_config.qk_nope_head_dim;
  size_t kv_lora_rank = new_deepseek_v3_config->mla_config.kv_lora_rank;
  size_t hidden_units = new_deepseek_v3_config->hidden_units;

  for (const auto & q_b_nope_name : q_b_nope_names) {
    std::string kv_b_nope_name = replace_first(q_b_nope_name, "q_b_nope_proj", "kv_b_nope_proj");
    std::string v_head_proj_name = replace_first(q_b_nope_name, "q_b_nope_proj", "v_head_proj");
    std::string attn_o_proj_name = replace_first(q_b_nope_name, "q_b_nope_proj", "o_proj");
    Tensor q_b_nope_tensor = dev_weights_map_.at(q_b_nope_name);
    Tensor kv_b_nope_tensor = dev_weights_map_.at(kv_b_nope_name);
    Tensor v_head_tensor = dev_weights_map_.at(v_head_proj_name);
    Tensor attn_o_tensor = dev_weights_map_.at(attn_o_proj_name);
    bool transpose_matrix = false;
    if (new_deepseek_v3_config->quant_config.is_fp8_blockwise) {
#ifdef ENABLE_FP8
      transpose_matrix = true;
      q_b_nope_tensor = this->DequantFp8E4m3BlockWiseTensor(
          q_b_nope_tensor,
          dev_weights_map_.at(q_b_nope_name + "_scale_inv"),
          dev_rank,
          new_deepseek_v3_config);
      kv_b_nope_tensor = this->DequantFp8E4m3BlockWiseTensor(
          kv_b_nope_tensor,
          dev_weights_map_.at(kv_b_nope_name + "_scale_inv"),
          dev_rank,
          new_deepseek_v3_config);
      v_head_tensor = this->DequantFp8E4m3BlockWiseTensor(
          v_head_tensor,
          dev_weights_map_.at(v_head_proj_name + "_scale_inv"),
          dev_rank,
          new_deepseek_v3_config);
      attn_o_tensor = this->DequantFp8E4m3BlockWiseTensor(
          attn_o_tensor,
          dev_weights_map_.at(attn_o_proj_name + "_scale_inv"),
          dev_rank,
          new_deepseek_v3_config);
#endif
    }

    // for compatible with deepseek v2
    q = (q == 0) ? static_cast<size_t>(q_b_nope_tensor.shape[0]) : q;
    Tensor w_q_uk_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
                                  {q, head_num_per_tp * kv_lora_rank}, dev_rank);
    Tensor w_uv_o_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
                                  {head_num_per_tp * kv_lora_rank, hidden_units}, dev_rank);
    MlaAbsorbWeight<T>(q_b_nope_tensor.GetPtr<void>(), kv_b_nope_tensor.GetPtr<void>(),
                      v_head_tensor.GetPtr<void>(), attn_o_tensor.GetPtr<void>(),
                      w_q_uk_tensor.GetPtr<void>(), w_uv_o_tensor.GetPtr<void>(),
                      q, head_num_per_tp, qk_nope_head_dim, kv_lora_rank, hidden_units,
                      transpose_matrix, dev_rank, context_->GetMemoryManageStreams()[dev_rank].Get());

    std::string w_q_uk_name = replace_first(q_b_nope_name, "q_b_nope_proj", "w_q_uk");
    std::string w_uv_o_name = replace_first(q_b_nope_name, "q_b_nope_proj", "w_uv_o");
    if (new_deepseek_v3_config->quant_config.is_fp8_blockwise) {
#ifdef ENABLE_FP8
      std::swap(w_q_uk_tensor.shape[0], w_q_uk_tensor.shape[1]);
      std::swap(w_uv_o_tensor.shape[0], w_uv_o_tensor.shape[1]);

      auto [quant_w_q_uk_tensor, w_q_uk_weight_scale] =
          this->QuantFp8E4m3BlockWiseTensor(w_q_uk_tensor, dev_rank, new_deepseek_v3_config);
      std::string w_q_uk_scale_name = w_q_uk_name + "_scale_inv";
      dev_weights_map_.insert({w_q_uk_name, quant_w_q_uk_tensor});
      dev_weights_map_.insert({w_q_uk_scale_name, w_q_uk_weight_scale});

      auto [quant_w_uv_o_tensor, w_uv_o_weight_scale] =
          this->QuantFp8E4m3BlockWiseTensor(w_uv_o_tensor, dev_rank, new_deepseek_v3_config);
      std::string w_uv_o_scale_name = w_uv_o_name + "_scale_inv";
      dev_weights_map_.insert({w_uv_o_name, quant_w_uv_o_tensor});
      dev_weights_map_.insert({w_uv_o_scale_name, w_uv_o_weight_scale});
#endif
    } else {
      dev_weights_map_.insert({w_q_uk_name, w_q_uk_tensor});
      dev_weights_map_.insert({w_uv_o_name, w_uv_o_tensor});
    }
  }
  return Status();
}
#endif

#ifdef ENABLE_FP8
template <typename T>
Tensor NewDeepSeekV3WeightImpl<T>::DequantFp8E4m3BlockWiseTensor(const Tensor & weight_tensor,
                                                                  const Tensor & weight_scale_tensor,
                                                                  int dev_rank,
                                                                  const std::shared_ptr<NewDeepSeekV3Config> &
                                                                      new_deepseek_v3_config) {
  Tensor dequant_weight_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
                                        weight_tensor.shape, dev_rank);
  DequantFp8E4m3BlockWise<T>(
      weight_tensor.GetPtr<void>(),
      weight_scale_tensor.GetPtr<void>(),
      dequant_weight_tensor.GetPtr<void>(),
      weight_tensor.shape[0],
      weight_tensor.shape[1],
      new_deepseek_v3_config->quant_config.weight_block_size[1],
      context_->GetMemoryManageStreams()[dev_rank].Get());

  return dequant_weight_tensor;
}

template <typename T>
std::pair<Tensor, Tensor> NewDeepSeekV3WeightImpl<T>::QuantFp8E4m3BlockWiseTensor(Tensor & weight_tensor,
                                                                      int dev_rank,
                                                                      const std::shared_ptr<NewDeepSeekV3Config> &
                                                                        new_deepseek_v3_config) {
  size_t scale_shape_0 = static_cast<size_t>(DivRoundUp(weight_tensor.shape[0],
      new_deepseek_v3_config->quant_config.weight_block_size[0]));
  size_t scale_shape_1 = static_cast<size_t>(DivRoundUp(weight_tensor.shape[1],
      new_deepseek_v3_config->quant_config.weight_block_size[1]));
  Tensor quant_weight_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
                                DataType::TYPE_FP8_E4M3, weight_tensor.shape, dev_rank);
  Tensor weight_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                                      {scale_shape_0, scale_shape_1}, dev_rank);
#ifdef ENABLE_FP8_TORCH
  ScaledQuantizeFp8E4m3<T>(
    static_cast<T*>(weight_tensor.GetPtr<void>()),
    quant_weight_tensor.GetPtr<void>(),
    static_cast<float*>(weight_scale_tensor.GetPtr<void>()),
    new_deepseek_v3_config->quant_config.weight_block_size,
    quant_weight_tensor.shape[0],
    quant_weight_tensor.shape[1],
    dev_rank);
#endif
  return std::make_pair(quant_weight_tensor, weight_scale_tensor);
}

#ifdef ENABLE_FP8_TORCH
template <typename T>
Status NewDeepSeekV3WeightImpl<T>::ProcessMlaFp8E4m3BlockWiseScaleOfWeight(
                                          std::unordered_set<std::string> & processed_weights,
                                          std::unordered_set<std::string> & dequant_weights,
                                          int dev_rank,
                                          const std::shared_ptr<NewDeepSeekV3Config> &
                                            new_deepseek_v3_config,
                                          std::unordered_map<std::string, Tensor> &
                                            device_model_weights) {
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
      Tensor dequant_q_b_proj = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
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
      Tensor dequant_q_b_nope_proj = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
                                            {head_num_tp * qk_nope_head_dim, dequant_q_b_proj.shape[1]}, dev_rank);
      size_t nope_dst_pitch =
          qk_nope_head_dim * dequant_q_b_proj.shape[1] * GetTypeSize(new_deepseek_v3_config->weight_data_type);
      size_t src_pitch = (qk_nope_head_dim + qk_rope_head_dim) * dequant_q_b_proj.shape[1] *
                          GetTypeSize(new_deepseek_v3_config->weight_data_type);
      Memcpy2DAsync(dequant_q_b_nope_proj.GetPtr<void>(), nope_dst_pitch,
                    dequant_q_b_proj.GetPtr<void>(), src_pitch, nope_dst_pitch, head_num_tp,
                    MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);

      Tensor dequant_q_b_rope_proj = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
                                            {head_num_tp * qk_rope_head_dim, dequant_q_b_proj.shape[1]}, dev_rank);
      size_t rope_dst_pitch = qk_rope_head_dim * dequant_q_b_rope_proj.shape[1]*
                              GetTypeSize(new_deepseek_v3_config->weight_data_type);
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
      Tensor dequant_kv_b_proj = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
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
      Tensor dequant_kv_b_nope_proj = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
                                            {head_num_tp * qk_nope_head_dim, dequant_kv_b_proj.shape[1]}, dev_rank);
      size_t nope_dst_pitch =
          qk_nope_head_dim * dequant_kv_b_proj.shape[1] * GetTypeSize(new_deepseek_v3_config->weight_data_type);
      size_t src_pitch = (qk_nope_head_dim + v_head_dim) * dequant_kv_b_proj.shape[1] *
                          GetTypeSize(new_deepseek_v3_config->weight_data_type);
      Memcpy2DAsync(dequant_kv_b_nope_proj.GetPtr<void>(), nope_dst_pitch,
                    dequant_kv_b_proj.GetPtr<void>(), src_pitch, nope_dst_pitch, head_num_tp,
                    MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);

      Tensor dequant_v_head_proj = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
                                          {head_num_tp * v_head_dim, dequant_kv_b_proj.shape[1]}, dev_rank);
      size_t v_head_dst_pitch =
          v_head_dim * dequant_kv_b_proj.shape[1] * GetTypeSize(new_deepseek_v3_config->weight_data_type);
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

template <typename T>
bool NewDeepSeekV3WeightImpl<T>::LoadMoeFp8E4m3BlockWiseScale(const std::string & host_weight_name,
                                  const Tensor & host_weight_tensor,
                                  int dev_rank,
                                  std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                  std::unordered_map<std::string, Tensor> & device_model_weights) {
  if (new_deepseek_v3_config->quant_config.method != QUANT_FP8_E4M3 &&
    !new_deepseek_v3_config->quant_config.is_fp8_blockwise) {
    return false;
  }
  if (host_weight_name.find(".experts.") == std::string::npos ||
      (host_weight_name.find(".weight_scale") == std::string::npos &&
       host_weight_name.find(".input_scale") == std::string::npos)) {
    return false;
  }
  if (host_weight_tensor.dtype != DataType::TYPE_FP32) {
    KLLM_THROW("Not support data type of scale: " + host_weight_name);
  }

  constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();
  size_t layer_idx = kInvalidIndex;
  size_t expert_idx = kInvalidIndex;
  GetExpertsIdx(host_weight_name, layer_idx, expert_idx);
  if (layer_idx == kInvalidIndex || expert_idx == kInvalidIndex) {
    KLLM_LOG_ERROR << "Failed to find valid indices for weight: " << host_weight_name;
    return false;
  }

  size_t block_n = new_deepseek_v3_config->quant_config.weight_block_size[0];
  size_t block_k = new_deepseek_v3_config->quant_config.weight_block_size[1];
  size_t moe_inter_size_per_rank = DivRoundUp(
    new_deepseek_v3_config->moe_config.moe_inter_size,
    new_deepseek_v3_config->moe_tensor_para_size);
  if (moe_inter_size_per_rank % block_n != 0) {
    KLLM_THROW(fmt::format(
        "The moe_inter_size_per_rank of gate's and up's weight = {}, is not divisible by weight quant block_n = {}",
        moe_inter_size_per_rank, block_n));
  }
  if (context_->GetTensorParallelSize() > 1 && moe_inter_size_per_rank % block_k != 0) {
    KLLM_THROW(fmt::format(
        "The moe_inter_size_per_rank of down's weight = {}, is not divisible by weight quant block_k = {}",
        moe_inter_size_per_rank, block_k));
  }
  size_t hidden_units = new_deepseek_v3_config->hidden_units;
  std::vector<size_t> up_gate_experts_scale_shape = {
    new_deepseek_v3_config->moe_config.num_experts,
    static_cast<size_t>(DivRoundUp(moe_inter_size_per_rank, block_n) * 2),
    static_cast<size_t>(DivRoundUp(hidden_units, block_k))};
  std::vector<size_t> down_experts_scale_shape = {
    new_deepseek_v3_config->moe_config.num_experts,
    static_cast<size_t>(DivRoundUp(hidden_units, block_n)),
    static_cast<size_t>(DivRoundUp(moe_inter_size_per_rank, block_k))};
  // For up_gate proj scale
  if (host_weight_name.find("up_proj.weight_scale") != std::string::npos ||
      host_weight_name.find("gate_proj.weight_scale") != std::string::npos) {
    if (host_weight_tensor.shape[0] !=
          static_cast<size_t>(DivRoundUp(new_deepseek_v3_config->moe_config.moe_inter_size, block_n))) {
      KLLM_THROW(fmt::format("Not support shape of scale: {}", host_weight_name));
    }
    std::string up_gate_experts_scale_name =
        "model.layers." + std::to_string(layer_idx) + ".mlp.experts.up_gate_proj.weight_scale_inv";
    if (device_model_weights.find(up_gate_experts_scale_name) == device_model_weights.end()) {
      device_model_weights[up_gate_experts_scale_name] = Tensor(MemoryLocation::LOCATION_DEVICE,
        DataType::TYPE_FP32, up_gate_experts_scale_shape, dev_rank);
    }

    size_t expert_scale_pitch = up_gate_experts_scale_shape[1] / 2 *
      up_gate_experts_scale_shape[2] * GetTypeSize(host_weight_tensor.dtype);
    size_t double_expert_scale_pitch = expert_scale_pitch * 2;
    size_t src_upgate_offset =
      new_deepseek_v3_config->moe_tensor_para_size > 1 ? dev_rank * expert_scale_pitch : 0;
    if (host_weight_name.find(".gate_proj.") != std::string::npos) {
      MemcpyAsync(device_model_weights.at(up_gate_experts_scale_name).GetPtr<void>() +
                  static_cast<size_t>(expert_idx) * double_expert_scale_pitch,
                  host_weight_tensor.GetPtr<void>() + src_upgate_offset, expert_scale_pitch, MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[dev_rank]);
    } else if (host_weight_name.find(".up_proj.") != std::string::npos) {
      MemcpyAsync(device_model_weights.at(up_gate_experts_scale_name).GetPtr<void>() +
                  static_cast<size_t>(expert_idx) * double_expert_scale_pitch + expert_scale_pitch,
                  host_weight_tensor.GetPtr<void>() + src_upgate_offset, expert_scale_pitch, MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[dev_rank]);
    }
  }
  if (host_weight_name.find(".down_proj.weight_scale") != std::string::npos) {
    std::string down_experts_scale_name =
      "model.layers." + std::to_string(layer_idx) + ".mlp.experts.down_proj.weight_scale_inv";
    if (device_model_weights.find(down_experts_scale_name) == device_model_weights.end()) {
      device_model_weights[down_experts_scale_name] = Tensor(MemoryLocation::LOCATION_DEVICE,
          DataType::TYPE_FP32, down_experts_scale_shape, dev_rank);
    }

    size_t dst_pitch = down_experts_scale_shape[2] * GetTypeSize(host_weight_tensor.dtype);
    size_t src_pitch = down_experts_scale_shape[2] *
      new_deepseek_v3_config->moe_tensor_para_size * GetTypeSize(host_weight_tensor.dtype);
    size_t expert_scale_pitch =
      down_experts_scale_shape[2] * down_experts_scale_shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t src_down_offset = dev_rank * dst_pitch;

    Memcpy2DAsync(device_model_weights.at(down_experts_scale_name).GetPtr<void>() +
                  static_cast<size_t>(expert_idx) * expert_scale_pitch,
                  dst_pitch, host_weight_tensor.GetPtr<void>() + src_down_offset,
                  src_pitch, dst_pitch, down_experts_scale_shape[1],
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  }
  return true;
}

template <typename T>
bool NewDeepSeekV3WeightImpl<T>::LoadMlaFp8E4m3BlockWiseScale(const std::string & host_weight_name,
                                    const Tensor & host_weight_tensor,
                                    int dev_rank,
                                    std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                    std::unordered_map<std::string, Tensor> & device_model_weights) {
  // q_a_proj: Weights are not split and copied to each GPU
  // q_b_proj: Weights are split, requiring dequantization, splitting, requantization, and distribution to each GPU
  // kv_a_proj: Copied to each GPU, split directly on each GPU without dequantization
  // kv_b_proj: Weights are split, requiring dequantization, splitting, requantization, and distribution to each GPU
  if (new_deepseek_v3_config->quant_config.method != QUANT_FP8_E4M3 &&
      !new_deepseek_v3_config->quant_config.is_fp8_blockwise) {
    return false;
  }
  if (host_weight_name.find(".self_attn.") == std::string::npos ||
      (host_weight_name.find(".weight_scale") == std::string::npos &&
      host_weight_name.find(".input_scale") == std::string::npos)) {
    return false;
  }
  // scale is float scalar
  if (host_weight_tensor.dtype != TYPE_FP32) {
    KLLM_THROW("Not support data type of scale:" + host_weight_name);
  }

  if (host_weight_name.find(".q_a_proj.weight_scale_inv") != std::string::npos) {
    // Weights are not split and are copied to each GPU.
    Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
      host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

    MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(),
                host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[host_weight_name] = dev_tensor;
  }
  // For q_b_proj scale
  if (host_weight_name.find(".q_b_proj.weight_scale_inv") != std::string::npos) {
    size_t para_pitch =
        static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize())) *
        host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t tensor_para_offset = dev_rank * para_pitch;
    std::vector<size_t> q_b_proj_scale_shape = {
      static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0],
        context_->GetTensorParallelSize())), host_weight_tensor.shape[1]
    };

    Tensor weight_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
      DataType::TYPE_FP32, q_b_proj_scale_shape, dev_rank);
    MemcpyAsync(weight_scale_tensor.GetPtr<void>(),
                host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                weight_scale_tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[host_weight_name] = weight_scale_tensor;
  }

  // For kv_a_proj scale
  if (host_weight_name.find(".kv_a_proj_with_mqa.weight_scale_inv") != std::string::npos) {
    // For kv_a_lora_proj scale
    size_t kv_lora_rank = new_deepseek_v3_config->mla_config.kv_lora_rank;
    if (kv_lora_rank % new_deepseek_v3_config->quant_config.weight_block_size[0] != 0) {
      KLLM_THROW("Not support shape of scale:" + host_weight_name);
    }
    std::string kv_a_lora_scale_name =
        host_weight_name.substr(0, host_weight_name.find_first_of('_')) + "_attn.kv_a_lora_proj.weight_scale_inv";
    size_t kv_a_lora_scale_shape_0 = kv_lora_rank / new_deepseek_v3_config->quant_config.weight_block_size[0];
    std::vector<size_t> kv_a_lora_scale_shape = { kv_a_lora_scale_shape_0, host_weight_tensor.shape[1]};
    Tensor kv_a_lora_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
      DataType::TYPE_FP32, kv_a_lora_scale_shape, dev_rank);
    MemcpyAsync(kv_a_lora_scale_tensor.GetPtr<void>(),
                host_weight_tensor.GetPtr<void>(),
                kv_a_lora_scale_tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[kv_a_lora_scale_name] = kv_a_lora_scale_tensor;

    // For kv_a_rope_proj scale
    std::string kv_a_rope_scale_name =
        host_weight_name.substr(0, host_weight_name.find_first_of('_')) + "_attn.kv_a_rope_proj.weight_scale_inv";
    size_t qk_rope_head_dim = new_deepseek_v3_config->mla_config.qk_rope_head_dim;
    size_t kv_a_rope_scale_shape_0 = DivRoundUp(qk_rope_head_dim,
      new_deepseek_v3_config->quant_config.weight_block_size[0]);
    if (kv_a_rope_scale_shape_0 + kv_a_lora_scale_shape_0 != host_weight_tensor.shape[0]) {
      KLLM_THROW("Not support shape of scale:" + host_weight_name);
    }
    std::vector<size_t> kv_a_rope_scale_shape = {kv_a_rope_scale_shape_0, host_weight_tensor.shape[1]};
    size_t kv_a_rope_allocate_size = std::max(static_cast<size_t>(256),
        kv_a_rope_scale_shape_0 * host_weight_tensor.shape[1]);
    Tensor kv_a_rope_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
      DataType::TYPE_FP32, {kv_a_rope_allocate_size}, dev_rank);
    kv_a_rope_scale_tensor.Fill(0);
    kv_a_rope_scale_tensor.shape = kv_a_rope_scale_shape;
    MemcpyAsync(kv_a_rope_scale_tensor.GetPtr<void>(),
                host_weight_tensor.GetPtr<void>() + kv_a_lora_scale_tensor.GetTotalBytes(),
                kv_a_rope_scale_tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[kv_a_rope_scale_name] = kv_a_rope_scale_tensor;
  }

  // for kv_b_proj_scale
  if (host_weight_name.find(".kv_b_proj.weight_scale_inv") != std::string::npos) {
    size_t para_pitch =
        DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize()) *
        host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t tensor_para_offset = dev_rank * para_pitch;
    std::vector<size_t> kv_b_proj_scale_shape = {
        static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize())),
        host_weight_tensor.shape[1]
    };

    Tensor weight_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE,
      DataType::TYPE_FP32, kv_b_proj_scale_shape, dev_rank);
    MemcpyAsync(weight_scale_tensor.GetPtr<void>(),
                host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                weight_scale_tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[host_weight_name] = weight_scale_tensor;
  }

  if (host_weight_name.find(".o_proj.weight_scale_inv") != std::string::npos) {
    // fp8: Transpose, then split along axis = 0, then transpose
    Tensor dev_tensor;
    TransSplitOptTrans(host_weight_tensor,
      dev_tensor, dev_rank, new_deepseek_v3_config, new_deepseek_v3_config->is_quant);

    device_model_weights[host_weight_name] = dev_tensor;
  }
  return true;
}
#endif

template class NewDeepSeekV3WeightImpl<float>;
template class NewDeepSeekV3WeightImpl<float16>;
#ifdef ENABLE_BFLOAT16
template class NewDeepSeekV3WeightImpl<bfloat16>;
#endif
}  // namespace ksana_llm
