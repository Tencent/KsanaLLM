/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_weight_impl.h"
#include <numeric>
#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {
template <typename T>
NewDeepSeekV3WeightImpl<T>::NewDeepSeekV3WeightImpl(const std::shared_ptr<Context>& context,
                                                    const RuntimeConfig& runtime_config)
    : context_(context), runtime_config_(runtime_config) {
  permute_buffers_.resize(context_->GetTensorParallelSize());
  tensor_buffers_.resize(context_->GetTensorParallelSize());
}

template <typename T>
Tensor& NewDeepSeekV3WeightImpl<T>::GetTempTensor(const std::vector<size_t>& shape, DataType data_type, int dev_rank) {
  const size_t key = std::accumulate(shape.begin(), shape.end(), GetTypeSize(data_type), std::multiplies<size_t>());
  if (tensor_buffers_[dev_rank].find(key) == tensor_buffers_[dev_rank].end()) {
    tensor_buffers_[dev_rank][key] = Tensor(MemoryLocation::LOCATION_DEVICE, data_type, shape, dev_rank, nullptr,
                                            &(context_->GetMemoryManageStreams()[dev_rank]));
  }
  Tensor& temp_tensor = tensor_buffers_[dev_rank].at(key);
  temp_tensor.dtype = data_type;
  temp_tensor.shape = shape;
  return temp_tensor;
}

template <typename T>
Status NewDeepSeekV3WeightImpl<T>::PermuteWeight(Tensor& input_tensor, const std::vector<size_t>& permutation,
                                                 int dev_rank) {
  const size_t key = input_tensor.GetTotalBytes();
  if (permute_buffers_[dev_rank].find(key) == permute_buffers_[dev_rank].end()) {
    permute_buffers_[dev_rank][key] = Tensor(MemoryLocation::LOCATION_DEVICE, input_tensor.dtype, input_tensor.shape,
                                             dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
  }
  Tensor& permute_tensor = permute_buffers_[dev_rank].at(key);
  permute_tensor.dtype = input_tensor.dtype;
  permute_tensor.shape = input_tensor.shape;
  Permute(input_tensor, permute_tensor, permutation, context_->GetMemoryManageStreams()[dev_rank]);
  for (size_t i = 0; i < permutation.size(); i++) {
    permute_tensor.shape[i] = input_tensor.shape[permutation[i]];
  }
  std::swap(input_tensor, permute_tensor);
  return Status();
}

// TODO(huicongyao): compare TransplitOptTrans with Memcpy2dAsync
template <typename T>
Status NewDeepSeekV3WeightImpl<T>::TransSplitOptTrans(const Tensor& host_weight_tensor, Tensor& output_tensor,
                                                      int dev_rank,
                                                      std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
                                                      size_t para_size, bool transpose) {
  Tensor& full_dev_tensor = GetTempTensor(host_weight_tensor.shape, host_weight_tensor.dtype, dev_rank);

  MemcpyAsync(full_dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
              MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  PermuteWeight(full_dev_tensor, {1, 0}, dev_rank);

  std::vector<size_t> slice_shape = {static_cast<size_t>(DivRoundUp(full_dev_tensor.shape[0], para_size)),
                                     full_dev_tensor.shape[1]};

  Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, slice_shape, dev_rank, nullptr,
                             &(context_->GetMemoryManageStreams()[dev_rank]));

  size_t slice_offset = dev_tensor.GetTotalBytes() * (dev_rank % para_size);
  size_t slice_bytes = dev_tensor.GetTotalBytes();
  if (static_cast<size_t>(dev_rank) == para_size - 1) {
    slice_bytes = host_weight_tensor.GetTotalBytes() - slice_offset;
  }

  MemcpyAsync(dev_tensor.GetPtr<void>(), full_dev_tensor.GetPtr<void>() + slice_offset, slice_bytes,
              MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);

  if (transpose) {
    PermuteWeight(dev_tensor, {1, 0}, dev_rank);
  }
  output_tensor = dev_tensor;

  return Status();
}

template <typename T>
Status NewDeepSeekV3WeightImpl<T>::SplitOptTrans(const Tensor& host_weight_tensor, Tensor& output_tensor, int dev_rank,
                                                 std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
                                                 size_t para_size, bool transpose) {
  std::vector<size_t> slice_shape = {static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], para_size)),
                                     host_weight_tensor.shape[1]};
  Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, slice_shape, dev_rank, nullptr,
                             &(context_->GetMemoryManageStreams()[dev_rank]));

  size_t slice_offset = dev_tensor.GetTotalBytes() * (dev_rank % para_size);
  size_t slice_bytes = dev_tensor.GetTotalBytes();
  if (static_cast<size_t>(dev_rank) == para_size - 1) {
    slice_bytes = host_weight_tensor.GetTotalBytes() - slice_offset;
  }

  MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + slice_offset, slice_bytes,
              MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  if (transpose) {
    PermuteWeight(dev_tensor, {1, 0}, dev_rank);
  }
  output_tensor = dev_tensor;

  return Status();
}

template <typename T>
Status NewDeepSeekV3WeightImpl<T>::GetExpertsIdx(const std::string& expert_name, int32_t& layer_idx,
                                                 int32_t& expert_idx) {
  // Get the index of the moe layer and the index of each expert
  static const std::regex re(R"(\d+)");
  std::sregex_iterator next(expert_name.begin(), expert_name.end(), re);
  std::sregex_iterator end;
  if (next != end) {
    std::smatch match = *next;
    layer_idx = std::stoi(match.str());
    next++;
    match = *next;
    expert_idx = std::stoi(match.str());
  } else {
    layer_idx = -1;
    expert_idx = -1;
  }
  return Status();
}

template <typename T>
Status NewDeepSeekV3WeightImpl<T>::ProcessGateUpProjWeight(
    const std::string& file_weight_name, const Tensor& dev_tensor,
    std::unordered_map<std::string, Tensor>& device_model_weights, int dev_rank, bool is_quant_weight) {
  int concat_offset = 0;
  std::string replacement = "gate_up_proj";
  std::string file_weight_name_replace;
  if (file_weight_name.find("gate_proj") != std::string::npos) {
    concat_offset = 0;
    static const std::regex pattern("gate_proj");
    file_weight_name_replace = std::regex_replace(file_weight_name, pattern, replacement);
  } else {
    concat_offset = 1;
    static const std::regex pattern("up_proj");
    file_weight_name_replace = std::regex_replace(file_weight_name, pattern, replacement);
  }

  if (device_model_weights.find(file_weight_name_replace) == device_model_weights.end()) {
    device_model_weights[file_weight_name_replace] =
        Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype, {dev_tensor.shape[0] * 2, dev_tensor.shape[1]},
               dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
  }
  Tensor& gate_up_proj_tensor = device_model_weights[file_weight_name_replace];
  size_t total_bytes = gate_up_proj_tensor.GetTotalBytes() / 2;
  if (is_quant_weight) {
    MemcpyAsync(gate_up_proj_tensor.GetPtr<void>() + concat_offset * total_bytes, dev_tensor.GetPtr<void>(),
                total_bytes, MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  } else {
    gate_up_proj_tensor.shape = {dev_tensor.shape[0], dev_tensor.shape[1] * 2};
    size_t dst_pitch = gate_up_proj_tensor.shape[1] * GetTypeSize(gate_up_proj_tensor.dtype);
    size_t src_pitch = dev_tensor.shape[1] * GetTypeSize(dev_tensor.dtype);
    Memcpy2DAsync(gate_up_proj_tensor.GetPtr<void>() + concat_offset * src_pitch, dst_pitch, dev_tensor.GetPtr<void>(),
                  src_pitch, src_pitch, dev_tensor.shape[0], MEMCPY_DEVICE_TO_DEVICE,
                  context_->GetMemoryManageStreams()[dev_rank]);
  }

  return Status();
}

#ifdef ENABLE_FP8
template <typename T>
Tensor NewDeepSeekV3WeightImpl<T>::DequantFp8E4m3BlockWiseTensor(
    const Tensor& weight_tensor, const Tensor& weight_scale_tensor, int dev_rank,
    const std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) {
  Tensor dequant_weight_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type, weight_tensor.shape, dev_rank,
             nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
  DequantFp8E4m3BlockWise<T>(weight_tensor.GetPtr<void>(), weight_scale_tensor.GetPtr<void>(),
                             dequant_weight_tensor.GetPtr<void>(), weight_tensor.shape[0], weight_tensor.shape[1],
                             new_deepseek_v3_config->quant_config.weight_block_size[1],
                             context_->GetMemoryManageStreams()[dev_rank].Get());

  return dequant_weight_tensor;
}

template <typename T>
std::pair<Tensor, Tensor> NewDeepSeekV3WeightImpl<T>::QuantFp8E4m3BlockWiseTensor(
    Tensor& weight_tensor, int dev_rank, const std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) {
  size_t scale_shape_0 = static_cast<size_t>(
      DivRoundUp(weight_tensor.shape[0], new_deepseek_v3_config->quant_config.weight_block_size[0]));
  size_t scale_shape_1 = static_cast<size_t>(
      DivRoundUp(weight_tensor.shape[1], new_deepseek_v3_config->quant_config.weight_block_size[1]));
  Tensor quant_weight_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP8_E4M3, weight_tensor.shape,
                                      dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
  Tensor weight_scale_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, {scale_shape_0, scale_shape_1}, dev_rank, nullptr,
             &(context_->GetMemoryManageStreams()[dev_rank]));
#  ifdef ENABLE_FP8_TORCH
  ScaledQuantizeFp8E4m3<T>(static_cast<T*>(weight_tensor.GetPtr<void>()), quant_weight_tensor.GetPtr<void>(),
                           static_cast<float*>(weight_scale_tensor.GetPtr<void>()),
                           new_deepseek_v3_config->quant_config.weight_block_size, quant_weight_tensor.shape[0],
                           quant_weight_tensor.shape[1], dev_rank);
#  endif
  return std::make_pair(quant_weight_tensor, weight_scale_tensor);
}

#  ifdef ENABLE_FP8_TORCH
template <typename T>
Status NewDeepSeekV3WeightImpl<T>::PostProcessFp8E4m3BlockWiseQuantWeights(
    std::unordered_map<std::string, Tensor>& device_model_weights, int dev_rank,
    std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) {
  size_t tp_size = context_->GetAttentionTensorParallelSize();
  size_t qk_nope_head_dim = new_deepseek_v3_config->mla_config.qk_nope_head_dim;
  size_t v_head_dim = new_deepseek_v3_config->mla_config.v_head_dim;
  size_t head_num_tp = static_cast<size_t>(DivRoundUp(new_deepseek_v3_config->head_num, tp_size));

  // Filter name of weights to be processed
  std::vector<std::string> weight_names;
  for (const auto& [weight_name, _] : device_model_weights) {
    if (weight_name.find("_scale_inv") != std::string::npos) {
      continue;
    }
    if (weight_name.find(".q_b_proj.weight") != std::string::npos ||
        weight_name.find(".kv_b_proj.weight") != std::string::npos) {
      weight_names.push_back(weight_name);
    }
  }

  for (const auto& weight_name : weight_names) {
    if (weight_name.find(".q_b_proj.weight") != std::string::npos) {
      const std::string weight_scale_name = weight_name + "_scale_inv";
      const std::string quant_nope_rope_weight_name =
          weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.q_b_nope_rope_proj.weight";
      const std::string quant_nope_rope_weight_scale_name =
          weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.q_b_nope_rope_proj.weight_scale_inv";
      device_model_weights[quant_nope_rope_weight_name] = std::move(device_model_weights[weight_name]);
      device_model_weights[quant_nope_rope_weight_scale_name] = std::move(device_model_weights[weight_scale_name]);
      device_model_weights.erase(weight_name);
      device_model_weights.erase(weight_scale_name);
      continue;
    }
    if (weight_name.find(".kv_b_proj.weight") != std::string::npos) {
      const std::string weight_scale_name = weight_name + "_scale_inv";
      Tensor& quant_weight = device_model_weights.at(weight_name);
      Tensor& quant_weight_scale = device_model_weights.at(weight_scale_name);
      // Dequant kv_b_proj
      Tensor& dequant_kv_b_proj = GetTempTensor(quant_weight.shape, new_deepseek_v3_config->weight_data_type, dev_rank);
      DequantFp8E4m3BlockWise<T>(quant_weight.GetPtr<void>(), quant_weight_scale.GetPtr<void>(),
                                 dequant_kv_b_proj.GetPtr<void>(), quant_weight.shape[0], quant_weight.shape[1],
                                 new_deepseek_v3_config->quant_config.weight_block_size[1],
                                 context_->GetMemoryManageStreams()[dev_rank].Get());

      // split dequant kv_b_proj
      if (dequant_kv_b_proj.shape[0] != (head_num_tp * (qk_nope_head_dim + v_head_dim))) {
        KLLM_THROW(fmt::format("Not support shape of dequant weight: {}", weight_name));
      }
      Tensor dequant_kv_b_nope_proj = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
                                             {head_num_tp * qk_nope_head_dim, dequant_kv_b_proj.shape[1]}, dev_rank,
                                             nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
      size_t nope_dst_pitch =
          qk_nope_head_dim * dequant_kv_b_proj.shape[1] * GetTypeSize(new_deepseek_v3_config->weight_data_type);
      size_t src_pitch = (qk_nope_head_dim + v_head_dim) * dequant_kv_b_proj.shape[1] *
                         GetTypeSize(new_deepseek_v3_config->weight_data_type);
      Memcpy2DAsync(dequant_kv_b_nope_proj.GetPtr<void>(), nope_dst_pitch, dequant_kv_b_proj.GetPtr<void>(), src_pitch,
                    nope_dst_pitch, head_num_tp, MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);

      Tensor& dequant_v_head_proj = GetTempTensor({head_num_tp * v_head_dim, dequant_kv_b_proj.shape[1]},
                                                  new_deepseek_v3_config->weight_data_type, dev_rank);
      size_t v_head_dst_pitch =
          v_head_dim * dequant_kv_b_proj.shape[1] * GetTypeSize(new_deepseek_v3_config->weight_data_type);
      Memcpy2DAsync(dequant_v_head_proj.GetPtr<void>(), v_head_dst_pitch,
                    dequant_kv_b_proj.GetPtr<void>() + nope_dst_pitch, src_pitch, v_head_dst_pitch, head_num_tp,
                    MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      // For the latest weight absorption version process
      // Copy dequant kv_b_proj to w_uk_t
      Tensor w_uk_t_tensor = dequant_kv_b_nope_proj;
      w_uk_t_tensor.shape = {head_num_tp, qk_nope_head_dim, dequant_kv_b_nope_proj.shape[1]};
      std::string w_uk_t_name = weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.w_uk_t.weight";
      device_model_weights[w_uk_t_name] = w_uk_t_tensor;

      // Permute dequant_nope_weight to w_uv
      Tensor w_uv_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, w_uk_t_tensor.dtype,
                                  {head_num_tp, dequant_v_head_proj.shape[1], v_head_dim}, dev_rank, nullptr,
                                  &(context_->GetMemoryManageStreams()[dev_rank]));
      dequant_v_head_proj.shape = {head_num_tp, v_head_dim, dequant_v_head_proj.shape[1]};
      Permute(dequant_v_head_proj, w_uv_tensor, {0, 2, 1}, context_->GetMemoryManageStreams()[dev_rank]);
      std::string w_uv_name = weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.w_uv.weight";
      device_model_weights[w_uv_name] = w_uv_tensor;
      dequant_v_head_proj.shape = {head_num_tp * v_head_dim, dequant_kv_b_proj.shape[1]};

      // Quant kv_b_nope_proj and v_head_proj
      std::string quant_nope_weight_name =
          weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.kv_b_nope_proj.weight";
      std::string quant_nope_weight_scale_name =
          weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.kv_b_nope_proj.weight_scale_inv";
      size_t weight_scale_shape_0 =
          DivRoundUp(dequant_kv_b_nope_proj.shape[0], new_deepseek_v3_config->quant_config.weight_block_size[0]);
      size_t weight_scale_shape_1 =
          DivRoundUp(dequant_kv_b_nope_proj.shape[1], new_deepseek_v3_config->quant_config.weight_block_size[1]);
      Tensor quant_nope_weight =
          Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP8_E4M3, dequant_kv_b_nope_proj.shape, dev_rank,
                 nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
      Tensor quant_nope_weight_scale =
          Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, {weight_scale_shape_0, weight_scale_shape_1},
                 dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
      ScaledQuantizeFp8E4m3<T>(static_cast<T*>(dequant_kv_b_nope_proj.GetPtr<void>()), quant_nope_weight.GetPtr<void>(),
                               static_cast<float*>(quant_nope_weight_scale.GetPtr<void>()),
                               new_deepseek_v3_config->quant_config.weight_block_size, dequant_kv_b_nope_proj.shape[0],
                               dequant_kv_b_nope_proj.shape[1], dev_rank);
      device_model_weights[quant_nope_weight_name] = quant_nope_weight;
      device_model_weights[quant_nope_weight_scale_name] = quant_nope_weight_scale;

      std::string quant_v_head_weight_name =
          weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.v_head_proj.weight";
      std::string quant_v_head_weight_scale_name =
          weight_name.substr(0, weight_name.find_first_of('_')) + "_attn.v_head_proj.weight_scale_inv";
      Tensor quant_v_head_weight =
          Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP8_E4M3, dequant_v_head_proj.shape, dev_rank, nullptr,
                 &(context_->GetMemoryManageStreams()[dev_rank]));
      weight_scale_shape_0 =
          DivRoundUp(dequant_v_head_proj.shape[0], new_deepseek_v3_config->quant_config.weight_block_size[0]);
      weight_scale_shape_1 =
          DivRoundUp(dequant_v_head_proj.shape[1], new_deepseek_v3_config->quant_config.weight_block_size[1]);
      Tensor quant_v_head_weight_scale =
          Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, {weight_scale_shape_0, weight_scale_shape_1},
                 dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
      ScaledQuantizeFp8E4m3<T>(static_cast<T*>(dequant_v_head_proj.GetPtr<void>()), quant_v_head_weight.GetPtr<void>(),
                               static_cast<float*>(quant_v_head_weight_scale.GetPtr<void>()),
                               new_deepseek_v3_config->quant_config.weight_block_size, dequant_v_head_proj.shape[0],
                               dequant_v_head_proj.shape[1], dev_rank);
      device_model_weights[quant_v_head_weight_name] = quant_v_head_weight;
      device_model_weights[quant_v_head_weight_scale_name] = quant_v_head_weight_scale;
      device_model_weights.erase(weight_name);
      device_model_weights.erase(weight_scale_name);
      continue;
    }
  }
  return Status();
}
#  endif

template <typename T>
bool NewDeepSeekV3WeightImpl<T>::LoadMoeFp8E4m3BlockWiseScale(
    const std::string& host_weight_name, const Tensor& host_weight_tensor, int dev_rank,
    std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
    std::unordered_map<std::string, Tensor>& device_model_weights, int32_t expert_idx) {
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

  int32_t layer_idx = -1, expert_idx_ = -1;
  GetExpertsIdx(host_weight_name, layer_idx, expert_idx_);
  if (layer_idx == -1 || expert_idx_ == -1) {
    KLLM_LOG_ERROR << "Failed to find valid indices for weight: " << host_weight_name;
    return false;
  }

  size_t block_n = new_deepseek_v3_config->quant_config.weight_block_size[0];
  size_t block_k = new_deepseek_v3_config->quant_config.weight_block_size[1];
  size_t moe_inter_size_per_rank =
      DivRoundUp(new_deepseek_v3_config->moe_config.moe_inter_size, new_deepseek_v3_config->moe_tensor_para_size);
  ExpertParallelConfig& expert_parallel_config = new_deepseek_v3_config->expert_parallel_config;
  size_t global_expert_para_size = expert_parallel_config.expert_world_size * expert_parallel_config.expert_para_size;
  size_t num_experts_per_rank = DivRoundUp(new_deepseek_v3_config->moe_config.num_experts, global_expert_para_size);
  if (moe_inter_size_per_rank % block_n != 0) {
    KLLM_THROW(fmt::format(
        "The moe_inter_size_per_rank of gate's and up's weight = {}, is not divisible by weight quant block_n = {}",
        moe_inter_size_per_rank, block_n));
  }
  if (context_->GetTensorParallelSize() > 1 && moe_inter_size_per_rank % block_k != 0) {
    KLLM_THROW(
        fmt::format("The moe_inter_size_per_rank of down's weight = {}, is not divisible by weight quant block_k = {}",
                    moe_inter_size_per_rank, block_k));
  }
  size_t hidden_units = new_deepseek_v3_config->hidden_units;
  std::vector<size_t> up_gate_experts_scale_shape = {
      num_experts_per_rank, static_cast<size_t>(DivRoundUp(moe_inter_size_per_rank, block_n) * 2),
      static_cast<size_t>(DivRoundUp(hidden_units, block_k))};
  std::vector<size_t> down_experts_scale_shape = {num_experts_per_rank,
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
      device_model_weights[up_gate_experts_scale_name] =
          Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, up_gate_experts_scale_shape, dev_rank, nullptr,
                 &(context_->GetMemoryManageStreams()[dev_rank]));
    }

    size_t expert_scale_pitch =
        up_gate_experts_scale_shape[1] / 2 * up_gate_experts_scale_shape[2] * GetTypeSize(host_weight_tensor.dtype);
    size_t double_expert_scale_pitch = expert_scale_pitch * 2;
    size_t src_upgate_offset = new_deepseek_v3_config->moe_tensor_para_size > 1
                                   ? (dev_rank / expert_parallel_config.expert_para_size) * expert_scale_pitch
                                   : 0;
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
      device_model_weights[down_experts_scale_name] =
          Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, down_experts_scale_shape, dev_rank, nullptr,
                 &(context_->GetMemoryManageStreams()[dev_rank]));
    }

    size_t dst_pitch = down_experts_scale_shape[2] * GetTypeSize(host_weight_tensor.dtype);
    size_t src_pitch = down_experts_scale_shape[2] * new_deepseek_v3_config->moe_tensor_para_size *
                       GetTypeSize(host_weight_tensor.dtype);
    size_t expert_scale_pitch =
        down_experts_scale_shape[2] * down_experts_scale_shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t src_down_offset = new_deepseek_v3_config->moe_tensor_para_size > 1
                                 ? (dev_rank / expert_parallel_config.expert_para_size) * dst_pitch
                                 : 0;

    Memcpy2DAsync(device_model_weights.at(down_experts_scale_name).GetPtr<void>() +
                      static_cast<size_t>(expert_idx) * expert_scale_pitch,
                  dst_pitch, host_weight_tensor.GetPtr<void>() + src_down_offset, src_pitch, dst_pitch,
                  down_experts_scale_shape[1], MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  }
  return true;
}

template <typename T>
bool NewDeepSeekV3WeightImpl<T>::LoadMlaFp8E4m3BlockWiseScale(
    const std::string& host_weight_name, const Tensor& host_weight_tensor, int dev_rank,
    std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
    std::unordered_map<std::string, Tensor>& device_model_weights) {
  // indexer.wk: Weights are not split and copied to each GPU
  // indexer.wq_b: Weights are not split and copied to each GPU
  // q_a_proj: Weights are not split and copied to each GPU
  // q_b_proj: Weights are split, requiring dequantization, splitting, requantization, and distribution to each GPU
  // kv_a_proj: Copied to each GPU, split directly on each GPU without dequantization
  // kv_b_proj: Weights are split, requiring dequantization, splitting, requantization, and distribution to each GPU
  size_t attn_tp_size = context_->GetAttentionTensorParallelSize();
  size_t attn_tp_rank = dev_rank % attn_tp_size;
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

  const std::string fused_name = ".fused_lora_a_proj.weight_scale_inv";
  const size_t kv_lora_rank = new_deepseek_v3_config->mla_config.kv_lora_rank;
  const size_t qk_rope_head_dim = new_deepseek_v3_config->mla_config.qk_rope_head_dim;
  const size_t q_lora_rank = new_deepseek_v3_config->mla_config.q_lora_rank;
  const size_t weight_block_size = new_deepseek_v3_config->quant_config.weight_block_size[0];

  // For indexer.wk/indexer.wq_b scale
  if (host_weight_name.find(".indexer.wk.weight_scale_inv") != std::string::npos ||
      host_weight_name.find(".indexer.wq_b.weight_scale_inv") != std::string::npos) {
    Tensor weight_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, host_weight_tensor.shape,
                                        dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
    MemcpyAsync(weight_scale_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(),
                weight_scale_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[host_weight_name] = weight_scale_tensor;
  }
  // For q_a_proj scale
  if (host_weight_name.find(".q_a_proj.weight_scale_inv") != std::string::npos) {
    std::vector<size_t> fused_q_a_proj_scale_shape = {
        DivRoundUp(q_lora_rank + kv_lora_rank + qk_rope_head_dim, weight_block_size), host_weight_tensor.shape[1]};
    std::string fused_tensor_name = GetReplacedName(host_weight_name, ".q_a_proj.weight_scale_inv", fused_name);
    Tensor fused_tensor;
    if (device_model_weights.find(fused_tensor_name) != device_model_weights.end()) {
      fused_tensor = device_model_weights.at(fused_tensor_name);
    } else {
      fused_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, fused_q_a_proj_scale_shape,
                            dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
      fused_tensor.Fill(0);
    }
    MemcpyAsync(fused_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);

    if (device_model_weights.find(fused_tensor_name) == device_model_weights.end()) {
      device_model_weights[fused_tensor_name] = fused_tensor;
    }
  }
  // For q_b_proj scale
  if (host_weight_name.find(".q_b_proj.weight_scale_inv") != std::string::npos) {
    size_t para_pitch = static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], attn_tp_size)) *
                        host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
    size_t tensor_para_offset = attn_tp_rank * para_pitch;
    std::vector<size_t> q_b_proj_scale_shape = {
        static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], attn_tp_size)), host_weight_tensor.shape[1]};

    Tensor weight_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, q_b_proj_scale_shape,
                                        dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
    MemcpyAsync(weight_scale_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                weight_scale_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[host_weight_name] = weight_scale_tensor;
  }

  // For kv_a_proj scale
  if (host_weight_name.find(".kv_a_proj_with_mqa.weight_scale_inv") != std::string::npos) {
    std::vector<size_t> fused_q_a_proj_scale_shape = {
        DivRoundUp(q_lora_rank + kv_lora_rank + qk_rope_head_dim, weight_block_size), host_weight_tensor.shape[1]};
    std::string fused_tensor_name =
        GetReplacedName(host_weight_name, ".kv_a_proj_with_mqa.weight_scale_inv", fused_name);
    Tensor fused_tensor;
    if (device_model_weights.find(fused_tensor_name) != device_model_weights.end()) {
      fused_tensor = device_model_weights.at(fused_tensor_name);
    } else {
      fused_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, fused_q_a_proj_scale_shape,
                            dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
      fused_tensor.Fill(0);
    }
    const size_t offset =
        (q_lora_rank / weight_block_size) * host_weight_tensor.shape[1] * host_weight_tensor.GetDTypeSize();
    MemcpyAsync(fused_tensor.GetPtr<void>() + offset, host_weight_tensor.GetPtr<void>(),
                host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);

    if (device_model_weights.find(fused_tensor_name) == device_model_weights.end()) {
      device_model_weights[fused_tensor_name] = fused_tensor;
    }
  }
  // For kv_b_proj scale
  if (host_weight_name.find(".kv_b_proj.weight_scale_inv") != std::string::npos) {
    size_t para_pitch = DivRoundUp(host_weight_tensor.shape[0], attn_tp_size) * host_weight_tensor.shape[1] *
                        GetTypeSize(host_weight_tensor.dtype);
    size_t tensor_para_offset = attn_tp_rank * para_pitch;
    std::vector<size_t> kv_b_proj_scale_shape = {
        static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], attn_tp_size)), host_weight_tensor.shape[1]};

    Tensor weight_scale_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, kv_b_proj_scale_shape,
                                        dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
    MemcpyAsync(weight_scale_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                weight_scale_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[dev_rank]);
    device_model_weights[host_weight_name] = weight_scale_tensor;
  }
  // For o_proj scale
  if (host_weight_name.find(".o_proj.weight_scale_inv") != std::string::npos) {
    // fp8: Transpose, then split along axis = 0, then transpose
    size_t split_size =
        runtime_config_.enable_o_proj_out_of_dp ? new_deepseek_v3_config->tensor_para_size : attn_tp_size;
    Tensor dev_tensor;
    TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, new_deepseek_v3_config, split_size,
                       new_deepseek_v3_config->is_quant);

    device_model_weights[host_weight_name] = dev_tensor;
  }
  return true;
}
#endif

template <typename T>
Status NewDeepSeekV3WeightImpl<T>::LoadInt4QuantWeight(std::unordered_map<std::string, Tensor>& host_gptq_weights,
                                                       int dev_rank,
                                                       std::unordered_map<std::string, Tensor>& device_model_weights,
                                                       std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
                                                       std::vector<std::vector<int>>& expert_map) {
  SetDevice(dev_rank);
  // Prapare params that need to use
  int32_t layer_idx = -1, expert_idx = -1;
  size_t num_experts = new_deepseek_v3_config->moe_config.num_experts;
  // moe combines tensor parallel and moe parallel
  ExpertParallelConfig& expert_parallel_config = new_deepseek_v3_config->expert_parallel_config;
  size_t global_expert_para_size = expert_parallel_config.expert_world_size * expert_parallel_config.expert_para_size;
  size_t num_experts_per_rank = DivRoundUp(num_experts, global_expert_para_size);
  size_t moe_tp_rank = dev_rank / expert_parallel_config.expert_para_size;

  size_t moe_inter_size_per_rank =
      DivRoundUp(new_deepseek_v3_config->moe_config.moe_inter_size, new_deepseek_v3_config->moe_tensor_para_size);
  size_t hidden_units = new_deepseek_v3_config->hidden_units;

  size_t attn_tp_size = context_->GetAttentionTensorParallelSize();
  size_t kv_lora_rank = new_deepseek_v3_config->mla_config.kv_lora_rank;
  size_t qk_rope_head_dim = new_deepseek_v3_config->mla_config.qk_rope_head_dim;
  size_t qk_nope_head_dim = new_deepseek_v3_config->mla_config.qk_nope_head_dim;
  size_t v_head_dim = new_deepseek_v3_config->mla_config.v_head_dim;
  size_t head_num = new_deepseek_v3_config->head_num;
  size_t head_num_tp = DivRoundUp(head_num, attn_tp_size);

  for (const auto& [host_weight_name, host_weight_tensor] : host_gptq_weights) {
    // Skip qzeros and g_idx
    if (host_weight_name.find(".qzeros") != std::string::npos || host_weight_name.find(".g_idx") != std::string::npos) {
      continue;
    }
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, dtype: {}, shape: {}", dev_rank,
                                  host_weight_name, host_weight_tensor.dtype,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));
    size_t mlp_tensor_para_size = runtime_config_.enable_full_shared_expert ? 1 : context_->GetTensorParallelSize();
    // 1, quant MLP layers
    if (host_weight_name.find(".mlp.down_proj.") != std::string::npos ||
        host_weight_name.find(".mlp.shared_expert.down_proj.") != std::string::npos) {
      Tensor dev_tensor;
      SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, new_deepseek_v3_config, mlp_tensor_para_size, false);
      device_model_weights[host_weight_name] = dev_tensor;
      continue;
    }
    if (host_weight_name.find(".mlp.up_proj.") != std::string::npos ||
        host_weight_name.find(".mlp.shared_expert.up_proj.") != std::string::npos ||
        host_weight_name.find(".mlp.shared_expert.gate_proj.") != std::string::npos ||
        host_weight_name.find(".mlp.gate_proj.") != std::string::npos) {
      Tensor dev_tensor;
      TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, new_deepseek_v3_config, mlp_tensor_para_size, true);
      device_model_weights[host_weight_name] = dev_tensor;
      continue;
    }

    // 2, quant MOE layers
    if (host_weight_name.find(".experts.") != std::string::npos) {
      STATUS_CHECK_RETURN(GetExpertsIdx(host_weight_name, layer_idx, expert_idx));
      if (layer_idx < 0 || expert_idx < 0) {
        continue;
      }
      int expert_idx_ = expert_map[layer_idx][expert_idx];
      if (expert_idx_ < 0) {
        continue;
      }
      if (host_weight_name.find(".input_scale") != std::string::npos) {
        size_t input_scale_size = GetTypeSize(host_weight_tensor.dtype);
        if (host_weight_name.find(".up_proj.") != std::string::npos ||
            host_weight_name.find(".gate_proj.") != std::string::npos) {
          std::string up_gate_experts_name =
              fmt::format("model.layers.{}.mlp.experts.up_gate_proj.input_scale", layer_idx);
          if (device_model_weights.find(up_gate_experts_name) == device_model_weights.end()) {
            device_model_weights[up_gate_experts_name] =
                Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, {num_experts_per_rank, 2}, dev_rank,
                       nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
          }
          Tensor& up_gate_experts_tensor = device_model_weights.at(up_gate_experts_name);
          if (host_weight_name.find(".up_proj.") != std::string::npos) {
            MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + expert_idx_ * input_scale_size * 2,
                        host_weight_tensor.template GetPtr<void>(), input_scale_size, MEMCPY_HOST_TO_DEVICE,
                        context_->GetMemoryManageStreams()[dev_rank]);
          } else if (host_weight_name.find(".gate_proj.") != std::string::npos) {
            MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + expert_idx_ * input_scale_size * 2 + input_scale_size,
                        host_weight_tensor.template GetPtr<void>(), input_scale_size, MEMCPY_HOST_TO_DEVICE,
                        context_->GetMemoryManageStreams()[dev_rank]);
          }
          continue;
        }
        if (host_weight_name.find(".down_proj.") != std::string::npos) {
          std::string down_experts_name = fmt::format("model.layers.{}.mlp.experts.down_proj.input_scale", layer_idx);
          if (device_model_weights.find(down_experts_name) == device_model_weights.end()) {
            device_model_weights[down_experts_name] =
                Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, {num_experts_per_rank, 1}, dev_rank,
                       nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
          }
          Tensor& down_experts_tensor = device_model_weights.at(down_experts_name);
          MemcpyAsync(down_experts_tensor.GetPtr<void>() + expert_idx_ * input_scale_size,
                      host_weight_tensor.template GetPtr<void>(), input_scale_size, MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);
          continue;
        }
      }

      bool is_qweight = host_weight_name.find(".qweight") != std::string::npos;
      Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape,
                                 dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
      MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.template GetPtr<void>(),
                  host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[dev_rank]);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
      if (!new_deepseek_v3_config->GetGptqQuantConfig().input_scale) {
        PermuteWeight(dev_tensor, {1, 0}, dev_rank);
      }
      if (host_weight_name.find(".up_proj.") != std::string::npos ||
          host_weight_name.find(".gate_proj.") != std::string::npos) {
        std::string up_gate_experts_name =
            fmt::format("model.layers.{}.mlp.experts.up_gate_proj.{}", layer_idx, is_qweight ? "weight" : "scales");

        if (is_qweight && dev_tensor.dtype == DataType::TYPE_INT32) {
          // an uint8 weight actually contains two int4 weights
          dev_tensor.dtype = DataType::TYPE_UINT8;
          dev_tensor.shape[1] = dev_tensor.shape[1] * 4;  // int32->uint8
        }

        size_t expert_pitch = moe_inter_size_per_rank * dev_tensor.shape[1] * GetTypeSize(dev_tensor.dtype);
        size_t double_expert_pitch = expert_pitch * 2;
        size_t src_upgate_offset = new_deepseek_v3_config->moe_tensor_para_size > 1 ? moe_tp_rank * expert_pitch : 0;
        if (device_model_weights.find(up_gate_experts_name) == device_model_weights.end()) {
          device_model_weights[up_gate_experts_name] =
              Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype,
                     {num_experts_per_rank, moe_inter_size_per_rank * 2, dev_tensor.shape[1]}, dev_rank, nullptr,
                     &(context_->GetMemoryManageStreams()[dev_rank]));
        }
        Tensor& up_gate_experts_tensor = device_model_weights.at(up_gate_experts_name);
        // TODO(jinxcwu) up_gate权重名字有问题，实际是up在后，gate在前
        if (host_weight_name.find(".up_proj.") != std::string::npos) {
          MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + expert_idx_ * double_expert_pitch + expert_pitch,
                      dev_tensor.GetPtr<void>() + src_upgate_offset, expert_pitch, MEMCPY_DEVICE_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);
        } else if (host_weight_name.find(".gate_proj.") != std::string::npos) {
          MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + expert_idx_ * double_expert_pitch,
                      dev_tensor.GetPtr<void>() + src_upgate_offset, expert_pitch, MEMCPY_DEVICE_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);
        }
        continue;
      }
      if (host_weight_name.find(".down_proj.") != std::string::npos) {
        std::string down_experts_name =
            fmt::format("model.layers.{}.mlp.experts.down_proj.{}", layer_idx, is_qweight ? "weight" : "scales");

        if (is_qweight && dev_tensor.dtype == DataType::TYPE_INT32) {
          // an uint8 weight actually contains two int4 weights
          dev_tensor.dtype = DataType::TYPE_UINT8;
          dev_tensor.shape[1] = dev_tensor.shape[1] * 4;  // int32->uint8
        }

        size_t down_inter_size_per_rank = DivRoundUp(dev_tensor.shape[1], new_deepseek_v3_config->moe_tensor_para_size);
        if (device_model_weights.find(down_experts_name) == device_model_weights.end()) {
          device_model_weights[down_experts_name] =
              Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype,
                     {num_experts_per_rank, new_deepseek_v3_config->hidden_units, down_inter_size_per_rank}, dev_rank,
                     nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
        }

        size_t dst_pitch = down_inter_size_per_rank * GetTypeSize(dev_tensor.dtype);
        size_t src_pitch = dev_tensor.shape[1] * GetTypeSize(dev_tensor.dtype);
        size_t expert_pitch = down_inter_size_per_rank * hidden_units * GetTypeSize(dev_tensor.dtype);
        size_t src_down_offset = new_deepseek_v3_config->moe_tensor_para_size > 1 ? moe_tp_rank * dst_pitch : 0;
        Tensor& down_experts_tensor = device_model_weights.at(down_experts_name);
        Memcpy2DAsync(down_experts_tensor.GetPtr<void>() + expert_idx_ * expert_pitch, dst_pitch,
                      dev_tensor.GetPtr<void>() + src_down_offset, src_pitch, dst_pitch, hidden_units,
                      MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
        continue;
      }
    }

    // TODO(huicongyao): add fuse lora for gptq int4 and pad fuse weight to a multiple of 128
    const std::string fused_lora_a_weight_name = ".fused_lora_a_proj.";
    // 3. quant MLA layers
    if (host_weight_name.find(".self_attn.") != std::string::npos) {
      if (host_weight_name.find(".self_attn.q_a_proj.") != std::string::npos) {
        Tensor dev_tensor =
            Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

        MemcpyAsync(dev_tensor.template GetPtr<void>(), host_weight_tensor.template GetPtr<void>(),
                    host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[dev_rank]);
        device_model_weights[host_weight_name] = dev_tensor;
        continue;
      }
      if (host_weight_name.find(".self_attn.q_b_proj.") != std::string::npos) {
        if ((qk_nope_head_dim + qk_rope_head_dim) * head_num != host_weight_tensor.shape[1]) {
          KLLM_THROW(fmt::format(
              "The shape of the 0th dim of the weight named '{} ({})' is not equal to the sum of qk_nope_head_dim {} "
              "and qk_rope_head_dim {}.",
              host_weight_name, host_weight_tensor.shape[0], qk_nope_head_dim, qk_rope_head_dim));
        }
        // Split along axis=1 first
        Tensor dev_tensor;
        TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, new_deepseek_v3_config, attn_tp_size, true);
        std::string q_b_nope_rope_name = GetReplacedName(host_weight_name, ".q_b_proj.", ".q_b_nope_rope_proj.");
        device_model_weights[q_b_nope_rope_name] = dev_tensor;
        continue;
      }
      if (host_weight_name.find(".self_attn.kv_a_proj_with_mqa.") != std::string::npos) {
        if ((kv_lora_rank + qk_rope_head_dim) != host_weight_tensor.shape[1]) {
          KLLM_THROW(fmt::format(
              "The shape of the 0th dim of the weight named `{}` is not equal to the sum of kv_lora_rank {} "
              "and qk_rope_head_dim {}.",
              host_weight_name, kv_lora_rank, qk_rope_head_dim));
        }

        std::string kv_a_lora_name = GetReplacedName(host_weight_name, ".kv_a_proj_with_mqa.", ".kv_a_lora_proj.");
        Tensor kv_a_lora_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                         {host_weight_tensor.shape[0], kv_lora_rank}, dev_rank);
        size_t host_tensor_pitch = host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
        size_t kv_a_lora_pitch = kv_lora_rank * GetTypeSize(host_weight_tensor.dtype);
        Memcpy2DAsync(kv_a_lora_tensor.GetPtr<void>(), kv_a_lora_pitch, host_weight_tensor.template GetPtr<void>(),
                      host_tensor_pitch, kv_a_lora_pitch, host_weight_tensor.shape[0], MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);
        device_model_weights[kv_a_lora_name] = kv_a_lora_tensor;

        std::string kv_a_rope_name = GetReplacedName(host_weight_name, ".kv_a_proj_with_mqa.", ".kv_a_rope_proj.");
        Tensor kv_a_rope_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                         {host_weight_tensor.shape[0], qk_rope_head_dim}, dev_rank);
        size_t kv_a_rope_pitch = qk_rope_head_dim * GetTypeSize(host_weight_tensor.dtype);
        Memcpy2DAsync(kv_a_rope_tensor.GetPtr<void>(), kv_a_rope_pitch,
                      host_weight_tensor.template GetPtr<void>() + kv_a_lora_pitch, host_tensor_pitch, kv_a_rope_pitch,
                      host_weight_tensor.shape[0], MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
        device_model_weights[kv_a_rope_name] = kv_a_rope_tensor;
        continue;
      }
      if (host_weight_name.find(".self_attn.kv_b_proj.") != std::string::npos) {
        if (head_num * (qk_nope_head_dim + v_head_dim) != host_weight_tensor.shape[1]) {
          KLLM_THROW(fmt::format(
              "The shape of the 0th dim of the weight named '{}' is not equal to the sum of qk_nope_head_dim {} "
              "and v_head_dim {}.",
              host_weight_name, kv_lora_rank, qk_rope_head_dim));
        }
        Tensor dev_tensor;
        TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, new_deepseek_v3_config, attn_tp_size, true);

        // For kv_b_nope_proj
        std::string kv_b_nope_name = GetReplacedName(host_weight_name, ".kv_b_proj.", ".kv_b_nope_proj.");
        Tensor kv_b_nope_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                         {host_weight_tensor.shape[0], head_num_tp * qk_nope_head_dim}, dev_rank,
                                         nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
        size_t nope_dst_pitch = qk_nope_head_dim * GetTypeSize(host_weight_tensor.dtype);
        size_t src_pitch = (qk_nope_head_dim + v_head_dim) * GetTypeSize(host_weight_tensor.dtype);
        Memcpy2DAsync(kv_b_nope_tensor.GetPtr<void>(), nope_dst_pitch, dev_tensor.template GetPtr<void>(), src_pitch,
                      nope_dst_pitch, host_weight_tensor.shape[0] * head_num_tp, MEMCPY_DEVICE_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);
        device_model_weights[kv_b_nope_name] = kv_b_nope_tensor;

        // For v_head_proj
        std::string v_head_name = GetReplacedName(host_weight_name, ".kv_b_proj.", ".v_head_proj.");
        Tensor v_head_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                      {host_weight_tensor.shape[0], head_num_tp * v_head_dim}, dev_rank, nullptr,
                                      &(context_->GetMemoryManageStreams()[dev_rank]));
        size_t v_head_dst_pitch = v_head_dim * GetTypeSize(host_weight_tensor.dtype);
        Memcpy2DAsync(v_head_tensor.GetPtr<void>(), v_head_dst_pitch,
                      dev_tensor.template GetPtr<void>() + nope_dst_pitch, src_pitch, v_head_dst_pitch,
                      host_weight_tensor.shape[0] * head_num_tp, MEMCPY_DEVICE_TO_DEVICE,
                      context_->GetMemoryManageStreams()[dev_rank]);
        device_model_weights[v_head_name] = v_head_tensor;
        continue;
      }
      if (host_weight_name.find(".self_attn.o_proj.") != std::string::npos) {
        Tensor dev_tensor;
        SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, new_deepseek_v3_config, attn_tp_size,
                      !new_deepseek_v3_config->is_quant);
        device_model_weights[host_weight_name] = dev_tensor;
        continue;
      }
    }
  }
  return Status();
}

template <typename T>
Status NewDeepSeekV3WeightImpl<T>::PostProcessInt4QuantWeights(
    std::unordered_map<std::string, Tensor>& device_model_weights, int dev_rank,
    std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) {
  if (!new_deepseek_v3_config->ContainGptqWeights()) {
    return Status();
  }

  QuantConfig quant_config = new_deepseek_v3_config->GetGptqQuantConfig();
  std::vector<std::string> experts_weight_names;
  std::vector<std::string> qweight_names;
  qweight_names.reserve(device_model_weights.size());
  for (const auto& [weight_name, weight_tensor] : device_model_weights) {
    if (weight_name.find(".experts.") != std::string::npos && new_deepseek_v3_config->IsWeightMatchGptq(weight_name)) {
      experts_weight_names.push_back(weight_name);
    }
    if (weight_name.find(".qweight") != std::string::npos) {
      qweight_names.push_back(weight_name);
    }
  }
  // bind for moe weights
  for (const auto& experts_weight_name : experts_weight_names) {
    if (experts_weight_name.find(".weight") == std::string::npos) {
      continue;
    }
    const std::string scales_name = GetReplacedName(experts_weight_name, "weight", "scales");
    auto itr = device_model_weights.find(scales_name);
    if (itr != device_model_weights.end()) {
      Tensor& scale = itr->second;
#ifdef ENABLE_CUDA
      const std::string input_scale_name = GetReplacedName(experts_weight_name, "weight", "input_scale");
      if (device_model_weights.find(input_scale_name) != device_model_weights.end()) {
        // 强制要求TP=EP
        if (new_deepseek_v3_config->moe_tensor_para_size != 1) {
          KLLM_THROW("Currently, W4AFP8 strictly enforces TP=EP.");
        }
        // cutlass moe int4 算子对权重的特殊处理
        if (runtime_config_.w4afp8_moe_backend == W4AFP8_MOE_BACKEND::Default) {
          // scale转换
          std::vector<size_t> interleaves = GetCutlassMoeInterleave(new_deepseek_v3_config->hidden_units,
                                                                    new_deepseek_v3_config->moe_config.moe_inter_size);
          size_t interleave = interleaves[0];  // fc31也就是up_gate用0
          if (experts_weight_name.find(".down_proj.") != std::string::npos) {
            interleave = interleaves[1];  // fc2也就是down用1
          }
          std::vector<size_t> origin_shape = scale.shape;
          scale.shape = {origin_shape[0], origin_shape[1], origin_shape[2] / interleave, interleave};
          PermuteWeight(scale, {0, 2, 1, 3}, dev_rank);
          scale.shape = {origin_shape[0], origin_shape[2] / interleave, origin_shape[1] * interleave};
        }
        // 计算input_scale的最大值
        Tensor& input_scale = device_model_weights.at(input_scale_name);
        torch::Tensor input_scale_torch = GetTorchTensorFromTensor(input_scale, dev_rank);
        float input_scale_max = torch::max(input_scale_torch).item<float>();
        auto float_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, dev_rank);
        // 生成act_scale
        torch::Tensor act_scale_tensor = torch::full({1}, 1 / input_scale_max, float_option).to(GetTorchDataType<T>());
        const std::string act_scale_name = GetReplacedName(experts_weight_name, "weight", "act_scale");
        device_model_weights[act_scale_name] = Tensor(MemoryLocation::LOCATION_DEVICE, GetDataType<T>(), {1}, dev_rank,
                                                      nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
        MemcpyAsync(device_model_weights[act_scale_name].GetPtr<void>(), act_scale_tensor.data_ptr(),
                    act_scale_tensor.numel() * act_scale_tensor.element_size(), MEMCPY_DEVICE_TO_DEVICE,
                    context_->GetMemoryManageStreams()[dev_rank]);
        device_model_weights.at(experts_weight_name).input_scales = &(device_model_weights[act_scale_name]);
        // 生成alpha
        torch::Tensor alpha_tensor = torch::full({input_scale_torch.size(0), 1}, input_scale_max, float_option);
        const std::string alpha_name = GetReplacedName(experts_weight_name, "weight", "alpha");
        device_model_weights[alpha_name] =
            Tensor(MemoryLocation::LOCATION_DEVICE, GetDataType<float>(), {input_scale_torch.size(0), 1}, dev_rank,
                   nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
        MemcpyAsync(device_model_weights[alpha_name].GetPtr<void>(), alpha_tensor.data_ptr(),
                    alpha_tensor.numel() * alpha_tensor.element_size(), MEMCPY_DEVICE_TO_DEVICE,
                    context_->GetMemoryManageStreams()[dev_rank]);
        device_model_weights.at(experts_weight_name).input_alpha = &(device_model_weights[alpha_name]);
      }
#endif
      device_model_weights.at(experts_weight_name).scales = &scale;
    } else {
      KLLM_THROW(fmt::format("Can not find scales for weight {}", experts_weight_name));
    }
  }

  // pack and bind for all no moe weights
  for (const auto& qweight_name : qweight_names) {
    const std::string scales_name = GetReplacedName(qweight_name, "qweight", "scales");
    std::string weight_name_replaced = GetReplacedName(qweight_name, "qweight", "weight");

    KLLM_LOG_DEBUG << fmt::format(
        "Pack and bind int4 quant weight: {}, on dev_rank: {}, "
        "shape: {}",
        qweight_name, dev_rank, Vector2Str(std::vector<size_t>(device_model_weights.at(qweight_name).shape)));
#ifdef ENABLE_CUDA
    // pack for non experts weights
    Tensor& qweight_tensor = device_model_weights.at(qweight_name);

    // tmp fix for fused_lora_a_proj
    if (qweight_tensor.shape[1] % 128) {
      const std::string perm_name = GetReplacedName(qweight_name, "qweight", "perm");
      Tensor perm_tensor = device_model_weights.find(perm_name) == device_model_weights.end()
                               ? Tensor()
                               : device_model_weights.at(perm_name);
      Tensor packed_tensor =
          MarlinPackGptqWeight(qweight_tensor, perm_tensor, dev_rank, quant_config.bits, 32 / quant_config.bits);

      device_model_weights[weight_name_replaced] = packed_tensor;
      device_model_weights.erase(qweight_name);

      Tensor scales_tensor = device_model_weights.at(scales_name);
      int k = quant_config.group_size * scales_tensor.shape[scales_tensor.shape.size() - 2];
      int n = scales_tensor.shape[scales_tensor.shape.size() - 1];
      Tensor scales_permute = MarlinPermuteScales(scales_tensor, dev_rank, k, n, quant_config.group_size);
      device_model_weights[scales_name] = scales_permute;
    } else {
      Tensor packed_tensor = MachetePackWeight(qweight_tensor, dev_rank, quant_config.method);
      device_model_weights[weight_name_replaced] = packed_tensor;
      // remove previous qweight
      device_model_weights.erase(qweight_name);
    }
#endif

    // bind
    auto scale_itr = device_model_weights.find(scales_name);
    if (scale_itr != device_model_weights.end()) {
      device_model_weights.at(weight_name_replaced).scales = &(scale_itr->second);
    } else {
      KLLM_THROW(fmt::format("Can not find scales for weight: {}", qweight_name));
    }
  }

  // Absorb V2 for GPTQ
  if (new_deepseek_v3_config->quant_config.method == QUANT_GPTQ) {
    size_t kv_lora_rank = new_deepseek_v3_config->mla_config.kv_lora_rank;
    size_t qk_nope_head_dim = new_deepseek_v3_config->mla_config.qk_nope_head_dim;
    size_t v_head_dim = new_deepseek_v3_config->mla_config.v_head_dim;
    size_t attn_tp_size = context_->GetAttentionTensorParallelSize();
    size_t head_num_tp = DivRoundUp(new_deepseek_v3_config->head_num, attn_tp_size);
    std::vector<std::string> kv_b_nope_weights;
    for (const auto& [weight_name, weight_tensor] : device_model_weights) {
      if (weight_name.find(".kv_b_nope_proj.weight") != std::string::npos) {
        kv_b_nope_weights.push_back(weight_name);
      }
    }
    for (const auto& kv_b_nope_name : kv_b_nope_weights) {
#ifdef ENABLE_CUDA
      const std::string v_head_name = GetReplacedName(kv_b_nope_name, ".kv_b_nope_proj.", ".v_head_proj.");
      KLLM_LOG_DEBUG << fmt::format("Start to dequant {} and {} on dev_rank: {}", kv_b_nope_name, v_head_name,
                                    dev_rank);
      // For W_UK_T
      Tensor& kv_b_nope_tensor = device_model_weights.at(kv_b_nope_name);
      Tensor dequant_kv_b_nope_tensor = DequantGptqWeight(kv_b_nope_tensor, dev_rank, new_deepseek_v3_config);
      std::string w_uk_t_name = GetReplacedName(kv_b_nope_name, ".kv_b_nope_proj.", ".w_uk_t.");
      dequant_kv_b_nope_tensor.shape = {kv_lora_rank, head_num_tp, qk_nope_head_dim};
      PermuteWeight(dequant_kv_b_nope_tensor, {1, 2, 0}, dev_rank);
      device_model_weights[w_uk_t_name] = dequant_kv_b_nope_tensor;

      // For W_UV
      Tensor& v_head_tensor = device_model_weights.at(v_head_name);
      Tensor dequant_v_head_tensor = DequantGptqWeight(v_head_tensor, dev_rank, new_deepseek_v3_config);
      std::string w_uv_name = GetReplacedName(v_head_name, ".v_head_proj.", ".w_uv.");
      dequant_v_head_tensor.shape = {kv_lora_rank, head_num_tp, v_head_dim};
      PermuteWeight(dequant_v_head_tensor, {1, 0, 2}, dev_rank);
      device_model_weights[w_uv_name] = dequant_v_head_tensor;
#endif
    }
  }
  return Status();
}

#ifdef ENABLE_CUDA
template <typename T>
std::vector<size_t> NewDeepSeekV3WeightImpl<T>::GetCutlassMoeInterleave(size_t hidden_size_per_partition,
                                                                        size_t intermediate_size_per_partition) {
  uint32_t sm = context_->ext->GetComputeCapacity();
  std::vector<size_t> interleave;
  if (sm == 90) {
    interleave.clear();
    std::vector<size_t> k_shapes = {hidden_size_per_partition, intermediate_size_per_partition};
    for (const size_t& k_shape : k_shapes) {
      if (k_shape % 512 == 0) {
        interleave.push_back(4);
      } else if (k_shape % 256 == 0) {
        interleave.push_back(2);
      } else if (k_shape % 128 == 0) {
        interleave.push_back(1);
      } else {
        KLLM_LOG_ERROR << fmt::format("K shape is required to be multiple of 128, received {}.", k_shape);
      }
    }
  } else {
    KLLM_LOG_ERROR << fmt::format("W4AFP8 MoE is unsupported on SM{}.", sm);
  }
  return interleave;
}

template <typename T>
torch::Tensor NewDeepSeekV3WeightImpl<T>::GetTorchTensorFromTensor(const Tensor& tensor, int dev_rank) {
  auto options = torch::TensorOptions().device(torch::kCUDA, dev_rank).dtype(GetTorchTypeFromDataType(tensor.dtype));
  std::vector<size_t> tensor_shape = tensor.shape;
  torch::Tensor tensor_gpu =
      torch::from_blob(tensor.GetPtr<void>(), std::vector<int64_t>(tensor_shape.begin(), tensor_shape.end()), options);
  return tensor_gpu;
}

template <typename T>
Tensor NewDeepSeekV3WeightImpl<T>::MachetePackWeight(Tensor& weight, int dev_rank, QuantMode quant_method) {
  llm_kernels::nvidia::vllm_dtype::ScalarType weight_type =
      (quant_method == QUANT_GPTQ) ? llm_kernels::nvidia::vllm_dtype::kU4B8 : llm_kernels::nvidia::vllm_dtype::kU4;
  Tensor prepack_weight = Tensor(MemoryLocation::LOCATION_DEVICE, weight.dtype, weight.shape, dev_rank, nullptr,
                                 &(context_->GetMemoryManageStreams()[dev_rank]));
  PermuteWeight(weight, {1, 0}, dev_rank);
  InvokeMachetePrepackWeight(weight.template GetPtr<void>(), {weight.shape[1], weight.shape[0]},
                             prepack_weight.GetPtr<void>(), GetMacheteDataType<T>(), weight_type,
                             GetMacheteDataType<T>(), context_->GetMemoryManageStreams()[dev_rank].Get());
  return prepack_weight;
}

template <typename T>
Tensor NewDeepSeekV3WeightImpl<T>::MarlinPackGptqWeight(Tensor& qweight, Tensor& perm, int dev_rank, int bits,
                                                        int pack_factor) {
  Tensor processed_tensor;
  bool has_perm = perm.GetElementNumber() != 0;
  if (qweight.shape.size() == 2) {
    int64_t num_experts = 1;
    int64_t k = static_cast<int64_t>(qweight.shape[0]) * pack_factor;
    int64_t n = static_cast<int64_t>(qweight.shape[1]);
    auto repack_shape_i64 = GetMarlinGptqRepackMeta(k, n, bits);
    std::vector<size_t> repack_shape = std::vector<size_t>(repack_shape_i64.begin(), repack_shape_i64.end());
    processed_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, qweight.dtype, repack_shape, dev_rank, nullptr,
                              &(context_->GetMemoryManageStreams()[dev_rank]));
    InvokeMarlinGptqRepack(qweight.GetPtr<void>(), has_perm ? perm.GetPtr<void>() : nullptr,
                           processed_tensor.GetPtr<void>(), num_experts, k, n, bits, has_perm, dev_rank,
                           context_->GetMemoryManageStreams()[dev_rank].Get());
  } else if (qweight.shape.size() == 3) {
    int64_t num_experts = static_cast<int64_t>(qweight.shape[0]);
    int64_t k = static_cast<int64_t>(qweight.shape[1]) * pack_factor;
    int64_t n = static_cast<int64_t>(qweight.shape[2]);
    auto repack_shape_i64 = GetMarlinGptqRepackMeta(k, n, bits);
    std::vector<size_t> repack_shape = std::vector<size_t>(repack_shape_i64.begin(), repack_shape_i64.end());
    processed_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, qweight.dtype, repack_shape, dev_rank, nullptr,
                              &(context_->GetMemoryManageStreams()[dev_rank]));
    InvokeMarlinGptqRepack(qweight.GetPtr<void>(), has_perm ? perm.GetPtr<void>() : nullptr,
                           processed_tensor.GetPtr<void>(), num_experts, k, n, bits, has_perm, dev_rank,
                           context_->GetMemoryManageStreams()[dev_rank].Get());
  }
  return processed_tensor;
}

template <typename T>
Tensor NewDeepSeekV3WeightImpl<T>::MarlinPermuteScales(Tensor& s, int dev_rank, int k, int n, int group_size) {
  Tensor permute_s = Tensor(MemoryLocation::LOCATION_DEVICE, s.dtype, s.shape, dev_rank, nullptr,
                            &(context_->GetMemoryManageStreams()[dev_rank]));
  if (s.shape.size() == 2) {
    InvokeMarlinPermuteScales<T>(context_->GetMemoryManageStreams()[dev_rank].Get(), s.GetPtr<void>(),
                                 permute_s.GetPtr<void>(), k, n, group_size);
  } else if (s.shape.size() == 3) {  // first dim is num_experts
    for (size_t i = 0; i < s.shape[0]; i++) {
      InvokeMarlinPermuteScales<T>(context_->GetMemoryManageStreams()[dev_rank].Get(), s.GetPtr<void>() + i * k * n,
                                   permute_s.GetPtr<void>() + i * k * n, k, n, group_size);
    }
  }
  return permute_s;
}

// origin weight matrix @ identity matrix = dequantized weight matrix
template <typename T>
Tensor NewDeepSeekV3WeightImpl<T>::DequantGptqWeight(Tensor& qweight, int dev_rank,
                                                     std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) {
  QuantConfig quant_config = new_deepseek_v3_config->GetGptqQuantConfig();
  size_t input_size_per_tp = qweight.shape[0];
  size_t pack_factor = 32 / quant_config.bits;
  Tensor eye_matrix;
  Tensor dequant_weight;
  if (new_deepseek_v3_config->ContainGptqWeights()) {
    eye_matrix = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
                        {input_size_per_tp * pack_factor, input_size_per_tp * pack_factor}, dev_rank, nullptr,
                        &(context_->GetMemoryManageStreams()[dev_rank]));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InitIdentityMatrixAdaptive<T>(
        reinterpret_cast<T*>(eye_matrix.GetPtr<void>()), eye_matrix.shape[0], eye_matrix.shape[1],
        context_->GetMemoryManageStreams()[dev_rank].Get()));
    dequant_weight = Tensor(MemoryLocation::LOCATION_DEVICE, new_deepseek_v3_config->weight_data_type,
                            {input_size_per_tp * pack_factor, qweight.shape[1]}, dev_rank, nullptr,
                            &(context_->GetMemoryManageStreams()[dev_rank]));
  }
  if (new_deepseek_v3_config->quant_config.backend == MACHETE_LINEAR_BACKEND &&
      new_deepseek_v3_config->ContainGptqWeights()) {
    size_t m = input_size_per_tp * pack_factor;
    size_t n = qweight.shape[1];
    // 获取 workspace
    int64_t current_workspace_size = -1;
    InvokeMacheteGemm(current_workspace_size, nullptr, context_->GetMemoryManageStreams()[dev_rank].Get(), m, n, m,
                      eye_matrix.GetPtr<void>(), qweight.GetPtr<void>(), dequant_weight.GetPtr<void>(),
                      GetMacheteDataType<T>(), llm_kernels::nvidia::vllm_dtype::kU4B8, qweight.scales->GetPtr<void>(),
                      qweight.scales->shape, GetMacheteDataType<T>(), std::nullopt, std::nullopt, std::nullopt,
                      quant_config.group_size, std::nullopt);
    if (current_workspace_size > -1) {
      Tensor workspace =
          Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT8, {static_cast<size_t>(current_workspace_size)},
                 dev_rank, nullptr, &(context_->GetMemoryManageStreams()[dev_rank]));
      InvokeMacheteGemm(
          current_workspace_size, workspace.GetPtr<void>(), context_->GetMemoryManageStreams()[dev_rank].Get(), m, n, m,
          eye_matrix.GetPtr<void>(), qweight.GetPtr<void>(), dequant_weight.GetPtr<void>(), GetMacheteDataType<T>(),
          llm_kernels::nvidia::vllm_dtype::kU4B8, qweight.scales->GetPtr<void>(), qweight.scales->shape,
          GetMacheteDataType<T>(), std::nullopt, std::nullopt, std::nullopt, quant_config.group_size, std::nullopt);
    } else {
      KLLM_THROW("Machete GEMM failed for Dequant");
    }
  }

  return dequant_weight;
}
#endif

template class NewDeepSeekV3WeightImpl<float>;
template class NewDeepSeekV3WeightImpl<float16>;
template class NewDeepSeekV3WeightImpl<bfloat16>;

}  // namespace ksana_llm
