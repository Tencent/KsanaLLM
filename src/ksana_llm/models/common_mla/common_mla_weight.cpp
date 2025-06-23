/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/common_mla/common_mla_weight.h"

#include <numeric>

#include "ksana_llm/kernels/permute.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
CommonMlaWeight<T>::CommonMlaWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, rank, context) {}

template <typename T>
Status CommonMlaWeight<T>::LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                               const std::vector<std::string>& weight_name_list,
                                               const std::vector<std::string>& custom_name_list) {
  SetDevice(rank_);
  size_t kv_lora_rank = model_config_.mla_config.kv_lora_rank;
  size_t qk_rope_head_dim = model_config_.mla_config.qk_rope_head_dim;
  size_t qk_nope_head_dim = model_config_.mla_config.qk_nope_head_dim;
  size_t v_head_dim = model_config_.mla_config.v_head_dim;
  size_t head_num = model_config_.head_num;

  int dp_tensor_para_size = Singleton<Environment>::GetInstance()->GetAttentionTensorParallel();
  int dp_rank = rank_ % dp_tensor_para_size;

  KLLM_LOG_DEBUG << fmt::format("Loading weights from file: {}\nweight_name_list:{}\ncustom_name_list{}",
                                weights_loader->GetTensorFileName(), Vector2Str(weight_name_list),
                                Vector2Str(custom_name_list));

  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    const std::string& tensor_name = custom_name_list[idx];
    const std::string& weight_name = weight_name_list[idx];

    if (!BaseWeight::IsPipelineNodeWeight(tensor_name)) {
      continue;
    }

    if (quant_weight_solver_->IsEnable()) {
      break;
    }
    auto [weight_ptr, weight_size] = weights_loader->GetTensor(weight_name);
    DataType weight_data_type = weights_loader->GetTensorDataType(weight_name);
    std::vector<size_t> weight_shape = weights_loader->GetTensorShape(weight_name);
#ifdef ENABLE_FP8
    if (quant_weight_solver_->LoadMlaFp8E4m3BlockWiseScale(tensor_name, weight_shape, weight_data_type, weight_ptr)) {
      continue;
    }
#endif
    // Split q_b_proj weight for easy to calculate
    if (tensor_name.find(".q_b_proj.weight") != std::string::npos) {
      // 3072 is deepseek v2
      if (weight_shape[0] != 3072 && (qk_nope_head_dim + qk_rope_head_dim) * head_num != weight_shape[0]) {
        KLLM_THROW(fmt::format(
            "The shape of the 0th dim of the weight named '{} ({})' is not equal to the sum of qk_nope_head_dim {} "
            "and qk_rope_head_dim {}.",
            tensor_name, weight_shape[0], qk_nope_head_dim, qk_rope_head_dim));
      }

      if (!model_config_.quant_config.is_fp8_blockwise) {
        // For q_b_nope_proj weight load
        std::string q_b_nope_name =
            tensor_name.substr(0, tensor_name.find_first_of('_')) + "_attn.q_b_nope_proj.weight";
        weights_data_type_map_[q_b_nope_name] = weight_data_type;
        size_t q_b_nope_shape_0 = DivRoundUp(head_num * qk_nope_head_dim, dp_tensor_para_size);
        std::vector<size_t> q_b_nope_shape = {q_b_nope_shape_0, weight_shape[1]};
        tensor_manager_->AddWeightTensor(q_b_nope_name, q_b_nope_shape, weight_data_type);

        size_t tensor_para_offset = dp_rank;
        size_t para_pitch = DivRoundUp(head_num, dp_tensor_para_size) * (qk_nope_head_dim + qk_rope_head_dim) *
                            weight_shape[1] * GetTypeSize(weight_data_type);
        tensor_para_offset *= para_pitch;

        Tensor& q_b_nope_tensor = weights_map_[q_b_nope_name];
        size_t nope_dst_pitch = qk_nope_head_dim * weight_shape[1] * GetTypeSize(weight_data_type);
        size_t nope_src_pitch = (qk_nope_head_dim + qk_rope_head_dim) * weight_shape[1] * GetTypeSize(weight_data_type);
        Memcpy2DAsync(q_b_nope_tensor.GetPtr<void>(), nope_dst_pitch, weight_ptr + tensor_para_offset, nope_src_pitch,
                      nope_dst_pitch, DivRoundUp(head_num, dp_tensor_para_size), MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[rank_]);

        // For q_b_rope_proj weight load
        std::string q_b_rope_name =
            tensor_name.substr(0, tensor_name.find_first_of('_')) + "_attn.q_b_rope_proj.weight";
        size_t q_b_rope_shape_0 = DivRoundUp(head_num * qk_rope_head_dim, dp_tensor_para_size);
        std::vector<size_t> q_b_rope_shape = {q_b_rope_shape_0, weight_shape[1]};
        tensor_manager_->AddWeightTensor(q_b_rope_name, q_b_rope_shape, weight_data_type);
        weights_data_type_map_[q_b_rope_name] = weight_data_type;
        Tensor& q_b_rope_tensor = weights_map_[q_b_rope_name];
        size_t rope_dst_pitch = qk_rope_head_dim * weight_shape[1] * GetTypeSize(weight_data_type);
        Memcpy2DAsync(q_b_rope_tensor.GetPtr<void>(), rope_dst_pitch, weight_ptr + nope_dst_pitch + tensor_para_offset,
                      nope_src_pitch, rope_dst_pitch, DivRoundUp(head_num, dp_tensor_para_size), MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[rank_]);
      } else {
        // For fp8 blockwise quant, do not split the weights initially, split them after dequantization later.
        tensor_manager_->AddWeightTensor(
            tensor_name, {DivRoundUp(weight_shape[0], dp_tensor_para_size), weight_shape[1]}, weight_data_type);
        weights_data_type_map_[tensor_name] = weight_data_type;
        Tensor& q_b_proj_tensor = weights_map_[tensor_name];
        size_t tensor_para_offset = dp_rank;
        size_t para_pitch = DivRoundUp(head_num, dp_tensor_para_size) * (qk_nope_head_dim + qk_rope_head_dim) *
                            weight_shape[1] * GetTypeSize(weight_data_type);
        tensor_para_offset *= para_pitch;
        MemcpyAsync(q_b_proj_tensor.GetPtr<void>(), weight_ptr + tensor_para_offset, para_pitch, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      }
    }
    // Split kv_a_proj weight for easy to calculate
    if (tensor_name.find(".kv_a_proj_with_mqa.weight") != std::string::npos) {
      if ((kv_lora_rank + qk_rope_head_dim) != weight_shape[0]) {
        KLLM_THROW(
            fmt::format("The shape of the 0th dim of the weight named '{}' is not equal to the sum of kv_lora_rank {} "
                        "and qk_rope_head_dim {}.",
                        tensor_name, kv_lora_rank, qk_rope_head_dim));
      }

      std::string kv_a_lora_name =
          tensor_name.substr(0, tensor_name.find_first_of('_')) + "_attn.kv_a_lora_proj.weight";
      std::vector<size_t> kv_a_lora_shape = {kv_lora_rank, weight_shape[1]};
      tensor_manager_->AddWeightTensor(kv_a_lora_name, kv_a_lora_shape, weight_data_type);
      weights_data_type_map_[kv_a_lora_name] = weight_data_type;
      Tensor& kv_a_lora_tensor = weights_map_[kv_a_lora_name];
      size_t kv_a_lora_size = kv_lora_rank * weight_shape[1] * GetTypeSize(weight_data_type);
      MemcpyAsync(kv_a_lora_tensor.GetPtr<void>(), weight_ptr, kv_a_lora_size, MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);

      // For kv_a_rope_proj weight load
      std::string kv_a_rope_name =
          tensor_name.substr(0, tensor_name.find_first_of('_')) + "_attn.kv_a_rope_proj.weight";
      std::vector<size_t> kv_a_rope_shape = {qk_rope_head_dim, weight_shape[1]};
      tensor_manager_->AddWeightTensor(kv_a_rope_name, kv_a_rope_shape, weight_data_type);
      weights_data_type_map_[kv_a_rope_name] = weight_data_type;
      Tensor& kv_a_rope_tensor = weights_map_[kv_a_rope_name];
      MemcpyAsync(kv_a_rope_tensor.GetPtr<void>(), weight_ptr + kv_a_lora_size,
                  qk_rope_head_dim * weight_shape[1] * GetTypeSize(weight_data_type), MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[rank_]);
    }
    // Split kv_b_proj weight for easy to calculate
    if (tensor_name.find(".kv_b_proj.weight") != std::string::npos) {
      if (head_num * (qk_nope_head_dim + v_head_dim) != weight_shape[0]) {
        KLLM_THROW(
            fmt::format("The shape of the 1th dim of the weight named '{}' is not equal to (num_head {} * "
                        "(qk_nope_head_dim {} + v_head_dim {})).",
                        tensor_name, head_num, qk_nope_head_dim, v_head_dim));
      }

      if (!model_config_.quant_config.is_fp8_blockwise) {
        // For kv_b_nope_proj weight load
        std::string kv_b_nope_name =
            tensor_name.substr(0, tensor_name.find_first_of('_')) + "_attn.kv_b_nope_proj.weight";
        weights_data_type_map_[kv_b_nope_name] = weight_data_type;
        size_t kv_b_nope_shape_0 = DivRoundUp(head_num * qk_nope_head_dim, dp_tensor_para_size);
        std::vector<size_t> kv_b_nope_shape = {kv_b_nope_shape_0, weight_shape[1]};
        tensor_manager_->AddWeightTensor(kv_b_nope_name, kv_b_nope_shape, weight_data_type);

        size_t tensor_para_offset = dp_rank;
        size_t para_pitch = DivRoundUp(head_num, dp_tensor_para_size) * (qk_nope_head_dim + v_head_dim) *
                            weight_shape[1] * GetTypeSize(weight_data_type);
        tensor_para_offset *= para_pitch;

        Tensor& kv_b_nope_tensor = weights_map_[kv_b_nope_name];
        size_t nope_dst_pitch = qk_nope_head_dim * weight_shape[1] * GetTypeSize(weight_data_type);
        size_t nope_src_pitch = (qk_nope_head_dim + v_head_dim) * weight_shape[1] * GetTypeSize(weight_data_type);
        Memcpy2DAsync(kv_b_nope_tensor.GetPtr<void>(), nope_dst_pitch, weight_ptr + tensor_para_offset, nope_src_pitch,
                      nope_dst_pitch, DivRoundUp(head_num, dp_tensor_para_size), MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[rank_]);

        // For v_head_proj weight load
        std::string v_head_name = tensor_name.substr(0, tensor_name.find_first_of('_')) + "_attn.v_head_proj.weight";
        weights_data_type_map_[v_head_name] = weight_data_type;
        size_t v_head_shape_0 = DivRoundUp(head_num * v_head_dim, dp_tensor_para_size);
        std::vector<size_t> v_head_shape = {v_head_shape_0, weight_shape[1]};
        tensor_manager_->AddWeightTensor(v_head_name, v_head_shape, weight_data_type);
        Tensor& v_head_tensor = weights_map_[v_head_name];
        size_t v_head_dst_pitch = v_head_dim * weight_shape[1] * GetTypeSize(weight_data_type);
        Memcpy2DAsync(v_head_tensor.GetPtr<void>(), v_head_dst_pitch, weight_ptr + nope_dst_pitch + tensor_para_offset,
                      nope_src_pitch, v_head_dst_pitch, DivRoundUp(head_num, dp_tensor_para_size),
                      MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
        if (GetAbsorbWeightsType() == AbsorbWeightsType::kAbsorbTypeBMM) {
          size_t head_num_tp = DivRoundUp(head_num, dp_tensor_para_size);
          // Copy kv_b_nope_proj to w_uk_t
          std::string w_uk_t_name = tensor_name.substr(0, tensor_name.find_first_of('_')) + "_attn.w_uk_t.weight";
          tensor_manager_->AddWeightTensor(w_uk_t_name, {head_num_tp, qk_nope_head_dim, kv_b_nope_tensor.shape[1]},
                                           weight_data_type);
          Tensor& w_uk_t_tensor = weights_map_[w_uk_t_name];

          MemcpyAsync(w_uk_t_tensor.template GetPtr<void>(), kv_b_nope_tensor.template GetPtr<void>(),
                      kv_b_nope_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
                      context_->GetMemoryManageStreams()[rank_]);
          weights_data_type_map_[w_uk_t_name] = weight_data_type;
          // Permute vhead_weight_name to w_uv
          std::string w_uv_name = tensor_name.substr(0, tensor_name.find_first_of('_')) + "_attn.w_uv.weight";
          tensor_manager_->AddWeightTensor(w_uv_name, {head_num_tp, weights_map_[v_head_name].shape[1], v_head_dim},
                                           weight_data_type);
          weights_map_[v_head_name].shape = {head_num_tp, v_head_dim, weights_map_[v_head_name].shape[1]};
          Permute(weights_map_[v_head_name], weights_map_[w_uv_name], {0, 2, 1},
                  context_->GetMemoryManageStreams()[rank_]);
          weights_data_type_map_[w_uv_name] = weight_data_type;
          weights_map_[v_head_name].shape = {head_num_tp * v_head_dim, weights_map_[v_head_name].shape[2]};
        }
      } else {
        tensor_manager_->AddWeightTensor(
            tensor_name, {DivRoundUp(weight_shape[0], dp_tensor_para_size), weight_shape[1]}, weight_data_type);
        weights_data_type_map_[tensor_name] = weight_data_type;
        Tensor& kv_b_proj_tensor = weights_map_[tensor_name];
        size_t tensor_para_offset = dp_rank;
        size_t para_pitch = DivRoundUp(head_num, dp_tensor_para_size) * (qk_nope_head_dim + v_head_dim) *
                            weight_shape[1] * GetTypeSize(weight_data_type);
        tensor_para_offset *= para_pitch;
        MemcpyAsync(kv_b_proj_tensor.GetPtr<void>(), weight_ptr + tensor_para_offset, para_pitch, MEMCPY_HOST_TO_DEVICE,
                    context_->GetMemoryManageStreams()[rank_]);
      }
    }
    KLLM_LOG_DEBUG << "Success load weight:" << tensor_name << " on rank " << rank_;
  }  // end for loop
  return Status();
}
template <typename T>
Status CommonMlaWeight<T>::PermuteQaWeight(Tensor& last_q_a_proj_tensor, bool is_weight_scale) {
  SetDevice(rank_);
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string q_a_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.q_a_proj" +
                           (is_weight_scale ? ".weight_scale_inv" : ".weight");
    CommonWeight<T>::CommonPermuteWeight(q_a_name, last_q_a_proj_tensor);
  }
  return Status();
}

template <typename T>
Status CommonMlaWeight<T>::PermuteQbNopeWeight(Tensor& last_q_b_nope_tensor, bool is_weight_scale) {
  SetDevice(rank_);
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string q_b_nope_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.q_b_nope_proj" +
                                (is_weight_scale ? ".weight_scale_inv" : ".weight");
    CommonWeight<T>::CommonPermuteWeight(q_b_nope_name, last_q_b_nope_tensor);
  }
  return Status();
}

template <typename T>
Status CommonMlaWeight<T>::PermuteQbRopeWeight(Tensor& last_q_b_rope_tensor, bool is_weight_scale) {
  SetDevice(rank_);
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string q_b_rope_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.q_b_rope_proj" +
                                (is_weight_scale ? ".weight_scale_inv" : ".weight");
    CommonWeight<T>::CommonPermuteWeight(q_b_rope_name, last_q_b_rope_tensor);
  }
  return Status();
}

template <typename T>
Status CommonMlaWeight<T>::PermuteKVaLoraWeight(Tensor& last_kv_a_lora_tensor, bool is_weight_scale) {
  SetDevice(rank_);
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string kv_a_lora_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.kv_a_lora_proj" +
                                 (is_weight_scale ? ".weight_scale_inv" : ".weight");
    CommonWeight<T>::CommonPermuteWeight(kv_a_lora_name, last_kv_a_lora_tensor);
  }
  return Status();
}

template <typename T>
Status CommonMlaWeight<T>::PermuteKVaRopeWeight(Tensor& last_kv_a_rope_tensor, bool is_weight_scale) {
  SetDevice(rank_);
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string kv_a_rope_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.kv_a_rope_proj" +
                                 (is_weight_scale ? ".weight_scale_inv" : ".weight");
    CommonWeight<T>::CommonPermuteWeight(kv_a_rope_name, last_kv_a_rope_tensor);
  }
  return Status();
}

template <typename T>
Status CommonMlaWeight<T>::PermuteKVbNopeWeight(Tensor& last_kv_b_nope_tensor, bool is_weight_scale) {
  SetDevice(rank_);
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string kv_b_nope_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.kv_b_nope_proj" +
                                 (is_weight_scale ? ".weight_scale_inv" : ".weight");
    CommonWeight<T>::CommonPermuteWeight(kv_b_nope_name, last_kv_b_nope_tensor);
  }
  return Status();
}

template <typename T>
Status CommonMlaWeight<T>::PermuteVHeadWeight(Tensor& last_v_head_tensor, bool is_weight_scale) {
  SetDevice(rank_);
  for (const auto layer_idx : required_layer_idx_.all) {
    std::string v_head_name = "model.layers." + std::to_string(layer_idx) + ".self_attn.v_head_proj" +
                              (is_weight_scale ? ".weight_scale_inv" : ".weight");
    CommonWeight<T>::CommonPermuteWeight(v_head_name, last_v_head_tensor);
  }
  return Status();
}

template <typename T>
Status CommonMlaWeight<T>::PermuteMlaWeight(bool is_weight_scale) {
  if (model_config_.mla_config.q_lora_rank != 0) {
    // 测试 lite v2 需要先注释 Permute q_a_proj Weight
    tensor_manager_->CreateTensorWithSameShape("model.layers." + std::to_string(pipeline_config_.lower_layer_idx) +
                                                   ".self_attn.q_a_proj" +
                                                   (is_weight_scale ? ".weight_scale_inv" : ".weight"),
                                               "empty_q_a_proj_tensor");
    Tensor& last_q_a_tensor = weights_map_["empty_q_a_proj_tensor"];
    PermuteQaWeight(last_q_a_tensor, is_weight_scale);
    weights_map_.erase("empty_q_a_proj_tensor");
  }

  // Permute q_b_nope_proj Weight
  tensor_manager_->CreateTensorWithSameShape("model.layers." + std::to_string(pipeline_config_.lower_layer_idx) +
                                                 ".self_attn.q_b_nope_proj" +
                                                 (is_weight_scale ? ".weight_scale_inv" : ".weight"),
                                             "empty_q_b_nope_proj_tensor");
  Tensor& last_q_b_nope_tensor = weights_map_["empty_q_b_nope_proj_tensor"];
  PermuteQbNopeWeight(last_q_b_nope_tensor, is_weight_scale);
  weights_map_.erase("empty_q_b_nope_proj_tensor");

  // Permute q_b_rope_proj Weight
  tensor_manager_->CreateTensorWithSameShape("model.layers." + std::to_string(pipeline_config_.lower_layer_idx) +
                                                 ".self_attn.q_b_rope_proj" +
                                                 (is_weight_scale ? ".weight_scale_inv" : ".weight"),
                                             "empty_q_b_rope_proj_tensor");
  Tensor& last_q_b_rope_tensor = weights_map_["empty_q_b_rope_proj_tensor"];
  PermuteQbRopeWeight(last_q_b_rope_tensor, is_weight_scale);
  weights_map_.erase("empty_q_b_rope_proj_tensor");

  // Permute kv_a_lora_proj Weight
  tensor_manager_->CreateTensorWithSameShape("model.layers." + std::to_string(pipeline_config_.lower_layer_idx) +
                                                 ".self_attn.kv_a_lora_proj" +
                                                 (is_weight_scale ? ".weight_scale_inv" : ".weight"),
                                             "empty_kv_a_lora_proj_tensor");
  Tensor& last_kv_a_lora_tensor = weights_map_["empty_kv_a_lora_proj_tensor"];
  PermuteKVaLoraWeight(last_kv_a_lora_tensor, is_weight_scale);
  weights_map_.erase("empty_kv_a_lora_proj_tensor");

  // Permute kv_a_rope_proj Weight
  tensor_manager_->CreateTensorWithSameShape("model.layers." + std::to_string(pipeline_config_.lower_layer_idx) +
                                                 ".self_attn.kv_a_rope_proj" +
                                                 (is_weight_scale ? ".weight_scale_inv" : ".weight"),
                                             "empty_kv_a_rope_proj_tensor");
  Tensor& last_kv_a_rope_tensor = weights_map_["empty_kv_a_rope_proj_tensor"];
  PermuteKVaRopeWeight(last_kv_a_rope_tensor, is_weight_scale);
  weights_map_.erase("empty_kv_a_rope_proj_tensor");

  // Permute kv_b_nope_proj Weight
  tensor_manager_->CreateTensorWithSameShape("model.layers." + std::to_string(pipeline_config_.lower_layer_idx) +
                                                 ".self_attn.kv_b_nope_proj" +
                                                 (is_weight_scale ? ".weight_scale_inv" : ".weight"),
                                             "empty_kv_b_nope_proj_tensor");
  Tensor& last_kv_b_nope_tensor = weights_map_["empty_kv_b_nope_proj_tensor"];
  PermuteKVbNopeWeight(last_kv_b_nope_tensor, is_weight_scale);
  weights_map_.erase("empty_kv_b_nope_proj_tensor");

  // Permute v_head_proj Weight
  tensor_manager_->CreateTensorWithSameShape("model.layers." + std::to_string(pipeline_config_.lower_layer_idx) +
                                                 ".self_attn.v_head_proj" +
                                                 (is_weight_scale ? ".weight_scale_inv" : ".weight"),
                                             "empty_v_head_proj_tensor");
  Tensor& last_v_head_tensor = weights_map_["empty_v_head_proj_tensor"];
  PermuteVHeadWeight(last_v_head_tensor, is_weight_scale);
  weights_map_.erase("empty_v_head_proj_tensor");
  return Status();
}

template <typename T>
void CommonMlaWeight<T>::ProcessWeights() {
  // 避免权重重复处理
  if (model_config_.quant_config.is_fp8_blockwise) {
#ifdef ENABLE_FP8
#  ifdef ENABLE_FP8_TORCH
    quant_weight_solver_->ProcessMlaFp8E4m3BlockWiseScaleOfWeight();
#  endif
    quant_weight_solver_->BindMlaFp8E4m3BlockWiseScaleOfWeight();
#else
    KLLM_THROW("Device not support Fp8");
#endif
  } else if (!quant_weight_solver_->IsEnable()) {
    PermuteMlaWeight(false);
  }
}

template class CommonMlaWeight<float>;
template class CommonMlaWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class CommonMlaWeight<bfloat16>;
#endif
}  // namespace ksana_llm
