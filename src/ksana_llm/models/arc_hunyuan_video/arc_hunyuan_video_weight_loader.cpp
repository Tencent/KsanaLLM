/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/arc_hunyuan_video/arc_hunyuan_video_weight_loader.h"

#include "ksana_llm/model_loader/model_loader_utils.h"

namespace ksana_llm {
ArcHunyuanVideoWeightLoader::ArcHunyuanVideoWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                                         std::shared_ptr<Environment> env,
                                                         std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context),
      common_weight_loader_(std::make_unique<CommonModelWeightLoader>(model_config, env, context)) {
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  tp_ = runtime_config_.parallel_basic_config.tensor_parallel_size;
  arc_hunyuan_video_config_ = std::dynamic_pointer_cast<ArcHunyuanVideoConfig>(model_config_);
  // TODO(jinxcwu) 暂时只做单卡的适配，后续看情况补充
  KLLM_CHECK_WITH_INFO(tp_ == 1, "Currently ArcHunyuanVideo only support TP=1");
}

ArcHunyuanVideoWeightLoader::~ArcHunyuanVideoWeightLoader() {}

Status ArcHunyuanVideoWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::vector<std::string> skip_list = {"speech_encoder.", "vision_model.", "mlp2.0", "mlp2.1", "mlp2.3"};
  std::vector<std::string> master_only_list = {"language_model.model.embed_tokens.weight"};

  int lower_layer_idx = pipeline_config_.lower_layer_idx;
  int upper_layer_idx = pipeline_config_.upper_layer_idx;
  int lower_nextn_layer_idx = pipeline_config_.lower_nextn_layer_idx;
  int upper_nextn_layer_idx = pipeline_config_.upper_nextn_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      weight_names.erase(it);
      continue;
    }

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && ((layer_idx < lower_layer_idx || layer_idx > upper_layer_idx) &&
                             (layer_idx < lower_nextn_layer_idx || layer_idx > upper_nextn_layer_idx))) {
        weight_names.erase(it);
        continue;
      }
    }

    // Skip embedding and lm_head on worker node in distributed mode.
    if (!context_->IsStandalone() && !context_->IsChief()) {
      if (CheckWeightNameMatched(*it, master_only_list, false)) {
        weight_names.erase(it);
        continue;
      }
    }
    ++it;
  }
  return Status();
}

Status ArcHunyuanVideoWeightLoader::ProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
    std::unordered_map<std::string, Tensor>& device_model_weights,
    std::unordered_map<std::string, Tensor>& left_host_weights) {
  for (auto& [host_weight_name, host_weight_tensor] : host_model_weights) {
    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, host_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    std::string new_host_weight_name = CutPrefix(host_weight_name, "language_model.");

    // 1. attn weights
    if (CheckWeightNameMatched(new_host_weight_name, {".self_attn.qkv_proj.weight"}, MatchMode::PartialMatch)) {
      Tensor dev_tensor;
      common_weight_loader_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, tp_, true);
      torch::Tensor qkv = common_weight_loader_->GetTorchTensorFromTensor(dev_tensor);
      const size_t heads = arc_hunyuan_video_config_->num_key_value_heads;
      const size_t head_groups = arc_hunyuan_video_config_->head_groups;
      qkv = qkv.reshape({qkv.size(0), heads, head_groups + 2, -1});
      auto tt = qkv.split({head_groups, 1, 1}, -2);
      tt[0] = tt[0].reshape({tt[0].size(0), -1});
      tt[1] = tt[1].reshape({tt[1].size(0), -1});
      tt[2] = tt[2].reshape({tt[2].size(0), -1});
      qkv = torch::concat(tt, -1).contiguous();
      const std::string qkv_proj_name = GetReplacedName(new_host_weight_name, "qkv_proj", "query_key_value");
      device_model_weights[qkv_proj_name] = common_weight_loader_->GetTensorFromTorchTensor(qkv);
      continue;
    }
    if (CheckWeightNameMatched(new_host_weight_name, {".self_attn.o_proj.weight"}, MatchMode::PartialMatch)) {
      Tensor dev_tensor;
      common_weight_loader_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, tp_, false);
      device_model_weights[new_host_weight_name] = dev_tensor;
      continue;
    }
    if (CheckWeightNameMatched(new_host_weight_name, {".self_attn.query_layernorm.weight"}, MatchMode::PartialMatch)) {
      device_model_weights[new_host_weight_name] = common_weight_loader_->MoveToDevice(host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(new_host_weight_name, {".self_attn.key_layernorm.weight"}, MatchMode::PartialMatch)) {
      device_model_weights[new_host_weight_name] = common_weight_loader_->MoveToDevice(host_weight_tensor, dev_rank);
      continue;
    }

    // 2. mlp weights
    if (CheckWeightNameMatched(new_host_weight_name, {".mlp.gate_and_up_proj.weight"}, MatchMode::PartialMatch)) {
      Tensor first(host_weight_tensor.location, host_weight_tensor.dtype,
                   {host_weight_tensor.shape[0] / 2, host_weight_tensor.shape[1]}, host_weight_tensor.device_id,
                   host_weight_tensor.GetPtr<void>());
      Tensor second(first.location, first.dtype, first.shape, first.device_id,
                    host_weight_tensor.GetPtr<void>() + first.GetTotalBytes());
      Tensor swapped_tensor;
      common_weight_loader_->Concat({second, first}, swapped_tensor, dev_rank);
      Tensor dev_tensor;
      common_weight_loader_->SplitOptTrans(swapped_tensor, dev_tensor, dev_rank, tp_, true);
      const std::string gate_up_proj_name = GetReplacedName(new_host_weight_name, "gate_and_up_proj", "gate_up_proj");
      device_model_weights[gate_up_proj_name] = dev_tensor;
      continue;
    }
    if (CheckWeightNameMatched(new_host_weight_name, {".mlp.down_proj.weight"}, MatchMode::PartialMatch)) {
      Tensor dev_tensor;
      common_weight_loader_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, tp_, false);
      device_model_weights[new_host_weight_name] = dev_tensor;
      continue;
    }

    // 3. norm weights
    if (CheckWeightNameMatched(new_host_weight_name, {".input_layernorm.weight"}, MatchMode::PartialMatch)) {
      device_model_weights[new_host_weight_name] = common_weight_loader_->MoveToDevice(host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(new_host_weight_name, {".post_attention_layernorm.weight"}, MatchMode::PartialMatch)) {
      device_model_weights[new_host_weight_name] = common_weight_loader_->MoveToDevice(host_weight_tensor, dev_rank);
      continue;
    }
    if (CheckWeightNameMatched(new_host_weight_name, {"model.norm.weight"}, MatchMode::FullMatch)) {
      device_model_weights[new_host_weight_name] = common_weight_loader_->MoveToDevice(host_weight_tensor, dev_rank);
      continue;
    }

    // 4. embedding weights && lm_head weights
    if (CheckWeightNameMatched(new_host_weight_name, {"model.embed_tokens.weight"}, MatchMode::FullMatch)) {
      Tensor embed_tokens_tensor;
      common_weight_loader_->TransSplitOptTrans(host_weight_tensor, embed_tokens_tensor, dev_rank, tp_, true);
      device_model_weights[new_host_weight_name] = embed_tokens_tensor;

      Tensor lm_head_tensor;
      common_weight_loader_->SplitOptTrans(host_weight_tensor, lm_head_tensor, dev_rank, tp_, true);
      device_model_weights["lm_head.weight"] = lm_head_tensor;
      continue;
    }
  }
  return Status();
}

}  // namespace ksana_llm
