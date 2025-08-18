/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/llama/llama_model_weight_loader.h"
#include <regex>
#include <string>

#include <unordered_set>

#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/models/base/model_arch.h"
#include "ksana_llm/models/base/model_format.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"

#include "ksana_llm/kernels/trans_layout.h"
#include "ksana_llm/models/llama/llama_model_config.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

LlamaModelWeightLoader::LlamaModelWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                               std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context) {
  // Initialize pipeline config, for distributed mode.
  env->GetPipelineConfig(pipeline_config_);
}

LlamaModelWeightLoader::~LlamaModelWeightLoader() {}

Status LlamaModelWeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq"};
  std::vector<std::string> master_only_list = {"model.embed_tokens.weight", "lm_head.weight"};

  int lower_layer_idx = pipeline_config_.lower_layer_idx;
  int upper_layer_idx = pipeline_config_.upper_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      weight_names.erase(it);
      continue;
    }

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && (layer_idx < lower_layer_idx || layer_idx > upper_layer_idx)) {
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

Status LlamaModelWeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                   int dev_rank,
                                                   std::unordered_map<std::string, Tensor>& device_model_weights,
                                                   std::unordered_map<std::string, Tensor>& left_host_weights) {
  std::shared_ptr<LlamaModelConfig> llama_model_config = std::dynamic_pointer_cast<LlamaModelConfig>(model_config_);

  if (llama_model_config->model_format == ModelFormat::GGUF) {
    return Status(RET_INVALID_ARGUMENT, "Not support GGUF format yet.");
  }

  std::unordered_set<std::string> processed_weights;
  for (auto& [file_weight_name, host_weight_tensor] : host_model_weights) {
    if (processed_weights.find(file_weight_name) != processed_weights.end()) {
      continue;
    }

    if (CheckWeightNameMatched(file_weight_name,
                               {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"},
                               false)) {
      std::string qkv_weight_name_prefix = file_weight_name.substr(0, file_weight_name.find("_proj") - 1);

      // Merge q & k & v to query_key_value.
      std::string qkv_weight_name = qkv_weight_name_prefix + "query_key_value.weight";

      // Make sure q & k & v all exist in one file.
      if (!CheckAllWeightsExist(host_model_weights,
                                {qkv_weight_name_prefix + "q_proj.weight", qkv_weight_name_prefix + "k_proj.weight",
                                 qkv_weight_name_prefix + "v_proj.weight"})) {
        left_host_weights[file_weight_name] = host_weight_tensor;
        continue;
      }

      const Tensor& q_host_weight_tensor = host_model_weights.at(qkv_weight_name_prefix + "q_proj.weight");
      const Tensor& k_host_weight_tensor = host_model_weights.at(qkv_weight_name_prefix + "k_proj.weight");
      const Tensor& v_host_weight_tensor = host_model_weights.at(qkv_weight_name_prefix + "v_proj.weight");

      size_t qkv_shape_0 =
          q_host_weight_tensor.shape[0] + k_host_weight_tensor.shape[0] + v_host_weight_tensor.shape[0];
      size_t qkv_slice_shape_0 = static_cast<size_t>(DivRoundUp(qkv_shape_0, context_->GetTensorParallelSize()));
      std::vector<size_t> qkv_slice_shape = {qkv_slice_shape_0, host_weight_tensor.shape[1]};

      Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, qkv_slice_shape, dev_rank);
      Tensor qkv_dev_tensor =
          Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, qkv_slice_shape, dev_rank);

      size_t q_slice_offset =
          static_cast<size_t>(DivRoundUp(q_host_weight_tensor.shape[0], context_->GetTensorParallelSize())) *
          q_host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype) * dev_rank;
      size_t q_slice_bytes =
          static_cast<size_t>(DivRoundUp(q_host_weight_tensor.shape[0], context_->GetTensorParallelSize())) *
          q_host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);

      if (k_host_weight_tensor.shape[0] != v_host_weight_tensor.shape[0]) {
        KLLM_THROW(fmt::format("k & v should have same shape for llama"));
      }
      size_t kv_slice_offset =
          static_cast<size_t>(DivRoundUp(k_host_weight_tensor.shape[0], context_->GetTensorParallelSize())) *
          k_host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype) * dev_rank;
      size_t kv_slice_bytes =
          static_cast<size_t>(DivRoundUp(k_host_weight_tensor.shape[0], context_->GetTensorParallelSize())) *
          k_host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
      if (static_cast<size_t>(dev_rank) == context_->GetTensorParallelSize() - 1) {
        q_slice_bytes = q_host_weight_tensor.GetTotalBytes() - q_slice_offset;
        kv_slice_bytes = k_host_weight_tensor.GetTotalBytes() - kv_slice_offset;
      }

      MemcpyAsync(qkv_dev_tensor.GetPtr<void>(), q_host_weight_tensor.GetPtr<void>() + q_slice_offset, q_slice_bytes,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      MemcpyAsync(qkv_dev_tensor.GetPtr<void>() + q_slice_bytes, k_host_weight_tensor.GetPtr<void>() + kv_slice_offset,
                  kv_slice_bytes, MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      MemcpyAsync(qkv_dev_tensor.GetPtr<void>() + q_slice_bytes + kv_slice_bytes,
                  v_host_weight_tensor.GetPtr<void>() + kv_slice_offset, kv_slice_bytes, MEMCPY_HOST_TO_DEVICE,
                  context_->GetMemoryManageStreams()[dev_rank]);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      PermuteDeviceTensor(qkv_dev_tensor, {1, 0}, dev_rank, dev_tensor);

      Status status = CastDeviceTensorType(dev_tensor, llama_model_config->weight_data_type, dev_rank);
      if (!status.OK()) {
        return status;
      }

      device_model_weights[qkv_weight_name] = dev_tensor;

      processed_weights.insert(qkv_weight_name_prefix + "q_proj.weight");
      processed_weights.insert(qkv_weight_name_prefix + "k_proj.weight");
      processed_weights.insert(qkv_weight_name_prefix + "v_proj.weight");

      continue;
    }

    if (CheckWeightNameMatched(file_weight_name, {"model.norm.weight"}, true) ||
        CheckWeightNameMatched(file_weight_name, {"input_layernorm.weight", "post_attention_layernorm.weight"},
                               false)) {
      Tensor dev_tensor;
      STATUS_CHECK_RETURN(CopyHostTensorToDevice(host_weight_tensor, dev_rank, dev_tensor));
      STATUS_CHECK_RETURN(CastDeviceTensorType(dev_tensor, llama_model_config->weight_data_type, dev_rank));

      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
      device_model_weights[file_weight_name] = dev_tensor;
      continue;
    }

    if (CheckWeightNameMatched(file_weight_name, {"self_attn.o_proj.weight", "mlp.down_proj.weight"}, false)) {
      Tensor full_dev_tensor =
          Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

      MemcpyAsync(full_dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      Tensor permute_dev_tensor =
          Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);
      PermuteDeviceTensor(full_dev_tensor, {1, 0}, dev_rank, permute_dev_tensor);
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

      Status status = CastDeviceTensorType(dev_tensor, llama_model_config->weight_data_type, dev_rank);
      if (!status.OK()) {
        return status;
      }

      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      device_model_weights[file_weight_name] = dev_tensor;

      continue;
    }

    if (CheckWeightNameMatched(file_weight_name, {"mlp.gate_proj.weight", "mlp.up_proj.weight"}, false)) {
      std::vector<size_t> slice_shape = {
          static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], context_->GetTensorParallelSize())),
          host_weight_tensor.shape[1]};

      Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, slice_shape, dev_rank);

      size_t slice_offset = dev_tensor.GetTotalBytes() * dev_rank;
      size_t slice_bytes = dev_tensor.GetTotalBytes();
      if (static_cast<size_t>(dev_rank) == context_->GetTensorParallelSize() - 1) {
        slice_bytes = host_weight_tensor.GetTotalBytes() - slice_offset;
      }

      Tensor full_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, slice_shape, dev_rank);
      MemcpyAsync(full_dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + slice_offset, slice_bytes,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      PermuteDeviceTensor(full_dev_tensor, {1, 0}, dev_rank, dev_tensor);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      Status status = CastDeviceTensorType(dev_tensor, llama_model_config->weight_data_type, dev_rank);
      if (!status.OK()) {
        return status;
      }

      device_model_weights[file_weight_name] = dev_tensor;

      continue;
    }

    if (CheckWeightNameMatched(file_weight_name, {"model.embed_tokens.weight"}, true)) {
      Tensor full_dev_tensor =
          Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);

      MemcpyAsync(full_dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      Tensor permute_dev_tensor =
          Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape, dev_rank);
      PermuteDeviceTensor(full_dev_tensor, {1, 0}, dev_rank, permute_dev_tensor);
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

      MemcpyAsync(full_dev_tensor.GetPtr<void>(), permute_dev_tensor.GetPtr<void>() + slice_offset, slice_bytes,
                  MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      full_dev_tensor.shape = dev_tensor.shape;
      PermuteDeviceTensor(full_dev_tensor, {1, 0}, dev_rank, dev_tensor);

      Status status = CastDeviceTensorType(dev_tensor, llama_model_config->weight_data_type, dev_rank);
      if (!status.OK()) {
        return status;
      }

      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      device_model_weights[file_weight_name] = dev_tensor;

      continue;
    }

    if (CheckWeightNameMatched(file_weight_name, {"lm_head.weight"}, true)) {
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

      // Cast to expected type.
      Status status = CastDeviceTensorType(dev_tensor, llama_model_config->weight_data_type, dev_rank);
      if (!status.OK()) {
        return status;
      }

      Tensor permute_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype, dev_tensor.shape, dev_rank);
      PermuteDeviceTensor(dev_tensor, {1, 0}, dev_rank, permute_dev_tensor);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);

      device_model_weights[file_weight_name] = permute_dev_tensor;

      continue;
    }
  }

  return Status();
}

}  // namespace ksana_llm
