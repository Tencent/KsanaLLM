/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/base/base_weight.h"

#include <future>
#include <regex>

#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/safetensors_file_saver.h"
#include "ksana_llm/utils/safetensors_file_tensor_loader.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

BaseWeight::BaseWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                       std::shared_ptr<Context> context)
    : rank_(rank), context_(context), model_config_(model_config), runtime_config_(runtime_config) {
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config_);
  tensor_manager_ = std::make_shared<TensorManager>(rank, weights_map_);

  // extract required layer idx
  for (auto idx = pipeline_config_.lower_layer_idx; idx <= pipeline_config_.upper_layer_idx; ++idx) {
    required_layer_idx_.all.emplace(idx);
  }
  KLLM_LOG_INFO << "CommonWeight IsChief:" << context_->IsChief() << ", layer:[" << pipeline_config_.lower_layer_idx
                << ", " << pipeline_config_.upper_layer_idx << "].";
  // add nextn predict layers
  if (static_cast<int>(pipeline_config_.lower_nextn_layer_idx) >= static_cast<int>(model_config_.num_layer)) {
    for (auto idx = pipeline_config_.lower_nextn_layer_idx; idx <= pipeline_config_.upper_nextn_layer_idx; ++idx) {
      required_layer_idx_.all.emplace(idx);
    }
    KLLM_LOG_INFO << "CommonWeight IsChief:" << context_->IsChief() << ", nextn layer:["
                  << pipeline_config_.lower_nextn_layer_idx << ", " << pipeline_config_.upper_nextn_layer_idx << "].";
  }
  // extract dense and moe layer idx
  const std::vector<size_t>& moe_layers = model_config_.moe_config.moe_layers;
  for (const auto idx : required_layer_idx_.all) {
    if (model_config_.is_moe && idx >= static_cast<int>(model_config_.moe_config.first_k_dense_replace) &&
        (moe_layers.empty() || std::find(moe_layers.begin(), moe_layers.end(), idx) != moe_layers.end())) {
      required_layer_idx_.moe.emplace(idx);
    } else {
      required_layer_idx_.dense.emplace(idx);
    }
  }
}

bool BaseWeight::IsPipelineNodeWeight(const std::string& tensor_name) {
  // Start get layer_idx
  int layer_idx = 0;
  static const std::regex re("\\d+");
  std::smatch match;
  if (std::regex_search(tensor_name, match, re)) {
    const std::string& layer_idx_str = match.str(0);
    layer_idx = std::stoi(layer_idx_str);
  }  // End get layer_idx
  if (required_layer_idx_.all.find(layer_idx) != required_layer_idx_.all.end()) {
    KLLM_LOG_DEBUG << "The weight named " << tensor_name << " layer idx : " << layer_idx << " loaders on node_rank "
                   << pipeline_config_.node_rank;
    return true;
  }
  return false;
}

bool BaseWeight::TryToLoadWeightsFromCache() {
  const std::filesystem::path cache_path(GetCacheFolder());
  if (!std::filesystem::exists(cache_path) || !std::filesystem::is_directory(cache_path)) {
    KLLM_LOG_INFO << fmt::format("CacheFolder: {} does not exist, skip load cache", cache_path.string());
    return false;
  }

  ModelFileFormat model_file_format;
  const std::vector<std::string>& weights_file_list = SearchLocalPath(cache_path, model_file_format);
  auto worker_loader_threadpool = std::make_shared<ThreadPool>(16);
  worker_loader_threadpool->Start();
  std::atomic_bool all_ok = true;
  std::vector<std::future<void>> futures;
  futures.reserve(weights_file_list.size());
  std::mutex tensor_manager_mutex;
  for (const std::string& file_name : weights_file_list) {
    futures.emplace_back(worker_loader_threadpool->Submit([&, file_name]() {
      try {
        auto loader = std::make_shared<SafeTensorsLoader>(file_name, model_config_.load_bias);
        const auto& tensor_names = loader->GetTensorNameList();
        for (const std::string& tensor_name : tensor_names) {
          if (!IsPipelineNodeWeight(tensor_name)) {
            continue;
          }
          const auto& shape = loader->GetTensorShape(tensor_name);
          const auto& data_type = loader->GetTensorDataType(tensor_name);
          const auto& [weight_ptr, weight_size] = loader->GetTensor(tensor_name);
          {
            std::lock_guard<std::mutex> lock(tensor_manager_mutex);
            weights_data_type_map_[tensor_name] = data_type;
            const Status status = tensor_manager_->AddWeightTensor(tensor_name, shape, data_type);
            if (!status.OK()) {
              KLLM_LOG_INFO << fmt::format("Failed to load cache model from {}", cache_path.string());
              all_ok = false;
              break;
            }
            MemcpyAsync(weights_map_[tensor_name].GetPtr<void>(), weight_ptr, weights_map_[tensor_name].GetTotalBytes(),
                        MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
          }
        }
      } catch (...) {
        all_ok = false;
      }
    }));
  }
  // Wait for all file processing
  for (auto& future : futures) {
    future.wait();
  }

  if (!all_ok.load(std::memory_order_relaxed)) {
    KLLM_LOG_INFO << "Error occurred during parallel file loading";
    return false;
  }
  StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
  worker_loader_threadpool->Stop();
  return true;
}

bool BaseWeight::SaveWeightsToCacheFolder() {
  std::filesystem::path target_cache_path(GetCacheFolder());
  if (std::filesystem::exists(target_cache_path) && std::filesystem::is_directory(target_cache_path)) {
    KLLM_LOG_INFO << fmt::format("Skip save cache model: CacheFolder {} exists", target_cache_path.string());
    return true;
  }
  SafetensorsFileSaver saver(target_cache_path.append("model"), rank_, context_, 1024 * 1024 * 1024);
  saver.SaveTensors(weights_map_);
  if (std::filesystem::exists(target_cache_path) && std::filesystem::is_directory(target_cache_path)) {
    KLLM_LOG_INFO << fmt::format("cache model was saved to CacheFolder: {}", target_cache_path.string());
    return true;
  }
  return false;
}

std::string BaseWeight::GetCacheFolder() {
  const char* const model_cache_path_env = std::getenv("MODEL_CACHE_PATH");
  std::string base_path = model_config_.path;
  if (model_cache_path_env != nullptr) {
    base_path = std::string(model_cache_path_env);
  }
  std::string model_info = fmt::format("/cached_model_{}/tp{}", GetTypeString(model_config_.weight_data_type),
                                       runtime_config_.parallel_basic_config.tensor_parallel_size);
  std::string model_index = fmt::format("/tp_rank_{}", rank_);
  std::string model_cache_path = base_path + model_info + model_index;
  KLLM_LOG_INFO << fmt::format("using cache folder: {}", model_cache_path);
  return model_cache_path;
}

}  // namespace ksana_llm
