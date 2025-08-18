/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/weight_loader/model_weight_loader.h"
#include <future>

#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/model_loader/weight_loader/model_weight_loader_factory.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

ModelWeightLoader::ModelWeightLoader(std::shared_ptr<Environment> env, std::shared_ptr<Context> context) {
  env_ = env;
  context_ = context;

  weight_loader_threadpool_ = std::make_shared<ThreadPool>(context->GetTensorParallelSize());
  weight_loader_threadpool_->Start();
}

ModelWeightLoader::~ModelWeightLoader() { weight_loader_threadpool_->Stop(); }

Status ModelWeightLoader::LoadWeights(std::shared_ptr<BaseModelConfig>& model_config,
                                      std::vector<std::shared_ptr<ModelWeight>>& dev_weights) {
  const auto start_time = std::chrono::high_resolution_clock::now();
  std::shared_ptr<BaseModelWeightLoader> model_weight_loader;
  Status status = ModelWeightLoaderFactory::CreateModelWeightLoader(model_config->model_arch, model_config, env_,
                                                                    context_, model_weight_loader);
  STATUS_CHECK_RETURN(status);

  std::string model_type;
  status = GetModelTypeFromArchitecture(model_config->model_arch, model_type);
  STATUS_CHECK_RETURN(status);

  status = model_weight_loader->InitRegexPatterns(model_config->model_dir, model_type, model_config->model_format);
  STATUS_CHECK_RETURN(status);

  std::vector<std::string> model_file_list;
  status = GetModelFileList(model_config->model_dir, model_file_list);
  STATUS_CHECK_RETURN(status);

  status = FilterModelFormatFiles(model_config->model_format, model_file_list);
  STATUS_CHECK_RETURN(status);

  status = model_weight_loader->FilterModelFiles(model_file_list);
  STATUS_CHECK_RETURN(status);

  const size_t tp_size = context_->GetTensorParallelSize();
  dev_weights.reserve(tp_size);
  for (size_t i = 0; i < tp_size; ++i) {
    dev_weights.emplace_back(std::make_shared<ModelWeight>());
  }

  std::vector<std::unordered_map<std::string, Tensor>> left_host_weights(tp_size);
  std::vector<std::mutex> tp_mutex(tp_size);
  std::vector<std::future<void>> load_weight_tasks;

  // file loader may use mmap, hold it until finish load weight
  std::vector<FileLoader> file_loaders;
  file_loaders.reserve(model_file_list.size());
  for (const std::string& model_file : model_file_list) {
    auto& file_loader = file_loaders.emplace_back(model_file);

    std::vector<std::string> weight_names;
    status = file_loader.LoadWeightNames(model_config->model_format, weight_names);
    STATUS_CHECK_RETURN(status);

    status = model_weight_loader->FilterWeightNames(weight_names);
    STATUS_CHECK_RETURN(status);

    auto host_model_weights = std::make_shared<std::unordered_map<std::string, Tensor>>();
    status = file_loader.LoadModelWeights(model_config->model_format, weight_names, *host_model_weights);
    STATUS_CHECK_RETURN(status);

    // Process common task for all tp devices.
    model_weight_loader->PreProcessModelWeights(*host_model_weights);

    for (size_t dev_rank = 0; dev_rank < tp_size; ++dev_rank) {
      load_weight_tasks.emplace_back(weight_loader_threadpool_->Submit([&, dev_rank, host_model_weights]() {
        // The current implementation can only process serially per card.
        std::lock_guard<std::mutex> lock(tp_mutex[dev_rank]);
        SetDevice(dev_rank);
        std::unordered_map<std::string, Tensor> model_weights = *host_model_weights;
        model_weight_loader->GetCustomWeightMap(model_config->model_dir, model_type, model_weights,
                                                model_config->model_format);
        auto& left_tensor = left_host_weights[dev_rank];
        model_weights.insert(left_tensor.begin(), left_tensor.end());
        left_tensor.clear();
        model_weight_loader->ProcessModelWeights(model_weights, dev_rank, dev_weights[dev_rank]->weights_map_,
                                                 left_tensor);
        StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
      }));
    }
  }

  // Wait all task finished.
  for (auto& task : load_weight_tasks) {
    task.get();
  }
  load_weight_tasks.clear();

  // post process
  for (size_t dev_rank = 0; dev_rank < tp_size; dev_rank++) {
    load_weight_tasks.emplace_back(weight_loader_threadpool_->Submit([&, dev_rank]() {
      SetDevice(dev_rank);
      model_weight_loader->PostProcessModelWeights(dev_weights[dev_rank]->weights_map_, dev_rank);
      StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
    }));
  }

  for (auto& task : load_weight_tasks) {
    task.get();
  }

  const std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start_time;
  KLLM_LOG_INFO << fmt::format("LoadWeights cost: {} seconds", elapsed.count());
  return Status();
}

}  // namespace ksana_llm
