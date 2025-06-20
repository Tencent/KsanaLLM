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
  std::shared_ptr<BaseModelWeightLoader> model_weight_loader;
  Status status = ModelWeightLoaderFactory::CreateModelWeightLoader(model_config->model_arch, model_config, env_,
                                                                    context_, model_weight_loader);
  STATUS_CHECK_RETURN(status);

  std::vector<std::string> model_file_list;
  status = GetModelFileList(model_config->model_dir, model_file_list);
  STATUS_CHECK_RETURN(status);

  status = FilterModelFormatFiles(model_config->model_format, model_file_list);
  STATUS_CHECK_RETURN(status);

  status = model_weight_loader->FilterModelFiles(model_file_list);
  STATUS_CHECK_RETURN(status);

  size_t tp_size = context_->GetTensorParallelSize();
  for (size_t i = 0; i < tp_size; ++i) {
    std::shared_ptr<ModelWeight> weight = std::make_shared<ModelWeight>();
    dev_weights.push_back(weight);
  }

  std::vector<std::unordered_map<std::string, Tensor>> left_host_weights(tp_size);
  std::vector<std::unordered_map<std::string, Tensor>> dev_model_weights;
  dev_model_weights.resize(tp_size);

  for (const std::string& model_file : model_file_list) {
    FileLoader file_loader(model_file);

    std::vector<std::string> weight_names;
    status = file_loader.LoadWeightNames(model_config->model_format, weight_names);
    STATUS_CHECK_RETURN(status);

    status = model_weight_loader->FilterWeightNames(weight_names);
    STATUS_CHECK_RETURN(status);

    std::unordered_map<std::string, Tensor> host_model_weights;
    status = file_loader.LoadModelWeights(model_config->model_format, weight_names, host_model_weights);
    STATUS_CHECK_RETURN(status);

    // Process common task for all tp devices.
    model_weight_loader->PreProcessModelWeights(host_model_weights);

    std::vector<std::unordered_map<std::string, Tensor>> merged_tensors(tp_size);

    std::vector<std::future<void>> process_weight_tasks;
    for (int dev_rank = 0; dev_rank < tp_size; ++dev_rank) {
      // Merge left host weights.
      merged_tensors[dev_rank].insert(host_model_weights.begin(), host_model_weights.end());
      merged_tensors[dev_rank].insert(left_host_weights[dev_rank].begin(), left_host_weights[dev_rank].end());
      left_host_weights[dev_rank].clear();

      process_weight_tasks.push_back(weight_loader_threadpool_->Submit([&, dev_rank]() {
        SetDevice(dev_rank);
        model_weight_loader->ProcessModelWeights(merged_tensors[dev_rank], dev_rank, dev_model_weights[dev_rank],
                                                 left_host_weights[dev_rank]);
        StreamSynchronize(context_->GetMemoryManageStreams()[dev_rank]);
      }));
    }

    // Wait all task finished.
    for (size_t i = 0; i < process_weight_tasks.size(); ++i) {
      process_weight_tasks[i].get();
    }
  }

  for (size_t i = 0; i < tp_size; i++) {
    dev_weights[i]->weights_map_.insert(dev_model_weights[i].begin(), dev_model_weights[i].end());
    model_weight_loader->PostProcessModelWeights(dev_weights[i]->weights_map_, i);
  }
  return Status();
}

}  // namespace ksana_llm
