/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/model_performance/model_performance_runner.h"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <filesystem>
#include <random>

#include "ksana_llm/cache_manager/cache_manager_factory.h"
#include "ksana_llm/runtime/weight_instance.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/yaml_reader.h"

namespace py = pybind11;

namespace ksana_llm {

ModelPerformanceRunner::ModelPerformanceRunner(const std::string& config_path, const PerfProfileConfig& max_config,
                                               int16_t lower_layer_idx, int16_t upper_layer_idx) {
  InitEnvs(config_path, max_config, lower_layer_idx, upper_layer_idx);
  LoadModel();
  model_instance_->AllocResources(multi_batch_id_);
}

ModelPerformanceRunner::~ModelPerformanceRunner() {
  model_instance_->FreeResources(multi_batch_id_);
  model_instance_->Reset();
  py::finalize_interpreter();
}

void ModelPerformanceRunner::InitEnvs(const std::string& config_path, const PerfProfileConfig& max_config,
                                      int16_t lower_layer_idx, int16_t upper_layer_idx) {
  py::initialize_interpreter();

  const auto& env = Singleton<Environment>::GetInstance();
  env->ParseConfig(config_path);

  // init context
  env->GetRuntimeConfig(runtime_config_);
  constexpr int max_multi_batch_num = 1;
  context_.reset(new Context(runtime_config_.parallel_basic_config.tensor_parallel_size,
                             runtime_config_.parallel_basic_config.attn_data_parallel_size, max_multi_batch_num));

  // init model_config
  Status status = env->GetModelConfig(model_config_);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "GetModelConfig failed. status: " << status.ToString();
    return;
  }

#ifdef ENABLE_CUDA
  // load gemm_algo_map
  if (context_->ext->GetGPUGemmAlgoHelper().LoadFromYaml(fmt::format("{}/gemm_algo_map.yaml", model_config_.path))) {
    KLLM_LOG_INFO << fmt::format("Load gemm algo from {}/gemm_algo_map.yaml success.", model_config_.path);
  }
#endif

  // set pipeline_config
  PipelineConfig pipeline_config;
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
  if (lower_layer_idx > 0) {
    pipeline_config.lower_layer_idx = lower_layer_idx;
  } else {
    pipeline_config.lower_layer_idx = 0;
  }
  if (upper_layer_idx > 0) {
    pipeline_config.upper_layer_idx = upper_layer_idx;
  } else {
    pipeline_config.upper_layer_idx = model_config_.num_layer - 1;
  }
  Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);

  BatchSchedulerConfig batch_scheduler_config;
  env->GetBatchSchedulerConfig(batch_scheduler_config);
  llm_runtime_ = std::make_shared<LlmRuntime>(batch_scheduler_config, runtime_config_, context_);

  // init BlockManager
  BlockManagerConfig block_manager_config;
  env->InitializeBlockManagerConfig();
  env->GetBlockManagerConfig(block_manager_config);
  OptimizeBlockManagerConfig(block_manager_config, max_config);
  env->SetBlockManagerConfig(block_manager_config);

  // init CacheManager
  CacheManagerConfig cache_manager_config;
  env->GetCacheManagerConfig(cache_manager_config);
  cache_manager_config.enable_prefix_caching = false;

  // RuntimeConfig is set in InitializeBlockManagerConfig
  env->GetRuntimeConfig(runtime_config_);

  BlockAllocatorManagerConfig block_allocator_manager_config;
  attn_dp_worker_num_ = runtime_config_.parallel_basic_config.attn_data_parallel_size;
  for (int dp_id = 0; dp_id < static_cast<uint32_t>(attn_dp_worker_num_); ++dp_id) {
    BlockAllocatorGroupConfig dp_group_config;
    dp_group_config.devices = env->GetDataParaGroupDevices(dp_id);
    dp_group_config.device_block_num = env->GetTotalDeviceBlockNum();
    dp_group_config.host_block_num = env->GetTotalHostBlockNum();
    dp_group_config.block_size = runtime_config_.attn_backend_config.block_size;
    block_allocator_manager_config[dp_id] = dp_group_config;
  }
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = std::make_shared<MemoryAllocator>();
  BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_);
  for (int dp_id = 0; dp_id < attn_dp_worker_num_; ++dp_id) {
    std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group =
        block_allocator_manager.GetBlockAllocatorGroup(dp_id);
    cache_managers_.emplace_back(CacheManagerFactory::CreateCacheManager(cache_manager_config, block_allocator_group));
    cache_managers_.back()->InitializeCachedBlocks();
  }

  llm_runtime_->SetCacheManagers(cache_managers_);

  // init WorkerGroup
  size_t pp_batch_num = 1;
  worker_group_ = std::make_shared<WorkerGroup>(context_->GetTensorParallelSize(), pp_batch_num, context_);
}

void ModelPerformanceRunner::OptimizeBlockManagerConfig(BlockManagerConfig& block_manager_config,
                                                        const PerfProfileConfig& max_config) {
  // reset blocks_num to speedup
  size_t block_token_num = block_manager_config.device_allocator_config.block_token_num;
  size_t needed_block_num = GetNeededBlockNum(block_token_num, max_config);
  block_manager_config.device_allocator_config.blocks_num = needed_block_num;
  // do not need many blocks on host
  block_manager_config.host_allocator_config.blocks_num = 10;
  KLLM_LOG_INFO << fmt::format("Reset block_manager_config.device_allocator_config.blocks_num to {}", needed_block_num);
}

size_t ModelPerformanceRunner::GetNeededBlockNum(size_t block_token_num, const PerfProfileConfig& max_config) const {
  size_t tp_size = runtime_config_.parallel_basic_config.tensor_parallel_size;
  static constexpr size_t kExtraBlockNum = 10;

  size_t single_token_request_num = 0;
  size_t single_token_request_cached_token_num = 0;
  size_t multi_token_request_num = 0;
  size_t multi_token_request_token_num = 0;

  for (size_t dp_idx = 0; dp_idx < max_config.req_configs.size(); dp_idx++) {
    auto& req_config = max_config.req_configs[0];
    single_token_request_num += req_config.single_token_request_num;
    single_token_request_cached_token_num += req_config.single_token_request_cached_token_num;
    multi_token_request_num += req_config.multi_token_request_num;
    multi_token_request_token_num += req_config.multi_token_request_token_num;
  }

  size_t multi_token_request_block_num = (multi_token_request_token_num + block_token_num) / block_token_num;
  size_t single_token_request_block_num =
      (single_token_request_cached_token_num + 1 + block_token_num) / block_token_num;
  return tp_size * (single_token_request_num * single_token_request_block_num +
                    multi_token_request_num * multi_token_request_block_num + kExtraBlockNum);
}

void ModelPerformanceRunner::LoadModel() {
  std::shared_ptr<WeightInstanceInterface> weight_instance =
      std::make_shared<WeightInstance>(model_config_, runtime_config_, context_);
  weight_instance->Load();
  model_instance_ = std::make_shared<ModelInstance>(model_config_, runtime_config_, context_, weight_instance);
  model_instance_->Load();
}

Status ModelPerformanceRunner::RunPerformanceForward(const PerfProfileConfig& profile_config,
                                                     PerfProfileResult& result) {
#ifndef ENABLE_CUDA
  KLLM_LOG_INFO << "Currently RunPerformanceForward only supports cuda";
  return Status();
#endif

  InitInferRequests(profile_config);
  int device_id = 0;
  SetDevice(device_id);
  Event start;
  Event stop;
  EventCreate(&start);
  EventCreate(&stop);
  float milliseconds = 0;
  llm_runtime_->ReorderInferRequests<InferRequest>(infer_reqs_);

  // warmup
  Status warmup_status = Status();
  for (size_t i = 0; i < profile_config.warmup_round; ++i) {
    warmup_status = llm_runtime_->Forward(multi_batch_id_, infer_reqs_, false);
  }

  // Sleep 100 ms to make nsys results easier to be analyzed
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  // run
  KLLM_LOG_INFO << fmt::format("Start run model performance of {} rounds", profile_config.warmup_round);
  EventRecord(start, context_->GetComputeStreams()[device_id]);
  Status result_status = Status();
  for (size_t i = 0; i < profile_config.profile_round; ++i) {
    result_status = llm_runtime_->Forward(multi_batch_id_, infer_reqs_, false);
    if (!result_status.OK()) {
      KLLM_LOG_ERROR << "Forward error. Msg: " << result_status.GetMessage();
      return result_status;
    }
  }
  EventRecord(stop, context_->GetComputeStreams()[device_id]);
  EventSynchronize(stop);
  EventElapsedTime(&milliseconds, start, stop);

  result.config_id = profile_config.config_id;
  result.time_cost_ms = milliseconds;
  return result_status;
}

void ModelPerformanceRunner::ResetInferRequests() {
  infer_reqs_.clear();
  input_ids_map_.clear();
  input_refit_pos_.clear();
  embeddings_.clear();
  embedding_tensors_.clear();
}

void ModelPerformanceRunner::InitInferRequests(const PerfProfileConfig& profile_config) {
  static constexpr size_t kVocabSize = 10000;
  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_int_distribution<> dis(0, kVocabSize);

  ResetInferRequests();

  embedding_slice_.pos = input_refit_pos_;
  embedding_slice_.embeddings = embeddings_;
  embedding_slice_.embedding_tensors = embedding_tensors_;
  ksana_python_input_ = std::make_shared<KsanaPythonInput>();
  ksana_python_input_->input_refit_embedding = embedding_slice_;
  std::shared_ptr<std::unordered_map<std::string, std::string>> req_ctx =
      std::make_shared<std::unordered_map<std::string, std::string>>();
  std::shared_ptr<Request> request = std::make_shared<Request>(ksana_python_input_, req_ctx);
  size_t req_id = 0;
  for (size_t dp_idx = 0; dp_idx < profile_config.req_configs.size(); dp_idx++) {
    auto& req_config = profile_config.req_configs[dp_idx];

    size_t single_token_request_num = req_config.single_token_request_num;
    size_t single_token_request_cached_token_num = req_config.single_token_request_cached_token_num;
    size_t multi_token_request_num = req_config.multi_token_request_num;
    size_t multi_token_request_token_num = req_config.multi_token_request_token_num;

    size_t dp_req_num = single_token_request_num + multi_token_request_num;
    for (size_t req_idx = 0; req_idx < dp_req_num; req_idx++) {
      req_id++;
      std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(request, req_id);

      infer_reqs_.push_back(infer_req);
      infer_req->attn_dp_group_id = dp_idx;
      infer_req->kv_cache_blocks.resize(runtime_config_.parallel_basic_config.attn_tensor_parallel_size);
      infer_req->cache_manager = cache_managers_[infer_req->attn_dp_group_id];
      infer_req->block_token_num = runtime_config_.attn_backend_config.block_token_num;
      infer_req->model_instance = model_instance_;
      infer_req->step = 0;  // not using

      if (req_id < multi_token_request_num) {  // multi_token_request
        input_ids_map_[req_id].resize(multi_token_request_token_num);
        infer_req->input_tokens = input_ids_map_[req_id];
        std::generate(infer_req->input_tokens.begin(), infer_req->input_tokens.end(), [&]() { return dis(gen); });
        infer_req->infer_stage = InferStage::STAGE_CONTEXT;
        infer_req->kv_cached_token_num = 0;
        infer_req->prefix_cache_len = 0;
      } else {  // single_token_request
        input_ids_map_[req_id].resize(single_token_request_cached_token_num + 1);
        infer_req->input_tokens = input_ids_map_[req_id];
        std::generate(infer_req->input_tokens.begin(), infer_req->input_tokens.end(), [&]() { return dis(gen); });
        infer_req->infer_stage = InferStage::STATE_DECODE;
        infer_req->kv_cached_token_num = infer_req->input_tokens.size() - 1;
      }
      infer_req->forwarding_tokens = infer_req->input_tokens;
      Status status = infer_req->cache_manager->AllocateRequestBlocks(infer_req->req_id, GetBlockNum(infer_req),
                                                                      infer_req->kv_cache_blocks);
      if (!status.OK()) {
        KLLM_THROW(fmt::format("AllocateRequestBlocks failed. status: {}", status.ToString()));
      }
    }
  }
  CheckRequests();
}

void ModelPerformanceRunner::CheckRequests() const {
  BatchSchedulerConfig batch_scheduler_config;
  KLLM_CHECK_WITH_INFO(Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config).OK(),
                       "Failed to get batch scheduler config error");
  KLLM_CHECK_WITH_INFO(batch_scheduler_config.max_batch_size >= infer_reqs_.size(),
                       fmt::format("max_batch_size {} should not less than number of requests {}",
                                   batch_scheduler_config.max_batch_size, infer_reqs_.size()));
  size_t step_tokens = std::accumulate(
      infer_reqs_.begin(), infer_reqs_.end(), size_t{0}, [](size_t acc, std::shared_ptr<InferRequest> req) {
        return acc + (req->infer_stage == InferStage::STAGE_CONTEXT ? req->forwarding_tokens.size() : 1);
      });
  KLLM_CHECK_WITH_INFO(batch_scheduler_config.max_step_token_num >= step_tokens,
                       fmt::format("max_step_token_num {} should not less than step_tokens {}",
                                   batch_scheduler_config.max_step_token_num, step_tokens));
}

size_t ModelPerformanceRunner::GetBlockNum(std::shared_ptr<InferRequest> req) const {
  size_t shared_block_num = 0;
  size_t unique_block_num = 0;
  size_t shared_token_num = 0;
  req->cache_manager->GetRequestPrefixBlockNumber(req->req_id, req->forwarding_tokens, shared_block_num,
                                                  unique_block_num, shared_token_num);
  return shared_block_num + unique_block_num;
}

/***************************************************************************************************
 * PerfProfileConfigBuilderWithYaml
 */

PerfProfileConfigBuilderWithYaml::PerfProfileConfigBuilderWithYaml(const std::string& config_file) {
  ParsePerformanceRunnerConfig(config_file);
}

PerfProfileConfig PerfProfileConfigBuilderWithYaml::GetMaxPerfProfileConfig() {
  // TODO(robertyuan): build multiple configs
  PerfProfileConfig max_config;
  max_config.config_id = 0;
  max_config.warmup_round = warmup_round_;
  max_config.profile_round = profile_round_;
  max_config.req_configs.resize(1);
  max_config.req_configs[0].single_token_request_num = single_token_request_num_;
  max_config.req_configs[0].single_token_request_cached_token_num = single_token_request_cached_token_num_;
  max_config.req_configs[0].multi_token_request_num = multi_token_request_num_;
  max_config.req_configs[0].multi_token_request_token_num = multi_token_request_token_num_;
  return max_config;
}

void PerfProfileConfigBuilderWithYaml::GetPerfProfileConfigs(std::vector<PerfProfileConfig>& configs) {
  configs.resize(1);
  auto& config = configs[0];
  config.config_id = 0;
  config.warmup_round = warmup_round_;
  config.profile_round = profile_round_;
  config.req_configs.resize(dp_num_);
  for (size_t dp_idx = 0; dp_idx < dp_num_; dp_idx++) {
    config.req_configs[dp_idx].single_token_request_num = single_token_request_num_;
    config.req_configs[dp_idx].single_token_request_cached_token_num = single_token_request_cached_token_num_;
    config.req_configs[dp_idx].multi_token_request_num = multi_token_request_num_;
    config.req_configs[dp_idx].multi_token_request_token_num = multi_token_request_token_num_;
  }
}

Status PerfProfileConfigBuilderWithYaml::ParsePerformanceRunnerConfig(const std::string& config_file) {
  YamlReader yaml_reader;
  Status status = yaml_reader.LoadFile(config_file);
  if (!status.OK()) {
    KLLM_THROW(fmt::format("Load yaml config error. {}", status.GetMessage()));
  }

  // Read input_config
  single_token_request_num_ = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "model_performance_runner_config.input_config.single_token_request_num", 4);
  single_token_request_cached_token_num_ = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "model_performance_runner_config.input_config.single_token_request_cached_token_num",
      32);
  multi_token_request_num_ = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "model_performance_runner_config.input_config.multi_token_request_num", 4);
  multi_token_request_token_num_ = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "model_performance_runner_config.input_config.multi_token_request_token_num", 32);

  // Read runner_config
  warmup_round_ = yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(),
                                                "model_performance_runner_config.runner_config.warmup_rounds", 10);
  profile_round_ = yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(),
                                                 "model_performance_runner_config.runner_config.rounds", 100);

  if (!(single_token_request_num_ > 0 || multi_token_request_num_ > 0)) {
    KLLM_THROW(fmt::format("single_token_request_num {} or multi_token_request_num {} should > 0",
                           single_token_request_num_, multi_token_request_num_));
  }
  return Status();
}

}  // namespace ksana_llm
