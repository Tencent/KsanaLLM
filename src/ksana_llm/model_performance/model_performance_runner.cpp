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

ModelPerformanceRunner::ModelPerformanceRunner(const std::string& config_path) {
  InitEnvs(config_path);
  LoadModel();
  model_instance_->AllocResources(schedule_id_);
  InitRequests();
}

ModelPerformanceRunner::~ModelPerformanceRunner() {
  model_instance_->FreeResources(schedule_id_);
  model_instance_->Reset();
  py::finalize_interpreter();
}

void ModelPerformanceRunner::InitEnvs(const std::string& config_path) {
  InitLoguru();
  py::initialize_interpreter();
  ParsePerformanceRunnerConfig(config_path);

  const auto& env = Singleton<Environment>::GetInstance();
  env->ParseConfig(config_path);

  // init context
  context_.reset(new Context(env->GetTensorParallelSize(), env->GetAttnDataParallelSize()));

  // init model_config
  env->GetModelConfig("", model_config_);

#ifdef ENABLE_CUDA
  // load gemm_algo_map
  if (context_->ext->GetGPUGemmAlgoHelper().LoadFromYaml(fmt::format("{}/gemm_algo_map.yaml", model_config_.path))) {
    KLLM_LOG_INFO << fmt::format("Load gemm algo from {}/gemm_algo_map.yaml success.", model_config_.path);
  }
#endif

  // set pipeline_config
  PipelineConfig pipeline_config;
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
  pipeline_config.lower_layer_idx = 0;
  pipeline_config.upper_layer_idx = model_config_.num_layer - 1;
  Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);

  // init BlockManager
  BlockManagerConfig block_manager_config;
  env->InitializeBlockManagerConfig();
  env->GetBlockManagerConfig(block_manager_config);
  OptimizeBlockManagerConfig(block_manager_config);
  env->SetBlockManagerConfig(block_manager_config);

  // init CacheManager
  CacheManagerConfig cache_manager_config;
  env->GetCacheManagerConfig(cache_manager_config);
  cache_manager_config.enable_prefix_caching = false;

  BlockAllocatorManagerConfig block_allocator_manager_config;
  attn_dp_worker_num_ = env->GetAttnDataParallelSize();
  // TODO(rockcao): support attn_dp_worker_num_ > 1
  KLLM_CHECK_WITH_INFO(attn_dp_worker_num_ == 1, "Currently only support attn_dp == 1");
  for (int dp_id = 0; dp_id < static_cast<uint32_t>(attn_dp_worker_num_); ++dp_id) {
    BlockAllocatorGroupConfig dp_group_config;
    dp_group_config.devices = env->GetDataParaGroupDevices(dp_id);
    dp_group_config.device_block_num = env->GetTotalDeviceBlockNum();
    dp_group_config.host_block_num = env->GetTotalHostBlockNum();
    dp_group_config.block_size = env->GetBlockSize();
    dp_group_config.convert_size = env->GetConvertSize();
    block_allocator_manager_config[dp_id] = dp_group_config;
    std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = std::make_shared<MemoryAllocator>();
    BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_);
    std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group =
        block_allocator_manager.GetBlockAllocatorGroup(dp_id);

    cache_managers_.emplace_back(CacheManagerFactory::CreateCacheManager(cache_manager_config, block_allocator_group));
  }
  cache_manager_ = cache_managers_[0];

  // init WorkerGroup
  size_t pp_batch_num = 1;
  worker_group_ = std::make_shared<WorkerGroup>(context_->GetTensorParallelSize(), pp_batch_num, context_);
}

void ModelPerformanceRunner::OptimizeBlockManagerConfig(BlockManagerConfig& block_manager_config) {
  // reset blocks_num to speedup
  size_t block_token_num = block_manager_config.device_allocator_config.block_token_num;
  size_t needed_block_num = GetNeededBlockNum(block_token_num);
  block_manager_config.device_allocator_config.blocks_num = needed_block_num;
  // do not need many blocks on host
  block_manager_config.host_allocator_config.blocks_num = 10;
  KLLM_LOG_INFO << fmt::format("Reset block_manager_config.device_allocator_config.blocks_num to {}", needed_block_num);
}

size_t ModelPerformanceRunner::GetNeededBlockNum(size_t block_token_num) {
  size_t tp_size = Singleton<Environment>::GetInstance()->GetTensorParallelSize();
  static constexpr size_t kExtraBlockNum = 10;
  size_t multi_token_request_block_num = (multi_token_request_token_num_ + block_token_num) / block_token_num;
  size_t single_token_request_block_num =
      (single_token_request_cached_token_num_ + 1 + block_token_num) / block_token_num;
  return tp_size * (single_token_request_num_ * single_token_request_block_num +
                    multi_token_request_num_ * multi_token_request_block_num + kExtraBlockNum);
}

void ModelPerformanceRunner::LoadModel() {
  std::shared_ptr<WeightInstanceInterface> weight_instance = std::make_shared<WeightInstance>(model_config_, context_);
  weight_instance->Load();
  model_instance_ = std::make_shared<ModelInstance>(model_config_, context_, weight_instance);
  model_instance_->Load();
}

Status ModelPerformanceRunner::RunPerformanceForward() {
#ifndef ENABLE_CUDA
  KLLM_LOG_INFO << "Currently RunPerformanceForward only supports cuda";
  return Status();
#endif
  int device_id = 0;
  SetDevice(device_id);
  Event start;
  Event stop;
  EventCreate(&start);
  EventCreate(&stop);
  float milliseconds = 0;

  // warmup
  KLLM_LOG_INFO << fmt::format("Start warmup of {} rounds", warmp_up_rounds_);
  Status warmup_status = Status();
  for (size_t i = 0; i < warmp_up_rounds_; ++i) {
    std::vector<std::future<Status>> inst_results =
        model_instance_->ForwardAsync(schedule_id_, worker_group_, InferStage::STATE_DECODE, forward_reqs_, false);
    for (auto& worker_result : inst_results) {
      Status status = worker_result.get();
      if (!status.OK()) {
        warmup_status = status;
      }
    }
  }
  KLLM_LOG_INFO << fmt::format("Warmup Done with status {}", warmup_status.GetMessage());

  // run
  Status result_status = Status();
  KLLM_LOG_INFO << fmt::format("Start run model performance of {} rounds", warmp_up_rounds_);
  EventRecord(start, context_->GetComputeStreams()[device_id]);
  for (size_t i = 0; i < rounds_; ++i) {
    std::vector<std::future<Status>> inst_results =
        model_instance_->ForwardAsync(schedule_id_, worker_group_, InferStage::STATE_DECODE, forward_reqs_, false);
    for (auto& worker_result : inst_results) {
      Status status = worker_result.get();
      if (!status.OK()) {
        result_status = status;
      }
    }
  }
  EventRecord(stop, context_->GetComputeStreams()[device_id]);
  EventSynchronize(stop);
  EventElapsedTime(&milliseconds, start, stop);

  std::stringstream ss;
  if (result_status.OK()) {
    ss << fmt::format("\n Performance Results: {} rounds cost {} milliseconds \n", rounds_, milliseconds)
       << fmt::format("Average single Forward of {} cost : {} milliseconds/round \n", model_config_.type,
                      milliseconds / rounds_);
    KLLM_LOG_INFO << ss.str();
    std::cout << ss.str();
  } else {
    ss << fmt::format("Faild to run model_preformance. End with status {}", result_status.GetMessage());
    KLLM_LOG_ERROR << ss.str();
    std::cout << ss.str();
  }
  return result_status;
}

void ModelPerformanceRunner::InitRequests() {
  // random token generater
  static constexpr size_t kVocabSize = 10000;
  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_int_distribution<> dis(0, kVocabSize);

  size_t total_req_num = single_token_request_num_ + multi_token_request_num_;
  input_ids_vec_.resize(total_req_num);
  forward_reqs_.resize(total_req_num);
  embedding_slice_.pos = input_refit_pos_;
  embedding_slice_.embeddings = input_refit_embedding_;

  size_t tp_size = Singleton<Environment>::GetInstance()->GetTensorParallelSize();
  size_t req_id = 0;
  for (; req_id < total_req_num; req_id++) {
    ForwardRequest& req = forward_reqs_[req_id];
    req.cache_manager = cache_manager_;
    req.req_id = req_id;
    req.sampling_config = &sampling_config_;
    req.draft_token_num = 0;
    req.flexible_cached_copy_tasks = &flexible_cached_copy_tasks_;
    req.input_refit_embedding = &embedding_slice_;
    req.logits_buf = model_instance_->GetLogitsPtr(schedule_id_);
    req.logits_offset = 0;
    req.attn_dp_group_id = 0;
    if (req_id < multi_token_request_num_) {  // multi_token_request
      input_ids_vec_[req_id].resize(multi_token_request_token_num_);
      req.forwarding_tokens = &input_ids_vec_[req_id];
      std::generate(req.forwarding_tokens->begin(), req.forwarding_tokens->end(), [&]() { return dis(gen); });
      req.infer_stage = InferStage::STAGE_CONTEXT;
      req.kv_cached_token_num = multi_token_cached_token_num_;
      req.prefix_cache_len = multi_token_cached_token_num_;
    } else {  // single_token_request
      input_ids_vec_[req_id].resize(single_token_request_cached_token_num_ + 1);
      req.forwarding_tokens = &input_ids_vec_[req_id];
      std::generate(req.forwarding_tokens->begin(), req.forwarding_tokens->end(), [&]() { return dis(gen); });
      req.infer_stage = InferStage::STATE_DECODE;
      req.kv_cached_token_num = req.forwarding_tokens->size() - 1;
    }
    req.kv_cache_ptrs.resize(tp_size);
    std::vector<std::vector<int>> block_ids(tp_size);
    for (size_t rank = 0; rank < tp_size; ++rank) {
      SetDevice(rank);
      KLLM_CHECK_WITH_INFO(cache_manager_->GetBlockAllocatorGroup()
                               ->GetDeviceBlockAllocator(rank)
                               ->AllocateBlocks(GetBlockNum(req), block_ids[rank])
                               .OK(),
                           "faild to allocate blocks");
      cache_manager_->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(rank)->GetBlockPtrs(block_ids[rank],
                                                                                            req.kv_cache_ptrs[rank]);
    }
#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
    uint32_t layer_num = model_instance_->GetLayerNum();
    LlmRuntime::BuildFlatKVCacheBlkIds(layer_num, block_ids, req.atb_kv_cache_base_blk_ids, req.cache_manager);
#endif
  }

  CheckRequests();
}

void ModelPerformanceRunner::CheckRequests() {
  BatchSchedulerConfig batch_scheduler_config;
  KLLM_CHECK_WITH_INFO(Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config).OK(),
                       "Failed to get batch scheduler config error");
  KLLM_CHECK_WITH_INFO(batch_scheduler_config.max_batch_size >= forward_reqs_.size(),
                       fmt::format("max_batch_size {} should not less than number of requests {}",
                                   batch_scheduler_config.max_batch_size, forward_reqs_.size()));
  size_t step_tokens =
      std::accumulate(forward_reqs_.begin(), forward_reqs_.end(), size_t{0}, [](size_t acc, const ForwardRequest& req) {
        return acc + (req.infer_stage == InferStage::STAGE_CONTEXT ? req.forwarding_tokens->size() : 1);
      });
  KLLM_CHECK_WITH_INFO(batch_scheduler_config.max_step_token_num >= step_tokens,
                       fmt::format("max_step_token_num {} should not less than step_tokens {}",
                                   batch_scheduler_config.max_step_token_num, step_tokens));
}

Status ModelPerformanceRunner::ParsePerformanceRunnerConfig(const std::string& config_file) {
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
  multi_token_cached_token_num_ = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "model_performance_runner_config.input_config.multi_token_cached_token_num", 0);
  multi_token_request_token_num_ = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "model_performance_runner_config.input_config.multi_token_request_token_num", 32);

  // Read runner_config
  warmp_up_rounds_ = yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(),
                                                   "model_performance_runner_config.runner_config.warmup_rounds", 10);
  rounds_ = yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(),
                                          "model_performance_runner_config.runner_config.rounds", 100);

  if (!(single_token_request_num_ > 0 || multi_token_request_num_ > 0)) {
    KLLM_THROW(fmt::format("single_token_request_num {} or multi_token_request_num {} should > 0",
                           single_token_request_num_, multi_token_request_num_));
  }
  return Status();
}

size_t ModelPerformanceRunner::GetBlockNum(const ForwardRequest& req) {
  size_t shared_block_num = 0;
  size_t unique_block_num = 0;
  size_t shared_token_num = 0;
  cache_manager_->GetRequestPrefixBlockNumber(req.req_id, *req.forwarding_tokens, shared_block_num, unique_block_num,
                                              shared_token_num);
  return shared_block_num + unique_block_num;
}
}  // namespace ksana_llm
