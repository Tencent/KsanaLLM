/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/service/inference_engine.h"

#include <chrono>
#include <memory>
#include <thread>

#include "ksana_llm/cache_manager/cache_manager_factory.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/expert_data_hub.h"
#include "ksana_llm/periphery/version_reporter.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/draft_generator/trie_generator.h"
#include "ksana_llm/runtime/layer_progress_tracker.h"
#include "ksana_llm/runtime/weight_instance.h"
#include "ksana_llm/transfer/transfer_engine.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tokenizer.h"
#include "ksana_llm/utils/waiter.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/runtime/cuda_graph_runner.h"
#endif

namespace ksana_llm {

InferenceEngine::InferenceEngine(Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : request_queue_(request_queue) {
  Initialize();
}

InferenceEngine::~InferenceEngine() { KLLM_LOG_DEBUG << "InferenceEngine destroyed."; }

Status InferenceEngine::Initialize() {
  std::shared_ptr<Environment> env = Singleton<Environment>::GetInstance();
  if (!env) {
    return Status(RET_INVALID_ARGUMENT, "The Environment is nullptr.");
  }

  // get batch schedule config;
  BatchSchedulerConfig batch_scheduler_config;
  Status status = env->GetBatchSchedulerConfig(batch_scheduler_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get batch manager config error:" + status.ToString());
  }
  // TODO(TJ): maybe cloud move IsChief and IsStandalone to env.
  PipelineConfig pipeline_config;
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
  bool is_chief = pipeline_config.world_size == 1 || pipeline_config.node_rank == 0;
  if (!is_chief) {
    batch_scheduler_config.max_pp_batch_num = 1;
  }
  // Environment is must be initialized befroe context.
  KLLM_LOG_INFO << "Get tensor parallel: " << env->GetTensorParallelSize()
                 << " attention data parallel: " << env->GetAttnDataParallelSize()
                 << " max_pp_batch_num: " << batch_scheduler_config.max_pp_batch_num;
  context_.reset(new Context(env->GetTensorParallelSize(), env->GetAttnDataParallelSize(),
                             batch_scheduler_config.max_pp_batch_num));

  // Load model configs.
  ModelConfig model_config;
  status = env->GetModelConfig(model_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get model config error:" + status.ToString());
  }

  // Initialize schedule output and hidden unit buffer pool.
  // Must be called after block manager is set.
  InitializeScheduleOutputPool();

  // 初始化 LayerProgressTracker，为每个设备的每一层创建 CUDA event
  Singleton<LayerProgressTracker>::GetInstance()->Initialize(
      env->GetTensorParallelSize(), model_config.num_layer + model_config.num_nextn_predict_layers);

  Singleton<LayerProgressTracker>::GetInstance()->RegisterCallback(
      [&](int device_id, int layer_index) { TransferEngine::GetInstance()->Send(device_id, layer_index); });

  // Only for distributed mode.
  if (!context_->IsStandalone() || !context_->IsExpertParallelStandalone()) {
    if (!context_->IsStandalone()) {
      InitializeHiddenUnitBufferPool();  // Does HiddenUnitBufferPool must has one comm type? (No)
      GetHiddenUnitBufferPool()->SetCommType(DistributedCommunicationType::ALLTOALL);
    }

    if (!context_->IsExpertParallelStandalone()) {
      InitializeExpertHiddenUnitBufferPool();
      GetExpertHiddenUnitBufferPool()->SetCommType(DistributedCommunicationType::SCATTER);
    }

    distributed_coordinator_ =
        std::make_shared<DistributedCoordinator>(context_, GetPacketObject, GetScheduleOutputPool(),
                                                 GetHiddenUnitBufferPool(), GetExpertHiddenUnitBufferPool(), env);

    KLLM_LOG_INFO << "Initialize distributed coordinator succeed.";
    distributed_coordinator_->InitializeCluster();
  }

  // Set model layers for standalone mode, assume only one model now.
  KLLM_LOG_INFO << "InferenceEngine PiplineParallel IsStandalone:" << context_->IsStandalone();
  if (context_->IsStandalone()) {
    PipelineConfig pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
    pipeline_config.lower_layer_idx = 0;
    pipeline_config.upper_layer_idx = model_config.num_layer - 1;
    if (model_config.num_nextn_predict_layers != 0 && env->IsMTPEnabled()) {
      pipeline_config.lower_nextn_layer_idx = model_config.num_layer;
      pipeline_config.upper_nextn_layer_idx = model_config.num_layer + model_config.num_nextn_predict_layers - 1;
    }
    Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);
    KLLM_LOG_INFO << "InferenceEngine Set layer range:[" << pipeline_config.lower_layer_idx << ", "
                  << pipeline_config.upper_layer_idx << "].";
  } else {
    KLLM_LOG_INFO << "Start to synchronize node layers.";
    // Get master_offload_layer_num from environment variable, default is 0
    const char *master_offload_layer_num_env = std::getenv("MASTER_OFFLOAD_LAYER_NUM");
    size_t master_offload_layer_num = master_offload_layer_num_env ? std::stoi(master_offload_layer_num_env) : 0;
    KLLM_LOG_INFO << "Master offload layer num: " << master_offload_layer_num;
    distributed_coordinator_->SynchronizeNodeLayers(master_offload_layer_num);

    PipelineConfig pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
    KLLM_LOG_INFO << "InferenceEngine Synchronize layer range:[" << pipeline_config.lower_layer_idx << ", "
                  << pipeline_config.upper_layer_idx << "].";
  }

  // Synchronize info for expert parallel.
  if (!context_->IsExpertParallelStandalone()) {
    distributed_coordinator_->SynchronizeExpertParallelExperts();
  }

  // Get block manager config of specific layers.
  status = env->InitializeBlockManagerConfig();
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Initialize block manager config error:" + status.ToString());
  }

  // Initialize global block manager.
  BlockManagerConfig block_manager_config;
  status = env->GetBlockManagerConfig(block_manager_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Get block manager config error:" + status.ToString());
  }

  // Allocate necessary device memory after block manager initialized.
  if (!context_->IsStandalone()) {
    GetHiddenUnitBufferPool()->PreAllocateDeviceBuffer();
  }

  // Maybe need to set communication type.
  if (!context_->IsExpertParallelStandalone()) {
    GetExpertHiddenUnitBufferPool()->PreAllocateDeviceBuffer();
  }

  ProfilerConfig profiler_config;
  status = env->GetProfilerConfig(profiler_config);
  Singleton<Profiler>::GetInstance()->Init(profiler_config);

  size_t max_batch_size = (size_t)model_config.max_batch_size;
  size_t max_vocab_size = (size_t)model_config.vocab_size;

  status = LoadOperatorOptimization(model_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Load optimization config error:" + status.ToString());
  }

  // Create batch manager.
  batch_scheduler_config.max_batch_size = max_batch_size;
  batch_scheduler_config.max_vocab_size = max_vocab_size;
  KLLM_LOG_DEBUG << "Batch Scheduler Config Max Batch Size= " << max_batch_size
                 << ", Max Vocab Size= " << max_vocab_size
                 << ", max_pp_batch_num=" << batch_scheduler_config.max_pp_batch_num;

  InitHiddenUnitsMetaInfoMap(batch_scheduler_config.max_pp_batch_num);
  batch_manager_ = std::make_unique<BatchManager>(context_, batch_scheduler_config.max_pp_batch_num);

  // Register model instance.
  {
    // Update pipeline_config first, and then load model.
    model_config.max_pp_batch_num = batch_scheduler_config.max_pp_batch_num;
    std::shared_ptr<WeightInstanceInterface> weight_instance = std::make_shared<WeightInstance>(model_config, context_);
    weight_instance->Load();
    std::shared_ptr<ModelInstance> model_instance =
        std::make_shared<ModelInstance>(model_config, context_, weight_instance);
    model_instance->Load();

    // Register model instance.
    weight_instances_.push_back(weight_instance);
    model_instances_.push_back(model_instance);
    batch_manager_->RegisterModelInstance(model_instance);

    // Register to data hub.
    SetModelInstance(model_config.name, model_instance);
  }

  // Calc block number after model loaded.
  status = Singleton<Environment>::GetInstance()->CalculateBlockNumber();
  if (!status.OK()) {
    return Status(RET_RUNTIME_FAILED, "calc block number error:" + status.ToString());
  }

  if (!context_->IsStandalone()) {
    KLLM_LOG_INFO << "Start to synchronize cache block num.";
    distributed_coordinator_->SynchronizeCacheBlockNum();

    PipelineConfig pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
    KLLM_LOG_INFO << "InferenceEngine Synchronize device block num " << pipeline_config.device_block_num
                  << ", host block_num " << pipeline_config.host_block_num << ".";

    Singleton<Environment>::GetInstance()->ResetPipelineBlockNumber();
  }

  // Create cache manager.
  int attn_dp_worker_num = env->GetAttnDataParallelSize();
  CacheManagerConfig cache_manager_config;
  status = env->GetCacheManagerConfig(cache_manager_config);

  BlockAllocatorManagerConfig block_allocator_manager_config;
  for (int dp_id = 0; dp_id < attn_dp_worker_num; ++dp_id) {
    BlockAllocatorGroupConfig dp_group_config;
    dp_group_config.devices = env->GetDataParaGroupDevices(dp_id);
    dp_group_config.device_block_num = env->GetTotalDeviceBlockNum();
    dp_group_config.host_block_num = env->GetTotalHostBlockNum();
    dp_group_config.block_size = env->GetBlockSize();
    dp_group_config.convert_size = env->GetConvertSize();

    block_allocator_manager_config[dp_id] = dp_group_config;
  }

  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = std::make_shared<MemoryAllocator>();
  BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_);
  for (int dp_id = 0; dp_id < attn_dp_worker_num; ++dp_id) {
    std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group =
        block_allocator_manager.GetBlockAllocatorGroup(dp_id);
    cache_managers_.emplace_back(CacheManagerFactory::CreateCacheManager(cache_manager_config, block_allocator_group));
  }

  // Register to data hub.
  SetCacheManagers(cache_managers_);

  if (!context_->IsStandalone()) {
    distributed_coordinator_->Barrier();
  }

  // Initialize tokenzier
  Singleton<Tokenizer>::GetInstance()->InitTokenizer(model_instances_[0]->GetModelConfig().path);

  // Create batch scheduler.
  batch_scheduler_ =
      std::make_shared<BatchScheduler>(batch_scheduler_config, attn_dp_worker_num, context_->GetTensorParallelSize());
  for (int dp_id = 0; dp_id < attn_dp_worker_num; ++dp_id) {
    batch_scheduler_->SetCacheManager(cache_managers_[dp_id], dp_id);
  }

  // Create llm runtime
  llm_runtime_ = std::make_shared<LlmRuntime>(batch_scheduler_config, context_);
  llm_runtime_->SetCacheManagers(cache_managers_);
  llm_runtime_->SetMtpForward(env->IsMTPEnabled() && model_config.num_nextn_predict_layers > 0);
#ifdef ENABLE_CUDA
  // create draft generator for speculative decoding
  if (batch_scheduler_config.enable_speculative_decoding) {
    auto draft_generator = std::make_shared<TrieGenerator>();
    llm_runtime_->SetDraftGenerator(draft_generator);
  }
#endif

  batch_manager_->SetBatchScheduler(batch_scheduler_);
  batch_manager_->SetLlmRuntime(llm_runtime_);

  if (Singleton<Environment>::GetInstance()->IsReportVersion()) {
    VersionReporter::GetInstance().Init();
  }

  return Status();
}

Status InferenceEngine::HandleRequest(std::shared_ptr<Request> &req) {
  std::unordered_map<std::string, std::string> filtered_ctx = *req->req_ctx;
  filtered_ctx.erase("kv-comm-request-id");
  opentelemetry::common::KeyValueIterableView<std::unordered_map<std::string, std::string>> attributes(filtered_ctx);
  REPORT_COUNTER(forward_req_total_num, static_cast<size_t>(1), attributes);
  REPORT_METRIC(metric_input_tokens_num, req->input_tokens.size(), attributes);

  Status handle_req_status = batch_manager_->Enqueue(req);
  if (!handle_req_status.OK()) {
    REPORT_COUNTER(forward_req_error_num, static_cast<size_t>(1), attributes);
    return handle_req_status;
  }
  return Status();
}

Status InferenceEngine::HandleLoop() {
  KLLM_LOG_DEBUG << "Start handler";

  while (!terminated_) {
    std::pair<Status, std::shared_ptr<Request>> req_pair;
    request_queue_.Read(&req_pair);
    if (terminated_) {
      break;
    }

    Status status = req_pair.first;
    if (status.GetCode() == RET_REQUEST_TERMINATED) {
      break;
    }

    std::shared_ptr<Request> req = req_pair.second;
    if (req) {
      HandleRequest(req);
    }
  }

  return Status();
}

Status InferenceEngine::StartHandler() {
  handle_thread_ = std::thread(&InferenceEngine::HandleLoop, this);
  return Status();
}

Status InferenceEngine::DoWarmupRun() {
  if (std::getenv("DISABLE_WARMUP") != nullptr) {
    KLLM_LOG_DEBUG << "warmup is disabled";
    return Status();
  }
  ModelConfig model_config;
  Status status = Singleton<Environment>::GetInstance()->GetModelConfig(model_config);
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "No model config found for warmup run. " + status.ToString());
  }
  size_t max_warmup_input_length = std::min(model_config.max_token_num, (size_t)2048);
  if (std::getenv("MAX_WARMUP_INPUT_LENGTH") != nullptr) {
    max_warmup_input_length = std::stoi(std::getenv("MAX_WARMUP_INPUT_LENGTH"));
  }
  std::vector<int> warmup_input_lengths;
  const size_t warmup_input_length_step = 64;
  for (int i = warmup_input_length_step; i < max_warmup_input_length; i += warmup_input_length_step) {
    warmup_input_lengths.push_back(i);
  }

  pybind11::gil_scoped_release release;
  KLLM_LOG_INFO << "Start to do warmup run";
  for (int input_length : warmup_input_lengths) {
    auto warmup_run_input = std::make_shared<KsanaPythonInput>();
    // Prepare the warm up input.
    std::vector<int> input_tokens(input_length, 0);
    for (int i = 0; i < input_length; ++i) {
      input_tokens[i] = i % 100;  // Fill with some dummy tokens.
    }
    warmup_run_input->input_tokens = input_tokens;
    // Warm up with one context and one decoding.
    warmup_run_input->sampling_config.max_new_tokens = 2;
    warmup_run_input->sampling_config.ignore_eos = true;

    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto req = std::make_shared<Request>(warmup_run_input, req_ctx);
    req->waiter = std::make_shared<Waiter>(1);
    HandleRequest(req);

    // Wait the warm up.
    req->waiter->Wait();
    for (const auto &[output, req_logprobs, total_score] : req->output_group) {
      KLLM_CHECK_WITH_INFO(req->input_tokens.size() < output.size(),
                           "Ksana warmup run generate empty output tokens. Warmup inference run failed");
    }
  }

  KLLM_LOG_INFO << "End to do warmup run";
  pybind11::gil_scoped_acquire acquire;
  return Status();
}

#ifdef ENABLE_CUDA
Status InferenceEngine::CudaGraphCapture() {
  if (!Singleton<Environment>::GetInstance()->IsCudagraphEnabled()) {
    KLLM_LOG_INFO << "cuda graph capture is disabled";
    return Status();
  }
  pybind11::gil_scoped_release release;
  auto cuda_graph_builder = std::make_shared<CudaGraphBuilder>();
  // currently support cudagraph bs=1,2,3
  // for VRAM usage consideration (each cudagraph for specific bs takes 15~25mb VRAM)
  const int capture_batch_sizes = 3;
  size_t max_batch_size =
      cuda_graph_builder->GetMaxGraphBatchSize(model_instances_[0]->GetModelConfig().max_batch_size);
  std::vector<int> batch_size_capture_list;
  batch_size_capture_list.reserve(cuda_graph_builder->GetBatchSizeCaptureList().size());
  std::copy_if(cuda_graph_builder->GetBatchSizeCaptureList().begin(),
               cuda_graph_builder->GetBatchSizeCaptureList().end(), std::back_inserter(batch_size_capture_list),
               [&](size_t bs) { return bs <= max_batch_size; });
  std::vector<int> input_tokens(batch_size_capture_list.back(), 0);
  for (int batchsize = 1; batchsize <= capture_batch_sizes; ++batchsize) {
    KLLM_LOG_INFO << "start to capture graph: batchsize: " << batchsize;
    auto warmup_run_input = std::make_shared<KsanaPythonInput>();
    warmup_run_input->input_tokens = std::vector<int>(input_tokens.begin(), input_tokens.begin() + capture_batch_sizes);
    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto req = std::make_shared<Request>(warmup_run_input, req_ctx);
    for (int i = 0; i <= batchsize; ++i) {
      std::vector<int> output_tuple_;
      output_tuple_.emplace_back(std::get<0>(req->output_group[0])[0]);
      std::vector<std::vector<std::pair<int, float>>> req_logprobs;
      auto req_tuple = std::make_tuple(output_tuple_, req_logprobs, std::get<2>(req->output_group[0]));
      req->output_group.emplace_back(req_tuple);
    }
    req->is_cudagraph_capture_request = true;
    // we only need one context decode + one decode process
    req->sampling_config.max_new_tokens = 2;
    req->waiter = std::make_shared<Waiter>(1);
    HandleRequest(req);
    req->waiter->Wait();
    KLLM_LOG_INFO << "end to capture graph batchsize: " << batchsize;
  }
  pybind11::gil_scoped_acquire acquire;
  return Status();
}
#endif

Status InferenceEngine::LoadOperatorOptimization(ModelConfig &model_config) {
#ifdef ENABLE_CUDA
  if (std::getenv("KSANA_GEMM_ALGO_MAP_DIR") != nullptr) {
    std::string gemm_algo_map_path =
        fmt::format("{}/gemm_algo_map.yaml", std::string(std::getenv("KSANA_GEMM_ALGO_MAP_DIR")));
    if (context_->ext->GetGPUGemmAlgoHelper().LoadFromYaml(gemm_algo_map_path)) {
      KLLM_LOG_INFO << fmt::format("Load gemm algo from {} success.", gemm_algo_map_path);
    } else {
      KLLM_LOG_ERROR << fmt::format("Load gemm algo from {} failed.", gemm_algo_map_path);
    }
  }
  // NOTE(karlluo): GEMM algo file is in model dir, so we have to load gemm best algo here
  if (context_->ext->GetGPUGemmAlgoHelper().LoadFromYaml(fmt::format("{}/gemm_algo_map.yaml", model_config.path))) {
    KLLM_LOG_INFO << fmt::format("Load gemm algo from {}/gemm_algo_map.yaml success.", model_config.path);
  }
#endif
  return Status();
}

Status InferenceEngine::Start() {
  // Initialize cached block tree only for chief node.
  if (context_->IsChief()) {
    for (auto &cache_manager : cache_managers_) {
      cache_manager->InitializeCachedBlocks();
    }
  }

  // Start batch manager.
  batch_manager_->Start();

  // Start service handler.
  if (context_->IsChief()) {
    StartHandler();
  }

  if (!context_->IsStandalone()) {
    distributed_coordinator_->Barrier();
    KLLM_LOG_INFO << "All Nodes are ready, ready to server";
  }
#ifndef ENABLE_ACL
  // Start warmup run
  if (context_->IsChief()) {
    DoWarmupRun();
  }
#endif

#ifdef ENABLE_CUDA
  CudaGraphCapture();
#endif

  return Status();
}

Status InferenceEngine::Stop() {
  if (terminated_) {
    return Status();
  }

  terminated_ = true;

  request_queue_.Write({Status(RET_REQUEST_TERMINATED), nullptr});

  if (handle_thread_.joinable()) {
    handle_thread_.join();
  }

  request_queue_.Close();

  // Wait all request done.
  KLLM_LOG_INFO << "Waiting all running request.";
  Status status = batch_manager_->WaitAllDone();
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Wait all requests done error:" << status.ToString();
  }

  // Destroy the tokenizer.
  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
  if (!context_->IsStandalone()) {
    KLLM_LOG_INFO << "Destroy distributed coordinator.";
    distributed_coordinator_->DestroyCluster();
    distributed_coordinator_ = nullptr;
  }

  // Stop the batch manger.
  KLLM_LOG_INFO << "Stop batch manager.";
  batch_manager_->Stop();
  batch_manager_ = nullptr;
  llm_runtime_ = nullptr;

  // Destroy all model instances.
  KLLM_LOG_INFO << "Destroy model instances.";
  DestroyModelInstance();
  for (auto model_instance : model_instances_) {
    model_instance->Reset();
  }
  model_instances_.clear();

  // Destroy all CacheManager
  KLLM_LOG_INFO << "Destroy batch scheduler.";
  DestroyCacheManager();

  // Clear batch scheduler
  KLLM_LOG_INFO << "Destroy batch scheduler.";
  batch_scheduler_.reset();

  // Destroy schedule output and hidden unit buffer pool.
  DestroyScheduleOutputPool();
  if (!context_->IsStandalone()) {
    DestroyHiddenUnitBufferPool();
  }

  // Clear weights after model instances are cleared
  weight_instances_.clear();

  if (Singleton<Environment>::GetInstance()->IsReportVersion()) {
    KLLM_LOG_INFO << "Stop version reporter.";
    VersionReporter::GetInstance().StopReporting();
    VersionReporter::GetInstance().Destroy();
  }

  KLLM_LOG_INFO << "Destroy Profiler";
  Singleton<Profiler>::DeleteInstance();

  KLLM_LOG_INFO << "The Inference Engine has stopped.";
  return Status();
}

}  // namespace ksana_llm
