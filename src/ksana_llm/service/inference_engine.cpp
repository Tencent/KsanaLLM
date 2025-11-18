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
#include "ksana_llm/runtime/draft_generator/draft_generator_controller.h"
#include "ksana_llm/runtime/draft_generator/ptp_generator.h"
#include "ksana_llm/runtime/draft_generator/trie_generator.h"
#include "ksana_llm/runtime/generation_controller.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/layer_progress_tracker.h"
#include "ksana_llm/runtime/ptp_generation_controller.h"
#include "ksana_llm/runtime/structured_generation/xgrammar/xgrammar_structured_generator_creator.h"
#include "ksana_llm/runtime/weight_instance.h"
#include "ksana_llm/transfer/transfer_engine.h"
#include "ksana_llm/utils/dynamic_memory_counter.h"
#include "ksana_llm/utils/dynamic_memory_pool.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/tokenizer.h"
#include "ksana_llm/utils/waiter.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/runtime/cuda_graph_runner.h"
#endif

namespace ksana_llm {

InferenceEngine::InferenceEngine(Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : request_queue_(request_queue) {
#ifdef ENABLE_ACL
  // Disable dynamic memory buffer for ascend device, because atb attention maybe crash in this.
  DeviceMemoryPool::Disable();
#else
  // TODO(yancyliu): Enable memory pool later.
  DeviceMemoryPool::Disable();
#endif
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
  RuntimeConfig runtime_config;
  env->GetRuntimeConfig(runtime_config);
  size_t tp_num = runtime_config.parallel_basic_config.tensor_parallel_size;
  size_t attn_data_parallel_size = runtime_config.parallel_basic_config.attn_data_parallel_size;

  // TODO(TJ): maybe cloud move IsChief and IsStandalone to env.
  PipelineConfig pipeline_config;
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
  bool is_chief = pipeline_config.world_size == 1 || pipeline_config.node_rank == 0;
  if (!is_chief) {
    batch_scheduler_config.max_pp_batch_num = 1;
  }
  // Environment is must be initialized befroe context.
  KLLM_LOG_INFO << "Get tensor parallel: " << tp_num << " attention data parallel: " << attn_data_parallel_size
                << " max_pp_batch_num: " << batch_scheduler_config.max_pp_batch_num;
  context_.reset(new Context(tp_num, attn_data_parallel_size, batch_scheduler_config.max_pp_batch_num));

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
      tp_num, model_config.num_layer + model_config.num_nextn_predict_layers);

  Singleton<LayerProgressTracker>::GetInstance()->RegisterCallback(
      [&](int device_id, int layer_index) { TransferEngine::GetInstance()->Send(device_id, layer_index); });

  // Only for distributed mode.
  if (!context_->IsStandalone() || !context_->IsExpertParallelStandalone()) {
    if (!context_->IsStandalone()) {
      InitializeHiddenUnitBufferPool();  // Does HiddenUnitBufferPool must has one comm type? (No)
      GetHiddenUnitBufferPool()->SetCommType(DistributedCommunicationType::ALLTOALL);
    }

    distributed_coordinator_ = std::make_shared<DistributedCoordinator>(
        context_, GetPacketObject, GetScheduleOutputPool(), GetHiddenUnitBufferPool(), env);

    KLLM_LOG_INFO << "Initialize distributed coordinator succeed.";
    distributed_coordinator_->InitializeCluster();
  }

  // TODO(zezhao): DeepEP 相关逻辑后续应迁移至 nvidia 目录下
  InitializeExpertParallelDeepepWrapper(model_config, runtime_config, context_);
  // TODO(zezhao): 当前仅有开启 enable_full_shared_expert 时才会使用 DeepEP，后续会将该变量更名并调整在 EP 中的耦合关系.
  if (!context_->IsExpertParallelStandalone() && runtime_config.enable_full_shared_expert) {
    distributed_coordinator_->SynchronizeNvshmemUniqueId();
  }

  // Set model layers for standalone mode, assume only one model now.
  KLLM_LOG_INFO << "InferenceEngine PiplineParallel IsStandalone:" << context_->IsStandalone();
  if (context_->IsStandalone()) {
    PipelineConfig pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
    pipeline_config.lower_layer_idx = 0;
    pipeline_config.upper_layer_idx = model_config.num_layer - 1;
    if (model_config.num_nextn_predict_layers > 0 && runtime_config.mtp_step_num > 0) {
      if (model_config.num_nextn_predict_layers <= runtime_config.mtp_step_num) {
        pipeline_config.lower_nextn_layer_idx = model_config.num_layer;
        pipeline_config.upper_nextn_layer_idx = model_config.num_layer + model_config.num_nextn_predict_layers - 1;
      } else if (model_config.num_nextn_predict_layers == 1) {
        // repeat MTP layer only support exist 1 MTP layer weight
        pipeline_config.upper_nextn_layer_idx += runtime_config.mtp_step_num - 1;
        KLLM_LOG_INFO << "repeat MTP layer " << runtime_config.mtp_step_num - 1 << " times";
      } else {
        KLLM_LOG_FATAL << fmt::format(
            "mtp_step_num({}) must <= num_nextn_predict_layers({}) in model config while num_nextn_predict_layers != 1",
            runtime_config.mtp_step_num, model_config.num_nextn_predict_layers);
      }
    }

    Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);
    KLLM_LOG_INFO << "InferenceEngine Set layer range:[" << pipeline_config.lower_layer_idx << ", "
                  << pipeline_config.upper_layer_idx << "], nextn layer range: ["
                  << pipeline_config.lower_nextn_layer_idx << ", " << pipeline_config.upper_nextn_layer_idx << "].";
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

  // Get block manager config of specific layers.
  status = env->InitializeBlockManagerConfig();
  if (!status.OK()) {
    return Status(RET_INVALID_ARGUMENT, "Initialize block manager config error:" + status.ToString());
  }

  // members in RuntimeConfig are set in InitializeBlockManagerConfig, get runtime_config again.
  env->GetRuntimeConfig(runtime_config);

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

  ProfilerConfig profiler_config;
  status = env->GetProfilerConfig(profiler_config);
  Singleton<Profiler>::GetInstance()->InitMetrics(profiler_config);

  size_t max_batch_size = runtime_config.max_batch_size;
  size_t max_vocab_size = model_config.vocab_size;

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
  if (context_->IsChief()) {
    multi_batch_controller_ = std::make_shared<MultiBatchController>(batch_scheduler_config.max_pp_batch_num);
  }
  InitHiddenUnitsMetaInfoMap(batch_scheduler_config.max_pp_batch_num);
  batch_manager_ = std::make_unique<BatchManager>(runtime_config, context_);

  // Register model instance.
  {
    // Update pipeline_config first, and then load model.
    runtime_config.max_pp_batch_num = batch_scheduler_config.max_pp_batch_num;  // TODO(robertyuan): to be removed
    std::shared_ptr<WeightInstanceInterface> weight_instance =
        std::make_shared<WeightInstance>(model_config, runtime_config, context_);
    weight_instance->Load();
    std::shared_ptr<ModelInstance> model_instance =
        std::make_shared<ModelInstance>(model_config, runtime_config, context_, weight_instance);
    model_instance->Load();

    // Register model instance.
    weight_instances_.push_back(weight_instance);
    model_instances_.push_back(model_instance);
    batch_manager_->RegisterModelInstance(model_instance);

    // Register to data hub.
    SetModelInstance(model_config.name, model_instance);
  }

  // Initialize memory pool, put it after model_instance, because the DEEPGEMM need very large memory.
  if (DeviceMemoryPool::IsEnabled()) {
    status = InitializeMemoryPool(env);
    if (!status.OK()) {
      return Status(RET_INVALID_ARGUMENT, "Initialize tensor memory pool error:" + status.ToString());
    }
    KLLM_LOG_DEBUG << "Initialize tensor memory pool finished.";
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
  CacheManagerConfig cache_manager_config;
  status = env->GetCacheManagerConfig(cache_manager_config);

  BlockAllocatorManagerConfig block_allocator_manager_config;
  for (size_t dp_id = 0; dp_id < attn_data_parallel_size; ++dp_id) {
    BlockAllocatorGroupConfig dp_group_config;
    dp_group_config.devices = env->GetDataParaGroupDevices(dp_id);
    dp_group_config.device_block_num = env->GetTotalDeviceBlockNum();
    dp_group_config.host_block_num = env->GetTotalHostBlockNum();
    dp_group_config.block_size = runtime_config.attn_backend_config.block_size;

    block_allocator_manager_config[dp_id] = dp_group_config;
  }

  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = std::make_shared<MemoryAllocator>();
  BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_);
  for (size_t dp_id = 0; dp_id < attn_data_parallel_size; ++dp_id) {
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

  std::shared_ptr<StructuredGeneratorFactory> structured_generator_factory;
  // Initialize structured generator factory for structured output
  if (batch_scheduler_config.enable_xgrammar) {
    std::vector<std::string> vocab;
    int vocab_size = static_cast<int>(model_config.vocab_size);
    std::vector<int> stop_token_ids;

    auto tokenizer = Singleton<Tokenizer>::GetInstance();
    status = tokenizer->GetVocabInfo(vocab, vocab_size, stop_token_ids);
    if (status.OK()) {
      try {
        structured_generator_factory = std::make_shared<StructuredGeneratorFactory>();
        structured_generator_factory->RegisterCreator(
            StructuredConstraintType::JSON,
            std::make_unique<GrammarGeneratorCreator>(vocab, vocab_size, stop_token_ids));
        KLLM_LOG_INFO << "Structured generator factory initialized with final "
                         "vocab_size: "
                      << vocab_size;

        // Get reasoning config from Environment and set it to factory
        ReasoningConfig reasoning_config;
        Status reasoning_status = env->GetReasoningConfig(reasoning_config);
        if (reasoning_status.OK() && reasoning_config.IsEnabled()) {
          structured_generator_factory->SetReasoningConfig(reasoning_config);
          KLLM_LOG_INFO << "Reasoning mode enabled with think_end_token_id: " << reasoning_config.think_end_token_id;
        }
      } catch (const std::exception &e) {
        return Status(RET_RUNTIME_FAILED,
                      "Structured generator factory initialization failed: " + std::string(e.what()));
      }
    } else {
      KLLM_LOG_WARNING << "Failed to get vocab info: " << status.GetMessage() << ". Structured output disabled.";
      structured_generator_factory = nullptr;
    }
  }

  // Create generation controller
  if (batch_scheduler_config.ptp_step_num > 0) {
    generation_controller_ = std::make_shared<PTPGenerationController>(structured_generator_factory);
  } else {
    generation_controller_ = std::make_shared<GenerationController>(structured_generator_factory);
  }

  ExpertParallelConfig ep_config;
  Singleton<Environment>::GetInstance()->GetExpertParallelConfig(ep_config);
  // Create batch scheduler.
  batch_scheduler_ = std::make_shared<BatchScheduler>(batch_scheduler_config, runtime_config,
                                                      ep_config.expert_world_size > 1, model_instances_);
  for (size_t dp_id = 0; dp_id < attn_data_parallel_size; ++dp_id) {
    batch_scheduler_->SetCacheManager(cache_managers_[dp_id], dp_id);
  }

  // Create llm runtime
  llm_runtime_ = std::make_shared<LlmRuntime>(batch_scheduler_config, runtime_config, context_);
  llm_runtime_->SetCacheManagers(cache_managers_);
  llm_runtime_->SetMultiBatchController(multi_batch_controller_);
  llm_runtime_->SetGenerationController(generation_controller_);

#ifdef ENABLE_CUDA
  // create draft generator for speculative decoding
  auto draft_generator_controller = std::make_shared<DraftGeneratorController>();
  if (batch_scheduler_config.enable_speculative_decoding) {
    draft_generator_controller->AppendDraftGenerator(std::make_shared<TrieGenerator>());
  }
  if (batch_scheduler_config.ptp_step_num > 0) {
    draft_generator_controller->AppendDraftGenerator(
        std::make_shared<PtpGenerator>(batch_scheduler_config.ptp_step_num, batch_scheduler_config.ptp_token_id));
  }
  llm_runtime_->SetDraftGeneratorController(draft_generator_controller);
#endif

  batch_manager_->SetBatchScheduler(batch_scheduler_);
  batch_manager_->SetMultiBatchController(multi_batch_controller_);
  batch_manager_->SetGenerationController(generation_controller_);
  batch_manager_->SetLlmRuntime(llm_runtime_);

  if (Singleton<Environment>::GetInstance()->IsReportVersion()) {
    VersionReporter::GetInstance().Init();
  }

  return Status();
}

Status InferenceEngine::InitializeMemoryPool(std::shared_ptr<Environment> env) {
  size_t free_bytes, total_bytes;
  Status status = GetDeviceMemoryInfo(MemoryDevice::MEMORY_DEVICE, &free_bytes, &total_bytes);
  if (!status.OK()) {
    return status;
  }

  BlockManagerConfig block_manager_config;
  env->GetBlockManagerConfig(block_manager_config);

  size_t dev_reserved_bytes = free_bytes * block_manager_config.reserved_device_memory_ratio;
  size_t dev_memory_pool_bytes = free_bytes - dev_reserved_bytes;
  KLLM_LOG_INFO << "InitializeMemoryPool, dev_reserved_mb:" << (dev_reserved_bytes / 1024 / 1024)
                << ", dev_memory_pool_mb:" << (dev_memory_pool_bytes / 1024 / 1024) << std::endl;

  constexpr size_t DEV_BLOCK_SIZE = 4096;

  RuntimeConfig runtime_config;
  status = env->GetRuntimeConfig(runtime_config);
  if (!status.OK()) {
    return status;
  }

  // Only for device memory now.
  size_t tp_size = runtime_config.parallel_basic_config.tensor_parallel_size;
  for (size_t rank = 0; rank < tp_size; ++rank) {
    SetDevice(rank);

    void *dev_ptr;
    Malloc(&dev_ptr, dev_memory_pool_bytes);
    DeviceMemoryPool::SetMemoryPool(rank, dev_ptr, dev_memory_pool_bytes, DEV_BLOCK_SIZE);
  }

  return Status();
}

Status InferenceEngine::HandleRequest(std::shared_ptr<Request> &req) {
  REPORT_COUNTER("forward_req_total_num", static_cast<size_t>(1));

  Status handle_req_status = batch_manager_->Enqueue(req);
  if (!handle_req_status.OK()) {
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
  RuntimeConfig runtime_config;
  Singleton<Environment>::GetInstance()->GetRuntimeConfig(runtime_config);
  size_t max_warmup_input_length = std::min(runtime_config.max_seq_len, 2048ul);
  if (std::getenv("MAX_WARMUP_INPUT_LENGTH") != nullptr) {
    max_warmup_input_length = std::stoi(std::getenv("MAX_WARMUP_INPUT_LENGTH"));
  }
  std::vector<int> warmup_input_lengths;
  const size_t warmup_input_length_step = 64;
  for (size_t i = warmup_input_length_step; i < max_warmup_input_length; i += warmup_input_length_step) {
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

    ConnectorConfig connector_config;
    Singleton<Environment>::GetInstance()->GetConnectorConfigs(connector_config);
    // NOTE(winminkong): To prevent PD disaggregate waiting in a loop for the configuration to be transmitted.
    if (connector_config.group_role == GroupRole::PREFILL) {
      req->kv_comm_request_id = -1;
    }
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

  // Wait until every expert-parallel rank is ready before allowing the mock request to proceed.
  if (!context_->IsExpertParallelStandalone()) {
    distributed_coordinator_->ExpertParallelBarrier();
  }

  // Start batch manager.
  batch_manager_->Start();

  // Start service handler.
  if (context_->IsChief()) {
    StartHandler();
  }

  if (!context_->IsStandalone()) {
    distributed_coordinator_->Barrier();
    distributed_coordinator_->Frozen();
    KLLM_LOG_INFO << "All Nodes are ready, ready to server";
  }
#ifndef ENABLE_ACL
  // Start warmup run
  if (context_->IsChief()) {
    DoWarmupRun();
  }
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
