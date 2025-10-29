/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/common/common_model.h"

#include <memory>
#include <vector>

#include "fmt/core.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/expert_data_hub.h"
#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/profiler/sched_event_tracer.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

void RecordRequestSchedEventWithFContext(ForwardingContext& forwarding_context, const char* type,
                                         RequestEventPhase phase) {
  RecordRequestSchedEvents(forwarding_context.GetBatchRequestSchedInfo(), forwarding_context.GetCurrentRank(),
                           forwarding_context.GetModelInput()->attn_dp_group_id_, type, phase);
}

CommonModel::CommonModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                         std::shared_ptr<Context> context) {
  model_config_ = model_config;
  runtime_config_ = runtime_config;
  context_ = context;
  rank_ = rank;
  GetBufferManager()->SetRank(rank_);

  KLLM_LOG_DEBUG << "Working mode info, is_standalone:" << context_->IsStandalone()
                 << ", is_chief:" << context_->IsChief();
}

CommonModel::~CommonModel() {}

void CommonModel::InitRunConfig(const ModelRunConfig& model_run_config, std::shared_ptr<BaseWeight> base_weight) {
  SetDevice(rank_);

  prefix_caching_enabled_ = runtime_config_.enable_prefix_caching;
  speculative_decoding_enabled_ = runtime_config_.enable_speculative_decoding;

  size_t free_device_mem_before_init, free_device_mem_after_init, total_device_mem;
  MemGetInfo(&free_device_mem_before_init, &total_device_mem);

  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config_);

  model_run_config_ = model_run_config;
  Singleton<Environment>::GetInstance()->GetExpertParallelConfig(expert_parallel_config_);

  model_buffers_.Init(context_, rank_, model_config_, runtime_config_, GetBufferManager());

  // Initialize the buffer of forwarding contexts based on max_pp_batch_num
  forwarding_context_buffer_size_ = runtime_config_.max_pp_batch_num > 0 ? runtime_config_.max_pp_batch_num : 1;
  // Clear any existing contexts in the buffer
  {
    forwarding_context_buffer_.clear();

    // Initialize all forwarding contexts in the buffer
    forwarding_context_buffer_.reserve(forwarding_context_buffer_size_);
    for (size_t multi_batch_id = 0; multi_batch_id < forwarding_context_buffer_size_; ++multi_batch_id) {
      auto forwarding_context = std::make_unique<ForwardingContext>();
      // TODO(karlluo): each forwarding_context binding different model buffer
      forwarding_context->Init(context_, rank_, model_config_, runtime_config_, pipeline_config_,
                               model_buffers_.buffers_.get(), GetBufferManager(), multi_batch_id);
      forwarding_context_buffer_.push_back(std::move(forwarding_context));
    }

    KLLM_LOG_DEBUG << "Initialized forwarding context buffer with " << forwarding_context_buffer_size_ << " contexts";
  }

  // When using Expert-Parallel, we should init hidden_buffer_ptr in DeepEPWrapper.
  if (forwarding_context_buffer_size_ == 1 && expert_parallel_config_.global_expert_para_size > 1) {
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0,
                        forwarding_context_buffer_[0]->GetForwardingBuffers()->hidden_buffer_0);
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1,
                        forwarding_context_buffer_[0]->GetForwardingBuffers()->hidden_buffer_1);
    CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context_buffer_[0]->GetForwardingBuffers()->shared_buffer);
    std::vector<Tensor> hidden_buffer_tensors = {hidden_buffer_tensors_0[0], hidden_buffer_tensors_1[0],
                                                 reduce_buffer_tensors[0]};
    if (GetExpertParallelDeepepWrapper()) {
      GetExpertParallelDeepepWrapper()->SetHiddenBuffers(hidden_buffer_tensors, rank_);
    } else {
      KLLM_LOG_ERROR << fmt::format(
          "Failed to initialize hidden buffer tensor data_ptr with DeepEPWrapper: GetExpertParallelDeepepWrapper "
          "failed.");
    }
  }

  layer_num_on_node_ = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
  if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(model_config_.num_layer)) {
    layer_num_on_node_ += pipeline_config_.upper_nextn_layer_idx - pipeline_config_.lower_nextn_layer_idx + 1;
  }

  int head_num = model_config_.head_num;
  int size_per_head = model_config_.size_per_head;
  int hidden_units = size_per_head * head_num;

  BlockManagerConfig block_manager_config;
  STATUS_CHECK_FAILURE(Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config));

  // Initialize instances for each layer.
  layer_creation_context_.Init(base_weight, context_, rank_, pipeline_config_, model_config_, runtime_config_,
                               GetBufferManager());

  emb_lookup_layer_ = std::make_shared<EmbLookupLayer>();
  if (model_run_config_.position_encoding == PositionEncoding::LEARNED_ABSOLUTE) {
    Tensor position_weight = base_weight->GetModelWeights("model.embed_positions.weight");
    emb_lookup_layer_->Init(
        {model_run_config_.use_emb_scale, model_run_config_.emb_scale, position_weight.GetPtr<void>()}, runtime_config_,
        context_, rank_);
  } else {
    emb_lookup_layer_->Init({model_run_config_.use_emb_scale, model_run_config_.emb_scale}, runtime_config_, context_,
                            rank_);
  }

  cpu_emb_lookup_layer_ = std::make_shared<CpuEmbLookupLayer>();
  cpu_emb_lookup_layer_->Init({}, runtime_config_, context_, rank_);

  assemble_tokens_hidden_layer_ = std::make_shared<AssembleTokensHiddenLayer>();
  assemble_tokens_hidden_layer_->Init({}, runtime_config_, context_, rank_);

  cast_layer_ = std::make_shared<CastLayer>();
  cast_layer_->Init({}, runtime_config_, context_, rank_);

  input_refit_layer_ = std::make_shared<InputRefitLayer>();
  input_refit_layer_->Init({}, runtime_config_, context_, rank_);

#ifdef ENABLE_CUDA
  set_torch_stream_layer_ = std::make_shared<SetTorchStreamLayer>();
  set_torch_stream_layer_->Init({}, runtime_config_, context_, rank_);
#endif

  if (runtime_config_.embed_tokens_use_cpu) {
    DataType input_data_type = TYPE_INT32;
    size_t max_token_num = runtime_config_.max_step_token_num;
    cpu_input_tokens_tensor_ = Tensor(MemoryLocation::LOCATION_HOST, input_data_type, {max_token_num}, rank_);
    cpu_tokens_emb_tensor_ =
        Tensor(MemoryLocation::LOCATION_HOST, input_data_type, {max_token_num * hidden_units}, rank_);
  }

  KLLM_LOG_DEBUG << "Total buffer tensors memory used: " << (GetBufferTensorsMemoryUsed() >> 20) << " MB";

  ModelCreationConfig model_creation_config;
  model_creation_config.layernorm_config.layernorm_eps = model_config_.layernorm_eps;
  model_creation_config.layernorm_config.activation_function = model_config_.activation_function;

  // Flash Attention requires the input shape to match the actual token length.
  // When dealing with prefix_cache or speculative decoding, it is necessary to
  // first fill in the missing parts
  if (model_config_.type == "qwen2_vl") {
    mrotary_section_tensor_ = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {3}, rank_);
    MemcpyAsync(mrotary_section_tensor_.GetPtr<void>(), model_config_.rope_scaling_factor_config.mrope_section.data(),
                3 * sizeof(int), MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
  }
  bool reuse_prefix_config = prefix_caching_enabled_ || speculative_decoding_enabled_;
  model_creation_config.Init(model_config_, runtime_config_, model_buffers_.cos_sin_cache_tensor_,
                             model_run_config_.position_encoding, reuse_prefix_config, layer_num_on_node_,
                             mrotary_section_tensor_.GetPtr<const int>(false));

  // create matmul layer
  CreateLayers(layer_creation_context_, model_creation_config);

  if (context_->IsChief()) {
    lm_head_ = std::make_shared<Linear>("lm_head.weight", layer_creation_context_,
                                        model_creation_config.attn_config.model_config.quant_config.backend);
    if (model_run_config_.layernorm_position == LayerNormPosition::PRE_NORM) {
      lm_head_prenorm_ =
          std::make_shared<Layernorm>("model.norm.weight", model_config_.layernorm_eps, layer_creation_context_);
    }
  }

  MemGetInfo(&free_device_mem_after_init, &total_device_mem);
  KLLM_LOG_INFO << "rank=" << rank_ << ": BufferManager used "
                << GetBufferManager()->GetBufferTensorsMemoryUsed() / (1024 * 1024)
                << "MB, total_device_mem=" << total_device_mem / (1024 * 1024)
                << "MB, free_device_mem_before_init=" << free_device_mem_before_init / (1024 * 1024)
                << "MB, free_device_mem_after_init=" << free_device_mem_after_init / (1024 * 1024) << "MB";
}

float* CommonModel::GetLogitsPtr(size_t multi_batch_id) {
  SetDevice(rank_);
  ForwardingContext* forwarding_context = GetForwardingContext(multi_batch_id);
  return forwarding_context->GetModelOutput()->logits_tensor.template GetPtr<float>(false);
}

Status CommonModel::EmbedTokensUseCpu(Tensor& embedding_weight, std::vector<ForwardRequest*>& forward_reqs,
                                      ForwardingContext& forwarding_context) {
  void* input_tokens_ptr = cpu_input_tokens_tensor_.GetPtr<void>();
  memcpy(input_tokens_ptr, forwarding_context.GetModelInput()->input_ids_cpu.data(),
         forwarding_context.GetModelInput()->input_ids_cpu.size() * sizeof(int));
  cpu_input_tokens_tensor_.shape = {forwarding_context.GetModelInput()->input_ids_cpu.size()};

  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferRef(forwarding_context);
  cpu_emb_lookup_layer_->Forward({cpu_input_tokens_tensor_, cpu_tokens_emb_tensor_, embedding_weight}, residual_buffer);
  return Status();
}

Status CommonModel::EmbedTokensUseGpu(Tensor& embedding_weight, ForwardingContext& forwarding_context) {
  // Wait the computation of input_ids.
  StreamWaitEvent(context_->GetComputeStreams()[rank_], forwarding_context.GetModelInput()->input_ids_event);
  if (model_run_config_.emb_lookup_use_rotary_embedding_pos) {
    StreamWaitEvent(context_->GetComputeStreams()[rank_], forwarding_context.GetModelInput()->rotary_embedding_event);
  }

  std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferRef(forwarding_context);
  if (model_run_config_.emb_lookup_use_rotary_embedding_pos) {
    STATUS_CHECK_RETURN(emb_lookup_layer_->Forward(
        {forwarding_context.GetModelInput()->input_ids, forwarding_context.GetModelInput()->input_offset_uint64_tensor,
         forwarding_context.GetModelInput()->input_prefix_uint64_tensor, embedding_weight,
         forwarding_context.GetModelInput()->flash_input.rotary_embedding_pos},
        residual_buffer));
  } else {
    STATUS_CHECK_RETURN(emb_lookup_layer_->Forward(
        {forwarding_context.GetModelInput()->input_ids, forwarding_context.GetModelInput()->input_offset_uint64_tensor,
         forwarding_context.GetModelInput()->input_prefix_uint64_tensor, embedding_weight},
        residual_buffer));
  }

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(forwarding_context.GetModelOutput()->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetCommStreams()[rank_], forwarding_context.GetModelOutput()->compute_ready_event);
  }

  if (forwarding_context.GetModelCommunicator()) {
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
    forwarding_context.GetModelCommunicator()->AllGather({residual_buffer[0], hidden_buffer_tensors_1[0]},
                                                         residual_buffer);
  }
  return Status();
}

bool CommonModel::UpdateResponse(std::vector<ForwardRequest*>& forward_reqs, Tensor& output, const std::string& stage) {
  bool ret = true;
  int req_offset = 0;
  for (ForwardRequest& req : forward_reqs) {
    int output_token_num = req.forwarding_tokens->size();
    if (!req.request_target) {
      ret = false;
      continue;
    }
    auto it = req.request_target->find(stage);
    if (it == req.request_target->end()) {
      ret = false;
      continue;
    } else if (it->second.token_reduce_mode != TokenReduceMode::GATHER_ALL) {
      // GATHER_TOKEN_ID for "logits"
      ret = false;
      continue;
    }
    // Determine whether to exit early
    ret &= req.request_target->size() == req.response->size();
    if (rank_ != 0) continue;
    int output_len = 0;
    std::vector<std::pair<int, int>> slice_pos = it->second.slice_pos;
    // If specific token IDs are provided, add their positions to slice_pos.
    if (it->second.token_id.size() != 0) {
      std::set<int> token_id_set(it->second.token_id.begin(), it->second.token_id.end());
      for (int i = 0; i < output_token_num; i++) {
        if (token_id_set.count(req.forwarding_tokens->at(i)) > 0) {
          slice_pos.push_back({i, i});
        }
      }
    }
    // Calculate the total output length based on slice positions.
    for (auto [l, r] : slice_pos) {
      output_len += r - l + 1;
    }
    // Calculate the size of each chunk based on the output tensor's data type and shape.
    size_t chunk_size = GetTypeSize(output.dtype) * output.shape[1];
    // Update the response tensor with the sliced data.
    PythonTensor& ret_tensor = (*req.response)[stage];
    ret_tensor.shape = {static_cast<size_t>(output_len), output.shape[1]};
    ret_tensor.dtype = GetTypeString(output.dtype);
    ret_tensor.data.resize(output_len * chunk_size);
    if (stage == "logits") {
      // Update slice_pos as {[0, output_len - 1]} to skip cutting.
      slice_pos = {{0, output_len - 1}};
      output_token_num = output_len;
    }
    req_offset += output_token_num;
    output_len = 0;
    // Copy data from the output tensor to the output_data buffer based on slice positions.
    for (auto [l, r] : slice_pos) {
      MemcpyAsync(ret_tensor.data.data() + output_len * chunk_size,
                  output.GetPtr<void>() + (req_offset - output_token_num + l) * chunk_size, (r - l + 1) * chunk_size,
                  MEMCPY_DEVICE_TO_HOST, context_->GetComputeStreams()[rank_]);
      output_len += r - l + 1;
    }
    StreamSynchronize(context_->GetComputeStreams()[rank_]);
  }
  return ret;
}

std::vector<Tensor>& CommonModel::GetHiddenUnitBufferRef(ForwardingContext& forwarding_context) {
  if (context_->IsStandalone()) {
    return model_buffers_.local_residual_buffer_tensors_;
  }

#ifdef ENABLE_ACL
  if (forwarding_context.GetModelInput()->infer_stage == InferStage::STAGE_CONTEXT) {
    HiddenUnitDeviceBuffer* device_buffer = GetCurrentHiddenUnitBuffer(forwarding_context.GetMultiBatchId());
    if (distributed_device_buffer_prefill_.empty()) {
      distributed_device_buffer_prefill_.push_back(device_buffer->prefill_tensors[rank_]);
    } else {
      // keep shape and dtype, just assign memory reference
      auto shape = distributed_device_buffer_prefill_[0].shape;
      auto dtype = distributed_device_buffer_prefill_[0].dtype;
      distributed_device_buffer_prefill_[0] = device_buffer->prefill_tensors[rank_];
      distributed_device_buffer_prefill_[0].shape = shape;
      distributed_device_buffer_prefill_[0].dtype = dtype;
    }

    return distributed_device_buffer_prefill_;
  } else {
#endif
    HiddenUnitDeviceBuffer* device_buffer = GetCurrentHiddenUnitBuffer(forwarding_context.GetMultiBatchId());
    if (distributed_device_buffer_.empty()) {
      distributed_device_buffer_.push_back(device_buffer->tensors[rank_]);
    } else {
      // keep shape and dtype, just assign memory reference
      auto shape = distributed_device_buffer_[0].shape;
      auto dtype = distributed_device_buffer_[0].dtype;
      distributed_device_buffer_[0] = device_buffer->tensors[rank_];
      distributed_device_buffer_[0].shape = shape;
      distributed_device_buffer_[0].dtype = dtype;
    }

    return distributed_device_buffer_;
#ifdef ENABLE_ACL
  }
#endif
}

std::vector<Tensor>& CommonModel::GetHiddenUnitBuffer(ForwardingContext& forwarding_context, bool do_recv) {
  if (do_recv) {
    RecordRequestSchedEventWithFContext(forwarding_context, "RecvHiddenUnitBuffer", RequestEventPhase::Begin);
    std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferRef(forwarding_context);
    bool is_prefill = forwarding_context.GetModelInput()->infer_stage == InferStage::STAGE_CONTEXT;
    CopyFromHiddenUnitBuffer(residual_buffer[0], GetCurrentHiddenUnitBuffer(forwarding_context.GetMultiBatchId()),
                             forwarding_context.GetCurrentRank(), is_prefill);
    RecordRequestSchedEventWithFContext(forwarding_context, "RecvHiddenUnitBuffer", RequestEventPhase::End);

    if (forwarding_context.IsForwardingLayers()) {
      RecordRequestSchedEventWithFContext(forwarding_context, "ForwardingLayers", RequestEventPhase::Begin);
    }
    return residual_buffer;
  } else {
    if (forwarding_context.IsForwardingLayers()) {
      RecordRequestSchedEventWithFContext(forwarding_context, "ForwardingLayers", RequestEventPhase::Begin);
    }
    return GetHiddenUnitBufferRef(forwarding_context);
  }
}

Status CommonModel::AllocResources(size_t multi_batch_id) {
  if (context_->IsChief()) {
    KLLM_CHECK_WITH_INFO(multi_batch_id < forwarding_context_buffer_size_,
                         FormatStr("multi_batch_id: %d should be smaller than max_pp: %d.", multi_batch_id,
                                   forwarding_context_buffer_size_));
  }

  size_t id = context_->IsChief() ? multi_batch_id : 0;
  if (forwarding_context_buffer_.at(id)->GetMultiBatchId() == multi_batch_id) {
    KLLM_LOG_DEBUG << "ForwardingContext for multi_batch_id=" << multi_batch_id << " already allocated";
    return Status();
  }
  forwarding_context_buffer_.at(id)->SetMultiBatchId(multi_batch_id);
  return Status();
}

Status CommonModel::FreeResources(size_t multi_batch_id) { return Status(); }

void CommonModel::SetHiddenUnitBuffer(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  if (forwarding_context.IsForwardingLayers()) {
    RecordRequestSchedEventWithFContext(forwarding_context, "ForwardingLayers", RequestEventPhase::End);
  }
  // Copy to hidden_unit_buffer if not standalone.
  if (!forwarding_context.GetContext()->IsStandalone()) {
    RecordRequestSchedEventWithFContext(forwarding_context, "StreamSynchronize", RequestEventPhase::Begin);
    bool is_prefill = forwarding_context.GetModelInput()->infer_stage == InferStage::STAGE_CONTEXT;

    auto working_stream = forwarding_context.GetContext()->GetComputeStreams()[forwarding_context.GetCurrentRank()];
    StreamSynchronize(working_stream);
    RecordRequestSchedEventWithFContext(forwarding_context, "StreamSynchronize", RequestEventPhase::End);
    CopyToHiddenUnitBuffer(GetCurrentHiddenUnitBuffer(forwarding_context.GetMultiBatchId()), residual_buffer[0],
                           forwarding_context.GetCurrentRank(), is_prefill, working_stream);
  }
}

ForwardingContext* CommonModel::GetForwardingContext(size_t multi_batch_id) {
  if (context_->IsChief()) {
    KLLM_CHECK_WITH_INFO(multi_batch_id < forwarding_context_buffer_size_,
                         FormatStr("multi_batch_id: %d should be smaller than max_pp: %d.", multi_batch_id,
                                   forwarding_context_buffer_size_));
  }
  size_t id = context_->IsChief() ? multi_batch_id : 0;
  return forwarding_context_buffer_[id].get();
}

Status CommonModel::Forward(size_t multi_batch_id, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                            std::vector<ForwardRequest*>& forward_reqs, bool epilogue, const RunMode run_mode) {
  // Get the forwarding context for this multi_batch_id
  time_t start_time_ms = ProfileTimer::GetCurrentTimeInMs();
  ForwardingContext* forwarding_context = GetForwardingContext(multi_batch_id);
  KLLM_LOG_DEBUG << "start forward multi_batch_id=" << forwarding_context->GetMultiBatchId() << ", rank=" << rank_;

  {}

  PROFILE_EVENT_SCOPE(
      CommonModel_Forward,
      fmt::format("CommonModel_Forward_{}_{}_rank{}", multi_batch_id, epilogue, forwarding_context->GetCurrentRank()),
      forwarding_context->GetCurrentRank());

  forwarding_context->GetBatchRequestSchedInfo() =
      BuildBatchRequestSchedInfoFromForwardingReqs(forward_reqs, multi_batch_id);

  forwarding_context->UpdateBeforeForward(forward_reqs, run_mode);
  forwarding_context->AcquireBuffers();
  model_buffers_.AcquireBuffers(forwarding_context->GetModelInput());

  // Set shape and type of hidden unit.
  SetHiddenUnitMeta(multi_batch_id,
                    {forwarding_context->GetModelInput()->input_ids.shape[0], model_config_.hidden_units},
                    model_config_.weight_data_type);
  if (context_->IsChief()) {
    RecordRequestSchedEventWithFContext(*forwarding_context, "PrepareForwarding", RequestEventPhase::End);
  }
  if (!epilogue || run_mode == RunMode::kNextN) {
    if (context_->IsChief()) {
      RecordRequestSchedEventWithFContext(*forwarding_context, "EmbLookup", RequestEventPhase::Begin);
      LookupEmbedding(*forwarding_context, base_weight, forward_reqs);
      RecordRequestSchedEventWithFContext(*forwarding_context, "EmbLookup", RequestEventPhase::End);
    }
    forwarding_context->SetIsForwardingLayers(true);
    if (!runtime_config_.is_profile_mode) {
      LayerForward(*forwarding_context, run_mode);
    } else {
      // Used for layer forwarding performance profile
      for (size_t idx = 0; idx < g_profile_layer_forwarding_round; idx++) {
        LayerForward(*forwarding_context, run_mode);
      }
    }
    forwarding_context->SetIsForwardingLayers(false);
  }
  // Invode lm head only in standalone mode.
  if (context_->IsStandalone() || epilogue) {
    LmHead(*forwarding_context, base_weight, forward_reqs, run_mode);
  }

  model_buffers_.ReleaseBuffers();
  forwarding_context->ReleaseBuffers();

  time_t end_time_ms = ProfileTimer::GetCurrentTimeInMs();
  KLLM_LOG_DEBUG << "CommonModel Forward multi_batch_id=" << multi_batch_id << ", epilogue=" << epilogue
                 << ", time cost=" << end_time_ms - start_time_ms << "ms";
  return Status();
}

Status CommonModel::LookupEmbedding(ForwardingContext& forwarding_context,
                                    std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                                    std::vector<ForwardRequest*>& forward_reqs, const RunMode run_mode) {
  KLLM_LOG_DEBUG << "start lookup embedding multi_batch_id=" << forwarding_context.GetMultiBatchId()
                 << ", rank=" << rank_ << "";
  PROFILE_EVENT_SCOPE(CommonModel_LookupEmbedding, "CommonModel_LookupEmbedding", forwarding_context.GetCurrentRank());
  // CPU embedding lookup
  // The output is stored in `residual_buffer` for residual connection in common
  // decoder.
  Tensor embedding_weight = base_weight->GetModelWeights("model.embed_tokens.weight");
  if (embedding_weight.location == MemoryLocation::LOCATION_HOST) {
    EmbedTokensUseCpu(embedding_weight, forward_reqs, forwarding_context);
  }

  if (forwarding_context.GetModelInput()->is_cudagraph_capture_request) {
    StreamWaitEvent(context_->GetComputeStreams()[rank_], forwarding_context.GetModelInput()->kvcache_offset_event);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], forwarding_context.GetModelInput()->rotary_embedding_event);
  }

  // GPU embedding lookup
  // The output is stored in `residual_buffer` for residual connection in common
  // decoder.
  if (embedding_weight.location == MemoryLocation::LOCATION_DEVICE) {
    EmbedTokensUseGpu(embedding_weight, forwarding_context);
  }

  // refit input needs to be processed only in the multi-token forwarding.
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  if (is_multi_token_forward && run_mode == RunMode::kMain) {
    std::vector<Tensor>& residual_buffer = GetHiddenUnitBufferRef(forwarding_context);
    input_refit_layer_->Forward({forwarding_context.GetModelInput()->cpu_input_refit_tensor.pos_pair_tensor,
                                 forwarding_context.GetModelInput()->cpu_input_refit_tensor.emb_fp32_ptr_tensor},
                                residual_buffer);
  }
  return Status();
}

Status CommonModel::LmHead(ForwardingContext& forwarding_context, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                           std::vector<ForwardRequest>& forward_reqs, RunMode run_mode) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  std::vector<Tensor>& residual_buffer = GetHiddenUnitBuffer(
      forwarding_context, !context_->IsStandalone() && context_->IsChief() && run_mode == RunMode::kMain);
  RecordRequestSchedEventWithFContext(forwarding_context, "LmHead", RequestEventPhase::Begin);
  // save hidden result if enable MTP model
  if (runtime_config_.mtp_step_num > 0 && context_->IsChief()) {
    auto& mtp_hidden_tensor = forwarding_context.GetForwardingBuffers()->mtp_hidden_buffer_tensors[0];
    mtp_hidden_tensor.shape = residual_buffer[0].shape;
    mtp_hidden_tensor.dtype = residual_buffer[0].dtype;
    MemcpyAsync(mtp_hidden_tensor.template GetPtr<void>(), residual_buffer[0].template GetPtr<void>(),
                residual_buffer[0].GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
  }

  if (is_multi_token_forward && run_mode == RunMode::kMain) {
    if (UpdateResponse(forward_reqs, residual_buffer[0], "transformer")) {
      StreamSynchronize(context_->GetComputeStreams()[rank_]);
      input_refit_layer_->Clear();
      return Status();
    }
  }

  // final norm
  // Only pre norm model performs final norm.
  // Both input and output are in `residual_buffer`.
  if (lm_head_prenorm_) {
    lm_head_prenorm_->Forward(residual_buffer, residual_buffer);
  }

  if (is_multi_token_forward && run_mode == RunMode::kMain) {
    if (UpdateResponse(forward_reqs, residual_buffer[0], "layernorm")) {
      StreamSynchronize(context_->GetComputeStreams()[rank_]);
      input_refit_layer_->Clear();
      return Status();
    }
  }

  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
// assemble last token
// The input is stored in `residual_buffer`.
#ifdef ENABLE_CUDA
  STATUS_CHECK_RETURN(assemble_tokens_hidden_layer_->Forward(
      {residual_buffer[0], forwarding_context.GetModelInput()->logits_idx_uint64_tensor}, hidden_buffer_tensors_0));
#elif defined(ENABLE_ACL)
  STATUS_CHECK_RETURN(assemble_tokens_hidden_layer_->Forward(
      {residual_buffer[0], forwarding_context.GetModelInput()->last_token_index_tensor,
       forwarding_context.GetModelInput()->input_prefix_uint64_tensor},
      hidden_buffer_tensors_0));
#endif

  // lm_head
  PROFILE_EVENT_SCOPE(CommonModel_LmHead_, fmt::format("CommonModel_LmHead_{}", forwarding_context.GetMultiBatchId()),
                      forwarding_context.GetCurrentRank());

  if (forwarding_context.GetModelCommunicator() && runtime_config_.enable_full_shared_expert) {
    forwarding_context.GetModelCommunicator()->ReduceSum(hidden_buffer_tensors_0, hidden_buffer_tensors_1,
                                                         is_multi_token_forward, /*use_custom*/ false);
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  }

  STATUS_CHECK_RETURN(lm_head_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1));
  std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(forwarding_context.GetModelOutput()->compute_ready_event, context_->GetComputeStreams()[rank_]);
    StreamWaitEvent(context_->GetCommStreams()[rank_], forwarding_context.GetModelOutput()->compute_ready_event);
  }

  if (forwarding_context.GetModelCommunicator()) {
    forwarding_context.GetModelCommunicator()->AllGather({hidden_buffer_tensors_0[0], hidden_buffer_tensors_1[0]},
                                                         hidden_buffer_tensors_0);
  }

  if (is_multi_token_forward && run_mode == RunMode::kMain) {
    if (UpdateResponse(forward_reqs, hidden_buffer_tensors_0[0], "logits")) {
      StreamSynchronize(context_->GetComputeStreams()[rank_]);
      return Status();
    }
  }

  PROFILE_EVENT_SCOPE(CommonModel_Cast_, fmt::format("CommonModel_Cast_{}", forwarding_context.GetMultiBatchId()),
                      forwarding_context.GetCurrentRank());
  forwarding_context.UpdateAfterForward(forward_reqs);
  std::vector<Tensor> logits_buffer{forwarding_context.GetModelOutput()->logits_tensor};
  STATUS_CHECK_RETURN(cast_layer_->Forward(
      {hidden_buffer_tensors_0[0], forwarding_context.GetAttentionForwardContext().forward_shape}, logits_buffer));

  StreamSynchronize(context_->GetComputeStreams()[rank_]);
  forwarding_context.GetModelInput()->VerifyChecksumAfterForward(forward_reqs);
  RecordRequestSchedEventWithFContext(forwarding_context, "LmHead", RequestEventPhase::End);
  input_refit_layer_->Clear();
  return Status();
}

}  // namespace ksana_llm
