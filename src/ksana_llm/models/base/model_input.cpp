/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_input.h"

#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#  include "csrc/kernels/nvidia/flash_mla/kernels/params.h"
#  include "csrc/utils/nvidia/cuda_utils.h"
#  include "ksana_llm/kernels/nvidia/deepseek_deepgemm_wrapper.h"
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif
#include "ksana_llm/cache_manager/block_allocator/block_allocator_interface.h"
#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/dynamic_memory_counter.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

ModelInput::ModelInput(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                       std::shared_ptr<Context> context)
    : model_config_(model_config), runtime_config_(runtime_config), rank_(rank), context_(context) {
  auto env = Singleton<Environment>::GetInstance();
  env->GetConnectorConfigs(connector_config_);
  PipelineConfig pipeline_config;
  env->GetPipelineConfig(pipeline_config);
  enable_blocked_multi_token_forwarding_kv_ =
      runtime_config.attn_backend_config.enable_blocked_multi_token_forwarding_kv;

  block_size_ = runtime_config_.attn_backend_config.block_size;
  const size_t max_batch_size = runtime_config_.max_batch_size;
  const size_t max_token_num = runtime_config.max_step_token_num;  // max step token num
  layer_num_on_node_ = pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1;
  if (pipeline_config.lower_nextn_layer_idx >= static_cast<int>(model_config_.num_layer)) {
    layer_num_on_node_ += pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1;
    KLLM_LOG_INFO << "ModelInput add next n, now layer: " << layer_num_on_node_;
  }

  attn_dp_group_id_ = rank_ / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  attn_dp_rank_id_ = rank_ % runtime_config.parallel_basic_config.attn_tensor_parallel_size;
  attn_dp_group_size_ = runtime_config_.parallel_basic_config.attn_data_parallel_size;
  attn_dp_group_offsets_.assign(attn_dp_group_size_, 0);
  KLLM_LOG_INFO << "rank:" << rank_ << ", attn_dp_group_id_: " << attn_dp_group_id_
                << ", attn_dp_rank_id_: " << attn_dp_rank_id_ << ", attn_dp_group_size_: " << attn_dp_group_size_;

  const size_t max_seq_len = runtime_config.max_seq_len;  // max seq len for one request
  size_t max_block_num = (max_seq_len * max_batch_size + runtime_config.attn_backend_config.block_token_num - 1) /
                         runtime_config.attn_backend_config.block_token_num;

  BlockManagerConfig block_manager_config;
  STATUS_CHECK_FAILURE(env->GetBlockManagerConfig(block_manager_config));

  size_t device_total, device_free;
  Status status = GetDeviceMemoryInfo(MemoryDevice::MEMORY_DEVICE, &device_free, &device_total);
  if (status.OK()) {
    size_t reserved_memory_size = device_total * block_manager_config.reserved_device_memory_ratio;
    max_block_num =
        std::min(max_block_num, (device_free - reserved_memory_size) / runtime_config_.attn_backend_config.block_size);
  }
  KLLM_LOG_INFO << "max_block_num " << max_block_num;

  // For prefix caching, the token will be used multiple times, reset it to max possible value.
  if (runtime_config.enable_prefix_caching) {
    max_block_num = (max_token_num * max_batch_size) / runtime_config_.attn_backend_config.block_token_num;
  }
  KLLM_LOG_INFO << "max_block_num: " << max_block_num;

  input_ids = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num}, rank_);
  input_offset_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);
  dp_input_offset_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);
  dp_input_offset_int32_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size + 1}, rank_);
  dp_prefill_q_offset_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);
  dp_prefill_q_offset_int32_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size + 1}, rank_);
  dp_flexible_rotary_embedding_pos = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num}, rank_);
  dp_flexible_rotary_embedding_mask = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num}, rank_);
  input_prefix_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);
  dp_input_prefix_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);
  dp_flexible_offset_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);

  BatchSchedulerConfig batch_scheduler_config;
  env->GetBatchSchedulerConfig(batch_scheduler_config);
  const size_t max_logits_tokens = max_batch_size * batch_scheduler_config.max_decode_tokens_per_req;
  logits_idx_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_logits_tokens}, rank_);

  nextn_hidden_idx_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_token_num}, rank_);

  input_length = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size}, rank_);
  kv_list = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER,
                   {static_cast<size_t>(layer_num_on_node_), max_block_num, 2}, rank_);
  layer_kv_cache_ptr =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1 + static_cast<size_t>(layer_num_on_node_ * 2)}, rank);
  if (model_config.use_dsa) {
    cur_seq_len_start = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num}, rank_);
    cur_seq_len_end = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num}, rank_);
    // kv cache meta for the indexer module
    indexer_kv_list = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER,
                             {static_cast<size_t>(layer_num_on_node_), max_block_num, 2}, rank_);
    layer_indexer_kv_cache_ptr =
        Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1 + static_cast<size_t>(layer_num_on_node_ * 2)}, rank);
    for (size_t q_len = 1; q_len <= GetDecodeTokenNumThreshold(); q_len++) {
      int num_sms = 0;
#ifdef ENABLE_CUDA
      num_sms = llm_kernels::utils::GetSMCount();
#endif
      paged_schedule_metas.emplace_back(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                                        std::vector<size_t>{static_cast<size_t>(num_sms + 1), 2}, rank_);
    }
  }
  kv_cache_offset =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size + 1 + GetDecodeTokenNumThreshold()}, rank_);
  rotary_embedding_pos = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num}, rank_);
  rotary_embedding_mask = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num}, rank_);
  block_table = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size * max_block_num}, rank);
#ifdef ENABLE_CUDA
  // Only for flashmla
  if (model_config_.use_mla) {
    num_splits =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size + 1 + GetDecodeTokenNumThreshold()}, rank_);
    for (size_t i = 0; i < GetDecodeTokenNumThreshold(); i++) {
      llm_kernels::nvidia::FlashMlaWorkspaceMap flash_mla_workspace_map;
      const size_t q_len = i + 1;
      const size_t head_num_per_tp =
          model_config.head_num / runtime_config.parallel_basic_config.attn_tensor_parallel_size;
      GetNumSmParts(flash_mla_workspace_map, q_len * head_num_per_tp, /*num_heads_k*/ 1, rank_);
      const size_t tile_scheduler_metadata_tensor_num =
          flash_mla_workspace_map.num_sm_parts * llm_kernels::nvidia::TileSchedulerMetaDataSize;
      info.tile_scheduler_metadata =
          Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {tile_scheduler_metadata_tensor_num}, rank_);
      info.num_splits = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size + 1}, rank_);
    }
#endif

  cpu_input_refit_tensor.pos_pair_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {input_ids.shape[0], 2}, rank_);
  cpu_input_refit_tensor.emb_fp32_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_POINTER, input_ids.shape, rank_);

  if (runtime_config_.enable_flexible_caching) {
    dp_flexible_rotary_embedding_mask = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num}, rank_);
    dp_src_flexible_rotary_embedding_pos = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num}, rank_);
    if (model_config_.use_mla) {
      dp_dst_flexible_rotary_embedding_pos =
          Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num}, rank_);
    }
    dp_flexible_offset_uint64_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);

    dp_dst_flexible_kv_cache_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER, {max_token_num * max_batch_size}, rank_);
    dp_src_flexible_kv_cache_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER, {max_token_num * max_batch_size}, rank_);
    dp_dst_flexible_token_idx_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num * max_batch_size}, rank_);
    dp_src_flexible_token_idx_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num * max_batch_size}, rank_);
  } else if (!enable_blocked_multi_token_forwarding_kv_ || model_config_.use_mla) {
    // Reserve space for configuration that supports flexible caching, even though flexible caching is disabled.
    dp_flexible_offset_uint64_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);
  }

  CreateVLTensors();

  EventCreateWithFlags(&kvcache_offset_event, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&rotary_embedding_event, EVENT_DISABLE_TIMING);
  EventCreateWithFlags(&input_ids_event, EVENT_DISABLE_TIMING);
#if defined(ENABLE_ACL)
  // NOTE(karlluo): for ATB, all device blocks locate on a flatten plane memory space.
  // The Ksana kv cache consists of blocks, each of which is an independent storage space. The blocks are not
  // guaranteed to be contiguous in memory. Each block has a shape of [2, layer_num, block_token_num, head_num,
  // head_dim], where 2 represents key and value. The Ascend ATB kv cache consists of kcache and vcache, which are
  // independent contiguous storage spaces. The shapes of kcache and vcache are [num_blocks * layer_num,
  // block_token_num, head_num, head_dim]. Each block has a size of [block_token_num, head_num, head_dim]. To
  // interface with the NPU, Ascend ATB (hereinafter referred to as ATB) needs to be used. In order for the NPU's
  // self/paged attention to utilize Ksana's kv cache and share the underlying memory/GPU memory management
  // capabilities, the Ksana kv cache needs to be converted to the Ascend ATB kv cache format.
  // 1. Change the block allocation method so that the blocks are contiguous in physical memory, while the upper-level
  // pointers point to different storage spaces. Originally, each block in the Ksana kv cache called malloc once. This
  // should be changed to pre-allocate a contiguous storage space of size [num_blocks, 2, layer_num, block_token_num,
  // head_num, head_dim]. The pointers of each block should then point to cache_base_ptr + (block index * 2 *
  // layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
  // 2. During each inference process, each prompt will carry an array of block IDs, which can be used to obtain the
  // pointers to the storage space. For ATB, conversion is required to use these pointers. The conversion process is
  // as follows:
  //    - Given a block ID array [b0, b1, b2, b3, b4] and the base address pointer of the Ksana kv cache after the
  //    modification in step 1, cache_base_ptr.
  //    - For ATB: The Ksana kv cache has a total of num_blocks * 2 * layer_num blocks.
  //    - Therefore, the block ID array for ATB is [b0 * layer_num * 2, b1 * layer_num * 2, b2 * layer_num * 2, b3 *
  //    layer_num * 2, b4 * layer_num * 2].
  //    - Ksana's kv cache swaps memory/GPU memory at the block level, so to reuse Ksana's kv cache's underlying
  //    memory/GPU memory management capabilities, ATB's kcache and vcache share the same Ksana kv cache.
  //    - Since each block in Ksana is divided into K and V parts, each part having a size of [layer_num,
  //    block_token_num, head_num, head_dim].
  //    - To allow ATB's kcache and vcache to share the same block ID array, the kcache pointer is cache_base_ptr, and
  //    the vcache pointer is cache_base_ptr + (layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
  //    - Therefore, the block ID array for kcache/vcache is [b0 * layer_num * 2 + layer_idx, b1 * layer_num * 2 +
  //    layer_idx, b2 * layer_num * 2 + layer_idx, b3 * layer_num * 2 + layer_idx, b4 * layer_num * 2 + layer_idx].
  SetDevice(rank);
  seq_len_host = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT32, {static_cast<uint64_t>(max_batch_size)}, rank);
  layers_slot_mapping = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                               {static_cast<uint64_t>(layer_num_on_node_), static_cast<uint64_t>(max_token_num)}, rank);
  layers_block_table =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
             {static_cast<uint64_t>(layer_num_on_node_), static_cast<uint64_t>(max_batch_size * max_block_num)}, rank);
  // https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/developmentguide/acce/ascendtb/ascendtb_01_0070.html
  // k/v_cache_blocks_base only support float16
  k_cache_blocks_base = Tensor(
      MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
      {1, runtime_config.attn_backend_config.block_token_num, model_config.head_num, model_config.size_per_head}, rank);
  v_cache_blocks_base = Tensor(
      MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
      {1, runtime_config.attn_backend_config.block_token_num, model_config.head_num, model_config.size_per_head}, rank);
  // 0: layers_slot_mapping_dim_1, 1: max_num_blocks_per_query
  atb_attention_attr = Tensor(MemoryLocation::LOCATION_HOST, TYPE_UINT64, {2}, rank);
  last_token_index_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_batch_size}, rank_);
  kv_cache_ptrs_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_POINTER,
                                {static_cast<uint64_t>(max_batch_size * max_block_num)}, rank_);
#endif
  if (Singleton<Environment>::GetInstance()->IsEnableBlockChecksum()) {
    checksum_ptrs_tensor_ =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER, {max_batch_size * max_block_num}, rank_);
    checksum_results_tensor_ =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size * max_block_num}, rank_);
  }
}

ModelInput::~ModelInput() {
  EventDestroy(kvcache_offset_event);
  EventDestroy(rotary_embedding_event);
  EventDestroy(input_ids_event);
}

void ModelInput::CreateVLTensors() {
  if (model_config_.type == "qwen2_vl") {
    dp_mrotary_embedding_pos =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {3, runtime_config_.max_step_token_num}, rank_);
  }
  if (model_config_.type == "internlmxcomposer2") {
    im_mask = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {runtime_config_.max_step_token_num}, rank_);
  }
}

void ModelInput::PrepareInputInfo(const std::vector<ForwardRequest*>& forward_reqs) {
  // Reset the offsets of shared tensors
  input_length.shape = {0};
  kv_list.shape = {0};
  indexer_kv_list.shape = {0};
  kv_cache_offset.shape = {0};
  rotary_embedding_pos.shape = {0};
  rotary_embedding_mask.shape = {0};
  block_table.shape = {0};
  num_splits.shape = {0};
  cur_seq_len_start.shape = {0};
  cur_seq_len_end.shape = {0};

  // Reset the input_infos
  flash_input.Reset();
  page_single_input.Reset();
  page_dual_input.Reset();

  for (const auto& req : forward_reqs) {
    // select input_info type
    input_info* target_input = nullptr;
    switch (req.GetType()) {
      case ForwardRequestType::kPageSingle:
        target_input = &page_single_input;
        break;
      case ForwardRequestType::kPageDual:
        target_input = &page_dual_input;
        break;
      default:  // ForwardRequestType::kFlash
        target_input = &flash_input;
    }
    target_input->reqs.emplace_back(const_cast<ForwardRequest*>(&req));
    if (req.attn_dp_group_id == attn_dp_group_id_) {
      target_input->dp_reqs.emplace_back(const_cast<ForwardRequest*>(&req));
    }
  }
}

void ModelInput::ParseFromRequests(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode) {
  // NOTE(karlluo): check batch size
  PROFILE_EVENT_SCOPE(StartPrepareReqs, "StartPrepareReqs", rank_);
  batch_size = forward_reqs.size();
  KLLM_CHECK_WITH_INFO(batch_size > 0, "ModelInput empty forward requests, batch_size == 0");
  KLLM_CHECK_WITH_INFO(
      !(connector_config_.group_role == GroupRole::DECODE && batch_size > runtime_config_.max_step_token_num),
      fmt::format("ModelInput batch_size exceed max_step_token_num at PD disaggregation. {} > {}", batch_size,
                  runtime_config_.max_step_token_num));
  KLLM_CHECK_WITH_INFO(
      !(connector_config_.group_role == GroupRole::NONE &&
        batch_size > static_cast<size_t>(runtime_config_.max_batch_size)),
      fmt::format("ModelInput batch_size exceed max_batch_size. {} > {}", batch_size, runtime_config_.max_batch_size));

  infer_stage = forward_reqs.front()->infer_stage;  // for NPU
  SetDevice(rank_);

  PrepareInputInfo(forward_reqs);

  dp_context_tokens = 0;
  dp_decode_tokens = 0;
  dp_total_prefix_len = 0;
  dp_multi_token_request_total_seq_len = 0;
  total_sampling_token_num_ = 0;
  for (const auto& req : forward_reqs) {
    if (req->attn_dp_group_id == attn_dp_group_id_) {
      if (req->GetType() == ForwardRequestType::kFlash) {
        dp_context_tokens += req->GetInputIdsLength();
        dp_total_prefix_len += req->prefix_cache_len;
        context_kv_cache_block_num += req->kv_cache_ptrs[attn_dp_rank_id_].size();

      } else {  // req->GetType() == ForwardRequestType::kPage
        dp_decode_tokens += req->GetInputIdsLength();
        decode_kv_cache_block_num += req->kv_cache_ptrs[attn_dp_rank_id_].size();
      }
    }
    total_sampling_token_num_ += req.sampling_token_num;
  }

  KLLM_LOG_DEBUG << fmt::format(
      "run_mode: {}, dp_multi_token_request_num: {}, dp_context_tokens: {}, dp_single_token_request_num: {}, "
      "dp_decode_tokens: {}, dp_total_prefix_len: {}, page_inputs.size(): {}",
      (run_mode == RunMode::kMain ? "main" : "next"), dp_multi_token_request_num, dp_context_tokens,
      dp_single_token_request_num, dp_decode_tokens, dp_total_prefix_len, page_inputs.size());
  ExecuteChecksumVerification(forward_reqs, false);

  PrepareInputIds(forward_reqs);

  PrepareVLInputRefit(forward_reqs);
  PrepareInputRefit(forward_reqs);

  PrepareVLRequest(forward_reqs);
  PrepareCutoffLayer(forward_reqs);
  PrepareNextNGatherIdx(forward_reqs, run_mode);

  PreparePrefill();
  PrepareDualDecode();
  PrepareSingleDecode();

#ifdef ENABLE_CUDA
  PrepareCudagraphParams(forward_reqs);
#endif

#ifdef ENABLE_ACL
  // NOTE(karlluo): please keep PrepareATBKVCache at the last of prepare process
  PrepareATBKVCache(forward_reqs, multi_token_request_num > 0);
#endif
}

void ModelInput::VerifyChecksumAfterForward(const std::vector<ForwardRequest*>& forward_reqs) {
  // 在 forward 计算之后执行校验和验证
  ExecuteChecksumVerification(forward_reqs, true);
}

void ModelInput::ExecuteChecksumVerification(const std::vector<ForwardRequest*>& forward_reqs, bool is_after_forward) {
  // 仅在开启校验且是最后一层时执行
  if (!(Singleton<Environment>::GetInstance()->IsEnableBlockChecksum() && context_->IsLastLayer())) {
    return;
  }
  if (forward_reqs.empty()) {
    return;
  }
  if (forward_reqs.front()->block_checksums == nullptr || forward_reqs.front()->checksummed_block_num == nullptr) {
    return;
  }

  // 需要计算校验和的块指针
  std::vector<void*> calc_ptrs;
  // <req_id, block_idx, checksum_ptr>
  std::vector<std::tuple<size_t, int, size_t*>> calc_map;
  // 需要验证校验和的块指针
  std::vector<void*> verify_ptrs;
  std::vector<std::tuple<size_t, int, size_t>> verify_map;

  for (auto& req : forward_reqs) {
    // 根据是 forward 之前还是之后，确定要处理的 token 数量
    size_t tokens_to_process = is_after_forward ? req->forwarding_tokens->size() : req->kv_cached_token_num;
    if (tokens_to_process > 0) {
      // 获取当前 rank 的校验和信息
      auto& checksums_on_rank = (*(req->block_checksums))[attn_dp_rank_id_];
      size_t& checksummed_num = (*(req->checksummed_block_num))[attn_dp_rank_id_];
      const auto& kv_cache_ptrs_on_rank = req->kv_cache_ptrs[attn_dp_rank_id_];

      // 计算需要处理的块数量
      int num_blocks_to_process = tokens_to_process / runtime_config_.attn_backend_config.block_token_num;

      if (checksums_on_rank.size() < static_cast<size_t>(num_blocks_to_process)) {
        checksums_on_rank.resize(num_blocks_to_process << 1);
      }

      for (int i = 0; i < num_blocks_to_process; ++i) {
        if (static_cast<size_t>(i) < checksummed_num) {
          // 如果块已经有校验和，则加入验证列表
          verify_ptrs.push_back(kv_cache_ptrs_on_rank[i]);
          verify_map.emplace_back(req->req_id, i, checksums_on_rank[i]);
        } else {
          // 如果块没有校验和，则加入计算列表
          calc_ptrs.push_back(kv_cache_ptrs_on_rank[i]);
          calc_map.emplace_back(req->req_id, i, &checksums_on_rank[i]);
        }
      }
      checksummed_num = num_blocks_to_process;
    }
  }

  // 合并需要计算和验证的块指针
  std::vector<void*> all_ptrs = verify_ptrs;
  all_ptrs.insert(all_ptrs.end(), calc_ptrs.begin(), calc_ptrs.end());

  if (!all_ptrs.empty()) {
    const size_t data_size = block_size_;
    void** d_ptrs = checksum_ptrs_tensor_.GetPtr<void*>();
    size_t* d_results = checksum_results_tensor_.GetPtr<size_t>();
    std::vector<size_t> h_results(all_ptrs.size());

#ifdef ENABLE_CUDA
    MemcpyAsync(d_ptrs, all_ptrs.data(), all_ptrs.size() * sizeof(void*), MEMCPY_HOST_TO_DEVICE,
                context_->GetComputeStreams()[rank_]);
    InvokeCalculateChecksum(d_ptrs, d_results, all_ptrs.size(), data_size, context_->GetComputeStreams()[rank_].Get());
    MemcpyAsync(h_results.data(), d_results, h_results.size() * sizeof(size_t), MEMCPY_DEVICE_TO_HOST,
                context_->GetComputeStreams()[rank_]);
    StreamSynchronize(context_->GetComputeStreams()[rank_]);
#endif
    // 验证校验和
    for (size_t i = 0; i < verify_ptrs.size(); ++i) {
      if (h_results[i] != std::get<2>(verify_map[i])) {
        KLLM_LOG_ERROR << "Checksum error, attn_dp_rank_id_: " << attn_dp_rank_id_
                       << ", req_id: " << std::get<0>(verify_map[i]) << ", block_idx: " << std::get<1>(verify_map[i])
                       << ", expect: " << std::get<2>(verify_map[i]) << ", actual: " << h_results[i]
                       << (is_after_forward ? " (after forward)" : "");
      }
    }

    // 保存新计算的校验和
    for (size_t i = 0; i < calc_ptrs.size(); ++i) {
      *(std::get<2>(calc_map[i])) = h_results[verify_ptrs.size() + i];
    }
  }
  if (is_after_forward) {
    for (auto& req : forward_reqs) {
      // 如果该请求的校验和错误信息尚未记录
      if (logged_checksum_error_req_ids_.find(req->req_id) == logged_checksum_error_req_ids_.end()) {
        logged_checksum_error_req_ids_.insert(req->req_id);
        // 将一个 attn_dp_rank_id_ 下的 block_checksums 求和
        size_t checksum_sum = 0;
        int checksummed_num = (*(req->checksummed_block_num))[attn_dp_rank_id_];
        if (checksummed_num == 0) {
          continue;
        }
        for (int i = 0; i < checksummed_num; ++i) {
          checksum_sum += (*(req->block_checksums))[attn_dp_rank_id_][i];
        }
        // 在首次计算完成后会打印一次 checksum_sum，方便 PD 分离校验传输的正确性
        KLLM_LOG_INFO << "req_id: " << req->req_id << ", attn_dp_rank_id_: " << attn_dp_rank_id_
                      << ", checksummed_num: " << (*(req->checksummed_block_num))[attn_dp_rank_id_]
                      << ", checksum_sum: " << checksum_sum;
      }
    }
  }
}

// TODO(ttsybyweng): VL_Model :Prepare moved into each Model Class
void ModelInput::PrepareVLInputRefit(const std::vector<ForwardRequest>& forward_reqs) {
  if (model_config_.type == "qwen2_vl") {
    PrepareMRopePos(forward_reqs);
  }
}

void ModelInput::PrepareCutoffLayer(const std::vector<ForwardRequest>& forward_reqs) {
  if (model_config_.type != "minicpm") {
    return;
  }
  for (const ForwardRequest& req : forward_reqs) {
    if (req.request_target == nullptr) {
      // if request_target is nullptr
      continue;
    }
    auto it = req.request_target->find("lm_head");
    if (it != req.request_target->end()) {
      std::vector<int> cutoff_layers = it->second.cutoff_layer;
      if (cutoff_layers.empty()) {
        cutoff_layer = model_config_.num_layer;
        continue;
      }
      auto max_layer = std::max_element(cutoff_layers.begin(), cutoff_layers.end());
      cutoff_layer = std::max((*max_layer), cutoff_layer);
    }
  }
}

void ModelInput::PrepareVLRequest(const std::vector<ForwardRequest>& forward_reqs) {
  PROFILE_EVENT_SCOPE(PrepareVLRequest, "PrepareVLRequest", rank_);
  if (model_config_.type == "internlmxcomposer2") {
    is_mask = false;
    size_t pos_num = cpu_input_refit_tensor.pos_pair_tensor.shape[0];
    if ((multi_token_request_num > 0) && (pos_num > 0)) {
#ifdef ENABLE_CUDA
      DataType weight_data_type_ = model_config_.weight_data_type;
      if (weight_data_type_ == TYPE_FP16) {
        PrepareImgMask<half>(pos_num);
      } else if (weight_data_type_ == TYPE_BF16) {
        PrepareImgMask<bfloat16>(pos_num);
      }
#endif
    }
  }
}

void ModelInput::PrepareNextNGatherIdx(const std::vector<ForwardRequest*>& forward_reqs, const RunMode run_mode) {
  PROFILE_EVENT_SCOPE(PrepareNextnGatherIdx, "PrepareNextnGatherIdx", rank_);
  std::unordered_map<size_t, size_t> updated_mtp_req_id_to_pos;
  updated_mtp_req_id_to_pos.reserve(forward_reqs.size());

  std::vector<size_t> mtp_hidden_gather_idx;
  size_t total_len = 0;
  for (const auto& req : forward_reqs) {
    const size_t input_ids_len = req->GetInputIdsLength();
    if (run_mode == RunMode::kMain) {
      updated_mtp_req_id_to_pos[req->req_id] = total_len;
    } else {
      updated_mtp_req_id_to_pos[req->req_id] = total_len + input_ids_len - 1;
    }
    total_len += input_ids_len;

    if (run_mode == RunMode::kNextN) {
      const size_t begin_idx = mtp_req_id_to_pos_[req->req_id];
      for (size_t idx = begin_idx; idx < begin_idx + input_ids_len; ++idx) {
        mtp_hidden_gather_idx.emplace_back(idx);
      }
    }
  }

  if (run_mode == RunMode::kMain) {
    return;
  }

  nextn_hidden_idx_uint64_tensor.shape = {mtp_hidden_gather_idx.size()};
  MemcpyAsync(nextn_hidden_idx_uint64_tensor.GetPtr<void>(), mtp_hidden_gather_idx.data(),
              mtp_hidden_gather_idx.size() * sizeof(decltype(mtp_hidden_gather_idx)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);
  KLLM_LOG_DEBUG << "mtp_hidden_gather_idx: " << mtp_hidden_gather_idx;
}

#ifdef ENABLE_CUDA
void ModelInput::PrepareCudagraphParams(const std::vector<ForwardRequest>& forward_reqs) {
  is_cudagraph_batchsize_matched = false;
  is_cudagraph_capture_request = false;
  if (forward_reqs[0].is_cudagraph_capture_request) {
    is_cudagraph_capture_request = true;
  }
  if (multi_token_request_num == 0 &&
      (single_token_request_num == 1 || single_token_request_num == 2 || single_token_request_num == 3)) {
    is_cudagraph_batchsize_matched = true;
  }
}
#endif

/**
 * Process the input refit information for the current batch of requests.
 *
 * Inputs:
 * 1. input_refit_embeddings (`std::vector<std::vector<float>>`) is obtained from the user request and placed on the
 * CPU.
 * 2. input_refit_embedding_tensors (`std::vector<py::object>)` is obtained from the Python plugin, which can be placed
 * on the CPU or GPU (not supported yet).
 *
 * Outputs:
 * 1. input_refit_pos_pair contains pairs of (start refit position offset in this batch, embedding length) for each
 * input refit. e.g., [(emb_pos_offset1, emb_length1), (emb_pos_offset2, emb_length2), ...]
 * 2. input_refit_emb_fp32_ptr contains pointers to all input refit on the CPU. e.g., [emb_ptr1, emb_ptr2, ...]
 *
 * After embedding lookup, the input refit embeddings will be placed to their respective intervals according to the
 * above outputs (by `input_refit_layer`).
 */
void ModelInput::PrepareInputRefit(const std::vector<ForwardRequest>& forward_reqs) {
  size_t pos_offset = 0;
  size_t cpu_input_refit_pos_pair_idx = 0;
  // Get pointers to the CPU input_refit position pair and CPU input_refit embedding float32 tensors
  int64_t* cpu_input_refit_pos_pair = cpu_input_refit_tensor.pos_pair_tensor.GetPtr<int64_t>();
  float** cpu_input_refit_emb_fp32_ptr = cpu_input_refit_tensor.emb_fp32_ptr_tensor.GetPtr<float*>();

  for (const auto& forward_req : forward_reqs) {
    // Only handle input refit for prefill requests
    if (forward_req.GetType() == ForwardRequestType::kFlash) {
      const std::vector<int>& input_refit_pos = (*forward_req.input_refit_embedding).pos;
      std::vector<std::vector<float>>& input_refit_embeddings = (*forward_req.input_refit_embedding).embeddings;
      std::vector<py::object>& input_refit_embedding_tensors = (*forward_req.input_refit_embedding).embedding_tensors;
      KLLM_CHECK_WITH_INFO(input_refit_pos.size() == input_refit_embeddings.size() ||
                               input_refit_pos.size() == input_refit_embedding_tensors.size(),
                           "`input_refit_pos.size()` should be equal to `input_refit_embeddings.size()` or "
                           "`input_refit_embedding_tensors.size()`.");

      // Iterate over the input_refit positions and embeddings
      for (size_t input_refit_idx = 0; input_refit_idx < input_refit_pos.size(); input_refit_idx++) {
        int64_t input_refit_pos_offset = input_refit_pos[input_refit_idx] + pos_offset;
        int64_t input_refit_size = 0;
        float* input_refit_fp32_ptr = nullptr;

        if (!input_refit_embedding_tensors.empty()) {
          // Get pointers from input refit embedding tensors first
          torch::Tensor input_refit_embedding_tensor;
          {
            py::gil_scoped_acquire acquire;
            input_refit_embedding_tensor = THPVariable_Unpack(input_refit_embedding_tensors[input_refit_idx].ptr());
          }
          if (input_refit_embedding_tensor.get_device() != -1) {
            KLLM_THROW("Input refit embedding tensor on GPU is not supported.");
          }
          // The input refit embedding tensor is on CPU.
          input_refit_size = input_refit_embedding_tensor.numel();
          input_refit_fp32_ptr = reinterpret_cast<float*>(input_refit_embedding_tensor.data_ptr());
        } else {
          // Get pointers from input refit embeddings
          input_refit_size = input_refit_embeddings[input_refit_idx].size();
          input_refit_fp32_ptr = input_refit_embeddings[input_refit_idx].data();
        }

        // Store the input refit information
        cpu_input_refit_pos_pair[cpu_input_refit_pos_pair_idx] = input_refit_pos_offset;
        cpu_input_refit_pos_pair[cpu_input_refit_pos_pair_idx + 1] = input_refit_size;
        cpu_input_refit_emb_fp32_ptr[cpu_input_refit_pos_pair_idx / 2] = input_refit_fp32_ptr;
        cpu_input_refit_pos_pair_idx += 2;
      }
    }
    pos_offset +=
        forward_req.forwarding_tokens->size() - std::max(forward_req.kv_cached_token_num, forward_req.prefix_cache_len);
  }

  cpu_input_refit_tensor.emb_fp32_ptr_tensor.shape = {cpu_input_refit_pos_pair_idx / 2};
  cpu_input_refit_tensor.pos_pair_tensor.shape = {cpu_input_refit_pos_pair_idx / 2, 2};
}

/**
 * The MRope position information (position and offset) of qwen2_vl is computed by the `_get_input_positions` function
 * in the Python plugin and is passed as additional tensors.
 * Before model inference, copy the position tensor (`additional_tensors[0]`) to the corresponding GPU tensor
 * (`dp_mrotary_embedding_pos`), and record the offset value (`additional_tensors[1]`).
 */
void ModelInput::PrepareMRopePos(const std::vector<ForwardRequest>& forward_reqs) {
  int64_t dp_mrotary_embedding_pos_size = 0;

  for (size_t bs_idx = 0; bs_idx < forward_reqs.size(); ++bs_idx) {
    // qwen2_vl only needs to handle the prefill requests in its dp group
    if (forward_reqs[bs_idx].GetType() == ForwardRequestType::kFlash &&
        forward_reqs[bs_idx].attn_dp_group_id == attn_dp_group_id_) {
      auto& additional_tensors = (*forward_reqs[bs_idx].input_refit_embedding).additional_tensors;
      // This is a plain text input.
      if (additional_tensors.empty()) {
        int64_t list_size = forward_reqs[bs_idx].forwarding_tokens->size() * 3;
        std::vector<int64_t> dp_mrotary_embedding_pos_list(list_size);
        for (int64_t i = 0; i < list_size; i += 3) {
          dp_mrotary_embedding_pos_list[i] = dp_mrotary_embedding_pos_list[i + 1] =
              dp_mrotary_embedding_pos_list[i + 2] = i;
        }
        MemcpyAsync(dp_mrotary_embedding_pos.GetPtr<void>() + sizeof(int64_t) * dp_mrotary_embedding_pos_size,
                    dp_mrotary_embedding_pos_list.data(), sizeof(int64_t) * list_size, MEMCPY_HOST_TO_DEVICE,
                    context_->GetD2HStreams()[rank_]);
        dp_mrotary_embedding_pos_size += list_size;
        *forward_reqs[bs_idx].mrotary_embedding_pos_offset = 0;
#ifdef ENABLE_ACL
        StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
        continue;
      }
      KLLM_CHECK_WITH_INFO(additional_tensors.size() >= 2,
                           "For visual inputs, additional_tensors should contain at least 2 tensors: position tensor "
                           "and offset tensor.");
      // This is a input with visual information.
      torch::Tensor dp_mrotary_embedding_pos_tensor;
      {
        py::gil_scoped_acquire acquire;
        dp_mrotary_embedding_pos_tensor = THPVariable_Unpack(additional_tensors[0].ptr());
      }
      int64_t tensor_size = dp_mrotary_embedding_pos_tensor.numel();
      MemcpyAsync(dp_mrotary_embedding_pos.GetPtr<void>() + sizeof(int64_t) * dp_mrotary_embedding_pos_size,
                  dp_mrotary_embedding_pos_tensor.data_ptr(), sizeof(int64_t) * tensor_size, MEMCPY_HOST_TO_DEVICE,
                  context_->GetD2HStreams()[rank_]);
      dp_mrotary_embedding_pos_size += tensor_size;

      torch::Tensor dp_mrotary_embedding_pos_offset_tensor = THPVariable_Unpack(additional_tensors[1].ptr());
      *forward_reqs[bs_idx].mrotary_embedding_pos_offset = dp_mrotary_embedding_pos_offset_tensor.item().toLong();
    }
  }
#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
}

#ifdef ENABLE_ACL
void ModelInput::PrepareATBKVCache(const std::vector<ForwardRequest>& forward_reqs, bool is_multi_token_forward) {
  std::shared_ptr<CacheManagerInterface> cache_manager = forward_reqs.front().cache_manager;
  std::shared_ptr<BlockAllocatorInterface> device_allocator =
      cache_manager->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(rank_);

  // NOTE(karlluo): block manager will change the block number in
  // ResetPreAllocatedBlocks, block_manager's allocator's blocks_num is difference from the allocator's member config,
  // so we need get it from allocator instance.
  size_t total_block_num = Singleton<Environment>::GetInstance()->GetTotalDeviceBlockNum() * 2 * layer_num_on_node_;
  if (total_block_num != k_cache_blocks_base.shape[0]) {
    void* cur_rank_block_base_ptr = device_allocator->GetBlocksBasePtr();
    void* k_cache_base_ptr = cur_rank_block_base_ptr;
    void* v_cache_base_ptr = cur_rank_block_base_ptr + block_size_ / 2;
    k_cache_blocks_base = Tensor(
        MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
        {Singleton<Environment>::GetInstance()->GetTotalDeviceBlockNum() * 2 * layer_num_on_node_,
         runtime_config_.attn_backend_config.block_token_num, model_config_.head_num, model_config_.size_per_head},
        rank_, k_cache_base_ptr);
    v_cache_blocks_base = Tensor(
        MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
        {Singleton<Environment>::GetInstance()->GetTotalDeviceBlockNum() * 2 * layer_num_on_node_,
         runtime_config_.attn_backend_config.block_token_num, model_config_.head_num, model_config_.size_per_head},
        rank_, v_cache_base_ptr);
  }

  uint32_t batch_size = forward_reqs.size();
  layers_slot_mapping_host.clear();
  layers_block_table_host.clear();
  size_t max_num_blocks_per_query = 0;
  last_token_index_tensor.shape = {batch_size};
  last_token_index_tensor.dtype = TYPE_UINT64;
  std::vector<int64_t> last_token_index_host(batch_size, 0);
  // for multi-token forwarding: slot_mapping shape is [num_layers, all_reqs_tokens]
  // for single-token forwarding: slot_mapping shape is [num_layers, batch_size]
  size_t all_seq_len = 0;
  size_t slot_mapping_dim_1 = is_multi_token_forward ? 0ul : batch_size;
  for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
    seq_len_host.GetPtr<int32_t>()[f_req_idx] = forward_reqs[f_req_idx].forwarding_tokens->size();
    if (is_multi_token_forward) {
      slot_mapping_dim_1 += forward_reqs[f_req_idx].forwarding_tokens->size();
      last_token_index_host[f_req_idx] = all_seq_len + forward_reqs[f_req_idx].forwarding_tokens->size() - 1;
    } else {
      max_num_blocks_per_query = std::max(max_num_blocks_per_query,
                                          forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[attn_dp_rank_id_].size());
      last_token_index_host[f_req_idx] = f_req_idx;
    }
    all_seq_len += forward_reqs[f_req_idx].forwarding_tokens->size();
  }
  layers_slot_mapping_host.resize(layer_num_on_node_ * slot_mapping_dim_1, 0);
  // NOTE(karlluo): for ATB, all device blocks locate on a flatten plane memory space.
  // The Ksana kv cache consists of blocks, each of which is an independent storage space. The blocks are not
  // guaranteed to be contiguous in memory. Each block has a shape of [2, layer_num, block_token_num, head_num,
  // head_dim], where 2 represents key and value. The Ascend ATB kv cache consists of kcache and vcache, which are
  // independent contiguous storage spaces. The shapes of kcache and vcache are [num_blocks * layer_num,
  // block_token_num, head_num, head_dim]. Each block has a size of [block_token_num, head_num, head_dim]. To
  // interface with the NPU, Ascend ATB (hereinafter referred to as ATB) needs to be used. In order for the NPU's
  // self/paged attention to utilize Ksana's kv cache and share the underlying memory/GPU memory management
  // capabilities, the Ksana kv cache needs to be converted to the Ascend ATB kv cache format.
  // 1. Change the block allocation method so that the blocks are contiguous in physical memory, while the upper-level
  // pointers point to different storage spaces. Originally, each block in the Ksana kv cache called malloc once. This
  // should be changed to pre-allocate a contiguous storage space of size [num_blocks, 2, layer_num, block_token_num,
  // head_num, head_dim]. The pointers of each block should then point to cache_base_ptr + (block index * 2 *
  // layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
  // 2. During each inference process, each prompt will carry an array of block IDs, which can be used to obtain the
  // pointers to the storage space. For ATB, conversion is required to use these pointers. The conversion process is
  // as follows:
  //    - Given a block ID array [b0, b1, b2, b3, b4] and the base address pointer of the Ksana kv cache after the
  //    modification in step 1, cache_base_ptr.
  //    - For ATB: The Ksana kv cache has a total of num_blocks * 2 * layer_num blocks.
  //    - Therefore, the block ID array for ATB is [b0 * layer_num * 2, b1 * layer_num * 2, b2 * layer_num * 2, b3 *
  //    layer_num * 2, b4 * layer_num * 2].
  //    - Ksana's kv cache swaps memory/GPU memory at the block level, so to reuse Ksana's kv cache's underlying
  //    memory/GPU memory management capabilities, ATB's kcache and vcache share the same Ksana kv cache.
  //    - Since each block in Ksana is divided into K and V parts, each part having a size of [layer_num,
  //    block_token_num, head_num, head_dim].
  //    - To allow ATB's kcache and vcache to share the same block ID array, the kcache pointer is cache_base_ptr, and
  //    the vcache pointer is cache_base_ptr + (layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
  //    - Therefore, the block ID array for kcache/vcache is [b0 * layer_num * 2 + layer_idx, b1 * layer_num * 2 +
  //    layer_idx, b2 * layer_num * 2 + layer_idx, b3 * layer_num * 2 + layer_idx, b4 * layer_num * 2 + layer_idx].
  // More detail refer to docs/Technology/kvcache-relationship-between-ascend-atb-and-ksana.md

  kv_cache_ptrs.clear();
  for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
    if (forward_reqs[f_req_idx].attn_dp_group_id == attn_dp_group_id_) {
      kv_cache_ptrs.insert(kv_cache_ptrs.end(), forward_reqs[f_req_idx].kv_cache_ptrs[attn_dp_rank_id_].begin(),
                           forward_reqs[f_req_idx].kv_cache_ptrs[attn_dp_rank_id_].end());
    }
  }
  if (!kv_cache_ptrs.empty()) {
    memcpy(kv_cache_ptrs_tensor.GetPtr<void>(), kv_cache_ptrs.data(), kv_cache_ptrs.size() * sizeof(void*));
  }

  if (is_multi_token_forward) {
    size_t layers_slot_mapping_offset = 0;
    for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
      if (forward_reqs[f_req_idx].attn_dp_group_id == attn_dp_group_id_) {
        for (size_t layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
          for (size_t token_idx = 0; token_idx < forward_reqs[f_req_idx].forwarding_tokens->size(); ++token_idx) {
            int32_t inner_block_offset = token_idx % runtime_config_.attn_backend_config.block_token_num;
            layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + layers_slot_mapping_offset + token_idx] =
                (forward_reqs[f_req_idx]
                     .atb_kv_cache_base_blk_ids[attn_dp_rank_id_]
                                               [token_idx / runtime_config_.attn_backend_config.block_token_num] +
                 layer_idx) *
                    runtime_config_.attn_backend_config.block_token_num +
                inner_block_offset;
          }
        }
        layers_slot_mapping_offset += forward_reqs[f_req_idx].forwarding_tokens->size();
      }
    }
  } else {
    layers_block_table_host.resize(layer_num_on_node_ * batch_size * max_num_blocks_per_query, -1);
    for (size_t f_req_idx = 0; f_req_idx < batch_size; ++f_req_idx) {
      if (forward_reqs[f_req_idx].attn_dp_group_id == attn_dp_group_id_) {
        size_t cur_query_blocks_num = forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[attn_dp_rank_id_].size();
        for (size_t layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
          for (uint32_t base_block_idx = 0; base_block_idx < cur_query_blocks_num; ++base_block_idx) {
            layers_block_table_host[layer_idx * batch_size * max_num_blocks_per_query +
                                    f_req_idx * max_num_blocks_per_query + base_block_idx] =
                forward_reqs[f_req_idx].atb_kv_cache_base_blk_ids[attn_dp_rank_id_][base_block_idx] + layer_idx;
          }
        }
        for (size_t layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
          int32_t block_id =
              forward_reqs[f_req_idx]
                  .atb_kv_cache_base_blk_ids[attn_dp_rank_id_][(seq_len_host.GetPtr<int32_t>()[f_req_idx] - 1) /
                                                               runtime_config_.attn_backend_config.block_token_num];
          layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + f_req_idx] =
              (block_id + layer_idx) * runtime_config_.attn_backend_config.block_token_num +
              ((seq_len_host.GetPtr<int32_t>()[f_req_idx] - 1) % runtime_config_.attn_backend_config.block_token_num);
        }
      }
    }
    if (!layers_block_table_host.empty()) {
      MemcpyAsync(layers_block_table.GetPtr<void>(), layers_block_table_host.data(),
                  layers_block_table_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                  context_->GetH2DStreams()[rank_]);
    }
  }
  MemcpyAsync(last_token_index_tensor.GetPtr<void>(), last_token_index_host.data(), batch_size * sizeof(int64_t),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
  MemcpyAsync(layers_slot_mapping.GetPtr<void>(), layers_slot_mapping_host.data(),
              layer_num_on_node_ * slot_mapping_dim_1 * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);
  atb_attention_attr.GetPtr<uint64_t>()[0] = slot_mapping_dim_1;
  atb_attention_attr.GetPtr<uint64_t>()[1] = max_num_blocks_per_query;
  StreamSynchronize(context_->GetH2DStreams()[rank_]);
}
#endif

// If the batch of multi token requests all require the next token (`max_new_tokens = 1`),
// and all the caching optimizations are disabled, then the kv cache is unnecessary.
void ModelInput::PrepareUseCache(input_info& input) {
  use_cache = runtime_config_.enable_prefix_caching || runtime_config_.enable_flexible_caching ||
              runtime_config_.separate_prefill_decode;
  if (enable_blocked_multi_token_forwarding_kv_) {
    use_cache = true;
  }
  if (use_cache) {
    return;
  }

  for (const ForwardRequest* dp_req : input.dp_reqs) {
    if (dp_req->sampling_config == nullptr || dp_req->sampling_config->max_new_tokens != 1) {
      use_cache = true;
      return;
    }
  }
}

#ifdef ENABLE_CUDA
template <typename T>
void ModelInput::PrepareImgMask(size_t pos_num) {
  std::vector<T> mask(input_ids.shape[0], 0.0f);
  int64_t* cpu_input_refit_pos_pair = reinterpret_cast<int64_t*>(cpu_input_refit_tensor.pos_pair_tensor.GetPtr<void>());
  size_t hidden_size = model_config_.hidden_units;
  for (size_t i = 0; i < pos_num; i++) {
    int64_t pos = cpu_input_refit_pos_pair[i * 2];
    int64_t len = cpu_input_refit_pos_pair[i * 2 + 1] / hidden_size;
    KLLM_LOG_DEBUG << "PrepareImgMask mask : " << static_cast<int>(input_ids.shape[0]) << " , start pos : " << pos
                   << " , pos len : " << len;

    if (pos + len > static_cast<int64_t>(input_ids.shape[0])) {
      KLLM_LOG_INFO << "pos + len exceeds input_ids length, set is_mask -> False";
      return;
    }
    for (int64_t j = pos; j < pos + len; j++) {
      mask[j] = 1.0f;
    }
  }
  is_mask = true;
  im_mask.shape = {input_ids.shape[0], 1};
  MemcpyAsync(im_mask.GetPtr<void>(), mask.data(), input_ids.shape[0] * sizeof(T), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);
}
#endif

void ModelInput::PrepareInputLength(input_info& input) {
  PROFILE_EVENT_SCOPE(PrepareInputLength, "PrepareInputLength", rank_);
  const auto& reqs = input.dp_reqs;

  std::vector<int> input_length_host(reqs.size());
  for (size_t i = 0; i < reqs.size(); ++i) {
    input_length_host[i] = reqs[i]->forwarding_tokens->size();
  }

  KLLM_LOG_DEBUG << "input_length_host " << input_length_host;
  input.input_length.shape = {input_length_host.size()};
  MemcpyAsync(input.input_length.GetPtr<void>(), input_length_host.data(),
              input_length_host.size() * sizeof(decltype(input_length_host)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);
}

void ModelInput::PrepareKVCacheBlocks(input_info& info) {
  PROFILE_EVENT_SCOPE(PrepareKVCacheBlocks, "PrepareKVCacheBlocks", rank_);
  const auto& reqs = info.dp_reqs;

  std::vector<int> kv_cache_offset_host(reqs.size() + 1);
  kv_cache_offset_host[0] = 0;  // first is 0
  for (size_t i = 0; i < reqs.size(); ++i) {
    kv_cache_offset_host[i + 1] = reqs[i]->kv_cache_ptrs[attn_dp_rank_id_].size() + kv_cache_offset_host[i];
  }
  info.kv_cache_block_num = kv_cache_offset_host.back();

  KLLM_LOG_DEBUG << "kv_cache_offset_host " << kv_cache_offset_host;
  info.kv_cache_offset.shape = {kv_cache_offset_host.size()};
  MemcpyAsync(info.kv_cache_offset.GetPtr<void>(), kv_cache_offset_host.data(),
              kv_cache_offset_host.size() * sizeof(decltype(kv_cache_offset_host)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->GetD2HStreams()[rank_]);

  const int total_block_num = kv_cache_offset_host.back();
  const int block_size_per_layer = block_size_ / layer_num_on_node_;
  for (const auto& [kv_cache_type, kv_cache_config] : runtime_config_.attn_backend_config.kv_cache_configs) {
#ifdef ENABLE_CUDA
    std::vector<void*> kv_list_host(total_block_num * 2);
#else
    std::vector<void*> kv_list_host(layer_num_on_node_ * total_block_num * 2);
#endif
#ifdef ENABLE_CUDA
    // layer_i = [0, 1)
    for (size_t layer_i = 0; layer_i < 1; ++layer_i) {
#else
    for (size_t layer_i = 0; layer_i < layer_num_on_node_; ++layer_i) {
#endif
      const size_t cache_block_offset = layer_i * block_size_per_layer;
      void** k_ptr = kv_list_host.data() + layer_i * total_block_num * 2;
      void** v_ptr = kv_list_host.data() + layer_i * total_block_num * 2 + total_block_num;
      for (size_t req_i = 0; req_i < reqs.size(); ++req_i) {
        for (const auto& cache_ptr : reqs[req_i]->kv_cache_ptrs[attn_dp_rank_id_]) {
          *k_ptr++ = cache_ptr + cache_block_offset + kv_cache_config.k_offset;
          *v_ptr++ = cache_ptr + cache_block_offset + kv_cache_config.v_offset;
        }
      }
    }

    Tensor* current_info_kv_list;
    Tensor* current_kv_list;
    if (kv_cache_type == "attention") {
      current_info_kv_list = &info.kv_list;
      current_kv_list = &kv_list;
    } else {  // kv_cache_type == "indexer"
      current_info_kv_list = &info.indexer_kv_list;
      current_kv_list = &indexer_kv_list;
    }
    KLLM_LOG_DEBUG << kv_cache_type << " kv_list_host " << kv_list_host;
    *current_info_kv_list = current_kv_list->GetView(
        {static_cast<size_t>(layer_num_on_node_), static_cast<size_t>(total_block_num * 2)}, current_kv_list->shape[0]);
    current_kv_list->shape[0] += current_info_kv_list->GetElementNumber();
    MemcpyAsync(current_info_kv_list->GetPtr<void>(), kv_list_host.data(),
                kv_list_host.size() * sizeof(decltype(kv_list_host)::value_type), MEMCPY_HOST_TO_DEVICE,
                context_->GetD2HStreams()[rank_]);

#ifdef ENABLE_CUDA
    // layer_i = [1, layer_num_on_node_)
    if (total_block_num > 0 && layer_num_on_node_ > 1) {
      InvokeProcessKvList(static_cast<void**>(current_info_kv_list->GetPtr<void>()), layer_num_on_node_,
                          total_block_num * 2, block_size_, context_->GetD2HStreams()[rank_].Get());
    }
#endif
  }

#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
  EventRecord(kvcache_offset_event, context_->GetD2HStreams()[rank_]);
}

void ModelInput::PrepareDecodeRotary(input_info& input) {
  PROFILE_EVENT_SCOPE(PrepareDecodeRotary, "PrepareDecodeRotary", rank_);
  size_t total_input_len = 0;
  for (const auto& req : input.dp_reqs) {
    total_input_len += req->forwarding_tokens->size() - req->kv_cached_token_num;
  }
  input.total_dp_input_ids_len = total_input_len;

  // prepare mask
  std::vector<int64_t> rotary_mask_host(total_input_len, 1);
  input.rotary_embedding_mask.shape = {rotary_mask_host.size()};
  MemcpyAsync(input.rotary_embedding_mask.GetPtr<void>(), rotary_mask_host.data(),
              sizeof(decltype(rotary_mask_host)::value_type) * rotary_mask_host.size(), MEMCPY_HOST_TO_DEVICE,
              context_->GetD2HStreams()[rank_]);

  // prepare pos
  std::vector<int64_t> rotary_pos_host(total_input_len, 1);
  size_t rotary_data_offset = 0;
  for (const auto& req : input.dp_reqs) {
    const size_t input_len = req->forwarding_tokens->size() - req->kv_cached_token_num;
    const auto pos_offset = model_config_.type == "qwen2_vl" ? *req->mrotary_embedding_pos_offset : 0;
    std::iota(rotary_pos_host.begin() + rotary_data_offset, rotary_pos_host.begin() + rotary_data_offset + input_len,
              req->kv_cached_token_num + pos_offset);
    rotary_data_offset += input_len;
  }
  KLLM_LOG_DEBUG << "rotary_pos_host " << rotary_pos_host;
  input.rotary_embedding_pos.shape = {rotary_pos_host.size()};
  MemcpyAsync(input.rotary_embedding_pos.GetPtr<void>(), rotary_pos_host.data(),
              sizeof(decltype(rotary_pos_host)::value_type) * rotary_pos_host.size(), MEMCPY_HOST_TO_DEVICE,
              context_->GetD2HStreams()[rank_]);

  EventRecord(rotary_embedding_event, context_->GetD2HStreams()[rank_]);
#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
}

void ModelInput::PrepareFlashRotary(input_info& input) {
  size_t total_input_len = 0;
  for (const auto& req : input.dp_reqs) {
    total_input_len += req->forwarding_tokens->size();
  }
  input.total_dp_input_ids_len = total_input_len - dp_total_prefix_len;

  std::vector<int64_t> rotary_mask_host(total_input_len, 1);
  std::vector<int64_t> rotary_pos_host(total_input_len);

  size_t rotary_host_idx = 0;
  for (const auto& req : input.dp_reqs) {
    if (enable_blocked_multi_token_forwarding_kv_) {
      std::iota(rotary_pos_host.begin() + rotary_host_idx,
                rotary_pos_host.begin() + rotary_host_idx + req->forwarding_tokens->size() - req->prefix_cache_len,
                req->prefix_cache_len);
      rotary_host_idx += req->forwarding_tokens->size() - req->prefix_cache_len;
    } else {
      // mask real prefix(exclude flexible cache), now equals kv_cached_token_num
      std::fill_n(rotary_mask_host.begin() + rotary_host_idx, req->kv_cached_token_num, 0);
      // Assign rotary positional values
      std::iota(rotary_pos_host.begin() + rotary_host_idx,
                rotary_pos_host.begin() + rotary_host_idx + req->forwarding_tokens->size(), 0);
      rotary_host_idx += req->forwarding_tokens->size();
    }
  }

  MemcpyAsync(input.rotary_embedding_pos.GetPtr<void>(), rotary_pos_host.data(),
              sizeof(decltype(rotary_pos_host)::value_type) * rotary_host_idx, MEMCPY_HOST_TO_DEVICE,
              context_->GetD2HStreams()[rank_]);
  MemcpyAsync(input.rotary_embedding_mask.GetPtr<void>(), rotary_mask_host.data(),
              sizeof(decltype(rotary_mask_host)::value_type) * rotary_host_idx, MEMCPY_HOST_TO_DEVICE,
              context_->GetD2HStreams()[rank_]);

  // For DeepSeek model, rope of flexible cached tokens and forwarding new tokens cannot be fused
  // because q&k buffers differ in total_tokens when there is prefix
  // If q&k forwarding parts is processed together, k flexible part must be handled separately
  if (dp_dst_flexible_kv_cache_tensor.shape[0] != 0) {
    // rotary_mask_host can be shared for rotary_pos_host and mla_dst_flexible_rotary_pos_host
    rotary_mask_host.assign(total_input_len, 0);
    // rotary_pos_host stores reverse rope information of flexible cached tokens
    rotary_pos_host.assign(total_input_len, 0);
    // mla_dst_flexible_rotary_pos_host stores correct rope information of flexible cached tokens
    std::vector<int64_t> mla_dst_flexible_rotary_pos_host(total_input_len, 0);

    size_t flexible_rotary_idx = 0;
    for (const auto& req : input.dp_reqs) {
      if (req->flexible_cache_len > 0) {
        std::fill(rotary_mask_host.begin() + flexible_rotary_idx + req->prefix_cache_len - req->flexible_cache_len,
                  rotary_mask_host.begin() + flexible_rotary_idx + req->prefix_cache_len, 1);
        for (auto& task : *req->flexible_cached_copy_tasks) {
          // Reverse rope information stores src_token_idx_ to clear existing rope information
          rotary_pos_host[flexible_rotary_idx + task.dst_token_idx_] = task.src_token_idx_;
          // For DeepSeek model, prepare correct rope information of flexible part
          if (model_config_.use_mla) {
            // Correct rope information stores dst_token_idx_ to embed correct rope information
            mla_dst_flexible_rotary_pos_host[flexible_rotary_idx + task.dst_token_idx_] = task.dst_token_idx_;
          }
        }
      }
      flexible_rotary_idx += req->forwarding_tokens->size();
    }
    MemcpyAsync(dp_flexible_rotary_embedding_mask.GetPtr<void>(), rotary_mask_host.data(),
                sizeof(decltype(rotary_mask_host)::value_type) * flexible_rotary_idx, MEMCPY_HOST_TO_DEVICE,
                context_->GetD2HStreams()[rank_]);
    MemcpyAsync(dp_src_flexible_rotary_embedding_pos.GetPtr<void>(), rotary_pos_host.data(),
                sizeof(decltype(rotary_pos_host)::value_type) * flexible_rotary_idx, MEMCPY_HOST_TO_DEVICE,
                context_->GetD2HStreams()[rank_]);
    if (model_config_.use_mla) {
      MemcpyAsync(dp_dst_flexible_rotary_embedding_pos.GetPtr<void>(), mla_dst_flexible_rotary_pos_host.data(),
                  sizeof(decltype(mla_dst_flexible_rotary_pos_host)::value_type) * flexible_rotary_idx,
                  MEMCPY_HOST_TO_DEVICE, context_->GetD2HStreams()[rank_]);
    }
  }
  EventRecord(rotary_embedding_event, context_->GetD2HStreams()[rank_]);
#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
}

void ModelInput::PrepareKVCacheBlockTable(input_info& info) {
  PROFILE_EVENT_SCOPE(PrepareKVCacheBlockTable, "PrepareKVCacheBlockTable", rank_);
  if (!enable_blocked_multi_token_forwarding_kv_) {
    return;
  }

  const auto& reqs = info.dp_reqs;

  // Initialize layer_kv_cache_ptr once
  if (!layer_kv_cache_ptr_initialized_) {
    layer_kv_cache_ptr_initialized_ = true;

    const auto cache_manager = reqs.front()->cache_manager;
    const auto device_allocator = cache_manager->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(attn_dp_rank_id_);

    const int block_size_per_layer = block_size_ / layer_num_on_node_;
    for (const auto& [kv_cache_type, kv_cache_config] : runtime_config_.attn_backend_config.kv_cache_configs) {
      void** current_layer_kv_cache_ptr;
      if (kv_cache_type == "attention") {
        current_layer_kv_cache_ptr = layer_kv_cache_ptr.GetPtr<void*>();
      } else {  // kv_cache_type == "indexer"
        current_layer_kv_cache_ptr = layer_indexer_kv_cache_ptr.GetPtr<void*>();
      }
      *reinterpret_cast<int64_t*>(current_layer_kv_cache_ptr) =
          Singleton<Environment>::GetInstance()->GetTotalDeviceBlockNum() * layer_num_on_node_ *
          (model_config_.use_mla ? 1 : 2);
      ++current_layer_kv_cache_ptr;

      // Get each layer's raw pointer of k_cache and v_cache tensor from
      // kv_cache[num_blocks, num_layers, 2, block_size, num_kv_heads, head_size]
      // block_size is [num_layers, 2, block_size, num_kv_heads, head_size]
      void* const k_cache_base_ptr = device_allocator->GetBlocksBasePtr() + kv_cache_config.k_offset;
      void* const v_cache_base_ptr = device_allocator->GetBlocksBasePtr() + kv_cache_config.v_offset;
      for (int layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
        *current_layer_kv_cache_ptr++ = k_cache_base_ptr + layer_idx * block_size_per_layer;
        *current_layer_kv_cache_ptr++ = v_cache_base_ptr + layer_idx * block_size_per_layer;
      }
    }
  }

  size_t max_num_blocks_per_query = 0;
  for (const auto& req : reqs) {
    max_num_blocks_per_query =
        std::max(max_num_blocks_per_query, req->atb_kv_cache_base_blk_ids[attn_dp_rank_id_].size());
  }

  std::vector<int32_t> block_table_host(reqs.size() * max_num_blocks_per_query, -1);
  // The pointer has already been offset by layer_idx, so all layers can use the same block_table.
  for (size_t i = 0; i < reqs.size(); ++i) {
    const auto& block_ids = reqs[i]->atb_kv_cache_base_blk_ids[attn_dp_rank_id_];
    for (size_t base_block_idx = 0; base_block_idx < block_ids.size(); ++base_block_idx) {
      block_table_host[i * max_num_blocks_per_query + base_block_idx] =
          block_ids[base_block_idx] / (model_config_.use_mla ? 2 : 1);
    }
  }
  KLLM_LOG_DEBUG << "block_table_host " << block_table_host;
  info.block_table.shape = {reqs.size(), max_num_blocks_per_query};
  MemcpyAsync(info.block_table.GetPtr<void>(), block_table_host.data(),
              block_table_host.size() * sizeof(decltype(block_table_host)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);

#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
  EventRecord(kvcache_offset_event, context_->GetD2HStreams()[rank_]);
}

void ModelInput::PrepareFlashMla(input_info& input) {
#ifdef ENABLE_CUDA
  if (model_config_.mla_config.kv_lora_rank == 0 || input.dp_reqs.empty()) {
    return;
  }
  PROFILE_EVENT_SCOPE(PrepareFlashMla, "PrepareFlashMla", rank_);
  Stream stream = context_->GetH2DStreams()[rank_];
  static const int head_num_per_tp =
      model_config_.head_num / runtime_config_.parallel_basic_config.attn_tensor_parallel_size;
  const int q_len = input.total_dp_input_ids_len / input.dp_reqs.size();
  llm_kernels::nvidia::FlashMlaWorkspaceMap flash_mla_workspace_map;
  GetNumSmParts(flash_mla_workspace_map, q_len * head_num_per_tp, 1, rank_, stream.Get());
  flash_mla_workspace_map.tile_scheduler_metadata_ptr = input.tile_scheduler_metadata.GetPtr<int>();
  flash_mla_workspace_map.num_splits_ptr = input.num_splits.GetPtr<int>();
  InvokeGetMlaMetadata(input.input_length.GetPtr<int>(), flash_mla_workspace_map, input.dp_reqs.size(), stream.Get());
#endif
}

void ModelInput::PrepareSingleDecode() {
  if (page_single_input.dp_reqs.empty()) {
    KLLM_LOG_DEBUG << "page_single_input empty";
    return;
  }
  PROFILE_EVENT_SCOPE(PrepareSingleDecode, "PrepareSingleDecode", rank_);
  PreparePageInput(page_single_input);
  PrepareKVCacheBlocks(page_single_input);
  PrepareKVCacheBlockTable(page_single_input);
  PrepareDecodeRotary(page_single_input);
  PrepareFlashMla(page_single_input);
}

void ModelInput::PrepareDualDecode() {
  if (page_dual_input.dp_reqs.empty()) {
    KLLM_LOG_DEBUG << "page_dual_input empty";
    return;
  }
  PROFILE_EVENT_SCOPE(PrepareDualDecode, "PrepareDualDecode", rank_);
  PreparePageInput(page_dual_input);
  PrepareKVCacheBlocks(page_dual_input);
  PrepareKVCacheBlockTable(page_dual_input);
  PrepareDecodeRotary(page_dual_input);
  PrepareFlashMla(page_dual_input);
}

void ModelInput::PreparePrefill() {
  if (flash_input.dp_reqs.empty()) {
    KLLM_LOG_DEBUG << "flash_input empty";
    return;
  }

  PROFILE_EVENT_SCOPE(PreparePrefill, "PreparePrefill", rank_);
  PrepareUseCache(flash_input);
  if (use_cache) {
    PrepareFlexibleCache(flash_input);
    PrepareKVCacheBlocks(flash_input);
    PrepareKVCacheBlockTable(flash_input);
  }

  PrepareFlashRotary(flash_input);
  PrepareCuSeqLen(flash_input, false);
}

void ModelInput::PrepareDecode() {
  PROFILE_EVENT_SCOPE(PrepareDecode, "PrepareDecode", rank_);
  for (auto& page_input : page_inputs) {
    if (page_input.dp_reqs.empty()) {
      continue;
    }
    PrepareInputLength(page_input);
    PrepareKVCacheBlocks(page_input);
    PrepareKVCacheBlockTable(page_input);
    PrepareDecodeRotary(page_input);
    PrepareFlashMla(page_input);
    PrepareCuSeqLen(page_input, true);
    PreparePagedScheduleMeta(page_input);
  }
}

void ModelInput::PrepareFlexibleCache(input_info& input) {
  // Flexible cache supported scope: dense model with paged attn layout, deepseek model
  // Flexible cache not supported scope: dense model with flash attn layout
  const bool is_dense_model_flash_layout = !model_config_.use_mla && enable_blocked_multi_token_forwarding_kv_;
  if (!runtime_config_.enable_flexible_caching && is_dense_model_flash_layout) {
    return;
  }

  std::vector<int> dst_flexible_kv_cache_id_cpu;
  std::vector<int> src_flexible_kv_cache_id_cpu;
  std::vector<int> dst_flexible_token_idx_cpu;
  std::vector<int> src_flexible_token_idx_cpu;
  std::vector<uint64_t> flexible_offset_uint64_cpu = {0};

  for (const auto& req : input.dp_reqs) {
    if (req->attn_dp_group_id != attn_dp_group_id_) {
      continue;
    }

    flexible_offset_uint64_cpu.emplace_back(flexible_offset_uint64_cpu.back() + req->prefix_cache_len -
                                            req->flexible_cache_len);
    for (const auto& task : *req->flexible_cached_copy_tasks) {
      dst_flexible_kv_cache_id_cpu.emplace_back(task.dst_block_id_[attn_dp_rank_id_]);
      src_flexible_kv_cache_id_cpu.emplace_back(task.src_block_id_[attn_dp_rank_id_]);
      dst_flexible_token_idx_cpu.emplace_back(task.dst_token_idx_);
      src_flexible_token_idx_cpu.emplace_back(task.src_token_idx_);
    }
  }

  dp_flexible_offset_uint64_tensor.shape = {flexible_offset_uint64_cpu.size()};
  MemcpyAsync(dp_flexible_offset_uint64_tensor.GetPtr<void>(), flexible_offset_uint64_cpu.data(),
              flexible_offset_uint64_cpu.size() * sizeof(decltype(flexible_offset_uint64_cpu)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);

  if (dst_flexible_kv_cache_id_cpu.empty() ||
      (!runtime_config_.enable_flexible_caching && !is_dense_model_flash_layout)) {
    dp_dst_flexible_kv_cache_tensor.shape = {0};
    return;
  }

  std::vector<void*> dst_flexible_kv_cache_cpu(dst_flexible_kv_cache_id_cpu.size());
  std::vector<void*> src_flexible_kv_cache_cpu(src_flexible_kv_cache_id_cpu.size());
  const auto cache_manager = input.dp_reqs.front()->cache_manager;
  auto device_allocator = cache_manager->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(attn_dp_rank_id_);
  device_allocator->GetBlockPtrs(dst_flexible_kv_cache_id_cpu, dst_flexible_kv_cache_cpu);
  device_allocator->GetBlockPtrs(src_flexible_kv_cache_id_cpu, src_flexible_kv_cache_cpu);

  dp_dst_flexible_kv_cache_tensor.shape = {dst_flexible_kv_cache_cpu.size()};
  MemcpyAsync(dp_dst_flexible_kv_cache_tensor.GetPtr<void>(), dst_flexible_kv_cache_cpu.data(),
              dst_flexible_kv_cache_cpu.size() * sizeof(decltype(dst_flexible_kv_cache_cpu)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
  MemcpyAsync(dp_src_flexible_kv_cache_tensor.GetPtr<void>(), src_flexible_kv_cache_cpu.data(),
              src_flexible_kv_cache_cpu.size() * sizeof(decltype(src_flexible_kv_cache_cpu)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
  MemcpyAsync(dp_dst_flexible_token_idx_tensor.GetPtr<void>(), dst_flexible_token_idx_cpu.data(),
              dst_flexible_token_idx_cpu.size() * sizeof(decltype(dst_flexible_token_idx_cpu)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
  MemcpyAsync(dp_src_flexible_token_idx_tensor.GetPtr<void>(), src_flexible_token_idx_cpu.data(),
              src_flexible_token_idx_cpu.size() * sizeof(decltype(src_flexible_token_idx_cpu)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
}

void ModelInput::PrepareInputIds(const std::vector<ForwardRequest>& forward_reqs) {
  PROFILE_EVENT_SCOPE(PrepareInputIds, "PrepareInputIds", rank_);
  input_ids_cpu.clear();
  input_offset_list_uint64.assign(1, 0);
  input_prefix_list_uint64.assign(1, 0);
  dp_input_offset_list_uint64.assign(1, 0);
  dp_input_prefix_list_uint64.assign(1, 0);
  multi_token_request_max_tokens = 0;
  single_token_request_max_tokens = 0;
  dp_multi_token_request_max_tokens = 0;
  dp_single_token_request_max_tokens = 0;
  dp_max_forwarding_tokens = 0;  // used for blocked_prefill

  std::vector<size_t> logits_idx_list(total_sampling_token_num_);
  size_t logits_idx_list_idx = 0;
  std::vector<size_t> dp_prefill_q_offset(1, 0);
  std::vector<size_t> dp_input_ids_lens(attn_dp_group_size_, 0);

  auto process_func = [&](const ForwardRequest& req) {
    const auto& forwarding_tokens = *(req.forwarding_tokens);
    const size_t input_length = forwarding_tokens.size();
    const bool in_dp_group = req.attn_dp_group_id == attn_dp_group_id_;

    // Skip prefix token(include flexible cache token)
    const size_t skip_token_num = std::max(req.kv_cached_token_num, req.prefix_cache_len);
    const size_t input_ids_len = input_length - skip_token_num;
    KLLM_LOG_DEBUG << "forwarding_tokens_num " << input_length << ", skip_token_num " << skip_token_num
                   << ", kv_cached_token_num " << req.kv_cached_token_num << ", prefix_cache_len "
                   << req.prefix_cache_len;

    input_ids_cpu.insert(input_ids_cpu.end(), forwarding_tokens.begin() + skip_token_num, forwarding_tokens.end());
    dp_max_forwarding_tokens = std::max(dp_max_forwarding_tokens, input_ids_len);
    dp_input_ids_lens[req.attn_dp_group_id] += input_ids_len;

    if (req.GetType() == ForwardRequestType::kFlash) {
      multi_token_request_max_tokens = std::max(multi_token_request_max_tokens, input_length);
      input_offset_list_uint64.emplace_back(input_offset_list_uint64.back() + input_length);
      input_prefix_list_uint64.emplace_back(input_prefix_list_uint64.back() + skip_token_num);
      if (in_dp_group) {
        dp_multi_token_request_max_tokens = std::max(dp_multi_token_request_max_tokens, input_length);
        dp_prefill_q_offset.emplace_back(dp_prefill_q_offset.back() + input_ids_len);
        dp_input_offset_list_uint64.emplace_back(dp_input_offset_list_uint64.back() + input_length);
        dp_input_prefix_list_uint64.emplace_back(dp_input_prefix_list_uint64.back() + skip_token_num);
      }
    } else {
      single_token_request_max_tokens = std::max(single_token_request_max_tokens, input_length);
      input_offset_list_uint64.emplace_back(input_offset_list_uint64.back() + input_ids_len);
      input_prefix_list_uint64.emplace_back(input_prefix_list_uint64.back());
      if (in_dp_group) {
        dp_single_token_request_max_tokens = std::max(dp_single_token_request_max_tokens, input_length);
        dp_input_offset_list_uint64.emplace_back(dp_input_offset_list_uint64.back() + 1);
        dp_input_prefix_list_uint64.emplace_back(dp_input_prefix_list_uint64.back());
      }
    }

    if (req.GetType() == ForwardRequestType::kFlash &&
        req.logits_custom_length > 0) {  // Specify the range of logits required
      for (const auto& [l, r] : req.request_target->at("logits").slice_pos) {
        std::iota(logits_idx_list.begin() + logits_idx_list_idx,
                  logits_idx_list.begin() + logits_idx_list_idx + r - l + 1, input_ids_cpu.size() - input_length + l);
        logits_idx_list_idx += r - l + 1;
      }
    } else {
      // In the standard case, only the logits of the last token are needed
      // In the case of speculative decoding, logits are required for both the last token and the predicted token
      std::iota(logits_idx_list.begin() + logits_idx_list_idx,
                logits_idx_list.begin() + logits_idx_list_idx + req.sampling_token_num,
                input_ids_cpu.size() - req.sampling_token_num);
      logits_idx_list_idx += req.sampling_token_num;
    }
  };

  for (const auto& req : forward_reqs) {
    process_func(req);
  }

  KLLM_LOG_DEBUG << "input_ids_cpu " << input_ids_cpu;
  KLLM_LOG_DEBUG << "logits_idx_list " << logits_idx_list;
  KLLM_LOG_DEBUG << "input_offset_list_uint64 " << input_offset_list_uint64;
  KLLM_LOG_DEBUG << "input_prefix_list_uint64 " << input_prefix_list_uint64;
  KLLM_LOG_DEBUG << "dp_input_offset_list_uint64 " << dp_input_offset_list_uint64;
  KLLM_LOG_DEBUG << "dp_input_prefix_list_uint64 " << dp_input_prefix_list_uint64;
  KLLM_LOG_DEBUG << "dp_prefill_q_offset " << dp_prefill_q_offset;

  input_ids.shape = {input_ids_cpu.size()};
  input_ids.dtype = TYPE_INT32;
  MemcpyAsync(input_ids.GetPtr<void>(), input_ids_cpu.data(),
              input_ids_cpu.size() * sizeof(decltype(input_ids_cpu)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);

  logits_idx_uint64_tensor.shape = {logits_idx_list.size()};
  logits_idx_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(logits_idx_uint64_tensor.GetPtr<void>(), logits_idx_list.data(),
              logits_idx_list.size() * sizeof(decltype(logits_idx_list)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);

  dp_prefill_q_offset_uint64_tensor.shape = {dp_prefill_q_offset.size()};
  dp_prefill_q_offset_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(dp_prefill_q_offset_uint64_tensor.GetPtr<void>(), dp_prefill_q_offset.data(),
              dp_prefill_q_offset.size() * sizeof(decltype(dp_prefill_q_offset)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);

  std::vector<int> int32_vector(dp_prefill_q_offset.size());
  for (size_t i = 0; i < dp_prefill_q_offset.size(); ++i) {
    int32_vector[i] = static_cast<int>(dp_prefill_q_offset[i]);
  }
  dp_prefill_q_offset_int32_tensor.shape = {int32_vector.size()};
  dp_prefill_q_offset_int32_tensor.dtype = TYPE_INT32;
  MemcpyAsync(dp_prefill_q_offset_int32_tensor.GetPtr<void>(), int32_vector.data(),
              int32_vector.size() * sizeof(decltype(int32_vector)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);

  input_offset_uint64_tensor.shape = {input_offset_list_uint64.size()};
  input_offset_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(input_offset_uint64_tensor.GetPtr<void>(), input_offset_list_uint64.data(),
              input_offset_list_uint64.size() * sizeof(decltype(input_offset_list_uint64)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);

  dp_input_offset_uint64_tensor.shape = {dp_input_offset_list_uint64.size()};
  dp_input_offset_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(dp_input_offset_uint64_tensor.GetPtr<void>(), dp_input_offset_list_uint64.data(),
              dp_input_offset_list_uint64.size() * sizeof(decltype(dp_input_offset_list_uint64)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);

  int32_vector.resize(dp_input_offset_list_uint64.size());
  for (size_t i = 0; i < dp_input_offset_list_uint64.size(); ++i) {
    int32_vector[i] = static_cast<int>(dp_input_offset_list_uint64[i]);
  }
  dp_input_offset_int32_tensor.shape = {int32_vector.size()};
  dp_input_offset_int32_tensor.dtype = TYPE_INT32;
  MemcpyAsync(dp_input_offset_int32_tensor.GetPtr<void>(), int32_vector.data(),
              int32_vector.size() * sizeof(decltype(int32_vector)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->GetH2DStreams()[rank_]);

  input_prefix_uint64_tensor.shape = {input_prefix_list_uint64.size()};
  input_prefix_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(input_prefix_uint64_tensor.GetPtr<void>(), input_prefix_list_uint64.data(),
              input_prefix_list_uint64.size() * sizeof(decltype(input_prefix_list_uint64)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);

  dp_input_prefix_uint64_tensor.shape = {dp_input_prefix_list_uint64.size()};
  dp_input_prefix_uint64_tensor.dtype = TYPE_UINT64;
  MemcpyAsync(dp_input_prefix_uint64_tensor.GetPtr<void>(), dp_input_prefix_list_uint64.data(),
              dp_input_prefix_list_uint64.size() * sizeof(decltype(dp_input_prefix_list_uint64)::value_type),
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);

  size_t attn_dp_group_offset = 0;
  for (size_t i = 0; i < attn_dp_group_size_; ++i) {
    attn_dp_group_offsets_[i] = attn_dp_group_offset;
    attn_dp_group_offset += dp_input_ids_lens[i];
  }
  KLLM_LOG_DEBUG << "attn_dp_group_offsets_ " << attn_dp_group_offsets_;

  EventRecord(input_ids_event, context_->GetH2DStreams()[rank_]);
#ifdef ENABLE_ACL
  // Event wait between streams seems not work, force sync here.
  StreamSynchronize(context_->GetH2DStreams()[rank_]);
#endif
}

// TODO(qiannan): consider prefix cache len
// Prepare cumulative sequence length arrays for both flash (prefill) and paged (decode) inputs
void ModelInput::PrepareCuSeqLen(input_info& input, bool is_paged) {
  if (!model_config_.use_dsa || input.dp_reqs.empty()) {
    return;
  }
  PROFILE_EVENT_SCOPE(PrepareCuSeqLen, "PrepareCuSeqLen", rank_);

  const size_t batch_size = input.dp_reqs.size();

  // Prepare cumulative_seq_lens array
  std::vector<int> cumulative_seq_lens(batch_size + 1, 0);
  for (size_t i = 0; i < batch_size; ++i) {
    cumulative_seq_lens[i + 1] = cumulative_seq_lens[i] + input.dp_reqs[i]->GetInputIdsLength();
  }

  // Calculate cur_seq_len_start and cur_seq_len_end with unified batch-first iteration
  std::vector<int> cur_seq_len_start_host(cumulative_seq_lens.back());
  std::vector<int> cur_seq_len_end_host(cumulative_seq_lens.back());

  for (size_t b = 0; b < batch_size; ++b) {
    const auto& req = input.dp_reqs[b];
    const int token_begin = cumulative_seq_lens[b];
    const int token_end = cumulative_seq_lens[b + 1];
    const size_t input_ids_len = token_end - token_begin;

    for (size_t i = 0; i < input_ids_len; ++i) {
      const int idx = token_begin + i;
      const int local_idx = idx - token_begin;

      if (is_paged) {
        // Paged: start is always 0, end is cumulative visible length
        const size_t skip_token_num = std::max(req->kv_cached_token_num, req->prefix_cache_len);
        cur_seq_len_start_host[idx] = 0;
        cur_seq_len_end_host[idx] = skip_token_num + local_idx + 1;
      } else {
        // Flash: start points to sequence beginning, end increments within sequence
        cur_seq_len_start_host[idx] = token_begin;
        cur_seq_len_end_host[idx] = token_begin + local_idx + 1;
      }
    }
  }

  // Set tensor shapes and copy data to device
  input.cur_seq_len_start = cur_seq_len_start.GetView({cur_seq_len_start_host.size()}, cur_seq_len_start.shape[0]);
  cur_seq_len_start.shape[0] += input.cur_seq_len_start.GetElementNumber();
  MemcpyAsync(input.cur_seq_len_start.GetPtr<void>(), cur_seq_len_start_host.data(),
              cur_seq_len_start_host.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);

  input.cur_seq_len_end = cur_seq_len_end.GetView({cur_seq_len_end_host.size()}, cur_seq_len_end.shape[0]);
  cur_seq_len_end.shape[0] += input.cur_seq_len_end.GetElementNumber();
  MemcpyAsync(input.cur_seq_len_end.GetPtr<void>(), cur_seq_len_end_host.data(),
              cur_seq_len_end_host.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);

#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetH2DStreams()[rank_]);
#endif
}

void ModelInput::PreparePagedScheduleMeta(input_info& input) {
#ifdef ENABLE_CUDA
  if (!model_config_.use_dsa || input.dp_reqs.empty()) {
    return;
  }
  PROFILE_EVENT_SCOPE(PreparePagedScheduleMeta, "PreparePagedScheduleMeta", rank_);

  // Get the appropriate paged_schedule_meta tensor based on q_seq_len
  // For page_inputs, q_seq_len is the number of tokens per request
  KLLM_CHECK_WITH_INFO(input.q_seq_len > 0 && input.q_seq_len <= paged_schedule_metas.size(),
                       fmt::format("q_seq_len is out of range, q_seq_len: {}, paged_schedule_metas.size(): {}",
                                   input.q_seq_len, paged_schedule_metas.size()));
  input.paged_schedule_meta = paged_schedule_metas[input.q_seq_len - 1];

  // Use input_length as context_lens (already prepared in PrepareInputLength)
  // For paged input, input_length contains the forwarding_tokens->size() for each request
  const int batch_size = static_cast<int>(input.input_length.shape[0]);

  // Call PagedMqaLogitsMetadata to compute schedule metadata
  auto deepseek_deepgemm_wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance(rank_);

  // Wrap input.input_length as torch tensor (context_lens)
  auto context_lens_torch = torch::from_blob(input.input_length.GetPtr<void>(), {batch_size},
                                             torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank_));

  auto schedule_meta_torch = torch::from_blob(input.paged_schedule_meta.GetPtr<void>(),
                                              {static_cast<int64_t>(input.paged_schedule_meta.shape[0]),
                                               static_cast<int64_t>(input.paged_schedule_meta.shape[1])},
                                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank_));

  const int block_size = runtime_config_.attn_backend_config.block_token_num;
  deepseek_deepgemm_wrapper->PagedMqaLogitsMetadata(context_lens_torch, schedule_meta_torch, batch_size, block_size);

  KLLM_LOG_DEBUG << fmt::format("PreparePagedScheduleMeta completed: q_seq_len={}, batch_size={}", input.q_seq_len,
                                batch_size);
#endif  // ENABLE_CUDA
}

}  // namespace ksana_llm
