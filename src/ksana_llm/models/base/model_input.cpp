/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_input.h"

#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>
#include "ksana_llm/cache_manager/block_allocator/block_allocator_interface.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#  include "csrc/kernels/nvidia/flash_mla/kernels/params.h"
#endif

#include "ksana_llm/profiler/profile_event.h"

#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

ModelInput::ModelInput(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : model_config_(model_config), rank_(rank), context_(context) {
  auto env = Singleton<Environment>::GetInstance();
  env->GetPipelineConfig(pipeline_config_);
  env->GetAttnBackendConfig(attn_backend_config_);

  block_size_ = env->GetBlockSize();
  const size_t max_batch_size = model_config_.max_batch_size;
  const size_t max_token_num = model_config.max_step_token_num;  // max step token num
  layer_num_on_node_ = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
  if (pipeline_config_.lower_nextn_layer_idx >= static_cast<int>(model_config_.num_layer)) {
    layer_num_on_node_ += pipeline_config_.upper_nextn_layer_idx - pipeline_config_.lower_nextn_layer_idx + 1;
    KLLM_LOG_INFO << "ModelInput add next n, now layer: " << layer_num_on_node_;
  }

  const char* const enable_flash_mla_env = std::getenv("ENABLE_FLASH_MLA");
  enable_flash_mla_ = (enable_flash_mla_env != nullptr && strcmp(enable_flash_mla_env, "1") == 0);

  attn_dp_group_id_ = rank_ / env->GetAttentionTensorParallel();
  attn_dp_rank_id_ = rank_ % env->GetAttentionTensorParallel();
  attn_dp_group_size_ = env->GetAttnDataParallelSize();
  attn_dp_group_offsets_.assign(attn_dp_group_size_ * 4, 0);
  KLLM_LOG_INFO << "rank:" << rank_ << ", attn_dp_group_id_: " << attn_dp_group_id_
                << ", attn_dp_rank_id_: " << attn_dp_rank_id_ << ", attn_dp_group_size_: " << attn_dp_group_size_;

  const size_t max_seq_len = model_config.max_token_num;  // max seq len for one request
  size_t max_block_num =
      (max_seq_len * max_batch_size + model_config.block_token_num - 1) / model_config.block_token_num;

  BlockManagerConfig block_manager_config;
  STATUS_CHECK_FAILURE(env->GetBlockManagerConfig(block_manager_config));

  size_t device_total, device_free;
  Status status = GetDeviceMemoryInfo(MemoryDevice::MEMORY_DEVICE, &device_free, &device_total);
  if (status.OK()) {
    size_t reserved_memory_size = device_total * block_manager_config.reserved_device_memory_ratio;
    max_block_num = std::min(max_block_num, (device_free - reserved_memory_size) / env->GetBlockSize());
  }
  KLLM_LOG_INFO << "max_block_num " << max_block_num;

  // For prefix caching, the token will be used multiple times, reset it to max possible value.
  if (env->IsPrefixCachingEnabled()) {
    max_block_num = (max_token_num * max_batch_size) / env->GetBlockTokenNum();
  }

  input_ids = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num}, rank_);
  input_offset_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);
  dp_input_offset_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);
  dp_prefill_q_offset_uint64_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);
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

  auto allocate_input_info = [&](input_info& info) {
    const size_t head_num_per_tp = model_config.head_num / env->GetAttentionTensorParallel();

    info.input_length = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size}, rank_);
    info.kv_list = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER,
                          {static_cast<size_t>(layer_num_on_node_), max_block_num, 2}, rank_);
    info.kv_cache_offset = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size + 1}, rank_);
    info.rotary_embedding_pos = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num}, rank_);
    info.rotary_embedding_mask = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_token_num}, rank_);
    info.kv_cache_buffer =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
               {max_batch_size, (max_seq_len + 511) / 512, head_num_per_tp, model_config.size_per_head + 2}, rank_);
    info.layer_kv_cache_ptr =
        Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {1 + static_cast<size_t>(layer_num_on_node_ * 2)}, rank);
    info.metadata = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {4}, rank);
    info.block_table = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size * max_block_num * 2}, rank);

#ifdef ENABLE_CUDA
    if (model_config_.mla_config.kv_lora_rank > 0) {
      llm_kernels::nvidia::FlashMlaWorkspaceMap flash_mla_workspace_map;
      GetNumSmParts(flash_mla_workspace_map, head_num_per_tp, 1, rank_, 0);
      const size_t tile_scheduler_metadata_tensor_num =
          flash_mla_workspace_map.num_sm_parts * llm_kernels::nvidia::TileSchedulerMetaDataSize;
      info.tile_scheduler_metadata =
          Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {tile_scheduler_metadata_tensor_num}, rank_);
      info.num_splits = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch_size + 1}, rank_);
    }
#endif
  };

  allocate_input_info(flash_input);
  allocate_input_info(page_dual_input);
  allocate_input_info(page_single_input);

  cpu_input_refit_tensor.pos_pair_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {input_ids.shape[0], 2}, rank_);
  cpu_input_refit_tensor.emb_fp32_ptr_tensor =
      Tensor(MemoryLocation::LOCATION_HOST, TYPE_POINTER, input_ids.shape, rank_);

  dp_dst_flexible_kv_cache_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER, {max_token_num * max_batch_size}, rank_);
  dp_src_flexible_kv_cache_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_POINTER, {max_token_num * max_batch_size}, rank_);
  dp_dst_flexible_token_idx_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num * max_batch_size}, rank_);
  dp_src_flexible_token_idx_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_token_num * max_batch_size}, rank_);

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
  k_cache_blocks_base =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
             {1, model_config.block_token_num, model_config.head_num, model_config.size_per_head}, rank);
  v_cache_blocks_base =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
             {1, model_config.block_token_num, model_config.head_num, model_config.size_per_head}, rank);
  // 0: layers_slot_mapping_dim_1, 1: max_num_blocks_per_query
  atb_attention_attr = Tensor(MemoryLocation::LOCATION_HOST, TYPE_UINT64, {2}, rank);
  last_token_index_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {max_batch_size}, rank_);
  kv_cache_ptrs_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_POINTER,
                                {static_cast<uint64_t>(max_batch_size * max_block_num)}, rank_);
#endif

  // Create buffer for enable_blocked_multi_token_forwarding_kv.
  dp_input_without_prefix_uint64_tensor =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {max_batch_size + 1}, rank_);
}

ModelInput::~ModelInput() {
  EventDestroy(kvcache_offset_event);
  EventDestroy(rotary_embedding_event);
  EventDestroy(input_ids_event);
}

void ModelInput::CreateVLTensors() {
  if (model_config_.type == "qwen2_vl") {
    dp_mrotary_embedding_pos =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {3, model_config_.max_step_token_num}, rank_);
  }
  if (model_config_.type == "internlmxcomposer2") {
    im_mask = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {model_config_.max_step_token_num}, rank_);
  }
}

void ModelInput::ParseFromRequests(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode) {
  // NOTE(karlluo): check batch size
  batch_size = forward_reqs.size();
  if (batch_size == 0) {
    KLLM_THROW(fmt::format("ModelInput empty forward requests, batch_size == 0"));
  } else if (batch_size > (size_t)model_config_.max_batch_size) {
    KLLM_THROW(
        fmt::format("ModelInput batch_size exceed max_batch_size. {} > {}", batch_size, model_config_.max_batch_size));
  }
  multi_token_request_total_seq_len = 0;
  dp_multi_token_request_total_seq_len = 0;

  total_sampling_token_num_ = 0;
  total_prefix_len = 0;
  dp_total_prefix_len = 0;
  infer_stage = forward_reqs.front().infer_stage;  // for NPU

  if (run_mode == RunMode::kMain) {
    mtp_req_id_to_pos_.clear();
  }

  flash_input.Reset();
  page_single_input.Reset();
  page_dual_input.Reset();

  SetDevice(rank_);

  for (const auto& forward_req : forward_reqs) {
    // select input_info type
    const size_t input_ids_len = forward_req.forwarding_tokens->size() - forward_req.kv_cached_token_num;
    input_info* target_input = nullptr;
    if (input_ids_len == 1) {
      target_input = &page_single_input;
    } else if (input_ids_len == 2 && IsAbsorbWeightsEnabled() && forward_req.kv_cached_token_num != 0) {
      target_input = &page_dual_input;
    } else {
      target_input = &flash_input;

      total_prefix_len += forward_req.prefix_cache_len;
      multi_token_request_total_seq_len += forward_req.forwarding_tokens->size();
      if (forward_req.attn_dp_group_id == attn_dp_group_id_) {
        dp_total_prefix_len += forward_req.prefix_cache_len;
        dp_multi_token_request_total_seq_len += forward_req.forwarding_tokens->size();
      }
    }

    target_input->reqs.emplace_back(const_cast<ForwardRequest*>(&forward_req));
    if (forward_req.attn_dp_group_id == attn_dp_group_id_) {
      target_input->dp_reqs.emplace_back(const_cast<ForwardRequest*>(&forward_req));
    }

    total_sampling_token_num_ += forward_req.sampling_token_num;
  }

  multi_token_request_num = flash_input.reqs.size();
  dp_multi_token_request_num = flash_input.dp_reqs.size();
  single_token_request_num = page_single_input.reqs.size() + page_dual_input.reqs.size();
  dp_single_token_request_num = page_single_input.dp_reqs.size() + page_dual_input.dp_reqs.size();
  dp_batch_size = dp_single_token_request_num + dp_multi_token_request_num;

  KLLM_LOG_DEBUG << "run mode: " << (run_mode == RunMode::kMain ? "main" : "next")
                 << ", flash_input: " << flash_input.reqs.size()
                 << ", page_single_input: " << page_single_input.reqs.size()
                 << ", page_dual_input: " << page_dual_input.reqs.size()
                 << ", flash_dp_input: " << flash_input.dp_reqs.size()
                 << ", page_dp_single_input: " << page_single_input.dp_reqs.size()
                 << ", page_dp_dual_input: " << page_dual_input.dp_reqs.size();

  ProfileEvent::PushEvent("StartPrepareReqs", rank_);
  PrepareFlexibleCache(flash_input);
  CheckUseCache(forward_reqs);

  PrepareInputIds({&flash_input}, {&page_dual_input, &page_single_input});

  PrepareVLInputRefit(forward_reqs);
  PrepareInputRefit(forward_reqs);

  size_t gather_offset = 0;
  attn_dp_group_gather_offsets_.clear();
  for (size_t group_id = 0; group_id < attn_dp_group_size_; ++group_id) {
    size_t prefill_size = attn_dp_group_offsets_[group_id * 4 + 1] - attn_dp_group_offsets_[group_id * 4];
    size_t decode_size = attn_dp_group_offsets_[group_id * 4 + 3] - attn_dp_group_offsets_[group_id * 4 + 2];

    attn_dp_group_gather_offsets_.push_back(gather_offset);
    gather_offset += prefill_size;
    attn_dp_group_gather_offsets_.push_back(gather_offset);
    attn_dp_group_gather_offsets_.push_back(gather_offset);
    gather_offset += decode_size;
    attn_dp_group_gather_offsets_.push_back(gather_offset);
  }

  PrepareVLRequest(forward_reqs);
  PrepareNetxnGatherIdx(forward_reqs, run_mode);

  PreparePrefill();
  PrepareDualDecode();
  PrepareSingleDecode();

  PrepareMetadata();

#ifdef ENABLE_CUDA
  PrepareCudagraphParams(forward_reqs);
#endif

#ifdef ENABLE_ACL
  // NOTE(karlluo): please keep PrepareATBKVCache at the last of prepare process
  PrepareATBKVCache(forward_reqs, multi_token_request_num > 0);
#endif
  ProfileEvent::PopEvent();
}

void ModelInput::PrepareMetadata() {
  size_t meta_offset = 0;
  // index 0: 在input_ids中的offset
  page_single_input.metadata.GetPtr<size_t>()[meta_offset] =
      flash_input.total_dp_input_ids_len + page_dual_input.total_dp_input_ids_len;
  page_dual_input.metadata.GetPtr<size_t>()[meta_offset] = flash_input.total_dp_input_ids_len;

  // index 1: 单请求的固定input_ids长度
  ++meta_offset;
  page_single_input.metadata.GetPtr<size_t>()[meta_offset] = 1;
  page_dual_input.metadata.GetPtr<size_t>()[meta_offset] = 2;

  // index 2: 前面已有的qlen，用于mla里计算query offset muliti - 2 - 1 input_ids_cpu todo
  ++meta_offset;
  page_single_input.metadata.GetPtr<size_t>()[meta_offset] = page_dual_input.total_dp_input_ids_len;
  page_dual_input.metadata.GetPtr<size_t>()[meta_offset] = 0;
}

// TODO(ttsybyweng): VL_Model :Prepare moved into each Model Class
void ModelInput::PrepareVLInputRefit(const std::vector<ForwardRequest>& forward_reqs) {
  if (model_config_.type == "qwen2_vl") {
    PrepareMRopePos(forward_reqs);
  }
}

void ModelInput::PrepareVLRequest(const std::vector<ForwardRequest>& forward_reqs) {
  is_mask = false;
  if (model_config_.type == "internlmxcomposer2") {
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

void ModelInput::PrepareNetxnGatherIdx(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode) {
  std::vector<size_t> mtp_hidden_gather_idx;
  size_t total_len = 0;
  for (size_t i = 0; i < forward_reqs.size(); ++i) {
    const auto& req = forward_reqs[i];
    const size_t input_ids_len =
        req.forwarding_tokens->size() - std::max(req.kv_cached_token_num, req.prefix_cache_len);
    if (run_mode == RunMode::kMain) {  // record len before nextn
      mtp_req_id_to_pos_[req.req_id] = total_len;
      total_len += input_ids_len;
    } else {  // calc gather idx while nextn
      const size_t begin_idx = mtp_req_id_to_pos_[req.req_id];
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

int ModelInput::GetKoffsetInBlockLayer() {
  if (model_config_.mla_config.kv_lora_rank == 0) {
    // For normal kv.
    return 0;
  } else {
    // For MLA compress-kv.
    if (IsAbsorbWeightsEnabled()) {
      // Merge k&v, k will take all block space.
      return 0;
    } else {
      return 0;
    }
  }
}

int ModelInput::GetVoffsetInBlockLayer() {
  if (model_config_.mla_config.kv_lora_rank == 0) {
    // For normal kv.
    return block_size_ / layer_num_on_node_ / 2;
  } else {
    // For MLA compress-kv.
    if (IsAbsorbWeightsEnabled()) {
      // Merge k&v, k will take all block space, v is useless.
      return 0;
    } else {
      return block_size_ / layer_num_on_node_ / 2;
    }
  }
}

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

  for (size_t bs_idx = 0; bs_idx < multi_token_request_num; ++bs_idx) {
    const ForwardRequest& forward_req = forward_reqs[bs_idx];
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
    pos_offset += forward_req.forwarding_tokens->size();
  }

  cpu_input_refit_tensor.emb_fp32_ptr_tensor.shape = {cpu_input_refit_pos_pair_idx / 2};
  cpu_input_refit_tensor.pos_pair_tensor.shape = {cpu_input_refit_pos_pair_idx / 2, 2};
}

/**
 * The MRope position information (position and offset) of qwen2_vl is computed by the `_get_input_positions` function
 * in the Python plugin and is passed as additional tensors.
 * Before model inference, copy the position tensor (`additional_tensors[0]`) to the corresponding GPU tensor
 * (`dp_mrotary_embedding_pos`), and record the offset value (`additioanl_tensors[1]`).
 */
void ModelInput::PrepareMRopePos(const std::vector<ForwardRequest>& forward_reqs) {
  int64_t dp_mrotary_embedding_pos_size = 0;

  for (size_t bs_idx = 0; bs_idx < multi_token_request_num; ++bs_idx) {
    if (forward_reqs[bs_idx].attn_dp_group_id == attn_dp_group_id_) {
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
  // ResetPreAllocatedBlocks, block_managr's allocator's blocks_num is difference from the allocator's member config, so
  // we need get it from allocator instance.
  size_t total_block_num = Singleton<Environment>::GetInstance()->GetTotalDeviceBlockNum() * 2 * layer_num_on_node_;
  if (total_block_num != k_cache_blocks_base.shape[0]) {
    void* cur_rank_block_base_ptr = device_allocator->GetBlocksBasePtr();
    void* k_cache_base_ptr = cur_rank_block_base_ptr;
    void* v_cache_base_ptr = cur_rank_block_base_ptr + (block_size_ / 2);
    k_cache_blocks_base =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
               {Singleton<Environment>::GetInstance()->GetTotalDeviceBlockNum() * 2 * layer_num_on_node_,
                model_config_.block_token_num, model_config_.head_num, model_config_.size_per_head},
               rank_, k_cache_base_ptr);
    v_cache_blocks_base =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
               {Singleton<Environment>::GetInstance()->GetTotalDeviceBlockNum() * 2 * layer_num_on_node_,
                model_config_.block_token_num, model_config_.head_num, model_config_.size_per_head},
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
            int32_t inner_block_offset = token_idx % model_config_.block_token_num;
            layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + layers_slot_mapping_offset + token_idx] =
                (forward_reqs[f_req_idx]
                     .atb_kv_cache_base_blk_ids[attn_dp_rank_id_][token_idx / model_config_.block_token_num] +
                 layer_idx) *
                    model_config_.block_token_num +
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
                                                               model_config_.block_token_num];
          layers_slot_mapping_host[layer_idx * slot_mapping_dim_1 + f_req_idx] =
              (block_id + layer_idx) * model_config_.block_token_num +
              ((seq_len_host.GetPtr<int32_t>()[f_req_idx] - 1) % model_config_.block_token_num);
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
void ModelInput::CheckUseCache(const std::vector<ForwardRequest>& forward_reqs) {
  const auto& env = Singleton<Environment>::GetInstance();
  use_cache = env->IsPrefixCachingEnabled() || env->IsFlexibleCachingEnabled() || env->IsPrefillDecodeSeparation();

  if (attn_backend_config_.enable_blocked_multi_token_forwarding_kv) use_cache = true;

  if (use_cache) {
    return;
  }

  for (size_t i = 0; i < multi_token_request_num; i++) {
    if (forward_reqs[i].sampling_config == nullptr || forward_reqs[i].sampling_config->max_new_tokens != 1) {
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

    if (pos + len > input_ids.shape[0]) {
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

void ModelInput::PreparePageInput(input_info& input) {
  const auto reqs = input.dp_reqs;

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
  std::vector<void*> kv_list_host(layer_num_on_node_ * total_block_num * 2);
  const int k_offset_in_block_layer = GetKoffsetInBlockLayer();
  const int v_offset_in_block_layer = GetVoffsetInBlockLayer();
  for (size_t layer_i = 0; layer_i < layer_num_on_node_; ++layer_i) {
    const size_t cache_block_offset = layer_i * block_size_ / layer_num_on_node_;
    void** k_ptr = kv_list_host.data() + layer_i * total_block_num * 2;
    void** v_ptr = kv_list_host.data() + layer_i * total_block_num * 2 + total_block_num;
    for (size_t req_i = 0; req_i < reqs.size(); ++req_i) {
      for (const auto& cache_ptr : reqs[req_i]->kv_cache_ptrs[attn_dp_rank_id_]) {
        *k_ptr++ = cache_ptr + cache_block_offset + k_offset_in_block_layer;
        *v_ptr++ = cache_ptr + cache_block_offset + v_offset_in_block_layer;
      }
    }
  }

  KLLM_LOG_DEBUG << "kv_list_host " << kv_list_host;
  info.kv_list.shape = {static_cast<size_t>(layer_num_on_node_), static_cast<size_t>(total_block_num * 2)};
  MemcpyAsync(info.kv_list.GetPtr<void>(), kv_list_host.data(),
              kv_list_host.size() * sizeof(decltype(kv_list_host)::value_type), MEMCPY_HOST_TO_DEVICE,
              context_->GetD2HStreams()[rank_]);
#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
  EventRecord(kvcache_offset_event, context_->GetD2HStreams()[rank_]);
}

void ModelInput::PrepareDecodeRotary(input_info& input) {
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

  // preapare pos
  std::vector<int64_t> rotary_pos_host(total_input_len, 1);
  size_t rotary_data_offset = 0;
  for (size_t i = 0; i < input.dp_reqs.size(); ++i) {
    const auto& req = *input.dp_reqs[i];
    const size_t input_len = req.forwarding_tokens->size() - req.kv_cached_token_num;
    const auto pos_offset = model_config_.type == "qwen2_vl" ? *req.mrotary_embedding_pos_offset : 0;
    std::iota(rotary_pos_host.begin() + rotary_data_offset, rotary_pos_host.begin() + rotary_data_offset + input_len,
              req.kv_cached_token_num + pos_offset);
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
    if (attn_backend_config_.enable_blocked_multi_token_forwarding_kv) {
      std::iota(rotary_pos_host.begin() + rotary_host_idx,
                rotary_pos_host.begin() + rotary_host_idx + req->forwarding_tokens->size() - req->prefix_cache_len,
                req->prefix_cache_len);
      rotary_host_idx += req->forwarding_tokens->size() - req->prefix_cache_len;
    } else {
      // mask real prefix(exclude flexible cache), now eques kv_cached_token_num
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

  if (dp_dst_flexible_kv_cache_tensor.shape[0] != 0) {
    rotary_mask_host.assign(total_input_len, 0);
    rotary_pos_host.assign(total_input_len, 0);

    size_t flexible_rotary_idx = 0;
    for (const auto& req : input.dp_reqs) {
      if (req->flexible_cache_len > 0) {
        std::fill(rotary_mask_host.begin() + flexible_rotary_idx + req->prefix_cache_len - req->flexible_cache_len,
                  rotary_mask_host.begin() + flexible_rotary_idx + req->prefix_cache_len, 1);
        for (auto& task : *req->flexible_cached_copy_tasks) {
          rotary_pos_host[flexible_rotary_idx + task.dst_token_idx_] = task.src_token_idx_;
        }
      }
      flexible_rotary_idx += req->forwarding_tokens->size();
    }
    MemcpyAsync(dp_flexible_rotary_embedding_pos.GetPtr<void>(), rotary_pos_host.data(),
                sizeof(decltype(rotary_pos_host)::value_type) * flexible_rotary_idx, MEMCPY_HOST_TO_DEVICE,
                context_->GetD2HStreams()[rank_]);
    MemcpyAsync(dp_flexible_rotary_embedding_mask.GetPtr<void>(), rotary_mask_host.data(),
                sizeof(decltype(rotary_mask_host)::value_type) * flexible_rotary_idx, MEMCPY_HOST_TO_DEVICE,
                context_->GetD2HStreams()[rank_]);
  }
  EventRecord(rotary_embedding_event, context_->GetD2HStreams()[rank_]);
#ifdef ENABLE_ACL
  StreamSynchronize(context_->GetD2HStreams()[rank_]);
#endif
}

void ModelInput::PrepareKVCacheBlockTable(input_info& info) {
  if (!attn_backend_config_.enable_blocked_multi_token_forwarding_kv) {
    return;
  }

  const auto& reqs = info.dp_reqs;

  const int k_offset_in_block_layer = GetKoffsetInBlockLayer();
  const int v_offset_in_block_layer = GetVoffsetInBlockLayer();

  // Get each layer's raw pointer of k_cache and v_cache tensor from
  // kv_cache[num_blocks, num_layers, 2, block_size, num_kv_heads, head_size]
  // block_size is [num_layers, 2, block_size, num_kv_heads, head_size]
  const auto cache_manager = reqs.front()->cache_manager;
  const auto device_allocator = cache_manager->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(attn_dp_rank_id_);
  void* const k_cache_base_ptr = device_allocator->GetBlocksBasePtr() + k_offset_in_block_layer;
  void* const v_cache_base_ptr = device_allocator->GetBlocksBasePtr() + v_offset_in_block_layer;

  // first num is kv_cache_block_num, after store ptrs
  int64_t* const kv_cache_block_num = info.layer_kv_cache_ptr.GetPtr<int64_t>();
  *kv_cache_block_num = Singleton<Environment>::GetInstance()->GetTotalDeviceBlockNum() * layer_num_on_node_ * 2;

  const int block_size_per_layer = block_size_ / layer_num_on_node_;
  void** layer_kv_cache_ptr = reinterpret_cast<void**>(info.layer_kv_cache_ptr.GetPtr<int64_t>() + 1);
  for (int layer_idx = 0; layer_idx < layer_num_on_node_; ++layer_idx) {
    *layer_kv_cache_ptr++ = k_cache_base_ptr + layer_idx * block_size_per_layer;
    *layer_kv_cache_ptr++ = v_cache_base_ptr + layer_idx * block_size_per_layer;
  }
  info.layer_kv_cache_ptr.shape = {1ul + layer_num_on_node_ * 2};

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
      block_table_host[i * max_num_blocks_per_query + base_block_idx] = block_ids[base_block_idx];
    }
  }

  if (IsAbsorbWeightsEnabled()) {
    *kv_cache_block_num = *kv_cache_block_num / 2;
    for (auto& i : block_table_host) {
      i = i / 2;
    }
  }
  KLLM_LOG_DEBUG << "block_table_host " << block_table_host;
  if (Singleton<Environment>::GetInstance()->GetKVCacheType() == DataType::TYPE_FP8_E5M2 ||
      Singleton<Environment>::GetInstance()->GetKVCacheType() == DataType::TYPE_FP8_E4M3) {
    size_t block_table_host_size = block_table_host.size();
    block_table_host.resize(block_table_host.size() * 2);
    for (size_t i = 0; i < block_table_host_size; i++) {
      block_table_host[block_table_host_size + i] = block_table_host[i] / layer_num_on_node_;
    }
    KLLM_LOG_DEBUG << "block_table_host " << block_table_host;
  }
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
  if (!enable_flash_mla_ || model_config_.mla_config.kv_lora_rank == 0 || input.dp_reqs.empty()) {
    return;
  }

  Stream stream = context_->GetH2DStreams()[rank_];
  const int head_num = model_config_.head_num;
  const int tensor_para_size = model_config_.tensor_para_size;
  const int head_num_per_tp = head_num / tensor_para_size;
  llm_kernels::nvidia::FlashMlaWorkspaceMap flash_mla_workspace_map;
  GetNumSmParts(flash_mla_workspace_map, head_num_per_tp, 1, rank_, stream.Get());
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
  ProfileEvent::PushEvent("PrepareSingleDecode", rank_);
  PreparePageInput(page_single_input);
  PrepareKVCacheBlocks(page_single_input);
  PrepareKVCacheBlockTable(page_single_input);
  PrepareDecodeRotary(page_single_input);
  PrepareFlashMla(page_single_input);
  ProfileEvent::PopEvent();
}

void ModelInput::PrepareDualDecode() {
  if (page_dual_input.dp_reqs.empty()) {
    KLLM_LOG_DEBUG << "page_dual_input empty";
    return;
  }
  ProfileEvent::PushEvent("PrepareDualDecode", rank_);
  PreparePageInput(page_dual_input);
  PrepareKVCacheBlocks(page_dual_input);
  PrepareKVCacheBlockTable(page_dual_input);
  PrepareDecodeRotary(page_dual_input);
  PrepareFlashMla(page_dual_input);
  ProfileEvent::PopEvent();
}

void ModelInput::PreparePrefill() {
  if (flash_input.dp_reqs.empty()) {
    KLLM_LOG_DEBUG << "flash_input empty";
    return;
  }

  ProfileEvent::PushEvent("PreparePrefill", rank_);
  if (use_cache) {
    PrepareKVCacheBlocks(flash_input);
    PrepareKVCacheBlockTable(flash_input);
  }

  PrepareFlashRotary(flash_input);
  ProfileEvent::PopEvent();
}

void ModelInput::PrepareFlexibleCache(input_info& input) {
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

  if (dst_flexible_kv_cache_id_cpu.empty()) {
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

void ModelInput::PrepareInputIds(const std::initializer_list<input_info*>& flash_inputs,
                                 const std::initializer_list<input_info*>& page_inputs) {
  input_ids_cpu.clear();
  input_offset_list_uint64.assign(1, 0);
  input_prefix_list_uint64.assign(1, 0);
  dp_input_offset_list_uint64.assign(1, 0);
  dp_input_prefix_list_uint64.assign(1, 0);
  dp_input_without_prefix_list_uint64.assign(1, 0);
  multi_token_request_max_tokens = 0;
  single_token_request_max_tokens = 0;
  dp_multi_token_request_max_tokens = 0;
  dp_single_token_request_max_tokens = 0;
  dp_max_forwarding_tokens = 0;  // used for blocked_prefill

  std::vector<size_t> logits_idx_list(total_sampling_token_num_);
  size_t logits_idx_list_idx = 0;
  std::vector<size_t> dp_prefill_q_offset(1, 0);
  std::vector<size_t> group_prefill_count(attn_dp_group_size_, 0);
  std::vector<size_t> group_decode_count(attn_dp_group_size_, 0);

  auto process_func = [&](const ForwardRequest& req, const bool is_page) {
    const auto& forwarding_tokens = *(req.forwarding_tokens);
    const size_t input_length = forwarding_tokens.size();
    const bool in_dp_group = req.attn_dp_group_id == attn_dp_group_id_;

    // Skip prefix token(include flexible cache token)
    const size_t skip_token_num = std::max(req.kv_cached_token_num, req.prefix_cache_len);
    const size_t input_ids_len = input_length - skip_token_num;
    input_ids_cpu.insert(input_ids_cpu.end(), forwarding_tokens.begin() + skip_token_num, forwarding_tokens.end());
    dp_max_forwarding_tokens = std::max(dp_max_forwarding_tokens, input_ids_len);
    if (attn_backend_config_.enable_blocked_multi_token_forwarding_kv && in_dp_group) {
      dp_input_without_prefix_list_uint64.emplace_back(dp_input_without_prefix_list_uint64.back() + input_ids_len);
    }

    if (!is_page) {
      multi_token_request_max_tokens = std::max(multi_token_request_max_tokens, input_length);
      input_offset_list_uint64.emplace_back(input_offset_list_uint64.back() + input_length);
      input_prefix_list_uint64.emplace_back(input_prefix_list_uint64.back() + skip_token_num);
      dp_multi_token_request_max_tokens = std::max(dp_multi_token_request_max_tokens, in_dp_group ? input_length : 0);
      group_prefill_count[req.attn_dp_group_id] += input_ids_len;
      if (in_dp_group) {
        dp_prefill_q_offset.emplace_back(dp_prefill_q_offset.back() + input_ids_len);
        dp_input_offset_list_uint64.emplace_back(dp_input_offset_list_uint64.back() + input_length);
        dp_input_prefix_list_uint64.emplace_back(dp_input_prefix_list_uint64.back() + skip_token_num);
      }
    } else {
      single_token_request_max_tokens = std::max(single_token_request_max_tokens, input_length);
      input_offset_list_uint64.emplace_back(input_offset_list_uint64.back() + input_ids_len);
      input_prefix_list_uint64.emplace_back(input_prefix_list_uint64.back());
      dp_single_token_request_max_tokens = std::max(dp_single_token_request_max_tokens, in_dp_group ? input_length : 0);
      group_decode_count[req.attn_dp_group_id] += input_ids_len;
      if (in_dp_group) {
        dp_input_offset_list_uint64.emplace_back(dp_input_offset_list_uint64.back() + 1);
        dp_input_prefix_list_uint64.emplace_back(dp_input_prefix_list_uint64.back());
      }
    }

    if (!is_page && req.logits_custom_length > 0) {  // Specify the range of logits required
      for (auto [l, r] : req.request_target->at("logits").slice_pos) {
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

  // process flash attention input
  for (const auto& flash_input : flash_inputs) {
    for (const auto& req : flash_input->reqs) {
      process_func(*req, false);
    }
  }

  // process page attention input
  for (const auto& page_input : page_inputs) {
    for (const auto& req : page_input->reqs) {
      process_func(*req, true);
    }
  }

  KLLM_LOG_DEBUG << "input_ids_cpu " << input_ids_cpu;
  KLLM_LOG_DEBUG << "logits_idx_list " << logits_idx_list;
  KLLM_LOG_DEBUG << "input_offset_list_uint64 " << input_offset_list_uint64;
  KLLM_LOG_DEBUG << "input_prefix_list_uint64 " << input_prefix_list_uint64;
  KLLM_LOG_DEBUG << "dp_input_offset_list_uint64 " << dp_input_offset_list_uint64;
  KLLM_LOG_DEBUG << "dp_input_prefix_list_uint64 " << dp_input_prefix_list_uint64;
  KLLM_LOG_DEBUG << "dp_input_without_prefix_list_uint64 " << dp_input_without_prefix_list_uint64;
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

  if (attn_backend_config_.enable_blocked_multi_token_forwarding_kv) {
    dp_input_without_prefix_uint64_tensor.shape = {dp_input_without_prefix_list_uint64.size()};
    dp_input_without_prefix_uint64_tensor.dtype = TYPE_UINT64;
    MemcpyAsync(
        dp_input_without_prefix_uint64_tensor.GetPtr<void>(), dp_input_without_prefix_list_uint64.data(),
        dp_input_without_prefix_list_uint64.size() * sizeof(decltype(dp_input_without_prefix_list_uint64)::value_type),
        MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);
  }

  size_t prefill_offset = 0, decode_offset = std::accumulate(group_prefill_count.begin(), group_prefill_count.end(), 0);
  auto* attn_dp_group_offsets_ptr = attn_dp_group_offsets_.data();
  for (size_t i = 0; i < attn_dp_group_size_; ++i) {
    *attn_dp_group_offsets_ptr++ = prefill_offset;
    *attn_dp_group_offsets_ptr++ = prefill_offset + group_prefill_count[i];
    *attn_dp_group_offsets_ptr++ = decode_offset;
    *attn_dp_group_offsets_ptr++ = decode_offset + group_decode_count[i];
    prefill_offset += group_prefill_count[i];
    decode_offset += group_decode_count[i];
  }
  KLLM_LOG_DEBUG << "attn_dp_group_offsets_ " << attn_dp_group_offsets_;

  EventRecord(input_ids_event, context_->GetH2DStreams()[rank_]);
#ifdef ENABLE_ACL
  // Event wait between streams seems not work, force sync here.
  StreamSynchronize(context_->GetH2DStreams()[rank_]);
#endif
}

}  // namespace ksana_llm
