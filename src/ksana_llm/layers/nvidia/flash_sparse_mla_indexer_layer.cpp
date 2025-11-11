/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_sparse_mla_indexer_layer.h"

#include <fmt/core.h>
#include <torch/torch.h>
#include <vector>

#include "ksana_llm/layers/attention_layer.h"  // For yarn mscale functions and InitYarnRotaryEmbedding
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"

#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "csrc/kernels/nvidia/weight_scale/weight_scale_kernel.h"
#include "ksana_llm/kernels/nvidia/basic_kernel_wrapper.h"

#include "ksana_llm/kernels/nvidia/deepseek_deepgemm_wrapper.h"

#include "csrc/kernels/nvidia/logits_topk/logits_topk.h"

namespace ksana_llm {

// 解析输入张量的辅助结构
template <typename SCALAR_T>
struct FlashMlaInputData {
  const Tensor& q_tensor;
  const Tensor& k_tensor;
  const Tensor& weights_tensor;
  const Tensor& rotary_pos;
  const Tensor& rotary_mask;
  const Tensor& indexer_kv_list_tensor;
  const Tensor& indexer_kv_cache_offset_tensor;
  const Tensor& seqlens_without_prefix_tensor;
  const Tensor& prefix_offsets_tensor;
  const Tensor* block_table_tensor;
  const Tensor* cur_seq_len_start_tensor;
  const Tensor* cur_seq_len_end_tensor;
  const Tensor* layer_indexer_kv_cache_ptr_tensor;

  size_t total_tokens;
  int batch_size;
  int64_t prefill_token_count;
  cudaStream_t stream;
};

// FP8 量化工作空间布局
struct FlashQuantWorkspace {
  uint8_t* q_fp8_ptr;
  float* q_scale_ptr;
  uint8_t* k_fp8_ptr;
  float* k_scale_ptr;
  float* logits_ptr;
  int aligned_seq_len;
  int aligned_seq_len_kv;
};

Status FlashSparseMlaIndexerLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                        std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  // Parse parameters
  if (parameters.size() < 14) {
    KLLM_THROW("FlashSparseMlaIndexerLayer requires at least 14 parameters");
  }

  dim_ = std::any_cast<int>(parameters[0]);
  n_heads_ = std::any_cast<int>(parameters[1]);
  head_dim_ = std::any_cast<int>(parameters[2]);
  rope_head_dim_ = std::any_cast<int>(parameters[3]);
  index_topk_ = std::any_cast<int>(parameters[4]);
  block_size_ = std::any_cast<int>(parameters[5]);
  quant_block_size_ = std::any_cast<int>(parameters[6]);

  softmax_scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  // Initialize RoPE
  PositionEncoding position_encoding = std::any_cast<PositionEncoding>(parameters[7]);
  if (position_encoding != PositionEncoding::ROPE) {
    KLLM_THROW(fmt::format("FlashSparseMlaIndexer only supports ROPE, got {}", position_encoding));
  }

  // Initialize RoPE using YARN scaling (only for DeepSeek models)
  float rope_theta = std::any_cast<float>(parameters[8]);
  void* cos_sin_cache_ptr = std::any_cast<void*>(parameters[9]);
  auto rope_scaling_factor_config = std::any_cast<RoPEScalingFactor>(parameters[10]);
  max_batch_size_ = std::any_cast<int>(parameters[11]);
  max_position_embeddings_ = std::any_cast<int>(parameters[12]);
  layer_index_ = std::any_cast<int>(parameters[13]);
  max_seq_len_ = runtime_config.max_seq_len;
  max_step_token_num_ = runtime_config.max_step_token_num;  // Save max_step_token_num from runtime_config
  KLLM_CHECK_WITH_INFO(rope_scaling_factor_config.type == "yarn", "Only YARN scaling is supported for indexer");

  // Initialize RoPE based on data type
  DISPATCH_BY_3_DTYPE(inter_data_type_, InitYarnRotaryEmbedding, rotary_embedding_cuda_, rope_scaling_factor_config,
                      cos_sin_cache_ptr, rope_theta, rope_head_dim_, max_position_embeddings_, head_dim_, n_heads_,
                      context_->GetComputeStreams()[rank_].Get());

  KLLM_LOG_DEBUG << fmt::format(
      "FlashSparseMlaIndexerLayer initialized: dim={}, n_heads={}, "
      "head_dim={}, rope_head_dim={}, index_topk={}, block_size={}",
      dim_, n_heads_, head_dim_, rope_head_dim_, index_topk_, block_size_);
  return Status();
}

Status FlashSparseMlaIndexerLayer::Forward(const std::vector<Tensor>& input_tensors,
                                           std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_2_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename SCALAR_T>
static FlashMlaInputData<SCALAR_T> ParseInputTensors(const std::vector<Tensor>& input_tensors,
                                                     std::shared_ptr<Context> context, int rank) {
  if (input_tensors.size() < 13) {
    KLLM_THROW(
        fmt::format("FlashSparseMlaIndexerLayer requires at least 13 input tensors, got {}", input_tensors.size()));
  }
  auto input_iter = input_tensors.cbegin();

  // 提取所有引用类型的 tensor
  const Tensor& q_tensor = *input_iter++;                              // 0: query
  const Tensor& k_tensor = *input_iter++;                              // 1: key
  const Tensor& weights_tensor = *input_iter++;                        // 2: weights
  const Tensor& rotary_pos = *input_iter++;                            // 3: rotary pos
  const Tensor& rotary_mask = *input_iter++;                           // 4: rotary mask
  const Tensor& indexer_kv_list_tensor = *input_iter++;                // 5: indexer_kv_list
  const Tensor& indexer_kv_cache_offset_tensor = *input_iter++;        // 6: indexer_kv_cache_offset
  const Tensor& seqlens_without_prefix_tensor = *input_iter++;         // 7: seqlens_without_prefix
  const Tensor& prefix_offsets_tensor = *input_iter++;                 // 8: prefix_offsets
  const Tensor* block_table_tensor = &(*input_iter++);                 // 9: block_table
  const Tensor* cur_seq_len_start_tensor = &(*input_iter++);           // 10: cur_seq_len_start
  const Tensor* cur_seq_len_end_tensor = &(*input_iter++);             // 11: cur_seq_len_end
  const Tensor* layer_indexer_kv_cache_ptr_tensor = &(*input_iter++);  // 12: layer_indexer_kv_cache_ptr

  // 使用聚合初始化创建对象
  FlashMlaInputData<SCALAR_T> data{.q_tensor = q_tensor,
                                   .k_tensor = k_tensor,
                                   .weights_tensor = weights_tensor,
                                   .rotary_pos = rotary_pos,
                                   .rotary_mask = rotary_mask,
                                   .indexer_kv_list_tensor = indexer_kv_list_tensor,
                                   .indexer_kv_cache_offset_tensor = indexer_kv_cache_offset_tensor,
                                   .seqlens_without_prefix_tensor = seqlens_without_prefix_tensor,
                                   .prefix_offsets_tensor = prefix_offsets_tensor,
                                   .block_table_tensor = block_table_tensor,
                                   .cur_seq_len_start_tensor = cur_seq_len_start_tensor,
                                   .cur_seq_len_end_tensor = cur_seq_len_end_tensor,
                                   .layer_indexer_kv_cache_ptr_tensor = layer_indexer_kv_cache_ptr_tensor,
                                   .total_tokens = q_tensor.shape[0],
                                   .batch_size = static_cast<int>(seqlens_without_prefix_tensor.shape[0] - 1),
                                   .prefill_token_count = static_cast<int64_t>(q_tensor.shape[0]),
                                   .stream = context->GetComputeStreams()[rank].Get()};

  return data;
}

template <typename SCALAR_T>
static Status ApplyRoPE(SCALAR_T* q_ptr, SCALAR_T* k_ptr, const Tensor& rotary_pos, const Tensor& rotary_mask,
                        int64_t prefill_token_count, cudaStream_t stream,
                        std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda, int n_heads,
                        int head_dim) {
  if (!rotary_embedding_cuda.has_value() || prefill_token_count == 0 || rotary_pos.GetElementNumber() == 0) {
    return Status(RET_RUNTIME_FAILED, "RoPE not initialized or invalid input");
  }

  // Q layout: [token_num, head_num, rope_dim + nope_dim]
  // K layout: [token_num, rope_dim + nope_dim]
  SCALAR_T* q_rope_ptr = q_ptr;  // rope_dim comes first
  SCALAR_T* k_rope_ptr = k_ptr;  // rope_dim comes first

  rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_pos.template GetPtr<void>()),
                                  reinterpret_cast<int64_t*>(rotary_mask.template GetPtr<void>()),
                                  q_rope_ptr,           // query: pointer to Q's rope portion (in-place)
                                  k_rope_ptr,           // key: pointer to K's rope portion (in-place)
                                  prefill_token_count,  // num_tokens: prefill token count
                                  stream,
                                  n_heads * head_dim,  // query_stride: total elements per token for Q
                                  head_dim,            // key_stride: total elements per token for K
                                  head_dim,            // query_head_size
                                  head_dim);           // key_head_size
  CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward<SCALAR_T>());
  KLLM_LOG_DEBUG << "RoPE computation completed";

  return Status();
}

static void PrepareQuantWorkspace(size_t total_tokens, FlashQuantWorkspace& workspace,
                                  std::shared_ptr<Tensor> workspace_buffer, int n_heads, int head_dim,
                                  int quant_block_size, int block_size) {
  // Calculate FP8 quantization buffer sizes
  size_t q_fp8_bytes = total_tokens * n_heads * head_dim;
  size_t q_scale_groups = DivRoundUp(head_dim, quant_block_size);
  KLLM_CHECK_WITH_INFO(q_scale_groups == 1,
                       fmt::format("q_scale_groups must be 1 for weight scaling, got {}", q_scale_groups));
  size_t q_scale_bytes = total_tokens * n_heads * q_scale_groups * sizeof(float);
  size_t k_fp8_bytes = total_tokens * head_dim;
  size_t k_scale_bytes = total_tokens * q_scale_groups * sizeof(float);

  // Calculate logits space requirements for DeepGEMM
  const int block_kv = 256;
  const int seq_len_alignment = 4;
  const int aligned_seq_len = RoundUp(static_cast<int>(total_tokens), seq_len_alignment);
  const int aligned_seq_len_kv = RoundUp(static_cast<int>(total_tokens) + block_kv, seq_len_alignment);
  size_t logits_bytes = static_cast<size_t>(aligned_seq_len) * static_cast<size_t>(aligned_seq_len_kv) * sizeof(float);

  // Calculate offsets with 256-byte alignment
  size_t quant_data_bytes = q_fp8_bytes + q_scale_bytes + k_fp8_bytes + k_scale_bytes;
  size_t logits_offset = RoundUp(quant_data_bytes, static_cast<size_t>(256));
  size_t total_workspace_bytes = logits_offset + logits_bytes;

  KLLM_CHECK_WITH_INFO(
      workspace_buffer->GetTotalBytes() >= total_workspace_bytes,
      fmt::format(
          "workspace_buffer_ not large enough. Required: {} (quant_data: {} + padding: {} + logits: {}), Available: {}",
          total_workspace_bytes, quant_data_bytes, logits_offset - quant_data_bytes, logits_bytes,
          workspace_buffer->GetTotalBytes()));

  // Setup workspace pointers
  uint8_t* workspace_ptr = workspace_buffer->GetPtr<uint8_t>();
  workspace.q_fp8_ptr = workspace_ptr;
  workspace.q_scale_ptr = reinterpret_cast<float*>(workspace_ptr + q_fp8_bytes);
  workspace.k_fp8_ptr = workspace_ptr + q_fp8_bytes + q_scale_bytes;
  workspace.k_scale_ptr = reinterpret_cast<float*>(workspace_ptr + q_fp8_bytes + q_scale_bytes + k_fp8_bytes);
  workspace.logits_ptr = reinterpret_cast<float*>(workspace_ptr + logits_offset);
  workspace.aligned_seq_len = aligned_seq_len;
  workspace.aligned_seq_len_kv = aligned_seq_len_kv;
}

template <typename SCALAR_T>
static Status QuantizeQK(SCALAR_T* q_ptr, SCALAR_T* k_ptr, FlashQuantWorkspace& workspace, size_t total_tokens,
                         cudaStream_t stream, int n_heads, int head_dim, int quant_block_size) {
  // Quantize Q: [total_tokens, n_heads, head_dim]
  InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(q_ptr, workspace.q_fp8_ptr, workspace.q_scale_ptr, total_tokens * n_heads,
                                            head_dim,
                                            /*is_column_major=*/false, stream, quant_block_size);

  // Quantize K: [total_tokens, head_dim]
  InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(k_ptr, workspace.k_fp8_ptr, workspace.k_scale_ptr, total_tokens, head_dim,
                                            /*is_column_major=*/false, stream, quant_block_size);

  KLLM_LOG_DEBUG << "FP8 quantization completed";
  return Status();
}

template <typename SCALAR_T>
static Status ApplyWeightScaling(SCALAR_T* weights_ptr, FlashQuantWorkspace& workspace, size_t total_tokens,
                                 cudaStream_t stream, int n_heads, float softmax_scale) {
  // Apply weight scaling: weights = weights * n_heads**-0.5 * q_scale * softmax_scale
  // Result is stored back in q_scale_ptr (reusing memory)
  float n_heads_inv_sqrt = 1.0f / std::sqrt(static_cast<float>(n_heads));
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokeWeightScale<SCALAR_T>(weights_ptr, workspace.q_scale_ptr, n_heads_inv_sqrt,
                                                       softmax_scale, static_cast<int>(total_tokens), n_heads, stream));
  KLLM_LOG_DEBUG << "Weight scaling completed";

  return Status();
}

template <typename SCALAR_T>
static Status StoreKVCache(const FlashMlaInputData<SCALAR_T>& input_data, FlashQuantWorkspace& workspace,
                           int layer_index, int block_size, int head_dim) {
  // Store K/V to cache
  const size_t layer_block_num = input_data.indexer_kv_list_tensor.shape[1] / 2;
  const size_t row_stride = input_data.indexer_kv_list_tensor.shape[1];
  void** indexer_kv_list_base = input_data.indexer_kv_list_tensor.template GetPtr<void*>();
  void** indexer_k_list = indexer_kv_list_base + (layer_index * row_stride);
  void** indexer_v_list = indexer_k_list + layer_block_num;

  const int k_stride_size = head_dim;
  const int v_stride_size = 1;  // k_scale_groups = 1

  KLLM_LOG_DEBUG << fmt::format(
      "MlaIndexerFlashKVCacheCopy params: layer_index={}, layer_block_num={}, row_stride={}, "
      "batch_size={}, total_tokens={}, block_size={}, k_stride={}, v_stride={}",
      layer_index, layer_block_num, row_stride, input_data.batch_size, input_data.total_tokens, block_size,
      k_stride_size, v_stride_size);

  KLLM_CHECK_WITH_INFO(indexer_k_list != nullptr && indexer_v_list != nullptr, "indexer k/v list pointers are null");
  KLLM_CHECK_WITH_INFO(
      layer_index < static_cast<int>(input_data.indexer_kv_list_tensor.shape[0]),
      fmt::format("layer_index {} >= tensor shape[0] {}", layer_index, input_data.indexer_kv_list_tensor.shape[0]));

  // Store K and V to KV cache
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaIndexerFlashKVCacheCopy(
      reinterpret_cast<__nv_fp8_e4m3*>(workspace.k_fp8_ptr), workspace.k_scale_ptr, indexer_k_list, indexer_v_list,
      input_data.prefix_offsets_tensor.template GetPtr<size_t>(),
      input_data.seqlens_without_prefix_tensor.template GetPtr<size_t>(),
      input_data.indexer_kv_cache_offset_tensor.template GetPtr<int>(), static_cast<int>(block_size),
      input_data.batch_size, static_cast<int>(input_data.total_tokens), k_stride_size, v_stride_size,
      input_data.stream));
  KLLM_LOG_DEBUG << "KV cache copy completed";

  return Status();
}

template <typename SCALAR_T>
static Status ComputeMqaLogits(const FlashMlaInputData<SCALAR_T>& input_data, FlashQuantWorkspace& workspace, int rank,
                               int n_heads, int head_dim) {
  const int seq_len = static_cast<int>(input_data.total_tokens);
  const int seq_len_kv = static_cast<int>(input_data.total_tokens);

  KLLM_CHECK_WITH_INFO(input_data.cur_seq_len_start_tensor != nullptr && input_data.cur_seq_len_end_tensor != nullptr,
                       "cur_seq_len tensors must be provided from model input");
  KLLM_CHECK_WITH_INFO(input_data.cur_seq_len_start_tensor->shape[0] == static_cast<size_t>(seq_len),
                       fmt::format("cur_seq_len_start shape mismatch: {} vs {}",
                                   input_data.cur_seq_len_start_tensor->shape[0], seq_len));
  KLLM_CHECK_WITH_INFO(
      input_data.cur_seq_len_end_tensor->shape[0] == static_cast<size_t>(seq_len),
      fmt::format("cur_seq_len_end shape mismatch: {} vs {}", input_data.cur_seq_len_end_tensor->shape[0], seq_len));

  // Prepare aligned dimensions for DeepGEMM
  const int seq_len_alignment = 4;
  const int block_kv = 256;
  const int aligned_seq_len = RoundUp(seq_len, seq_len_alignment);
  const int aligned_seq_len_kv = RoundUp(seq_len_kv + block_kv, seq_len_alignment);
  auto device = torch::Device(torch::kCUDA, rank);

  // Create torch tensors from workspace pointers
  auto q_torch = torch::from_blob(const_cast<uint8_t*>(workspace.q_fp8_ptr),
                                  std::vector<int64_t>{static_cast<int64_t>(seq_len), static_cast<int64_t>(n_heads),
                                                       static_cast<int64_t>(head_dim)},
                                  torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));

  auto weights_torch = torch::from_blob(
      workspace.q_scale_ptr, std::vector<int64_t>{static_cast<int64_t>(seq_len), static_cast<int64_t>(n_heads)},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));

  auto cur_seq_len_start_torch =
      torch::from_blob(const_cast<int32_t*>(input_data.cur_seq_len_start_tensor->template GetPtr<int32_t>()),
                       {static_cast<int64_t>(seq_len)}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  auto cur_seq_len_end_torch =
      torch::from_blob(const_cast<int32_t*>(input_data.cur_seq_len_end_tensor->template GetPtr<int32_t>()),
                       {static_cast<int64_t>(seq_len)}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  auto logits_torch_aligned = torch::from_blob(
      workspace.logits_ptr,
      std::vector<int64_t>{static_cast<int64_t>(aligned_seq_len), static_cast<int64_t>(aligned_seq_len_kv)},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));

  auto k_fp8_torch = torch::from_blob(const_cast<uint8_t*>(workspace.k_fp8_ptr),
                                      {static_cast<int64_t>(seq_len_kv), static_cast<int64_t>(head_dim)},
                                      torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));

  auto k_scale_torch = torch::from_blob(workspace.k_scale_ptr, {static_cast<int64_t>(seq_len_kv)},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(device));

  auto kv_fp8 = std::make_pair(k_fp8_torch, k_scale_torch);

  // Call DeepGEMM wrapper
  auto wrapper = ksana_llm::nvidia::DeepSeekDeepGEMMWrapper::GetInstance(rank);
  wrapper->Fp8MqaLogits(q_torch, kv_fp8, weights_torch, cur_seq_len_start_torch, cur_seq_len_end_torch,
                       logits_torch_aligned,
                       /*clean_logits=*/true);

  return Status();
}

template <typename SCALAR_T>
static Status SelectTopK(const FlashMlaInputData<SCALAR_T>& input_data, const FlashQuantWorkspace& workspace,
                         Tensor& topk_indices, int index_topk) {
  KLLM_CHECK_WITH_INFO(input_data.cur_seq_len_start_tensor != nullptr && input_data.cur_seq_len_end_tensor != nullptr,
                       "cur_seq_len tensors are required for TopK selection");

  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeFastTopK(
      workspace.logits_ptr,                                             // logits
      input_data.cur_seq_len_start_tensor->template GetPtr<int32_t>(),  // cur_seq_len_start
      input_data.cur_seq_len_end_tensor->template GetPtr<int32_t>(),    // cur_seq_len_end
      topk_indices.template GetPtr<int32_t>(),                          // topk_indices
      input_data.total_tokens,                                          // total_tokens
      static_cast<int64_t>(workspace.aligned_seq_len_kv),               // aligned_seq_len_kv
      static_cast<int64_t>(1),                                          // topk
      input_data.stream));                                              // stream

  KLLM_LOG_DEBUG << fmt::format("TopK selection completed for total_tokens={}, topk={}, seq_len_kv={}",
                                input_data.total_tokens, index_topk, input_data.total_tokens);

  return Status();
}

template <typename SCALAR_T>
Status FlashSparseMlaIndexerLayer::ForwardT(const std::vector<Tensor>& input_tensors,
                                            std::vector<Tensor>& output_tensors) {
  // Step 1: Parse input tensors
  FlashMlaInputData<SCALAR_T> input_data = ParseInputTensors<SCALAR_T>(input_tensors, context_, rank_);
  Tensor& topk_indices = output_tensors[0];

  KLLM_LOG_DEBUG << fmt::format("FlashSparseMlaIndexerLayer Forward: total_tokens={}, prefill_tokens={}",
                                input_data.total_tokens, input_data.prefill_token_count);

  // Get pointers
  SCALAR_T* q_ptr = input_data.q_tensor.template GetPtr<SCALAR_T>();
  SCALAR_T* k_ptr = input_data.k_tensor.template GetPtr<SCALAR_T>();
  SCALAR_T* weights_ptr = input_data.weights_tensor.template GetPtr<SCALAR_T>();

  // Step 2: Apply RoPE to Q and K
  ApplyRoPE<SCALAR_T>(q_ptr, k_ptr, input_data.rotary_pos, input_data.rotary_mask, input_data.prefill_token_count,
                      input_data.stream, rotary_embedding_cuda_, n_heads_, head_dim_);

  // Step 3: Prepare workspace
  FlashQuantWorkspace workspace;
  PrepareQuantWorkspace(input_data.total_tokens, workspace, workspace_buffer_, n_heads_, head_dim_, quant_block_size_,
                        block_size_);

  // Step 4: Quantize Q/K
  QuantizeQK<SCALAR_T>(q_ptr, k_ptr, workspace, input_data.total_tokens, input_data.stream, n_heads_, head_dim_,
                       quant_block_size_);

  // Step 5: Store KV to cache (only the storage part, quantization is already done)
  StoreKVCache<SCALAR_T>(input_data, workspace, layer_index_, block_size_, head_dim_);

  // Step 6: Apply weight scaling
  ApplyWeightScaling<SCALAR_T>(weights_ptr, workspace, input_data.total_tokens, input_data.stream, n_heads_,
                               softmax_scale_);

  // Step 7: Compute MQA logits
  ComputeMqaLogits<SCALAR_T>(input_data, workspace, rank_, n_heads_, head_dim_);

  // Step 8: TopK selection
  SelectTopK<SCALAR_T>(input_data, workspace, topk_indices, index_topk_);

  return Status();
}

size_t FlashSparseMlaIndexerLayer::GetWorkspaceSize() {
  // Calculate workspace size needed for FP8 quantization and logits (with alignment)
  // Layout: [q_fp8 | q_scale | k_fp8 | k_scale | padding | logits(aligned to 256 bytes)]
  // Use max_step_token_num_ from runtime_config for more accurate calculation

  const size_t max_total_tokens = max_step_token_num_;

  // FP8 quantization workspace requirements:
  // 1. Q FP8 data: [max_total_tokens, n_heads_, head_dim_] * sizeof(uint8_t)
  size_t q_fp8_bytes = max_total_tokens * n_heads_ * head_dim_;

  // 2. Q scale data: [max_total_tokens, n_heads_] * sizeof(float)
  // (assuming q_scale_groups = 1 based on the check in ForwardT)
  size_t q_scale_bytes = max_total_tokens * n_heads_ * sizeof(float);

  // 3. K FP8 data: [max_total_tokens, head_dim_] * sizeof(uint8_t)
  size_t k_fp8_bytes = max_total_tokens * head_dim_;

  // 4. K scale data: [max_total_tokens] * sizeof(float)
  size_t k_scale_bytes = max_total_tokens * sizeof(float);

  // 5. Logits data: [aligned_seq_len, aligned_seq_len_kv] * sizeof(float), aligned to 256 bytes
  const int block_kv = 256;
  const int seq_len_alignment = 4;
  const int aligned_seq_len = RoundUp(static_cast<int>(max_total_tokens), seq_len_alignment);
  const int aligned_seq_len_kv = RoundUp(static_cast<int>(max_total_tokens) + block_kv, seq_len_alignment);
  size_t logits_bytes = static_cast<size_t>(aligned_seq_len) * static_cast<size_t>(aligned_seq_len_kv) * sizeof(float);

  // Calculate total size with 256-byte alignment for logits (same as ForwardT)
  size_t quant_data_bytes = q_fp8_bytes + q_scale_bytes + k_fp8_bytes + k_scale_bytes;
  size_t logits_offset = RoundUp(quant_data_bytes, static_cast<size_t>(256));  // Align to 256 bytes
  size_t padding_bytes = logits_offset - quant_data_bytes;
  size_t total_workspace_size = logits_offset + logits_bytes;

  KLLM_LOG_DEBUG << fmt::format(
      "FlashSparseMlaIndexerLayer GetWorkspaceSize: q_fp8={}, q_scale={}, k_fp8={}, k_scale={}, padding={}, logits={}, "
      "total={}",
      q_fp8_bytes, q_scale_bytes, k_fp8_bytes, k_scale_bytes, padding_bytes, logits_bytes, total_workspace_size);

  return total_workspace_size;
}

}  // namespace ksana_llm
