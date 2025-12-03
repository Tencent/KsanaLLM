/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "flashinfer_ops.cuh"
#include "flashinfer_prefill.h"

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

namespace {
constexpr int kThreadsPerBlock = 64;
constexpr size_t kByteAlignedSize = 16;
constexpr size_t kInitWorkspaceSize = 8 * 1024 * 1024;  // 8MB
}  // namespace

// According to FlashInfer's worspace usage, the float workspace size is computed as follows:
// tmp_v_device_size_aligned = align16(num_heads * max_batch_size_if_split * cta_tile_q * head_dim * sizeof(float))
// tmp_s_device_size_aligned = align16(num_heads * max_batch_size_if_split * cta_tile_q * sizeof(float))
// float_workspace_size_in_bytes = tmp_v_device_size_aligned + tmp_s_device_size_aligned
// Note: an additional 8MB init buffer (kinit_workspace_size_in_bytes) is allocated separately.

size_t GetFlashInferDeviceWorkspaceSize(size_t num_heads, size_t num_kv_heads, size_t head_dim) {
  static size_t device_workspace_size_in_bytes{0};
  if (device_workspace_size_in_bytes == 0) {
    size_t cta_tile_q = 0;
    size_t gqa_size = num_heads / num_kv_heads;
    int32_t compute_capacity = GetSMVersion();
    // Adapted from the workspace usage logic in FlashInfer's PrefillPlan.
    if (gqa_size > 64 && head_dim < 256) {
      cta_tile_q = 128;
    } else {
      if (compute_capacity >= 80) {
        if (gqa_size > 16) {
          cta_tile_q = 64;
        } else {
          cta_tile_q = 16;
        }
      } else {
        cta_tile_q = 64;
      }
    }
    int num_sm = GetSMCount();
    int num_blocks_per_sm = 2;
    int max_grid_size = num_blocks_per_sm * num_sm;
    size_t max_batch_size_if_split = max_grid_size / num_kv_heads;

    size_t tmp_v_device_size_aligned =
        (num_heads * max_batch_size_if_split * cta_tile_q * head_dim * sizeof(float) + kByteAlignedSize - 1) /
        kByteAlignedSize * kByteAlignedSize;
    size_t tmp_s_device_size_aligned =
        (num_heads * max_batch_size_if_split * cta_tile_q * sizeof(float) + kByteAlignedSize - 1) / kByteAlignedSize *
        kByteAlignedSize;
    size_t float_workspace_size_in_bytes = tmp_v_device_size_aligned + tmp_s_device_size_aligned;

    device_workspace_size_in_bytes = float_workspace_size_in_bytes + kInitWorkspaceSize;
  }

  return device_workspace_size_in_bytes;
}

/*
 * GPU Kernel for constructing kv_indices from block_table.
 *
 * This kernel transforms a 2D block_table [num_seqs, max_blocks_per_seq]
 * into a 1D kv_indices array by removing padding and concatenating valid block indices.
 */
__global__ void ConstructKvIndicesKernel(int32_t* block_table_ptr, int32_t* kv_indptr_ptr, size_t num_seqs,
                                         size_t max_blocks_per_seq, int32_t* kv_indices_ptr) {
  __shared__ int32_t start_idx;
  __shared__ int32_t end_idx;
  const size_t seq_id = blockIdx.x;  // Each block handles one sequence.

  if (seq_id >= num_seqs) return;

  if (threadIdx.x == 0) {
    start_idx = kv_indptr_ptr[seq_id];
    end_idx = kv_indptr_ptr[seq_id + 1];
  }
  __syncthreads();

  const size_t block_table_base_offset =
      seq_id * max_blocks_per_seq;  // Base offset for the current sequence in block_table.

  for (size_t idx = threadIdx.x; idx < (end_idx - start_idx); idx += blockDim.x) {
    kv_indices_ptr[start_idx + idx] = block_table_ptr[block_table_base_offset + idx];
  }
}

/*
 * GPU Kernel for computing kv_last_page_len from context_lens_ptr.
 */
__global__ void ComputeKvLastPageLenKernel(const int32_t* context_lens_ptr, int32_t* kv_last_page_len_ptr,
                                           size_t num_seqs, size_t page_size) {
  const size_t seq_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (seq_id >= num_seqs) return;

  int32_t total_len = context_lens_ptr[seq_id];
  int32_t last_page_len = total_len % static_cast<int32_t>(page_size);

  kv_last_page_len_ptr[seq_id] =
      (last_page_len == 0) ? static_cast<int32_t>(page_size) : static_cast<int32_t>(last_page_len);
}

void LaunchConstructKvIndices(int32_t* block_table_ptr, int32_t* kv_indptr_ptr, size_t num_seqs,
                              size_t max_blocks_per_seq, int32_t* kv_indices_ptr, cudaStream_t& stream) {
  dim3 grid(num_seqs);
  dim3 block(kThreadsPerBlock);
  ConstructKvIndicesKernel<<<grid, block, 0, stream>>>(block_table_ptr, kv_indptr_ptr, num_seqs, max_blocks_per_seq,
                                                       kv_indices_ptr);
}

void LaunchComputeKvLastPageLen(const int32_t* context_lens_ptr, size_t num_seqs, size_t page_size,
                                int32_t* kv_last_page_len_ptr, cudaStream_t& stream) {
  int num_blocks = (num_seqs + kThreadsPerBlock - 1) / kThreadsPerBlock;

  ComputeKvLastPageLenKernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(context_lens_ptr, kv_last_page_len_ptr,
                                                                          num_seqs, page_size);
}

template <typename Q_DTYPE, llm_kernels::utils::KVCacheType KV_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
FlashinferBatchPrefillHelper<Q_DTYPE, KV_DTYPE, O_DTYPE, INDEX_DTYPE>::FlashinferBatchPrefillHelper() = default;

template <typename Q_DTYPE, llm_kernels::utils::KVCacheType KV_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
FlashinferBatchPrefillHelper<Q_DTYPE, KV_DTYPE, O_DTYPE, INDEX_DTYPE>::~FlashinferBatchPrefillHelper() = default;

/*
 * Function for pre-processing for FlashInfer::BatchPrefillWithPagedKVCacheWrapper.
 *
 * This function includes workspace allocation, parameter setup, and FlashinferBatchPrefillHelper::plan.
 */
template <typename Q_DTYPE, llm_kernels::utils::KVCacheType KV_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
void FlashinferBatchPrefillHelper<Q_DTYPE, KV_DTYPE, O_DTYPE, INDEX_DTYPE>::Prepare(
    size_t num_heads, size_t head_size, size_t num_kv_heads, size_t block_size, size_t max_blocks_per_seq,
    size_t num_seqs, void* query_ptr, void* output_ptr, void* context_lens_ptr, void* k_cache_ptr, void* v_cache_ptr,
    INDEX_DTYPE* block_table_ptr, float* alibi_slopes_ptr, bool is_causal, float softmax_scale, void* workspace,
    size_t work_size, void* flashinfer_extra_workspace, void* page_locked_workspace, bool is_first_layer_on_node,
    cudaStream_t& stream) {
#ifdef ENABLE_FLASHINFER
  // Map KV_DTYPE to CUDA type.
  using CacheCudaType = llm_kernels::utils::KVCacheCudaType<Q_DTYPE, KV_DTYPE>;

  if (batch_prefill_handler_ == nullptr) {
    batch_prefill_handler_ = std::make_unique<flashinfer::BatchPrefillHandler>(page_locked_workspace, stream);
  }

  if (is_first_layer_on_node) {
    params_.float_workspace_size_in_bytes =
        GetFlashInferDeviceWorkspaceSize(num_heads, num_kv_heads, head_size) - params_.kinit_workspace_size_in_bytes;
    params_.batch_size = num_seqs;
    params_.max_blocks_per_seq = max_blocks_per_seq;
    params_.causal = is_causal;
    params_.maybe_sm_scale = softmax_scale;
    params_.alibi_slopes_ptr = alibi_slopes_ptr;

    // Compute host-side qo_indptr_h / kv_indptr_h on CPU.
    params_.GenParamsAtHost(num_seqs, context_lens_ptr, block_size, stream);

    // Allocate float_buffer/init_buffer/kv_indices/kv_last_page_len/qo_inptr in workspace.
    // Only can be called after GenParamsAtHost.
    params_.SetWorkSpace(flashinfer_extra_workspace, workspace, work_size);

    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(params_.qo_indptr, params_.qo_indptr_h.data(),
                                            (num_seqs + 1) * sizeof(INDEX_DTYPE), cudaMemcpyHostToDevice, stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(params_.kv_indptr, params_.kv_indptr_h.data(),
                                            (num_seqs + 1) * sizeof(INDEX_DTYPE), cudaMemcpyHostToDevice, stream));
    // Construct kv_indices from block_table.
    LaunchConstructKvIndices(block_table_ptr, params_.kv_indptr, num_seqs, max_blocks_per_seq, params_.kv_indices,
                             stream);

    // Compute kv_last_page_len.
    LaunchComputeKvLastPageLen(reinterpret_cast<int32_t*>(context_lens_ptr), num_seqs, block_size,
                               params_.kv_last_page_len, stream);

    // Build plan_info (requires host-side qo_indptr_h and kv_indptr_h)
    batch_prefill_handler_->Plan<Q_DTYPE, INDEX_DTYPE>(
        params_.float_buffer, params_.float_workspace_size_in_bytes, params_.init_buffer,
        params_.kinit_workspace_size_in_bytes, params_.qo_indptr_h.data(), params_.kv_indptr_h.data(),
        /*total_num_rows=*/num_seqs, num_seqs, num_heads, num_kv_heads, head_size, block_size);

    // For wrapper.
    params_.stream = stream;
    params_.num_qo_heads = num_heads;
    params_.pos_encoding_mode = flashinfer::PosEncodingMode::kNone;
  }
  params_.query = reinterpret_cast<Q_DTYPE*>(query_ptr);
  params_.output = reinterpret_cast<O_DTYPE*>(output_ptr);
  params_.paged_kv = std::make_unique<flashinfer::paged_kv_t<CacheCudaType, INDEX_DTYPE>>(
      num_kv_heads, block_size, head_size, num_seqs, flashinfer::QKVLayout::kNHD,
      reinterpret_cast<CacheCudaType*>(k_cache_ptr), reinterpret_cast<CacheCudaType*>(v_cache_ptr), params_.kv_indices,
      params_.kv_indptr, params_.kv_last_page_len);
#else
  std::cerr << "ENABLE_FLASHINFER is not defined. skipping call FlashinferBatchPrefillHelper::Prepare()." << std::endl;
#endif
}

template <typename Q_DTYPE, llm_kernels::utils::KVCacheType KV_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
void FlashinferBatchPrefillHelper<Q_DTYPE, KV_DTYPE, O_DTYPE, INDEX_DTYPE>::Forward() {
#ifdef ENABLE_FLASHINFER
  // Map KV_DTYPE to CUDA type.
  using CacheCudaType = llm_kernels::utils::KVCacheCudaType<Q_DTYPE, KV_DTYPE>;

  flashinfer::BatchPrefillWithPagedKVCacheWrapper<Q_DTYPE, CacheCudaType, O_DTYPE, INDEX_DTYPE>(
      batch_prefill_handler_.get(), params_.query, params_.qo_indptr, params_.q_rope_offset, *params_.paged_kv,
      params_.output, params_.lse, params_.num_qo_heads, params_.causal, params_.pos_encoding_mode,
      params_.use_fp16_qk_reduction, params_.maybe_sm_scale, params_.rope_scale, params_.rope_theta,
      params_.alibi_slopes_ptr, params_.stream);
#else
  std::cerr << "ENABLE_FLASHINFER is not defined. skipping call FlashinferBatchPrefillHelper::Prepare()." << std::endl;
#endif
}

/*
 * Function for allocating the workspace for the FlashinferBatchPrefillPagedAttentionParams.
 *
 * Workspace Layout(Elements, dtype: INDEX_DTYPE):
 * kv_last_page_len / kv_indptr / qo_indptr / kv_indices
 * batch_size / batch_size + 1 /  batch_size + 1 / SUM(num of valid blocks of seq_i)
 *
 * Flashinfer_extra_workspace Layout(Bytes):
 * float_buffer / init_buffer
 * float_workspace_size_in_bytes / kinit_workspace_size_in_bytes
 *
 */
template <typename Q_DTYPE, typename CACHE_CUDA_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
void FlashinferBatchPrefillPagedAttentionParams<Q_DTYPE, CACHE_CUDA_DTYPE, O_DTYPE, INDEX_DTYPE>::SetWorkSpace(
    void* flashinfer_extra_workspace, void* workspace, size_t work_size) {
  size_t total_workspace_size =
      (GetKvIndicesNumel() + GetKvLastPageLenNumel() + GetQoIndptrNumel() + GetKvindptrNumel()) * sizeof(INDEX_DTYPE);
  if (work_size < total_workspace_size) {
    throw std::runtime_error("workspace less than needed.");
  }
  char* workspace_ptr = reinterpret_cast<char*>(workspace);
  // kv_last_page_len / kv_indptr / qo_indptr / kv_indices
  kv_last_page_len = reinterpret_cast<INDEX_DTYPE*>(workspace_ptr);
  workspace_ptr += GetKvLastPageLenNumel() * sizeof(INDEX_DTYPE);
  kv_indptr = reinterpret_cast<INDEX_DTYPE*>(workspace_ptr);
  workspace_ptr += GetKvindptrNumel() * sizeof(INDEX_DTYPE);
  qo_indptr = reinterpret_cast<INDEX_DTYPE*>(workspace_ptr);
  workspace_ptr += GetQoIndptrNumel() * sizeof(INDEX_DTYPE);
  kv_indices = reinterpret_cast<INDEX_DTYPE*>(workspace_ptr);

  // float_buffer / init_buffer
  workspace_ptr = reinterpret_cast<char*>(flashinfer_extra_workspace);
  float_buffer = reinterpret_cast<void*>(workspace_ptr);
  workspace_ptr += float_workspace_size_in_bytes;
  init_buffer = reinterpret_cast<void*>(workspace_ptr);
}

template <typename Q_DTYPE, typename CACHE_CUDA_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
size_t FlashinferBatchPrefillPagedAttentionParams<Q_DTYPE, CACHE_CUDA_DTYPE, O_DTYPE, INDEX_DTYPE>::GetKvLastPageLenNumel()
    const {
  // Element size for kv_last_page_len: [batch_size].
  return batch_size;
}

template <typename Q_DTYPE, typename CACHE_CUDA_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
size_t FlashinferBatchPrefillPagedAttentionParams<Q_DTYPE, CACHE_CUDA_DTYPE, O_DTYPE, INDEX_DTYPE>::GetKvIndicesNumel()
    const {
  // Element size for kv_indices: sum of valid blocks across all requests.
  return static_cast<size_t>(kv_indptr_h[batch_size]);
}

template <typename Q_DTYPE, typename CACHE_CUDA_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
size_t FlashinferBatchPrefillPagedAttentionParams<Q_DTYPE, CACHE_CUDA_DTYPE, O_DTYPE, INDEX_DTYPE>::GetQoIndptrNumel()
    const {
  // Element size for qo_indptr: [batch_size + 1].
  return batch_size + 1;
}

template <typename Q_DTYPE, typename CACHE_CUDA_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
size_t FlashinferBatchPrefillPagedAttentionParams<Q_DTYPE, CACHE_CUDA_DTYPE, O_DTYPE, INDEX_DTYPE>::GetKvindptrNumel()
    const {
  // Element size for kv_indptr: [batch_size + 1].
  return batch_size + 1;
}

template <typename Q_DTYPE, typename CACHE_CUDA_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
void FlashinferBatchPrefillPagedAttentionParams<Q_DTYPE, CACHE_CUDA_DTYPE, O_DTYPE, INDEX_DTYPE>::GenParamsAtHost(
    size_t num_seqs, void* context_lens_ptr, size_t block_size, cudaStream_t& stream) {
  // Generate qo_indptr_h / kv_indptr_h.
  std::vector<INDEX_DTYPE> context_len_h(num_seqs);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(context_len_h.data(), context_lens_ptr, num_seqs * sizeof(INDEX_DTYPE),
                                          cudaMemcpyDeviceToHost, stream));
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  qo_indptr_h.resize(num_seqs + 1);
  kv_indptr_h.resize(num_seqs + 1);

  kv_indptr_h[0] = static_cast<INDEX_DTYPE>(0);
  qo_indptr_h[0] = static_cast<INDEX_DTYPE>(0);

  for (size_t i = 0; i < num_seqs; i++) {
    INDEX_DTYPE num_blocks = (context_len_h[i] + block_size - 1) / block_size;
    kv_indptr_h[i + 1] = kv_indptr_h[i] + num_blocks;
    qo_indptr_h[i + 1] = static_cast<INDEX_DTYPE>(i + 1);
  }
}

// Explicit template instantiation.
// (note: ryanyhuang) FlashInfer does not support float32 currently.
#ifdef ENABLE_FP8
template class FlashinferBatchPrefillHelper<__half, llm_kernels::utils::KVCacheType::kFp8E5M2, __half, int32_t>;
template class FlashinferBatchPrefillHelper<__half, llm_kernels::utils::KVCacheType::kFp8E4M3, __half, int32_t>;
template class FlashinferBatchPrefillHelper<__nv_bfloat16, llm_kernels::utils::KVCacheType::kFp8E5M2, __nv_bfloat16,
                                            int32_t>;
template class FlashinferBatchPrefillHelper<__nv_bfloat16, llm_kernels::utils::KVCacheType::kFp8E4M3, __nv_bfloat16,
                                            int32_t>;
#endif
template class FlashinferBatchPrefillHelper<__half, llm_kernels::utils::KVCacheType::kAuto, __half, int32_t>;
template class FlashinferBatchPrefillHelper<__nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto, __nv_bfloat16,
                                            int32_t>;

}  // namespace nvidia
}  // namespace llm_kernels