/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once
#include "csrc/utils/quant_type.h"

#include <memory>
#include <optional>
#include <vector>

namespace flashinfer {

class BatchPrefillHandler;

template <typename DType, typename IdType>
struct paged_kv_t;
enum class PosEncodingMode;
enum class QKVLayout;
}  // namespace flashinfer

namespace llm_kernels {
namespace nvidia {

size_t GetFlashInferDeviceWorkspaceSize(size_t num_heads, size_t num_kv_heads, size_t head_dim);

template <typename Q_DTYPE, typename CACHE_CUDA_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
struct FlashinferBatchPrefillPagedAttentionParams {
 public:
  Q_DTYPE* query;
  O_DTYPE* output;
  INDEX_DTYPE* qo_indptr;         // (Device) The indptr of the query/output tensor, shape: [batch_size + 1].
  INDEX_DTYPE* kv_indptr;         // (Device) The start position of each sequence in the kv_indices array.
  INDEX_DTYPE* kv_last_page_len;  // (Device) Valid length of the last block for each sequence.
  INDEX_DTYPE* kv_indices;        // (Device) The page indices of the paged kv-cache, shape: [total_num_blocks].

  std::vector<INDEX_DTYPE> kv_indptr_h;  // (Host) The start position of each sequence in the kv_indices array.
  std::vector<INDEX_DTYPE> qo_indptr_h;  // (Host) The indptr of the query/output tensor, shape: [batch_size + 1].

  size_t float_workspace_size_in_bytes = 0;
  static constexpr size_t kinit_workspace_size_in_bytes =
      8 * 1024 * 1024;  // According to the Python wrapper logic of flashinfer, init_buffer_ requires 8 MB.
  void* float_buffer;
  void* init_buffer;

  size_t batch_size;
  size_t max_blocks_per_seq;
  size_t num_qo_heads;
  size_t page_size;

  std::unique_ptr<flashinfer::paged_kv_t<CACHE_CUDA_DTYPE, INDEX_DTYPE>> paged_kv;
  flashinfer::PosEncodingMode pos_encoding_mode;
  cudaStream_t stream;

  bool causal = false;
  float* lse = nullptr;
  INDEX_DTYPE* q_rope_offset = nullptr;
  bool use_fp16_qk_reduction = false;

  std::optional<float> maybe_sm_scale = std::nullopt;
  float* alibi_slopes_ptr = nullptr;
  double softcap = 0.0f;  // unused

  float rope_scale = 1.0f;
  float rope_theta = 1e4f;

 public:
  size_t GetKvLastPageLenNumel() const;  // Get the Number of elements in kv_last_page_len (device buffer)
  size_t GetKvIndicesNumel() const;      // Get the Number of elements in kv_indices (device buffer)
  size_t GetQoIndptrNumel() const;       // Get the Number of elements in qo_indptr (device buffer)
  size_t GetKvindptrNumel() const;       // Get the Number of elements in kv_indptr (device buffer)
  void GenParamsAtHost(size_t num_seqs, void* context_lens_ptr, size_t block_size,
                       cudaStream_t& stream);  // Compute host-side qo_indptr_h / kv_indptr_h on CPU
  void SetWorkSpace(void* flashinfer_extra_workspace_buffer, void* workspace,
                    size_t work_size);  // Set device-side workspace
};

template <typename Q_DTYPE, llm_kernels::utils::KVCacheType KV_DTYPE, typename O_DTYPE, typename INDEX_DTYPE>
class FlashinferBatchPrefillHelper {
 public:
  FlashinferBatchPrefillHelper();
  ~FlashinferBatchPrefillHelper();
  void Prepare(size_t num_heads, size_t head_size, size_t num_kv_heads, size_t block_size, size_t max_blocks_per_seq,
               size_t num_seqs, void* query_ptr, void* output_ptr, void* context_lens_ptr, void* k_cache_ptr,
               void* v_cache_ptr, INDEX_DTYPE* block_table_ptr, float* alibi_slopes_ptr, bool is_causal,
               float softmax_scale, void* workspace, size_t work_size, void* flashinfer_extra_workspace,
               void* page_locked_workspace, bool is_first_layer_on_node, cudaStream_t& stream);

  void Forward();

 private:
  using CacheCudaType = llm_kernels::utils::KVCacheCudaType<Q_DTYPE, KV_DTYPE>;
  std::unique_ptr<flashinfer::BatchPrefillHandler> batch_prefill_handler_;
  FlashinferBatchPrefillPagedAttentionParams<Q_DTYPE, CacheCudaType, O_DTYPE, INDEX_DTYPE> params_;
};

}  // namespace nvidia
}  // namespace llm_kernels