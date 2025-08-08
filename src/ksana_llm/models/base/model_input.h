/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// Convert input ids to expected format.
class ModelInput {
 public:
  ModelInput(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
             std::shared_ptr<Context> context);
  ~ModelInput();

  // Parse forward request.
  void ParseFromRequests(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode = RunMode::kMain);

 private:
  // Get the offser of k & v in one layer of the cache block.
  int GetKoffsetInBlockLayer();
  int GetVoffsetInBlockLayer();

  void PrepareInputRefit(const std::vector<ForwardRequest>& forward_reqs);
  void PrepareVLInputRefit(const std::vector<ForwardRequest>& forward_reqs);
  void CreateVLTensors();
  void DestroyVLTensors();
  void PrepareVLRequest(const std::vector<ForwardRequest>& forward_reqs);
  void PrepareNextNGatherIdx(const std::vector<ForwardRequest>& forward_reqs, const RunMode run_mode);

  void PrepareMRopePos(const std::vector<ForwardRequest>& forward_reqs);

#ifdef ENABLE_CUDA
  template <typename T>
  void PrepareImgMask(size_t pos_num);

  void PrepareCudagraphParams(const std::vector<ForwardRequest>& forward_reqs);
#endif

#ifdef ENABLE_ACL
  void PrepareATBKVCache(const std::vector<ForwardRequest>& forward_reqs, bool is_multi_token_forward);
#endif

  // Determine whether to use cache for the current batch of multi token requests.
  void CheckUseCache(const std::vector<ForwardRequest>& forward_reqs);

 public:
  // The input batch size.
  size_t batch_size;
  size_t dp_batch_size;

  // The multi-token forwarding request total sequence length.
  size_t multi_token_request_total_seq_len = 0;
  size_t dp_multi_token_request_total_seq_len = 0;

  // Number of requests who are forwarding multi-tokens in this step.
  size_t multi_token_request_num = 0;
  size_t dp_multi_token_request_num = 0;

  // Number of requests who are forwarding single-token in this step.
  size_t single_token_request_num = 0;
  size_t dp_single_token_request_num = 0;

  // The max tokens.
  size_t multi_token_request_max_tokens = 0;
  size_t single_token_request_max_tokens = 0;
  size_t dp_multi_token_request_max_tokens = 0;
  size_t dp_single_token_request_max_tokens = 0;

  // The total prefix length.
  size_t total_prefix_len = 0;
  size_t dp_total_prefix_len = 0;

  // current request batchsize matches cudagraph catpure range
  bool is_cudagraph_batchsize_matched = false;

  // if current req is cudagraph capture request
  bool is_cudagraph_capture_request = false;

  // Whether to use kv cache.
  bool use_cache = true;

  std::vector<size_t> dp_input_offset_list_uint64;
  std::vector<size_t> dp_input_prefix_list_uint64;
  std::vector<size_t> input_offset_list_uint64;
  std::vector<size_t> input_prefix_list_uint64;

  std::vector<int> input_ids_cpu;

  // The infer stage, context decode or decode.
  InferStage infer_stage;

  // The input ids, int32
  Tensor input_ids;

  // The ids offset tensor, uint64
  Tensor input_offset_uint64_tensor;
  Tensor dp_input_offset_uint64_tensor;
  Tensor dp_input_offset_int32_tensor;

  // The input's prefix length
  Tensor input_prefix_uint64_tensor;
  Tensor dp_input_prefix_uint64_tensor;

  Tensor dp_prefill_q_offset_uint64_tensor;
  Tensor dp_prefill_q_offset_int32_tensor;

  // Indicate the corresponding index position of the input during the flexible rotary_embedding kernel computation,
  // considering the impact of flexible cache optimization.
  Tensor dp_flexible_rotary_embedding_pos;

  // Due to the optimization of PrefixCaching for computation reuse, incorporating the effects of flexible caching, a
  // mask is used during the flexible rotary_embedding computation to avoid multiple executions of flexible
  // rotary_embedding on the prefix block.
  Tensor dp_flexible_rotary_embedding_mask;

  // The 3-dimentional index position for multimodal rotarty embedding.
  Tensor dp_mrotary_embedding_pos;

  // Record which logits in the output of all tokens need to be extracted for subsequent sampling calculations
  // Due to the presence of logits_custom_length and speculative_decoding, a single request may require extracting more
  // than one logit. In the standard case, only the last logit of each request needs to be retrieved
  Tensor logits_idx_uint64_tensor;

  Tensor nextn_hidden_idx_uint64_tensor;

  Tensor dp_dst_flexible_kv_cache_tensor;
  Tensor dp_src_flexible_kv_cache_tensor;
  Tensor dp_dst_flexible_token_idx_tensor;
  Tensor dp_src_flexible_token_idx_tensor;
  Tensor dp_flexible_offset_uint64_tensor;

  // Tensors to hold pairs(pos, data_length) and embeddings ptr of positions for input_refit on the CPU.
  struct {
    Tensor pos_pair_tensor, emb_fp32_ptr_tensor;
  } cpu_input_refit_tensor;

  // IXC model use PLoRA
  bool is_mask = false;
  Tensor im_mask;

  Event kvcache_offset_event;
  Event rotary_embedding_event;
  Event input_ids_event;

#ifdef ENABLE_ACL
  // record all reqs token number on host, shape: [batch_size]
  Tensor seq_len_host;
  // Tensor to save kv cache base. detail doc please refer:
  // docs/Technology/kvcache-relationship-between-ascend-atb-and-ksana.md shape: [total_k/v_blocks, block_token_num,
  // kv_head_num, head_dim]
  Tensor k_cache_blocks_base;
  Tensor v_cache_blocks_base;

  // for multi-token forwarding: layers_slot_mapping shape is [num_layers, all_reqs_tokens_num]
  // for single-token forwarding: layers_block_table shape is [num_layers, batch_size]
  std::vector<int32_t> layers_slot_mapping_host;
  Tensor layers_slot_mapping;

  // only used for single-token forwarding: layers_block_table shape is [num_layers, batch_size *
  // max_num_blocks_per_query]
  std::vector<int32_t> layers_block_table_host;
  Tensor layers_block_table;

  // since layer's forward only support Tensor as input (nothing to do with karlluo), such crappy design ignore runtime
  // attribute, so we need a tensor to be attribute.
  // shape: [2]; 0: layers_slot_mapping_dim_1; 1: max_num_blocks_per_query
  Tensor atb_attention_attr;

  // assemble last token index for gather, dtype is int64_t
  Tensor last_token_index_tensor;

  std::vector<void*> kv_cache_ptrs;
  Tensor kv_cache_ptrs_tensor;
#endif

  size_t dp_max_forwarding_tokens = 0;

  // current rank related attention data para group id
  // NOTE(karlluo): for example: machine has 4 GPUs, Attention Data Parallelism is 2, Tensor Parallelism is 2.
  // |----Attn DP Group id 0----|----Attn DP Group id 1----|
  // |     TP 0   |     TP1     |     TP0    |     TP1     |
  // |     GPU0   |     GPU1    |     GPU2   |     GPU3    |
  size_t attn_dp_group_id_ = 0;
  int attn_dp_rank_id_ = 0;

  size_t attn_dp_group_size_;

  // The beg and end token offset of every dp group,
  // in format [dp0_prefill_beg, dp0_prefill_end, dp0_decode_beg, dp0_decode_end,
  //            dp1_prefill_beg, dp1_prefill_end, dp1_decode_beg, dp1_decode_end ...]
  std::vector<int> attn_dp_group_offsets_;

 private:
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;

  bool enable_blocked_multi_token_forwarding_kv_;

  const int rank_;
  std::shared_ptr<Context> context_;
  bool enable_flash_mla_;

  int block_size_;
  int layer_num_on_node_;
  size_t total_sampling_token_num_;

  // for nextn layer(MTP), record each req's first token index in hidden output
  std::unordered_map<size_t, size_t> mtp_req_id_to_pos_;

 public:
  struct input_info {
    std::vector<ForwardRequest*> reqs;
    std::vector<ForwardRequest*> dp_reqs;

    Tensor input_length;  // only for page, forwarding_tokens.size()
    Tensor kv_list;
    Tensor kv_cache_offset;
    Tensor rotary_embedding_pos;
    Tensor rotary_embedding_mask;
    Tensor layer_kv_cache_ptr;  // host
    Tensor block_table;
    Tensor tile_scheduler_metadata;
    Tensor num_splits;
    Tensor metadata;  // host, only for page, size_t

    size_t total_dp_input_ids_len = 0;
    size_t kv_cache_block_num = 0;

    void Reset() {
      reqs.clear();
      dp_reqs.clear();
      total_dp_input_ids_len = 0;
      kv_cache_block_num = 0;
    }
  };

  input_info flash_input;        // input_ids length is non-specialized, use flash attention
  input_info page_single_input;  // input_ids length is 1, use page attention
  input_info page_dual_input;    // input_ids length is 2, use page attention

  void PreparePrefill();
  void PrepareDualDecode();
  void PrepareSingleDecode();
  void PrepareMetadata();

  void PrepareInputIds(const std::vector<ForwardRequest>& forward_reqs);

  void PreparePageInput(input_info& input);
  void PrepareKVCacheBlocks(input_info& info);
  void PrepareKVCacheBlockTable(input_info& info);
  void PrepareDecodeRotary(input_info& input);
  void PrepareFlashMla(input_info& input);
  void PrepareFlashRotary(input_info& input);
  void PrepareFlexibleCache(input_info& input);
};

}  // namespace ksana_llm
