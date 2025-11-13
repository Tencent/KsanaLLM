/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <set>
#include <vector>

#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

// The extension type define.
template <int T>
struct ExtensionTypeTraits {
  typedef DummyClass value_type;
};

// The global context, like cuda stream, nccl handler.
template <int T>
class ContextT {
 public:
  explicit ContextT(const size_t tensor_parallel_size, const size_t attn_data_parallel_size,
                    const size_t max_multi_batch_size);
  ~ContextT();

  size_t GetTensorParallelSize() { return tensor_parallel_size_; }
  size_t GetAttentionTensorParallelSize() { return tensor_parallel_size_ / attn_data_parallel_size_; }

  inline bool IsRunContextDecodeAndDecodeSerially() { return is_contextdecode_and_decode_run_serially_; }

  std::vector<Stream>& GetMemoryManageStreams() { return memory_manage_streams_; }

  size_t GetMaxMultiBatchSize() const { return max_multi_batch_size_; }

  std::vector<Stream>& GetComputeStreams() { return compute_streams_; }

  std::vector<Stream>& GetH2DStreams() { return h2d_streams_; }

  std::vector<Stream>& GetD2HStreams() { return d2h_streams_; }

  std::vector<Stream>& GetD2DStreams() { return d2d_streams_; }

  std::vector<Stream>& GetCommStreams() { return comm_streams_; }

  std::vector<Stream>& GetCommNodesStreams() { return comm_nodes_streams_; }

  const std::set<int>& GetSupportedCudaGraphCaptureSizes() { return cudagraph_captured_batchsizes; }

  inline bool IsGemmFp8Supported() { return is_gemm_fp8_supported_; }

  // Whether current node is standalone mode.
  bool IsStandalone() const;

  // Whether current node is master node.
  bool IsChief() const;

  // Whether current node is standalone mode in Expert-Parallel.
  bool IsExpertParallelStandalone() const;

  // Whether current node is master node in Expert-Parallel.
  bool IsExpertParallelChief() const;

  // Total number of machines participating in Expert-Parallel
  size_t GetExpertParallelWorldSize() { return expert_parallel_config_.expert_world_size; }

  // Rank value of the current node in multi-machine Expert-Parallel scenario
  size_t GetExpertParallelExpertNodeRank() { return expert_parallel_config_.expert_node_rank; }

  bool IsLastLayer() { return is_last_layer_; }

  void SetIsLastLayer(bool is_last_layer) { is_last_layer_ = is_last_layer; }

 public:
  friend class ExtensionTypeTraits<T>::value_type;
  typename ExtensionTypeTraits<T>::value_type* ext = nullptr;

 private:
  // init streams
  void InitStreams(const int worker_id);

 private:
  bool is_last_layer_ = false;
  const size_t tensor_parallel_size_{1};
  const size_t attn_data_parallel_size_{1};
  static constexpr int defalt_device_id_{0};
  int driver_version_;
  // if true, only one thread execute context_decode/decode and context_decode decode run in sync
  // TODO(karlluo): load from environment
  bool is_contextdecode_and_decode_run_serially_{true};
  bool is_gemm_fp8_supported_{false};
  const std::set<int> cudagraph_captured_batchsizes = {1, 2, 3};

  size_t max_multi_batch_size_;

  // streams
  std::vector<Stream> memory_manage_streams_;
  std::vector<Stream> compute_streams_;
  // these comm_streams seems not in use
  std::vector<Stream> comm_streams_;
  // btw nodes comm streams
  std::vector<Stream> comm_nodes_streams_;

  std::vector<Stream> h2d_streams_;
  std::vector<Stream> d2h_streams_;
  std::vector<Stream> d2d_streams_;

  // pipeline config.
  PipelineConfig pipeline_config_;
  ExpertParallelConfig expert_parallel_config_;

  // single node.
  bool is_standalone_ = false;
  bool is_expert_standalone_ = false;

  // Single node or master node of distributed model
  bool is_chief_ = false;

  // Single node or master node of expert parallel mode.
  bool is_expert_chief_ = false;

 private:
  // Initialize and destroy extension, implemented by device.
  void InitializeExtension();
  void DestroyExtension();
};

}  // namespace ksana_llm
