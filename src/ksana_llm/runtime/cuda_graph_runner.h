/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ksana_llm {

// The class used to build a cuda graph.
class CudaGraphBuilder {
 public:
  size_t GetMaxGraphBatchSize(size_t batch_size_step_);

  std::vector<size_t>& GetBatchSizeCaptureList() { return batch_sizes_to_catpure_list; }

 private:
  // The batch size config for cuda graph.
  size_t max_batch_size_ = 2;

  // Reserved for extension to align with vllm cudagraph
  // where batchsizes within batch_size_step_ will fall into a padded batchsize
  size_t batch_size_step_ = 8;

  const size_t default_batch_generation_limit = 1024;

  // batch size to catpure in warmup
  std::vector<size_t> batch_sizes_to_catpure_list;

  // Capture graphs for batch size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.....
  void GenerateBatchSizesConfig(std::vector<size_t>& batch_sizes_to_catpure);

  // Get the paded batch size.
  size_t GetPaddedBatchSize(size_t batch_size);
};

// Used to run inference forward with cuda graph.
class CudaGraphRunner {
 public:
  // Launch cuda graph with specified batch_size.
  void LaunchGraph(std::string batch_size, cudaStream_t stream);

  // Begin and end the graph capture.
  void BeginCapture(cudaStream_t stream, int rank_);

  // End to catpure cuda stream
  cudaGraphExec_t EndCapture(cudaStream_t stream, int rank_);

  // Check if the captured graph exists in cache.
  bool CheckIfGraphExec(std::string batch_size);

  void SetGraphInstance(const std::string& batch_size, cudaGraphExec_t& graph_exec);

 public:
  // Already captured batch sizes
  std::unordered_set<std::string> captured_batch_sizes;

  // Whether graph capturing is running
  bool is_capturing_graph = false;

 private:
  // Whether has been instantiated.
  bool initialized_ = false;

  // Batch_size => graph instances
  std::unordered_map<std::string, cudaGraphExec_t> graph_instances_;

  // The graph object.
  cudaGraph_t graph_;

  // The executable graph instance.
  cudaGraphExec_t exec_graph_;

  // Stream used for capture graph
  cudaStream_t capture_stream;
};

}  // namespace ksana_llm
