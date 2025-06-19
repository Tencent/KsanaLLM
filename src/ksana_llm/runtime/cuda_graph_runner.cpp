/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/cuda_graph_runner.h"
#include <algorithm>
#include <thread>
#include "ksana_llm/utils/nvidia/cuda_utils.h"
namespace ksana_llm {

void CudaGraphRunner::BeginCapture(cudaStream_t stream, int rank_) {
  cudaGraph_t capture_graph_;
  graph_ = capture_graph_;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
}

cudaGraphExec_t CudaGraphRunner::EndCapture(cudaStream_t stream, int rank_) {
  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph_, nullptr, nullptr, 0));
  CUDA_CHECK(cudaGraphDestroy(graph_));
  return graph_exec;
}

void CudaGraphRunner::SetGraphInstance(const std::string& batch_size, cudaGraphExec_t& graph_exec) {
  graph_instances_[batch_size] = graph_exec;
}

void CudaGraphRunner::LaunchGraph(std::string batch_size, cudaStream_t stream) {
  cudaGraphExec_t graph_exec = graph_instances_[batch_size];
  CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
}

bool CudaGraphRunner::CheckIfGraphExec(std::string batch_size) {
  if (graph_instances_.find(batch_size) != graph_instances_.end()) {
    return true;
  }
  return false;
}

size_t CudaGraphBuilder::GetPaddedBatchSize(size_t batch_size) {
  if (batch_size <= 2) {
    return batch_size;
  } else if (batch_size <= 4) {
    return 4;
  } else {
    return (batch_size + batch_size_step_ - 1) / batch_size_step_ * batch_size_step_;
  }
}

size_t CudaGraphBuilder::GetMaxGraphBatchSize(size_t max_num_seqs) {
  size_t padded_size = GetPaddedBatchSize(max_num_seqs);
  if (batch_sizes_to_catpure_list.empty()) {
    GenerateBatchSizesConfig(batch_sizes_to_catpure_list);
  }
  if (std::find(batch_sizes_to_catpure_list.begin(), batch_sizes_to_catpure_list.end(), padded_size) !=
      batch_sizes_to_catpure_list.end()) {
    return padded_size;
  }
  return batch_sizes_to_catpure_list.back();
}

void CudaGraphBuilder::GenerateBatchSizesConfig(std::vector<size_t>& batch_sizes_to_catpure) {
  batch_sizes_to_catpure = {1, 2, 4};
  batch_sizes_to_catpure.reserve(1027);
  for (int i = 1; i <= default_batch_generation_limit; ++i) {
    batch_sizes_to_catpure.push_back(batch_size_step_ * i);
  }
}

}  // namespace ksana_llm
