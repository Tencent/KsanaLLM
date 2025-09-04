/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "common.h"
#include "deep_ep.h"

namespace deepep_wrapper {

class Wrapper {
 public:
  Wrapper(IPCData* shared_data, int rank, int num_ranks, int world_size, int node_rank);

  ~Wrapper();

  void Dispatch();

  void Combine();

  void Run();

 private:
  void InitConfig();

  void SetNumSMs(int new_num_sms);

  void InitHiddenBuffers();

  void InitMoeBuffer();

  void DispatchLayout(torch::Tensor& topk_idx);

  void IntranodeDispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales, torch::Tensor& topk_idx,
                         torch::Tensor& topk_weights);

  void InternodeDispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales, torch::Tensor& topk_idx,
                         torch::Tensor& topk_weights);

  void IntranodeCombine();

  void InternodeCombine();

 private:
  IPCData* shared_data_;
  int local_rank_;
  int rank_;
  int num_ranks_;
  int deepep_rank_;
  int world_size_;
  int node_rank_;
  int num_sms_;
  std::shared_ptr<deepep_wrapper::Buffer> buffer_;
  std::shared_ptr<deep_ep::Config> dispatch_config_;
  std::shared_ptr<deep_ep::Config> combine_config_;
  std::optional<deep_ep::EventHandle> empty_event_ = std::nullopt;

  // event
  cudaEvent_t dispatch_event;
  cudaEvent_t combine_event;

  // input data
  void* x_scales_ptr;
  void* topk_ids_ptr;
  void* topk_weights_ptr;
  torch::Tensor x_;
  torch::Tensor x_scales_;
  torch::Tensor topk_ids_;
  torch::Tensor topk_weights_;
  void* output_ = nullptr;

  // Workspace buffer
  std::vector<void*> hidden_buffer_ = {nullptr, nullptr, nullptr};
  void* moe_buffer_ = nullptr;

  // dispatch layout
  torch::Tensor num_tokens_per_rank_;
  torch::Tensor num_tokens_per_rdma_rank_;
  torch::Tensor num_tokens_per_expert_;
  torch::Tensor is_token_in_rank_;

  // dispatch & combine
  int recv_num_tokens_;

  // intranode dispatch
  torch::Tensor send_head_;
  torch::Tensor rank_prefix_matrix_;
  torch::Tensor channel_prefix_matrix_;
  torch::Tensor src_idx_;

  // internode dispatch
  torch::Tensor rdma_channel_prefix_matrix_;
  torch::Tensor gbl_channel_prefix_matrix_;
  torch::Tensor recv_gbl_rank_prefix_sum_;
  torch::Tensor recv_rdma_channel_prefix_matrix_;
  torch::Tensor recv_rdma_rank_prefix_sum_;
  torch::Tensor recv_gbl_channel_prefix_matrix_;
  torch::Tensor recv_src_meta_;
  torch::Tensor send_rdma_head_;
  torch::Tensor send_nvl_head_;

  cudaStream_t stream_;
};

}  // namespace deepep_wrapper