/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "wrapper.h"

namespace deepep_wrapper {

Wrapper::Wrapper(IPCData* shared_data, int rank, int num_ranks, int world_size, int node_rank) {
  shared_data_ = shared_data;
  local_rank_ = rank;
  rank_ = rank + node_rank * num_ranks / world_size;
  num_ranks_ = num_ranks;
  world_size_ = world_size;
  node_rank_ = node_rank;
  InitConfig();

  // 每个进程持有一份 DeepEP::Buffer
  bool low_latency_mode = false;
  size_t hidden_bytes = shared_data_->hidden_size * 2;
  size_t num_nvl_bytes = std::max(dispatch_config_->get_nvl_buffer_size_hint(hidden_bytes, num_ranks_),
                                  combine_config_->get_nvl_buffer_size_hint(hidden_bytes, num_ranks_));
  size_t num_rdma_bytes = 0;
  if (num_ranks_ > kNumMaxNvlPeers || low_latency_mode) {
    num_rdma_bytes =
        std::max(dispatch_config_->get_rdma_buffer_size_hint(static_cast<int64_t>(hidden_bytes), num_ranks_),
                 combine_config_->get_rdma_buffer_size_hint(static_cast<int64_t>(hidden_bytes), num_ranks_));
  }
  buffer_ = std::make_shared<deepep_wrapper::Buffer>(rank_, num_ranks_, static_cast<int64_t>(num_nvl_bytes),
                                                     static_cast<int64_t>(num_rdma_bytes), low_latency_mode,
                                                     torch::cuda::getCurrentCUDAStream(local_rank_));
  stream_ = torch::cuda::getCurrentCUDAStream(local_rank_).stream();
  // 同机各卡间交换 IPC Handle 地址
  std::string ipc_handle = buffer_->get_local_ipc_handle();
  std::memcpy(shared_data->ipc_handles[rank_], ipc_handle.data(), CUDA_IPC_HANDLE_SIZE);
  shared_data->ipc_handle_ready[rank_] = true;

  // 同机各卡间交换 nvshmem_unique_id
  auto unique_id = deep_ep::internode::get_unique_id();
  std::memcpy(shared_data->shared_unique_id[rank_], unique_id.data(), sizeof(nvshmemx_uniqueid_t));
  shared_data->unique_id_ready[rank_] = true;

  // 初始化nvshmem
  int wait_count = 0;
  bool buffer_connect_ready = false;
  while (!buffer_connect_ready) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    wait_count++;
    if (wait_count > 600) {  // 60秒超时
      std::cerr << "Process " << rank_ << ": Timeout waiting for unique ID" << std::endl;
      return;
    }
    buffer_connect_ready = true;
    for (int i = 0; i < num_ranks_; i++) {
      if (!shared_data->ipc_handle_ready[i]) {
        buffer_connect_ready = false;
        break;
      }
    }
    for (int i = 0; i < num_ranks_; i++) {
      if (!shared_data->unique_id_ready[i]) {
        buffer_connect_ready = false;
        break;
      }
    }
  }
  buffer_->sync(shared_data);

  // 打开共享显存
  if (shared_data_->use_scales) {
    cudaIpcOpenMemHandle(&x_scales_ptr, shared_data_->x_scales[local_rank_], cudaIpcMemLazyEnablePeerAccess);
  }
  cudaIpcOpenMemHandle(&topk_ids_ptr, shared_data_->topk_ids[local_rank_], cudaIpcMemLazyEnablePeerAccess);
  cudaIpcOpenMemHandle(&topk_weights_ptr, shared_data_->topk_weights[local_rank_], cudaIpcMemLazyEnablePeerAccess);

  cudaIpcOpenEventHandle(&dispatch_event, shared_data_->dispatch_events[local_rank_]);
  cudaIpcOpenEventHandle(&combine_event, shared_data_->combine_events[local_rank_]);

  if (buffer_->get_num_rdma_ranks() > 1) {
    nvshmem_barrier_all();
  }
}

Wrapper::~Wrapper() {
  if (buffer_->get_num_rdma_ranks() > 1) {
    nvshmem_finalize();
  }
}

void Wrapper::InitConfig() {
  // Adapted from DeepEP [https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py#L218]
  std::unordered_map<int, std::vector<int>> dispatch_config_map = {
      {2, {24, 256, 6, 128}},    {4, {6, 256, 6, 128}},    {8, {6, 256, 6, 128}},    {16, {36, 288, 20, 128}},
      {24, {8, 288, 32, 128}},   {32, {32, 288, 32, 128}}, {64, {20, 288, 28, 128}}, {128, {20, 560, 32, 128}},
      {144, {32, 720, 12, 128}}, {160, {28, 720, 12, 128}}};
  // TODO(zezhao): 后续修改为可动态配置的
  num_sms_ = kDeepEPDefaultNumSMs;
  std::vector<int> dispatch_config_param = dispatch_config_map[num_ranks_];
  dispatch_config_ = std::make_shared<deep_ep::Config>(num_sms_, dispatch_config_param[0], dispatch_config_param[1],
                                                       dispatch_config_param[2], dispatch_config_param[3]);

  // Adapted from DeepEP [https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/buffer.py#L246]
  std::unordered_map<int, std::vector<int>> combine_config_map = {
      {2, {10, 256, 6, 128}},  {4, {9, 256, 6, 128}},  {8, {4, 256, 6, 128}},   {16, {4, 288, 12, 128}},
      {24, {1, 288, 8, 128}},  {32, {1, 288, 8, 128}}, {64, {1, 288, 20, 128}}, {128, {1, 560, 12, 128}},
      {144, {2, 720, 8, 128}}, {160, {2, 720, 8, 128}}};
  std::vector<int> combine_config_param = combine_config_map[num_ranks_];
  combine_config_ = std::make_shared<deep_ep::Config>(num_sms_, combine_config_param[0], combine_config_param[1],
                                                      combine_config_param[2], combine_config_param[3]);
}

void Wrapper::SetNumSMs(int new_num_sms) { num_sms_ = new_num_sms; }

void Wrapper::InitHiddenBuffers() {
  for (size_t i = 0; i < hidden_buffer_.size(); ++i) {
    cudaIpcOpenMemHandle(&hidden_buffer_[i], shared_data_->hidden_buffer[local_rank_][i],
                         cudaIpcMemLazyEnablePeerAccess);
  }
}

void Wrapper::InitMoeBuffer() {
  cudaIpcOpenMemHandle(&moe_buffer_, shared_data_->moe_buffer[local_rank_], cudaIpcMemLazyEnablePeerAccess);
}

void Wrapper::DispatchLayout(torch::Tensor& topk_ids) {
  auto dispatch_layout_result =
      buffer_->get_dispatch_layout(topk_ids, shared_data_->num_experts, empty_event_, false, false);
  num_tokens_per_rank_ = std::get<0>(dispatch_layout_result);
  auto opt_num_tokens_per_rdma_rank = std::get<1>(dispatch_layout_result);
  if (opt_num_tokens_per_rdma_rank.has_value()) {
    num_tokens_per_rdma_rank_ = opt_num_tokens_per_rdma_rank.value();
  }
  num_tokens_per_expert_ = std::get<2>(dispatch_layout_result);
  is_token_in_rank_ = std::get<3>(dispatch_layout_result);
}

void Wrapper::IntranodeDispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                                torch::Tensor& topk_ids, torch::Tensor& topk_weights) {
  int expert_alignment = 1;
  int num_worst_tokens = 0;
  auto result =
      buffer_->intranode_dispatch(x,                                           // const at::Tensor&
                                  x_scales,                                    // const std::optional<at::Tensor>&
                                  std::make_optional(topk_ids),                // const std::optional<at::Tensor>&
                                  std::make_optional(topk_weights),            // const std::optional<at::Tensor>&
                                  std::make_optional(num_tokens_per_rank_),    // const std::optional<at::Tensor>&
                                  is_token_in_rank_,                           // const at::Tensor&
                                  std::make_optional(num_tokens_per_expert_),  // const std::optional<at::Tensor>&
                                  0,                                           // int
                                  std::nullopt,                                // const std::optional<at::Tensor>&
                                  std::nullopt,                                // const std::optional<at::Tensor>&
                                  expert_alignment,                            // int (不是引用!)
                                  num_worst_tokens,                            // int (不是引用!)
                                  *dispatch_config_,                           // const deep_ep::Config&
                                  empty_event_,                                // std::optional<deep_ep::EventHandle>&
                                  false,                                       // bool
                                  false, output_);

  // Return values that are not used in subsequent computations:
  // 4: num_recv_tokens_per_expert_list
  // 6: channel_prefix_matrix
  // 10: event
  recv_num_tokens_ = std::get<0>(result);
  auto opt_recv_x_scales = std::get<1>(result);
  if (opt_recv_x_scales.has_value()) {
    x_scales_ = opt_recv_x_scales;
  }
  auto opt_recv_topk_ids = std::get<2>(result);
  if (opt_recv_topk_ids.has_value()) {
    topk_ids_ = opt_recv_topk_ids.value();
  }
  auto opt_recv_topk_weights = std::get<3>(result);
  if (opt_recv_topk_weights.has_value()) {
    topk_weights_ = opt_recv_topk_weights.value();
  }
  rank_prefix_matrix_ = std::get<5>(result);
  channel_prefix_matrix_ = std::get<7>(result);
  src_idx_ = std::get<8>(result);
  send_head_ = std::get<9>(result);
}

void Wrapper::InternodeDispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                                torch::Tensor& topk_ids, torch::Tensor& topk_weights) {
  int expert_alignment = 1;
  auto result =
      buffer_->internode_dispatch(x,                                              // const at::Tensor&
                                  x_scales,                                       // const std::optional<at::Tensor>&
                                  std::make_optional(topk_ids),                   // const std::optional<at::Tensor>&
                                  std::make_optional(topk_weights),               // const std::optional<at::Tensor>&
                                  std::make_optional(num_tokens_per_rank_),       // const std::optional<at::Tensor>&
                                  std::make_optional(num_tokens_per_rdma_rank_),  // const std::optional<at::Tensor>&
                                  is_token_in_rank_,                              // const at::Tensor&
                                  std::make_optional(num_tokens_per_expert_),     // const std::optional<at::Tensor>&
                                  0,                                              // int
                                  0,                                              // int
                                  std::nullopt,                                   // const std::optional<at::Tensor>&
                                  std::nullopt,                                   // const std::optional<at::Tensor>&
                                  std::nullopt,                                   // const std::optional<at::Tensor>&
                                  std::nullopt,                                   // const std::optional<at::Tensor>&
                                  expert_alignment,                               // int
                                  *dispatch_config_,                              // const deep_ep::Config&
                                  empty_event_,  // std::optional<deep_ep::EventHandle>&
                                  false,         // bool
                                  false, output_);
  // 0: recv_num_tokens_
  // 1: recv_x_scales
  // 2: recv_topk_idx
  // 3: recv_topk_weights
  // 4: num_recv_tokens_per_expert_list
  // 5: rdma_channel_prefix_matrix
  // 6: gbl_channel_prefix_matrix
  // 7: recv_rdma_channel_prefix_matrix
  // 8: recv_rdma_rank_prefix_sum
  // 9: recv_gbl_channel_prefix_matrix
  // 10: recv_gbl_rank_prefix_sum
  // 11: recv_src_meta
  // 12: send_rdma_head
  // 13: send_nvl_head
  // 14: event

  // Return values that are not used in subsequent computations:
  // 4: num_recv_tokens_per_expert_list
  // 5: rdma_channel_prefix_matrix
  // 6: gbl_channel_prefix_matrix
  // 14: event
  recv_num_tokens_ = std::get<0>(result);
  auto opt_recv_x_scales = std::get<1>(result);
  if (opt_recv_x_scales.has_value()) {
    x_scales_ = opt_recv_x_scales;
  }
  auto opt_recv_topk_ids = std::get<2>(result);
  if (opt_recv_topk_ids.has_value()) {
    topk_ids_ = opt_recv_topk_ids.value();
  }
  auto opt_recv_topk_weights = std::get<3>(result);
  if (opt_recv_topk_weights.has_value()) {
    topk_weights_ = opt_recv_topk_weights.value();
  }

  auto opt_recv_rdma_channel_prefix_matrix = std::get<7>(result);
  if (opt_recv_rdma_channel_prefix_matrix.has_value()) {
    recv_rdma_channel_prefix_matrix_ = opt_recv_rdma_channel_prefix_matrix.value();
  }
  recv_rdma_rank_prefix_sum_ = std::get<8>(result);
  auto opt_recv_gbl_channel_prefix_matrix = std::get<9>(result);
  if (opt_recv_gbl_channel_prefix_matrix.has_value()) {
    recv_gbl_channel_prefix_matrix_ = opt_recv_gbl_channel_prefix_matrix.value();
  }

  recv_gbl_rank_prefix_sum_ = std::get<10>(result);

  auto opt_recv_src_meta = std::get<11>(result);
  if (opt_recv_src_meta.has_value()) {
    recv_src_meta_ = opt_recv_src_meta.value();
  }
  auto opt_send_rdma_head = std::get<12>(result);
  if (opt_send_rdma_head.has_value()) {
    send_rdma_head_ = opt_send_rdma_head.value();
  }
  auto opt_send_nvl_head = std::get<13>(result);
  if (opt_send_nvl_head.has_value()) {
    send_nvl_head_ = opt_send_nvl_head.value();
  }
}

void Wrapper::Dispatch() {
  // 输入存在 hidden_buffer[input_buffer_idx]
  // 输出存在 hidden_buffer[output_buffer_idx]
  cudaStreamWaitEvent(stream_, dispatch_event, 0);

  // Step 1: 准备输入输出
  int input_buffer_idx = shared_data_->input_buffer_idx[local_rank_];
  void* input_data = hidden_buffer_[input_buffer_idx];

  int output_buffer_idx = shared_data_->output_buffer_idx[local_rank_];
  output_ = hidden_buffer_[output_buffer_idx];

  // Step 2: 准备输入Tensor
  auto bfp16_options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto int32_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  x_ = torch::from_blob(input_data, {shared_data_->recv_token_num[local_rank_], shared_data_->hidden_size},
                        bfp16_options);
  if (shared_data_->use_scales) {
    x_scales_ = torch::from_blob(
        x_scales_ptr, {shared_data_->recv_token_num[local_rank_], shared_data_->hidden_size / 128}, float_options);
    if (x_fp8_ptr == nullptr) {
      cudaIpcOpenMemHandle(&x_fp8_ptr, shared_data_->x_workspace[local_rank_], cudaIpcMemLazyEnablePeerAccess);
      x_fp8_ptr += shared_data_->x_fp8_offsets[local_rank_];
    }
    auto fp8_options = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA);
    x_ = torch::from_blob(x_fp8_ptr, {shared_data_->recv_token_num[local_rank_], shared_data_->hidden_size},
                          fp8_options);
  } else {
    x_scales_ = std::nullopt;
  }
  auto topk_ids_int32 = torch::from_blob(
      topk_ids_ptr, {shared_data_->recv_token_num[local_rank_], shared_data_->num_topk}, int32_options);
  topk_ids_ = topk_ids_int32.to(torch::kInt64);
  topk_weights_ = torch::from_blob(topk_weights_ptr,
                                   {shared_data_->recv_token_num[local_rank_], shared_data_->num_topk}, float_options);

  // Step 3: 先执行 dispatch_layout
  DispatchLayout(topk_ids_);

  // Step 4: 执行 Dispatch
  if (buffer_->get_num_rdma_ranks() > 1) {
    InternodeDispatch(x_, x_scales_, topk_ids_, topk_weights_);
  } else {
    IntranodeDispatch(x_, x_scales_, topk_ids_, topk_weights_);
  }

  // Step 5: 将处理结果拷贝回共享显存
  shared_data_->recv_token_num[local_rank_] = recv_num_tokens_;
  if (recv_num_tokens_ > 0) {
    if (shared_data_->use_scales) {
      cudaMemcpyAsync(x_fp8_ptr, output_, recv_num_tokens_ * shared_data_->hidden_size, cudaMemcpyDeviceToDevice,
                      stream_);
      cudaMemcpyAsync(x_scales_ptr, x_scales_.value().data_ptr(),
                      x_scales_.value().numel() * x_scales_.value().element_size(), cudaMemcpyDeviceToDevice, stream_);
    }
    auto topk_ids_int32 = topk_ids_.to(torch::kInt32);
    cudaMemcpyAsync(topk_ids_ptr, topk_ids_int32.data_ptr(), topk_ids_int32.numel() * topk_ids_int32.element_size(),
                    cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(topk_weights_ptr, topk_weights_.data_ptr(), topk_weights_.numel() * topk_weights_.element_size(),
                    cudaMemcpyDeviceToDevice, stream_);
  }

  cudaEventRecord(dispatch_event, stream_);
}

void Wrapper::IntranodeCombine() {
  auto result = buffer_->intranode_combine(x_, std::nullopt, std::nullopt, std::nullopt, src_idx_, rank_prefix_matrix_,
                                           channel_prefix_matrix_, send_head_, *combine_config_, empty_event_, false,
                                           false, output_);
  recv_num_tokens_ = std::get<0>(result);
}

void Wrapper::InternodeCombine() {
  auto result = buffer_->internode_combine(x_, std::nullopt, std::nullopt, std::nullopt, recv_src_meta_,
                                           is_token_in_rank_, recv_rdma_channel_prefix_matrix_,
                                           recv_rdma_rank_prefix_sum_, recv_gbl_channel_prefix_matrix_, send_rdma_head_,
                                           send_nvl_head_, *combine_config_, empty_event_, false, false, output_);
  recv_num_tokens_ = std::get<0>(result);
}

void Wrapper::Combine() {
  cudaStreamWaitEvent(stream_, combine_event, 0);

  // 准备 输入输出
  // 沿用 Dispatch 时传入的 output_buffer_idx
  int output_buffer_idx = shared_data_->output_buffer_idx[local_rank_];
  void* input_data = hidden_buffer_[output_buffer_idx];
  auto bfp16_options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
  x_ = torch::from_blob(input_data, {shared_data_->recv_token_num[local_rank_], shared_data_->hidden_size},
                        bfp16_options);
  output_ = moe_buffer_;

  // RestoreFromDeduplicated();
  if (buffer_->get_num_rdma_ranks() > 1) {
    InternodeCombine();
  } else {
    IntranodeCombine();
  }
  shared_data_->recv_token_num[local_rank_] = recv_num_tokens_;
  cudaEventRecord(combine_event, stream_);
}

void Wrapper::Run() {
  if (local_rank_ == 0) {
    std::cout << "===== deepep_wrapper Init Success =====" << std::endl;
  }
  while (!shared_data_->trigger_exit) {
    if (shared_data_->trigger_dispatch[local_rank_]) {
      Dispatch();
      shared_data_->trigger_dispatch[local_rank_] = false;
    } else if (shared_data_->trigger_combine[local_rank_]) {
      Combine();
      shared_data_->trigger_combine[local_rank_] = false;
    } else if (shared_data_->trigger_init_hidden_buffers[local_rank_]) {
      InitHiddenBuffers();
      shared_data_->trigger_init_hidden_buffers[local_rank_] = false;
    } else if (shared_data_->trigger_init_moe_buffer[local_rank_]) {
      InitMoeBuffer();
      shared_data_->trigger_init_moe_buffer[local_rank_] = false;
    }
  }
}

}  // namespace deepep_wrapper