/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/data_hub/expert_parallel_deepep_wrapper.h"

namespace ksana_llm {

// Expert parallel wrapper tensor数量常量
constexpr size_t kXTensorIndex = 3;           // x tensor在input_tensors中的索引（use_scales时）
constexpr size_t kXScalesTensorIndex = 4;     // x_scales tensor在input_tensors中的索引
constexpr size_t kWorkspaceTensorIndex = 5;   // workspace tensor在input_tensors中的索引

static std::atomic_int barrier_count = 0;

ExpertParallelDeepepWrapper::ExpertParallelDeepepWrapper(size_t num_ranks, size_t num_ranks_per_node, size_t node_rank,
                                                         size_t max_token_num, size_t hidden_size, size_t expert_topk,
                                                         size_t num_experts, std::shared_ptr<Context> context)
    : context_(context),
      num_ranks_(num_ranks),
      num_ranks_per_node_(num_ranks_per_node),
      node_rank_(node_rank),
      max_token_num_(max_token_num),
      hidden_size_(hidden_size),
      expert_topk_(expert_topk),
      num_experts_(num_experts),
      initialized_(false) {}

ExpertParallelDeepepWrapper::~ExpertParallelDeepepWrapper() {
  if (!initialized_) {
    return;
  }
#ifdef ENABLE_CUDA
  for (size_t rank = 0; rank < num_ranks_per_node_; rank++) {
    SetDevice(rank);
    Free(x_scales_ptrs_[rank]);
    Free(topk_ids_ptrs_[rank]);
    Free(topk_weights_ptrs_[rank]);
    cudaEventDestroy(dispatch_events_[rank]);
    cudaEventDestroy(combine_events_[rank]);
  }
#endif
  if (shared_data_) {
    shared_data_->trigger_exit = true;
    sleep(1);
    munmap(shared_data_, sizeof(IPCData));
    shm_unlink(shm_name_.c_str());
    shared_data_ = nullptr;
  }
}


void ExpertParallelDeepepWrapper::Init() {
#ifdef ENABLE_CUDA
  // Create Shared Memory
  shm_name_ = fmt::format("/nvshmem_ipc_data_{}", node_rank_);
  int shm_fd = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
  if (ftruncate(shm_fd, sizeof(IPCData)) == -1) {
    KLLM_LOG_ERROR << "Failed to set shared memory size";
    close(shm_fd);
    shm_unlink(shm_name_.c_str());
    return;
  }
  void* addr = mmap(nullptr, sizeof(IPCData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  close(shm_fd);
  if (addr == MAP_FAILED) {
    KLLM_LOG_ERROR << "Failed to map shared memory";
    shm_unlink(shm_name_.c_str());
    return;
  }
  shared_data_ = static_cast<IPCData*>(addr);

  // Initialize Shared Memory Data
  memset(shared_data_, 0, sizeof(IPCData));
  for (int i = 0; i < kMaxNumRanks; ++i) {
    shared_data_->ipc_handle_ready[i] = false;
    shared_data_->unique_id_ready[i] = false;
    shared_data_->trigger_dispatch[i] = false;
    shared_data_->trigger_combine[i] = false;
    shared_data_->recv_token_num[i] = 0;
    shared_data_->input_buffer_idx[i] = 0;
    shared_data_->output_buffer_idx[i] = 0;
  }
  shared_data_->error_code = 0;
  shared_data_->error_message[0] = '\0';
  shared_data_->hidden_size = hidden_size_;
  shared_data_->num_topk = expert_topk_;
  shared_data_->num_experts = num_experts_;
  shared_data_->use_scales = true;

  // Create Cuda Device Buffer
  x_scales_ptrs_.resize(num_ranks_per_node_);
  topk_ids_ptrs_.resize(num_ranks_per_node_);
  topk_weights_ptrs_.resize(num_ranks_per_node_);
  for (size_t rank = 0; rank < num_ranks_per_node_; rank++) {
    SetDevice(rank);
    size_t dtype_size = shared_data_->use_scales ? sizeof(uint8_t) : sizeof(half);
    if (shared_data_->use_scales) {
      Malloc(&x_scales_ptrs_[rank], max_token_num_ * hidden_size_ / 128 * sizeof(float));
      cudaIpcGetMemHandle(&shared_data_->x_scales[rank], x_scales_ptrs_[rank]);
    }

    Malloc(&topk_ids_ptrs_[rank], max_token_num_ * expert_topk_ * sizeof(int32_t));
    cudaIpcGetMemHandle(&shared_data_->topk_ids[rank], topk_ids_ptrs_[rank]);

    Malloc(&topk_weights_ptrs_[rank], max_token_num_ * expert_topk_ * sizeof(float));
    cudaIpcGetMemHandle(&shared_data_->topk_weights[rank], topk_weights_ptrs_[rank]);
  }

  // Create CUDA event
  dispatch_events_.resize(num_ranks_per_node_);
  combine_events_.resize(num_ranks_per_node_);
  for (size_t rank = 0; rank < num_ranks_per_node_; rank++) {
    SetDevice(rank);
    CUDA_CHECK(cudaEventCreate(&dispatch_events_[rank], cudaEventDisableTiming | cudaEventInterprocess));
    CUDA_CHECK(cudaEventCreate(&combine_events_[rank], cudaEventDisableTiming | cudaEventInterprocess));
    CUDA_CHECK(cudaIpcGetEventHandle(&shared_data_->dispatch_events[rank], dispatch_events_[rank]));
    CUDA_CHECK(cudaIpcGetEventHandle(&shared_data_->combine_events[rank], combine_events_[rank]));
  }

  // Wating for the deepep_wrapper to retrieve nvshmem_id and ipc handle
  shared_data_->ready = true;
  bool unique_id_ready = false;
  while (!unique_id_ready) {
    unique_id_ready = true;
    size_t rank_start = node_rank_ * num_ranks_per_node_;
    size_t rank_end = std::min(num_ranks_, rank_start + num_ranks_per_node_);
    for (size_t i = rank_start; i < rank_end; ++i) {
      if (!shared_data_->ipc_handle_ready[i] || !shared_data_->unique_id_ready[i]) {
        unique_id_ready = false;
        break;
      }
    }
    usleep(1);
  }
#endif
  tensor_address_to_id_.resize(num_ranks_);
  initialized_ = true;
}

Status ExpertParallelDeepepWrapper::SetReady() {
  for (size_t i = 0; i < num_ranks_; ++i) {
    shared_data_->ipc_handle_ready[i] = true;
    shared_data_->unique_id_ready[i] = true;
  }
  KLLM_LOG_INFO << fmt::format("DeepEPWrapper Init Success!");
  return Status();
}

Status ExpertParallelDeepepWrapper::SetHiddenBuffers(const std::vector<Tensor>& hidden_buffers, int rank) {
#ifdef ENABLE_CUDA
  // Transfer important buffer addresses (hidden_buffer_0, hidden_buffer_1, reduce_buffer)
  // from KsanaLLM computation to DeepEPWrapper
  SetDevice(rank);
  for (auto& hidden_buffer_tensor : hidden_buffers) {
    void* buffer_ptr = hidden_buffer_tensor.GetPtr<void>();
    int hidden_idx = tensor_address_to_id_[rank].size();
    tensor_address_to_id_[rank][buffer_ptr] = hidden_idx;
    cudaIpcGetMemHandle(&shared_data_->hidden_buffer[rank][hidden_idx], buffer_ptr);
  }
  shared_data_->trigger_init_hidden_buffers[rank] = true;
#endif
  return Status();
}

Status ExpertParallelDeepepWrapper::SetMoeBuffer(const std::vector<Tensor>& moe_buffers, int rank) {
#ifdef ENABLE_CUDA
  SetDevice(rank);
  void* moe_buffer_ptr = moe_buffers[0].GetPtr<void>();
  cudaIpcGetMemHandle(&shared_data_->moe_buffer[rank], moe_buffer_ptr);
  shared_data_->trigger_init_moe_buffer[rank] = true;
#endif
  return Status();
}

Status ExpertParallelDeepepWrapper::Dispatch(const std::vector<Tensor>& input_tensors,
                                             std::vector<Tensor>& output_tensors, int rank) {
#ifdef ENABLE_CUDA
  // input_tensors:
  //   0: x
  //   1: topk_ids
  //   2: topk_weights
  // output_tensors:
  //   0: moe_buffer
  //   1: workspace_buffer

  // 0.获取基础配置
  void* x = input_tensors[0].GetPtr<void>();
  void* x_scales = nullptr;
  void* topk_ids = input_tensors[1].GetPtr<void>();
  void* topk_weights = input_tensors[2].GetPtr<void>();
  void* out = output_tensors[1].GetPtr<void>();

  int num_tokens = input_tensors[0].shape[0];

  // 1.告知 DeepEP 关键 shape 信息
  shared_data_->input_buffer_idx[rank] = tensor_address_to_id_[rank][x];
  shared_data_->output_buffer_idx[rank] = tensor_address_to_id_[rank][out];
  shared_data_->recv_token_num[rank] = num_tokens;

  shared_data_->use_scales = input_tensors.size() > kWorkspaceTensorIndex;
  // 2.拷贝x_scales, topk_ids, topk_weights 到共享空间
  if (shared_data_->use_scales) {
    x = input_tensors[kXTensorIndex].GetPtr<void>();
    x_scales = input_tensors[kXScalesTensorIndex].GetPtr<void>();
    void* workspace_ptr = input_tensors[kWorkspaceTensorIndex].GetPtr<void>();
    MemcpyAsync(x_scales_ptrs_[rank], x_scales, num_tokens * hidden_size_ / 128 * sizeof(float),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank]);

    Tensor a1_q_tensor(input_tensors[1].location, TYPE_INT8, {1}, input_tensors[1].device_id, x);
    if (!fp8_initialized_[rank]) {
      shared_data_->x_fp8_offsets[rank] = reinterpret_cast<char*>(x) - reinterpret_cast<char*>(workspace_ptr);
      cudaIpcGetMemHandle(&shared_data_->x_workspace[rank], workspace_ptr);
      fp8_initialized_[rank] = true;
    }
  }
  MemcpyAsync(topk_ids_ptrs_[rank], topk_ids, num_tokens * expert_topk_ * sizeof(int32_t), MEMCPY_DEVICE_TO_DEVICE,
              context_->GetComputeStreams()[rank]);
  MemcpyAsync(topk_weights_ptrs_[rank], topk_weights, num_tokens * expert_topk_ * sizeof(float),
              MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank]);

  CUDA_CHECK(cudaEventRecord(dispatch_events_[rank], context_->GetComputeStreams()[rank].Get()));

  // 3.配置 trigger_dispatch 信号，等待 DeepEP 处理
  KLLM_LOG_DEBUG << fmt::format("Will run dispatch. Rank {} send {} tokens.", rank, num_tokens);
  barrier_count = 0;
  shared_data_->trigger_dispatch[rank] = true;
  while (shared_data_->trigger_dispatch[rank]) {
  }

  CUDA_CHECK(cudaStreamWaitEvent(context_->GetComputeStreams()[rank].Get(), dispatch_events_[rank], 0));

  // 4. 获取 DeepEP 处理结果
  int recv_token_num = shared_data_->recv_token_num[rank];
  KLLM_LOG_DEBUG << fmt::format("Dispatch finished. Rank {} get {} tokens.", rank, recv_token_num);
  if (recv_token_num > 0) {
    // TODO(zezhao): 这里 x 在 DeepEP 中已存到 shared_data_->reduce_buffer 中，但提示不够明显
    size_t dtype_size = shared_data_->use_scales ? sizeof(uint8_t) : sizeof(half);
    if (shared_data_->use_scales) {
      MemcpyAsync(x_scales, x_scales_ptrs_[rank], recv_token_num * hidden_size_ / 128 * sizeof(float),
                  MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank]);
    }
    MemcpyAsync(topk_ids, topk_ids_ptrs_[rank], recv_token_num * expert_topk_ * sizeof(int32_t),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank]);
    MemcpyAsync(topk_weights, topk_weights_ptrs_[rank], recv_token_num * expert_topk_ * sizeof(float),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank]);
  }

  // 5. 更新输出 shape
  output_tensors[1].shape[0] = recv_token_num;
  output_tensors[1].shape[1] = input_tensors[0].shape[1];
#endif
  return Status();
}

Status ExpertParallelDeepepWrapper::Combine(const std::vector<Tensor>& input_tensors,
                                            std::vector<Tensor>& output_tensors, int rank) {
#ifdef ENABLE_CUDA
  // 0. 获取基础配置
  size_t dtype_size = shared_data_->use_scales ? sizeof(uint8_t) : sizeof(half);
  int num_tokens = shared_data_->recv_token_num[rank];
  void* x = input_tensors[0].GetPtr<void>();
  // TODO(zezhao): 这里默认输入指向的是reduce_buffer,提示不够明显
  void* x_combined = output_tensors[0].GetPtr<void>();
  // TODO(zezhao): 这里默认输出指向 moe_buffer，提示不够明显

  // 2. 配置 event 同步信号
  CUDA_CHECK(cudaEventRecord(combine_events_[rank], context_->GetComputeStreams()[rank].Get()));

  // 3.配置 trigger_combine 信号，等待 DeepEP 处理
  KLLM_LOG_DEBUG << fmt::format("Will run dispatch. Rank {} send {} tokens.", rank, num_tokens);
  barrier_count++;
  while (barrier_count != num_ranks_per_node_) {}
  shared_data_->trigger_combine[rank] = true;
  while (shared_data_->trigger_combine[rank]) {
  }

  CUDA_CHECK(cudaStreamWaitEvent(context_->GetComputeStreams()[rank].Get(), combine_events_[rank], 0));

  // 4. 获取 DeepEP 处理结果
  int recv_token_num = shared_data_->recv_token_num[rank];
  KLLM_LOG_DEBUG << fmt::format("Combine finished. Rank {} get {} tokens.", rank, recv_token_num);

  // 5. 更新输出 shape
  output_tensors[0].shape[0] = recv_token_num;
  output_tensors[0].shape[1] = input_tensors[0].shape[1];
#endif
  return Status();
}

}  // namespace ksana_llm
