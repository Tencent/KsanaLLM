/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <unistd.h>

#include "ksana_llm/layers/custom_all_reduce_sum_layer.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

template <typename T>
Status CustomAllReduceSumLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                        int rank) {
  context_ = context;
  rank_ = rank;

  // No need to initialize custom reduce sum when `tp == 1`.
  if (context_->GetTensorParallelSize() == 1) {
    return Status();
  }

  int parameter_index = 0;
  void* input = std::any_cast<void*>(parameters[parameter_index++]);
  void* signal = std::any_cast<void*>(parameters[parameter_index++]);
  size_t signal_sz = std::any_cast<size_t>(parameters[parameter_index++]);
  rank_data_ = std::any_cast<void*>(parameters[parameter_index++]);
  rank_data_sz_ = std::any_cast<size_t>(parameters[parameter_index++]);
  is_group_custom_all_reduce_ = std::any_cast<bool>(parameters[parameter_index++]);

  // NOTE(karlluo): For attention data parallelism, we do all reduce as group allreduce: just do allreduce with between
  // some gpus. The root rank is the first rank of the attention data parallel group. For example, if the rank is 0, 1,
  // 2, 3, and the attention data parallel size is 2, the root rank is 0. If the rank is 4, 5, 6, 7, and the attention
  // data parallel size is 2, the root rank is 4. The root rank is used to determine the group of ranks that will
  // perform the all-reduce operation. The root rank is the first rank of the attention data parallel group.
  uint32_t attn_dp_para_size = context_->GetAttnDataParallelSize();
  if (attn_dp_para_size > 1 && is_group_custom_all_reduce_) {
    uint32_t tp_para_size = context_->GetTensorParallelSize();
    uint32_t world_size = tp_para_size / attn_dp_para_size;
    uint32_t attn_dp_group_id = rank_ / world_size;
    world_size_ = world_size;
    if (world_size_ == 1) {
      // NOTE(karlluo): We do not need to do all reduce for attention data parallelism when the world size is 1.
      return Status();
    }
    root_rank_ = attn_dp_group_id * world_size;
  } else {
    world_size_ = context_->GetTensorParallelSize();
  }

  CUDA_CHECK(cudaMemset(signal, 0x0, signal_sz));
  CUDA_CHECK(cudaMemset(rank_data_, 0x0, rank_data_sz_));

  signals_ = context_->ext->GetCustomAllReduceSignals();
  input_handles_ = context_->ext->GetCustomAllReduceInputs();

  signals_[rank_] = signal;
  input_handles_[rank_] = input;

  // is full nvlink on each device
  is_full_nvlink_ = context_->ext->IsFullNvLink();

  // When using cudaMalloc and reduce operations with P2P enabled, the system may hang. This issue may be a bug in NCCL
  // or CUDA. Resolving it requires switching PyTorch's memory allocator to asynchronous mode. Alternatively, adding a
  // synchronization operation can prevent concurrent execution of malloc and reduce to avoid the hang.
  const char* const torch_alloc_config = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
  const std::string torch_alloc_config_str = torch_alloc_config == nullptr ? "" : std::string(torch_alloc_config);
  need_sync_ = torch_alloc_config_str.find("backend:cudaMallocAsync") == std::string::npos;
  return Status();
}

template <typename T>
Status CustomAllReduceSumLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                           std::vector<Tensor>& output_tensors) {
  cudaStream_t* stream;
  if (context_->IsRunContextDecodeAndDecodeSerially()) {
    stream = &(context_->GetComputeStreams()[rank_].Get());
  } else {
    stream = &(context_->GetCommStreams()[rank_].Get());
  }
  if (context_->GetTensorParallelSize() > 1) {
    void* input = input_tensors[0].GetPtr<void>();
    void* result = output_tensors[0].GetPtr<void>();
    int data_size = input_tensors[0].GetElementNumber();
    if (!is_init_) {
      // TODO(jinxcwu): layer的init是卡间串行的，但allreduce的init需要卡间并行，可以考虑并行创建commonmodel
      CustomAllReduceInit<T>(&reduce_op_, signals_, rank_data_, rank_data_sz_, rank_, world_size_, is_full_nvlink_,
                             root_rank_);
      CustomAllReduceRegisterBuffer<T>(reduce_op_, input_handles_, *stream);
      is_init_ = true;
    }
    CustomAllReduceRun<T>(reduce_op_, input, result, data_size, *stream);
    // To avoid getting stuck during CustomAllReduce.
    if (need_sync_) {
      CUDA_CHECK(cudaStreamSynchronize(*stream));
    }
  } else {
    void* src = input_tensors[0].GetPtr<void>();
    void* dst = output_tensors[0].GetPtr<void>();
    CUDA_CHECK(cudaMemcpyAsync(dst, src, input_tensors[0].GetTotalBytes(), cudaMemcpyDeviceToDevice, *stream));
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

template class CustomAllReduceSumLayer<float>;
template class CustomAllReduceSumLayer<half>;
#ifdef ENABLE_BFLOAT16
template class CustomAllReduceSumLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
