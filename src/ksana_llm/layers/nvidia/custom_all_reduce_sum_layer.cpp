/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/custom_all_reduce_sum_layer.h"

#include <unistd.h>

#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/utils/dynamic_memory_pool.h"

namespace ksana_llm {

static std::shared_ptr<Barrier> s_barrier = nullptr;

Status CustomAllReduceSumLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                     std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;

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
  uint32_t attn_dp_para_size = runtime_config.parallel_basic_config.attn_data_parallel_size;
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

  if (signal != nullptr) {
    CUDA_CHECK(cudaMemset(signal, 0x0, signal_sz));
  }
  CUDA_CHECK(cudaMemset(rank_data_, 0x0, rank_data_sz_));

  signals_ = context_->ext->GetCustomAllReduceSignals();
  input_handles_ = context_->ext->GetCustomAllReduceInputs();

  signals_[rank_] = signal;
  input_handles_[rank_] = input;

  // is full nvlink on each device
  is_full_nvlink_ = context_->ext->IsFullNvLink();
  // Enable trt allreduce only when `tp_size == 2` and group all reduce is not used
  if ((enable_trt_reduce_ = (world_size_ == 2 && !is_group_custom_all_reduce_))) {
    // When nvlink is not available, lower the threshold for using trt allreduce
    if (!is_full_nvlink_) {
      TRT_REDUCE_THRESHOLD = 128;
    }

    ModelConfig model_config;
    Singleton<Environment>::GetInstance()->GetModelConfig(model_config);
    // For DeepSeek-V3, roughly 28MB of extra memory is allocated
    AllocTrtAllReduceWorkspace(rank_, /*max_token_num*/ TRT_REDUCE_THRESHOLD,
                               /*hidden_dim*/ model_config.hidden_units, GetTypeSize(inter_data_type_),
                               context_->ext->GetTrtAllReduceBuffers(), context_->ext->GetTrtAllReduceFlags(),
                               context_->ext->GetTrtAllReduceWorkspaces(), context_->GetComputeStreams()[rank_].Get());
  }

  s_barrier = std::make_shared<Barrier>(1);
  s_barrier->Init(context_->GetTensorParallelSize());

  // When using cudaMalloc and reduce operations with P2P enabled, the system may hang. This issue may be a bug in NCCL
  // or CUDA. Resolving it requires switching PyTorch's memory allocator to asynchronous mode. Alternatively, adding a
  // synchronization operation can prevent concurrent execution of malloc and reduce to avoid the hang.
  const char* const torch_alloc_config = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
  const std::string torch_alloc_config_str = torch_alloc_config == nullptr ? "" : std::string(torch_alloc_config);
  need_sync_ = torch_alloc_config_str.find("backend:cudaMallocAsync") == std::string::npos;
  return Status();
}

Status CustomAllReduceSumLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

void CustomAllReduceSumLayer::Clear() {
  if (reduce_op_ != nullptr) {
    delete static_cast<llm_kernels::nvidia::CustomAllreduce*>(reduce_op_);
  }
}

void CustomAllReduceSumLayer::ResetInputBuffer(void* input) { input_handles_[rank_] = input; }

void CustomAllReduceSumLayer::ResetSignalBuffer(void* signal, size_t signal_sz) {
  // The threads in reduce group must be barriered, otherwise the reduce myabe hanged.
  CUDA_CHECK(cudaMemset(signal, 0x0, signal_sz));
  signals_[rank_] = signal;
}

template <typename T>
Status CustomAllReduceSumLayer::ForwardT(const std::vector<Tensor>& input_tensors,
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
    // Initialize the workspace on the first forward
    if (!is_init_) {
      // TODO(jinxcwu): layer的init是卡间串行的，但allreduce的init需要卡间并行，可以考虑并行创建commonmodel
      CustomAllReduceInit<T>(&reduce_op_, rank_data_, rank_data_sz_, rank_, world_size_, is_full_nvlink_, root_rank_);

      // Should register new signal buffer every time later.
      CustomAllReduceRegisterSignalBuffer<T>(reduce_op_, signals_);
      CustomAllReduceRegisterBuffer<T>(reduce_op_, input_handles_, *stream);

      if (enable_trt_reduce_) {
        InitTrtAllReduceWorkspace(rank_, context_->ext->GetTrtAllReduceBuffers(), context_->ext->GetTrtAllReduceFlags(),
                                  context_->ext->GetTrtAllReduceWorkspaces(), *stream);
      }

      // Must sync, otherwise the CustomAllReduceRun maybe hanged.
      cudaStreamSynchronize(*stream);

      // Without barrier, the reduce maybe hanged.
      s_barrier->arrive_and_wait();
      is_init_ = true;
    }
    // Use only when trt allreduce is enabled and the token number is below the threshold to ensure the performance
    if (const size_t token_num = input_tensors[0].shape[0]; token_num <= TRT_REDUCE_THRESHOLD && enable_trt_reduce_) {
      const size_t hidden_dim = input_tensors[0].shape[1];
      RunTrtAllReduce<T>(input, rank_, token_num, hidden_dim, context_->ext->GetTrtAllReduceWorkspaces(), result,
                         *stream);
    } else {
      CustomAllReduceRun<T>(reduce_op_, input, result, data_size, *stream);
    }
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

}  // namespace ksana_llm
