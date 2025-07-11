/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_communicator.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"

#ifdef ENABLE_CUDA
#  include "3rdparty/LLM_kernels/csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#endif

namespace ksana_llm {

template <typename T>
ModelCommunicator<T>::ModelCommunicator(Tensor* buffer, Tensor* input, int rank, std::shared_ptr<Context> context)
    : rank_(rank), context_(context), buffer_(buffer), input_(input) {
  EventCreateWithFlags(&comm_finish_event_, EVENT_DISABLE_TIMING);

#ifdef ENABLE_CUDA
  // TODO(rockcao): set SELECT_ALL_REDUCE_BY_SIZE as true by default
  if (std::getenv("SELECT_ALL_REDUCE_BY_SIZE") != nullptr) {
    KLLM_LOG_INFO << "SELECT_ALL_REDUCE_BY_SIZE is enabled";
    select_all_reduce_by_size_ = true;
  }

  nccl_all_reduce_sum_layer_ = std::make_shared<NcclAllReduceSumLayer<T>>();
  nccl_all_reduce_sum_layer_->Init({}, context_, rank_);

  nccl_all_gather_layer_ = std::make_shared<NcclAllGatherLayer<T>>();
  nccl_all_gather_layer_->Init({}, context_, rank_);

  use_cuda_graph_ = Singleton<Environment>::GetInstance()->IsCudagraphEnabled();

  is_full_nvlink_ = context_->ext->IsFullNvLink();

  tp_size_ = context_->GetTensorParallelSize();

  // ReduceSumLayer for tensor parallelism
  InitTensorParaCustomAllReduceSumLayer(input);

#elif defined(ENABLE_ACL)
  hccl_all_reduce_sum_layer_ = std::make_shared<HcclAllReduceSumLayer<T>>();
  hccl_all_reduce_sum_layer_->Init({}, context, rank);

  hccl_all_gather_layer_ = std::make_shared<HcclAllGatherLayer<T>>();
  hccl_all_gather_layer_->Init({}, context, rank);
#endif
}

#ifdef ENABLE_CUDA
template <typename T>
void ModelCommunicator<T>::InitTensorParaCustomAllReduceSumLayer(Tensor* input) {
  if (!context_->ext->IsSupportedP2PAccess()) {
    return;
  }
  size_t max_size = input->GetTotalBytes();
  size_t largest_part = max_size / tp_size_ + max_size % tp_size_;
  size_t signal_sz = sizeof(llm_kernels::nvidia::Signal) + largest_part;
  Stream* stream = &(context_->GetMemoryManageStreams()[rank_]);
  tp_signal_tensor_ = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {signal_sz}, rank_, nullptr, stream);

  // This is a buffer for storing the tuples of pointers pointing to
  // IPC buffers from all ranks. Each registered tuple has size of
  // 8*world_size bytes where world_size is at most 8. Allocating 8MB
  // is enough for 131072 such tuples. The largest model I've seen only
  // needs less than 10000 of registered tuples.
  constexpr size_t rank_data_sz = 8 * 1024 * 1024;
  tp_custom_all_reduce_sum_layer_ = std::make_shared<CustomAllReduceSumLayer<T>>();
  tp_custom_all_reduce_rank_tensor_ =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {rank_data_sz}, rank_, nullptr, stream);
  StreamSynchronize(*stream);
  tp_custom_all_reduce_sum_layer_->Init(
      {input->GetPtr<void>(), tp_signal_tensor_.GetPtr<void>(), signal_sz,
       tp_custom_all_reduce_rank_tensor_.GetPtr<void>(), rank_data_sz, /*is_group_custom_all_reduce*/ false},
      context_, rank_);
}
#endif

template <typename T>
ModelCommunicator<T>::~ModelCommunicator() {
  EventDestroy(comm_finish_event_);
}

template <typename T>
Status ModelCommunicator<T>::AllGather(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
#ifdef ENABLE_CUDA
  STATUS_CHECK_RETURN(nccl_all_gather_layer_->Forward(input_tensors, output_tensors));
  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(comm_finish_event_, context_->GetCommStreams()[rank_]);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], comm_finish_event_);
  }
#endif

#ifdef ENABLE_ACL
  if (context_->GetTensorParallelSize() > 1) {
    STATUS_CHECK_RETURN(hccl_all_gather_layer_->Forward(input_tensors, output_tensors));
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      EventRecord(comm_finish_event_, context_->GetCommStreams()[rank_]);
      StreamWaitEvent(context_->GetComputeStreams()[rank_], comm_finish_event_);
    }
  } else {
    MemcpyAsync(output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), input_tensors[0].GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
  }
#endif
  return Status();
}

template <typename T>
Status ModelCommunicator<T>::ReduceSum(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                                       bool is_multi_token_forward, bool use_custom) {
#ifdef ENABLE_CUDA
  if (CheckIfUseCustomReduceSum(input_tensors, use_custom)) {
    STATUS_CHECK_RETURN(tp_custom_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
  } else {
    STATUS_CHECK_RETURN(nccl_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
  }

  if (!context_->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(comm_finish_event_, context_->GetCommStreams()[rank_]);
    StreamWaitEvent(context_->GetComputeStreams()[rank_], comm_finish_event_);
  }
#endif

#ifdef ENABLE_ACL
  if (context_->GetTensorParallelSize() > 1) {
    STATUS_CHECK_RETURN(hccl_all_reduce_sum_layer_->Forward(input_tensors, output_tensors));
    if (!context_->IsRunContextDecodeAndDecodeSerially()) {
      EventRecord(comm_finish_event_, context_->GetCommStreams()[rank_]);
      StreamWaitEvent(context_->GetComputeStreams()[rank_], comm_finish_event_);
    }
  } else {
    MemcpyAsync(output_tensors[0].GetPtr<void>(), input_tensors[0].GetPtr<void>(), input_tensors[0].GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetComputeStreams()[rank_]);
    output_tensors[0].shape = input_tensors[0].shape;
    output_tensors[0].dtype = input_tensors[0].dtype;
  }
#endif

  return Status();
}

template <typename T>
bool ModelCommunicator<T>::CheckIfUseCustomReduceSum(const std::vector<Tensor>& input_tensors, bool use_custom) {
  if (select_all_reduce_by_size_ && input_tensors[0].GetTotalBytes() > kAllReduceThreshold) {
    return false;
  }
#ifdef ENABLE_CUDA
  if (!context_->ext->IsSupportedP2PAccess()) {
    return false;
  }
#endif
  int batch_size = input_tensors[0].shape[0];
  return use_custom && (tp_size_ == 2 || is_full_nvlink_) &&
         (!use_cuda_graph_ || (use_cuda_graph_ && context_->GetSupportedCudaGraphCaptureSizes().find(batch_size) ==
                                                      context_->GetSupportedCudaGraphCaptureSizes().end()));
}

template class ModelCommunicator<float>;
template class ModelCommunicator<float16>;
template class ModelCommunicator<bfloat16>;

}  // namespace ksana_llm
