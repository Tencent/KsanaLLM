/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/nccl_all_gather_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status NcclAllGatherLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  size_t tp_size = context_->GetTensorParallelSize();
  if (tp_size == 1) {
    return Status();
  }
  size_t h = input_tensors[0].shape[0];
  size_t w_per = input_tensors[0].shape[1];

  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl stream just enable when IsRunContextDecodeAndDecodeSerially == false
  cudaStream_t* stream;
  if (context_->IsRunContextDecodeAndDecodeSerially()) {
    stream = &(context_->GetComputeStreams()[rank_].Get());
  } else {
    stream = &(context_->GetCommStreams()[rank_].Get());
  }

  NCCL_CHECK(ncclGroupStart());
  ncclResult_t ncclError =
      ncclAllGather(reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()),
                    reinterpret_cast<void*>(input_tensors[1].GetPtr<void>()), input_tensors[0].GetElementNumber(),
                    GetNcclDataType<T>(), context_->ext->GetNCCLParam()[rank_].nccl_comm, *stream);
  if (ncclError != ncclSuccess) {
    KLLM_LOG_ERROR << fmt::format("NCCL error: {}\n", ncclGetErrorString(ncclError));
    return Status(RetCode::RET_INFER_FAILED, "NCCL error");
  }
  NCCL_CHECK(ncclGroupEnd());
  InvokePermute<T>(input_tensors[1].GetPtr<void>(), output_tensors[0].GetPtr<void>(), {tp_size, h, w_per}, {1, 0, 2},
                   *stream);
  output_tensors[0].shape = {h, tp_size * w_per};
  return Status();
}

template class NcclAllGatherLayer<float>;
template class NcclAllGatherLayer<half>;
#ifdef ENABLE_BFLOAT16
template class NcclAllGatherLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
