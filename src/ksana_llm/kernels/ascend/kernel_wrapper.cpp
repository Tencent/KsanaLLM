/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "aclnnop/aclnn_embedding.h"
#include "atb/infer_op_params.h"

#include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

void LookupEmbedding(const aclTensor* input_ids, const aclTensor* embedding_table, const aclTensor* position_table,
                     aclTensor* output, aclrtStream stream, WorkSpaceFunc ws_func) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(aclnnEmbeddingGetWorkspaceSize(embedding_table, input_ids, output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnEmbedding(workspace, ws_size, executor, stream));
}

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  if (tensor.dtype != target_dtype) {
    llm_kernels::utils::ATBOperationExecutor atb_cast_op_executor;
    llm_kernels::ascend::CastParam cast_param;
    // NOTE(karlluo): there is no inplace in ATB, we have prepare a buffer simulate inplace operation
    cast_param.dataType = static_cast<aclDataType>(target_dtype);
    atb::Operation* cast_op = new llm_kernels::ascend::CastOperation(
        fmt::format("Cast{}To{}Inplace", GetTypeString(tensor.dtype), GetTypeString(target_dtype)), cast_param);
    atb_cast_op_executor.SetOperation(cast_op);
    int32_t rank;
    GetDevice(&rank);
    reinterpret_cast<atb::Context*>(GetRuntimeContext(rank))->SetExecuteStream(stream.Get());
    Tensor tmp_tensor(MemoryLocation::LOCATION_DEVICE, target_dtype, tensor.shape, rank);
    atb_cast_op_executor.ResetVariantPack();
    atb_cast_op_executor.SetInputTensor(tensor.GetPtr<void>(), tensor.shape,
                                        static_cast<aclDataType>(DataType(tensor.dtype)));
    atb_cast_op_executor.SetOutputTensor(tmp_tensor.GetPtr<void>(), tensor.shape,
                                         static_cast<aclDataType>(target_dtype));
    atb_cast_op_executor.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank)), GetWorkSpaceFunc());
    StreamSynchronize(stream);
    Memcpy(tensor.GetPtr<void>(), tmp_tensor.GetPtr<void>(), tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE);
    tensor.dtype = target_dtype;
  } else {
    // NOTE(karlluo): dtype same will skip cast
    KLLM_LOG_DEBUG << fmt::format("Cast{}To{}Inplace is ignore", GetTypeString(tensor.dtype),
                                  GetTypeString(target_dtype));
  }
  return Status();
}

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* workspace_ptr) {
  std::vector<size_t> input_shape = input_tensor.shape;
  std::vector<size_t> output_shape = input_tensor.shape;
  atb::infer::TransposeParam param;
  for (size_t i = 0; i < permutation.size(); ++i) {
    param.perm.push_back(static_cast<int32_t>(permutation[i]));
    output_shape[i] = input_shape[permutation[i]];
  }
  llm_kernels::utils::ATBOperationExecutor atb_op_executor;
  int32_t rank;
  GetDevice(&rank);
  atb_op_executor.Init(rank, param);
  output_tensor.dtype = input_tensor.dtype;
  output_tensor.shape = output_shape;
  reinterpret_cast<atb::Context*>(GetRuntimeContext(rank))->SetExecuteStream(stream.Get());
  atb_op_executor.ResetVariantPack();
  atb_op_executor.SetInputTensor(input_tensor.GetPtr<void>(), input_tensor.shape,
                                 static_cast<aclDataType>(DataType(input_tensor.dtype)));
  atb_op_executor.SetOutputTensor(output_tensor.GetPtr<void>(), output_tensor.shape,
                                  static_cast<aclDataType>(DataType(output_tensor.dtype)));
  atb_op_executor.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank)), GetWorkSpaceFunc());
  StreamSynchronize(stream);
  return Status();
}

template <typename T>
Status ArgMaxATBExecutor<T>::Init(const int rank, const size_t max_batch_size) {
  llm_kernels::ascend::ArgmaxParam argmax_param;
  argmax_param.dim = 1;
  atb::Operation* argmax_op = new llm_kernels::ascend::ArgmaxOperation("argmax_exec_op_1", argmax_param);

  llm_kernels::ascend::CastParam cast_param;
  cast_param.dataType = aclDataType::ACL_UINT32;
  atb::Operation* cast_op = new llm_kernels::ascend::CastOperation("argmax_exec_op_2", cast_param);

  atb_argmax_op_executor_.SetOperation(argmax_op);
  atb_cast_op_executor_.SetOperation(cast_op);
  internal_tensor_ = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT32, {max_batch_size}, rank);
  return Status();
}

template <typename T>
Status ArgMaxATBExecutor<T>::Run(const int rank, const T* input, const int32_t batch_size, const int32_t vocab_size,
                                 uint32_t* result, Stream& stream) {
  // NOTE(karlluo): get argmax and output int32 type
  atb_argmax_op_executor_.ResetVariantPack();
  atb_argmax_op_executor_.SetInputTensor(reinterpret_cast<void*>(const_cast<T*>(input)),
                                         {static_cast<uint32_t>(batch_size), static_cast<uint32_t>(vocab_size)},
                                         static_cast<aclDataType>(TYPE_FP32));
  atb_argmax_op_executor_.SetOutputTensor(internal_tensor_.GetPtr<void>(), {static_cast<uint32_t>(batch_size)},
                                          static_cast<aclDataType>(TYPE_INT32));
  atb_argmax_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank)), GetWorkSpaceFunc());
  // NOTE(karlluo): cast int32 to uint32 type
  atb_cast_op_executor_.ResetVariantPack();
  atb_cast_op_executor_.SetInputTensor(internal_tensor_.GetPtr<void>(), {static_cast<uint32_t>(batch_size)},
                                       static_cast<aclDataType>(TYPE_INT32));
  atb_cast_op_executor_.SetOutputTensor(result, {static_cast<uint32_t>(batch_size)},
                                        static_cast<aclDataType>(TYPE_UINT32));
  atb_cast_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank)), GetWorkSpaceFunc());
  return Status();
}

template class ArgMaxATBExecutor<float>;

template <typename T>
Status ArgMax(const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result, Stream& stream,
              void* buffer_ptr) {
  if (std::is_same<T, float>::value) {
    int32_t rank;
    GetDevice(&rank);
    reinterpret_cast<atb::Context*>(GetRuntimeContext(rank))->SetExecuteStream(stream.Get());
    ArgMaxATBExecutor<T>* arg_max_atb_executor_ptr = reinterpret_cast<ArgMaxATBExecutor<T>*>(buffer_ptr);
    return arg_max_atb_executor_ptr->Run(rank, input, batch_size, vocab_size, result, stream);
  }

  return Status(RET_UNDEFINED_REFERENCE, "Not supported argmax data type.");
}

#define INSTANTIATE_ARG_MAX(T)                                                                                    \
  template Status ArgMax<T>(const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result, \
                            Stream& stream, void* buffer_ptr);

INSTANTIATE_ARG_MAX(float);
INSTANTIATE_ARG_MAX(float16);
INSTANTIATE_ARG_MAX(bfloat16);

#undef INSTANTIATE_ARG_MAX

}  // namespace ksana_llm
