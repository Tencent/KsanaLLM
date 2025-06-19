/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/kernels/trans_layout.h"

#include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {
Status TransLayout(Tensor& tensor, Stream& stream) {
  // NOTE(karlluo): 2 condition: shape[0] and shape[1] must divide by 16 and only support FP16
  // TODO(karlluo): support INT8
  if ((tensor.shape[1] % 16 != 0 || tensor.shape[0] % 16 != 0) || tensor.dtype != TYPE_FP16) {
    return Status();
  }
  constexpr size_t TRANSDATA_IN_DIMS = 3ul;
  constexpr size_t TRANSDATA_TRUNK = 16ul;
  std::vector<size_t> src_shape = tensor.shape;
  std::vector<size_t> in_tensor_shape = src_shape;
  // NOTE(karlluo): transdata need 3 dims input
  if (src_shape.size() < TRANSDATA_IN_DIMS) {
    in_tensor_shape.clear();
    in_tensor_shape.resize(TRANSDATA_IN_DIMS, 1);
    for (size_t dim_idx = TRANSDATA_IN_DIMS - src_shape.size(), src_dim_idx = 0;
         dim_idx < TRANSDATA_IN_DIMS && src_dim_idx < src_shape.size(); ++dim_idx, ++src_dim_idx) {
      in_tensor_shape[dim_idx] = src_shape[src_dim_idx];
    }
  } else if (src_shape.size() > TRANSDATA_IN_DIMS) {
    in_tensor_shape.clear();
    in_tensor_shape.resize(TRANSDATA_IN_DIMS, 1);
    in_tensor_shape[0] = src_shape[0];
    in_tensor_shape[1] = src_shape[1];
    size_t final_dim = 1ul;
    for (size_t dim_idx = 2ul; dim_idx < src_shape.size(); ++dim_idx) {
      final_dim *= src_shape[dim_idx];
    }
    in_tensor_shape[2] = final_dim;
  }

  if (in_tensor_shape.at(2) % TRANSDATA_TRUNK != 0) {
    return Status();
  }

  std::vector<size_t> out_tensor_shape = {in_tensor_shape.at(0), in_tensor_shape.at(2) / TRANSDATA_TRUNK,
                                          in_tensor_shape.at(1), TRANSDATA_TRUNK};

  const aclFormat src_format = GetACLFormat(tensor.data_format);
  aclFormat target_format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
  atb::infer::TransdataParam::TransdataType transdata_type =
      atb::infer::TransdataParam::TransdataType::ND_TO_FRACTAL_NZ;
  atb::infer::TransdataParam transdata_param;
  if (src_format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
    // for nz to nd
    // NOTE(karlluo): nz to nd only support shape size > 2
    // ref:
    // https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/ascendtbapi/ascendtb_01_0083.html
    target_format = aclFormat::ACL_FORMAT_ND;
    transdata_type = atb::infer::TransdataParam::TransdataType::FRACTAL_NZ_TO_ND;
    transdata_param.outCrops[0] = tensor.shape[in_tensor_shape.size() - 2];
    transdata_param.outCrops[1] = tensor.shape[in_tensor_shape.size() - 1] * tensor.shape[in_tensor_shape.size() - 3];
  }
  void* workspace_ptr;
  Malloc(&workspace_ptr, tensor.GetTotalBytes());
  transdata_param.transdataType = transdata_type;
  llm_kernels::utils::ATBOperationExecutor atb_op_executor;
  int32_t rank;
  GetDevice(&rank);
  atb_op_executor.Init(rank, transdata_param);
  atb_op_executor.SetInputTensor(tensor.GetPtr<void>(), in_tensor_shape,
                                 static_cast<aclDataType>(DataType(tensor.dtype)), src_format);
  atb_op_executor.SetOutputTensor(workspace_ptr, out_tensor_shape, static_cast<aclDataType>(DataType(tensor.dtype)),
                                  target_format);
  atb_op_executor.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank)), GetWorkSpaceFunc());
  tensor.data_format = GetTensorFormat(target_format);
  tensor.shape = out_tensor_shape;
  MemcpyAsync(tensor.GetPtr<void>(), workspace_ptr, tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE, stream);
  StreamSynchronize(stream);
  Free(workspace_ptr);
  return Status();
}
}  // namespace ksana_llm
