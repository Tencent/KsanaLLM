/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>

#include "ksana_llm/models/base/base_model_config.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

// 张量并行切分模式
enum TensorParallelMode {
  RowPara,  // 按行切分（沿第0维切分）
  ColPara,  // 按列切分（沿第1维切分）
};

class CommonModelWeightLoader {
 public:
  CommonModelWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                          std::shared_ptr<Context> context);

  ~CommonModelWeightLoader() = default;

  Status LoadMhaWeights(const std::string& host_weight_name, const Tensor& host_weight_tensor,
                        std::unordered_map<std::string, Tensor>& device_model_weights, int dev_rank, size_t num_heads,
                        size_t num_kv_heads, size_t size_per_head);

  // Reuse temporary created tensor while processing weights
  // these tensors should not be inserted into device_model_weights
  // Careful with this function because it may cause memory issue
  Tensor& GetTempTensor(const std::vector<size_t>& shape, DataType data_type, int dev_rank);

  // Permutation with buffer
  Status PermuteWeight(Tensor& input_tensor, const std::vector<size_t>& permutation, int dev_rank);

  // Permutation with buffer (auto permutation for 2D tensors)
  Status Permute2D(Tensor& input_tensor, int dev_rank);

  // Split weight along axis = 0, then with param `skip_transpose` to decide whether skip transpose
  Status SplitOptTrans(const Tensor& host_weight_tensor, Tensor& output_tensor, int dev_rank, size_t para_size,
                       bool transpose);

  // Transpose and split weight along axis = 0, then with param `transpose` to decide whether to transpose back
  Status TransSplitOptTrans(const Tensor& host_weight_tensor, Tensor& output_tensor, int dev_rank, size_t para_size,
                            bool transpose);

  // 张量并行切分：支持1D/2D张量按指定模式（行/列）切分到多个设备
  Tensor TensorParallelSplit(const Tensor& input_tensor, int dev_rank, size_t para_size, TensorParallelMode tp_mode);

  // 将主机张量拷贝到指定设备（GPU）
  Tensor MoveToDevice(const Tensor& host_tensor, int dev_rank);

  // 自动合并多个权重张量为一个，并从权重映射中删除原权重
  Status AutoMergeWeight(const std::vector<std::string>& inputs_weight_name, const std::string& merge_weight_name,
                         std::unordered_map<std::string, Tensor>& dev_weights_map, int dev_rank);

  // 转换设备张量的数据类型（支持FP32/FP16/BF16之间的转换）
  Status CastDeviceTensorType(Tensor& input_tensor, DataType new_dtype, int dev_rank);

  // 将Tensor转换为torch::Tensor（零拷贝，共享底层内存）
  torch::Tensor GetTorchTensorFromTensor(const Tensor& tensor);

  // 将torch::Tensor转换为Tensor（拷贝数据到新Tensor）
  Tensor GetTensorFromTorchTensor(const torch::Tensor& torch_tensor);

  // 获取context指针
  std::shared_ptr<Context> GetContext() const { return context_; }

 private:
  // 沿第0维拼接多个张量（要求第1维大小相同）
  Status Concat(const std::vector<Tensor>& inputs_tensor, Tensor& output, int dev_rank);

  // 根据源张量和目标张量的内存位置自动判断内存拷贝类型（Host/Device）
  MemcpyKind GetMemcpyKind(const Tensor& from_tensor, const Tensor& to_tensor);

  // 1D张量并行切分：将1D张量按指定切片形状切分到指定设备
  Tensor TensorParallelSplit1D(const Tensor& input_tensor, const std::vector<size_t>& slice_shape, int dev_rank,
                               size_t para_size);

  // 2D张量并行切分：按行或列模式切分2D张量（列切分时会自动转置）
  Tensor TensorParallelSplit2D(const Tensor& input_tensor, int dev_rank, size_t para_size, TensorParallelMode tp_mode);

  std::shared_ptr<BaseModelConfig> model_config_;
  std::shared_ptr<Environment> env_;
  std::shared_ptr<Context> context_;

  std::vector<std::unordered_map<size_t, Tensor>> permute_buffers_;
  std::vector<std::unordered_map<size_t, Tensor>> tensor_buffers_;
};

}  // namespace ksana_llm
