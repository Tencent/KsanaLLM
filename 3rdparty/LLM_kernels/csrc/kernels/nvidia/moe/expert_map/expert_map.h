/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {
namespace moe {

class ExpertMap {
 public:
  ExpertMap(size_t ep_size, size_t ep_rank, size_t expert_num);

  void InvokeExpertMapInplace(int32_t* data, size_t data_size, cudaStream_t stream);

  // 原地(in-place)逆映射：
  // - 对于不为 -1 的本地 id，加上 start_expert_ 恢复为全局 expert id；
  // - 对于 -1 保持不变（表示该 token 不属于当前 rank 的任何 expert）。
  void InvokeExpertMapInverseInplace(int32_t* data, size_t data_size, cudaStream_t stream);

  // 原地(in-place)映射： 返回 expert_map[expert_id]
  void InvokeExpertMapInplace(int32_t* data, size_t data_size, int32_t* expert_map, cudaStream_t stream);

 private:
  int32_t start_expert_;
  int32_t end_expert_;
};

}  // namespace moe
}  // namespace nvidia
}  // namespace llm_kernels