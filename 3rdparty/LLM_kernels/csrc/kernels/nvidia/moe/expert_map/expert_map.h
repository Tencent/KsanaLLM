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

 private:
  int32_t start_expert_;
  int32_t end_expert_;
};

}  // namespace moe
}  // namespace nvidia
}  // namespace llm_kernels