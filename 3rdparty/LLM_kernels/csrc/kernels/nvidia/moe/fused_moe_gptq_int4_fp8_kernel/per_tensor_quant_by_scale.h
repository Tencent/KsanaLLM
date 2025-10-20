/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

template <class InputT, class OutputT>
void per_tensor_quant_by_scale_launcher(void* output, const void* input, const void* scales, const size_t num_elements,
                                        cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
