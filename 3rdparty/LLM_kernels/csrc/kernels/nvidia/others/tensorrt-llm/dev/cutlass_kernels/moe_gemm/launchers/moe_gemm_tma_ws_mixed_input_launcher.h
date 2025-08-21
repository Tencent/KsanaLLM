/*
 * Adapted from
 * [TensorRT-LLM Project]
 * https://github.com/NVIDIA/TensorRT-LLM/tree/v1.0.0rc3
 */

/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime_api.h>
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_extensions/gemm_configs.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_extensions/weight_only_quant_op.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/cutlass_kernels/include/moe_gemm_kernels.h"

namespace llm_kernels::nvidia::tensorrt_llm::dev {
namespace kernels {
namespace cutlass_kernels {

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag, typename CTAShape,
          typename ClusterShape, typename MainloopScheduleType, typename EpilogueScheduleType,
          cutlass::WeightOnlyQuantOp QuantOp>
void sm90_generic_mixed_moe_gemm_kernelLauncher(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
                                                TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_,
                                                size_t* workspace_size);

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace llm_kernels::nvidia::tensorrt_llm::dev
