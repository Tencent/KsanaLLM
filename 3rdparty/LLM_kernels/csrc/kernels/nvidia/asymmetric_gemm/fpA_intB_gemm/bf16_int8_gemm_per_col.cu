/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
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

#include "csrc/kernels/nvidia/asymmetric_gemm/fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace llm_kernels {
namespace nvidia {

template class CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>;

}  // namespace nvidia
}  // namespace llm_kernels
