/*
 * Copyright 2025 vLLM Team
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
 *
 * Adapted from
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/9798b2fb0052092a6420172e41c0c8a307eedfa6/csrc/quantization/cutlass_w8a8/c3x/cutlass_gemm_caller.cuh
 */

#pragma once
// clang-format will break include orders
// clang-format off
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
// clang-format on
namespace ksana_llm::c3x {

#define CUTLASS_CHECK(status)                                                                                  \
  {                                                                                                            \
    cutlass::Status error = status;                                                                            \
    if (error != cutlass::Status::kSuccess) {                                                                  \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl; \
      exit(EXIT_FAILURE);                                                                                      \
    }                                                                                                          \
  }

template <typename GemmKernel>
void cutlass_gemm_caller(cute::Shape<int, int, int, int> prob_shape,
                         typename GemmKernel::MainloopArguments mainloop_args,
                         typename GemmKernel::EpilogueArguments epilogue_args, void* workspace, cudaStream_t& stream,
                         typename GemmKernel::TileSchedulerArguments scheduler = {}) {
  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args, epilogue_args, hw_info, scheduler};
  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));
  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::Status status = gemm_op.run(args, workspace, stream);
  CUTLASS_CHECK(status);
}

 }  // namespace ksana_llm::c3x
 