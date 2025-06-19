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
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/quantization/machete/machete_mm_launcher.cuh
 */

#pragma once

#include "csrc/kernels/nvidia/machete/cutlass_extensions/torch_utils.hpp"
#include "csrc/kernels/nvidia/machete/machete_mm_kernel.cuh"
#include "csrc/utils/nvidia/scalar_type.hpp"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace machete {

struct MMArgs {
  int64_t& workspace_size;
  void* workspace;

  cudaStream_t stream;
  int M;
  int N;
  int K;
  const void* Aptr;
  const std::vector<size_t> A_shape;
  const void* Bptr;
  void* Dptr;
  const std::vector<size_t> D_shape;

  llm_kernels::nvidia::vllm_dtype::ScalarType const& a_type;
  llm_kernels::nvidia::vllm_dtype::ScalarType const& b_type;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_out_type;

  std::optional<void*> const& maybe_group_scales_ptr;
  std::optional<std::vector<size_t>> const& maybe_group_scales_shape;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_scales_type;

  std::optional<void*> const& maybe_group_zeros_ptr;
  std::optional<std::vector<size_t>> const& maybe_group_zeros_shape;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_zeros_type;

  std::optional<int64_t> maybe_group_size;

  std::optional<void*> const& maybe_channel_scales_ptr;
  std::optional<int64_t> const& maybe_channel_scales_numel;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_channel_scales_type;

  std::optional<void*> const& maybe_token_scales_ptr;
  std::optional<int64_t> const& maybe_token_scales_numel;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_token_scales_type;

  std::optional<std::string> maybe_schedule;
};

struct SupportedSchedulesArgs {
  llm_kernels::nvidia::vllm_dtype::ScalarType a_type;
  llm_kernels::nvidia::vllm_dtype::ScalarType b_type;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_group_scales_type;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_group_zeros_type;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_channel_scales_type;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_token_scales_type;
  std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_out_type;
};

void mm_dispatch(MMArgs args);

std::vector<std::string> supported_schedules_dispatch(SupportedSchedulesArgs args);

template <typename MacheteKernel>
void run_impl(MMArgs args) {
  auto arguments = MacheteKernel::create_arguments(
      args.stream, args.M, args.N, args.K, args.Aptr, args.A_shape, args.Bptr, args.Dptr, args.D_shape,
      args.maybe_group_scales_ptr, args.maybe_group_scales_shape, args.maybe_group_zeros_ptr,
      args.maybe_group_zeros_shape, args.maybe_group_size, args.maybe_channel_scales_ptr,
      args.maybe_channel_scales_numel, args.maybe_token_scales_ptr, args.maybe_token_scales_numel);
  KLLM_KERNEL_CHECK_WITH_INFO(MacheteKernel::can_implement(arguments),
                              "Machete kernel cannot be run with these arguments");

  // 构建gemm后才能获取workspace size，实际run前需要调用一次获取大小
  if (args.workspace_size == -1) {
    args.workspace_size = MacheteKernel::get_workspace_size(arguments);
  } else {
    MacheteKernel::run(arguments, args.workspace, args.stream);
  }
};

}  // namespace machete
}  // namespace nvidia
}  // namespace llm_kernels