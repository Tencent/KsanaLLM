/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
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
 * [vLLM Project]
 * https://github.com/vllm-project/vllm/blob/9798b2fb0052092a6420172e41c0c8a307eedfa6/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh
 * [vLLM Project]
 * https://github.com/vllm-project/vllm/blob/9798b2fb0052092a6420172e41c0c8a307eedfa6/csrc/quantization/cutlass_w8a8/c3x/cutlass_gemm_caller.cuh
 * [vLLM Project]
 * https://github.com/vllm-project/vllm/blob/9798b2fb0052092a6420172e41c0c8a307eedfa6/csrc/cutlass_extensions/common.hpp
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"

#include "csrc/kernels/nvidia/blockwise_gemm/collective/collective_builder.hpp"
#include "csrc/kernels/nvidia/blockwise_gemm/dispatch_policy.hpp"

#include "cutlass_gemm_caller.cuh"

#include <cuda_runtime.h>

namespace ksana_llm {
using namespace cute;

template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

template <typename SchedulerType, typename OutType, int GroupSizeM_, int GroupSizeN_, int GroupSizeK_,
          int TileSizeM_ = 128, class ClusterShape = Shape<_1, _2, _1>>
struct Cutlass3xGemmFp8Blockwise {
  using GroupSizeM = Int<GroupSizeM_>;
  using GroupSizeN = Int<GroupSizeN_>;
  using GroupSizeK = Int<GroupSizeK_>;
  using TileSizeM = Int<TileSizeM_>;
  static_assert(TileSizeM_ % GroupSizeM_ == 0, "TileSizeM must be a multiple of GroupSizeM");

  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using StrideD = Stride<int64_t, Int<1>, Int<0>>;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using StrideC = StrideD;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementBlockScale = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<TileSizeM, GroupSizeN, GroupSizeK>;

  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<GroupSizeM_>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using StoreEpilogueCompute = typename cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementCompute, ElementC,
      StrideC, AlignmentC, ElementD, StrideD, AlignmentD, EpilogueSchedule, StoreEpilogueCompute>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using KernelType =
      enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                                CollectiveEpilogue, SchedulerType>>;

  struct GemmKernel : public KernelType {};

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
};

template <typename Gemm>
typename Gemm::GemmKernel::Arguments GetCutlassGemmBlockwiseKernelArgs(const void* a, const float* a_scales,
                                                                       const void* b, const float* b_scales, void* out,
                                                                       int m, int k, int n) {
  using GemmKernel = typename Gemm::GemmKernel;

  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  cute::Shape<int, int, int, int> prob_shape = {m, n, k, 1};

  int64_t lda = k;
  int64_t ldb = k;
  int64_t ldc = n;

  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideB = Stride<int64_t, Int<1>, int64_t>;
  using StrideC = typename Gemm::StrideC;

  StrideA a_stride{lda, Int<1>{}, 0};
  StrideB b_stride{ldb, Int<1>{}, 0};
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a_ptr = static_cast<ElementAB*>(const_cast<void*>(a));
  auto b_ptr = static_cast<ElementAB*>(const_cast<void*>(b));
  auto a_scales_ptr = const_cast<float*>(a_scales);
  auto b_scales_ptr = const_cast<float*>(b_scales);

  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr, b_stride, a_scales_ptr, b_scales_ptr};

  auto c_ptr = static_cast<ElementD*>(out);
  typename GemmKernel::EpilogueArguments epilogue_args{{}, c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::TileSchedulerArguments scheduler;

  static constexpr bool UsesStreamKScheduler =
      cute::is_same_v<typename GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>;

  if constexpr (UsesStreamKScheduler) {
    using DecompositionMode =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
    using ReductionMode =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90StreamKParams::ReductionMode;

    scheduler.decomposition_mode = DecompositionMode::StreamK;
    scheduler.reduction_mode = ReductionMode::Nondeterministic;
  }

  cutlass::KernelHardwareInfo hw_info;
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args, epilogue_args, hw_info, scheduler};
  return args;
}

template <typename SchedulerType, typename OutType>
void InvokeGemmBlockwise(void* a, float* a_scales, void* b, float* b_scales, void* out, void* workspace,
                         size_t workspace_size, int m, int k, int n, cudaStream_t& stream) {
  using GemmFp8Blockwise = Cutlass3xGemmFp8Blockwise<SchedulerType, OutType, 1, 128, 128>;
  auto args = GetCutlassGemmBlockwiseKernelArgs<GemmFp8Blockwise>(a, a_scales, b, b_scales, out, m, k, n);
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<typename GemmFp8Blockwise::GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op.run(args, workspace, stream);
  CUTLASS_CHECK(status);
}

template <typename OutType>
void DispatchCutlassGemmBlockwiseSm90Fp8(void* a, float* a_scales, void* b, float* b_scales, void* out, void* workspace,
                                         size_t workspace_size, int m, int k, int n, cudaStream_t& stream) {
  // NOTE(karlluo): adapted from
  // https://github.com/vllm-project/vllm/blob/v0.8.4/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh#L183

  if (k > 3 * n && workspace != nullptr && workspace_size > 0) {
    // NOTE(karlluo): Use StreamKScheduler if k is large enough and workspace is provided
    InvokeGemmBlockwise<cutlass::gemm::StreamKScheduler, OutType>(
        a, a_scales, b, b_scales, out, workspace, workspace_size, m, k, n, stream);
  } else {
    // NOTE(karlluo): Use PersistentScheduler if k is small or workspace is not provided but performance regression
    InvokeGemmBlockwise<cutlass::gemm::PersistentScheduler, OutType>(
        a, a_scales, b, b_scales, out, workspace, workspace_size, m, k, n, stream);
  }
}

template <typename SchedulerType, typename OutType>
size_t GetGemmBlockwiseWorkspace(int m, int k, int n) {
  using GemmFp8Blockwise = Cutlass3xGemmFp8Blockwise<SchedulerType, OutType, 1, 128, 128>;
  auto args = GetCutlassGemmBlockwiseKernelArgs<GemmFp8Blockwise>(nullptr, nullptr, nullptr, nullptr, nullptr, m, k, n);
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<typename GemmFp8Blockwise::GemmKernel>;
  GemmOp gemm_op;
  return gemm_op.get_workspace_size(args);
}

template <typename OutType>
size_t GetCutlassGemmBlockwiseSm90Fp8Workspace(int m, int k, int n) {
  if (k > 3 * n) {
    return GetGemmBlockwiseWorkspace<cutlass::gemm::StreamKScheduler, OutType>(m, k, n);
  } else {
    return GetGemmBlockwiseWorkspace<cutlass::gemm::PersistentScheduler, OutType>(m, k, n);
  }
}

}  // namespace ksana_llm
