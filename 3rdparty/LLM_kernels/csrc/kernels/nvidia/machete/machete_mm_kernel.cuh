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
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/quantization/machete/machete_mm_kernel.cuh
 */

#pragma once

// clang-format off
// The cutlass include order matters (annoyingly)
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
// clang-format on

#include "csrc/kernels/nvidia/machete/cutlass_extensions/cute_utils.cuh"
#include "csrc/kernels/nvidia/machete/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "csrc/kernels/nvidia/machete/cutlass_extensions/torch_utils.hpp"
#include "csrc/kernels/nvidia/machete/cutlass_extensions/vllm_numeric_conversion.cuh"
#include "csrc/kernels/nvidia/machete/machete_collective_builder.cuh"
#include "csrc/kernels/nvidia/machete/machete_interleaving_utils.cuh"
#include "csrc/kernels/nvidia/machete/machete_prepacked_layout.cuh"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace machete {

using namespace cute;

// NOTE This kernel computes D = alpha * A * B + beta * C by computing
//   D^t = alpha * B^t * A^t + beta * C^t, this is because the wgmma
//   instructions only support sourcing from registers for the left-hand
//   operand, we want to upconvert/decompress the quantized operand in
//   register. Since the primary use case we want to support is Y = XW^t where
//   W is quantized, in this situation or right-hand operand is quantized so
//   we compute the transpose to move it to the left-hand side.
template <typename ElementA_, typename ElementB_, typename ElementD_, typename AccumulatorT, typename GroupScaleT,
          typename GroupZeroT, typename ChannelScaleT, typename TokenScaleT, class KernelSchedule,
          typename ScheduleConfig>
struct MacheteKernelTemplate {
  static constexpr bool with_C = false;  // not ever used
  static constexpr bool with_group_scales = !std::is_same_v<GroupScaleT, void>;
  static constexpr bool with_group_zeropoints = !std::is_same_v<GroupZeroT, void>;
  static constexpr bool with_channel_scales = !std::is_same_v<ChannelScaleT, void>;
  static constexpr bool with_token_scales = !std::is_same_v<TokenScaleT, void>;

  using MmaType = ElementA_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementD = ElementD_;
  using ElementC = cute::conditional_t<with_C, ElementD, void>;
  using ElementAccumulator = AccumulatorT;
  using ElementCompute = AccumulatorT;  // For Epilogue
  // Use dummy values when we don't have scales or zeropoints
  using ElementZGroup = cute::conditional_t<with_group_zeropoints, GroupZeroT, MmaType>;
  using ElementSGroup = cute::conditional_t<with_group_scales, GroupScaleT, MmaType>;
  using ElementConvertGroup = cute::conditional_t<with_group_scales, GroupScaleT, MmaType>;
  using ElementSChannel = cute::conditional_t<with_channel_scales, ChannelScaleT, AccumulatorT>;
  using ElementSToken = cute::conditional_t<with_token_scales, TokenScaleT, AccumulatorT>;

  using BTypeTuple = cute::conditional_t<
      with_group_scales,
      cute::conditional_t<with_group_zeropoints, cute::tuple<ElementB, ElementSGroup, ElementZGroup>,
                          cute::tuple<ElementB, ElementSGroup>>,
      ElementB>;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;
  using LayoutScale = cutlass::layout::RowMajor;
  // not actually used since B has the prepacked layout, but required by cutlass
  using _LayoutB = cutlass::layout::ColumnMajor;

  // Interface strides expected by create_arguments (will get transposed)
  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideC = cutlass::detail::TagToStrideA_t<LayoutC>;
  using StrideD = cutlass::detail::TagToStrideA_t<LayoutD>;
  using StrideSGroup = cutlass::detail::TagToStrideA_t<LayoutScale>;
  using StrideZGroup = StrideSGroup;

  using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
  using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using PrepackedLayoutB = PrepackedLayoutBTemplate<ElementA_, ElementB_, ElementConvertGroup, AccumulatorT,
                                                    LayoutA_Transpose, KernelSchedule>;

  static int constexpr TileShapeK = 128 * 8 / cutlass::sizeof_bits<MmaType>::value;
  static int constexpr AlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>;
  static int constexpr AlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>;
  static int constexpr AlignmentC = (with_C) ? 128 / cutlass::sizeof_bits_v<ElementC> : 0;
  static int constexpr AlignmentD = 128 / cutlass::sizeof_bits_v<ElementD>;

  using TileShape = decltype(append(typename ScheduleConfig::TileShapeNM{}, cute::Int<TileShapeK>{}));
  using ClusterShape = typename ScheduleConfig::ClusterShape;
  using EpilogueSchedule = typename ScheduleConfig::EpilogueSchedule;
  using EpilogueTileType = typename ScheduleConfig::EpilogueTileType;
  using TileScheduler = typename ScheduleConfig::TileScheduler;

  static_assert((!with_channel_scales && !with_token_scales) ||
                    ((with_channel_scales && with_token_scales) && std::is_same_v<ElementSChannel, ElementSToken>),
                "Currently token and channel scales (if present) must be the same type");

  // Currently only supports float scales
  using ChTokScalesEpilogue = typename vllm::c3x::ScaledEpilogue<ElementAccumulator, ElementD, TileShape>;
  static_assert((with_channel_scales || with_token_scales) ||
                    (std::is_same_v<ElementSChannel, float> && std::is_same_v<ElementSToken, float>),
                "Currently token and channel scales (if present) must be float "
                "(and if one is present the other must be too)");

  using StoreEpilogueCompute = typename cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>;

  using EVTCompute = std::conditional_t<with_channel_scales || with_token_scales,
                                        typename ChTokScalesEpilogue::EVTCompute, StoreEpilogueCompute>;

  // EVTCompute
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementSChannel, ElementC,
      LayoutC_Transpose, AlignmentC, ElementD, LayoutD_Transpose, AlignmentD, EpilogueSchedule,
      EVTCompute>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::VLLMCollectiveBuilder<
      cutlass::gemm::collective::MacheteKernelTag, ArchTag, OperatorClass, BTypeTuple, PrepackedLayoutB, AlignmentB,
      ElementA, LayoutA_Transpose, AlignmentA, ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,  // Indicates ProblemShape
                                                          CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // stride_B is unused (since B is prepacked), but still required by cutlass
  using _StrideB = cutlass::detail::TagToStrideB_t<_LayoutB>;

  using Arguments = typename Gemm::Arguments;
  using MainloopArguments = typename GemmKernel::MainloopArguments;
  using EpilogueArguments = typename GemmKernel::EpilogueArguments;

  static Arguments create_arguments(
      cudaStream_t stream, int M, int N, int K,
      // torch::Tensor const& A,  // MxK matrix
      const void* Aptr, const std::vector<size_t>& A_shape,
      // torch::Tensor const& B,  // KxN prepacked matrix
      const void* Bptr,
      // torch::Tensor& D,        // MxN matrix
      void* Dptr, const std::vector<size_t>& D_shape,
      // std::optional<torch::Tensor> const& maybe_g_scales,  // scale_KxN matrix
      std::optional<void*> const& maybe_g_scales_ptr, std::optional<std::vector<size_t>> const& maybe_g_scales_shape,
      // std::optional<torch::Tensor> const& maybe_g_zeros,   // scale_KxN matrix
      std::optional<void*> const& maybe_g_zeros_ptr, std::optional<std::vector<size_t>> const& maybe_g_zeros_shape,
      std::optional<int64_t> maybe_group_size,
      // std::optional<torch::Tensor> const& maybe_ch_scales,   // len N vector
      std::optional<void*> const& maybe_ch_scales_ptr, std::optional<int64_t> maybe_ch_scales_numel,
      // std::optional<torch::Tensor> const& maybe_tok_scales)  // len M vector
      std::optional<void*> const& maybe_tok_scales_ptr, std::optional<int64_t> maybe_tok_scales_numel) {
    static_assert(!with_group_zeropoints || with_group_scales);

    // auto layout_A = make_cute_layout<StrideA>(A, "A");
    auto layout_A = make_cute_layout_from_shape<StrideA>(A_shape, "A");
    auto layout_D = make_cute_layout_from_shape<StrideD>(D_shape, "D");
    auto layout_S_group = maybe_make_cute_layout_from_shape<StrideSGroup>(maybe_g_scales_shape, "group_scales");
    auto layout_Z_group = maybe_make_cute_layout_from_shape<StrideZGroup>(maybe_g_zeros_shape, "group_zeros");
    int64_t numel_S_channel = maybe_ch_scales_numel ? *maybe_ch_scales_numel : 0;
    int64_t numel_S_token = maybe_tok_scales_numel ? *maybe_tok_scales_numel : 0;

    auto unwrap_ptr = [](auto const& t) { return t ? *t : nullptr; };
    auto A_ptr = static_cast<ElementA const*>(Aptr);
    auto B_ptr = static_cast<ElementB const*>(Bptr);
    auto D_ptr = static_cast<ElementD*>(Dptr);
    auto S_group_ptr = static_cast<ElementSGroup const*>(unwrap_ptr(maybe_g_scales_ptr));
    auto Z_group_ptr = static_cast<ElementZGroup const*>(unwrap_ptr(maybe_g_zeros_ptr));
    auto S_channel_ptr = static_cast<ElementSChannel const*>(unwrap_ptr(maybe_ch_scales_ptr));
    auto S_token_ptr = static_cast<ElementSToken const*>(unwrap_ptr(maybe_tok_scales_ptr));

    int const group_size = maybe_group_size == -1 ? K : maybe_group_size.value_or(K);
    int const scale_k = (K + group_size - 1) / group_size;

    KLLM_KERNEL_CHECK(size<0>(layout_A) == M && size<1>(layout_A) == K);
    KLLM_KERNEL_CHECK(size<0>(layout_D) == M && size<1>(layout_D) == N);

    if constexpr (with_group_scales) {
      KLLM_KERNEL_CHECK(S_group_ptr && layout_S_group);
      KLLM_KERNEL_CHECK((size<0>(*layout_S_group) == scale_k && size<1>(*layout_S_group) == N));
    } else {
      KLLM_KERNEL_CHECK_WITH_INFO(!S_group_ptr, "Scales not supported");
    }

    if constexpr (with_group_zeropoints) {
      KLLM_KERNEL_CHECK(Z_group_ptr && layout_Z_group);
      KLLM_KERNEL_CHECK((size<0>(*layout_Z_group) == scale_k && size<1>(*layout_Z_group) == N));
      KLLM_KERNEL_CHECK_WITH_INFO(layout_S_group && *layout_Z_group == *layout_S_group,
                                  "Scales and zeros must have the same layout");
    } else {
      KLLM_KERNEL_CHECK_WITH_INFO(!Z_group_ptr, "Zeropoints not supported");
    }

    if constexpr (with_channel_scales || with_token_scales) {
      KLLM_KERNEL_CHECK((*maybe_ch_scales_numel == N || *maybe_ch_scales_numel == 1) &&
                        (*maybe_tok_scales_numel == M || *maybe_tok_scales_numel == 1));
    }

    // Transpose A and D
    // A doesn't need to be transposed since cutlass expects a NxK matrix
    //  for B (which is At)
    auto stride_At = layout_A.stride();
    auto stride_Dt = permute_layout<1, 0, 2>(layout_D).stride();

    MainloopArguments mainloop_arguments{};
    // {Accum, C, C_layout, D, D}
    EpilogueArguments epilogue_arguments{};

    if constexpr (with_channel_scales || with_token_scales) {
      epilogue_arguments =
          EpilogueArguments{ChTokScalesEpilogue::prepare_args(*maybe_ch_scales_ptr, *maybe_ch_scales_numel,
                                                              *maybe_tok_scales_ptr, *maybe_tok_scales_numel),
                            nullptr,
                            {},
                            D_ptr,
                            stride_Dt};
    } else {
      epilogue_arguments = EpilogueArguments{{}, nullptr, {}, D_ptr, stride_Dt};
    }

    if constexpr (with_group_scales && with_group_zeropoints) {
      auto stride_S_group = permute_layout<1, 0, 2>(*layout_S_group).stride();
      mainloop_arguments =
          MainloopArguments{B_ptr, _StrideB{}, A_ptr, stride_At, S_group_ptr, stride_S_group, group_size, Z_group_ptr};
    } else if constexpr (with_group_scales) {
      auto stride_S_group = permute_layout<1, 0, 2>(*layout_S_group).stride();
      mainloop_arguments =
          MainloopArguments{B_ptr, _StrideB{}, A_ptr, stride_At, S_group_ptr, stride_S_group, group_size};
    } else {
      mainloop_arguments = MainloopArguments{B_ptr, _StrideB{}, A_ptr, stride_At};
    }

    return Arguments{cutlass::gemm::GemmUniversalMode::kGemm, {N, M, K, 1}, mainloop_arguments, epilogue_arguments};
  };

  static size_t get_workspace_size(Arguments const& args) { return Gemm::get_workspace_size(args); }

  static bool can_implement(Arguments const& args) { return Gemm::can_implement(args) == cutlass::Status::kSuccess; }

  static void run(Arguments const& args, void* workspace, cudaStream_t stream) {
    Gemm gemm_op;

    cutlass::Status status = gemm_op.initialize(args, workspace, stream);
    KLLM_KERNEL_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "Machete kernel failed to initialize workspace");

    status = gemm_op.run(stream);
    KLLM_KERNEL_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "Machete kernel failed");
  }
};

}  // namespace machete
}  // namespace nvidia
}  // namespace llm_kernels
