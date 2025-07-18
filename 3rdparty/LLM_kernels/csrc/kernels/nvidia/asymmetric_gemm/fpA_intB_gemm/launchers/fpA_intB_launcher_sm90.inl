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

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "csrc/kernels/nvidia/cutlass_extensions/compute_occupancy.h"
#include "csrc/kernels/nvidia/cutlass_extensions/epilogue_helpers.h"
#include "csrc/kernels/nvidia/cutlass_extensions/gemm_configs.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/assert.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_heuristic.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_type_conversion.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.h"

using namespace llm_kernels::utils;

using namespace cute;

namespace llm_kernels {
namespace nvidia {

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename CTAShape, typename ClusterShape,
    typename MainloopScheduleType, typename EpilogueScheduleType>
void sm90_generic_mixed_gemm_kernelLauncher(ActivationType const* A, WeightType const* B,
    ScaleZeroType const* weight_scales, ScaleZeroType const* weight_zero_points, BiasType const* biases,
    float const alpha, OutputType* C, int m, int n, int k, int const group_size, llm_kernels::nvidia::cutlass_extensions::CutlassGemmConfig gemm_config,
    char* workspace, size_t workspace_bytes, cudaStream_t stream, int* occupancy)
{

#ifdef COMPILE_HOPPER_TMA_GEMMS
    using CutlassActivationType = typename TllmToCutlassTypeAdapter<ActivationType>::type;

    if constexpr (!should_filter_sm90_gemm_problem_shape_v<CTAShape, ClusterShape, ActivationType>)
    {
        using CutlassWeightType__ = typename TllmToCutlassTypeAdapter<WeightType>::type;
        // We need to remap this since SM90 uses a different layout for the weight matrix.
        using CutlassWeightType_ = std::conditional_t<std::is_same_v<CutlassWeightType__, cutlass::uint4b_t>,
            cutlass::int4b_t, CutlassWeightType__>;

        using CutlassWeightType
            = std::conditional_t<std::is_same_v<CutlassWeightType_, uint8_t>, int8_t, CutlassWeightType_>;

        using CutlassScaleZeroType = typename TllmToCutlassTypeAdapter<ScaleZeroType>::type;
        using CutlassBiasType = typename TllmToCutlassTypeAdapter<BiasType>::type;
        using CutlassOutputType = typename TllmToCutlassTypeAdapter<OutputType>::type;

        static_assert(std::is_same_v<CutlassActivationType, cutlass::half_t>
                || std::is_same_v<CutlassActivationType, cutlass::bfloat16_t>
                || std::is_same_v<CutlassActivationType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassActivationType, cutlass::float_e5m2_t>,
            "Activation type must be bfloat16, half, FP8");

        static_assert(std::is_same_v<CutlassWeightType, int8_t> || std::is_same_v<CutlassWeightType, cutlass::int4b_t>
                || std::is_same_v<CutlassWeightType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassWeightType, cutlass::float_e5m2_t>,
            "Weight type must be fp8, int8_t or int4_t");

        using LayoutA = cutlass::layout::RowMajor; // Layout type for A matrix operand
        constexpr int AlignmentA = 128 / cutlass::sizeof_bits<CutlassActivationType>::value;

        using LayoutB = cutlass::layout::ColumnMajor; // Layout type for B matrix operand
        constexpr int AlignmentB = 128 / cutlass::sizeof_bits<CutlassWeightType>::value;

        // This example manually swaps and transposes, so keep transpose of input layouts
        using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
        using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

        using ElementZero = CutlassScaleZeroType;
        using ElementScale = CutlassScaleZeroType;

        // C/D matrix configuration. We reuse the C operand for the bias and set the stride for broadcast.
        using LayoutBias = cutlass::layout::RowMajor;
        constexpr int AlignmentBias = 128 / cutlass::sizeof_bits<CutlassBiasType>::value;

        // D matrix configuration
        using LayoutOutput = cutlass::layout::RowMajor;
        constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<CutlassOutputType>::value;

        // Core kernel configurations
        using ElementAccumulator = float;    // Element type for internal accumulation
        using ElementCompute = float;        // Element type for epilogue computation
        using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that supports the intended feature
        using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
        using TileShape = CTAShape;                           // Threadblock-level tile size
        using KernelSchedule = MainloopScheduleType;
        using EpilogueSchedule = EpilogueScheduleType;

        // Shrink the N dimension to match CTA_N if needed
        constexpr int epi_tile_M = cute::min(shape<0>(TileShape{}), 128); // 64 or 128
        constexpr int epi_tile_N = cute::min(shape<1>(TileShape{}), 32);  // Allow this to be 16 for some small N tiles.
        using EpilogueTileType = cute::Shape<cute::Int<epi_tile_M>, cute::Int<epi_tile_N>>;

        static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
        static_assert(std::is_same_v<EpilogueTag, llm_kernels::nvidia::cutlass_extensions::EpilogueOpBias>, "");
        using EVT_bias_addition = cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add, CutlassOutputType, ElementCompute,
                RoundStyle>,                                                    // alpha * acc + bias
            cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAccumulator>, // alpha
            cutlass::epilogue::fusion::Sm90AccFetch,                            // acc
            cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, CutlassBiasType, CutlassBiasType, Stride<_1, _0, _0>,
                AlignmentBias>                                                  // bias
            >;

        using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
            TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementAccumulator,
            // Transpose layout of D here since we use the explicit swap + transpose trick
            // Void C since we don't use it. Prevents smem allocation.
            void, typename cutlass::layout::LayoutTranspose<LayoutBias>::type, AlignmentBias, CutlassOutputType,
            typename cutlass::layout::LayoutTranspose<LayoutOutput>::type, AlignmentOutput, EpilogueSchedule,
            EVT_bias_addition>::CollectiveOp;

        using PackedScaleZero = cute::tuple<CutlassWeightType, ElementScale, ElementZero>;
        using PackedScale = cute::tuple<CutlassWeightType, ElementScale>;
        using ElementBCollectiveInfo = std::conditional_t<cutlass::hasZero(QuantOp), PackedScaleZero, PackedScale>;

        // We swap A and B operands to the builder here
        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass,
            ElementBCollectiveInfo, LayoutB_Transpose, AlignmentB, CutlassActivationType, LayoutA_Transpose, AlignmentA,
            ElementAccumulator, TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
            KernelSchedule>::CollectiveOp;

        using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, // Indicates ProblemShape
            CollectiveMainloop, CollectiveEpilogue>;

        if (occupancy != nullptr)
        {
            *occupancy = llm_kernels::nvidia::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel, true>();
            return;
        }

        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

        using StrideA = typename GemmKernel::StrideA;
        using StrideB = typename GemmKernel::StrideB;
        using StrideC = typename GemmKernel::StrideC;
        using StrideD = typename GemmKernel::StrideD;
        using StrideS = typename CollectiveMainloop::StrideScale;

        if (weight_scales == nullptr)
        {
            throw std::runtime_error("Weight scales must always be set to a non-null value.");
        }

        if constexpr (cutlass::isFinegrained(QuantOp))
        {
            int cta_shape_k = cute::size<2>(TileShape{});
            if (group_size % cta_shape_k != 0)
            {
                std::string err_msg = "The group size must a multiple of " + std::to_string(cta_shape_k);
                throw std::runtime_error("[fpA_intB Runner]" + err_msg);
            }

            if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY)
            {
                if (weight_zero_points != nullptr)
                {
                    throw std::runtime_error("Weight zero pointer must be a nullptr for scale only fine grained");
                }
            }
            else if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
            {
                if (weight_zero_points == nullptr)
                {
                    throw std::runtime_error("Weight zero pointer must be valid for scale and bias fine grained");
                }
            }
        }
        else
        {
            if (group_size != k)
            {
                throw std::runtime_error("Invalid group size for per column scaling kernels.");
            }

            if (weight_zero_points != nullptr)
            {
                throw std::runtime_error("Weight zero-points must be null when running per column scaling");
            }
        }

        auto cutlass_scale_k = (k + group_size - 1) / group_size;
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
        StrideS stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, cutlass_scale_k, 1));

        // Use the output as the bias to avoid making a tma descriptor with a nullptr.
        auto output_as_bias_type = reinterpret_cast<CutlassBiasType const*>(C);

        typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1},
            {reinterpret_cast<CutlassWeightType const*>(B), stride_B, reinterpret_cast<CutlassActivationType const*>(A),
                stride_A, reinterpret_cast<ElementScale const*>(weight_scales), stride_S, group_size,
                reinterpret_cast<ElementZero const*>(weight_zero_points)},
            {{}, output_as_bias_type, stride_D, reinterpret_cast<CutlassOutputType*>(C), stride_D}};

        args.epilogue.thread = {
            {alpha},                                                                  // alpha args
            {},                                                                       // accumulator
            {reinterpret_cast<CutlassBiasType const*>(biases), CutlassBiasType(0.f)}, // bias args
            {}                                                                        // end multiply_add
        };

        Gemm gemm;
        if (gemm.get_workspace_size(args) > workspace_bytes)
        {
            std::runtime_error("[fpA_intB Runner] given workspace size insufficient.");
        }

        auto can_implement = gemm.can_implement(args);
        if (can_implement != cutlass::Status::kSuccess)
        {
            std::string err_msg = "fpA_intB cutlass kernel will fail for params. Error: "
                + std::string(cutlassGetStatusString(can_implement));
            std::cout << err_msg << std::endl;
            throw std::runtime_error("[fpA_intB Runner] " + err_msg);
        }

        auto init_status = gemm.initialize(args, workspace, stream);
        if (init_status != cutlass::Status::kSuccess)
        {
            std::string err_msg = "Failed to initialize cutlass fpA_intB gemm. Error: "
                + std::string(cutlassGetStatusString(init_status));
            throw std::runtime_error("[fpA_intB Runner] " + err_msg);
        }

        auto run_status = gemm.run(stream);
        if (run_status != cutlass::Status::kSuccess)
        {
            std::string err_msg
                = "Failed to run cutlass fpA_intB gemm. Error: " + std::string(cutlassGetStatusString(run_status));
            throw std::runtime_error("[fpA_intB Runner] " + err_msg);
        }
    }
    else
    {
        std::stringstream ss;
        ss << "[fpA_intB Runner] Config (" << (int64_t) cute::size<0>(CTAShape{}) << ","
           << (int64_t) cute::size<1>(CTAShape{}) << "," << (int64_t) cute::size<2>(CTAShape{}) << ") ("
           << (int64_t) cute::size<0>(ClusterShape{}) << "," << (int64_t) cute::size<1>(ClusterShape{}) << ","
           << (int64_t) cute::size<2>(ClusterShape{}) << ") not compiled with FAST_BUILD.";

        throw std::runtime_error(ss.str());
    }

#else  // COMPILE_HOPPER_TMA_GEMMS
    throw std::runtime_error(
        "[fpA_intB Runner] Please recompile with support for hopper by passing 90-real as an arch "
        "to build_wheel.py.");
#endif // COMPILE_HOPPER_TMA_GEMMS
}

}  // namespace nvidia
}  // namespace llm_kernels
