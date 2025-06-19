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
 * [vLLM Project] https://github.com/vllm-project/vllm/tree/v0.8.2/csrc/cutlass_extensions
 */

#pragma once

#include "csrc/kernels/nvidia/machete/cutlass_extensions/gemm/collective/collective_builder.hpp"

namespace cutlass::gemm::collective {
using namespace cute;

//
// VLLMCollectiveBuilder is a wrapper around CollectiveBuilder that allows for
// for custom kernel tags, allowing you to build custom collectives. Without
// touching the cutlass library headers, using `CutlassKernelTag` will mean it
// will resort to using the standard cutlass collective builder.
//

// Use the default Cutlass collective builder, i.e. use an unmodified cutless
// collective
struct CutlassKernelTag {};

template <class KernelTag, class ArchTag, class OpClass, class ElementA,
          class GmemLayoutA, int AlignmentA, class ElementB, class GmemLayoutB,
          int AlignmentB, class ElementAccumulator, class TileShape_MNK,
          class ClusterShape_MNK, class StageCountType,
          class KernelScheduleType, class Enable = void>
struct VLLMCollectiveBuilder {
  static_assert(sizeof(ElementA) == 0,
                "Could not build a collective for given parameters.");
};

template <class ArchTag, class OpClass, class ElementA, class GmemLayoutA,
          int AlignmentA, class ElementB, class GmemLayoutB, int AlignmentB,
          class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK,
          class StageCountType, class KernelScheduleType>
struct VLLMCollectiveBuilder<
    CutlassKernelTag, ArchTag, OpClass, ElementA, GmemLayoutA, AlignmentA,
    ElementB, GmemLayoutB, AlignmentB, ElementAccumulator, TileShape_MNK,
    ClusterShape_MNK, StageCountType, KernelScheduleType> {
  using CollectiveOp = typename CollectiveBuilder<
      ArchTag, OpClass, ElementA, GmemLayoutA, AlignmentA, ElementB,
      GmemLayoutB, AlignmentB, ElementAccumulator, TileShape_MNK,
      ClusterShape_MNK, StageCountType, KernelScheduleType>::CollectiveOp;
};

};  // namespace cutlass::gemm::collective