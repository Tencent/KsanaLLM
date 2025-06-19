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
 * [vLLM Project]
 * https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/quantization/machete/machete_collective_builder.cuh
 */

#pragma once

#include "csrc/kernels/nvidia/machete/cutlass_extensions/vllm_collective_builder.cuh"
#include "csrc/kernels/nvidia/machete/machete_mainloop.cuh"

namespace cutlass::gemm::collective {
using namespace cute;

struct MacheteKernelTag {};

template <class ElementPairA_, class GmemLayoutA_, int AlignmentA, class ElementPairB_, class GmemLayoutB_,
          int AlignmentB, class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK, class StageCountType,
          class KernelScheduleType>
struct VLLMCollectiveBuilder<
    MacheteKernelTag, arch::Sm90, arch::OpClassTensorOp, ElementPairA_, GmemLayoutA_, AlignmentA, ElementPairB_,
    GmemLayoutB_, AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType, KernelScheduleType,
    cute::enable_if_t<(cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized> ||
                       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong> ||
                       cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative>)>> {
  using CollectiveOp =
      llm_kernels::nvidia::machete::MacheteCollectiveMma<ElementPairA_, GmemLayoutA_, AlignmentA, ElementPairB_,
                                                         GmemLayoutB_, AlignmentB, ElementAccumulator, TileShape_MNK,
                                                         ClusterShape_MNK, StageCountType, KernelScheduleType>;
};

};  // namespace cutlass::gemm::collective
