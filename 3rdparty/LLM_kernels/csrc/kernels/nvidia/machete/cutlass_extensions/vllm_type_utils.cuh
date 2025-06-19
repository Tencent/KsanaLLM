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

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cuda_bf16.h"

#include "csrc/kernels/nvidia/machete/cutlass_extensions/vllm_custom_types.cuh"

namespace cutlass {

template <typename T>
struct nameof {
  static constexpr char const* value = "unknown";
};

template <typename T>
inline constexpr auto nameof_v = nameof<T>::value;

#define NAMEOF_TYPE(T)                       \
  template <>                                \
  struct nameof<T> {                         \
    static constexpr char const* value = #T; \
  };

NAMEOF_TYPE(float_e4m3_t)
NAMEOF_TYPE(float_e5m2_t)
NAMEOF_TYPE(half_t)
NAMEOF_TYPE(nv_bfloat16)
NAMEOF_TYPE(bfloat16_t)
NAMEOF_TYPE(float)

NAMEOF_TYPE(int4b_t)
NAMEOF_TYPE(int8_t)
NAMEOF_TYPE(int32_t)
NAMEOF_TYPE(int64_t)

NAMEOF_TYPE(vllm_uint4b8_t)
NAMEOF_TYPE(uint4b_t)
NAMEOF_TYPE(uint8_t)
NAMEOF_TYPE(vllm_uint8b128_t)
NAMEOF_TYPE(uint32_t)
NAMEOF_TYPE(uint64_t)

};  // namespace cutlass