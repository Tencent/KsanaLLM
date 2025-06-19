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

#include "cutlass/integer_subbyte.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, int Bias, bool Signed = false>
struct vllm_biased_integer_subbyte : public integer_subbyte<Bits, Signed> {
  using Base = integer_subbyte<Bits, Signed>;

  using Storage = typename Base::Storage;
  using xint_t = typename Base::xint_t;

  using Base::bits_mask_;
  using Base::sign_mask_;
  using Base::storage;

  //
  // Methods
  //

  /// No operation
  vllm_biased_integer_subbyte() = default;

  /// Conversion from integer type
  CUTLASS_HOST_DEVICE explicit vllm_biased_integer_subbyte(int value)
      : Base(value) {}
  CUTLASS_HOST_DEVICE explicit vllm_biased_integer_subbyte(unsigned value)
      : Base(value) {}
  CUTLASS_HOST_DEVICE explicit vllm_biased_integer_subbyte(double value)
      : Base(value) {}
};
///////////////////////////////////////////////////////////////////////////////////////////////////

// "GPTQ" types, i.e. symmetric quantization
using vllm_uint4b8_t = vllm_biased_integer_subbyte<4, 8>;      // u4b8
using vllm_uint8b128_t = vllm_biased_integer_subbyte<8, 128>;  // u8b128

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Bits, int Bias, bool Signed>
struct sizeof_bits<vllm_biased_integer_subbyte<Bits, Bias, Signed>> {
  static constexpr int value = Bits;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass
