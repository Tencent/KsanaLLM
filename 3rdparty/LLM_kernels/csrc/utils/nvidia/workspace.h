/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Copyright (c) 1993-2023, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once
#include <cstddef>
#include <cstdint>

namespace llm_kernels {
namespace utils {

std::uintptr_t constexpr kCudaMemAlign = 128;

namespace {

inline int8_t* alignPtr(int8_t* ptr, uintptr_t to) {
  uintptr_t addr = (uintptr_t)ptr;
  if (addr % to) {
    addr += to - addr % to;
  }
  return (int8_t*)addr;
}

inline int8_t* nextWorkspacePtrCommon(int8_t* ptr, uintptr_t previousWorkspaceSize, const uintptr_t alignment) {
  uintptr_t addr = (uintptr_t)ptr;
  addr += previousWorkspaceSize;
  return alignPtr((int8_t*)addr, alignment);
}

inline int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize) {
  return nextWorkspacePtrCommon(ptr, previousWorkspaceSize, kCudaMemAlign);
}

inline int8_t* nextWorkspacePtr(int8_t* const base, uintptr_t& offset, const uintptr_t size,
                                const uintptr_t alignment = kCudaMemAlign) {
  uintptr_t curr_offset = offset;
  uintptr_t next_offset = curr_offset + ((size + alignment - 1) / alignment) * alignment;
  int8_t* newptr = size == 0 ? nullptr : base + curr_offset;
  offset = next_offset;
  return newptr;
}

inline int8_t* nextWorkspacePtrWithAlignment(int8_t* ptr, uintptr_t previousWorkspaceSize,
                                             const uintptr_t alignment = kCudaMemAlign) {
  return nextWorkspacePtrCommon(ptr, previousWorkspaceSize, alignment);
}

inline size_t calculateTotalWorkspaceSize(size_t const* workspaces, int count,
                                          const uintptr_t alignment = kCudaMemAlign) {
  size_t total = 0;
  for (int i = 0; i < count; i++) {
    total += workspaces[i];
    if (workspaces[i] % alignment) {
      total += alignment - (workspaces[i] % alignment);
    }
  }
  return total;
}

}  // namespace

}  // namespace utils
}  // namespace llm_kernels
