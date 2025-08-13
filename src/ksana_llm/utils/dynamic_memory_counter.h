/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <unordered_map>

namespace ksana_llm {

class DynamicMemoryCounter {
 public:
  static size_t GetMemoryBytes(int rank);

  static void Increase(int rank, size_t bytes);
  static void Decrease(int rank, size_t bytes);

 private:
  static std::unordered_map<int, size_t> rank_memory_bytes;
};

}  // namespace ksana_llm
