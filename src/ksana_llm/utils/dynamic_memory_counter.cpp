/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/dynamic_memory_counter.h"

namespace ksana_llm {

std::unordered_map<int, size_t> DynamicMemoryCounter::rank_memory_bytes;

void DynamicMemoryCounter::Increase(int rank, size_t bytes) {
  if (rank_memory_bytes.find(rank) == rank_memory_bytes.end()) {
    rank_memory_bytes[rank] = 0;
  }
  rank_memory_bytes[rank] += bytes;
}

void DynamicMemoryCounter::Decrease(int rank, size_t bytes) {
  if (rank_memory_bytes.find(rank) == rank_memory_bytes.end()) {
    rank_memory_bytes[rank] = 0;
  }
  rank_memory_bytes[rank] -= bytes;
}

size_t DynamicMemoryCounter::GetMemoryBytes(int rank) {
  if (rank_memory_bytes.find(rank) == rank_memory_bytes.end()) {
    return 0;
  }

  return rank_memory_bytes.at(rank);
}

}  // namespace ksana_llm
