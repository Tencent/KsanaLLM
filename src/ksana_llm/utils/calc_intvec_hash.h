/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>
#include <vector>

namespace ksana_llm {

// Calculate a hash code for specific token ids.
inline size_t CalcIntVecHash(const int* start, size_t len) {
  size_t hash_code = 0;
  std::hash<int> hasher;
  for (size_t i = 0; i < len; ++i) {
    hash_code ^= hasher(*(start + i)) + 0x9e3779b9 + (hash_code << 6) + (hash_code >> 2);
  }
  return hash_code;
}

// Custom hash function
struct TokensHash {
  size_t operator()(const std::vector<int>& tokens) const { return CalcIntVecHash(tokens.data(), tokens.size()); }
};

// Custom comparison function
struct TokensEqual {
  bool operator()(const std::vector<int>& l_tokens, const std::vector<int>& r_tokens) const {
    return l_tokens == r_tokens;
  }
};

typedef std::unordered_map<std::vector<int>, std::vector<int>, TokensHash, TokensEqual> NgramDict;

}  // namespace ksana_llm