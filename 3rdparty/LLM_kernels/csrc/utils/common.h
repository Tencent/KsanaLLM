/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

namespace llm_kernels {
namespace utils {

void ParseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data);

int32_t ParseNpyHeader(FILE*& f_ptr, uint32_t header_len, std::vector<size_t>& shape);

}  // namespace utils
}  // namespace llm_kernels