/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <vector>

namespace ksana_llm {
struct expert_parallel_comm_meta {
  // number of communication bytes.
  uint32_t shape_0;
  uint32_t shape_1;
  uint32_t src_rank;
  uint32_t dst_rank;
  // 1: control message, 2: hidden_data for ep computing; 3 hidden_data: results of ep computing.
  uint32_t mode;
};

}  // namespace ksana_llm