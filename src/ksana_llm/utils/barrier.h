/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <vector>

namespace ksana_llm {
class Barrier {
 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  size_t thread_count_;
  size_t remaining_;
  size_t generation_ = 0;

 public:
  Barrier() : thread_count_(1), remaining_(1) {}
  explicit Barrier(size_t count) : thread_count_(count), remaining_(count) {}
  int Init(size_t count) {
    std::unique_lock<std::mutex> lock(mtx_);
    thread_count_ = count;
    remaining_ = count;

    return 0;
  }

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    size_t current_gen = generation_;
    if (--remaining_ == 0) {
      remaining_ = thread_count_;
      generation_++;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, current_gen] { return generation_ != current_gen; });
    }
  }

  size_t get_thread_count() { return thread_count_; }
  size_t get_remaining() { return remaining_; }
  size_t get_generation() { return generation_; }
};

}  // namespace ksana_llm