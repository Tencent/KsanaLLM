/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>

namespace ksana_llm {
class Barrier {
 private:
  size_t thread_count_;
  std::atomic_size_t remaining_;
  std::atomic_size_t generation_ = 0;

 public:
  Barrier() : thread_count_(1), remaining_(1) {}
  explicit Barrier(const size_t count) : thread_count_(count), remaining_(count) {}
  void Init(const size_t count) {
    thread_count_ = count;
    remaining_ = count;
  }

  void arrive_and_wait() {
    const size_t current_gen = generation_.load(std::memory_order_acquire);
    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      remaining_.store(thread_count_, std::memory_order_relaxed);
      generation_.store(current_gen + 1, std::memory_order_release);
    } else {
      while (generation_.load(std::memory_order_acquire) == current_gen) {
        __builtin_ia32_pause();
      }
    }
  }

  size_t get_thread_count() const { return thread_count_; }
  size_t get_remaining() const { return remaining_.load(std::memory_order_acquire); }
  size_t get_generation() const { return generation_.load(std::memory_order_acquire); }
};

}  // namespace ksana_llm