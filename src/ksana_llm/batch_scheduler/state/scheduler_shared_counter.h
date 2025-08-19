/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <mutex>

namespace ksana_llm {

class SharedCounter {
 public:
  explicit SharedCounter(size_t num = 1) {
    std::lock_guard<std::mutex> lock(mutex_);
    num_ = num;
  }

  void Reset(size_t new_num = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    num_ = new_num;
  }

  size_t Get() {
    std::lock_guard<std::mutex> lock(mutex_);
    return num_;
  }

  void Increase(size_t num = 1) {
    std::lock_guard<std::mutex> lock(mutex_);
    num_ += num;
  }

  void Decrease(size_t num = 1) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (num_ > 0) {
      num_ -= num;
    }
  }

 private:
  size_t num_ = 0;
  std::mutex mutex_;
};

struct SchedulerSharedCounter {
  explicit SchedulerSharedCounter(size_t num) : step_batch_size(num), step_token_num(num) {}

  SharedCounter step_batch_size;
  SharedCounter step_token_num;
  SharedCounter step_logits_num;
};

}  // namespace ksana_llm
