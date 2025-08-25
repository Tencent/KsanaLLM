/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <pthread.h>

#include <cstddef>

namespace ksana_llm {

class SharedCounter {
 public:
  explicit SharedCounter(size_t num = 1) {
    num_ = num;
    pthread_spin_init(&spinlock_, PTHREAD_PROCESS_PRIVATE);
  }

  ~SharedCounter() { pthread_spin_destroy(&spinlock_); }

  void __attribute__((always_inline)) Reset(size_t new_num = 0) { num_ = new_num; }

  size_t __attribute__((always_inline)) Get() const { return num_; }

  void __attribute__((always_inline)) Increase(size_t num = 1) {
    pthread_spin_lock(&spinlock_);
    num_ += num;
    pthread_spin_unlock(&spinlock_);
  }

  void __attribute__((always_inline)) Decrease(size_t num = 1) {
    pthread_spin_lock(&spinlock_);
    if (num_ > 0) {
      num_ -= num;
    }
    pthread_spin_unlock(&spinlock_);
  }

 private:
  size_t num_ = 0;
  pthread_spinlock_t spinlock_;
};

class SchedulerSharedCounter {
 public:
  explicit SchedulerSharedCounter(size_t num) : step_batch_size(num), step_token_num(num) {}

  SharedCounter step_batch_size;
  SharedCounter step_token_num;
  SharedCounter step_logits_num;
};

}  // namespace ksana_llm
