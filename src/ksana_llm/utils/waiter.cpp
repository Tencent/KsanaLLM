/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

void Waiter::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this]() { return num_wait_ <= 0 || stop_; });
}

void Waiter::Notify() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (--num_wait_ == 0) {
    cv_.notify_all();
    // call done call back
    if (done_) {
      done_();
    }
  }
}

void Waiter::Reset(int num_wait) {
  std::lock_guard<std::mutex> lock(mutex_);
  num_wait_ = num_wait;
}

void Waiter::Inc() {
  std::lock_guard<std::mutex> lock(mutex_);
  num_wait_++;
}

int Waiter::Cnt() {
  std::lock_guard<std::mutex> lock(mutex_);
  return num_wait_;
}

void Waiter::Stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  stop_ = true;
  cv_.notify_all();
}

void WaitGroup::Done(int64_t n) {
  std::lock_guard<std::mutex> guard(mutex_);
  count_ -= n;
  if (count_ <= 0) {
    cond_.notify_all();
  }
}

void WaitGroup::Wait() {
  if (count_ == 0) {
    return;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [&]() { return count_ == 0; });
}

}  // namespace ksana_llm
