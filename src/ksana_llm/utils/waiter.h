/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>

namespace ksana_llm {

class AtomicCounter {
 public:
  explicit AtomicCounter(int num = 1) : num_(num) {}

  AtomicCounter() = delete;

  void Reset(int num) { num_ = num; }

  bool IsZero() { return num_ == 0; }

  bool DecAndIsZero() { return (--num_ == 0) ? true : false; }

 private:
  std::atomic<int> num_;
};

class Waiter {
 public:
  explicit Waiter(int num_wait = 1) : num_wait_(num_wait), stop_(false) {}
  explicit Waiter(int num_wait, std::function<void()> done) : num_wait_(num_wait), stop_(false), done_(done) {}

  Waiter() = delete;

  // Wait until zero.
  void Wait();

  void Notify();

  void Reset(int num_wait);

  void Inc();

  int Cnt();

  void Stop();

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int num_wait_;
  bool stop_;
  // done call back
  std::function<void()> done_;
};

// Note: Maybe use Latch insteaded in c++20.
class WaitGroup {
 public:
  WaitGroup() : count_(0) {}

  explicit WaitGroup(const int64_t n) : count_(n) {}

  void Add(const int64_t n = 1) { count_ += n; }

  void Done(const int64_t n = 1);

  int64_t Count() const { return count_; }

  void Wait();

  // Return true if successed, false for timeout.
  template <typename Rep, typename Period>
  bool WaitFor(const std::chrono::duration<Rep, Period>& timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cond_.wait_for(lock, timeout, [&]() { return count_ == 0; });
  }

 private:
  std::atomic_int64_t count_ = 0;
  std::mutex mutex_;
  std::condition_variable cond_;
};

}  // namespace ksana_llm
