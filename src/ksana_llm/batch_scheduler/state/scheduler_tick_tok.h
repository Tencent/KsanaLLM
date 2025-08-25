/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <vector>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace ksana_llm {

class SchedulerTickTok {
 public:
  explicit SchedulerTickTok(size_t group_size = 1) {
    group_size_ = group_size;
    skipped_groups_.resize(group_size_, false);

    for (size_t i = 0; i < group_size_; ++i) {
      visit_order_.push_back(i);
    }
  }

  static void SetThreadIndex(size_t thread_index) { thread_index_ = thread_index; }

  __attribute__((hot)) void Lock() {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    // Blocking on wait only if pred not true.
    // Otherwise, if all other thread is skipped, no notify() invoked.
    if (is_lockable_.load() && thread_index_ == visit_order_[current_idx_]) {
      is_lockable_.store(false);
      return;
    }

    guard_cv_.wait(
        lock, [this]() { return is_lockable_.load() && thread_index_ == visit_order_[current_idx_]; });
    is_lockable_.store(false);
  }

  __attribute__((hot)) void Unlock() {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    // If not locked, return immediately.
    if (unlikely(is_lockable_.load())) {
      return;
    }

    // If thread not matched, that is, other thread have change index via Skip(), keep current_idx_ not changed.
    if (thread_index_ != visit_order_[current_idx_]) {
      is_lockable_.store(true);
      guard_cv_.notify_all();
      return;
    }

    // Change current idx to next position.
#pragma GCC unroll 8
    for (size_t i = current_idx_ + 1; i <= current_idx_ + group_size_; ++i) {
      size_t tmp_current_idx = i % group_size_;
      if (!skipped_groups_[visit_order_[tmp_current_idx]]) {
        current_idx_ = tmp_current_idx;
        break;
      }
    }

    is_lockable_.store(true);
    guard_cv_.notify_all();
  }

  // Reset all skip list, the vist order will be keepped.
  __attribute__((hot)) void Reset() {
    std::unique_lock<std::mutex> lock(guard_mutex_);

#pragma GCC unroll 8
    for (size_t i = 0; i < group_size_; ++i) {
      skipped_groups_[i] = false;
    }
    current_idx_ = 0;

    // No need to reset wait_num_ & instance_
    is_lockable_.store(true);
  }

  // The visit list will skip current thread index.
  __attribute__((hot)) void Skip() {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    if (likely(thread_index_ < skipped_groups_.size())) {
      skipped_groups_[thread_index_] = true;
    }

    // Change current idx to next not skipped position.
#pragma GCC unroll 8
    for (size_t i = current_idx_ + 1; i <= current_idx_ + group_size_; ++i) {
      size_t tmp_current_idx = i % group_size_;
      if (!skipped_groups_[visit_order_[tmp_current_idx]]) {
        current_idx_ = tmp_current_idx;
        break;
      }
    }

    // Notify other threads.
    guard_cv_.notify_all();
  }

  __attribute__((hot)) void Reorder() {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    size_t first_val = visit_order_[0];
#pragma GCC unroll 8
    for (size_t i = 1; i < group_size_; ++i) {
      visit_order_[i - 1] = visit_order_[i];
    }
    visit_order_[group_size_ - 1] = first_val;
    current_idx_ = 0;
  }

  // Make all threads arrive same check point.
  __attribute__((hot)) void Barrier() {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    size_t cur_inst = instance_;
    if (++wait_num_ == group_size_) {
      wait_num_ = 0;
      instance_++;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, &cur_inst]() { return instance_ != cur_inst; });
    }
  }

 private:
  size_t group_size_ = 1;
  size_t current_idx_ = 0;

  // The index is thread idx
  std::vector<bool> skipped_groups_;

  // The value is thread idx
  std::vector<size_t> visit_order_;

  std::atomic<bool> is_lockable_ = true;

  std::mutex guard_mutex_;
  std::condition_variable guard_cv_;

  std::condition_variable cv_;

  size_t wait_num_ = 0;
  size_t instance_ = 0;

  inline static thread_local size_t thread_index_;
};

}  // namespace ksana_llm
