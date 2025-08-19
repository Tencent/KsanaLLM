/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <vector>

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

  void Lock(size_t thread_index) {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    // Blocking on wait only if pred not true.
    // Otherwise, if all other thread is skipped, no notify() invoked.
    if (is_lockable_.load() && thread_index == visit_order_[current_idx_]) {
      mutex_.lock();
      is_lockable_.store(false);
      return;
    }

    guard_cv_.wait(
        lock, [this, thread_index]() { return is_lockable_.load() && thread_index == visit_order_[current_idx_]; });
    mutex_.lock();
    is_lockable_.store(false);
  }

  void Unlock(size_t thread_index) {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    // If not locked, or Wrong thread, do nothing and return immediately.
    if (is_lockable_.load() || thread_index != visit_order_[current_idx_]) {
      return;
    }

    // Change current idx to next position.
    for (size_t i = current_idx_ + 1; i <= current_idx_ + group_size_; ++i) {
      size_t tmp_current_idx = i % group_size_;
      if (!skipped_groups_[visit_order_[tmp_current_idx]]) {
        current_idx_ = tmp_current_idx;
        break;
      }
    }

    is_lockable_.store(true);
    guard_cv_.notify_all();
    mutex_.unlock();
  }

  // Reset all skip list, the vist order will be keepped.
  void Reset() {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    for (size_t i = 0; i < group_size_; ++i) {
      skipped_groups_[i] = false;
    }
    current_idx_ = 0;

    // No need to reset wait_num_ & instance_
    is_lockable_.store(true);
  }

  // The visit list will skip current thread index.
  void Skip(size_t thread_index) {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    if (thread_index < skipped_groups_.size()) {
      skipped_groups_[thread_index] = true;
    }

    // Change current idx to next not skipped position.
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

  // Set the process order, such as {3, 1, 2, 0}
  void Reorder(const std::vector<size_t>& visit_order) {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    if (visit_order.size() == group_size_) {
      visit_order_ = visit_order;
      current_idx_ = 0;
    }
  }

  void Reorder() {
    std::unique_lock<std::mutex> lock(guard_mutex_);

    size_t first_val = visit_order_[0];
    for (size_t i = 1; i < group_size_; ++i) {
      visit_order_[i - 1] = visit_order_[i];
    }
    visit_order_[group_size_ - 1] = first_val;
    current_idx_ = 0;
  }

  // Make all threads arrive same check point.
  void Barrier() {
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

  std::mutex mutex_;
  std::condition_variable cv_;

  size_t wait_num_ = 0;
  size_t instance_ = 0;
};

class SchedulerTickTokLockGuard {
 public:
  SchedulerTickTokLockGuard(std::shared_ptr<SchedulerTickTok> tick_tok, size_t thread_idx) {
    tick_tok_ = tick_tok;
    thread_idx_ = thread_idx;
    tick_tok_->Lock(thread_idx_);
  }

  ~SchedulerTickTokLockGuard() { tick_tok_->Unlock(thread_idx_); }

 private:
  std::shared_ptr<SchedulerTickTok> tick_tok_;
  size_t thread_idx_;
};

}  // namespace ksana_llm
