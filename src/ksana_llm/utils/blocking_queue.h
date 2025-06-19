/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <limits.h>
#include <stdint.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

template <typename T, typename Queue = std::queue<T>>
class BlockingQueue {
 public:
  explicit BlockingQueue(uint32_t max_size = UINT_MAX) : max_size_(max_size), is_closed_(false) {}

  ~BlockingQueue() = default;

  template <typename V>
  bool Put(V&& new_value) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return queue_.size() < max_size_ || is_closed_; });
    if (is_closed_) {
      return false;
    }

    queue_.push(std::forward<V>(new_value));
    cond_.notify_all();
    return true;
  }

  T Get() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty() || is_closed_; });
    if (is_closed_) {
      return T();
    }

    T value = std::move(queue_.front());
    queue_.pop();
    cond_.notify_all();
    return value;
  }

  T NonBlockingGet() {
    std::unique_lock<std::mutex> lock(mutex_);
    // cond_.wait(lock, [this] { return !queue_.empty() || is_closed_; });
    if (is_closed_ || queue_.empty()) {
      return T();
    }

    T value = std::move(queue_.front());
    queue_.pop();
    cond_.notify_all();
    return value;
  }

  T Get(size_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, std::chrono::milliseconds(timeout_ms), [this] { return !queue_.empty() || is_closed_; });
    if (is_closed_) {
      return T();
    }

    T value = std::move(queue_.front());
    queue_.pop();
    cond_.notify_all();
    return value;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void Stop() {
    std::lock_guard<std::mutex> lk(mutex_);
    is_closed_ = true;
    cond_.notify_all();
  }

 private:
  Queue queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  uint32_t max_size_;
  bool is_closed_;
};

// NOTE: cond_not_empty_.size() increases with id numbers. Don't use it with lots of ids.
template <typename T, typename IdType, typename Queue = std::queue<T>>
class BlockingQueueWithId {
 public:
  explicit BlockingQueueWithId(uint32_t max_size = UINT_MAX) : max_size_(max_size), is_closed_(false) {}

  ~BlockingQueueWithId() = default;

  template <typename V>
  bool Put(IdType id, V&& new_value) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (is_closed_) {
      return false;
    }

    // Check if we've reached the maximum total size across all queues
    size_t total_size = 0;
    for (const auto& pair : queues_) {
      total_size += pair.second.size();
    }

    // Wait if we've reached the maximum size
    if (total_size >= max_size_) {
      cond_not_full_.wait(lock, [this] {
        size_t total = 0;
        for (const auto& pair : queues_) {
          total += pair.second.size();
        }
        return total < max_size_ || is_closed_;
      });

      if (is_closed_) {
        return false;
      }
    }

    // Add the item to the queue for this ID
    queues_[id].push(std::forward<V>(new_value));

    // Notify any threads waiting for this specific ID
    auto it = cond_not_empty_.find(id);
    if (it != cond_not_empty_.end()) {
      it->second.notify_all();
    }

    return true;
  }

  T Get(IdType id) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Ensure we have a condition variable for this ID
    cond_not_empty_.try_emplace(id);

    // Wait until there's an item with the requested ID or the queue is closed
    cond_not_empty_[id].wait(
        lock, [this, id] { return (queues_.find(id) != queues_.end() && !queues_[id].empty()) || is_closed_; });

    if (is_closed_) {
      return T();
    }

    // Get the item from the queue for this ID
    T value = std::move(queues_[id].front());
    queues_[id].pop();

    // If the queue for this ID is now empty, remove it
    if (queues_[id].empty()) {
      queues_.erase(id);
    }

    // Notify any threads waiting to put items (if we were at max capacity)
    cond_not_full_.notify_one();

    return value;
  }

  T Get(IdType id, size_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Ensure we have a condition variable for this ID
    if (cond_not_empty_.find(id) == cond_not_empty_.end()) {
      cond_not_empty_[id] = std::condition_variable();
    }

    // Wait until there's an item with the requested ID or the queue is closed or timeout
    bool success = cond_not_empty_[id].wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, id] {
      return (queues_.find(id) != queues_.end() && !queues_[id].empty()) || is_closed_;
    });

    if (!success || is_closed_) {
      return T();
    }

    // Get the item from the queue for this ID
    T value = std::move(queues_[id].front());
    queues_[id].pop();

    // If the queue for this ID is now empty, remove it
    if (queues_[id].empty()) {
      queues_.erase(id);
    }

    // Notify any threads waiting to put items (if we were at max capacity)
    cond_not_full_.notify_one();

    return value;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total_size = 0;
    for (const auto& pair : queues_) {
      total_size += pair.second.size();
    }
    return total_size;
  }

  size_t Size(IdType id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = queues_.find(id);
    if (it != queues_.end()) {
      return it->second.size();
    }
    return 0;
  }

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queues_.empty();
  }

  bool Empty(IdType id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = queues_.find(id);
    if (it != queues_.end()) {
      return it->second.empty();
    }
    return true;
  }

  void Stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_closed_ = true;

    // Notify all waiting threads
    cond_not_full_.notify_all();
    for (auto& pair : cond_not_empty_) {
      pair.second.notify_all();
    }
  }

 private:
  std::unordered_map<IdType, Queue> queues_;
  std::unordered_map<IdType, std::condition_variable> cond_not_empty_;
  mutable std::mutex mutex_;
  std::condition_variable cond_not_full_;
  uint32_t max_size_;
  bool is_closed_;
};

}  // namespace ksana_llm
