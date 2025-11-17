/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <math.h>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

// / The thread pool
class ThreadPool {
 public:
  explicit ThreadPool(const size_t thread_num) : thread_num_(thread_num) {}

  ~ThreadPool() {}

  void Stop() {
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      stopped_.store(true);
    }
    cv_.notify_all();
    for (std::thread& thread : pool_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  void Start() {
    stopped_.store(false);
    for (size_t i = 0; i < thread_num_; ++i) {
      pool_.emplace_back([this] {
        while (!this->stopped_) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock{this->mutex_};
            this->cv_.wait(lock, [this] { return this->stopped_.load() || !this->tasks_.empty(); });
            if (this->stopped_ && this->tasks_.empty()) {
              return;
            }
            task = std::move(this->tasks_.front());
            this->tasks_.pop();
          }
          task();
        }
      });
    }
  }

  template <class Fun, class... Args>
  auto Submit(Fun&& f, Args&&... args) -> std::future<decltype(f(args...))> {
    if (stopped_.load()) {
      KLLM_THROW("Submit on stopped threadpool.");
    }

    using RetType = decltype(f(args...));
    auto task =
        std::make_shared<std::packaged_task<RetType()>>(std::bind(std::forward<Fun>(f), std::forward<Args>(args)...));
    std::future<RetType> future = task->get_future();
    {
      std::lock_guard<std::mutex> lock{mutex_};
      tasks_.emplace([task]() { (*task)(); });
    }
    cv_.notify_one();
    return future;
  }

 private:
  std::condition_variable cv_;
  std::mutex mutex_;

  std::atomic<bool> stopped_;

  std::vector<std::thread> pool_;
  std::queue<std::function<void()>> tasks_;

  const size_t thread_num_;
};

}  // namespace ksana_llm
