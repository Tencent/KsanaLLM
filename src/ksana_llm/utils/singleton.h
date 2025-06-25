/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include <mutex>

namespace ksana_llm {

// The singleton instance implement
template <typename T>
class Singleton {
 public:
  // Get singleton instance, with constructor arguments
  template <typename... Args>
  static std::shared_ptr<T> GetInstance(Args&&... args) {
    if (!singleton_instance_) {
      std::lock_guard<std::mutex> lock(singleton_mutex_);
      if (!singleton_instance_) {
        singleton_instance_ = std::make_shared<T>(std::forward<Args>(args)...);
      }
    }
    return singleton_instance_;
  }

 private:
  Singleton();
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;

 private:
  inline static std::shared_ptr<T> singleton_instance_ = nullptr;
  inline static std::mutex singleton_mutex_;
};

}  // namespace ksana_llm
