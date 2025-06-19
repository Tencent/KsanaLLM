/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <chrono>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/transfer/transfer_types.h"
#include "ksana_llm/utils/blocking_queue.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

#define TASK_MANAGER_DEFAULT_EXPIRE_MINUTES 15

struct TaskKey {
  int req_id;
  int block_idx;
  int layer_idx;
  int device_idx;
  int tensor_size;
  int token;  // 可选：用于存储token数量
  std::time_t start_time_us;

  TaskKey() : req_id(0), block_idx(0), layer_idx(0), device_idx(0), tensor_size(0), token(0), start_time_us(0) {}
  TaskKey(int req, int block, int layer, int device, int tsize = 0, int ttoken = 0, std::time_t timestamp_us = 0)
      : req_id(req),
        block_idx(block),
        layer_idx(layer),
        device_idx(device),
        tensor_size(tsize),
        token(ttoken),
        start_time_us(timestamp_us) {}
  TaskKey(const TaskKey& other) = default;
  TaskKey(TaskKey&& other) noexcept = default;
  TaskKey& operator=(const TaskKey& other) = default;
  TaskKey& operator=(TaskKey&& other) noexcept = default;
  bool operator==(const TaskKey& other) const {
    return req_id == other.req_id && block_idx == other.block_idx && layer_idx == other.layer_idx &&
           device_idx == other.device_idx && tensor_size == other.tensor_size;
  }
  std::string ToString() const {
    std::ostringstream oss;
    oss << "req_id=" << req_id << ", block_idx=" << block_idx << ", layer_idx=" << layer_idx
        << ", device_idx=" << device_idx << ", tensor_size=" << tensor_size << ", token=" << token;
    return oss.str();
  }
  struct NoTokenHash {
    size_t operator()(const TaskKey& key) const {
      return std::hash<int>()(key.req_id) ^ (std::hash<int>()(key.block_idx) << 1) ^
             (std::hash<int>()(key.layer_idx) << 2) ^ (std::hash<int>()(key.device_idx) << 3) ^
             (std::hash<int>()(key.tensor_size) << 4);
    }
  };

  // Serialize TaskKey to binary data
  std::vector<uint8_t> Serialize() const {
    std::vector<uint8_t> data(sizeof(TaskKey));
    memcpy(data.data(), this, sizeof(TaskKey));
    return data;
  }

  // Deserialize binary data to TaskKey
  static TaskKey Deserialize(const std::vector<uint8_t>& data) {
    TaskKey key;
    if (data.size() >= sizeof(TaskKey)) {
      memcpy(&key, data.data(), sizeof(TaskKey));
    }
    return key;
  }

  // 批量反序列化：根据长度自动判断
  static std::vector<TaskKey> DeserializeBatch(const std::vector<uint8_t>& data) {
    std::vector<TaskKey> result;
    size_t n = data.size() / sizeof(TaskKey);
    if (n == 0 || data.size() % sizeof(TaskKey) != 0) return result;
    result.resize(n);
    memcpy(result.data(), data.data(), n * sizeof(TaskKey));
    return result;
  }

  // 批量反序列化：根据长度自动判断
  static std::vector<TaskKey> DeserializeTaskKeys(const char* data, size_t size) {
    std::vector<TaskKey> result;
    if (size == sizeof(TaskKey)) {
      std::vector<uint8_t> buf(data, data + size);
      result.push_back(TaskKey::Deserialize(buf));
    } else if (size % sizeof(TaskKey) == 0) {
      std::vector<uint8_t> buf(data, data + size);
      result = TaskKey::DeserializeBatch(buf);
    }
    return result;
  }

  // 批量序列化
  static std::vector<uint8_t> BatchSerialize(const std::vector<TaskKey>& keys) {
    std::vector<uint8_t> buf;
    buf.reserve(keys.size() * sizeof(TaskKey) + sizeof(std::time_t));
    for (const auto& k : keys) {
      auto single = k.Serialize();
      buf.insert(buf.end(), single.begin(), single.end());
    }
    return buf;
  }
};

class TaskManager {
 public:
  using TaskNotifyCallback = std::function<void()>;

  TaskManager() : send_waiter_(std::make_shared<Waiter>(1)) {}
  ~TaskManager() = default;

  // 记录淘汰时间默认（15分钟）
  static constexpr std::chrono::minutes kDefaultExpire = std::chrono::minutes(TASK_MANAGER_DEFAULT_EXPIRE_MINUTES);

  TaskKey CreateTaskKey(std::shared_ptr<TransferTask> const& task) {
    int tsize = 0;
    if (task && !task->tensor.shape.empty()) {
      tsize = task->tensor.GetElementNumber() * GetTypeSize(task->tensor.dtype);
    }
    return TaskKey(task->req_id, task->tensor.block_idx, task->tensor.layer_idx, task->tensor.device_idx, tsize,
                   task->token, ProfileTimer::GetCurrentTimeInUs());
  }

  using GroupDevKey = std::pair<std::string, int>;
  struct GroupDevKeyHash {
    std::size_t operator()(const GroupDevKey& k) const {
      return std::hash<std::string>()(k.first) ^ (std::hash<int>()(k.second) << 1);
    }
  };

  std::unordered_map<GroupDevKey, std::vector<TaskKey>, GroupDevKeyHash> GroupByGroupKeyAndDevice(
      const std::vector<TaskKey>& batch) {
    std::unordered_map<GroupDevKey, std::vector<TaskKey>, GroupDevKeyHash> grouped;
    for (const auto& tk : batch) {
      auto task = GetTask(tk);
      if (task == nullptr) {
        KLLM_LOG_ERROR << "Skipping task_key without corresponding task: " << tk.ToString();
        continue;
      }
      const std::string& group_key = task->addr;
      int device_idx = tk.device_idx;

      grouped[{group_key, device_idx}].push_back(tk);
    }
    return grouped;
  }

  void AddTask(const TaskKey& key, std::shared_ptr<TransferTask> task) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    task_map_[key] = task;
  }

  std::shared_ptr<TransferTask> GetTask(const TaskKey& key) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    auto it = task_map_.find(key);
    if (it != task_map_.end()) {
      return it->second;
    }
    return nullptr;
  }

  void CompleteTask(const TaskKey& key) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    auto it = task_map_.find(key);
    if (it != task_map_.end() && it->second != nullptr) {
      it->second->is_completed = true;
      task_map_.erase(it);
    }
    // If task not found, silently ignore - it may have been already completed or never existed
  }

  void Shutdown() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    task_map_.clear();
    prefill_pending_tasks_.clear();
    decode_confirmed_tasks_.clear();
    processing_buffer_.Stop();
  }

  // Promise 同步相关方法
  void CleanupExpiredTasks(int timeout_seconds) {
    std::lock_guard<std::mutex> lock(promises_mutex_);

    const auto now = ProfileTimer::GetCurrentTime();

    // 清理 prefill_pending_tasks_ 中的过期条目
    auto pending_it = prefill_pending_tasks_.begin();
    while (pending_it != prefill_pending_tasks_.end()) {
      if (now - pending_it->second > timeout_seconds) {
        // Note: 这里应该有日志，但为了避免依赖问题，暂时省略
        pending_it = prefill_pending_tasks_.erase(pending_it);
      } else {
        ++pending_it;
      }
    }

    // 清理 decode_confirmed_tasks_ 中的过期条目
    auto request_it = decode_confirmed_tasks_.begin();
    while (request_it != decode_confirmed_tasks_.end()) {
      if (now - request_it->second > timeout_seconds) {
        // Note: 这里应该有日志，但为了避免依赖问题，暂时省略
        request_it = decode_confirmed_tasks_.erase(request_it);
      } else {
        ++request_it;
      }
    }
  }

  // Promise 同步相关方法
  void RegisterDecodeConfirmedTasks(const std::vector<TaskKey>& task_keys) {
    std::lock_guard<std::mutex> lock(promises_mutex_);
    const auto now = ProfileTimer::GetCurrentTime();
    for (const auto& task_key : task_keys) {
      decode_confirmed_tasks_[task_key] = now;
      auto it = prefill_pending_tasks_.find(task_key);
      if (it != prefill_pending_tasks_.end()) {
        TaskKey actual_key = it->first;
        prefill_pending_tasks_.erase(it);
        processing_buffer_.Put(actual_key);
        if (send_waiter_) {
          send_waiter_->Notify();
        }
      }
    }
  }

  void AddPrefillPendingTask(const TaskKey& task_key) {
    std::lock_guard<std::mutex> lock(promises_mutex_);
    const auto now = ProfileTimer::GetCurrentTime();
    prefill_pending_tasks_[task_key] = now;
  }

  bool TryActivatePendingTask(const TaskKey& task_key) {
    std::lock_guard<std::mutex> lock(promises_mutex_);
    if (decode_confirmed_tasks_.find(task_key) != decode_confirmed_tasks_.end()) {
      // 如果找到匹配的TaskKey，从两个映射中移除
      prefill_pending_tasks_.erase(task_key);
      decode_confirmed_tasks_.erase(task_key);
      return true;  // 可以激活
    }
    return false;  // 不能激活，需要等待
  }

  // Prefill端等待Decode确认的任务映射，存储TaskKey和插入时间戳
  std::unordered_map<TaskKey, std::time_t, TaskKey::NoTokenHash> prefill_pending_tasks_;
  // Decode端已确认的任务映射，存储TaskKey和插入时间戳
  std::unordered_map<TaskKey, std::time_t, TaskKey::NoTokenHash> decode_confirmed_tasks_;

  std::mutex promises_mutex_;

  BlockingQueue<TaskKey> processing_buffer_;
  std::shared_ptr<Waiter> send_waiter_ = nullptr;
  std::unordered_map<TaskKey, std::shared_ptr<TransferTask>, TaskKey::NoTokenHash> task_map_;
  std::mutex buffer_mutex_;
};

}  // namespace ksana_llm