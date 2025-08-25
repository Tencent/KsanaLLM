/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/task_manager.h"

#include <algorithm>
#include <atomic>

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

//=============================================================================
// Constructor and Singleton Management
//=============================================================================

TaskManager::TaskManager(int circular_bucket_num, int bucket_size_hint, int circular_thread_num)
    : circular_bucket_num_(circular_bucket_num) {
  task_arena_.initialize(circular_thread_num);  // 设置最大并发为64
  // Initialize notification waiter
  notification_waiter_ = std::make_shared<Waiter>(1);

  // Initialize shards
  shards_.reserve(circular_bucket_num_);
  for (int i = 0; i < circular_bucket_num_; ++i) {
    auto shard = std::make_unique<TaskShard>();

    // Set initial bucket size for hash maps
    shard->request_map.rehash(bucket_size_hint / circular_bucket_num_);
    shard->prefill_pending_tasks.rehash(bucket_size_hint / circular_bucket_num_);
    shard->decode_confirmed_tasks.rehash(bucket_size_hint / circular_bucket_num_);

    shards_.push_back(std::move(shard));
  }

  KLLM_LOG_INFO << "TaskManager initialized with " << circular_bucket_num_
                << " shards, bucket_size_hint=" << bucket_size_hint << ", circular_thread_num=" << circular_thread_num;
}

std::shared_ptr<TaskManager> TaskManager::GetInstance(int circular_bucket_num, int bucket_size_hint) {
  return Singleton<TaskManager>::GetInstance(circular_bucket_num, bucket_size_hint);
}

//=============================================================================
// Processing Buffer Operations
//=============================================================================

void TaskManager::PutProcessingBuffer(const TaskKey& task_key) {
  processing_buffer_.push(task_key);

  if (notification_waiter_) {
    notification_waiter_->Notify();
  }
}

// void TaskManager::StopProcessing

TaskKey TaskManager::GetProcessingBuffer() {
  TaskKey task_key;

  if (processing_buffer_.try_pop(task_key)) {
    return task_key;
  }

  return TaskKey();  // Return default TaskKey if buffer is empty
}

std::vector<TaskKey> TaskManager::GetProcessingBufferBatch(int batch_size) {
  std::vector<TaskKey> batch;
  batch.reserve(batch_size);

  if (batch_size <= 0) {
    return batch;
  }

  // Simply try to pop tasks from the single processing buffer
  for (int i = 0; i < batch_size; ++i) {
    TaskKey task_key;
    if (processing_buffer_.try_pop(task_key)) {
      batch.push_back(task_key);
    } else {
      break;  // No more tasks available
    }
  }

  return batch;
}

bool TaskManager::IsProcessingBufferEmpty() const { return processing_buffer_.empty(); }

size_t TaskManager::GetProcessingBufferSize() const { return processing_buffer_.size(); }

//=============================================================================
// Core Task Management Operations
//=============================================================================

TaskKey TaskManager::CreateTaskKey(const std::shared_ptr<TransferTask>& task) {
  return TaskKey::CreateFromTransferTask(task);
}

void TaskManager::AddTask(const TaskKey& key, std::shared_ptr<TransferTask> task) {
  auto& shard = GetShard(key.req_id);
  typename decltype(shard.request_map)::accessor accessor;

  shard.request_map.insert(accessor, key);
  accessor->second = task;
}

std::shared_ptr<TransferTask> TaskManager::GetTask(const TaskKey& key) {
  const auto& shard = GetShard(key.req_id);
  typename decltype(shard.request_map)::const_accessor accessor;

  if (shard.request_map.find(accessor, key)) {
    return accessor->second;
  }

  return nullptr;
}

std::vector<std::shared_ptr<TransferTask>> TaskManager::GetTasksBatch(const std::vector<TaskKey>& keys) {
  std::vector<std::shared_ptr<TransferTask>> results;
  results.reserve(keys.size());

  if (keys.size() >= static_cast<size_t>(circular_bucket_num_) * 2) {
    // Parallel batch retrieval for large batches
    results.resize(keys.size());

    task_arena_.execute([&]() {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, keys.size()), [&](const tbb::blocked_range<size_t>& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          const auto& key = keys[i];
          const auto& shard = GetShard(key.req_id);
          typename decltype(shard.request_map)::const_accessor accessor;

          if (shard.request_map.find(accessor, key)) {
            results[i] = accessor->second;
          } else {
            results[i] = nullptr;
          }
        }
      });
    });
  } else {
    // Sequential retrieval for smaller batches
    for (const auto& key : keys) {
      results.push_back(GetTask(key));
    }
  }

  return results;
}

void TaskManager::CompleteTask(const TaskKey& key) {
  auto& shard = GetShard(key.req_id);
  typename decltype(shard.request_map)::accessor accessor;

  if (shard.request_map.find(accessor, key)) {
    if (accessor->second) {
      accessor->second->is_completed = true;
    }
    accessor.release();
    shard.request_map.erase(key);
  }
}

size_t TaskManager::GetTaskCount() const {
  size_t total = 0;
  for (const auto& shard : shards_) {
    total += shard->request_map.size();
  }
  return total;
}

float TaskManager::GetLoadFactor() const {
  size_t total_size = 0;
  size_t total_buckets = 0;

  for (const auto& shard : shards_) {
    total_size += shard->request_map.size();
    total_buckets += shard->request_map.bucket_count();
  }

  return total_buckets > 0 ? static_cast<float>(total_size) / total_buckets : 0.0f;
}

//=============================================================================
// Promise-based Task Synchronization
//=============================================================================

void TaskManager::RegisterDecodeConfirmedTasks(const std::vector<TaskKey>& task_keys) {
  if (task_keys.empty()) {
    return;
  }

  const auto now = ProfileTimer::GetCurrentTime();

  // Group tasks by shard for better cache locality
  std::vector<std::vector<TaskKey>> shard_tasks(circular_bucket_num_);
  for (const auto& task_key : task_keys) {
    size_t shard_idx = GetShardIndex(task_key.req_id);
    shard_tasks[shard_idx].push_back(task_key);
  }

  // Process each shard's tasks in parallel
  task_arena_.execute([&]() {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, circular_bucket_num_),
                      [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t shard_idx = range.begin(); shard_idx != range.end(); ++shard_idx) {
                          auto& shard = *shards_[shard_idx];
                          const auto& shard_task_keys = shard_tasks[shard_idx];

                          for (const auto& task_key : shard_task_keys) {
                            // Insert into decode confirmed tasks
                            typename decltype(shard.decode_confirmed_tasks)::accessor decode_accessor;
                            shard.decode_confirmed_tasks.insert(decode_accessor, task_key);
                            decode_accessor->second = now;
                            decode_accessor.release();

                            // Check if this task was waiting in prefill pending
                            typename decltype(shard.prefill_pending_tasks)::accessor prefill_accessor;
                            KLLM_LOG_DEBUG << "RegisterDecodeConfirmedTasks: " << task_key.ToString();
                            if (shard.prefill_pending_tasks.find(prefill_accessor, task_key)) {
                              TaskKey actual_key = prefill_accessor->first;
                              actual_key.decode_device_id = task_key.decode_device_id;
                              actual_key.decode_device_offset = task_key.decode_device_offset;
                              KLLM_LOG_DEBUG << "prefill_pending_tasks find and update: " << actual_key.ToString();
                              prefill_accessor.release();
                              shard.prefill_pending_tasks.erase(task_key);

                              // Move to processing buffer
                              processing_buffer_.push(actual_key);
                            }
                          }
                        }
                      });
  });

  // Notify waiting threads
  if (notification_waiter_) {
    notification_waiter_->Notify();
  }
}

void TaskManager::AddPrefillPendingTask(const TaskKey& task_key) {
  const auto now = ProfileTimer::GetCurrentTime();
  auto& shard = GetShard(task_key.req_id);

  typename decltype(shard.prefill_pending_tasks)::accessor accessor;
  shard.prefill_pending_tasks.insert(accessor, task_key);
  accessor->second = now;
}

bool TaskManager::TryActivatePendingTask(TaskKey& task_key) {
  auto& shard = GetShard(task_key.req_id);

  // Check if confirmed by decode phase
  typename decltype(shard.decode_confirmed_tasks)::const_accessor decode_accessor;
  KLLM_LOG_DEBUG << "Start find prefill Taskkey in decode_confirmed_tasks: " << task_key.ToString();
  if (shard.decode_confirmed_tasks.find(decode_accessor, task_key)) {
    if (task_key.decode_device_id == -1 || task_key.decode_device_offset == -1) {
      KLLM_LOG_DEBUG << "Invalid decode device id need assigned for task_key: " << task_key.ToString();
      const TaskKey& decode_task_key = decode_accessor->first;
      task_key.decode_device_id = decode_task_key.decode_device_id;
      task_key.decode_device_offset = decode_task_key.decode_device_offset;
      KLLM_LOG_DEBUG << "Assigned decode device id for task_key: " << task_key.ToString();
    }
    decode_accessor.release();

    // Remove from both maps
    shard.prefill_pending_tasks.erase(task_key);
    shard.decode_confirmed_tasks.erase(task_key);

    return true;  // Can be activated
  }

  return false;  // Cannot be activated, must wait
}

//=============================================================================
// Batch Operations and Utilities
//=============================================================================

std::unordered_map<TaskManager::GroupDevKey, std::vector<TaskKey>, TaskManager::GroupDevKeyHash>
TaskManager::GroupByGroupKeyAndDevice(const std::vector<TaskKey>& batch, bool is_prefill) {
  std::unordered_map<GroupDevKey, std::vector<TaskKey>, GroupDevKeyHash> grouped;
  auto tasks = GetTasksBatch(batch);

  for (size_t i = 0; i < batch.size(); ++i) {
    const auto& task_key = batch[i];
    auto task = tasks[i];

    if (!task) {
      KLLM_LOG_ERROR << "Skipping task_key without corresponding task: " << task_key.ToString();
      continue;
    }

    const std::string& group_key = task->addr;
    int prefill_device_id = task_key.prefill_device_id;
    int decode_device_id = task_key.decode_device_id;
    if (is_prefill) {
      grouped[{group_key, {prefill_device_id, decode_device_id}}].push_back(task_key);
    } else {
      grouped[{group_key, {decode_device_id, prefill_device_id}}].push_back(task_key);
    }
  }

  return grouped;
}

//=============================================================================
// Maintenance Operations
//=============================================================================

void TaskManager::CleanupExpiredTasks(int timeout_seconds) {
  const auto now = ProfileTimer::GetCurrentTime();
  const std::time_t timeout_threshold = timeout_seconds;

  task_arena_.execute([&]() {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, circular_bucket_num_), [&](const tbb::blocked_range<size_t>& range) {
          for (size_t shard_idx = range.begin(); shard_idx != range.end(); ++shard_idx) {
            auto& shard = *shards_[shard_idx];

            // Cleanup prefill pending tasks (erase by key, increment before erase)
            for (auto it = shard.prefill_pending_tasks.begin(); it != shard.prefill_pending_tasks.end();) {
              if (now - it->second >= timeout_threshold) {
                auto key = it->first;
                ++it;
                shard.prefill_pending_tasks.erase(key);
                KLLM_LOG_DEBUG << "Cleanup expired prefill pending task key: " << key.ToString();
              } else {
                ++it;
              }
            }

            // Cleanup decode confirmed tasks (erase by key, increment before erase)
            for (auto it = shard.decode_confirmed_tasks.begin(); it != shard.decode_confirmed_tasks.end();) {
              if (now - it->second >= timeout_threshold) {
                auto key = it->first;
                ++it;
                shard.decode_confirmed_tasks.erase(key);
                KLLM_LOG_DEBUG << "Cleanup expired decode confirmed task key: " << key.ToString();
              } else {
                ++it;
              }
            }
          }
        });
  });

  KLLM_LOG_DEBUG << "Cleaned up expired tasks with timeout " << timeout_seconds << " seconds";
}

void TaskManager::Shutdown() {
  if (shutdown_.exchange(true)) {
    KLLM_LOG_INFO << "TaskManager::Shutdown() called more than once, skipping.";
    return;
  }
  KLLM_LOG_INFO << "Shutting down TaskManager...";
  if (notification_waiter_) {
    notification_waiter_->Stop();
  }

  // 串行清理所有 shard，避免析构时 TBB 初始化
  for (size_t shard_idx = 0; shard_idx < shards_.size(); ++shard_idx) {
    auto& shard = *shards_[shard_idx];
    shard.request_map.clear();
    shard.prefill_pending_tasks.clear();
    shard.decode_confirmed_tasks.clear();
  }
  TaskKey dummy;
  while (processing_buffer_.try_pop(dummy)) {
    // Empty the queue
  }
  KLLM_LOG_INFO << "TaskManager shutdown completed";
}

void TaskManager::SetNotificationWaiter(std::shared_ptr<Waiter> waiter) { notification_waiter_ = waiter; }

}  // namespace ksana_llm
