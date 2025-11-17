/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_hash_map.h>
#include <oneapi/tbb/concurrent_priority_queue.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <oneapi/tbb/task_arena.h>

#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/connector/task_key.h"
#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/transfer/transfer_types.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

// Default expiration time for tasks in seconds
constexpr int TASK_MANAGER_DEFAULT_EXPIRE_SECONDS = 900;  // 15 minutes

/**
 * @brief Shard-based high-performance task manager for distributed tensor transfer operations
 *
 * TaskManager provides a thread-safe, singleton-based system for managing transfer tasks
 * in a distributed LLM inference environment. It uses sharded data structures based on
 * req_id for optimal concurrent access and scalability.
 *
 * Key features:
 * - Sharded data structures based on req_id for better concurrency
 * - Lock-free operations using TBB concurrent containers
 * - Promise-based task synchronization between prefill and decode phases
 * - Automatic cleanup of expired tasks with parallel processing
 * - High-performance batch operations
 * - Configurable shard size based on circular_bucket_num
 */
class TaskManager {
 public:
  using TaskNotifyCallback = std::function<void()>;
  using GroupDevKey =
      std::pair<std::string, std::pair<int, int>>;  ///< (group_key, (src_device_idx, dst_device_idx)) pair

  // The stage of a task in the pipeline
  struct TaskStage {
      std::time_t start_time = ProfileTimer::GetCurrentTime();

    // If Decode confirms this task, transfer it to DecodeNode and set decode_confirmed to 1.
    // If Prefill produces this task first and Decode does not confirm it, decode_confirmed remains 0.
      std::atomic<int> decode_confirmed = 0;
  };

  /**
   * @brief Task shard containing all data structures for a subset of req_ids
   */
  struct TaskShard {
    // Core task storage
    tbb::concurrent_hash_map<TaskKey, std::shared_ptr<TransferTask>, TaskKey::HashCompare> request_map;

    // Promise synchronization maps
    tbb::concurrent_hash_map<TaskKey, TaskStage, TaskKey::HashCompare> decode_confirmed_tasks;

    TaskShard() = default;
    TaskShard(const TaskShard&) = delete;
    TaskShard& operator=(const TaskShard&) = delete;
    TaskShard(TaskShard&&) = default;
    TaskShard& operator=(TaskShard&&) = default;
  };

  /**
   * @brief Construct TaskManager with specified parameters
   * @param circular_bucket_num Maximum number of concurrent batches (determines shard count)
   * @param bucket_size_hint Initial bucket size hint for hash tables
   */
  explicit TaskManager(int circular_bucket_num, int bucket_size_hint = 16384, int circular_thread_num = 4);

  ~TaskManager() = default;

  // Disable copy operations
  TaskManager(const TaskManager&) = delete;
  TaskManager& operator=(const TaskManager&) = delete;

  /**
   * @brief Get singleton instance of TaskManager
   * @param circular_bucket_num Maximum number of concurrent batches (default: 256)
   * @param bucket_size_hint Hash table bucket size hint (default: 16384)
   * @return Shared pointer to TaskManager singleton instance
   */
  static std::shared_ptr<TaskManager> GetInstance(int circular_bucket_num = 16, int bucket_size_hint = 16384);

  //=============================================================================
  // Processing Buffer Operations (Priority Queue Interface)
  //=============================================================================

  /**
   * @brief Add a task to the processing buffer
   * @param task_key TaskKey to add to the processing buffer
   */
  void PutProcessingBuffer(const TaskKey& task_key);

  /**
   * @brief Get a task from the processing buffer (round-robin across shards)
   * @return TaskKey with highest priority, or default TaskKey if empty
   */
  TaskKey GetProcessingBuffer();

  /**
   * @brief Get tasks from multiple shards in parallel
   * @param batch_size Maximum number of tasks to retrieve
   * @return Vector of TaskKeys from different shards
   */
  std::vector<TaskKey> GetProcessingBufferBatch(int batch_size);

  /**
   * @brief Check if all processing buffers are empty
   * @return true if all buffers are empty, false otherwise
   */
  bool IsProcessingBufferEmpty() const;

  /**
   * @brief Get the total size of all processing buffers
   * @return Number of tasks across all processing buffers
   */
  size_t GetProcessingBufferSize() const;

  //=============================================================================
  // Core Task Management Operations
  //=============================================================================

  /**
   * @brief Create a TaskKey from a TransferTask
   * @param task Source transfer task
   * @return Generated TaskKey with computed tensor size and timestamp
   */
  TaskKey CreateTaskKey(const std::shared_ptr<TransferTask>& task);

  /**
   * @brief Add a task to the manager
   * @param key TaskKey identifier
   * @param task Shared pointer to the TransferTask
   */
  void AddTask(const TaskKey& key, std::shared_ptr<TransferTask> task);

  /**
   * @brief Retrieve a task by its key
   * @param key TaskKey to look up
   * @return Shared pointer to TransferTask, or nullptr if not found
   */
  std::shared_ptr<TransferTask> GetTask(const TaskKey& key);

  /**
   * @brief Retrieve multiple tasks in a single operation
   * @param keys Vector of TaskKeys to look up
   * @return Vector of shared pointers (nullptr for missing tasks)
   */
  std::vector<std::shared_ptr<TransferTask>> GetTasksBatch(const std::vector<TaskKey>& keys);

  /**
   * @brief Mark a task as completed and remove it from the manager
   * @param key TaskKey of the task to complete
   */
  void CompleteTask(const TaskKey& key);

  /**
   * @brief Get total number of tasks across all shards
   * @return Total task count
   */
  size_t GetTaskCount() const;

  /**
   * @brief Get hash table load factor for performance monitoring
   * @return Average load factor across all shards
   */
  float GetLoadFactor() const;

  //=============================================================================
  // Promise-based Task Synchronization
  //=============================================================================

  /**
   * @brief Register a receive callback function for received tasks from decode phase
   * @param task_keys Vector of TaskKeys confirmed by decode phase
   */
  void RegisterDecodeConfirmedTasks(const std::vector<TaskKey>& task_keys);

  /**
   * @brief Add a task to the unconfirmed queue
   * @param task_key TaskKey to add to unconfirmed queue
   */
  void AddUnconfirmedTask(const TaskKey& task_key);

  /**
   * @brief Try to activate a unconfirmed task
   * @param task_key TaskKey to try to activate
   * @return true if task was activated, false if it needs to wait
   */
  bool TryActivateUnconfirmedTask(TaskKey& task_key);

  //=============================================================================
  // Batch Operations and Utilities
  //=============================================================================

  /**
   * @brief Hash function for GroupDevKey (group_key, (src_device_idx, dst_device_idx)) pairs
   */
  struct GroupDevKeyHash {
    std::size_t operator()(const GroupDevKey& k) const {
      return std::hash<std::string>()(k.first) ^ (std::hash<int>()(k.second.first) << 1) ^
             (std::hash<int>()(k.second.second) << 2);
    }
  };

  /**
   * @brief Group tasks by their destination group and device
   * @param batch Vector of TaskKeys to group
   * @return Map of (group_key, device_idx) -> vector of TaskKeys
   */
  std::unordered_map<GroupDevKey, std::vector<TaskKey>, GroupDevKeyHash> GroupByGroupKeyAndDevice(
      const std::vector<TaskKey>& batch, bool is_prefill);

  //=============================================================================
  // Maintenance Operations
  //=============================================================================

  /**
   * @brief Clean up expired tasks using parallel processing
   * @param timeout_seconds Age threshold for task expiration
   */
  void CleanupExpiredTasks(int timeout_seconds);

  /**
   * @brief Shutdown the task manager and clean up all resources
   */
  void Shutdown();

  /**
   * @brief Set task notification callback
   * @param waiter Shared pointer to waiter for notifications
   */
  void SetNotificationWaiter(std::shared_ptr<Waiter> waiter);

  //=============================================================================
  // Public Member Variables
  //=============================================================================

  // Notification system (public for direct access)
  std::shared_ptr<Waiter> notification_waiter_;  ///< Notifies when new tasks are available

  tbb::concurrent_priority_queue<TaskKey> usage_tasks_;

 private:
  //=============================================================================
  // Member Variables
  //=============================================================================

  // Sharded data structures
  std::vector<std::unique_ptr<TaskShard>> shards_;  ///< Task shards indexed by req_id % circular_bucket_num
  const int circular_bucket_num_;                   ///< Maximum batch size (number of shards)

  // Task arena for parallel operations
  mutable tbb::task_arena task_arena_;  ///< Arena for controlling parallel operations
  mutable std::atomic<size_t> round_robin_counter_{0};
  std::atomic<bool> shutdown_{false};  // 防止重复 Shutdown

  // Processing buffer (priority queue)
  tbb::concurrent_priority_queue<TaskKey> processing_buffer_;
  //=============================================================================
  // Private Methods
  //=============================================================================

  /**
   * @brief Get shard index for a request ID
   * @param req_id Request ID
   * @return Shard index in the range [0, circular_bucket_num)
   */
  size_t GetShardIndex(int req_id) const { return static_cast<size_t>(req_id) % circular_bucket_num_; }

  /**
   * @brief Get shard for a request ID
   * @param req_id Request ID
   * @return Reference to the corresponding TaskShard
   */
  TaskShard& GetShard(int req_id) { return *shards_[GetShardIndex(req_id)]; }

  /**
   * @brief Get shard for a request ID (const version)
   * @param req_id Request ID
   * @return Const reference to the corresponding TaskShard
   */
  const TaskShard& GetShard(int req_id) const { return *shards_[GetShardIndex(req_id)]; }
};

}  // namespace ksana_llm