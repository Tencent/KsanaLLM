/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstring>
#include <ctime>
#include <memory>
#include <sstream>
#include <vector>
#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/transfer/transfer_types.h"
#include "ksana_llm/utils/device_types.h"

namespace ksana_llm {

/**
 * @brief Key structure for identifying and managing transfer tasks
 *
 * TaskKey serves as a unique identifier for tasks in the transfer system,
 * combining request information, tensor metadata, and timing information.
 * The key is designed to support efficient sharding based on req_id.
 */
struct TaskKey {
  // Core identifiers
  int req_id;       ///< Request ID for grouping related tasks (primary sharding key)
  int block_idx;    ///< Block index in the computation pipeline
  int layer_idx;    ///< Layer index in the neural network
  int device_idx;   ///< Target device index
  int tensor_size;  ///< Size of the tensor data in bytes
  int token;        ///< Token identifier for sequencing

  // Timing information
  std::time_t start_time_us;  ///< Task creation timestamp in microseconds

  // Constructors
  TaskKey() : req_id(0), block_idx(0), layer_idx(0), device_idx(0), tensor_size(0), token(0), start_time_us(0) {}

  TaskKey(int req, int block, int layer, int device, int tsize = 0, int ttoken = 0, std::time_t timestamp_us = 0)
      : req_id(req),
        block_idx(block),
        layer_idx(layer),
        device_idx(device),
        tensor_size(tsize),
        token(ttoken),
        start_time_us(timestamp_us) {}

  // Default copy/move semantics
  TaskKey(const TaskKey& other) = default;
  TaskKey(TaskKey&& other) noexcept = default;
  TaskKey& operator=(const TaskKey& other) = default;
  TaskKey& operator=(TaskKey&& other) noexcept = default;

  // Equality comparison (excludes token and timestamp for logical equality)
  bool operator==(const TaskKey& other) const {
    return req_id == other.req_id && block_idx == other.block_idx && layer_idx == other.layer_idx &&
           device_idx == other.device_idx && tensor_size == other.tensor_size;
  }

  // Priority comparison for priority queue (earlier timestamp = higher priority)
  bool operator<(const TaskKey& other) const {
    return start_time_us > other.start_time_us;  // Earlier timestamp has higher priority
  }

  // String representation for debugging
  std::string ToString() const {
    std::ostringstream oss;
    oss << "req_id=" << req_id << ", block_idx=" << block_idx << ", layer_idx=" << layer_idx
        << ", device_idx=" << device_idx << ", tensor_size=" << tensor_size << ", token=" << token;
    return oss.str();
  }

  /**
   * @brief Get shard index based on req_id
   * @param max_shards Maximum number of shards
   * @return Shard index in range [0, max_shards)
   */
  size_t GetShardIndex(size_t max_shards) const { return static_cast<size_t>(req_id) % max_shards; }

  // Hash function for TaskKey (excludes token for consistent hashing)
  struct Hash {
    size_t operator()(const TaskKey& key) const {
      uint64_t combined = (static_cast<uint64_t>(key.req_id) << 32) | ((key.tensor_size > 0 ? 1ULL : 0ULL) << 16) |
                          (static_cast<uint64_t>(key.block_idx) << 8) | (static_cast<uint64_t>(key.layer_idx) << 4) |
                          (static_cast<uint64_t>(key.device_idx));
      return std::hash<uint64_t>()(combined);
    }
  };

  // TBB compatible hash comparator
  struct HashCompare {
    static size_t hash(const TaskKey& key) { return Hash {} (key); }

    static bool equal(const TaskKey& lhs, const TaskKey& rhs) { return lhs == rhs; }
  };

  // Serialization methods
  std::vector<uint8_t> Serialize() const {
    std::vector<uint8_t> data(sizeof(TaskKey));
    std::memcpy(data.data(), this, sizeof(TaskKey));
    return data;
  }

  static TaskKey Deserialize(const std::vector<uint8_t>& data) {
    TaskKey key;
    if (data.size() >= sizeof(TaskKey)) {
      std::memcpy(&key, data.data(), sizeof(TaskKey));
    }
    return key;
  }

  // Batch serialization/deserialization methods
  static std::vector<TaskKey> DeserializeBatch(const std::vector<uint8_t>& data) {
    std::vector<TaskKey> result;
    size_t count = data.size() / sizeof(TaskKey);
    if (count == 0 || data.size() % sizeof(TaskKey) != 0) {
      return result;
    }

    // Direct cast approach - more efficient, avoiding memcpy
    const TaskKey* keys_ptr = reinterpret_cast<const TaskKey*>(data.data());
    result.assign(keys_ptr, keys_ptr + count);

    return result;
  }

  static std::vector<TaskKey> DeserializeBatch(const char* data, size_t size) {
    std::vector<TaskKey> result;
    if (!data || size == 0) {
      return result;
    }

    size_t count = size / sizeof(TaskKey);
    if (count == 0 || size % sizeof(TaskKey) != 0) {
      return result;
    }

    // Direct cast approach - more efficient, avoiding memcpy
    const TaskKey* keys_ptr = reinterpret_cast<const TaskKey*>(data);
    result.assign(keys_ptr, keys_ptr + count);

    return result;
  }

  static std::vector<uint8_t> BatchSerialize(const std::vector<TaskKey>& keys) {
    std::vector<uint8_t> buffer(keys.size() * sizeof(TaskKey));
    std::memcpy(buffer.data(), keys.data(), buffer.size());
    return buffer;
  }

  /**
   * @brief Create a TaskKey from a TransferTask
   * @param task Source transfer task
   * @return Generated TaskKey with computed tensor size and timestamp
   */
  static TaskKey CreateFromTransferTask(const std::shared_ptr<TransferTask>& task) {
    if (!task) {
      KLLM_LOG_ERROR << "CreateFromTransferTask called with null task";
      return TaskKey();
    }

    int tensor_size = 0;
    if (!task->tensor.shape.empty()) {
      tensor_size = task->tensor.GetElementNumber() * GetTypeSize(task->tensor.dtype);
    }

    return TaskKey(task->req_id, task->tensor.block_idx, task->tensor.layer_idx, task->tensor.device_idx, tensor_size,
                   task->token, ProfileTimer::GetCurrentTimeInUs());
  }
};

}  // namespace ksana_llm
