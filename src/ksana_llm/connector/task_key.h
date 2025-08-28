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
  int req_id;            ///< Request ID for grouping related tasks (primary sharding key)
  int block_idx;         ///< Block index in the computation pipeline
  int layer_idx;         ///< Layer index in the neural network
  int tensor_size;       ///< Size of the tensor data in bytes
  int token;             ///< Token identifier for sequencing
  bool is_skipped_task;  ///< Flag indicating if the task is skipped

  int hash_device_id;  ///< Critical parameters for checking if Taskkeys are the same

  int decode_device_id = -1;      ///< The physical rank ID within the decode group
  int decode_device_offset = -1;  ///< The logical ID of the request within its DP group

  int prefill_device_id = -1;      ///< The physical rank ID within the prefill group
  int prefill_device_offset = -1;  ///< The logical ID of the request within its DP group

  // Timing information
  std::time_t start_time_us;  ///< Task creation timestamp in microseconds

  /*
  Diagram: Node Configuration
  For example: The P node and D node have a total of 4 ranks. The P node is divided into 2 DP groups,
  while the D node is divided into 4 DP groups

                    [P Node]                                       [D Node]
  ---------------------------------------------------------------------------------------------
  | prefill_device_id 0 | --> hash_device_id 0    | decode_device_id 0 | --> hash_device_id 0
  |---------------------|                         |--------------------|
  | prefill_device_id 1 | --> hash_device_id 1    | decode_device_id 1 | --> hash_device_id 0
  |---------------------|                         |--------------------|
  | prefill_device_id 2 | --> hash_device_id 0    | decode_device_id 2 | --> hash_device_id 0
  |---------------------|                         |--------------------|
  | prefill_device_id 3 | --> hash_device_id 1    | decode_device_id 3 | --> hash_device_id 0
  ----------------------------------------------------------------------------------------------
  */

  // Constructors
  TaskKey()
      : req_id(0),
        block_idx(0),
        layer_idx(0),
        hash_device_id(0),
        tensor_size(0),
        token(0),
        is_skipped_task(false),
        start_time_us(0) {}

  TaskKey(int req, int block, int layer, int hash_device_id, int tsize = 0, int ttoken = 0, bool skipped = false,
          int decode_device_id = -1, int decode_device_offset = -1, int prefill_device_id = -1,
          int prefill_device_offset = -1, std::time_t timestamp_us = 0)
      : req_id(req),
        block_idx(block),
        layer_idx(layer),
        tensor_size(tsize),
        token(ttoken),
        is_skipped_task(skipped),
        hash_device_id(hash_device_id),
        decode_device_id(decode_device_id),
        decode_device_offset(decode_device_offset),
        prefill_device_id(prefill_device_id),
        prefill_device_offset(prefill_device_offset),
        start_time_us(timestamp_us) {}

  // Default copy/move semantics
  TaskKey(const TaskKey& other) = default;
  TaskKey(TaskKey&& other) noexcept = default;
  TaskKey& operator=(const TaskKey& other) = default;
  TaskKey& operator=(TaskKey&& other) noexcept = default;

  // Equality comparison (excludes token and timestamp for logical equality)
  bool operator==(const TaskKey& other) const {
    return req_id == other.req_id && block_idx == other.block_idx && layer_idx == other.layer_idx &&
           hash_device_id == other.hash_device_id && tensor_size == other.tensor_size;
  }

  // Priority comparison for priority queue (earlier timestamp = higher priority)
  bool operator<(const TaskKey& other) const {
    return start_time_us > other.start_time_us;  // Earlier timestamp has higher priority
  }

  // String representation for debugging
  std::string ToString() const {
    std::ostringstream oss;
    oss << "req_id=" << req_id << ", block_idx=" << block_idx << ", layer_idx=" << layer_idx
        << ", tensor_size=" << tensor_size << ", token=" << token << ", is_skipped_task=" << is_skipped_task
        << ", hash_device_id=" << hash_device_id << ", decode_device_id=" << decode_device_id
        << ", decode_device_offset=" << decode_device_offset << ", prefill_device_id=" << prefill_device_id
        << ", prefill_device_offset=" << prefill_device_offset;
    return oss.str();
  }

  /**
   * @brief Get shard index based on req_id
   * @param max_shards Maximum number of shards
   * @return Shard index in range [0, max_shards)
   */
  size_t GetShardIndex(size_t max_shards) const { return static_cast<size_t>(req_id) % max_shards; }

  /**
   * @brief Set is_skipped_task flag
   * @param flag Value to set is_skipped_task flag
   */
  void SetIsSkippedTaskFlag(bool flag) { is_skipped_task = flag; }

  /**
   * @brief Get is_skipped_task flag
   * @return Value of is_skipped_task flag
   */
  bool GetIsSkippedTaskFlag() const { return is_skipped_task; }

  // Hash function for TaskKey (excludes token for consistent hashing)
  struct Hash {
    size_t operator()(const TaskKey& key) const {
      uint64_t combined = (static_cast<uint64_t>(key.req_id) << 32) | ((key.tensor_size > 0 ? 1ULL : 0ULL) << 16) |
                          (static_cast<uint64_t>(key.block_idx) << 8) | (static_cast<uint64_t>(key.layer_idx) << 4) |
                          (static_cast<uint64_t>(key.hash_device_id));
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

  // High-performance batch serialization: directly from pointer and count
  static std::vector<uint8_t> BatchSerialize(const TaskKey* keys, size_t count) {
    if (!keys || count == 0) {
      return std::vector<uint8_t>();
    }
    std::vector<uint8_t> buffer(count * sizeof(TaskKey));
    std::memcpy(buffer.data(), keys, buffer.size());
    return buffer;
  }

  // Zero-copy serialization: direct access to raw bytes (use with caution)
  // This provides direct access to the underlying bytes without copying
  // Only safe when data lifetime is guaranteed and platforms are compatible
  static const uint8_t* BatchSerializePtr(const TaskKey* keys, size_t count, size_t& out_size) {
    if (!keys || count == 0) {
      out_size = 0;
      return nullptr;
    }
    out_size = count * sizeof(TaskKey);
    return reinterpret_cast<const uint8_t*>(keys);
  }

  // Convenience wrapper for vector serialization
  static std::vector<uint8_t> BatchSerialize(const std::vector<TaskKey>& keys) {
    return BatchSerialize(keys.data(), keys.size());
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

    return TaskKey(task->req_id, task->tensor.block_idx, task->tensor.layer_idx, task->tensor.hash_device_id,
                   tensor_size, task->token, task->is_skipped_task, task->decode_device_id, task->decode_device_offset,
                   task->prefill_device_id, task->prefill_device_offset, ProfileTimer::GetCurrentTimeInUs());
  }

  static const TaskKey* DeserializeBatchPtr(const uint8_t* data, size_t size, size_t& out_count) {
    if (!data || size == 0 || size % sizeof(TaskKey) != 0) {
      out_count = 0;
      return nullptr;
    }
    out_count = size / sizeof(TaskKey);
    return reinterpret_cast<const TaskKey*>(data);
  }

  // High-performance batch deserialization: char* version
  static const TaskKey* DeserializeBatchPtr(const char* data, size_t size, size_t& out_count) {
    return DeserializeBatchPtr(reinterpret_cast<const uint8_t*>(data), size, out_count);
  }

  // High-performance batch deserialization: vector<uint8_t> version
  static const TaskKey* DeserializeBatchPtr(const std::vector<uint8_t>& data, size_t& out_count) {
    return DeserializeBatchPtr(data.data(), data.size(), out_count);
  }

  // Convenience wrapper for backward compatibility
  static std::vector<TaskKey> DeserializeBatch(const std::vector<uint8_t>& data) {
    size_t count;
    const TaskKey* keys = DeserializeBatchPtr(data, count);
    if (!keys || count == 0) {
      return std::vector<TaskKey>();
    }
    return std::vector<TaskKey>(keys, keys + count);
  }

  // Convenience wrapper: char* version for backward compatibility
  static std::vector<TaskKey> DeserializeBatch(const char* data, size_t size) {
    size_t count;
    const TaskKey* keys = DeserializeBatchPtr(data, size, count);
    if (!keys || count == 0) {
      return std::vector<TaskKey>();
    }
    return std::vector<TaskKey>(keys, keys + count);
  }
};

// Static assertions to ensure safe serialization (placed after complete type definition)
static_assert(std::is_trivially_copyable_v<TaskKey>, "TaskKey must be trivially copyable for safe serialization");
static_assert(std::is_standard_layout_v<TaskKey>, "TaskKey must have standard layout for safe serialization");

}  // namespace ksana_llm
