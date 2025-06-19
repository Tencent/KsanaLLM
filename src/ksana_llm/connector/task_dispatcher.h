/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ksana_llm/connector/communicator/communicator_manager.h"
#include "ksana_llm/connector/config.h"
#include "ksana_llm/connector/task_manager.h"
#include "ksana_llm/runtime/threadpool.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/connector/cuda_buffer_pool.h"
#  include "ksana_llm/utils/nvidia/cuda_utils.h"
#  include "ksana_llm/utils/nvidia/nccl_utils.h"
#endif

namespace ksana_llm {
// Forward declarations only - 避免头文件依赖传递
class ZmqCommunicator;   // forward declare
class NcclCommunicator;  // forward declare - 只在NCCL模式下使用
/**
 * @class TaskDispatcher
 * @brief Dispatches tasks across different communication protocols
 *
 * This class is responsible for dispatching tasks to appropriate
 * communicators based on the task type and configuration.
 * It uses Communicator interface rather than directly depending
 * on concrete implementations, improving extensibility.
 */
class TaskDispatcher {
 public:
  /**
   * @brief Constructor
   * @param config Connector configuration
   * @param task_manager Task manager instance
   * @param comm_manager Communicator manager instance（所有权转移）
   */
  TaskDispatcher(const ConnectorConfig& config, std::shared_ptr<TaskManager> task_manager,
                 std::shared_ptr<CommunicatorManager> comm_manager);
  /**
   * @brief Destructor
   */
  virtual ~TaskDispatcher();

  /**
   * @brief Shutdown the dispatcher
   */
  virtual void Shutdown();

  virtual Status Initialize();

  /**
   * @brief Send tasks to prefill
   */
  virtual void SendToPrefill();

  /**
   * @brief Register prefill receive callback
   */
  virtual void RegisterPrefillRecv();

  /**
   * @brief Register decode receive callback
   */
  virtual void RegisterDecodeRecv();

  /**
   * @brief Process prefill received tasks from queue
   */
  virtual void ProcessPrefillReceivedTasks();

  virtual void HandlePrefillGroupBatch(const std::pair<std::pair<std::string, int>, std::vector<TaskKey>>& group_batch);

  /**
   * @brief Retry failed tasks for a specific group
   * @param group_key The group key that failed
   * @param device_idx The device index that failed
   * @param failed_tasks The tasks that failed to send
   */
  void RetryFailedTasks(const std::string& group_key, int device_idx, const std::vector<TaskKey>& failed_tasks);

 private:
  /**
   * @brief 生成连接的唯一标识符
   * @param group_key 组标识
   * @param device_idx 设备索引
   * @return 连接的唯一标识符字符串
   */
  std::string MakeConnectionId(const std::string& group_key, int device_idx) const;
  std::pair<std::string, int> ParseConnectionId(const std::string& connection_id);

  /**
   * @brief 检查连接是否准备就绪，首次连接会尝试等待
   * @param group_key 组标识
   * @param device_idx 设备索引
   * @param group_vec 任务组向量（如果连接失败需要重试）
   * @return 连接是否准备好
   */
  bool CheckConnection(const std::string& group_key, int device_idx);

  /**
   * @brief 检查任务是否为首次尝试
   * @param conn_key 连接键
   * @return 是否为首次尝试
   */
  bool IsFirstAttempt(const std::string& conn_key);

  /**
   * @brief 为任务添加张量信息到集合中
   * @param task 需要处理的任务指针
   * @param tk 任务键
   * @param tensors 张量指针集合
   * @param tensor_sizes 张量大小集合
   */
  void AddTensorForTask(const std::shared_ptr<TransferTask>& task, const TaskKey& tk, std::vector<void*>& tensors,
                        std::vector<size_t>& tensor_sizes, std::vector<DataType>& data_types, bool use_dst_ptr = false);

  /**
   * @brief 批量获取任务
   * @param batch_size 批大小
   * @return 获取到的任务批次
   */
  std::vector<TaskKey> BatchTasks(int batch_size);

  /**
   * @brief 处理分组后的批次
   * @param batch 原始任务批次
   */
  void PrefillProcessGroupBatches(const std::vector<TaskKey>& batch);
#ifdef ENABLE_CUDA
  /**
   * @brief Handle NCCL send operation for tensor data transmission
   * @param group_key Group identifier for the batch
   * @param device_idx Target device index
   * @param group_vec Vector of TaskKeys to process
   * @return Status of the operation
   */
  void SendDataToDecodeWithNccl(const std::string& group_key, int device_idx, const std::vector<TaskKey>& group_vec);

  BufferBlock* CopyTaskKeysToDevice(BufferPool* buffer_pool, const std::vector<TaskKey>& task_keys,
                                    size_t task_keys_bytes, int device_idx, cudaEvent_t copy_done);
  void RecvTaskDataWithNccl(const std::string& group_key, int device_idx, const std::vector<TaskKey>& task_keys);
  void RecvTaskKeysWithNccl(const std::string& group_key, int device_idx, TaskKey* host_ptr, size_t bytes);
#endif

 private:  // Changed from private to protected to allow test access
  ZmqCommunicator* zmq_communicator_ = nullptr;
  NcclCommunicator* nccl_communicator_ = nullptr;

  /** @brief Map to track retry counts for task keys to avoid infinite retries */
  std::unordered_map<std::string, int> task_retry_counts_;

  /** @brief Mutex to protect retry counts map */
  std::mutex retry_counts_mutex_;

  /** @brief ThreadPool for parallel task processing */
  std::unique_ptr<ThreadPool> send_thread_pool_;

  std::atomic<size_t> job_id_{1};

  /** @brief Maximum number of retries per task */
  static constexpr int MAX_TASK_RETRIES = 3;

  /** @brief Running state flag */
  std::atomic<bool> running_{true};

  /** @brief Monitor thread */
  std::thread decode_process_thread_;

  /** @brief Prefill receive processing thread */
  std::thread prefill_recv_thread_;

  ConnectorConfig config_;

  /** @brief Task manager instance */
  std::shared_ptr<TaskManager> task_manager_;

  /** @brief Communicator manager（所有权转移） */
  std::shared_ptr<CommunicatorManager> comm_manager_;

#ifdef ENABLE_CUDA
  std::map<int, std::unique_ptr<BufferPool>> device_pools_;
  // cudaStream_t copy_stream = nullptr;
  std::map<int, cudaStream_t> device_streams_;
#endif
};
}  // namespace ksana_llm
