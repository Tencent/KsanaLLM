/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/connector/task_dispatcher.h"
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for_each.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include "ksana_llm/connector/communicator/communicator_manager.h"
#include "ksana_llm/connector/task_manager.h"
#include "ksana_llm/utils/device_types.h"

#include "ksana_llm/connector/communicator/zmq/zmq_communicator.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/connector/communicator/nvida/nccl_communicator.h"
#endif

namespace ksana_llm {
TaskDispatcher::TaskDispatcher(const ConnectorConfig& config, std::shared_ptr<TaskManager> task_manager,
                               std::shared_ptr<CommunicatorManager> comm_manager)
    : config_(config), task_manager_(task_manager), comm_manager_(comm_manager) {}

TaskDispatcher::~TaskDispatcher() {
  if (running_.load()) {
    Shutdown();
  }
}

void TaskDispatcher::Shutdown() {
  // Check if already shutdown to ensure idempotency
  if (!running_.load()) {
    return;
  }
  // 先设置 running_ = false，通知线程退出
  running_ = false;

  // 等待所有线程结束 - 修复：检查线程是否joinable而不是角色
  if (decode_process_thread_.joinable()) {
    decode_process_thread_.join();
  }
  if (prefill_recv_thread_.joinable()) {
    prefill_recv_thread_.join();
  }

  // 停止线程池
  if (send_thread_pool_) {
    send_thread_pool_->Stop();
  }

  // 关闭通信管理器
  if (comm_manager_) comm_manager_->Shutdown();

#ifdef ENABLE_CUDA
  // Clean up device buffer pools
  if (!device_pools_.empty()) {
    device_pools_.clear();  // smart pointers will automatically handle cleanup
  }

#endif
}

Status TaskDispatcher::Initialize() {
  // Check required dependencies first
  if (!comm_manager_ || !task_manager_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "TaskDispatcher initialize failed");
  }
  running_ = true;

  // 首先初始化 CommunicatorManager（如果还未初始化）
  if (!comm_manager_->IsInitialized()) {
    auto status = comm_manager_->Initialize();
    if (!status.OK()) {
      KLLM_LOG_ERROR << "CommunicatorManager initialization failed: " << status.GetMessage();
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "CommunicatorManager initialization failed");
    }
  }

  // 初始化线程池，线程数量优先使用配置中的send_thread_num
  int thread_count = config_.send_thread_num;
  KLLM_LOG_INFO << "Initializing task processing thread pool with " << thread_count << " threads";
  send_thread_pool_ = std::make_unique<ThreadPool>(thread_count);
  send_thread_pool_->Start();

  // ZMQ是必须的通信组件
  auto status = comm_manager_->CreateZmqCommunicator();
  if (!status.OK()) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "ZMQ communicator is not initialized");
  }

  zmq_communicator_ = comm_manager_->GetZmqCommunicator();
  if (!zmq_communicator_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to get ZMQ communicator");
  }

  // 只在NCCL模式下才初始化NCCL
#ifdef ENABLE_CUDA
  if (config_.communication_type == CommunicationType::NCCL) {
    status = comm_manager_->CreateNcclCommunicator();
    if (!status.OK()) {
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL communicator is not initialized");
    }
    nccl_communicator_ = comm_manager_->GetNcclCommunicator();
  }
  // CUDA is available, initialize buffer pools and streams for each device
  for (int dev = 0; dev < config_.device_count; ++dev) {
    device_pools_[dev] =
        std::make_unique<BufferPool>(dev, config_.send_thread_num * 2, config_.transfer_batch * sizeof(TaskKey));
    cudaSetDevice(dev);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    device_streams_[dev] = stream;
  }
#endif

  // 根据角色启动相应的线程和注册回调
  if (config_.group_role == GroupRole::PREFILL || config_.group_role == GroupRole::BOTH) {
    RegisterPrefillRecv();
    // Start prefill processing thread
    prefill_recv_thread_ = std::thread(&TaskDispatcher::ProcessPrefillReceivedTasks, this);
  }

  if (config_.group_role == GroupRole::DECODE || config_.group_role == GroupRole::BOTH) {
    RegisterDecodeRecv();
    decode_process_thread_ = std::thread(&TaskDispatcher::SendToPrefill, this);
  }

  return Status();
}

void TaskDispatcher::SendToPrefill() {
  while (running_) {
    if (task_manager_->IsProcessingBufferEmpty()) {
      task_manager_->notification_waiter_->Wait();
      task_manager_->notification_waiter_->Reset(1);
      continue;
    }
    std::time_t start_time = ProfileTimer::GetCurrentTimeInUs();
    std::vector<TaskKey> batch = BatchTasks(config_.transfer_batch);
    KLLM_LOG_DEBUG << "Fetched batch of size: " << batch.size()
                   << " GetTask batch time is: " << ProfileTimer::GetCurrentTimeInUs() - start_time;
    if (batch.empty()) continue;

    auto group_batches = task_manager_->GroupByGroupKeyAndDevice(batch);
    if (!send_thread_pool_) {
      KLLM_LOG_ERROR << "Thread pool is not initialized. Cannot process task batches.";
      continue;
    }

    for (const auto& [key, group_vec] : group_batches) {
      auto future = send_thread_pool_->Submit([this, key, group_vec] {
        const std::string& group_key = key.first;
        int device_idx = key.second;
        if (!CheckConnection(group_key, device_idx)) {
          return;
        }
        std::vector<uint8_t> buf = TaskKey::BatchSerialize(group_vec);
        Status status = zmq_communicator_->Send(group_key, device_idx, 0, buf.data(), buf.size(), DataType::TYPE_BYTES);
        KLLM_LOG_DEBUG << "Step_1 Decode task_keys sent to Prefill for group key: " << group_key
                       << " and first task_key is: " << group_vec[0].ToString();
        if (!status.OK()) {
          return;
        }
      });
    }
  }
}

void TaskDispatcher::HandlePrefillGroupBatch(
    const std::pair<std::pair<std::string, int>, std::vector<TaskKey>>& group_batch) {
  const std::string& group_key = group_batch.first.first;
  int device_idx = group_batch.first.second;
  if (!CheckConnection(group_key, device_idx)) {
    return;
  }
  const std::vector<TaskKey>& group_vec = group_batch.second;

  if (config_.communication_type == CommunicationType::ZMQ) {
    for (const TaskKey& tk : group_vec) {
      auto task = task_manager_->GetTask(tk);
      if (task) {
        KLLM_LOG_DEBUG << "Step_3: Prefill preparing send task_keys and tensors: " << tk.ToString();
        std::vector<char> buf(sizeof(TaskKey) + tk.tensor_size);
        memcpy(buf.data(), &tk, sizeof(TaskKey));
        if (tk.tensor_size > 0) {
#ifdef ENABLE_CUDA
          cudaStream_t cur_stream = device_streams_[device_idx];
          CUDA_CHECK(cudaMemcpyAsync(buf.data() + sizeof(TaskKey), task->tensor.src_ptr, tk.tensor_size,
                                     cudaMemcpyDeviceToHost, cur_stream));
          CUDA_CHECK(cudaStreamSynchronize(cur_stream));
#endif
        }
        Status status = zmq_communicator_->Send(group_key, device_idx, 0, buf.data(), buf.size(), DataType::TYPE_BYTES);
        if (!status.OK()) {
          KLLM_LOG_WARNING << "Failed to send task_keys to Decode";
          return;
        }
      }
    }
    return;
  }
  std::string connection_id = MakeConnectionId(group_key, device_idx);
  std::string signal = connection_id + "|" + std::to_string(group_vec.size());
  Status status = zmq_communicator_->Send(group_key, device_idx, 0, signal.data(), signal.size(), DataType::TYPE_BYTES);
  KLLM_LOG_INFO << "Fetched batch of size: " << group_vec.size();
  if (!status.OK()) {
    KLLM_LOG_WARNING << "Failed to send signal to Decode";
    return;
  }
  KLLM_LOG_DEBUG << "Step_3: Prefill signal sent to Decode for group_key=" << group_key << ", device_idx=" << device_idx
                 << ", first task_keys is: =" << group_vec[0].ToString();
  if (config_.communication_type == CommunicationType::NCCL) {
#ifdef ENABLE_CUDA
    SendDataToDecodeWithNccl(group_key, device_idx, group_vec);
    return;
#endif
  } else if (config_.communication_type == CommunicationType::ZMQ) {
    KLLM_LOG_ERROR << "Unsupported communication type: ZMQ";
    return;
  } else {
    KLLM_LOG_ERROR << "Unsupported communication type: default";
    return;
  }
}

void TaskDispatcher::RegisterPrefillRecv() {
  if (!zmq_communicator_) {
    KLLM_LOG_ERROR << "ZMQ communicator is null in RegisterPrefillRecv";
    return;
  }
  zmq_communicator_->SetReceiveCallback([this](const char* data, size_t size, uint64_t job_id, void* /*user_data*/) {
    std::vector<TaskKey> received = TaskKey::DeserializeBatch(data, size);
    if (received.empty()) {
      return;
    }

    // 使用TaskManager的封装方法来处理接收到的Decode确认
    KLLM_LOG_DEBUG << "Step_2 Prefill received " << received.size() << " task_keys from Decode";
    task_manager_->RegisterDecodeConfirmedTasks(received);
  });
}

void TaskDispatcher::ProcessPrefillReceivedTasks() {
  KLLM_LOG_INFO << "ProcessPrefillReceivedTasks thread started";

  while (running_) {
    if (task_manager_->IsProcessingBufferEmpty()) {
      task_manager_->notification_waiter_->Wait();
      task_manager_->notification_waiter_->Reset(1);
      continue;
    }
    std::time_t start_time = ProfileTimer::GetCurrentTimeInUs();
    std::vector<TaskKey> batch = BatchTasks(config_.transfer_batch);
    KLLM_LOG_DEBUG << "Fetched batch of size: " << batch.size()
                   << " GetTask batch time is: " << ProfileTimer::GetCurrentTimeInUs() - start_time;
    if (batch.empty()) continue;
    PrefillProcessGroupBatches(batch);
  }
}

void TaskDispatcher::RegisterDecodeRecv() {
  if (!zmq_communicator_) {
    KLLM_LOG_ERROR << "ZMQ communicator is null in RegisterDecodeRecv";
    return;
  }
  zmq_communicator_->SetReceiveCallback([this](const char* data, size_t size, uint64_t job_id, void* /*user_data*/) {
    if (size == 0 || data == nullptr) {
      KLLM_LOG_ERROR << "Received empty data in RegisterDecodeRecv";
      return;
    }
    if (config_.communication_type == CommunicationType::ZMQ) {
      TaskKey tk;
      memcpy(&tk, data, sizeof(TaskKey));

      KLLM_LOG_DEBUG << "Step_7: Decode preparing to receive tensor for task_key: " << tk.ToString();
      auto task = task_manager_->GetTask(tk);
      if (!task) {
        KLLM_LOG_ERROR << "Task not found for task_key: " << tk.ToString();
        return;
      }
      if (tk.tensor_size == 0) {
        std::memcpy(task->dst_ptr, &tk.token, sizeof(int32_t));
      } else {
#ifdef ENABLE_CUDA
        cudaStream_t cur_stream = device_streams_[tk.device_idx];
        CUDA_CHECK(
            cudaMemcpyAsync(task->dst_ptr, data + sizeof(TaskKey), tk.tensor_size, cudaMemcpyHostToDevice, cur_stream));
        CUDA_CHECK(cudaStreamSynchronize(cur_stream));
#endif
      }
      task_manager_->CompleteTask(tk);
      return;
    }
    std::string signal(data, size);
    // 解码信令，格式为 connection_id|count
    auto sep = signal.rfind('|');
    if (sep == std::string::npos) {
      KLLM_LOG_ERROR << "Invalid signal format in RegisterDecodeRecv: " << signal;
      return;
    }
    std::string connection_id = signal.substr(0, sep);
    size_t count = std::stoul(signal.substr(sep + 1));
    auto [group_key, device_idx] = TaskDispatcher::ParseConnectionId(connection_id);
    std::vector<TaskKey> task_keys(count);
    KLLM_LOG_DEBUG << "Step_4: Decode signal recved from Prefill for group_key=" << group_key
                   << ", device_idx=" << device_idx << ", count=" << count;
    switch (config_.communication_type) {
      case CommunicationType::NCCL:
#ifdef ENABLE_CUDA
        RecvTaskKeysWithNccl(group_key, device_idx, task_keys.data(), count * sizeof(TaskKey));
        KLLM_LOG_DEBUG << "Step_7: Decode received task_keys from Prefill for group_key: " << group_key
                       << ", first task_key is: " << task_keys[0].ToString();
        RecvTaskDataWithNccl(group_key, device_idx, task_keys);
#endif
        break;
      case CommunicationType::ZMQ:
        KLLM_LOG_ERROR << "Unsupported communication type for Decode role: ZMQ";
        break;
      default:
        KLLM_LOG_ERROR << "Unsupported communication type for Decode role";
        break;
    }
  });
}

void TaskDispatcher::RetryFailedTasks(const std::string& group_key, int device_idx,
                                      const std::vector<TaskKey>& failed_tasks) {
  if (failed_tasks.empty()) {
    return;
  }

  // Generate connection key for tracking retry counts
  std::string connection_key = group_key + "_" + std::to_string(device_idx);

  // Check retry count and determine if we should retry
  bool should_retry = true;
  int current_retry_count = 0;
  {
    std::lock_guard<std::mutex> lock(retry_counts_mutex_);
    // Increment retry count or initialize to 1
    if (task_retry_counts_.find(connection_key) == task_retry_counts_.end()) {
      task_retry_counts_[connection_key] = 1;
      current_retry_count = 1;
    } else {
      task_retry_counts_[connection_key]++;
      current_retry_count = task_retry_counts_[connection_key];
    }

    // Check if maximum retry count exceeded
    if (current_retry_count > MAX_TASK_RETRIES) {
      KLLM_LOG_WARNING << "Max retries exceeded for connection " << connection_key << ". Tasks will be dropped after "
                       << MAX_TASK_RETRIES << " attempts.";
      should_retry = false;
    }
  }

  if (!should_retry) {
    KLLM_LOG_ERROR << "Dropping " << failed_tasks.size() << " tasks for group " << group_key << " device " << device_idx
                   << " after maximum retry attempts.";

    return;
  }

  // Re-queue failed tasks immediately without detached threads
  // This avoids the use-after-free issues that detached threads can cause
  for (const auto& task_key : failed_tasks) {
    task_manager_->PutProcessingBuffer(task_key);
  }
}

std::string TaskDispatcher::MakeConnectionId(const std::string& group_key, int device_idx) const {
  return group_key + "_" + std::to_string(device_idx);
}

std::pair<std::string, int> TaskDispatcher::ParseConnectionId(const std::string& connection_id) {
  auto pos = connection_id.rfind('_');
  if (pos == std::string::npos) {
    throw std::invalid_argument("Invalid connection_id format: " + connection_id);
  }
  std::string group_key = connection_id.substr(0, pos);
  int device_idx = std::stoi(connection_id.substr(pos + 1));
  return {group_key, device_idx};
}

bool TaskDispatcher::IsFirstAttempt(const std::string& conn_key) {
  std::lock_guard<std::mutex> lock(retry_counts_mutex_);
  return task_retry_counts_.find(conn_key) == task_retry_counts_.end();
}

bool TaskDispatcher::CheckConnection(const std::string& group_key, int device_idx) {
  if (!zmq_communicator_) {
    KLLM_LOG_ERROR << "ZMQ communicator is null in CheckConnection";
    return false;
  }

#ifdef ENABLE_CUDA
  if (config_.communication_type == CommunicationType::NCCL && !nccl_communicator_) {
    KLLM_LOG_ERROR << "NCCL communicator is null in CheckConnection";
    return false;
  }
#endif

  std::string connection_id = MakeConnectionId(group_key, device_idx);
  bool is_first_attempt = IsFirstAttempt(connection_id);

  // 对于首次连接尝试，可以等待更长时间
  if (is_first_attempt) {
    KLLM_LOG_INFO << "First connection attempt for " << connection_id << ", waiting for connection to establish...";

    // 尝试等待连接建立，使用配置的超时时间，每5秒检查一次
    const int wait_interval_sec = 5;
    // 将毫秒转换为秒，最小等待30秒，最大3600秒（1小时）
    const int max_wait_sec = config_.connector_waiting_sec;
    int waited_sec = 0;
#ifdef ENABLE_CUDA
    while (config_.communication_type == CommunicationType::NCCL && waited_sec < max_wait_sec &&
           !nccl_communicator_->IsConnectionReady(group_key, device_idx)) {
      KLLM_LOG_INFO << "Waiting for connection to " << connection_id << ", " << waited_sec
                    << " seconds elapsed, will wait up to " << max_wait_sec << " seconds";
      std::this_thread::sleep_for(std::chrono::seconds(wait_interval_sec));
      waited_sec += wait_interval_sec;
    }

#endif
    while (waited_sec < max_wait_sec && !zmq_communicator_->IsConnectionReady(group_key, device_idx)) {
      KLLM_LOG_INFO << "Waiting for connection to " << connection_id << ", " << waited_sec
                    << " seconds elapsed, will wait up to " << max_wait_sec << " seconds";
      std::this_thread::sleep_for(std::chrono::seconds(wait_interval_sec));
      waited_sec += wait_interval_sec;
    }
  }
#ifdef ENABLE_CUDA
  if (config_.communication_type == CommunicationType::NCCL) {
    if (nccl_communicator_->IsConnectionReady(group_key, device_idx) &&
        zmq_communicator_->IsConnectionReady(group_key, device_idx)) {
      KLLM_LOG_INFO << "Connection " << connection_id << " is ready for both ZMQ and NCCL.";
      std::lock_guard<std::mutex> lock(retry_counts_mutex_);
      task_retry_counts_[connection_id] = 1;  // 重置重试计数
      return true;
    } else {
      std::lock_guard<std::mutex> lock(retry_counts_mutex_);
      task_retry_counts_.erase(connection_id);
      KLLM_LOG_ERROR << "Connection " << connection_id << " is not ready for ZMQ or NCCL.";
      return false;  // 连接未准备好
    }
  }
#endif
  if (zmq_communicator_->IsConnectionReady(group_key, device_idx)) {
    KLLM_LOG_INFO << "Connection " << connection_id << " is ready for ZMQ.";
    std::lock_guard<std::mutex> lock(retry_counts_mutex_);
    task_retry_counts_[connection_id] = 1;  // 重置重试计数
    return true;
  } else {
    std::lock_guard<std::mutex> lock(retry_counts_mutex_);
    task_retry_counts_.erase(connection_id);  // 清除重试计数
    KLLM_LOG_ERROR << "Connection " << connection_id << " is not ready for ZMQ or NCCL.";
    return false;  // 连接未准备好
  }
}

// 把零散的各个 task中的Tensor 合并为一个Batch 列表整组发送。
void TaskDispatcher::AddTensorForTask(const std::shared_ptr<TransferTask>& task, const TaskKey& tk,
                                      std::vector<void*>& tensors, std::vector<size_t>& tensor_sizes,
                                      std::vector<DataType>& data_types, bool use_dst_ptr) {
  if (use_dst_ptr) {
    tensors.push_back(task->dst_ptr);
  } else {
    tensors.push_back(task->tensor.src_ptr);
  }

  tensor_sizes.push_back(task->tensor.GetElementNumber() * GetTypeSize(task->tensor.dtype));
  data_types.push_back(task->tensor.dtype);
}

// batch 出队
std::vector<TaskKey> TaskDispatcher::BatchTasks(int batch_size) {
  // 定期清理过期的promise条目
  task_manager_->CleanupExpiredTasks(config_.connector_waiting_sec);

  // Use the new circular buffer batch method - naturally parallel by req_id
  std::vector<TaskKey> raw_tasks = task_manager_->GetProcessingBufferBatch(batch_size);

  if (raw_tasks.empty()) {
    return std::vector<TaskKey>();
  }

  // For decode role, simply return all tasks
  if (config_.group_role != GroupRole::PREFILL) {
    KLLM_LOG_DEBUG << "Taking " << raw_tasks.size() << " tasks from circular processing buffers for decode role";
    return raw_tasks;
  }

  // For prefill role, process tasks by req_id groups in parallel
  std::vector<TaskKey> batch;
  batch.reserve(raw_tasks.size());

  // Group by req_id for parallel processing
  std::unordered_map<uint64_t, std::vector<TaskKey>> grouped_by_req_id;
  for (const auto& task_key : raw_tasks) {
    grouped_by_req_id[task_key.req_id].push_back(task_key);
  }

  // Use adaptive threshold: parallel processing for multiple req_id groups
  if (grouped_by_req_id.size() <= 1) {
    // Serial processing for single req_id group
    for (const auto& task_key : raw_tasks) {
      KLLM_LOG_DEBUG << "Checking pending promises for task_key: " << task_key.ToString();
      if (task_manager_->TryActivatePendingTask(task_key)) {
        batch.push_back(task_key);
      } else {
        task_manager_->AddPrefillPendingTask(task_key);
      }
    }
  } else {
    // Parallel processing for multiple req_id groups
    tbb::concurrent_vector<TaskKey> concurrent_batch;

    // Process each req_id group in parallel
    tbb::parallel_for_each(grouped_by_req_id.begin(), grouped_by_req_id.end(), [&](const auto& req_group) {
      const auto& req_tasks = req_group.second;

      // Process all tasks for this req_id serially (maintain order within req_id)
      for (const auto& task_key : req_tasks) {
        KLLM_LOG_DEBUG << "Checking pending promises for task_key: " << task_key.ToString();
        if (task_manager_->TryActivatePendingTask(task_key)) {
          concurrent_batch.push_back(task_key);
        } else {
          task_manager_->AddPrefillPendingTask(task_key);
        }
      }
    });

    // Convert concurrent_vector to regular vector
    batch.assign(concurrent_batch.begin(), concurrent_batch.end());

    // Sort by timestamp to maintain priority order across req_ids
    std::sort(batch.begin(), batch.end(),
              [](const TaskKey& a, const TaskKey& b) { return a.start_time_us < b.start_time_us; });
  }

  return batch;
}

void TaskDispatcher::PrefillProcessGroupBatches(const std::vector<TaskKey>& batch) {
  auto group_batches = task_manager_->GroupByGroupKeyAndDevice(batch);
  if (group_batches.empty()) {
    return;
  }

  // Check if thread pool is initialized
  if (!send_thread_pool_) {
    throw std::runtime_error("Thread pool is not initialized. Cannot process task batches.");
  }

  // 使用线程池并行处理不同的group_batch，避免单个group_key的连接问题阻塞其他group_key
  std::vector<std::future<void>> futures;
  futures.reserve(group_batches.size());

  // 为每个group批次提交独立的任务到线程池
  for (const auto& group_batch : group_batches) {
    auto future = send_thread_pool_->Submit([this, group_batch]() { HandlePrefillGroupBatch(group_batch); });
    futures.push_back(std::move(future));
  }

  for (auto& future : futures) {
    future.wait();
  }

  // Complete tasks first
  for (auto key : batch) {
    task_manager_->CompleteTask(key);
  }
}

#ifdef ENABLE_CUDA

void TaskDispatcher::SendDataToDecodeWithNccl(const std::string& group_key, int device_idx,
                                              const std::vector<TaskKey>& group_vec) {
  CUDA_CHECK(cudaSetDevice(device_idx));
  BufferPool* buffer_pool = device_pools_[device_idx].get();
  size_t task_keys_bytes = group_vec.size() * sizeof(TaskKey);
  cudaEvent_t copy_done;
  cudaEventCreate(&copy_done);
  auto task_keys_gpu = CopyTaskKeysToDevice(buffer_pool, group_vec, task_keys_bytes, device_idx, copy_done);

  // Check if CopyTaskKeysToDevice failed
  if (task_keys_gpu == nullptr) {
    KLLM_LOG_ERROR << "Failed to copy task keys to device for group_key=" << group_key;
    CUDA_CHECK(cudaEventDestroy(copy_done));
    return;
  }

  // 收集和准备张量数据
  std::vector<void*> tensors;
  std::vector<size_t> tensor_sizes;
  std::vector<DataType> data_types;

  // 遍历所有任务，收集有效的张量
  for (const auto& tk : group_vec) {
    KLLM_LOG_DEBUG << "Step_5: Prefill preparing task_keys and tensors for group_key: " << group_key
                   << " and task_key is: " << tk.ToString()
                   << ", befor start nccl send time cost: " << (ProfileTimer::GetCurrentTimeInUs() - tk.start_time_us);
    auto task = task_manager_->GetTask(tk);
    if (!task) {
      KLLM_LOG_ERROR << "Task not found for key: " << tk.ToString();
      continue;  // 跳过未找到的任务
    }
    if (tk.tensor_size > 0) {
      AddTensorForTask(task, tk, tensors, tensor_sizes, data_types);
    }
  }

  cudaEventSynchronize(copy_done);
  nccl_communicator_->Send(group_key, device_idx, 0, task_keys_gpu->device_ptr, task_keys_bytes, DataType::TYPE_BYTES);
  CUDA_CHECK(cudaEventDestroy(copy_done));
  buffer_pool->put_block(task_keys_gpu);
  KLLM_LOG_DEBUG << "Step_6: Prefill task_keys sent to Decode for group_key=" << group_key
                 << ", and first task_key is:=" << group_vec[0].ToString();

  if (tensors.empty() || tensor_sizes.empty() || data_types.empty()) {
    return;
  }

  std::vector<const void*> const_tensors;
  for (auto ptr : tensors) const_tensors.push_back(ptr);
  nccl_communicator_->SendGroup(group_key, device_idx, 0, const_tensors, tensor_sizes, data_types[0]);

  KLLM_LOG_DEBUG << "Step_8: Prefill task data sent to Decode and task_key[0]= "
                 << (group_vec.empty() ? "" : group_vec[0].ToString()) << " batch size: " << group_vec.size();
  return;
}

BufferBlock* TaskDispatcher::CopyTaskKeysToDevice(BufferPool* buffer_pool, const std::vector<TaskKey>& task_keys,
                                                  size_t task_keys_bytes, int device_idx, cudaEvent_t copy_done) {
  // Check for null buffer_pool
  if (buffer_pool == nullptr) {
    KLLM_LOG_ERROR << "BufferPool is null in CopyTaskKeysToDevice";
    return nullptr;
  }

  // Check for empty task_keys
  if (task_keys.empty()) {
    KLLM_LOG_DEBUG << "Empty task_keys in CopyTaskKeysToDevice";
    return nullptr;
  }

  KLLM_LOG_DEBUG << "copy task_keys to device first task_key: " << task_keys[0].ToString()
                 << ", task_keys size: " << task_keys.size();
  BufferBlock* blk = buffer_pool->get_block();
  cudaStream_t cur_stream = device_streams_[device_idx];
  CUDA_CHECK(cudaEventRecord(copy_done, cur_stream));
  CUDA_CHECK(cudaMemcpyAsync(blk->device_ptr, task_keys.data(), task_keys_bytes, cudaMemcpyHostToDevice, cur_stream));
  return blk;
}

void TaskDispatcher::RecvTaskKeysWithNccl(const std::string& group_key, int device_idx, TaskKey* host_ptr,
                                          size_t bytes) {
  cudaSetDevice(device_idx);
  // Check if device_pools_ is properly initialized for this device
  auto it = device_pools_.find(device_idx);
  if (it == device_pools_.end() || !it->second) {
    throw std::runtime_error("BufferPool not initialized for device " + std::to_string(device_idx));
  }

  auto buffer_pool = it->second.get();
  BufferBlock* blk = buffer_pool->get_block();

  if (blk->capacity < bytes) throw std::runtime_error("Buffer too small");

  Status recv_status = nccl_communicator_->Recv(group_key, device_idx, 0, blk->device_ptr, bytes, DataType::TYPE_BYTES);
  if (!recv_status.OK()) {
    buffer_pool->put_block(blk);
    throw std::runtime_error("NCCL Recv failed");
  }

  // 2. 从 device buffer 拷贝回 host
  cudaEvent_t copy_done;
  cudaStream_t cur_stream = device_streams_[device_idx];
  CUDA_CHECK(cudaEventCreate(&copy_done));
  CUDA_CHECK(cudaMemcpyAsync(host_ptr, blk->device_ptr, bytes, cudaMemcpyDeviceToHost, cur_stream));
  CUDA_CHECK(cudaEventRecord(copy_done, cur_stream));
  CUDA_CHECK(cudaEventSynchronize(copy_done));
  CUDA_CHECK(cudaEventDestroy(copy_done));
  buffer_pool->put_block(blk);
}

void TaskDispatcher::RecvTaskDataWithNccl(const std::string& group_key, int device_idx,
                                          const std::vector<TaskKey>& task_keys) {
  std::vector<void*> recv_ptrs;
  std::vector<size_t> recv_sizes;
  std::vector<DataType> data_types;
  for (const auto& tk : task_keys) {
    KLLM_LOG_DEBUG << "Step_9: Decode preparing to receive tensor for task_key: " << tk.ToString()
                   << ", nccl send task_keys cost: " << (ProfileTimer::GetCurrentTimeInUs() - tk.start_time_us);
    auto task = task_manager_->GetTask(tk);
    if (!task) {
      continue;
    }

    if (tk.tensor_size > 0) {
      AddTensorForTask(task, tk, recv_ptrs, recv_sizes, data_types, true);
    } else {
      std::memcpy(task->dst_ptr, &tk.token, sizeof(int32_t));
    }
  }
  if (!recv_ptrs.empty()) {
    nccl_communicator_->RecvGroup(group_key, device_idx, 0, recv_ptrs, recv_sizes, data_types[0]);
  }
  for (const auto& tk : task_keys) {
    task_manager_->CompleteTask(tk);
    KLLM_LOG_DEBUG << "Step_10: Decode completed task_key: " << tk.ToString()
                   << " nccl send task_keys and tensors cost: "
                   << (ProfileTimer::GetCurrentTimeInUs() - tk.start_time_us);
  }
}
#endif
}  // namespace ksana_llm
