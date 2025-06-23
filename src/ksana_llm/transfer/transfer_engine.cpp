/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/transfer/transfer_engine.h"

#include <cstring>
#include <future>
#include <mutex>

#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

/**
 * @brief 初始化传输引擎
 *
 * @tparam EnvType 环境类型，默认为 Environment
 * @tparam ConnectorType 连接器类型，默认为 Connector
 * @param group_role 节点角色
 */
template <typename EnvType, typename ConnectorType>
void TransferEngine::Initialize(GroupRole group_role) {
  group_role_ = group_role;

  auto env = Singleton<EnvType>::GetInstance();

  // 从环境中获取配置
  env->GetPipelineConfig(pipeline_config_);
  env->GetBlockManagerConfig(block_manager_config_);
  tensor_parallel_size_ = env->GetTensorParallelSize();

  // 获取连接器配置
  ConnectorConfig connector_config;
  env->GetConnectorConfigs(connector_config);

  // 创建连接器实例
  connector_ =
      std::make_shared<ConnectorType>(connector_config, tensor_parallel_size_, pipeline_config_.node_rank, env);

  // 初始化并启动传输连接器
  connector_->Initialize(group_role);

  // 计算派生值
  layer_num_ = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
  block_size_ = block_manager_config_.device_allocator_config.block_size;
  kv_cache_dtype_ = block_manager_config_.device_allocator_config.kv_cache_dtype;

  // 判断是否处于不需要prefill的decode状态
  decode_node_benchmark =
      (std::getenv("DECODE_NODE_BENCHMARK") != nullptr) && (strcmp(std::getenv("DECODE_NODE_BENCHMARK"), "1") == 0);

  KLLM_LOG_DEBUG << "TransferEngine initialized";
}

// 显式实例化默认模板参数的版本
template void TransferEngine::Initialize<Environment, Connector>(GroupRole group_role);
// 显式实例化 TransferConnector 版本，兼容测试代码
template void TransferEngine::Initialize<Environment, TransferConnector>(GroupRole group_role);

/**
 * @brief 为请求添加传输元数据
 *
 * @param request_id 请求ID
 * @param shared_token_num 共享token数量
 * @param gpu_blocks 每个设备的GPU内存块
 */
void TransferEngine::AddTransferMeta(const std::string& kv_comm_group_key, int request_id, size_t shared_token_num,
                                     std::vector<std::vector<void*>>& gpu_blocks) {
  if (request_id < 0) {
    KLLM_LOG_ERROR << "Invalid request_id: " << request_id;
    return;
  }

  auto transfer_meta = std::make_shared<TransferMeta>();
  transfer_meta->shared_token_num = shared_token_num;
  transfer_meta->gpu_blocks = std::move(gpu_blocks);
  transfer_meta->kv_comm_group_key = kv_comm_group_key;

  // 初始化sent_tasks_跟踪矩阵
  const size_t device_num = transfer_meta->gpu_blocks.size();
  const size_t block_num = device_num > 0 ? transfer_meta->gpu_blocks[0].size() : 0;

  // 验证device_num和block_num
  if (device_num == 0 || block_num == 0) {
    KLLM_LOG_WARNING << "Invalid device_num or block_num in AddTransferMeta: " << device_num << ", " << block_num;
    return;
  }

  // 预分配适当维度的sent_tasks_矩阵
  transfer_meta->sent_tasks_.resize(device_num);
  for (size_t d = 0; d < device_num; ++d) {
    transfer_meta->sent_tasks_[d].resize(block_num);
    for (size_t b = 0; b < block_num; ++b) {
      transfer_meta->sent_tasks_[d][b].resize(layer_num_, false);
    }
  }
  // 对于decode节点，创建传输任务
  if (group_role_ == GroupRole::DECODE) {
    CreateTransferTasksForDecodeNode(request_id, transfer_meta, device_num, block_num);
  } else {
    transfer_meta->first_token = 0;
  }

  // 将元数据添加到映射
  {
    std::lock_guard<std::mutex> lock(meta_map_mutex_);
    meta_map_[request_id] = std::move(transfer_meta);
  }

  KLLM_LOG_DEBUG << "TransferMeta added for request ID: " << request_id << ", shared_token_num: " << shared_token_num
                 << ", gpu_blocks size: " << gpu_blocks.size() << ", kv_comm_group_key: " << kv_comm_group_key;
}

/**
 * @brief 检查请求的所有接收操作是否完成
 *
 * @param request_id 请求ID
 * @return int 如果完成则返回first_token值，否则返回-1
 */
int TransferEngine::IsRecvDone(int request_id) {
  std::shared_ptr<TransferMeta> meta = GetTransferMeta(request_id);
  if (!meta) {
    KLLM_LOG_DEBUG << "TransferTask not found, request id:" << request_id;
    return -1;
  }

  // 处理已完成的任务
  {
    std::lock_guard<std::mutex> lock(meta->mutex_);

    // 将已完成的任务从transfer_tasks_deque_移动到finished_tasks_deque_
    auto it = meta->transfer_tasks_deque_.begin();
    while (it != meta->transfer_tasks_deque_.end()) {
      auto& task = *it;
      if (task && task->is_completed) {
        meta->finished_tasks_deque_.push_back(std::move(task));
        it = meta->transfer_tasks_deque_.erase(it);
      } else {
        ++it;
      }
    }

    // 检查gpu_blocks是否为空
    if (meta->gpu_blocks.empty()) {
      KLLM_LOG_WARNING << "Empty gpu_blocks in IsDone for request id: " << request_id;
      return -1;
    }

    // 计算预期的任务数量
    const int block_num = meta->gpu_blocks[0].size();
    const size_t expected_tasks = block_num * layer_num_ * tensor_parallel_size_;

    KLLM_LOG_DEBUG << "TransferTask IsDone? request id:" << request_id
                   << " finished:" << meta->finished_tasks_deque_.size() << " expected:" << expected_tasks;

    if (decode_node_benchmark) {
      meta->first_token = 10;  // 模拟从prefill获取的首token
    }

    // 检查所有任务是否完成（管道并行异构模式）
    if (block_num > 0 && meta->finished_tasks_deque_.size() == expected_tasks && meta->first_token != -1) {
      return meta->first_token;
    }
  }

  return -1;
}

/**
 * @brief 检查请求的所有发送操作是否完成
 *
 * @param request_id 请求ID
 * @return true 如果所有发送操作完成则返回true，否则返回false
 */
bool TransferEngine::IsSendDone(int request_id) { return IsRecvDone(request_id) != -1; }

/**
 * @brief 为特定设备和层发送传输任务
 *
 * @param device_idx 设备索引
 * @param layer_idx 层索引
 */
void TransferEngine::Send(int device_idx, int layer_idx) {
  if (group_role_ != GroupRole::PREFILL) {
    return;
  }

  // 验证layer_idx参数
  if (!ValidateLayerIndex(layer_idx)) {
    KLLM_LOG_WARNING << "Layer index out of range. " << layer_idx << " not in [" << pipeline_config_.lower_layer_idx
                     << ", " << pipeline_config_.upper_layer_idx << "]";
    return;
  }

  const int layer_offset = CalculateLayerOffset(layer_idx);
  const size_t element_size = block_size_ / layer_num_;

  // 处理所有请求
  std::lock_guard<std::mutex> meta_lock(meta_map_mutex_);
  for (auto& meta_pair : meta_map_) {
    const int request_id = meta_pair.first;
    std::shared_ptr<TransferMeta> meta = meta_pair.second;

    if (!meta) {
      continue;
    }

    // 验证元数据
    if (meta->gpu_blocks.empty()) {
      KLLM_LOG_WARNING << "Empty gpu_blocks in Send for request id: " << request_id;
      continue;
    }

    if (device_idx >= meta->gpu_blocks.size()) {
      KLLM_LOG_DEBUG << "Invalid device_idx: " << device_idx << ", max: " << meta->gpu_blocks.size() - 1;
      continue;
    }

    if (meta->gpu_blocks[0].empty()) {
      KLLM_LOG_WARNING << "Empty gpu_blocks[0] in Send for request id: " << request_id;
      continue;
    }

    // 处理此设备和层的所有块
    for (size_t block_idx = 0; block_idx < meta->gpu_blocks[0].size(); ++block_idx) {
      // 检查是否已发送
      bool already_sent = false;
      {
        std::lock_guard<std::mutex> lock(meta->mutex_);
        if (device_idx < meta->sent_tasks_.size() && block_idx < meta->sent_tasks_[device_idx].size() &&
            layer_offset < meta->sent_tasks_[device_idx][block_idx].size()) {
          already_sent = meta->sent_tasks_[device_idx][block_idx][layer_offset];
        }
      }

      if (already_sent) {
        continue;
      }

      // 创建传输任务
      auto task = std::make_shared<TransferTask>();
      task->req_id = request_id;
      task->addr = meta->kv_comm_group_key;  // 设置通信组键
      task->tensor.block_idx = block_idx;
      task->tensor.layer_idx = layer_idx;
      task->tensor.device_idx = device_idx;

      // 设置张量属性
      task->tensor.shape = {block_size_ / layer_num_ / GetTypeSize(kv_cache_dtype_), 1};
      task->tensor.dtype = kv_cache_dtype_;

      // 如果block_idx有效，设置源指针
      if (block_idx < meta->gpu_blocks[device_idx].size()) {
        task->tensor.src_ptr =
            static_cast<char*>(meta->gpu_blocks[device_idx][block_idx]) + layer_offset * element_size;
      } else {
        KLLM_LOG_WARNING << "Invalid block_idx: " << block_idx << " for device: " << device_idx;
        continue;
      }

      // 将任务标记为已发送并添加到传输任务
      {
        std::lock_guard<std::mutex> lock(meta->mutex_);
        meta->sent_tasks_[device_idx][block_idx][layer_offset] = true;
        meta->transfer_tasks_deque_.push_back(task);
      }

      // 将任务推送到连接器队列
      connector_->PushTask(task);
      KLLM_LOG_DEBUG << "Sent transfer task for request " << request_id << ", device: " << device_idx
                     << ", layer: " << layer_idx << ", block: " << block_idx;
    }
  }
}

/**
 * @brief 为多个请求发送token传输任务
 *
 * @param reqs_tokens 请求ID和token对的向量
 */
void TransferEngine::Send(std::vector<std::tuple<std::string, int, int>>& reqs_tokens) {
  if (group_role_ != GroupRole::PREFILL) {
    return;
  }

  if (reqs_tokens.empty()) {
    KLLM_LOG_DEBUG << "No tokens to send";
    return;
  }

  // 处理所有请求-token对
  for (const auto& [kv_comm_group_key, request_id, token] : reqs_tokens) {
    if (request_id < 0) {
      KLLM_LOG_WARNING << "Invalid request_id: " << request_id;
      continue;
    }

    // 创建传输任务
    auto task = std::make_shared<TransferTask>();
    task->req_id = request_id;
    task->addr = kv_comm_group_key;
    task->token = token;

    // 将任务推送到连接器队列
    connector_->PushTask(task);
    KLLM_LOG_DEBUG << "Sent token transfer task for request " << request_id << ", token: " << token;
  }
}

/**
 * @brief 为decode节点创建传输任务
 *
 * @param request_id 请求ID
 * @param transfer_meta 传输元数据的共享指针
 * @param device_num 设备数量
 * @param block_num 块数量
 */
void TransferEngine::CreateTransferTasksForDecodeNode(int request_id, std::shared_ptr<TransferMeta>& transfer_meta,
                                                      size_t device_num, size_t block_num) {
  KLLM_LOG_DEBUG << "Creating transfer tasks for decode node, request_id: " << request_id
                 << ", device_num: " << device_num << ", block_num: " << block_num;
  const size_t element_size = block_size_ / layer_num_;

  // 为每个设备、块和层创建任务
  for (size_t device_idx = 0; device_idx < device_num; ++device_idx) {
    for (size_t block_idx = 0; block_idx < block_num; ++block_idx) {
      for (size_t layer_idx = pipeline_config_.lower_layer_idx; layer_idx <= pipeline_config_.upper_layer_idx;
           ++layer_idx) {
        auto task = std::make_shared<TransferTask>();
        task->req_id = request_id;
        task->addr = transfer_meta->kv_comm_group_key;
        task->tensor.block_idx = block_idx;
        task->tensor.layer_idx = layer_idx;
        task->tensor.device_idx = device_idx;

        // 设置张量形状和数据类型
        task->tensor.shape = {block_size_ / layer_num_ / GetTypeSize(kv_cache_dtype_), 1};
        task->tensor.dtype = kv_cache_dtype_;

        // 如果block_idx有效，设置目标指针
        if (block_idx < transfer_meta->gpu_blocks[device_idx].size()) {
          const int layer_offset = CalculateLayerOffset(layer_idx);
          task->dst_ptr =
              static_cast<char*>(transfer_meta->gpu_blocks[device_idx][block_idx]) + layer_offset * element_size;
        } else {
          KLLM_LOG_WARNING << "Invalid block_idx: " << block_idx << " for device: " << device_idx;
          continue;
        }

        // 将任务添加到连接器和元数据
        connector_->PushTask(task);

        {
          std::lock_guard<std::mutex> lock(transfer_meta->mutex_);
          transfer_meta->transfer_tasks_deque_.push_back(std::move(task));
        }
      }
    }
  }

  // 为第一个token创建任务
  auto token_task = std::make_shared<TransferTask>();
  token_task->req_id = request_id;
  token_task->dst_ptr = &transfer_meta->first_token;
  token_task->addr = transfer_meta->kv_comm_group_key;

  connector_->PushTask(token_task);
  KLLM_LOG_DEBUG << "Creating transfer tasks for decode node, request_id: " << request_id
                 << ", device_num: " << device_num << ", block_num: " << block_num;
}

}  // namespace ksana_llm
