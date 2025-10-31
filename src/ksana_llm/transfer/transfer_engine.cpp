/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/transfer/transfer_engine.h"

#include <cstring>
#include <mutex>

#include "ksana_llm/connector/device_info_manager.h"
#include "ksana_llm/profiler/reporter.h"
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
  std::call_once(init_once_flag_, [this, group_role]() {
    group_role_ = group_role;

    auto env = Singleton<EnvType>::GetInstance();

    // 从环境中获取配置
    env->GetPipelineConfig(pipeline_config_);
    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    RuntimeConfig runtime_config;
    env->GetRuntimeConfig(runtime_config);
    tensor_parallel_size_ = runtime_config.parallel_basic_config.tensor_parallel_size;
    attn_data_parallel_size_ = runtime_config.parallel_basic_config.attn_data_parallel_size;
    // 获取连接器配置
    ConnectorConfig connector_config;
    env->GetConnectorConfigs(connector_config);

    // 创建连接器单例
    connector_ = ConnectorType::GetInstance(connector_config, tensor_parallel_size_, pipeline_config_.node_rank, env);

    // 初始化并启动传输连接器
    connector_->Initialize(group_role, device_info_manager_);

    // 计算派生值
    int common_layer_num = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
    int mtp_layer_num = 0;
    if (runtime_config.enable_mtp_module) {
      mtp_layer_num = pipeline_config_.upper_nextn_layer_idx - pipeline_config_.lower_nextn_layer_idx + 1;
    }
    KLLM_LOG_DEBUG << "common_layer_num: " << common_layer_num << ", mtp_layer_num: " << mtp_layer_num;
    layer_num_ = common_layer_num + mtp_layer_num;
    block_size_ = block_manager_config.device_allocator_config.block_size;
    transfer_layer_chunk_size_ = env->GetTransferLayerChunkSize();
    // 判断是否处于不需要prefill的decode状态
    decode_node_benchmark =
        (std::getenv("DECODE_NODE_BENCHMARK") != nullptr) && (strcmp(std::getenv("DECODE_NODE_BENCHMARK"), "1") == 0);

    KLLM_LOG_DEBUG << "TransferEngine initialized";
  });
}

// 显式实例化默认模板参数的版本
template void TransferEngine::Initialize<Environment, Connector>(GroupRole group_role);
// 显式实例化 TransferConnector 版本，兼容测试代码
template void TransferEngine::Initialize<Environment, TransferConnector>(GroupRole group_role);

/**
 * @brief 为请求添加传输元数据
 *
 * @param request_id 请求ID
 * @param shared_block_num 共享block数量
 * @param gpu_blocks 每个设备的GPU内存块
 */
void TransferEngine::AddTransferMeta(const std::string& kv_comm_group_key, int request_id, size_t shared_block_num,
                                     std::vector<std::vector<void*>>& gpu_blocks,
                                     std::vector<int>& kv_occupied_devices) {
  if (group_role_ == GroupRole::PREFILL && request_id == -1) {
    KLLM_LOG_WARNING << "Invalid request_id: " << request_id << " to avoid warmup failed";
    return;
  }

  if (request_id < 0) {
    KLLM_LOG_ERROR << "Invalid request_id: " << request_id;
    return;
  }

  auto transfer_meta = std::make_shared<TransferMeta>();
  transfer_meta->shared_block_num = shared_block_num;
  transfer_meta->kv_ranks_in_node = kv_occupied_devices;
  transfer_meta->gpu_blocks = std::move(gpu_blocks);
  transfer_meta->kv_comm_group_key = kv_comm_group_key;

  if (group_role_ == GroupRole::DECODE &&
      !device_info_manager_->FindAndInsert(kv_comm_group_key, attn_data_parallel_size_, tensor_parallel_size_)) {
    connector_->SendConfigToPrefill(kv_comm_group_key, attn_data_parallel_size_, tensor_parallel_size_);
    KLLM_LOG_DEBUG << "SendConfigToPrefill for kv_comm_group_key: " << kv_comm_group_key
                   << ", ADP: " << attn_data_parallel_size_ << ", total_dev: " << tensor_parallel_size_;
  }

  // 初始化sent_tasks_跟踪矩阵
  const size_t device_num = transfer_meta->gpu_blocks.size();
  const size_t block_num = device_num > 0 ? transfer_meta->gpu_blocks[0].size() : 0;
  // 验证device_num和block_num
  if (device_num == 0 || block_num == 0) {
    KLLM_LOG_WARNING << "Invalid device_num or block_num in AddTransferMeta: " << device_num << ", " << block_num;
    return;
  }

  // 验证block_num和shared_block_num
  if (block_num < shared_block_num) {
    KLLM_LOG_ERROR << "shared_block_num larger than block_num in AddTransferMeta: " << shared_block_num << ", "
                   << block_num;
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
    CreateTransferTasksForDecodeNode(request_id, transfer_meta, device_num, block_num, shared_block_num);
  } else {
    transfer_meta->first_tokens = std::vector<int>(MAX_TRANSFER_TOKENS, 0);
  }

  KLLM_LOG_DEBUG << "TransferMeta added for request ID: " << request_id << ", shared_block_num: " << shared_block_num
                 << ", gpu_blocks size: " << transfer_meta->gpu_blocks.size()
                 << ", kv_comm_group_key: " << kv_comm_group_key;

  // 将元数据添加到映射
  {
    std::lock_guard<std::mutex> lock(meta_map_mutex_);
    meta_map_[request_id] = std::move(transfer_meta);
  }
}

/**
 * @brief 检查请求的所有接收操作是否完成
 *
 * @param request_id 请求ID
 * @return std::pair<int, int> 如果完成则返回first_tokens值，否则返回{-1, -1}
 */
std::vector<int> TransferEngine::IsRecvDone(int request_id) {
  // 预热请求直接完成
  if (request_id == -1) {
    return std::vector<int>(MAX_TRANSFER_TOKENS, 0);
  }
  std::shared_ptr<TransferMeta> meta = GetTransferMeta(request_id);
  if (!meta) {
    KLLM_LOG_DEBUG << "TransferTask not found, request id:" << request_id;
    return std::vector<int>(MAX_TRANSFER_TOKENS, -1);
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
      return std::vector<int>(MAX_TRANSFER_TOKENS, -1);
    }

    // 计算预期的任务数量 - 现在按chunk传输，任务数量会减少
    const int device_num = meta->gpu_blocks.size();
    const int block_num = meta->gpu_blocks[0].size();
    const size_t chunks_per_device =
        (layer_num_ + transfer_layer_chunk_size_ - 1) / transfer_layer_chunk_size_;  // 向上取整

    size_t expected_tasks = block_num * chunks_per_device * device_num;
    if (group_role_ == GroupRole::PREFILL) {
      std::pair<int, int> decode_dev_config;
      if (!device_info_manager_->TryGet(meta->kv_comm_group_key, decode_dev_config)) {
        device_info_manager_->WaitFor(meta->kv_comm_group_key, decode_dev_config);
      }
      KLLM_LOG_DEBUG << "Wait for config from Decode for kv_comm_group_key: " << meta->kv_comm_group_key
                     << ", ADP: " << decode_dev_config.first << ", total_dev: " << decode_dev_config.second;
      int decode_dp_num = decode_dev_config.first;
      int deocode_device_num = decode_dev_config.second;

      expected_tasks = block_num * chunks_per_device * (deocode_device_num / decode_dp_num);
    }
    KLLM_LOG_DEBUG << "TransferTask IsDone? request id:" << request_id
                   << " finished:" << meta->finished_tasks_deque_.size() << " expected:" << expected_tasks
                   << " (block_num:" << block_num << ", chunks_per_device:" << chunks_per_device
                   << ", device_num:" << device_num << ")";

    if (decode_node_benchmark) {
      meta->first_tokens = std::vector<int>(MAX_TRANSFER_TOKENS, 10);  // 模拟从prefill获取的首token
    }

    // 检查所有任务是否完成（管道并行异构模式）
    if (block_num > 0 && meta->finished_tasks_deque_.size() == expected_tasks && meta->first_tokens[0] != -1) {
      REPORT_METRIC("transfer_tasks_per_request", expected_tasks);
      return meta->first_tokens;
    }
  }

  return std::vector<int>(MAX_TRANSFER_TOKENS, -1);
}

/**
 * @brief 检查请求的所有发送操作是否完成
 *
 * @param request_id 请求ID
 * @return true 如果所有发送操作完成则返回true，否则返回false
 */
bool TransferEngine::IsSendDone(int request_id) {
  return IsRecvDone(request_id) != std::vector<int>(MAX_TRANSFER_TOKENS, -1);
}

/**
 * @brief 为特定设备和层发送传输任务
 *
 * @param device_idx 设备索引
 * @param layer_idx 层索引
 */
void TransferEngine::Send(int device_idx, int layer_idx) {
  int device_dp_offset = device_idx % (static_cast<int>(tensor_parallel_size_ / attn_data_parallel_size_));
  KLLM_LOG_DEBUG << "TransferEngine Send called for device_idx: " << device_idx
                 << ", device_dp_offset: " << device_dp_offset << ", layer_idx: " << layer_idx;

  if (group_role_ != GroupRole::PREFILL) {
    return;
  }

  int prefill_dp_num = attn_data_parallel_size_;
  int prefill_device_num = tensor_parallel_size_;

  // 验证layer_idx参数
  if (!ValidateLayerIndex(layer_idx)) {
    KLLM_LOG_WARNING << "Layer index out of range. " << layer_idx << " not in [" << pipeline_config_.lower_layer_idx
                     << ", " << pipeline_config_.upper_layer_idx << "]";
    return;
  }

  const int layer_offset = CalculateLayerOffset(layer_idx);

  // 计算chunk传输：只有chunk的最后一层或模型的最后一层才触发传输
  const int chunk_start_layer_offset = (layer_offset / transfer_layer_chunk_size_) * transfer_layer_chunk_size_;
  const size_t actual_chunk_size =
      std::min(transfer_layer_chunk_size_, static_cast<size_t>(layer_num_ - chunk_start_layer_offset));
  const int chunk_end_layer_offset = chunk_start_layer_offset + actual_chunk_size - 1;
  const bool is_model_last_layer = (layer_offset == layer_num_ - 1);

  if (layer_offset != chunk_end_layer_offset && !is_model_last_layer) {
    // 不是chunk的最后一层，也不是模型的最后一层，跳过传输
    KLLM_LOG_DEBUG << "Skipping layer " << layer_idx << " (offset " << layer_offset
                   << "), not chunk end (chunk end offset: " << chunk_end_layer_offset << ") and not model last layer";
    return;
  }
  const size_t element_size = block_size_ / layer_num_;
  const size_t chunk_element_size = element_size * actual_chunk_size;

  // 处理所有请求
  std::lock_guard<std::mutex> meta_lock(meta_map_mutex_);
  for (auto& meta_pair : meta_map_) {
    const int request_id = meta_pair.first;
    KLLM_LOG_DEBUG << " Send Processing request id: " << request_id << " on device_idx: " << device_idx;
    std::shared_ptr<TransferMeta> meta = meta_pair.second;

    if (!meta) {
      continue;
    }
    // 避免构建的task的设备号不对应
    if (std::find(meta->kv_ranks_in_node.begin(), meta->kv_ranks_in_node.end(), device_idx) ==
        meta->kv_ranks_in_node.end()) {
      // TODO(winminkong): 后续优化为hash查找
      KLLM_LOG_DEBUG << "Device " << device_idx << " not used in request " << request_id;
      continue;
    }

    // 验证元数据
    if (meta->gpu_blocks.empty()) {
      KLLM_LOG_WARNING << "Empty gpu_blocks in Send for request id: " << request_id;
      continue;
    }

    if (device_dp_offset >= static_cast<int>(meta->gpu_blocks.size())) {
      KLLM_LOG_DEBUG << "Invalid device_dp_offset: " << device_dp_offset << ", max: " << meta->gpu_blocks.size() - 1;
      continue;
    }

    if (meta->gpu_blocks[0].empty()) {
      KLLM_LOG_WARNING << "Empty gpu_blocks[0] in Send for request id: " << request_id;
      continue;
    }

    std::pair<int, int> decode_dev_config;
    if (!device_info_manager_->Find(meta->kv_comm_group_key, decode_dev_config)) {
      device_info_manager_->WaitFor(meta->kv_comm_group_key, decode_dev_config);
      KLLM_LOG_DEBUG << "WaitForConfigFromDecode for kv_comm_group_key: " << meta->kv_comm_group_key
                     << ", ADP: " << decode_dev_config.first << ", total_dev: " << decode_dev_config.second;
    }
    int decode_dp_num = decode_dev_config.first;
    int deocode_device_num = decode_dev_config.second;
    // 判断P端的taskkey是否复制或跳过
    int dp_ratio = 1;
    if ((deocode_device_num / decode_dp_num) < (prefill_device_num / prefill_dp_num)) {
      // 跳过某些task的创建
      if (device_dp_offset >= (deocode_device_num / decode_dp_num)) {
        // TODO(winminkong): 优化某个设备通信负载高,某个设备没有通信负载的问题
        KLLM_LOG_DEBUG << "Skip sending for device_dp_offset: " << device_dp_offset << " on device_idx: " << device_idx;
        continue;
      }
    } else if ((deocode_device_num / decode_dp_num) > (prefill_device_num / prefill_dp_num)) {
      dp_ratio = (deocode_device_num / decode_dp_num) / (prefill_device_num / prefill_dp_num);
    }
    KLLM_LOG_DEBUG << "dp_ratio: " << dp_ratio;
    for (int cp_idx = 0; cp_idx < dp_ratio; ++cp_idx) {
      // 处理此设备和层的所有块
      for (size_t block_idx = 0; block_idx < meta->gpu_blocks[0].size(); ++block_idx) {
        // 检查是否已发送
        bool already_sent = false;
        {
          std::lock_guard<std::mutex> lock(meta->mutex_);
          if (device_dp_offset < meta->sent_tasks_.size() && block_idx < meta->sent_tasks_[device_dp_offset].size() &&
              layer_offset < meta->sent_tasks_[device_dp_offset][block_idx].size()) {
            already_sent = meta->sent_tasks_[device_dp_offset][block_idx][layer_offset];
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
        task->tensor.layer_idx = layer_idx;  // 保持原始layer_idx用于标识
        task->tensor.hash_device_id = device_dp_offset * dp_ratio + cp_idx;
        // 额外附带信息
        task->prefill_device_id = device_idx;
        task->prefill_device_offset = device_dp_offset;

        // 设置张量属性 - 现在传输多层数据
        task->tensor.shape = {static_cast<int64_t>(chunk_element_size), 1};
        task->tensor.dtype = TYPE_UINT8;

        // 如果block_idx有效，设置源指针 - 从chunk起始层开始
        if (block_idx < meta->gpu_blocks[device_dp_offset].size()) {
          task->tensor.src_ptr = static_cast<char*>(meta->gpu_blocks[device_dp_offset][block_idx]) +
                                 chunk_start_layer_offset * element_size;
        } else {
          KLLM_LOG_WARNING << "Invalid block_idx: " << block_idx << " for device: " << device_idx;
          continue;
        }

        // 将chunk中的所有层标记为已发送
        {
          std::lock_guard<std::mutex> lock(meta->mutex_);
          for (size_t i = 0; i < actual_chunk_size; ++i) {
            const int chunk_layer_offset = chunk_start_layer_offset + i;
            if (chunk_layer_offset < layer_num_ && (cp_idx == dp_ratio - 1)) {
              meta->sent_tasks_[device_dp_offset][block_idx][chunk_layer_offset] = true;
            }
          }
          meta->transfer_tasks_deque_.push_back(task);
        }

        // 将任务推送到连接器队列
        connector_->PushTask(task);
        KLLM_LOG_DEBUG << "Sent chunk transfer task for request " << request_id << ", device: " << device_idx
                       << ", trigger layer: " << layer_idx << " (offset " << layer_offset << ")"
                       << ", chunk range: [" << chunk_start_layer_offset << "-" << chunk_end_layer_offset << "]"
                       << ", chunk size: " << actual_chunk_size << ", block: " << block_idx
                       << ", is_model_last_layer: " << is_model_last_layer;
      }
    }
  }
}

/**
 * @brief 为多个请求发送token传输任务
 *
 * @param reqs_tokens 请求ID和token对的向量
 */
void TransferEngine::Send(std::vector<std::tuple<std::string, int, std::vector<int>>>& reqs_tokens) {
  if (group_role_ != GroupRole::PREFILL) {
    return;
  }

  if (reqs_tokens.empty()) {
    KLLM_LOG_DEBUG << "No tokens to send";
    return;
  }

  // 处理所有请求-token对
  for (const auto& [kv_comm_group_key, request_id, tokens] : reqs_tokens) {
    if (request_id < 0) {
      KLLM_LOG_WARNING << "Invalid request_id: " << request_id;
      continue;
    }

    // 创建传输任务
    auto task = std::make_shared<TransferTask>();
    task->req_id = request_id;
    task->addr = kv_comm_group_key;
    task->tokens = tokens;

    // 额外附带信息
    task->prefill_device_id = 0;
    task->prefill_device_offset = 0;

    // 将任务推送到连接器队列
    connector_->PushTask(task);
    KLLM_LOG_DEBUG << "Sent token transfer task for request " << request_id << ", gen tokens: " << tokens[0]
                   << ", draft tokens: " << tokens[1];
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
                                                      size_t device_num, size_t block_num, size_t shared_block_num) {
  KLLM_LOG_DEBUG << "Creating transfer tasks for decode node, request_id: " << request_id
                 << ", device_num: " << device_num << ", block_num: " << block_num;
  const size_t element_size = block_size_ / layer_num_;

  // 为每个设备、块和chunk创建任务
  for (size_t device_idx = 0; device_idx < device_num; ++device_idx) {
    for (size_t block_idx = 0; block_idx < block_num; ++block_idx) {
      bool is_skipped = block_idx < shared_block_num;
      // 按chunk处理层 - 使用layer_offset而不是layer_idx来确保正确处理边界
      for (int chunk_start_offset = 0; chunk_start_offset < layer_num_;
           chunk_start_offset += transfer_layer_chunk_size_) {
        const size_t layer_idx = pipeline_config_.lower_layer_idx + chunk_start_offset;

        // 计算实际要接收的层数（考虑边界情况）
        // 例如：layer_num_=10, transfer_layer_chunk_size_=3, chunk_start_offset=9
        // 则 actual_chunk_size = min(3, 10-9) = min(3, 1) = 1
        const size_t actual_chunk_size =
            std::min(transfer_layer_chunk_size_, static_cast<size_t>(layer_num_ - chunk_start_offset));
        const size_t chunk_element_size = element_size * actual_chunk_size;

        auto task = std::make_shared<TransferTask>();
        task->req_id = request_id;
        task->is_skipped_task = is_skipped;
        task->addr = transfer_meta->kv_comm_group_key;
        task->tensor.block_idx = block_idx;
        task->tensor.layer_idx = layer_idx + actual_chunk_size - 1;  // chunk的最后一层layer_idx用于标识
        task->tensor.hash_device_id = device_idx;
        // 额外附带信息
        task->decode_device_id = transfer_meta->kv_ranks_in_node[device_idx];
        task->decode_device_offset = device_idx;

        // 设置张量形状和数据类型 - 现在接收多层数据
        task->tensor.shape = {static_cast<int64_t>(chunk_element_size), 1};
        task->tensor.dtype = TYPE_UINT8;

        // 如果block_idx有效，设置目标指针 - 从chunk起始层开始
        if (block_idx < transfer_meta->gpu_blocks[device_idx].size()) {
          task->dst_ptr =
              static_cast<char*>(transfer_meta->gpu_blocks[device_idx][block_idx]) + chunk_start_offset * element_size;
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

        KLLM_LOG_DEBUG << "Created chunk receive task for request " << request_id << ", device: " << device_idx
                       << ", chunk start layer: " << layer_idx << ", chunk size: " << actual_chunk_size
                       << ", block: " << block_idx << ", chunk_start_offset: " << chunk_start_offset;
      }
    }
  }

  // 计算预期的任务数量
  const size_t chunks_per_device =
      (layer_num_ + transfer_layer_chunk_size_ - 1) / transfer_layer_chunk_size_;  // 向上取整
  const size_t total_task_num = block_num * chunks_per_device * device_num;
  const size_t skipped_task_num = shared_block_num * chunks_per_device * device_num;
  KLLM_LOG_DEBUG << "Created block transfer tasks for decode node, request id:" << request_id
                 << " total_task_num:" << total_task_num << " among which " << skipped_task_num
                 << " actually do not transfer data due to hit prefix cache. "
                 << " (device_num:" << device_num << ", chunks_per_device:" << chunks_per_device
                 << ", block_num:" << block_num << ", shared_block_num:" << shared_block_num << ")";

  // 为第一个token创建任务
  auto token_task = std::make_shared<TransferTask>();
  token_task->req_id = request_id;
  token_task->dst_ptr = transfer_meta->first_tokens.data();
  token_task->addr = transfer_meta->kv_comm_group_key;

  // 额外附带信息
  token_task->decode_device_id = 0;
  token_task->decode_device_offset = 0;

  connector_->PushTask(token_task);
  KLLM_LOG_DEBUG << "Creating transfer tasks for decode node, request_id: " << request_id
                 << ", device_num: " << device_num << ", block_num: " << block_num;
}

}  // namespace ksana_llm
