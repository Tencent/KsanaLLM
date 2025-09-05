/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "ksana_llm/connector/connector.h"
#include "ksana_llm/transfer/transfer_types.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

/**
 * @brief 传输元数据结构，存储与传输相关的所有信息
 *
 * 该结构包含了传输过程中需要的所有元数据，包括共享token数量、
 * 第一个token、GPU内存块信息、传输任务列表等。
 */
struct TransferMeta {
  size_t shared_block_num = 0;  // 共享block数量
  std::vector<int> first_tokens =
      std::vector<int>(MAX_TRANSFER_TOKENS, -1);  // Prefill阶段所有的gen token和draft token的值，-1表示尚未接收到

  // 节点内KV存储使用的Rank号
  std::vector<int> kv_ranks_in_node;

  // 多个设备上的GPU物理内存块信息 [device_idx][block_idx]
  std::vector<std::vector<void*>> gpu_blocks;

  // 当前正在处理的传输任务列表
  std::deque<std::shared_ptr<TransferTask>> transfer_tasks_deque_;

  // 已完成的传输任务列表
  std::deque<std::shared_ptr<TransferTask>> finished_tasks_deque_;

  // 标记是否已发起传输 [device_idx][block_idx][layer_idx]
  std::vector<std::vector<std::vector<bool>>> sent_tasks_;

  std::string kv_comm_group_key;  // KV通信组键

  // 保护元数据访问的互斥锁
  std::mutex mutex_;
};

// 下游连接器接口
class TransferConnector : public Connector {
 public:
  TransferConnector(const ConnectorConfig& config, size_t tp_size, size_t node_rank, std::shared_ptr<Environment> env)
      : Connector(config, tp_size, node_rank, env) {}

  Status Initialize(GroupRole group_role, std::shared_ptr<DeviceInfoManager> device_info_manager) override {
    return Status();
  }

  // 启动传输任务处理线程
  void Start() override {}

  // 推送任务到任务队列
  void PushTask(const std::shared_ptr<TransferTask>& task) override { task->is_completed = true; }
};

/**
 * @brief 传输引擎类，负责管理和协调数据传输任务
 *
 * TransferEngine负责在多设备、多层之间协调数据传输，
 * 支持decode节点和prefill节点之间的数据交换。
 */
class TransferEngine {
 public:
  /**
   * @brief 获取TransferEngine单例实例
   * @return 返回TransferEngine的共享指针
   */
  static std::shared_ptr<TransferEngine> GetInstance() { return Singleton<TransferEngine>::GetInstance(); }

  /**
   * @brief 初始化传输引擎
   * @tparam EnvType 环境类型，默认为 Environment
   * @tparam ConnectorType 连接器类型，默认为 Connector
   * @param group_role 节点角色
   */
  template <typename EnvType = Environment, typename ConnectorType = Connector>
  void Initialize(GroupRole group_role);

  /**
   * @brief 添加传输元数据
   * @param request_id 请求ID
   * @param shared_block_num 共享block数量
   * @param gpu_blocks GPU内存块信息
   */
  void AddTransferMeta(const std::string& kv_comm_group_key, int request_id, size_t shared_block_num,
                       std::vector<std::vector<void*>>& gpu_blocks, std::vector<int>& kv_occupied_devices);

  /**
   * @brief 检查指定请求的发送操作是否完成
   * @param request_id 请求ID
   * @return bool 如果所有发送操作完成则返回true，否则返回false
   */
  bool IsSendDone(int request_id);

  /**
   * @brief 检查指定请求的接收操作是否完成
   * @param request_id 请求ID
   * @return std::vector<int> 如果完成则返回first_tokens值，否则返回值全为-1的向量
   */
  std::vector<int> IsRecvDone(int request_id);

  /**
   * @brief 在特定设备和层上发送传输任务
   * @param device_idx 设备索引
   * @param layer_idx 层索引
   */
  void Send(int device_idx, int layer_idx);

  /**
   * @brief 发送多个请求的token传输任务
   * @param reqs_tokens 请求ID和token对的向量
   */
  virtual void Send(std::vector<std::tuple<std::string, int, std::vector<int>>>& reqs_tokens);

  /**
   * @brief 获取指定请求的传输元数据
   * @param request_id 请求ID
   * @return std::shared_ptr<TransferMeta> 传输元数据的共享指针，如果不存在则返回nullptr
   */
  std::shared_ptr<TransferMeta> GetTransferMeta(int request_id) {
    std::lock_guard<std::mutex> lock(meta_map_mutex_);
    auto it = meta_map_.find(request_id);
    if (it != meta_map_.end()) {
      return it->second;
    }
    return nullptr;
  }

  /**
   * @brief 清理指定请求的传输元数据
   * @param request_id 请求ID
   * @return bool 如果成功清理则返回true，如果元数据不存在则返回false
   */
  bool CleanupTransferMeta(int request_id) {
    std::lock_guard<std::mutex> lock(meta_map_mutex_);
    return meta_map_.erase(request_id) > 0;
  }

  void SetGroupRole(GroupRole group_role) { group_role_ = group_role; }

 private:
  /**
   * @brief 为decode节点创建传输任务
   * @param request_id 请求ID
   * @param transfer_meta 传输元数据的共享指针
   * @param device_num 设备数量
   * @param block_num 块数量
   * @param shared_block_num 请求共享block数量
   */
  void CreateTransferTasksForDecodeNode(int request_id, std::shared_ptr<TransferMeta>& transfer_meta, size_t device_num,
                                        size_t block_num, size_t shared_block_num);

  PipelineConfig pipeline_config_;

  // 张量并行大小
  size_t tensor_parallel_size_ = 0;

  size_t attn_data_parallel_size_ = 0;

  // 保护元数据映射的互斥锁
  std::mutex meta_map_mutex_;

  // 请求ID到传输元数据的映射
  std::unordered_map<int, std::shared_ptr<TransferMeta>> meta_map_;

  // 节点角色
  GroupRole group_role_ = GroupRole::DECODE;

  BlockManagerConfig block_manager_config_;

  int layer_num_ = 0;
  // 每次传输的层数量
  size_t transfer_layer_chunk_size_ = 1;

  size_t block_size_ = 0;

  // KV缓存数据类型
  DataType kv_cache_dtype_;

  // 传输连接器
  std::shared_ptr<Connector> connector_;

  // 设备信息管理器
  std::shared_ptr<DeviceInfoManager> device_info_manager_ = std::make_shared<DeviceInfoManager>();

  // 不需要prefill的decode节点状态
  bool decode_node_benchmark = false;

  std::once_flag init_once_flag_;

  /**
   * @brief 验证层索引是否有效
   * @param layer_idx 层索引
   * @return bool 如果索引有效则返回true，否则返回false
   */

  inline bool ValidateLayerIndex(int layer_idx) const {
    return (layer_idx >= pipeline_config_.lower_layer_idx && layer_idx <= pipeline_config_.upper_layer_idx) ||
           (layer_idx >= pipeline_config_.lower_nextn_layer_idx && layer_idx <= pipeline_config_.upper_nextn_layer_idx);
  }

  /**
   * @brief 计算层偏移量
   * @param layer_idx 层索引
   * @return int 层偏移量
   */
  inline int CalculateLayerOffset(int layer_idx) const { return layer_idx - pipeline_config_.lower_layer_idx; }
};

}  // namespace ksana_llm