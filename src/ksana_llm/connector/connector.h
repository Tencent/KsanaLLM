/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ksana_llm/connector/communicator/communicator_manager.h"
#include "ksana_llm/connector/config.h"
#include "ksana_llm/connector/task_dispatcher.h"
#include "ksana_llm/connector/task_manager.h"
#include "ksana_llm/transfer/transfer_types.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

class Connector {
 public:
  explicit Connector(const ConnectorConfig& config, int attn_tensor_para_size, int node_rank,
                     std::shared_ptr<Environment> env);

  Connector(const Connector&) = delete;
  Connector& operator=(const Connector&) = delete;
  Connector() = default;  // 新增，允许子类默认构造
  virtual ~Connector();   // 声明为虚析构函数

  virtual Status Initialize(GroupRole group_role);

  // 启动传输任务处理线程，为兼容性添加
  virtual void Start();

  virtual void PushTask(const std::shared_ptr<TransferTask>& task);

 private:
  ConnectorConfig config_;
  std::shared_ptr<TaskManager> task_manager_;
  std::shared_ptr<TaskDispatcher> task_dispatcher_;
  std::shared_ptr<Environment> env_;
  std::shared_ptr<Coordinator> coordinator_;
  std::shared_ptr<CommunicatorManager> comm_manager_;
  bool is_prefill_ = false;
};

}  // namespace ksana_llm