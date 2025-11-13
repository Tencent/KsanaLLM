/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/connector.h"
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ksana_llm/connector/config.h"
#include "ksana_llm/connector/router_client/http_router_client.h"
#include "ksana_llm/connector/task_manager.h"
#include "ksana_llm/transfer/transfer_types.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {
Connector::Connector(const ConnectorConfig& config, int attn_tensor_para_size, int node_rank,
                     std::shared_ptr<Environment> env)
    : config_(config) {
  KLLM_LOG_DEBUG << "Initializing Connector with config: " << config_.toString()
                 << ", attn_tensor_para_size: " << attn_tensor_para_size << ", node_rank: " << node_rank;
  if (config_.world_size < 0) {
    PipelineConfig pipe_config;
    env->GetPipelineConfig(pipe_config);
    // TODO(shawnding): Currently, the `world_size` in pipeline parallelism refers to the number of nodes,
    // but NCCL communication actually requires the number of devices. Therefore, it should be multiplied
    // by the attention tensor parallelism size. This needs to be optimized and improved in the future.
    config_.world_size = pipe_config.world_size * attn_tensor_para_size;
  }

  config_.node_rank = node_rank;
  config_.device_count = attn_tensor_para_size;

  KLLM_LOG_INFO << "Initialize connector, config: " << config_.toString();
  // 1. 创建 RouterClient
  auto router_client = std::make_shared<HTTPRouterClient>(config_.router_addr);

  // 2. 创建 Coordinator
  coordinator_ = std::make_shared<DefaultCoordinator>(config_, router_client);

  // 3. 创建 CommunicatorManager
  comm_manager_ = std::make_shared<CommunicatorManager>(config_, coordinator_);
  int circular_bucket_num = config_.circular_bucket_num;
  if (circular_bucket_num >= env->GetMaxBatchSize()) {
    circular_bucket_num = env->GetMaxBatchSize();
  }
  task_manager_ =
      std::make_shared<TaskManager>(circular_bucket_num, config_.circular_bucket_size, config_.circular_thread_num);

  //  4. 注册/创建所有需要的 task_dispatcher_
  task_dispatcher_ = std::make_shared<TaskDispatcher>(config_, task_manager_, comm_manager_);

  is_prefill_ = (config_.group_role == GroupRole::PREFILL);
}
// 启动传输任务处理线程，为兼容性添加
void Connector::Start() {}

Status Connector::Initialize(GroupRole group_role, std::shared_ptr<DeviceInfoManager> device_info_manager) {
  if (group_role == GroupRole::NONE) {
    KLLM_LOG_ERROR << "Group role is NONE, cannot initialize connector";
    return Status(RetCode::RET_INVALID_ARGUMENT, "Group role is NONE");
  }
  Status status = coordinator_->Initialize();
  if (!status.OK()) {
    return status;
  }
  status = comm_manager_->Initialize();
  if (!status.OK()) {
    return status;
  }

  status = task_dispatcher_->Initialize(device_info_manager);
  if (!status.OK()) {
    return status;
  }
  return Status();
}

void Connector::SendConfigToPrefill(const std::string& kv_comm_group_key, size_t adp_num, size_t device_num) {
  task_dispatcher_->SendConfigToPrefill(kv_comm_group_key, adp_num, device_num);
}

void Connector::PushTask(const std::shared_ptr<TransferTask>& task) {
  std::time_t start_time = ProfileTimer::GetCurrentTimeInUs();
  if (!task || !task_manager_) {
    KLLM_LOG_ERROR << "Received null task or task_manager_ null, cannot push task";
    return;
  }

  if (task->addr.empty()) {
    if (task->req_id != 0) {
      KLLM_LOG_ERROR << "Task address is empty, cannot push task";
    }
    if (task->tensor.shape.empty() && task->dst_ptr) {
      std::vector<int> tmp_tokens(MAX_TRANSFER_TOKENS, 1);
      std::memcpy(static_cast<int*>(task->dst_ptr), tmp_tokens.data(), sizeof(int32_t) * tmp_tokens.size());
    }
    task->is_completed = true;
    return;
  }

  TaskKey task_key = task_manager_->CreateTaskKey(task);
  task_manager_->AddTask(task_key, task);
  task_manager_->PutProcessingBuffer(task_key);
  KLLM_LOG_DEBUG << "Pushed task: " << task_key.ToString() << ", total tasks: " << task->tensor.dtype
                 << ", shape: " << task->tensor.shape.size()
                 << ", time: " << ProfileTimer::GetCurrentTimeInUs() - start_time;
}

Connector::~Connector() {
  if (task_dispatcher_) {
    task_dispatcher_->Shutdown();
  }
  if (task_manager_) {
    task_manager_->Shutdown();
  }
  if (comm_manager_) {
    comm_manager_->Shutdown();
  }
  if (coordinator_) {
    coordinator_->Shutdown();
  }
}
}  // namespace ksana_llm