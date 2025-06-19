/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/communicator/communicator_manager.h"
#include "ksana_llm/connector/communicator/nvida/nccl_communicator.h"
#include "ksana_llm/connector/communicator/zmq/zmq_communicator.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

CommunicatorManager::CommunicatorManager(const ConnectorConfig& config, std::shared_ptr<Coordinator> coordinator)
    : config_(config), coordinator_(coordinator) {}

CommunicatorManager::~CommunicatorManager() { Shutdown(); }
bool CommunicatorManager::IsInitialized() const { return initialized_; }
Status CommunicatorManager::Initialize() {
  if (initialized_) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "CommunicatorManager already initialized");
  }
  if (!coordinator_ || !coordinator_->IsInitialized()) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Coordinator not initialized");
  }
  coordinator_->OnHeartbeatResponseCallback(
      [this](const std::unordered_map<std::string, std::string>& comm_group_to_id,
             const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>&
                 comm_group_to_address) { return ProcessHeartbeatData(comm_group_to_id, comm_group_to_address); });

  Status status = CreateZmqCommunicator();
  if (!status.OK()) return status;
#ifdef ENABLE_CUDA
  // Conditionally create NCCL communicator
  if (config_.communication_type == CommunicationType::NCCL) {
    status = CreateNcclCommunicator();
    if (!status.OK()) return status;
  }
#endif
  initialized_ = true;
  return Status();
}

void CommunicatorManager::Shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if already shutdown to ensure idempotency
  if (shutdown_) {
    KLLM_LOG_DEBUG << "CommunicatorManager already shutdown, skipping";
    return;
  }

  KLLM_LOG_INFO << "CommunicatorManager shutdown starting...";
  shutdown_ = true;

  // First explicitly shutdown all communicators before clearing
  for (auto& [type_name, communicator] : communicators_) {
    if (communicator) {
      KLLM_LOG_DEBUG << "Shutting down communicator: " << type_name;
      try {
        communicator->Shutdown();
      } catch (const std::exception& e) {
        KLLM_LOG_WARNING << "Exception during communicator shutdown (" << type_name << "): " << e.what();
      } catch (...) {
        KLLM_LOG_WARNING << "Unknown exception during communicator shutdown (" << type_name << ")";
      }
    }
  }

  // Then clear the containers
  communicators_.clear();
  enabled_communicator_types_.clear();

  KLLM_LOG_DEBUG << "CommunicatorManager shutdown complete";
}

Status CommunicatorManager::ProcessHeartbeatData(
    const std::unordered_map<std::string, std::string>& comm_group_to_id,
    const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>& comm_group_to_address) {
  std::lock_guard<std::mutex> lock(mutex_);
  Status status;
  for (auto& [type_name, communicator] : communicators_) {
    KLLM_LOG_DEBUG << "Processing heartbeat data for communicator type: " << type_name
                   << " And  first comm_group_to_id  is: ";
    status = communicator->ProcessHeartbeatData(comm_group_to_id, comm_group_to_address);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "Failed to process heartbeat data for communicator type %s: %s", type_name.c_str(),
          status.GetMessage();
      // 继续处理其他通信器，不中断
    }
  }
  return status;
}

Status CommunicatorManager::SendCommId(const std::string& group_key, const std::string& comm_id) {
  if (!coordinator_) return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Coordinator not initialized");
  return coordinator_->SendCommId(group_key, comm_id);
}

Communicator* CommunicatorManager::GetCommunicatorByType(const std::string& type_name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = communicators_.find(type_name);
  if (it != communicators_.end()) {
    return it->second.get();
  }
  return nullptr;
}

ZmqCommunicator* CommunicatorManager::GetZmqCommunicator() const {
  return static_cast<ZmqCommunicator*>(GetCommunicatorByType("ZMQ"));
}
#ifdef ENABLE_CUDA
NcclCommunicator* CommunicatorManager::GetNcclCommunicator() const {
  return static_cast<NcclCommunicator*>(GetCommunicatorByType("NCCL"));
}
#endif

Status CommunicatorManager::CreateZmqCommunicator() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (communicators_.find("ZMQ") != communicators_.end()) {
      return Status();  // 已存在，直接返回
    }
  }
  try {
    auto communicator = std::make_unique<ZmqCommunicator>(config_);
    Status status = communicator->Initialize();
    if (!status.OK()) return status;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      communicators_["ZMQ"] = std::move(communicator);
      enabled_communicator_types_.insert("ZMQ");
    }
    return Status();
  } catch (const std::exception& e) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, std::string("Failed to create ZMQ communicator: ") + e.what());
  }
}
#ifdef ENABLE_CUDA
Status CommunicatorManager::CreateNcclCommunicator() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (communicators_.find("NCCL") != communicators_.end()) {
      return Status();  // 已存在，直接返回
    }
  }
  try {
    auto communicator = std::make_unique<NcclCommunicator>(config_);
    auto send_comm_id_callback = [this](const std::string& group_key, const std::string& comm_id) -> Status {
      return this->SendCommId(group_key, comm_id);
    };
    communicator->SetSendCommIdCallback(send_comm_id_callback);
    Status status = communicator->Initialize();
    if (!status.OK()) return status;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      communicators_["NCCL"] = std::move(communicator);
      enabled_communicator_types_.insert("NCCL");
    }
    return Status();
  } catch (const std::exception& e) {
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, std::string("Failed to create NCCL communicator: ") + e.what());
  }
}
#endif
void CommunicatorManager::SetZmqCommunicator(std::unique_ptr<ZmqCommunicator> communicator) {
  std::lock_guard<std::mutex> lock(mutex_);
  communicators_["ZMQ"] = std::move(communicator);
  enabled_communicator_types_.insert("ZMQ");
}
#ifdef ENABLE_CUDA
void CommunicatorManager::SetNcclCommunicator(std::unique_ptr<NcclCommunicator> communicator) {
  std::lock_guard<std::mutex> lock(mutex_);
  communicators_["NCCL"] = std::move(communicator);
  enabled_communicator_types_.insert("NCCL");
}
#endif
}  // namespace ksana_llm
