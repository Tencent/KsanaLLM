/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/coordinator/default_coordinator.h"

namespace ksana_llm {

DefaultCoordinator::DefaultCoordinator(const ConnectorConfig& config, std::shared_ptr<RouterClient> router_client)
    : config_(config), router_client_(router_client) {
  if (!router_client_) {
    throw std::invalid_argument("client cannot be nullptr");
  }
}

Status DefaultCoordinator::RegisterNode() {
  node_info_.coordinator_addr = config_.coordinator_addr;
  node_info_.cluster_name = config_.cluster_name;
  node_info_.group_role = GroupRoleToString(config_.group_role);
  node_info_.node_rank = config_.node_rank;
  node_info_.world_size = config_.world_size;

  std::string host_ip;
  std::string interface;
  auto status = GetAvailableInterfaceAndIP(interface, host_ip);
  if (!status.OK()) {
    host_ip = "127.0.0.1";
  }

  if (config_.inference_addr.empty()) {
      KLLM_LOG_ERROR << "Inference address is empty";
  } else {
      node_info_.inference_addr = config_.inference_addr;
  }
  node_info_.devices = DeviceCollector::CollectDeviceInformation(config_.device_count, host_ip);

  // Send registration request to router
  status = router_client_->RegisterNode(node_info_);
  if (status.OK()) {
    node_info_.node_id = router_client_->GetNodeInfo().node_id;
    KLLM_LOG_INFO << "Node successfully registered with ID: " << node_info_.node_id;
  }
  return status;
}

Status DefaultCoordinator::Initialize() {
  if (initialized_) {
    return Status(RetCode::RET_INIT_FAILED, "Coordinator is already initialized");
  }
  Status status = RegisterNode();
  if (!status.OK()) return status;

  status = StartHeartbeat();
  if (!status.OK()) return status;

  KLLM_LOG_INFO << "Coordinator initialized successfully (DI version)";
  initialized_ = true;
  return Status();
}

bool DefaultCoordinator::IsInitialized() const { return initialized_; }

Status DefaultCoordinator::StartHeartbeat() {
  if (heartbeat_) {
    return Status();  // Already running
  }

  KLLM_LOG_INFO << "Starting heartbeat_thread_ thread";
  heartbeat_ = true;
  heartbeat_thread_ = std::make_unique<std::thread>(&DefaultCoordinator::HeartbeatThread, this);
  KLLM_LOG_INFO << "Heartbeat thread started";
  return Status();
}

void DefaultCoordinator::HeartbeatThread() {
  while (heartbeat_) {
    KVHeartbeatResponse heartbeat_response;
    Status status = router_client_->SendHeartbeat(node_info_.node_id, heartbeat_response);
    if (status.OK()) {
      if (comm_setup_callback_) {
        comm_setup_callback_(heartbeat_response.comm_group_to_id, heartbeat_response.comm_group_to_address);
      }

      KLLM_LOG_DEBUG << "Heartbeat sent to router and Status is: " << status.GetMessage();
    } else {
      KLLM_LOG_ERROR << "Failed to send heartbeat: " << status.GetMessage();
    }

    // Always sleep between heartbeat attempts, regardless of success or failure
    std::this_thread::sleep_for(std::chrono::milliseconds(config_.heartbeat_interval_ms));
  }
}

Status DefaultCoordinator::SendCommId(const std::string& comm_key, const std::string& comm_id) {
  if (router_client_) {
    return router_client_->SendCommId(node_info_.node_id, comm_key, comm_id);
  }
  return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Router client is not initialized");
}

Status DefaultCoordinator::SendHeartbeat(std::string& node_id, KVHeartbeatResponse& response) {
  if (router_client_) {
    return router_client_->SendHeartbeat(node_id, response);
  }
  return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Router client is not initialized");
}

void DefaultCoordinator::StopHeartbeat() {
  heartbeat_ = false;
  if (heartbeat_thread_ && heartbeat_thread_->joinable()) {
    heartbeat_thread_->join();
  }
  heartbeat_thread_.reset();
}

DefaultCoordinator::~DefaultCoordinator() { Shutdown(); }

void DefaultCoordinator::Shutdown() {
  StopHeartbeat();
  if (router_client_) {
    router_client_.reset();
  }
  KLLM_LOG_INFO << "DefaultCoordinator shutdown complete";
}

void DefaultCoordinator::OnHeartbeatResponseCallback(HeartbeatResponseCallback cb) {
  comm_setup_callback_ = std::move(cb);
}

const KVNodeInfo& DefaultCoordinator::GetNodeInfo() const { return node_info_; }
}  // namespace ksana_llm