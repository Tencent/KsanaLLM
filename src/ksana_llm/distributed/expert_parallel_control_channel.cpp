/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/distributed/expert_parallel_control_channel.h"

#include <torch/csrc/utils/variadic.h>

#include <chrono>
#include <complex>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/expert_data_hub.h"
#include "ksana_llm/distributed/control_channel.h"
#include "ksana_llm/distributed/control_message.h"
#include "ksana_llm/distributed/node_info.h"
#include "ksana_llm/distributed/packet_util.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/distributed/raw_socket.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/service_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {
ExpertParallelControlChannel::ExpertParallelControlChannel(const std::string& master_host, uint16_t master_port,
                                                           size_t world_size, int node_rank,
                                                           size_t global_expert_para_size,
                                                           PacketCreationFunc packet_creation_fn,
                                                           ScheduleOutputPool* schedule_output_pool,
                                                           std::shared_ptr<Environment> env)
    : ControlChannel(master_host, master_port, 1, node_rank, packet_creation_fn, nullptr, env) {
  world_size_ = world_size;
  node_rank_ = node_rank;
  global_expert_para_size_ = global_expert_para_size;
  master_host_ = master_host;
  master_port_ = master_port;

  raw_socket_ = std::make_shared<RawSocket>(packet_creation_fn);

  env_ = env ? env : Singleton<Environment>::GetInstance();

  // Start assisant threads.
  heartbeat_thread_ =
      std::unique_ptr<std::thread>(new std::thread(&ExpertParallelControlChannel::ProcessHeartbeatLoop, this));
  // send_packet_thread_ =
  //     std::unique_ptr<std::thread>(new std::thread(&ControlChannel::ProcessSendScheduleOutputLoop, this));
}

ExpertParallelControlChannel::~ExpertParallelControlChannel() {
  terminated_ = true;
  if (heartbeat_thread_) {
    heartbeat_thread_->join();
  }
}

Status ExpertParallelControlChannel::ProcessHeartbeatLoop() {
  while (!terminated_) {
    time_t curr_time_stamp = GetCurrentTime();

    {
      std::unique_lock<std::mutex> lock(mutex_);

      // For master and worker.
      for (auto it = node_heartbeat_timestamp_.begin(); it != node_heartbeat_timestamp_.end(); ++it) {
        time_t last_time_stamp = it->second;
        if (curr_time_stamp > last_time_stamp + heartbeat_timeout_secs_) {
          KLLM_LOG_ERROR << "Heartbeat timeout, cluster exited.";

          if (node_rank_ == 0) {
            // For master node, stop whole cluster.
            ShutdownCluster();
          } else {
            // For worker node, stop current service.
            GetServiceLifetimeManager()->ShutdownService();
          }
        }

        // For worker node.
        if (node_rank_ > 0) {
          if (raw_socket_->IsConnected() && curr_time_stamp > last_time_stamp + heartbeat_interval_secs_) {
            // Send heartbeat to master.
            Packet* packet = GetPacketObject(PacketType::CONTROL_REQ_HEARTBEAT, 0);
            if (packet == nullptr) {
              throw std::runtime_error(
                  "ExpertParallelControlChannel::ProcessHeartbeatLoop allocate memory "
                  "error.");
            }

            HeartbeatRequest* heartbeat_req = reinterpret_cast<HeartbeatRequest*>(packet->body);
            heartbeat_req->node_rank = node_rank_;

            Status status = raw_socket_->Send({master_host_, master_port_}, packet);
            free(packet);

            if (!status.OK()) {
              KLLM_LOG_ERROR << "ControlChannel heartbeat error, send packet failed, info:" << status.GetMessage();
            }
          }
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return Status();
}

Status ExpertParallelControlChannel::Listen() {
  auto listen_fn = [this](NodeInfo* node_info, Packet* packet) -> Status {
    return HandleServerPacket(node_info, packet);
  };

  KLLM_LOG_INFO << "ExpertParallelControlChannel listen on " << master_host_ << ":" << master_port_ << ".";
  Status status = raw_socket_->Listen(master_host_, master_port_, listen_fn);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Listen control channel error:" << status.GetMessage();
  }

  return status;
}

Status ExpertParallelControlChannel::Close() { return raw_socket_->Close(); }

Status ExpertParallelControlChannel::Connect() {
  auto connect_fn = [this](NodeInfo* node_info, Packet* packet) -> Status {
    return HandleClientPacket(node_info, packet);
  };

  KLLM_LOG_INFO << "ExpertParallelControlChannel connect to " << master_host_ << ":" << master_port_ << ".";
  Status status = raw_socket_->Connect(master_host_, master_port_, connect_fn);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "ExpertParallelControlChannel Connect control channel error:" << status.GetMessage();
  } else {
    KLLM_LOG_INFO << "ExpertParallelControlChannel connect to " << master_host_ << ":" << master_port_
                  << " succeed. \n";
  }

  return status;
}

Status ExpertParallelControlChannel::Disconnect() { return raw_socket_->Disconnect(); }

Status ExpertParallelControlChannel::ProcessAddNodeRequest(NodeInfo* node_info, Packet* req_packet) {
  auto it = node_ranks_.find(*node_info);
  if (it != node_ranks_.end()) {
    return Status(RET_RUNTIME_FAILED, fmt::format("Duplicated node {}:{}", node_info->host, node_info->port));
  }

  AddNodeRequest* add_node_req = reinterpret_cast<AddNodeRequest*>(req_packet->body);

  size_t req_node_rank = add_node_req->node_rank;
  node_ranks_[*node_info] = req_node_rank;
  rank_nodes_[req_node_rank] = *node_info;

  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_ADD_NODE, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ExpertParallelControlChannel::ProcessAddNodeRequest allocate memory error.");
  }

  Status status = raw_socket_->Send(*node_info, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ExpertParallelControlChannel process the add node reqeust error, send "
                      "packet failed, info:"
                   << status.GetMessage();
  }

  free(req_packet);
  return status;
}

Status ExpertParallelControlChannel::ProcessAddNodeResponse(NodeInfo* node_info, Packet* rsp_packet) {
  free(rsp_packet);
  return Status();
}

Status ExpertParallelControlChannel::ProcessHeartbeatRequest(NodeInfo* node_info, Packet* req_packet) {
  std::unique_lock<std::mutex> lock(mutex_);

  HeartbeatRequest* heartbeat_req = reinterpret_cast<HeartbeatRequest*>(req_packet->body);
  size_t req_node_rank = heartbeat_req->node_rank;
  node_heartbeat_timestamp_[req_node_rank] = GetCurrentTime();

  // Send response.
  Packet* packet = GetPacketObject(PacketType::CONTROL_RSP_HEARTBEAT, 0);
  if (packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessHeartbeatRequest allocate memory error.");
  }

  HeartbeatResponse* heartbeat_rsp = reinterpret_cast<HeartbeatResponse*>(packet->body);
  heartbeat_rsp->node_rank = node_rank_;

  Status status = raw_socket_->Send({master_host_, master_port_}, packet);
  free(packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel process heartbeat reqeust error, send "
                      "packet failed, info:"
                   << status.GetMessage();
  }

  free(req_packet);
  return Status();
}

Status ExpertParallelControlChannel::ProcessHeartbeatResponse(NodeInfo* node_info, Packet* rsp_packet) {
  std::unique_lock<std::mutex> lock(mutex_);

  HeartbeatResponse* heartbeat_rsp = reinterpret_cast<HeartbeatResponse*>(rsp_packet->body);
  size_t req_node_rank = heartbeat_rsp->node_rank;
  node_heartbeat_timestamp_[req_node_rank] = GetCurrentTime();

  free(rsp_packet);
  return Status();
}

Status ExpertParallelControlChannel::ProcessBarrierRequest(NodeInfo* node_info, Packet* req_packet) {
  BarrierRequest* barrier_req = reinterpret_cast<BarrierRequest*>(req_packet->body);

  int clock_idx = barrier_req->clock_idx;
  if (barrier_req_ranks_.find(clock_idx) == barrier_req_ranks_.end()) {
    barrier_req_ranks_.insert(std::make_pair(clock_idx, std::unordered_set<int>()));
  }

  size_t req_node_rank = barrier_req->node_rank;
  barrier_req_ranks_[clock_idx].insert(req_node_rank);

  // Notify if all nodes arrives.
  if (barrier_req_ranks_[clock_idx].size() == world_size_ - 1) {
    std::unique_lock<std::mutex> lock(mutex_);
    barrier_cv_.notify_all();
  }

  free(req_packet);
  return Status();
}

Status ExpertParallelControlChannel::ProcessBarrierResponse(NodeInfo* node_info, Packet* rsp_packet) {
  BarrierResponse* barrier_rsp = reinterpret_cast<BarrierResponse*>(rsp_packet->body);

  int clock_idx = barrier_rsp->clock_idx;
  if (barrier_rsp_clocks_.find(clock_idx) == barrier_rsp_clocks_.end()) {
    barrier_rsp_clocks_.insert(clock_idx);
  }

  // Notity thread to continue.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    barrier_cv_.notify_all();
  }

  free(rsp_packet);
  return Status();
}

Status ExpertParallelControlChannel::ProcessShutdownRequest(NodeInfo* node_info, Packet* req_packet) {
  // Send response.
  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_SHUTDOWN, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ExpertParallelControlChannel::ProcessShutdownRequest allocate memory error.");
  }

  Status status = raw_socket_->Send({master_host_, master_port_}, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ExpertParallelControlChannel process shutdown reqeust error, send "
                      "packet failed, info:"
                   << status.GetMessage();
  }

  GetServiceLifetimeManager()->ShutdownService();

  free(req_packet);
  return status;
}

Status ExpertParallelControlChannel::ProcessShutdownResponse(NodeInfo* node_info, Packet* rsp_packet) {
  std::unique_lock<std::mutex> lock(mutex_);

  auto it = node_ranks_.find(*node_info);
  if (it == node_ranks_.end()) {
    return Status(RET_RUNTIME_FAILED, "Unknown node received.");
  }

  size_t target_node_rank = it->second;
  shutdown_nodes_.insert(target_node_rank);

  if (shutdown_nodes_.size() == world_size_) {
    shutdown_cv_.notify_all();
  }

  free(rsp_packet);
  return Status();
}

Status ExpertParallelControlChannel::Barrier() {
  ++barrier_clock_idx_;

  // The master does not send request to itself.
  if (node_rank_ > 0) {
    Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_BARRIER);
    if (req_packet == nullptr) {
      throw std::runtime_error("ControlChannel::Barrier allocate memory error.");
    }

    BarrierRequest* barrier_req = reinterpret_cast<BarrierRequest*>(req_packet->body);
    barrier_req->node_rank = node_rank_;
    barrier_req->clock_idx = barrier_clock_idx_;

    Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
    free(req_packet);

    if (!status.OK()) {
      KLLM_LOG_ERROR << "ControlChannel barrier error, send packet failed, info:" << status.GetMessage();
      return status;
    }
  }

  if (node_rank_ == 0) {
    // Wait until all nodes
    std::unique_lock<std::mutex> lock(mutex_);

    barrier_cv_.wait(lock, [this]() -> bool {
      return (node_ranks_.size() == world_size_ - 1) &&
             (barrier_req_ranks_[barrier_clock_idx_].size() == world_size_ - 1);
    });

    // Send response to all nodes.
    for (auto it = node_ranks_.begin(); it != node_ranks_.end(); ++it) {
      NodeInfo node_info = it->first;

      Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_BARRIER, 0);
      if (rsp_packet == nullptr) {
        throw std::runtime_error("ControlChannel::Barrier allocate memory error.");
      }

      BarrierResponse* barrier_rsp = reinterpret_cast<BarrierResponse*>(rsp_packet->body);
      barrier_rsp->clock_idx = barrier_clock_idx_;

      Status status = raw_socket_->Send(node_info, rsp_packet);
      free(rsp_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel barrier error, send packet failed, info:" << status.GetMessage();
      }
    }

  } else {
    // Wait master response
    std::unique_lock<std::mutex> lock(mutex_);

    barrier_cv_.wait(
        lock, [this]() -> bool { return (barrier_rsp_clocks_.find(barrier_clock_idx_) != barrier_rsp_clocks_.end()); });
  }

  return Status();
}

Status ExpertParallelControlChannel::Frozen() { return raw_socket_->Frozen(); }

// ToDo.
Status ExpertParallelControlChannel::AddNode() {
  ExpertParallelConfig expert_parallel_config;
  env_->GetExpertParallelConfig(expert_parallel_config);

  Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_ADD_NODE, 0);
  if (req_packet == nullptr) {
    throw std::runtime_error("ControlChannel::AddNode allocate memory error.");
  }

  AddNodeRequest* add_node_req = reinterpret_cast<AddNodeRequest*>(req_packet->body);
  add_node_req->node_rank = node_rank_;

  Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
  free(req_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel add node error, send packet failed, info:" << status.GetMessage();
  }

  return status;
}

Status ExpertParallelControlChannel::ShutdownCluster() {
  // Only master can call shutdown.
  if (node_rank_ == 0) {
    for (size_t target_node_rank = 1; target_node_rank < world_size_; target_node_rank++) {
      Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_SHUTDOWN, 0);
      if (req_packet == nullptr) {
        throw std::runtime_error("ControlChannel::ShutdownCluster allocate memory error.");
      }

      Status status = raw_socket_->Send(rank_nodes_[target_node_rank], req_packet);
      free(req_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel shutdown error, send packet failed, info:" << status.GetMessage();
      }
    }

    // Wait for all workers.
    std::unique_lock<std::mutex> lock(mutex_);
    shutdown_nodes_.insert(0);

    // Wait at most 5 seconds.
    size_t timeout = 5;
    block_num_cv_.wait_for(lock, std::chrono::seconds(timeout),
                           [this]() -> bool { return shutdown_nodes_.size() == world_size_; });

    // Shutdown master node finally.
    GetServiceLifetimeManager()->ShutdownService();
  }

  return Status();
}

Status ExpertParallelControlChannel::SynchronizeNvshmemUniqueId() {
  KLLM_LOG_INFO << "SynchronizeNvshmemUniqueId";
  if (world_size_ == 1) {
    return Status();
  }
  if (deepep_wrapper_ == nullptr) {
    deepep_wrapper_ = GetExpertParallelDeepepWrapper();
  }
  // Workers report ipc_handles to Master for aggregation.
  char* ipc_handles = deepep_wrapper_->GetIPCHandles();
  uint8_t* nvshmem_unique_id = deepep_wrapper_->GetNvshmemUniqueId();

  int local_num_ranks = global_expert_para_size_ / world_size_;
  if (node_rank_ != 0) {
    // Worker nodes (node_rank_ != 0) send messages (ipc handles ptr) to master.
    Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_DEEPEP_META, 0);
    if (req_packet == nullptr) {
      throw std::runtime_error("ExpertParallelControlChannel::SynchronizeNvshmemUniqueId allocate memory error.");
    }
    NvshmemUniqueIdRequest* nvshmem_req = reinterpret_cast<NvshmemUniqueIdRequest*>(req_packet->body);
    nvshmem_req->node_rank = node_rank_;

    char* target_ipc_handles = reinterpret_cast<char*>(nvshmem_req->ipc_handles[node_rank_ * local_num_ranks]);
    std::memcpy(target_ipc_handles, ipc_handles + (node_rank_ * local_num_ranks * kIpcHandlesSize),
                kIpcHandlesSize * local_num_ranks);

    Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
    free(req_packet);
    if (!status.OK()) {
      KLLM_LOG_ERROR << node_rank_ << " send IPC Handle to master failed: " << status.GetMessage();
      return status;
    }
  } else {
    // Master node waits for data from worker nodes.
    std::unique_lock<std::mutex> lock(mutex_);
    auto ipc_handle_ptr = std::make_unique<char[]>(kIpcHandlesSize * local_num_ranks);
    std::memcpy(ipc_handle_ptr.get(), ipc_handles, kIpcHandlesSize * local_num_ranks);
    node_ipc_handles_[0] = std::move(ipc_handle_ptr);
    ipc_handles_cv_.wait(lock, [this]() -> bool { return node_ipc_handles_.size() == world_size_; });
    for (auto& pair : node_ipc_handles_) {
      int req_node_rank = pair.first;
      size_t ipc_handles_size_per_node = kIpcHandlesSize * local_num_ranks;
      char* src_ipc_handles = pair.second.get();
      char* dst_ipc_handles = ipc_handles + req_node_rank * ipc_handles_size_per_node;
      std::memcpy(dst_ipc_handles, src_ipc_handles, ipc_handles_size_per_node);
    }
  }

  // Master aggregates results and sends nvshmem_unique_id and all ipc_handles to workers.
  if (node_rank_ == 0) {
    uint8_t* nvshmem_unique_id = deepep_wrapper_->GetNvshmemUniqueId();
    for (size_t target_node_rank = 1; target_node_rank < world_size_; ++target_node_rank) {
      Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_DEEPEP_META, 0);
      if (req_packet == nullptr) {
        throw std::runtime_error("ExpertParallelControlChannel::SynchronizeNvshmemUniqueId allocate memory error.");
      }

      NvshmemUniqueIdRequest* nvshmem_req = reinterpret_cast<NvshmemUniqueIdRequest*>(req_packet->body);

      nvshmem_req->node_rank = node_rank_;
      std::memcpy(nvshmem_req->nvshmem_unique_id, nvshmem_unique_id, global_expert_para_size_ * kNvshmemUniqudIdSize);
      std::memcpy(reinterpret_cast<char*>(nvshmem_req->ipc_handles), ipc_handles,
                  global_expert_para_size_ * kIpcHandlesSize);

      Status status = raw_socket_->Send(rank_nodes_[target_node_rank], req_packet);
      free(req_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "Send Nvshmem Unique Id to worker " << target_node_rank << " failed: " << status.GetMessage();
        return status;
      }
    }
  } else {
    // Worker node waits for data from master nodes.
    std::unique_lock<std::mutex> lock(mutex_);
    nvshmem_unique_id_cv_.wait(lock, [this]() -> bool { return nvshmem_unique_id_synchronized_; });
  }
  deepep_wrapper_->SetReady();
  return Status();
}

Status ExpertParallelControlChannel::SetExpertParallelDeepepWrapper(
    const std::shared_ptr<ExpertParallelDeepepWrapper>& deepep_wrapper) {
  deepep_wrapper_ = deepep_wrapper;
  return Status();
}

Status ExpertParallelControlChannel::ProcessNvshmemUniqueIdRequest(NodeInfo* node_info, Packet* req_packet) {
  KLLM_LOG_INFO << "ProcessNvshmemUniqueIdRequest";
  NvshmemUniqueIdRequest* nvshmem_req = reinterpret_cast<NvshmemUniqueIdRequest*>(req_packet->body);
  if (deepep_wrapper_ == nullptr) {
    deepep_wrapper_ = GetExpertParallelDeepepWrapper();
  }
  uint8_t* nvshmem_unique_id = deepep_wrapper_->GetNvshmemUniqueId();
  if (nvshmem_req->node_rank > 0) {
    // Process requests received by Master.
    size_t req_node_rank = nvshmem_req->node_rank;
    size_t local_num_ranks = global_expert_para_size_ / world_size_;
    auto ipc_handle_ptr = std::make_unique<char[]>(kIpcHandlesSize * local_num_ranks);
    std::memcpy(ipc_handle_ptr.get(), nvshmem_req->ipc_handles[req_node_rank * local_num_ranks],
                kIpcHandlesSize * local_num_ranks);
    {
      std::unique_lock<std::mutex> lock(mutex_);
      node_ipc_handles_[req_node_rank] = std::move(ipc_handle_ptr);
      if (node_ipc_handles_.size() == world_size_) {
        ipc_handles_cv_.notify_all();
      }
    }
  } else {
    // Process requests received by Worker.
    std::memcpy(nvshmem_unique_id, nvshmem_req->nvshmem_unique_id, global_expert_para_size_ * kNvshmemUniqudIdSize);
    char* ipc_handles = deepep_wrapper_->GetIPCHandles();
    std::memcpy(ipc_handles, nvshmem_req->ipc_handles, global_expert_para_size_ * kIpcHandlesSize);
    nvshmem_unique_id_synchronized_ = true;
    nvshmem_unique_id_cv_.notify_all();
  }

  // Send response.
  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_DEEPEP_META, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ExpertParallelControlChannel::ProcessNvshmemUniqueIdRequest allocate memory error.");
  }

  NvshmemUniqueIdResponse* nvshmem_rsp = reinterpret_cast<NvshmemUniqueIdResponse*>(rsp_packet->body);
  nvshmem_rsp->status = 0;  // success
  Status status = raw_socket_->Send(*node_info, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "Send Nvshmem Unique Id response failed: " << status.GetMessage();
  }

  free(req_packet);
  return status;
}

Status ExpertParallelControlChannel::ProcessNvshmemUniqueIdResponse(NodeInfo* node_info, Packet* rsp_packet) {
  KLLM_LOG_INFO << "ProcessNvshmemUniqueIdResponse";
  free(rsp_packet);
  return Status();
}

}  // namespace ksana_llm
