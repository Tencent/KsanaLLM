/* Copyright 2024 Tencent Inc.  All rights reserved.

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
                                                           PacketCreationFunc packet_creation_fn,
                                                           ScheduleOutputPool* schedule_output_pool,
                                                           std::shared_ptr<Environment> env)
    : ControlChannel(master_host, master_port, 1, node_rank, packet_creation_fn, nullptr, env) {
  world_size_ = world_size;
  node_rank_ = node_rank;

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

  int node_rank = add_node_req->node_rank;
  node_ranks_[*node_info] = node_rank;
  rank_nodes_[node_rank] = *node_info;

  char* data_host = add_node_req->data_host;
  uint16_t data_port = add_node_req->data_port;
  rank_data_nodes_[node_rank] = {std::string(data_host), data_port};
  KLLM_LOG_INFO << "ExpertParallelControlChannel add node, data_host: " << std::string(data_host)
                << ", port: " << data_port << ", node_rank: " << node_rank;

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
  int node_rank = heartbeat_req->node_rank;
  node_heartbeat_timestamp_[node_rank] = GetCurrentTime();

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
  int node_rank = heartbeat_rsp->node_rank;
  node_heartbeat_timestamp_[node_rank] = GetCurrentTime();

  free(rsp_packet);
  return Status();
}

Status ExpertParallelControlChannel::ProcessBarrierRequest(NodeInfo* node_info, Packet* req_packet) {
  BarrierRequest* barrier_req = reinterpret_cast<BarrierRequest*>(req_packet->body);

  int clock_idx = barrier_req->clock_idx;
  if (barrier_req_ranks_.find(clock_idx) == barrier_req_ranks_.end()) {
    barrier_req_ranks_.insert(std::make_pair(clock_idx, std::unordered_set<int>()));
  }

  int node_rank = barrier_req->node_rank;
  barrier_req_ranks_[clock_idx].insert(node_rank);

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

Status ExpertParallelControlChannel::ProcessLayerRequest(NodeInfo* node_info, Packet* req_packet) {
  ExpertParallelConfig expert_parallel_config;
  env_->GetExpertParallelConfig(expert_parallel_config);

  AllocateExpertRequest* layer_req = reinterpret_cast<AllocateExpertRequest*>(req_packet->body);

  // update pipeline config.
  // expert_parallel_config.lower_layer_idx = layer_req->lower_layer_idx;
  // expert_parallel_config.upper_layer_idx = layer_req->upper_layer_idx;
  // expert_parallel_config.lower_nextn_layer_idx = layer_req->lower_nextn_layer_idx;
  // expert_parallel_config.upper_nextn_layer_idx = layer_req->upper_nextn_layer_idx;
  expert_parallel_config.downstream_host = layer_req->downstream_host;
  expert_parallel_config.downstream_port = layer_req->downstream_port;

#ifdef ENABLE_CUDA
  memcpy(expert_parallel_config.nccl_unique_id, layer_req->nccl_unique_id, sizeof(layer_req->nccl_unique_id));
  KLLM_LOG_INFO << "ProcessLayerRequest, recv nccl_unique_id: " << expert_parallel_config.nccl_unique_id
                << ", layer_req->nccl_unique_id: " << layer_req->nccl_unique_id;
#endif

  env_->SetExpertParallelConfig(expert_parallel_config);

  {
    std::unique_lock<std::mutex> lock(mutex_);
    layer_allocated_ = true;
    layer_allocation_cv_.notify_all();
  }

  // Send response.
  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_LAYER, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ExpertParallelControlChannel::ProcessLayerRequest allocate memory error.");
  }

  Status status = raw_socket_->Send({master_host_, master_port_}, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ExpertParallelControlChannel process allocate layer reqeust error, "
                      "send packet failed, info:"
                   << status.GetMessage();
  } else {
    KLLM_LOG_INFO << "ExpertParallelControlChannel process allocate layer reqeust succeed";
  }

  free(req_packet);
  return status;
}

Status ExpertParallelControlChannel::ProcessLayerResponse(NodeInfo* node_info, Packet* rsp_packet) {
  free(rsp_packet);
  return Status();
}

Status ExpertParallelControlChannel::ProcessExpertParallelRequest(NodeInfo* node_info, Packet* req_packet) {
  ExpertParallelConfig expert_parallel_config;
  env_->GetExpertParallelConfig(expert_parallel_config);

  AllocateExpertRequest layer_req;
  DeserializeAllocateExpertRequest(layer_req, req_packet->body, expert_parallel_config.expert_world_size);

  expert_parallel_config.downstream_host = layer_req.downstream_host;
  expert_parallel_config.downstream_port = layer_req.downstream_port;

#ifdef ENABLE_CUDA
  memcpy(expert_parallel_config.nccl_unique_id, layer_req.nccl_unique_id, sizeof(layer_req.nccl_unique_id));
  KLLM_LOG_INFO << "ProcessExpertParallelRequest, expert_parallele_config nccl_unique_id: "
                << expert_parallel_config.nccl_unique_id;

  expert_parallel_config.nccl_unique_ids.resize(world_size_);
  for (int i = 0; i < world_size_; i++) {
    memcpy(expert_parallel_config.nccl_unique_ids[i].data(), layer_req.nccl_unique_ids[i].data(),
           sizeof(layer_req.nccl_unique_id));
    KLLM_LOG_INFO << "ProcessExpertParallelRequest, rank: " << i << ", expert_parallele_config nccl_unique_ids: : "
                  << reinterpret_cast<char*>(expert_parallel_config.nccl_unique_ids[i].data());
  }
#endif

  env_->SetExpertParallelConfig(expert_parallel_config);

  {
    std::unique_lock<std::mutex> lock(mutex_);
    layer_allocated_ = true;
    layer_allocation_cv_.notify_all();
  }

  // Send response.
  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_EXPERT_PARALLEL, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ExpertParallelControlChannel::ProcessLayerRequest allocate memory error.");
  }

  Status status = raw_socket_->Send({master_host_, master_port_}, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ExpertParallelControlChannel process allocate layer reqeust error, "
                      "send packet failed, info:"
                   << status.GetMessage();
  } else {
    KLLM_LOG_INFO << "ExpertParallelControlChannel process allocate layer reqeust succeed";
  }

  free(req_packet);
  return status;
}

Status ExpertParallelControlChannel::ProcessExpertParallelResponse(NodeInfo* node_info, Packet* rsp_packet) {
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

  int node_rank = it->second;
  shutdown_nodes_.insert(node_rank);

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

  strcpy(add_node_req->data_host, expert_parallel_config.data_host.c_str());
  add_node_req->data_port = expert_parallel_config.data_port;

  KLLM_LOG_INFO << "ExpertParallelControlChannel add node, node_rank " << node_rank_ << ", data endpoint "
                << add_node_req->data_host << ":" << add_node_req->data_port;
  Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
  free(req_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel add node error, send packet failed, info:" << status.GetMessage();
  }

  return status;
}

// buffer = request.
Status ExpertParallelControlChannel::SerializeAllocateExpertRequest(char* buffer, AllocateExpertRequest& request,
                                                                    size_t world_size) {
  char* current = buffer;
  // downstream_host
  memcpy(current, request.downstream_host, 16);
  current += 16;

  // downstream port.
  memcpy(current, reinterpret_cast<char*>(&request.downstream_port), sizeof(uint16_t));
  current += sizeof(uint16_t);
#ifdef ENABLE_CUDA
  // 反序列化 nccl_unique_id
  memcpy(current, reinterpret_cast<char*>(request.nccl_unique_id), 128);
  current += 128;

  // 反序列化 nccccl_unique_ids 的每个元素
  for (uint32_t i = 0; i < world_size; i++) {
    memcpy(current, request.nccl_unique_ids[i].data(), 128);
    std::cout << "SerializeAllocateExpertRequest, unique_dst: " << std::hex << std::setw(2) << current
              << ", unique_src: " << request.nccl_unique_ids[i].data() << std::endl;
    current += 128;
  }
#endif

  return Status();
}

Status ExpertParallelControlChannel::DeserializeAllocateExpertRequest(AllocateExpertRequest& request,
                                                                      const char* buffer, size_t world_size) {
  std::cout << "DeserializeAllocateExpertRequest world_size: " << world_size << std::endl;
  const char* current = buffer;
  // downstream_host
  memcpy(request.downstream_host, current, 16);
  current += 16;

  // downstream port.
  memcpy(&request.downstream_port, current, sizeof(uint16_t));
  current += sizeof(uint16_t);

#ifdef ENABLE_CUDA
  // 反序列化 nccl_unique_id
  memcpy(request.nccl_unique_id, current, 128);
  current += 128;

  //
  // 反序列化 nccccl_unique_ids 的每个元素
  request.nccl_unique_ids.resize(world_size);
  for (uint32_t i = 0; i < world_size; ++i) {
    std::array<char, 128> unique_id;
    memcpy(request.nccl_unique_ids[i].data(), current, 128);
    current += 128;
  }
#endif

  return Status();
}

Status ExpertParallelControlChannel::SynchronizeExpertParallelExperts() {
  ModelConfig model_config;
  env_->GetModelConfig("", model_config);
  const size_t num_experts = model_config.moe_config.num_experts;

  if (node_rank_ == 0) {
    ExpertParallelConfig expert_parallel_config;
    env_->GetExpertParallelConfig(expert_parallel_config);
    expert_parallel_config.expert_node_host.resize(expert_parallel_config.expert_world_size);
    expert_parallel_config.expert_node_port.resize(expert_parallel_config.expert_world_size);

    const size_t local_num_experts = (num_experts + expert_parallel_config.global_expert_para_size - 1) /
                                     expert_parallel_config.global_expert_para_size;
    expert_parallel_config.local_num_experts = local_num_experts;

    // Set expert_id vs expert node mapping.
    for (size_t i = 0; i < world_size_; i++) {
      expert_parallel_config.expert_route_table[i * local_num_experts] = i;
    }

    // TODO(xingjinglu): Support more than two nodes later.
    KLLM_LOG_INFO << "rank_data_nodes_[1].host: " << rank_data_nodes_[1].host
                  << ", rank_data_nodes_[1].port: " << rank_data_nodes_[1].port;
    expert_parallel_config.expert_node_host[1] = rank_data_nodes_[1].host;
    expert_parallel_config.expert_node_port[1] = rank_data_nodes_[1].port;
    expert_parallel_config.downstream_host = rank_data_nodes_[1].host;
    expert_parallel_config.downstream_port = rank_data_nodes_[1].port;

    env_->SetExpertParallelConfig(expert_parallel_config);
    KLLM_LOG_INFO << "ExpertParallelControlChannel set master node "
                  << ", local_num_experts: " << local_num_experts << "\n";

    // Send comm info to every worker node.
    int padding = 0;
    for (int node_rank = 1; node_rank < world_size_; ++node_rank) {
      Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_EXPERT_PARALLEL, 0);
      if (req_packet == nullptr) {
        throw std::runtime_error("ControlChannel::SynchronizeNodeLayers allocate memory error.");
      }

      AllocateExpertRequest layer_req;
#ifdef ENABLE_CUDA
      layer_req.nccl_unique_ids.resize(world_size_);
#endif

      // post-data_node
      if (node_rank == world_size_ - 1) {
        strcpy(layer_req.downstream_host, expert_parallel_config.data_host.c_str());
        layer_req.downstream_port = expert_parallel_config.data_port;
      } else {
        strcpy(layer_req.downstream_host, rank_data_nodes_[node_rank + 1].host.c_str());
        layer_req.downstream_port = rank_data_nodes_[node_rank + 1].port;
      }

      Status status;
      // Broadcast nccl unique_id for all nodes.
      if (!expert_parallel_config.use_tcp) {
#ifdef ENABLE_CUDA
        memcpy(layer_req.nccl_unique_id, expert_parallel_config.nccl_unique_id,
               sizeof(expert_parallel_config.nccl_unique_id));
        for (size_t i = 0; i < world_size_; i++) {
          memcpy(layer_req.nccl_unique_ids[i].data(), expert_parallel_config.nccl_unique_ids[i].data(),
                 sizeof(expert_parallel_config.nccl_unique_id));
        }
        KLLM_LOG_INFO << "ExpertParallelControlChannel set worker node " << node_rank << ", send  nccl_unique_id"
                      << expert_parallel_config.nccl_unique_id << "\n";
#endif

        SerializeAllocateExpertRequest(req_packet->body, layer_req, world_size_);
        status = raw_socket_->Send(rank_nodes_[node_rank], req_packet);
      }

      free(req_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "ExpertParallelControlChannel sync expert parallel nodes error, send packet, failed, info:"
                       << status.GetMessage();
      }
    }
  } else {
    // for worker node,  wait master response
    std::unique_lock<std::mutex> lock(mutex_);

    layer_allocation_cv_.wait(lock, [this]() -> bool { return layer_allocated_; });
  }

  return Status();
}

Status ExpertParallelControlChannel::ShutdownCluster() {
  // Only master can call shutdown.
  if (node_rank_ == 0) {
    for (int node_rank = 1; node_rank < world_size_; ++node_rank) {
      Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_SHUTDOWN, 0);
      if (req_packet == nullptr) {
        throw std::runtime_error("ControlChannel::ShutdownCluster allocate memory error.");
      }

      Status status = raw_socket_->Send(rank_nodes_[node_rank], req_packet);
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

}  // namespace ksana_llm
