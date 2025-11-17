/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/distributed/control_channel.h"

#include <torch/csrc/utils/variadic.h>
#include <cassert>
#include <chrono>
#include <complex>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include "ksana_llm/data_hub/data_hub.h"
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

ControlChannel::ControlChannel(const std::string& master_host, uint16_t master_port, size_t world_size, int node_rank,
                               PacketCreationFunc packet_creation_fn, ScheduleOutputPool* schedule_output_pool,
                               std::shared_ptr<Environment> env) {
  master_host_ = master_host;
  master_port_ = master_port;

  world_size_ = world_size;
  node_rank_ = node_rank;

  if (world_size_ > 1) {
    raw_socket_ = std::make_shared<RawSocket>(packet_creation_fn);

    env_ = env ? env : Singleton<Environment>::GetInstance();
    schedule_output_pool_ = schedule_output_pool ? schedule_output_pool : GetScheduleOutputPool();

    // Start assisant threads.

    heartbeat_thread_ = std::unique_ptr<std::thread>(new std::thread(&ControlChannel::ProcessHeartbeatLoop, this));
    send_packet_thread_ =
        std::unique_ptr<std::thread>(new std::thread(&ControlChannel::ProcessSendScheduleOutputLoop, this));
  }
}

ControlChannel::~ControlChannel() {
  if (world_size_ > 1) {
    terminated_ = true;
    schedule_output_pool_->Stop();

    if (heartbeat_thread_) {
      heartbeat_thread_->join();
    }

    if (send_packet_thread_) {
      send_packet_thread_->join();
    }
  }
}

Status ControlChannel::ProcessHeartbeatLoop() {
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
              throw std::runtime_error("ControlChannel::ProcessHeartbeatLoop allocate memory error.");
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

Status ControlChannel::ProcessSendScheduleOutputLoop() {
  while (!terminated_) {
    Packet* packet = schedule_output_pool_->GetFromSendQueue();
    if (!packet) {
      KLLM_LOG_WARNING << "ProcessSendScheduleLoop empty packet from send queue, break..";
      break;
    }

    // Send to all workers
    for (auto& [rank_id, node_info] : rank_nodes_) {
      Status status = raw_socket_->Send(node_info, packet);
      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel broadcast schedule output error, send packet failed, info:"
                       << status.GetMessage();
      }
    }

    free(packet);
  }

  return Status();
}

Status ControlChannel::Listen() {
  auto listen_fn = [this](NodeInfo* node_info, Packet* packet) -> Status {
    return HandleServerPacket(node_info, packet);
  };

  KLLM_LOG_INFO << "ControlChannel listen on " << master_host_ << ":" << master_port_ << ".";
  Status status = raw_socket_->Listen(master_host_, master_port_, listen_fn);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Listen control channel error:" << status.GetMessage();
  }

  return status;
}

Status ControlChannel::Close() { return raw_socket_->Close(); }

Status ControlChannel::Connect() {
  auto connect_fn = [this](NodeInfo* node_info, Packet* packet) -> Status {
    return HandleClientPacket(node_info, packet);
  };

  KLLM_LOG_INFO << "ControlChannel connect to " << master_host_ << ":" << master_port_ << ".";
  Status status = raw_socket_->Connect(master_host_, master_port_, connect_fn);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Connect control channel error:" << status.GetMessage();
  } else {
    KLLM_LOG_INFO << "ControlChannel connect to " << master_host_ << ":" << master_port_ << " succeed. \n";
  }

  return status;
}

Status ControlChannel::Disconnect() { return raw_socket_->Disconnect(); }

Status ControlChannel::ProcessAddNodeRequest(NodeInfo* node_info, Packet* req_packet) {
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

  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_ADD_NODE, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessAddNodeRequest allocate memory error.");
  }

  Status status = raw_socket_->Send(*node_info, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel process the add node reqeust error, send packet failed, info:"
                   << status.GetMessage();
  }

  free(req_packet);
  return status;
}

Status ControlChannel::ProcessAddNodeResponse(NodeInfo* node_info, Packet* rsp_packet) {
  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessHeartbeatRequest(NodeInfo* node_info, Packet* req_packet) {
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
    KLLM_LOG_ERROR << "ControlChannel process heartbeat reqeust error, send packet failed, info:"
                   << status.GetMessage();
  }

  free(req_packet);
  return Status();
}

Status ControlChannel::ProcessHeartbeatResponse(NodeInfo* node_info, Packet* rsp_packet) {
  std::unique_lock<std::mutex> lock(mutex_);

  HeartbeatResponse* heartbeat_rsp = reinterpret_cast<HeartbeatResponse*>(rsp_packet->body);
  int node_rank = heartbeat_rsp->node_rank;
  node_heartbeat_timestamp_[node_rank] = GetCurrentTime();

  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessBarrierRequest(NodeInfo* node_info, Packet* req_packet) {
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

Status ControlChannel::ProcessBarrierResponse(NodeInfo* node_info, Packet* rsp_packet) {
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

Status ControlChannel::ProcessLayerRequest(NodeInfo* node_info, Packet* req_packet) {
  PipelineConfig pipeline_config;
  env_->GetPipelineConfig(pipeline_config);

  AllocateLayerRequest* layer_req = reinterpret_cast<AllocateLayerRequest*>(req_packet->body);

  // update pipeline config.
  pipeline_config.lower_layer_idx = layer_req->lower_layer_idx;
  pipeline_config.upper_layer_idx = layer_req->upper_layer_idx;
  pipeline_config.lower_nextn_layer_idx = layer_req->lower_nextn_layer_idx;
  pipeline_config.upper_nextn_layer_idx = layer_req->upper_nextn_layer_idx;
  pipeline_config.downstream_host = layer_req->downstream_host;
  pipeline_config.downstream_port = layer_req->downstream_port;
  memcpy(pipeline_config.nccl_unique_id, layer_req->nccl_unique_id, sizeof(layer_req->nccl_unique_id));
  env_->SetPipelineConfig(pipeline_config);

  {
    std::unique_lock<std::mutex> lock(mutex_);
    layer_allocated_ = true;
    layer_allocation_cv_.notify_all();
  }

  // Send response.
  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_LAYER, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessLayerRequest allocate memory error.");
  }

  Status status = raw_socket_->Send({master_host_, master_port_}, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel process allocate layer reqeust error, send packet failed, info:"
                   << status.GetMessage();
  }

  free(req_packet);
  return status;
}

Status ControlChannel::ProcessLayerResponse(NodeInfo* node_info, Packet* rsp_packet) {
  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessBlockNumRequest(NodeInfo* node_info, Packet* req_packet) {
  CacheBlockNumRequest* block_req = reinterpret_cast<CacheBlockNumRequest*>(req_packet->body);

  int node_rank = block_req->node_rank;
  size_t device_block_num = block_req->device_block_num;
  size_t host_block_num = block_req->host_block_num;
  rank_cach_block_num_[node_rank] = std::make_pair(device_block_num, host_block_num);

  if (rank_cach_block_num_.size() == world_size_) {
    std::unique_lock<std::mutex> lock(mutex_);
    block_num_cv_.notify_all();
  }

  free(req_packet);
  return Status();
}

Status ControlChannel::ProcessBlockNumResponse(NodeInfo* node_info, Packet* rsp_packet) {
  CacheBlockNumResponse* block_rsp = reinterpret_cast<CacheBlockNumResponse*>(rsp_packet->body);

  // Set pipeline config.
  PipelineConfig pipeline_config;
  env_->GetPipelineConfig(pipeline_config);

  pipeline_config.device_block_num = block_rsp->device_block_num;
  pipeline_config.host_block_num = block_rsp->host_block_num;
  env_->SetPipelineConfig(pipeline_config);

  {
    std::unique_lock<std::mutex> lock(mutex_);
    block_num_synchronized_ = true;
    block_num_cv_.notify_all();
  }

  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessScheduleRequest(NodeInfo* node_info, Packet* req_packet) {
  ScheduleOutput* schedule_output = schedule_output_pool_->GetScheduleOutput();
  if (schedule_output == nullptr) {
    return Status(RET_SERVICE_TERMINATED);
  }

  ScheduleOutputParser::DeserializeScheduleOutput(reinterpret_cast<void*>(req_packet->body), schedule_output);
  schedule_output_pool_->PutToRecvQueue(schedule_output);

  free(req_packet);
  return Status();
}

Status ControlChannel::ProcessScheduleResponse(NodeInfo* node_info, Packet* rsp_packet) {
  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessShutdownRequest(NodeInfo* node_info, Packet* req_packet) {
  // Send response.
  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_SHUTDOWN, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessShutdownRequest allocate memory error.");
  }

  Status status = raw_socket_->Send({master_host_, master_port_}, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel process shutdown reqeust error, send packet failed, info:" << status.GetMessage();
  }

  GetServiceLifetimeManager()->ShutdownService();

  free(req_packet);
  return status;
}

Status ControlChannel::ProcessShutdownResponse(NodeInfo* node_info, Packet* rsp_packet) {
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

Status ControlChannel::Barrier() {
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

Status ControlChannel::Frozen() { return raw_socket_->Frozen(); }

Status ControlChannel::AddNode() {
  PipelineConfig pipeline_config;
  env_->GetPipelineConfig(pipeline_config);

  Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_ADD_NODE, 0);
  if (req_packet == nullptr) {
    throw std::runtime_error("ControlChannel::AddNode allocate memory error.");
  }

  AddNodeRequest* add_node_req = reinterpret_cast<AddNodeRequest*>(req_packet->body);
  add_node_req->node_rank = node_rank_;

  strcpy(add_node_req->data_host, pipeline_config.data_host.c_str());
  add_node_req->data_port = pipeline_config.data_port;

  KLLM_LOG_INFO << "ControlChannel add node " << node_rank_ << ", data endpoint " << add_node_req->data_host << ":"
                << add_node_req->data_port;
  Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
  free(req_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel add node error, send packet failed, info:" << status.GetMessage();
  }

  return status;
}

std::map<int, std::pair<int, int>> ControlChannel::GenerateLayerDistribution(int num_layer,
                                                                             size_t master_offload_layer_num) {
  std::map<int, std::pair<int, int>> layer_distribution;

  // Fast path for single node case
  if (world_size_ <= 1) {
    layer_distribution[0] = std::make_pair(0, num_layer - 1);
    return layer_distribution;
  }

  int quotient = num_layer / world_size_;
  int remainder = num_layer % world_size_;

  // Calculate how many layers the master node has
  int master_layers = quotient;  // reminders are assigned to workers

  // If offloading layers, assert that we're not trying to offload more layers than the master has (minus 1)
  if (master_offload_layer_num > 0) {
    assert(master_offload_layer_num < static_cast<size_t>(master_layers) &&
           "Cannot offload more layers than the master node has (minus 1)");
  }

  // Set master node's range (with or without offloading)
  master_layers = master_layers - static_cast<int>(master_offload_layer_num);
  layer_distribution[0] = std::make_pair(0, master_layers - 1);

  // Distribute layers to worker nodes
  int current_upper_layer = master_layers - 1;

  // Recompute layer distribution for remaining nodes
  size_t remaining_worker_nodes = world_size_ - 1;
  int num_layer_left = num_layer - master_layers;
  quotient = num_layer_left / remaining_worker_nodes;
  remainder = num_layer_left % remaining_worker_nodes;

  for (size_t node_rank = 1; node_rank < world_size_; ++node_rank) {
    int remainder_pad = 0;
    if (remainder > 0) {
      remainder_pad++;
      remainder--;
    }

    const int lower_layer_idx = current_upper_layer + 1;
    const int upper_layer_idx = lower_layer_idx + quotient - 1 + remainder_pad;
    current_upper_layer = upper_layer_idx;

    layer_distribution[node_rank] = std::make_pair(lower_layer_idx, upper_layer_idx);
  }

  return layer_distribution;
}

Status ControlChannel::SynchronizeNodeLayers(size_t master_offload_layer_num) {
  if (node_rank_ != 0) {
    // for worker node, wait master response
    std::unique_lock<std::mutex> lock(mutex_);

    layer_allocation_cv_.wait(lock, [this]() -> bool { return layer_allocated_; });
    return Status();
  }

  // For master node.
  ModelConfig model_config;
  Status status = env_->GetModelConfig(model_config);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "SynchronizeNodeLayers failed. status: " << status.ToString();
    return status;
  }

  // Generate layer distribution for all nodes
  auto layer_distribution = GenerateLayerDistribution(model_config.num_layer, master_offload_layer_num);

  PipelineConfig pipeline_config;
  env_->GetPipelineConfig(pipeline_config);

  // update master pipeline config.
  pipeline_config.lower_layer_idx = layer_distribution[0].first;
  pipeline_config.upper_layer_idx = layer_distribution[0].second;

  // only master node do nextn predict
  RuntimeConfig runtime_config;
  env_->GetRuntimeConfig(runtime_config);
  if (model_config.num_nextn_predict_layers > 0 && runtime_config.mtp_step_num > 0) {
    if (model_config.num_nextn_predict_layers <= runtime_config.mtp_step_num) {
      pipeline_config.lower_nextn_layer_idx = model_config.num_layer;
      pipeline_config.upper_nextn_layer_idx = model_config.num_layer + model_config.num_nextn_predict_layers - 1;
    } else if (model_config.num_nextn_predict_layers == 1) {
      // repeat MTP layer only support exist 1 MTP layer weight
      pipeline_config.upper_nextn_layer_idx += runtime_config.mtp_step_num - 1;
      KLLM_LOG_INFO << "repeat MTP layer " << runtime_config.mtp_step_num - 1 << " times";
    } else {
      KLLM_LOG_FATAL << fmt::format(
          "mtp_step_num({}) should <= num_nextn_predict_layers({}) in model config while num_nextn_predict_layers != 1",
          runtime_config.mtp_step_num, model_config.num_nextn_predict_layers);
    }
  }

  pipeline_config.downstream_host = rank_data_nodes_[1].host;
  pipeline_config.downstream_port = rank_data_nodes_[1].port;
  env_->SetPipelineConfig(pipeline_config);

  KLLM_LOG_INFO << "ControlChannel set master node " << node_rank_ << ", data downstream endpoint "
                << pipeline_config.downstream_host << ":" << pipeline_config.downstream_port << ", normal layer range ["
                << pipeline_config.lower_layer_idx << ", " << pipeline_config.upper_layer_idx
                << "], nextn layer range [" << pipeline_config.lower_nextn_layer_idx << ", "
                << pipeline_config.upper_nextn_layer_idx << "].";

  // Send to every worker node.
  for (size_t node_rank = 1; node_rank < world_size_; ++node_rank) {
    Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_LAYER, 0);
    if (req_packet == nullptr) {
      throw std::runtime_error("ControlChannel::SynchronizeNodeLayers allocate memory error.");
    }

    AllocateLayerRequest* layer_req = reinterpret_cast<AllocateLayerRequest*>(req_packet->body);
    layer_req->lower_layer_idx = layer_distribution[node_rank].first;
    layer_req->upper_layer_idx = layer_distribution[node_rank].second;
    // only master node do nextn predict
    layer_req->lower_nextn_layer_idx = -1;
    layer_req->upper_nextn_layer_idx = -1;

    // post-data_node
    if (node_rank == world_size_ - 1) {
      strcpy(layer_req->downstream_host, pipeline_config.data_host.c_str());
      layer_req->downstream_port = pipeline_config.data_port;
    } else {
      strcpy(layer_req->downstream_host, rank_data_nodes_[node_rank + 1].host.c_str());
      layer_req->downstream_port = rank_data_nodes_[node_rank + 1].port;
    }

    // Broadcast nccl unique_id for all nodes.
    memcpy(layer_req->nccl_unique_id, pipeline_config.nccl_unique_id, sizeof(pipeline_config.nccl_unique_id));
    KLLM_LOG_INFO << "ControlChannel set worker node " << node_rank << ", data downstream endpoint "
                  << layer_req->downstream_host << ":" << layer_req->downstream_port << ", layer range ["
                  << layer_distribution[node_rank].first << ", " << layer_distribution[node_rank].second << "].";
    Status status = raw_socket_->Send(rank_nodes_[node_rank], req_packet);
    free(req_packet);

    if (!status.OK()) {
      KLLM_LOG_ERROR << "ControlChannel sync node layers error, send packet failed, info:" << status.GetMessage();
    }
  }

  return Status();
}

Status ControlChannel::SynchronizeCacheBlockNum() {
  size_t device_block_num = env_->GetTotalDeviceBlockNum();
  size_t host_block_num = env_->GetTotalHostBlockNum();

  // For master
  if (node_rank_ == 0) {
    rank_cach_block_num_[0] = std::make_pair(device_block_num, host_block_num);

    // Wait for all workers.
    std::unique_lock<std::mutex> lock(mutex_);

    block_num_cv_.wait(lock, [this]() -> bool { return rank_cach_block_num_.size() == world_size_; });

    // Get minimum block num.
    size_t real_device_block_num = std::numeric_limits<size_t>::max();
    size_t real_host_block_num = std::numeric_limits<size_t>::max();
    size_t i = 0;
    std::stringstream ss;
    for (auto pair : rank_cach_block_num_) {
      if (real_device_block_num > pair.second.first) {
        real_device_block_num = pair.second.first;
      }
      if (real_host_block_num > pair.second.second) {
        real_host_block_num = pair.second.second;
      }
      ss << "[" << i << "] device_block_num=" << pair.second.first << ", host_block_num=" << pair.second.second;
      i++;
    }
    KLLM_LOG_INFO << "Reset block nums after sync. real_device_block_num=" << real_device_block_num
                  << ", real_host_block_num=" << real_host_block_num << ", node block infos: " << ss.str();

    // Set master config.
    PipelineConfig pipeline_config;
    env_->GetPipelineConfig(pipeline_config);

    pipeline_config.device_block_num = real_device_block_num;
    pipeline_config.host_block_num = real_host_block_num;
    env_->SetPipelineConfig(pipeline_config);

    // Send response to all nodes.
    for (auto it = node_ranks_.begin(); it != node_ranks_.end(); ++it) {
      NodeInfo node_info = it->first;

      Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_BLOCKNUM, 0);
      if (rsp_packet == nullptr) {
        throw std::runtime_error("ControlChannel::SynchronizeCacheBlockNum allocate memory error.");
      }

      CacheBlockNumResponse* block_rsp = reinterpret_cast<CacheBlockNumResponse*>(rsp_packet->body);
      block_rsp->device_block_num = real_device_block_num;
      block_rsp->host_block_num = real_host_block_num;

      Status status = raw_socket_->Send(node_info, rsp_packet);
      free(rsp_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel synchronize cache block num error, send packet failed, info:"
                       << status.GetMessage();
        return status;
      }
    }
  } else {
    // For worker
    Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_BLOCKNUM);
    if (req_packet == nullptr) {
      throw std::runtime_error("ControlChannel::SynchronizeCacheBlockNum allocate memory error.");
    }

    CacheBlockNumRequest* block_req = reinterpret_cast<CacheBlockNumRequest*>(req_packet->body);
    block_req->device_block_num = device_block_num;
    block_req->host_block_num = host_block_num;
    block_req->node_rank = node_rank_;

    Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
    free(req_packet);

    if (!status.OK()) {
      KLLM_LOG_ERROR << "ControlChannel synchronize cache block num error, send packet failed, info:"
                     << status.GetMessage();
      return status;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    block_num_cv_.wait(lock, [this]() -> bool { return block_num_synchronized_; });
  }

  return Status();
}

Status ControlChannel::ShutdownCluster() {
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

Status ControlChannel::HandleServerPacket(NodeInfo* node_info, Packet* packet) {
  switch (packet->type) {
    case PacketType::CONTROL_REQ_ADD_NODE: {
      return ProcessAddNodeRequest(node_info, packet);
    }
    case PacketType::CONTROL_REQ_BARRIER: {
      return ProcessBarrierRequest(node_info, packet);
    }
    case PacketType::CONTROL_RSP_LAYER: {
      return ProcessLayerResponse(node_info, packet);
    }
    case PacketType::CONTROL_REQ_BLOCKNUM: {
      return ProcessBlockNumRequest(node_info, packet);
    }
    case PacketType::CONTROL_RSP_SCHEDULE: {
      return ProcessScheduleResponse(node_info, packet);
    }
    case PacketType::CONTROL_RSP_SHUTDOWN: {
      return ProcessShutdownResponse(node_info, packet);
    }
    case PacketType::CONTROL_RSP_DEEPEP_META: {
      return ProcessNvshmemUniqueIdResponse(node_info, packet);
    }
    case PacketType::CONTROL_REQ_DEEPEP_META: {
      return ProcessNvshmemUniqueIdRequest(node_info, packet);
    }
    default: {
      KLLM_LOG_ERROR << "Not supported packet type:" << packet->type;
      return Status(RET_RUNTIME_FAILED, FormatStr("Not supported packet type %d", packet->type));
    }
  }

  return Status();
}

Status ControlChannel::HandleClientPacket(NodeInfo* node_info, Packet* packet) {
  switch (packet->type) {
    case PacketType::CONTROL_RSP_ADD_NODE: {
      return ProcessAddNodeResponse(node_info, packet);
    }
    case PacketType::CONTROL_RSP_BARRIER: {
      return ProcessBarrierResponse(node_info, packet);
    }
    case PacketType::CONTROL_REQ_LAYER: {
      return ProcessLayerRequest(node_info, packet);
    }
    case PacketType::CONTROL_RSP_BLOCKNUM: {
      return ProcessBlockNumResponse(node_info, packet);
    }
    case PacketType::CONTROL_REQ_SCHEDULE: {
      return ProcessScheduleRequest(node_info, packet);
    }
    case PacketType::CONTROL_REQ_SHUTDOWN: {
      return ProcessShutdownRequest(node_info, packet);
    }
    case PacketType::CONTROL_REQ_DEEPEP_META: {
      return ProcessNvshmemUniqueIdRequest(node_info, packet);
    }
    case PacketType::CONTROL_RSP_DEEPEP_META: {
      return ProcessNvshmemUniqueIdResponse(node_info, packet);
    }
    default: {
      KLLM_LOG_ERROR << "Not supported packet type:" << packet->type;
      return Status(RET_RUNTIME_FAILED, FormatStr("Not supported packet type %d", packet->type));
    }
  }

  return Status();
}

}  // namespace ksana_llm
