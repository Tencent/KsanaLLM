/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include "ksana_llm/data_hub/expert_parallel_deepep_wrapper.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/distributed/control_message.h"
#include "ksana_llm/distributed/node_info.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/distributed/raw_socket.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"

#include "ksana_llm/distributed/control_channel.h"

namespace ksana_llm {
class ExpertParallelControlChannel : public ControlChannel {
 public:
  ExpertParallelControlChannel(const std::string& master_host, uint16_t master_port, size_t world_size, int node_rank,
                               size_t global_expert_para_size, PacketCreationFunc packet_creation_fn = GetPacketObject,
                               ScheduleOutputPool* schedule_output_pool = nullptr,
                               std::shared_ptr<Environment> env = nullptr);
  ~ExpertParallelControlChannel();
  // For master node only.
  Status Listen();

  // Close open port.
  Status Close();

  // For slave node only.
  Status Connect();

  // disconnect from master.
  Status Disconnect();

  // Wait until all nodes arrive same location.
  Status Barrier();

  // Stop to accept any new connection.
  Status Frozen();

  // Add node to cluster.
  Status AddNode();

  // Shutdown the pipeline cluster.
  Status ShutdownCluster();

  // Exchange Nvshmem unique-ID among Expert-Parallel nodes.
  Status SynchronizeNvshmemUniqueId();

  // Set the deepep_wrapper_
  Status SetExpertParallelDeepepWrapper(const std::shared_ptr<ExpertParallelDeepepWrapper>& deepep_wrapper);

 private:
  // Add node.
  virtual Status ProcessAddNodeRequest(NodeInfo* node_info, Packet* req_packet) override;
  virtual Status ProcessAddNodeResponse(NodeInfo* node_info, Packet* rsp_packet) override;

  // heartbeat
  virtual Status ProcessHeartbeatRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessHeartbeatResponse(NodeInfo* node_info, Packet* rsp_packet);

  // Barrier
  virtual Status ProcessBarrierRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessBarrierResponse(NodeInfo* node_info, Packet* rsp_packet);

  // Process shutdown message.
  virtual Status ProcessShutdownRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessShutdownResponse(NodeInfo* node_info, Packet* rsp_packet);

  // heartbeat thread handle.
  virtual Status ProcessHeartbeatLoop();

  // send schedule output to workers.
  // virtual Status ProcessSendScheduleOutputLoop();
  virtual Status ProcessNvshmemUniqueIdRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessNvshmemUniqueIdResponse(NodeInfo* node_info, Packet* rsp_packet);

 private:
  std::shared_ptr<RawSocket> raw_socket_ = nullptr;

  // The environments.
  std::shared_ptr<Environment> env_ = nullptr;

  std::string master_host_;
  uint16_t master_port_;

  size_t world_size_;
  int node_rank_;
  size_t global_expert_para_size_;

  // Used for barrier.
  int barrier_clock_idx_ = 0;

  // Whether the control channl is terminated..
  bool terminated_ = false;

  // The node ranks that has sent barrier message.
  std::unordered_map<int, std::unordered_set<int>> barrier_req_ranks_;

  bool layer_allocated_ = false;
  bool block_num_synchronized_ = false;

  // The barrier clocks that has checked by master.
  std::unordered_set<int> barrier_rsp_clocks_;

  // The shutdown nodes.
  std::unordered_set<int> shutdown_nodes_;

  // Notify for barrier utility.
  std::condition_variable barrier_cv_;
  std::condition_variable shutdown_cv_;
  std::condition_variable block_num_cv_;

  std::condition_variable nvshmem_unique_id_cv_;
  bool nvshmem_unique_id_synchronized_ = false;

  std::condition_variable ipc_handles_cv_;
  bool ipc_handles_synchronized_ = false;

  // rank to nodes and vice versa.
  std::unordered_map<int, NodeInfo> rank_nodes_;
  std::unordered_map<NodeInfo, int, NodeInfoHash, NodeInfoEqual> node_ranks_;

  // The heartbeat thread & timeout.
  std::shared_ptr<std::thread> heartbeat_thread_ = nullptr;
  size_t heartbeat_timeout_secs_ = 120;
  size_t heartbeat_interval_secs_ = 30;

  // rank to timestamp.
  std::unordered_map<int, time_t> node_heartbeat_timestamp_;

  std::unordered_map<int, std::unique_ptr<char[]>> node_ipc_handles_;

  // Protect multi-thread receive handles.
  std::mutex mutex_;

  // Unique DeepEP wrapper for each node in Expert-Parallel.
  std::shared_ptr<ExpertParallelDeepepWrapper> deepep_wrapper_ = nullptr;
};

}  // namespace ksana_llm
