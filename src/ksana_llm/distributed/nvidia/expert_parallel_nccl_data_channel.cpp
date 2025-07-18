/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/distributed/nvidia/expert_parallel_nccl_data_channel.h"

#include <nccl.h>

#include <cstring>
#include <ios>

#include "cuda_runtime.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/utils/barrier.h"
#include "ksana_llm/utils/expert_parallel_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/socket_util.h"

namespace ksana_llm {

ExpertParallelNcclDataChannel::ExpertParallelNcclDataChannel(
    ExpertParallelHiddenUnitBufferPool* expert_hidden_unit_buffer_pool, std::shared_ptr<Environment> env,
    std::shared_ptr<Context> context) {
  env_ = env ? env : Singleton<Environment>::GetInstance();
  context_ = context;
  hidden_unit_buffer_pool_ = expert_hidden_unit_buffer_pool;

  // Each node has its own nccl send-comm-group.
  send_thread_ =
      std::unique_ptr<std::thread>(new std::thread(&ExpertParallelNcclDataChannel::ProcessSendTwiceLoop, this));

  // Each node has independent recv thread for every send-comm-group.
  size_t world_size = context_->GetExpertParallelWorldSize();
  size_t cur_rank = context_->GetExpertParallelExpertNodeRank();
  recv_thread_.resize(world_size);
  for (int i = 0; i < world_size; i++) {
    if (i != cur_rank)
      recv_thread_[i] =
          std::unique_ptr<std::thread>(new std::thread(&ExpertParallelNcclDataChannel::ProcessRecvTwiceLoop, this, i));
  }

  KLLM_LOG_INFO << "ExpertParallelNcclDataChannel() succeed\n";
}

ExpertParallelNcclDataChannel::~ExpertParallelNcclDataChannel() {
  terminated_ = true;
  hidden_unit_buffer_pool_->Stop();

  if (send_thread_) {
    send_thread_->join();
  }

  if (!recv_thread_.empty()) {
    for (int i = 0; i < context_->GetExpertParallelWorldSize(); i++)
      if (i != context_->GetExpertParallelExpertNodeRank()) {
        recv_thread_[i]->join();
        recv_thread_.clear();
      }
  }
}

// Master node.
Status ExpertParallelNcclDataChannel::Listen() {
  KLLM_LOG_INFO << "ExpertParallelNcclDataChannel::Listen()";

  std::string interface;
  Status status = GetAvailableInterfaceAndIP(interface, data_host_);
  if (!status.OK()) {
    throw std::runtime_error(fmt::format("Get data ip error: {}", status.GetMessage()));
  }

  status = GetAvailablePort(data_port_);
  if (!status.OK()) {
    throw std::runtime_error(fmt::format("Get data port error: {}", status.GetMessage()));
  }

  env_->GetExpertParallelConfig(expert_parallel_config_);
  expert_parallel_config_.data_host = data_host_;
  expert_parallel_config_.data_port = data_port_;
  env_->SetExpertParallelConfig(expert_parallel_config_);

  // Skip if not master node.
  if (expert_parallel_config_.expert_node_rank > 0) {
    return Status();
  }

  // Create unique_id on master node only.
  expert_parallel_config_.nccl_unique_ids.resize(expert_parallel_config_.expert_world_size);
  nccl_unique_ids_.resize(expert_parallel_config_.expert_world_size);
  for (int i = 0; i < expert_parallel_config_.expert_world_size; i++) {
    NCCL_CHECK(ncclGetUniqueId(&nccl_unique_ids_[i]));
    KLLM_LOG_INFO << "Create unique_id, id: " << i << ", nccl_unique_ids"
                  << reinterpret_cast<char*>(&nccl_unique_ids_[i]);
    memcpy(expert_parallel_config_.nccl_unique_ids[i].data(), reinterpret_cast<const char*>(&nccl_unique_ids_[i]),
           sizeof(ncclUniqueId));
  }
  // NCCL_CHECK(ncclGetUniqueId(&nccl_unique_id_));
  // memcpy(expert_parallel_config_.nccl_unique_id, reinterpret_cast<const char*>(&nccl_unique_id_),
  // sizeof(ncclUniqueId));

  env_->SetExpertParallelConfig(expert_parallel_config_);

  return Status();
}

Status ExpertParallelNcclDataChannel::Close() {
  // do nothing.
  return Status();
}

//    All expert parallel worker nodes.
Status ExpertParallelNcclDataChannel::Connect() {
  env_->GetExpertParallelConfig(expert_parallel_config_);

  // Deserialize nccl unique id from master.
  memcpy(&nccl_unique_id_, expert_parallel_config_.nccl_unique_id, sizeof(ncclUniqueId));
  KLLM_LOG_INFO << "ExpertParallelNcclDataChannel::Connect nccl_unique_id " << expert_parallel_config_.nccl_unique_id
                << std::endl;

  int expert_parallel_world_size = expert_parallel_config_.expert_world_size;
  int expert_node_rank = expert_parallel_config_.expert_node_rank;
  int expert_parallel_size = expert_parallel_config_.expert_para_size;
  size_t global_expert_parallel_size = expert_parallel_config_.global_expert_para_size;
  size_t expert_tensor_para_size = expert_parallel_config_.expert_tensor_para_size;
  int tp_size = context_->GetTensorParallelSize();

  KLLM_LOG_INFO << "Initialize nccl communicators, expert_parallel_world_size:" << expert_parallel_world_size
                << ", global_expert_para_size:" << global_expert_parallel_size << ", node_rank:" << expert_node_rank
                << ", tp_size:" << tp_size;

  for (int i = 0; i < expert_parallel_world_size; i++) {
    KLLM_LOG_INFO << "ExpertParallelNcclDataChannel::Connect, node_rank i: " << i << ", nccl_unique_id "
                  << reinterpret_cast<char*>(expert_parallel_config_.nccl_unique_ids[i].data());
  }

  // Every ep rank communicate, to support pp later. @xingjinglu
  communicators_.resize(expert_parallel_world_size);
  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < expert_parallel_world_size; i++) {
    communicators_[i].resize(tp_size);
    ncclUniqueId nccl_unique_id;
    memcpy(&nccl_unique_id, expert_parallel_config_.nccl_unique_ids[i].data(), sizeof(ncclUniqueId));

    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&nccl_unique_id);
    std::cout << "nccl_unique_id rank: " << i;

    // NCCL_UNIQUE_ID_BYTES通常是128字节
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; i++) {
      std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]) << " ";
    }
    std::cout << std::endl;

    for (int dev_id = 0; dev_id < tp_size; ++dev_id) {
      CUDA_CHECK(cudaSetDevice(dev_id));
      int cur_rank = expert_node_rank * tp_size + dev_id;

      KLLM_LOG_INFO << "Initialize nccl communicator: node_rank: " << expert_node_rank << ", dev_id:" << dev_id
                    << ", cur rank:" << cur_rank;

      NCCL_CHECK(
          ncclCommInitRank(&communicators_[i][dev_id], expert_parallel_world_size * tp_size, nccl_unique_id, cur_rank));
    }
  }
  NCCL_CHECK(ncclGroupEnd());

  streams_.resize(tp_size);
  for (int dev_id = 0; dev_id < tp_size; ++dev_id) {
    CUDA_CHECK(cudaSetDevice(dev_id));
    CUDA_CHECK(cudaStreamCreate(&streams_[dev_id]));
  }

  // Build upstream and downstream ranks.
  int upstream_node_rank = expert_node_rank - 1;
  if (upstream_node_rank < 0) {
    upstream_node_rank = expert_parallel_world_size - 1;
  }

  int downstream_node_rank = expert_node_rank + 1;
  if (downstream_node_rank > expert_parallel_world_size - 1) {
    downstream_node_rank = 0;
  }

  for (int dev_id = 0; dev_id < tp_size; ++dev_id) {
    upstream_ranks_.push_back(upstream_node_rank * tp_size + dev_id);
    downstream_ranks_.push_back(downstream_node_rank * tp_size + dev_id);
  }
  KLLM_LOG_INFO << "Initialize nccl upstream node:" << upstream_node_rank
                << ", upstream ranks:" << Vector2Str(upstream_ranks_);
  KLLM_LOG_INFO << "Initialize nccl downstream node:" << downstream_node_rank
                << ", downstream ranks:" << Vector2Str(downstream_ranks_);

  // Build src and dst ranks. Redundant communication can be reduced later.
  // TBD @xingjinglu
  for (int dev_id = 0; dev_id < expert_parallel_size; ++dev_id) {
    for (int node_id = 0; node_id < expert_parallel_world_size; node_id++) {
      if (expert_node_rank != node_id) {
        src_ranks_.push_back(node_id * expert_parallel_size + dev_id);
        dst_ranks_.push_back(node_id * expert_parallel_size + dev_id);
      }
    }
  }

  return Status();
}

Status ExpertParallelNcclDataChannel::Disconnect() {
  for (auto& row : communicators_) {
    for (auto& comm : row) {
      NCCL_CHECK(ncclCommDestroy(comm));
    }
  }

  return Status();
}

Status ExpertParallelNcclDataChannel::Frozen() {
  KLLM_LOG_INFO << "ExpertParallelNcclDataChannel skip frozen." << std::endl;
  return Status();
}

Status ExpertParallelNcclDataChannel::ProcessRecvLoop(int node_rank) {
  while (!terminated_) {
    // Wait util recv invoked.
    hidden_unit_buffer_pool_->WaitUtilReadyToRecv();

    if (hidden_unit_buffer_pool_->Stopped()) {
      KLLM_LOG_WARNING << "ProcessRecvLoop hidden unit buffer pool stopped, break..";
      break;
    }

    // Recv comm meta.
    HiddenUnitDeviceBuffer* hidden_unit = hidden_unit_buffer_pool_->GetCommMetaDeviceBufferSingle();
    if (!hidden_unit) {
      KLLM_LOG_ERROR << "ProcessRecvLoop empty hidden unit from host send queue, break..";
      break;
    }

    Status status = ProcessRecvDeviceBuffer(hidden_unit, node_rank);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "ProcessRecvLoop send data failed, info:" << status.GetMessage();
    } else {
      KLLM_LOG_DEBUG << "ProcessRecvLoop succeed";
    }

    // Put to device receive queue.
    hidden_unit_buffer_pool_->PutToDeviceRecvQueue(hidden_unit);
  }

  return Status();
}

Status ExpertParallelNcclDataChannel::ProcessRecvTwiceLoop(int node_rank) {
  while (!terminated_) {
    // Wait util recv invoked.
    hidden_unit_buffer_pool_->WaitUtilReadyToRecv();

    if (hidden_unit_buffer_pool_->Stopped()) {
      KLLM_LOG_WARNING << "ProcessRecvLoop hidden unit buffer pool stopped, break..";
      break;
    }

    // Stage 1. Recv comm meta
    HiddenUnitDeviceBuffer* hidden_unit = hidden_unit_buffer_pool_->GetCommMetaDeviceBufferSingle();
    if (!hidden_unit) {
      KLLM_LOG_ERROR << "ProcessRecvTwiceLoop empty hidden unit from host send queue, break..";
      break;
    }
    for (int dev_id = 0; dev_id < hidden_unit->tensors.size(); dev_id++) {
      hidden_unit->tensors[dev_id].shape = {sizeof(expert_parallel_comm_meta), 1};
      hidden_unit->tensors[dev_id].dtype = DataType::TYPE_UINT8;
    }

    Status status = ProcessRecvDeviceBuffer(hidden_unit, node_rank);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "ProcessRecvTwiceLoop recv comm meta failed, info:" << status.GetMessage();
    } else {
      KLLM_LOG_DEBUG << "ProcessRecvTwiceLoop recv comm meta succeed";
    }

    // Parse comm meta.
    expert_parallel_comm_meta meta;
    CopyHiddenUnitBufferToHostMem(reinterpret_cast<char*>(&meta), hidden_unit, {sizeof(expert_parallel_comm_meta), 1},
                                  DataType::TYPE_UINT8, 0, false);
    hidden_unit_buffer_pool_->FreeCommMetaDeviceBuffer(hidden_unit);

    // Stage 2. Recv real data.
    HiddenUnitDeviceBuffer* hidden_unit_data = hidden_unit_buffer_pool_->GetDeviceBufferSingle();
    if (!hidden_unit_data) {
      KLLM_LOG_ERROR << "ProcessRecvTwiceLoop empty hidden unit from host send queue, break..";
      break;
    }

    for (int dev_id = 0; dev_id < hidden_unit->tensors.size(); dev_id++) {
      hidden_unit_data->tensors[dev_id].shape = {meta.shape_0, meta.shape_1};
      // cudaMemset(hidden_unit_data->tensors[dev_id].GetPtr<void>(), 0,
      //             hidden_unit_data->tensors[dev_id].GetTotalBytes());
    }

    KLLM_LOG_DEBUG << fmt::format("ProcessRecvTwiceLoop meta shape: {} {}", meta.shape_0, meta.shape_1);

    status = ProcessRecvDeviceBuffer(hidden_unit_data, node_rank);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "ProcessRecvTwiceLoop recv data failed, info:" << status.GetMessage();
    } else {
      KLLM_LOG_DEBUG << "ProcessRecvTwiceLoop recv real data succeed";
    }

    // Put to device receive queue.
    hidden_unit_buffer_pool_->PutToDeviceRecvQueue(hidden_unit_data);
  }

  return Status();
}

Status ExpertParallelNcclDataChannel::ProcessSendLoop() {
  while (!terminated_) {
    // Blocked, waiting util packet is ready.
    HiddenUnitDeviceBuffer* hidden_unit = hidden_unit_buffer_pool_->GetFromSendQueue();
    if (!hidden_unit) {
      KLLM_LOG_WARNING << "ProcessSendLoop empty hidden_unit from device send "
                          "queue, break..";
      break;
    }

    Status status = ProcessSendDeviceBuffer(hidden_unit);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "ProcessSendLoop send data failed, info:" << status.GetMessage();
    } else {
      KLLM_LOG_DEBUG << "ProcessSendLoop succeed";
    }

    // Notify that send operation finished.
    hidden_unit_buffer_pool_->NotifySendFinished();
  }

  return Status();
}

Status ExpertParallelNcclDataChannel::ProcessSendTwiceLoop() {
  while (!terminated_) {
    // Stage 1. Send comm meta.
    // Blocked, waiting util packet is ready.
    HiddenUnitDeviceBuffer* hidden_unit = hidden_unit_buffer_pool_->GetFromSendQueue();
    if (!hidden_unit) {
      KLLM_LOG_WARNING << "ProcessSendLoop empty hidden_unit from device send "
                          "queue, break..";
      break;
    }

    Status status = ProcessSendDeviceBuffer(hidden_unit);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "ProcessSendTwiceLoop send comm meta failed, info:" << status.GetMessage();
    } else {
      KLLM_LOG_DEBUG << "ProcessSendTwiceLoop send comm meta succeed";
    }

    hidden_unit_buffer_pool_->FreeCommMetaDeviceBuffer(hidden_unit);

    // Stage 2. Send real data.
    hidden_unit = hidden_unit_buffer_pool_->GetFromSendQueue();
    if (!hidden_unit) {
      KLLM_LOG_WARNING << "ProcessSendTwiceLoop empty hidden_unit from device send "
                          "queue, break..";
      break;
    }

    status = ProcessSendDeviceBuffer(hidden_unit);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "ProcessSendTwiceLoop send real data failed, info:" << status.GetMessage();
    } else {
      KLLM_LOG_DEBUG << "ProcessSendTwiceLoop send real data succeed";
    }

    hidden_unit_buffer_pool_->FreeDeviceBuffer(hidden_unit);

    // Notify that send operation finished.
    hidden_unit_buffer_pool_->NotifySendFinished();
  }

  return Status();
}

//
Status ExpertParallelNcclDataChannel::ProcessRecvDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit, int node_rank) {
  size_t rank = expert_parallel_config_.expert_node_rank;

  ncclGroupStart();
  for (size_t dev_id = 0; dev_id < hidden_unit->tensors.size(); ++dev_id) {
    CUDA_CHECK(cudaSetDevice(dev_id));

    DataType data_type = hidden_unit->tensors[dev_id].dtype;
    std::vector<size_t> shape = hidden_unit->tensors[dev_id].shape;

    ncclDataType_t nccl_dtype;
    GetNcclDataType(data_type, nccl_dtype);

    int64_t count = 1;
    for (auto i : shape) {
      count *= i;
    }

    Tensor& tensor = hidden_unit->tensors[dev_id];
    tensor.shape = shape;
    tensor.dtype = data_type;

    cudaMemset(tensor.GetPtr<void>(), 0, tensor.GetTotalBytes());

    KLLM_LOG_DEBUG << fmt::format(
        "ProcessRecvDeviceBuffer, dev_id: {}, bytes count: {}, comm_type: {}, send_comm_rank: {}, ref_ptr: {}, "
        "ref_ptr2: {}",
        dev_id, count, static_cast<int>(hidden_unit->comm_type), upstream_ranks_[hidden_unit->scatter_sender_rank],
        tensor.GetPtr<void>(), hidden_unit->tensors[dev_id].GetPtr<void>());

    if (hidden_unit->comm_type == DistributedCommunicationType::SCATTER) {
      NCCL_CHECK_CALL(
          ncclRecv(tensor.GetPtr<void>(), count, nccl_dtype, upstream_ranks_[hidden_unit->scatter_sender_rank],
                   communicators_[node_rank][dev_id], streams_[dev_id]),
          "ncclRecv");

    } else {
      NCCL_CHECK(ncclRecv(tensor.GetPtr<void>(), count, nccl_dtype, upstream_ranks_[dev_id],
                          communicators_[node_rank][dev_id], streams_[dev_id]));
    }
  }
  ncclGroupEnd();

  // Wait finished. Not need?
  for (size_t dev_id = 0; dev_id < hidden_unit->tensors.size(); ++dev_id) {
    CUDA_CHECK(cudaSetDevice(dev_id));
    CUDA_CHECK(cudaStreamSynchronize(streams_[dev_id]));
  }

  return Status();
}

// all-2-all commnunication, refactor later, not implemented now.
Status ExpertParallelNcclDataChannel::ProcessSendDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit) {
  size_t rank = expert_parallel_config_.expert_node_rank;

  ncclGroupStart();
  for (size_t dev_id = 0; dev_id < hidden_unit->tensors.size(); ++dev_id) {
    CUDA_CHECK(cudaSetDevice(dev_id));

    // CUDA_CHECK(cudaStreamSynchronize(context_->GetComputeStreams()[hidden_unit->scatter_sender_rank].Get()));

    DataType data_type = hidden_unit->tensors[dev_id].dtype;
    std::vector<size_t> shape = hidden_unit->tensors[dev_id].shape;

    ncclDataType_t nccl_dtype;
    GetNcclDataType(data_type, nccl_dtype);

    int64_t count = 1;
    for (auto i : shape) {
      count *= i;
    }

    Tensor& tensor = hidden_unit->tensors[dev_id];
    tensor.shape = shape;
    tensor.dtype = data_type;

    if (hidden_unit->comm_type == DistributedCommunicationType::SCATTER) {
      NCCL_CHECK(
          ncclSend(tensor.GetPtr<void>(), count, nccl_dtype,
                   ((rank + 1) % expert_parallel_config_.expert_world_size) * hidden_unit->tensors.size() + dev_id,
                   communicators_[rank][hidden_unit->scatter_sender_rank], streams_[0]));

    } else {
      NCCL_CHECK(ncclSend(tensor.GetPtr<void>(), count, nccl_dtype, downstream_ranks_[dev_id],
                          communicators_[rank][dev_id], streams_[dev_id]));
    }
  }

  ncclGroupEnd();

  // Wait finished. Not need.
  for (size_t dev_id = 0; dev_id < hidden_unit->tensors.size(); ++dev_id) {
    CUDA_CHECK(cudaSetDevice(dev_id));
    CUDA_CHECK(cudaStreamSynchronize(streams_[dev_id]));
  }

  return Status();
}

Status ExpertParallelNcclDataChannel::GetNcclDataType(DataType dtype, ncclDataType_t& nccl_dtype) {
  switch (dtype) {
    case DataType::TYPE_BYTES:
    case DataType::TYPE_BOOL: {
      nccl_dtype = ncclDataType_t::ncclChar;
      return Status();
    }
    case DataType::TYPE_INT8: {
      nccl_dtype = ncclDataType_t::ncclInt8;
      return Status();
    }
    case DataType::TYPE_UINT8: {
      nccl_dtype = ncclDataType_t::ncclUint8;
      return Status();
    }
    case DataType::TYPE_UINT32: {
      nccl_dtype = ncclDataType_t::ncclUint32;
      return Status();
    }
    case DataType::TYPE_UINT64: {
      nccl_dtype = ncclDataType_t::ncclUint64;
      return Status();
    }
    case DataType::TYPE_INT32: {
      nccl_dtype = ncclDataType_t::ncclInt32;
      return Status();
    }
    case DataType::TYPE_INT64: {
      nccl_dtype = ncclDataType_t::ncclInt64;
      return Status();
    }
    case DataType::TYPE_BF16: {
      nccl_dtype = ncclDataType_t::ncclBfloat16;
      return Status();
    }
    case DataType::TYPE_FP16: {
      nccl_dtype = ncclDataType_t::ncclFloat16;
      return Status();
    }
    case DataType::TYPE_FP32: {
      nccl_dtype = ncclDataType_t::ncclFloat32;
      return Status();
    }
    case DataType::TYPE_FP64: {
      nccl_dtype = ncclDataType_t::ncclFloat64;
      return Status();
    }
    case DataType::TYPE_INT16:
    case DataType::TYPE_UINT16:
    case DataType::TYPE_INVALID:
    case DataType::TYPE_FP8_E4M3:
    case DataType::TYPE_I4_GROUP:
    case DataType::TYPE_FP8_E5M2:
    case DataType::TYPE_VOID:
    case DataType::TYPE_POINTER: {
      return Status(RET_INVALID_ARGUMENT, FormatStr("Not supported dtype %d", dtype));
    }
  }
  return Status();
}

}  // namespace ksana_llm
