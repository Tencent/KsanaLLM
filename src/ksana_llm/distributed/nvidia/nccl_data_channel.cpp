/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/distributed/nvidia/nccl_data_channel.h"

#include <nccl.h>
#include <cstring>
#include <ios>
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

NcclDataChannel::NcclDataChannel(HiddenUnitBufferPool* hidden_unit_buffer_pool, std::shared_ptr<Environment> env,
                                 std::shared_ptr<Context> context) {
  env_ = env ? env : Singleton<Environment>::GetInstance();
  context_ = context;

  hidden_unit_buffer_pool_ = hidden_unit_buffer_pool ? hidden_unit_buffer_pool : GetHiddenUnitBufferPool();

  recv_thread_ = std::unique_ptr<std::thread>(new std::thread(&NcclDataChannel::ProcessRecvLoop, this));

  send_thread_ = std::unique_ptr<std::thread>(new std::thread(&NcclDataChannel::ProcessSendLoop, this));
  KLLM_LOG_DEBUG << "NcclDataChannel() succeed \n";
}

NcclDataChannel::~NcclDataChannel() {
  terminated_ = true;
  hidden_unit_buffer_pool_->Stop();

  if (send_thread_) {
    send_thread_->join();
  }

  if (recv_thread_) {
    recv_thread_->join();
  }
}

Status NcclDataChannel::Listen() {
  env_->GetPipelineConfig(pipeline_config_);

  // Skip if not master node.
  if (pipeline_config_.node_rank > 0) {
    return Status();
  }

  // Create unique_id on master node only.
  NCCL_CHECK(ncclGetUniqueId(&nccl_unique_id_));
  memcpy(pipeline_config_.nccl_unique_id, reinterpret_cast<const char*>(&nccl_unique_id_), sizeof(ncclUniqueId));

  env_->SetPipelineConfig(pipeline_config_);

  return Status();
}

Status NcclDataChannel::Close() {
  // do nothing.
  return Status();
}

Status NcclDataChannel::Connect() {
  env_->GetPipelineConfig(pipeline_config_);

  // Deserialize nccl unique id from master.
  memcpy(&nccl_unique_id_, pipeline_config_.nccl_unique_id, sizeof(ncclUniqueId));

  int pp_size = pipeline_config_.world_size;
  int node_rank = pipeline_config_.node_rank;
  int tp_size = context_->GetTensorParallelSize();
  KLLM_LOG_INFO << "Initialize nccl communicators, pp_size:" << pp_size << ", node_rank:" << node_rank
                << ", tp_size:" << tp_size;

  communicators_.resize(tp_size);

  NCCL_CHECK(ncclGroupStart());
  for (int dev_id = 0; dev_id < tp_size; ++dev_id) {
    CUDA_CHECK(cudaSetDevice(dev_id));

    int cur_rank = node_rank * tp_size + dev_id;
    KLLM_LOG_INFO << "Initialize nccl communicator, dev_id:" << dev_id << ", cur rank:" << cur_rank;
    NCCL_CHECK(ncclCommInitRank(&communicators_[dev_id], pp_size * tp_size, nccl_unique_id_, cur_rank));
  }
  NCCL_CHECK(ncclGroupEnd());

  // Build upstream and downstream ranks.
  int upstream_node_rank = node_rank - 1;
  if (upstream_node_rank < 0) {
    upstream_node_rank = pp_size - 1;
  }

  int downstream_node_rank = node_rank + 1;
  if (downstream_node_rank > pp_size - 1) {
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

  return Status();
}

Status NcclDataChannel::Disconnect() {
  for (auto& comm : communicators_) {
    NCCL_CHECK(ncclCommDestroy(comm));
  }

  return Status();
}

Status NcclDataChannel::ProcessRecvLoop() {
  while (!terminated_) {
    // Wait util recv invoked.
    hidden_unit_buffer_pool_->WaitUtilReadyToRecv();

    if (hidden_unit_buffer_pool_->Stopped()) {
      KLLM_LOG_WARNING << "ProcessRecvLoop hidden unit buffer pool stopped, break..";
      break;
    }

    HiddenUnitDeviceBuffer* hidden_unit = hidden_unit_buffer_pool_->GetDeviceBuffer();
    if (!hidden_unit) {
      KLLM_LOG_WARNING << "ProcessRecvLoop empty hidden unit from host send queue, break..";
      break;
    }

    Status status = ProcessDeviceBuffer(hidden_unit, false);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "ProcessRecvLoop send data failed, info:" << status.GetMessage();
    }

    // Put to device receive queue.
    hidden_unit_buffer_pool_->PutToDeviceRecvQueue(hidden_unit);
  }

  return Status();
}

Status NcclDataChannel::ProcessSendLoop() {
  while (!terminated_) {
    // Blocked, waiting util packet is ready.
    HiddenUnitDeviceBuffer* hidden_unit = hidden_unit_buffer_pool_->GetFromSendQueue();
    if (!hidden_unit) {
      KLLM_LOG_WARNING << "ProcessSendLoop empty hidden_unit from device send queue, break..";
      break;
    }

    Status status = ProcessDeviceBuffer(hidden_unit, true);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "ProcessSendLoop send data failed, info:" << status.GetMessage();
    }

    // Notify that send operation finished.
    hidden_unit_buffer_pool_->NotifySendFinished();
  }

  return Status();
}

Status NcclDataChannel::GetNcclDataType(DataType dtype, ncclDataType_t& nccl_dtype) {
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

Status NcclDataChannel::ProcessDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit, bool is_send) {
  int devs_num = static_cast<int>(hidden_unit->tensors.size());
  ncclGroupStart();
  for (int dev_id = 0; dev_id < devs_num; ++dev_id) {
    CUDA_CHECK(cudaSetDevice(dev_id));

    DataType data_type;
    std::vector<size_t> shape;
    GetHiddenUnitMeta(shape, data_type);

    ncclDataType_t nccl_dtype;
    GetNcclDataType(data_type, nccl_dtype);

    int64_t count = 1;
    for (auto i : shape) {
      count *= i;
    }

    Tensor& tensor = hidden_unit->tensors[dev_id];
    tensor.shape = shape;
    tensor.dtype = data_type;

    if (is_send) {
      if (hidden_unit->comm_type == DistributedCommunicationType::SCATTER) {
        NCCL_CHECK(ncclSend(tensor.GetPtr<void>(), count, nccl_dtype, downstream_ranks_[dev_id],
                            communicators_[hidden_unit->scatter_sender_rank],
                            context_->GetComputeStreams()[hidden_unit->scatter_sender_rank].Get()));
      } else {
        NCCL_CHECK(ncclSend(tensor.GetPtr<void>(), count, nccl_dtype, downstream_ranks_[dev_id], communicators_[dev_id],
                            context_->GetComputeStreams()[dev_id].Get()));
      }
    } else {
      if (hidden_unit->comm_type == DistributedCommunicationType::SCATTER) {
        NCCL_CHECK(ncclRecv(tensor.GetPtr<void>(), count, nccl_dtype, upstream_ranks_[hidden_unit->scatter_sender_rank],
                            communicators_[dev_id], context_->GetComputeStreams()[dev_id].Get()));
      } else {
        NCCL_CHECK(ncclRecv(tensor.GetPtr<void>(), count, nccl_dtype, upstream_ranks_[dev_id], communicators_[dev_id],
                            context_->GetComputeStreams()[dev_id].Get()));
      }
    }
  }
  ncclGroupEnd();

  return Status();
}

}  // namespace ksana_llm
