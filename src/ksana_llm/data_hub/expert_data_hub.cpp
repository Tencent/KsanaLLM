/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/data_hub/expert_data_hub.h"
#include <cstddef>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/distributed/control_message.h"
#include "ksana_llm/distributed/packet_type.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// For expert parallel. Every expert node has only one pool.
ExpertParallelHiddenUnitBufferPool* g_expert_hidden_unit_buffer_pool = nullptr;

// Used to sync hidden_unit_buffer pointer among devices on the same node.
HiddenUnitDeviceBuffer* g_expert_recv_hidden_unit_buffer = nullptr;
HiddenUnitDeviceBuffer* g_expert_send_hidden_unit_buffer = nullptr;
HiddenUnitDeviceBuffer* g_expert_recv_comm_meta_hidden_unit_buffer = nullptr;
HiddenUnitDeviceBuffer* g_expert_send_comm_meta_hidden_unit_buffer = nullptr;

// Used to sync devices on the same node, be careful to avoid potential data race.
std::shared_ptr<Waiter> g_expert_waiter = std::make_shared<Waiter>(1);

// Used to connect with DeepEP
std::shared_ptr<ExpertParallelDeepepWrapper> g_deepep_wrapper = nullptr;

void InitializeExpertHiddenUnitBufferPool() {
  KLLM_LOG_INFO << "InitializeExpertHiddenUnitBufferPool";
  g_expert_hidden_unit_buffer_pool = new ExpertParallelHiddenUnitBufferPool();
}

ExpertParallelHiddenUnitBufferPool* GetExpertHiddenUnitBufferPool() { return g_expert_hidden_unit_buffer_pool; }

const std::shared_ptr<ExpertParallelDeepepWrapper>& GetExpertParallelDeepepWrapper() { return g_deepep_wrapper; }

void SetExpertParallelDeepepWrapper(const std::shared_ptr<ExpertParallelDeepepWrapper>& deepep_wrapper) {
  g_deepep_wrapper = deepep_wrapper;
}

Status InitializeExpertParallelDeepepWrapper(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                             const std::shared_ptr<Context>& context) {
  size_t num_ranks = runtime_config.parallel_basic_config.expert_world_size *
                     runtime_config.parallel_basic_config.expert_parallel_size;
  size_t num_ranks_per_node = runtime_config.parallel_basic_config.expert_parallel_size;
  size_t max_token_num = runtime_config.max_step_token_num;
  size_t hidden_size = static_cast<size_t>(model_config.hidden_units);
  size_t expert_topk = model_config.moe_config.experts_topk;
  size_t num_experts = model_config.moe_config.num_experts;
  size_t node_rank = context->GetExpertParallelExpertNodeRank();
  // TODO(zezhao): 当前仅有开启 enable_full_shared_expert 时才会使用 DeepEP，后续会将该变量更名并调整在 EP 中的耦合关系.
  if (num_ranks > 1 && runtime_config.enable_full_shared_expert) {
    // TODO(zezhao): 临时限制，Expert-Parallel 开启时 Data-Parallel 需相等，后续将支持 ATP > 1
    if (runtime_config.parallel_basic_config.expert_parallel_size !=
        runtime_config.parallel_basic_config.attn_data_parallel_size) {
      KLLM_LOG_ERROR << fmt::format("Expert-Parallel only supports DP=EP");
      KLLM_THROW(fmt::format("Expert-Parallel only supports DP=EP"));
    }
    // Initialize deepep_wrapper when using Expert-Parallel
    g_deepep_wrapper = std::make_shared<ExpertParallelDeepepWrapper>(
        num_ranks, num_ranks_per_node, node_rank, max_token_num, hidden_size, expert_topk, num_experts, context);
    g_deepep_wrapper->Init();
  }
  return Status();
}

Status InitExpertHiddenUnits() {
  HiddenUnitDeviceBuffer* hidden_unit_buffer = GetExpertHiddenUnitBufferPool()->GetDeviceBufferSingle();
  if (!hidden_unit_buffer) {
    return Status(RET_RUNTIME_FAILED, "GetDeviceBuffer error, empty result.");
  }

  // Maybe not used when do send after recv.
  SetCurrentExpertSendHiddenUnitBuffer(hidden_unit_buffer);
  return Status();
}

Status FreeExpertRecvHiddenUnits(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
  // free recv buffer.
  GetExpertHiddenUnitBufferPool()->FreeDeviceBuffer(hidden_unit_buffer);
  return Status();
}

Status SendExpertHiddenUnits(HiddenUnitDeviceBuffer* hidden_unit_buffer, bool is_sync) {
  if (is_sync) {
    KLLM_LOG_DEBUG << fmt::format("SendExpertHiddenUnits::PutToSendQueue");
    GetExpertHiddenUnitBufferPool()->PutToSendQueue(hidden_unit_buffer);

  } else {
    KLLM_LOG_DEBUG << fmt::format("SendExpertHiddenUnits::AsyncPutToSendQueue");
    GetExpertHiddenUnitBufferPool()->AsyncPutToSendQueue(hidden_unit_buffer);
  }
  return Status();
}

// Notify recv thread and sync recv data.
HiddenUnitDeviceBuffer* RecvExpertHiddenUnits(int rank) {
  // All the model inference will call this method.
  // But only the thread that have true do_recv actually do the receiving operation.
  // Other threads are blocked until receiving operation finished, then start to do computation.
  if (rank == 0) {
    HiddenUnitDeviceBuffer* hidden_unit_buffer = GetExpertHiddenUnitBufferPool()->GetFromDeviceRecvQueue();
    if (!hidden_unit_buffer) {
      KLLM_LOG_WARNING << "GetFromDeviceRecvQueue , not recv data, empty result.";
    }

    SetCurrentExpertRecvHiddenUnitBuffer(hidden_unit_buffer);
  }

  GetExpertHiddenUnitBufferPool()->ExpertBarrier();

  return GetCurrentExpertRecvHiddenUnitBuffer();
}

// Notify recv thread and get recv data.
HiddenUnitDeviceBuffer* AsyncRecvExpertHiddenUnits(int rank) {
  // All the model inference will call this method.
  // But only the thread that have true do_recv actually do the receiving operation.
  // Other threads are blocked until receiving operation finished, then start to do computation.
  if (rank == 0) {
    HiddenUnitDeviceBuffer* hidden_unit_buffer = GetExpertHiddenUnitBufferPool()->AsyncGetFromDeviceRecvQueue();
    if (!hidden_unit_buffer) {
      KLLM_LOG_INFO << "AsyncGetFromDeviceRecvQueue , not recv data, empty result.";
    }

    SetCurrentExpertRecvHiddenUnitBuffer(hidden_unit_buffer);
    g_expert_waiter->Notify();

    return hidden_unit_buffer;
  } else {
    g_expert_waiter->Wait();
  }

  return GetCurrentExpertRecvHiddenUnitBuffer();
}

void DestroyExpertHiddenUnitBufferPool() {
  if (g_expert_hidden_unit_buffer_pool) {
    delete g_expert_hidden_unit_buffer_pool;
    g_expert_hidden_unit_buffer_pool = nullptr;
  }
}

void SetCurrentExpertRecvHiddenUnitBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
  g_expert_recv_hidden_unit_buffer = hidden_unit_buffer;
}

HiddenUnitDeviceBuffer* GetCurrentExpertRecvHiddenUnitBuffer() { return g_expert_recv_hidden_unit_buffer; }

void SetCurrentExpertSendHiddenUnitBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
  g_expert_send_hidden_unit_buffer = hidden_unit_buffer;
}

HiddenUnitDeviceBuffer* GetCurrentExpertSendHiddenUnitBuffer() { return g_expert_send_hidden_unit_buffer; }

void SetCurrentExpertRecvCommMetaHiddenUnitBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
  g_expert_recv_comm_meta_hidden_unit_buffer = hidden_unit_buffer;
}

HiddenUnitDeviceBuffer* GetCurrentExpertRecvCommMetaHiddenUnitBuffer() {
  return g_expert_recv_comm_meta_hidden_unit_buffer;
}

Status ResetExpertReceiveWaiter() {
  g_expert_waiter->Reset(1);
  return Status();
}

Status ResetExpertWaiter() {
  g_expert_waiter->Reset(1);
  return Status();
}

Status ExpertWait() {
  g_expert_waiter->Wait();
  return Status();
}

Status ExpertNotify() {
  g_expert_waiter->Notify();
  return Status();
}

void PrintExpertHiddenUnitBufferPoolInfo(std::string tag) {
  KLLM_LOG_INFO << fmt::format("{} free buffer size:{}, send_buffer_size:{}, recv_buffer_size:{}", tag,
                               GetExpertHiddenUnitBufferPool()->GetFreeDeviceBufferSize(),
                               GetExpertHiddenUnitBufferPool()->GetSendDeviceBufferSize(),
                               GetExpertHiddenUnitBufferPool()->GetRecvDeviceBufferSize());
  return;
}

}  // namespace ksana_llm
