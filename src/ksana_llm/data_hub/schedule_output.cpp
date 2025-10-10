/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/data_hub/schedule_output.h"

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/packet_util.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

ForwardRequest* WorkerInferRequest::GetForwardRequest() {
  if (forward_request_ == nullptr) {
    forward_request_ = std::make_unique<ForwardRequest>();
    forward_request_->req_id = req_id;
    forward_request_->response = &response;
    forward_request_->flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    forward_request_->input_refit_embedding = &input_refit_embedding;
    forward_request_->mrotary_embedding_pos_offset = &mrotary_embedding_pos_offset;
    forward_request_->forwarding_tokens =
        std::shared_ptr<decltype(forwarding_tokens)>(&forwarding_tokens, [](decltype(forwarding_tokens)*) {});
    forward_request_->request_target = std::make_shared<const std::map<std::string, TargetDescribe>>(request_target);
    forward_request_->logits_custom_length = 0;
    forward_request_->logits_buf.clear();
    forward_request_->is_cudagraph_capture_request = false;
    forward_request_->cache_manager = cache_manager;

    const size_t rank_num = kv_cache_blocks.size();
    forward_request_->kv_cache_ptrs.resize(rank_num);
    forward_request_->atb_kv_cache_base_blk_ids.resize(rank_num);
  }

  forward_request_->infer_stage = infer_stage;
  forward_request_->step = step;
  forward_request_->kv_cached_token_num = kv_cached_token_num;
  forward_request_->is_use_prefix_cache = is_use_prefix_cache;
  forward_request_->prefix_cache_len = prefix_cache_len + flexible_cached_copy_tasks.size();
  forward_request_->attn_dp_group_id = attn_dp_group_id;

  // rebuild in worker
  forward_request_->kv_cache_ptrs.clear();
  forward_request_->atb_kv_cache_base_blk_ids.clear();
  UpdateBlockPtrs(forward_request_->kv_cache_ptrs);
#if defined(ENABLE_ACL) || defined(ENABLE_CUDA)
  AppendFlatKVCacheBlkIds(model_instance->GetLayerNum(), kv_cache_blocks, forward_request_->atb_kv_cache_base_blk_ids,
                          cache_manager);
#endif

  return forward_request_.get();
}

std::string ScheduleOutput::ToString() {
  std::string result;
  result += "{\n";
  result += "  multi_batch_id: " + std::to_string(multi_batch_id) + "\n";

  result += "  finish_req_ids:\n";
  for (size_t finish_req_ids_on_attn_dp_idx = 0; finish_req_ids_on_attn_dp_idx < finish_req_ids.size();
       ++finish_req_ids_on_attn_dp_idx) {
    result += "    " + std::to_string(finish_req_ids_on_attn_dp_idx) + ": ";
    result += Vector2Str(finish_req_ids[finish_req_ids_on_attn_dp_idx]) + "\n";
  }

  result += "  merged_swapout_req_ids:\n";
  for (size_t merged_swapout_req_ids_on_attn_dp_idx = 0;
       merged_swapout_req_ids_on_attn_dp_idx < merged_swapout_req_ids.size(); ++merged_swapout_req_ids_on_attn_dp_idx) {
    result += "    " + std::to_string(merged_swapout_req_ids_on_attn_dp_idx) + ": ";
    result += Vector2Str(merged_swapout_req_ids[merged_swapout_req_ids_on_attn_dp_idx]) + "\n";
  }

  result += "  merged_swapin_req_ids:\n";
  for (size_t merged_swapin_req_ids_on_attn_dp_idx = 0;
       merged_swapin_req_ids_on_attn_dp_idx < merged_swapin_req_ids.size(); ++merged_swapin_req_ids_on_attn_dp_idx) {
    result += "    " + std::to_string(merged_swapin_req_ids_on_attn_dp_idx) + ": ";
    result += Vector2Str(merged_swapin_req_ids[merged_swapin_req_ids_on_attn_dp_idx]) + "\n";
  }

  result += "  swapout_req_block_ids:\n";
  for (size_t swapout_req_block_ids_on_attn_dp_idx = 0;
       swapout_req_block_ids_on_attn_dp_idx < swapout_req_block_ids.size(); ++swapout_req_block_ids_on_attn_dp_idx) {
    result += "    " + std::to_string(swapout_req_block_ids_on_attn_dp_idx) + ": ";
    for (auto pair : swapout_req_block_ids[swapout_req_block_ids_on_attn_dp_idx]) {
      result += "    " + std::to_string(pair.first) + ": " + Vector2Str(pair.second) + "\n";
    }
  }

  result += "  swapin_req_block_ids:\n";
  for (size_t swapin_req_block_ids_on_attn_dp_idx = 0;
       swapin_req_block_ids_on_attn_dp_idx < swapin_req_block_ids.size(); ++swapin_req_block_ids_on_attn_dp_idx) {
    result += "    " + std::to_string(swapin_req_block_ids_on_attn_dp_idx) + ": ";
    for (auto pair : swapin_req_block_ids[swapin_req_block_ids_on_attn_dp_idx]) {
      result += "    " + std::to_string(pair.first) + ": " + Vector2Str(pair.second) + "\n";
    }
  }

  result += "  running_reqs:\n";
  result += InferRequestToString(running_reqs);

  result += "  worker_running_reqs:\n";
  result += InferRequestToString(worker_running_reqs);

  result += "}";

  return result;
}

size_t ScheduleOutputParser::GetSerializedSize(const ScheduleOutput* schedule_output) {
  size_t serialized_bytes = 0;

  // multi_batch_id
  serialized_bytes += sizeof(size_t);

  // finish_req_ids
  serialized_bytes += sizeof(int);
  for (size_t dp_idx = 0; dp_idx < schedule_output->finish_req_ids.size(); ++dp_idx) {
    serialized_bytes += sizeof(int);
    serialized_bytes += schedule_output->finish_req_ids[dp_idx].size() * sizeof(int64_t);
  }

  // merged swapout reqs
  serialized_bytes += sizeof(int);
  for (size_t dp_idx = 0; dp_idx < schedule_output->merged_swapout_req_ids.size(); ++dp_idx) {
    serialized_bytes += sizeof(int);
    serialized_bytes += schedule_output->merged_swapout_req_ids[dp_idx].size() * sizeof(int64_t);
  }

  // merged swapin reqs
  serialized_bytes += sizeof(int);
  for (size_t dp_idx = 0; dp_idx < schedule_output->merged_swapin_req_ids.size(); ++dp_idx) {
    serialized_bytes += sizeof(int);
    serialized_bytes += schedule_output->merged_swapin_req_ids[dp_idx].size() * sizeof(int64_t);
  }

  // swapout req with blocks.
  serialized_bytes += sizeof(int);
  for (size_t dp_idx = 0; dp_idx < schedule_output->swapout_req_block_ids.size(); ++dp_idx) {
    auto& swapout_req_block_ids = schedule_output->swapout_req_block_ids[dp_idx];
    serialized_bytes += sizeof(int);
    for (auto& [k, v] : swapout_req_block_ids) {
      // key
      serialized_bytes += sizeof(int64_t);

      // value
      serialized_bytes += sizeof(int);
      serialized_bytes += v.size() * sizeof(int);
    }
  }

  // swapin req with blocks.
  serialized_bytes += sizeof(int);
  for (size_t dp_idx = 0; dp_idx < schedule_output->swapin_req_block_ids.size(); ++dp_idx) {
    auto& swapin_req_block_ids = schedule_output->swapin_req_block_ids[dp_idx];
    serialized_bytes += sizeof(int);
    for (auto& [k, v] : swapin_req_block_ids) {
      // key
      serialized_bytes += sizeof(int64_t);

      // value
      serialized_bytes += sizeof(int);
      serialized_bytes += v.size() * sizeof(int);
    }
  }

  // running reqs.
  serialized_bytes += sizeof(int);

  for (auto req : schedule_output->running_reqs) {
    // req_id
    serialized_bytes += sizeof(int64_t);

    // model_name
    serialized_bytes += sizeof(int);
    serialized_bytes += req->model_name.size();

    // output_tokens
    serialized_bytes += sizeof(int);
    serialized_bytes += req->forwarding_tokens.size() * sizeof(int);

    // infer_stage
    serialized_bytes += sizeof(InferStage);

    // step
    serialized_bytes += sizeof(int);

    // kv_cache_blocks
    serialized_bytes += sizeof(int);
    for (auto& v : req->kv_cache_blocks) {
      serialized_bytes += sizeof(int);
      serialized_bytes += v.size() * sizeof(int);
    }

    // is_use_prefix_cache
    serialized_bytes += sizeof(bool);

    // prefix_cache_len
    serialized_bytes += sizeof(int);

    // kv_cached_token_num
    serialized_bytes += sizeof(int);

    // mrotary_embedding_pos_offset
    serialized_bytes += sizeof(int64_t);

    // attn_dp_group_id
    serialized_bytes += sizeof(uint32_t);
  }

  return serialized_bytes;
}

Status ScheduleOutputParser::SerializeAsWorkerInferRequest(const std::vector<std::shared_ptr<InferRequest>>& infer_reqs,
                                                           void* data, size_t& bytes) {
  size_t offset = 0;

  int req_num = infer_reqs.size();
  std::memcpy(data + offset, &req_num, sizeof(int));
  offset += sizeof(int);

  size_t inner_bytes;

  for (auto req : infer_reqs) {
    // req_id
    std::memcpy(data + offset, &req->req_id, sizeof(int64_t));
    offset += sizeof(int64_t);

    // model_name
    int model_name_size = req->model_name.size();
    std::memcpy(data + offset, &model_name_size, sizeof(int));
    offset += sizeof(int);

    std::memcpy(data + offset, req->model_name.data(), req->model_name.size());
    offset += req->model_name.size();

    // forwarding_tokens
    SerializeVector(req->forwarding_tokens, data + offset, inner_bytes);
    offset += inner_bytes;

    // infer_stage
    std::memcpy(data + offset, &req->infer_stage, sizeof(InferStage));
    offset += sizeof(InferStage);

    // step
    std::memcpy(data + offset, &req->step, sizeof(int));
    offset += sizeof(int);

    // kv_cache_blocks
    SerializeVectorOfVector(req->kv_cache_blocks, data + offset, inner_bytes);
    offset += inner_bytes;

    // is_use_prefix_cache
    std::memcpy(data + offset, &req->is_use_prefix_cache, sizeof(bool));
    offset += sizeof(bool);

    // prefix_cache_len
    std::memcpy(data + offset, &req->prefix_cache_len, sizeof(int));
    offset += sizeof(int);

    // kv_cached_token_num
    std::memcpy(data + offset, &req->kv_cached_token_num, sizeof(int));
    offset += sizeof(int);

    // mrotary_embedding_pos_offset
    std::memcpy(data + offset, &req->mrotary_embedding_pos_offset, sizeof(int64_t));
    offset += sizeof(int64_t);

    // attn_dp_group_id
    std::memcpy(data + offset, &req->attn_dp_group_id, sizeof(uint32_t));
    offset += sizeof(uint32_t);
  }
  bytes = offset;

  return Status();
}

Status ScheduleOutputParser::DeserializeWorkerInferRequest(
    void* data, std::vector<std::shared_ptr<WorkerInferRequest>>& worker_infer_reqs, size_t& bytes) {
  size_t offset = 0;

  int req_num = *reinterpret_cast<int*>(data + offset);
  offset += sizeof(int);

  size_t inner_bytes;

  for (size_t i = 0; i < static_cast<size_t>(req_num); ++i) {
    std::shared_ptr<WorkerInferRequest> req = std::make_shared<WorkerInferRequest>();

    // req_id
    req->req_id = *reinterpret_cast<int64_t*>(data + offset);
    offset += sizeof(int64_t);

    // model_name
    int model_name_size = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    std::string model_name;
    model_name.assign(reinterpret_cast<char*>(data + offset), model_name_size);
    req->model_name = model_name;
    offset += model_name_size;

    // forwarding_tokens
    std::vector<int> forwarding_tokens;
    DeserializeVector(data + offset, forwarding_tokens, inner_bytes);
    req->forwarding_tokens = forwarding_tokens;
    offset += inner_bytes;

    // infer_stage
    req->infer_stage = *reinterpret_cast<InferStage*>(data + offset);
    offset += sizeof(InferStage);

    // step
    req->step = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    // kv_cache_blocks
    std::vector<std::vector<int>> kv_cache_blocks;
    DeserializeVectorOfVector(data + offset, kv_cache_blocks, inner_bytes);
    req->kv_cache_blocks = kv_cache_blocks;
    offset += inner_bytes;

    // is_use_prefix_cache
    req->is_use_prefix_cache = *reinterpret_cast<bool*>(data + offset);
    offset += sizeof(bool);

    // prefix_cache_len
    req->prefix_cache_len = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    // kv_cached_token_num
    req->kv_cached_token_num = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    // mrotary_embedding_pos_offset.
    req->mrotary_embedding_pos_offset = *reinterpret_cast<int64_t*>(data + offset);
    offset += sizeof(int64_t);

    // attn_dp_group_id
    req->attn_dp_group_id = *reinterpret_cast<uint32_t*>(data + offset);
    offset += sizeof(uint32_t);

    // Get model instance from data hub.
    req->model_instance = GetModelInstance(req->model_name);

    // Get cache manager from data hub.
    req->cache_manager = GetCacheManager(req->attn_dp_group_id);

    worker_infer_reqs.push_back(req);
  }
  bytes = offset;

  return Status();
}

Status ScheduleOutputParser::SerializeScheduleOutput(const ScheduleOutput* schedule_output, void* data) {
  size_t offset = 0;

  // multi_batch_id
  std::memcpy(data + offset, &schedule_output->multi_batch_id, sizeof(size_t));
  offset += sizeof(size_t);

  size_t bytes;

  // finished reqs.
  int vec_size = schedule_output->finish_req_ids.size();
  std::memcpy(data + offset, &vec_size, sizeof(int));
  offset += sizeof(int);
  for (int dp_idx = 0; dp_idx < vec_size; ++dp_idx) {
    SerializeVector(schedule_output->finish_req_ids[dp_idx], data + offset, bytes);
    offset += bytes;
  }

  // merged swapout reqs
  vec_size = schedule_output->merged_swapout_req_ids.size();
  std::memcpy(data + offset, &vec_size, sizeof(int));
  offset += sizeof(int);
  for (int dp_idx = 0; dp_idx < vec_size; ++dp_idx) {
    SerializeVector(schedule_output->merged_swapout_req_ids[dp_idx], data + offset, bytes);
    offset += bytes;
  }

  // merged swapin reqs
  vec_size = schedule_output->merged_swapin_req_ids.size();
  std::memcpy(data + offset, &vec_size, sizeof(int));
  offset += sizeof(int);
  for (int dp_idx = 0; dp_idx < vec_size; ++dp_idx) {
    SerializeVector(schedule_output->merged_swapin_req_ids[dp_idx], data + offset, bytes);
    offset += bytes;
  }

  // swapout req with blocks.
  vec_size = schedule_output->swapout_req_block_ids.size();
  std::memcpy(data + offset, &vec_size, sizeof(int));
  offset += sizeof(int);
  for (int dp_idx = 0; dp_idx < vec_size; ++dp_idx) {
    SerializeKeyToVector(schedule_output->swapout_req_block_ids[dp_idx], data + offset, bytes);
    offset += bytes;
  }

  // swapin req with blocks.
  vec_size = schedule_output->swapin_req_block_ids.size();
  std::memcpy(data + offset, &vec_size, sizeof(int));
  offset += sizeof(int);
  for (int dp_idx = 0; dp_idx < vec_size; ++dp_idx) {
    SerializeKeyToVector(schedule_output->swapin_req_block_ids[dp_idx], data + offset, bytes);
    offset += bytes;
  }

  // running reqs.
  SerializeAsWorkerInferRequest(schedule_output->running_reqs, data + offset, bytes);
  offset += bytes;

  return Status();
}

Status ScheduleOutputParser::DeserializeScheduleOutput(void* data, ScheduleOutput* schedule_output) {
  size_t offset = 0;

  // multi_batch_id
  schedule_output->multi_batch_id = *reinterpret_cast<size_t*>(data + offset);
  offset += sizeof(size_t);

  size_t bytes;

  // finished reqs
  int vec_size = *reinterpret_cast<int*>(data + offset);
  offset += sizeof(int);
  schedule_output->finish_req_ids.resize(vec_size);
  for (int dp_idx = 0; dp_idx < vec_size; ++dp_idx) {
    std::vector<int64_t> finish_req_ids;
    DeserializeVector(data + offset, finish_req_ids, bytes);
    schedule_output->finish_req_ids[dp_idx] = finish_req_ids;
    offset += bytes;
  }

  // merged swapout reqs
  vec_size = *reinterpret_cast<int*>(data + offset);
  offset += sizeof(int);
  schedule_output->merged_swapout_req_ids.resize(vec_size);
  for (int dp_idx = 0; dp_idx < vec_size; ++dp_idx) {
    std::vector<int64_t> merged_swapout_req_ids;
    DeserializeVector(data + offset, merged_swapout_req_ids, bytes);
    schedule_output->merged_swapout_req_ids[dp_idx] = merged_swapout_req_ids;
    offset += bytes;
  }

  // merged swapin reqs
  vec_size = *reinterpret_cast<int*>(data + offset);
  offset += sizeof(int);
  schedule_output->merged_swapin_req_ids.resize(vec_size);
  for (int dp_idx = 0; dp_idx < vec_size; ++dp_idx) {
    std::vector<int64_t> merged_swapin_req_ids;
    DeserializeVector(data + offset, merged_swapin_req_ids, bytes);
    schedule_output->merged_swapin_req_ids[dp_idx] = merged_swapin_req_ids;
    offset += bytes;
  }

  // swapout req with blocks.
  vec_size = *reinterpret_cast<int*>(data + offset);
  offset += sizeof(int);
  schedule_output->swapout_req_block_ids.resize(vec_size);
  for (int dp_idx = 0; dp_idx < vec_size; ++dp_idx) {
    std::unordered_map<int64_t, std::vector<int>> swapout_req_block_ids;
    DeserializeKeyToVector(data + offset, swapout_req_block_ids, bytes);
    schedule_output->swapout_req_block_ids[dp_idx] = swapout_req_block_ids;
    offset += bytes;
  }

  // swapin req with blocks.
  vec_size = *reinterpret_cast<int*>(data + offset);
  offset += sizeof(int);
  schedule_output->swapin_req_block_ids.resize(vec_size);
  for (int dp_idx = 0; dp_idx < vec_size; ++dp_idx) {
    std::unordered_map<int64_t, std::vector<int>> swapin_req_block_ids;
    DeserializeKeyToVector(data + offset, swapin_req_block_ids, bytes);
    schedule_output->swapin_req_block_ids[dp_idx] = swapin_req_block_ids;
    offset += bytes;
  }

  // running reqs.
  std::vector<std::shared_ptr<WorkerInferRequest>> worker_running_reqs;
  DeserializeWorkerInferRequest(data + offset, worker_running_reqs, bytes);
  schedule_output->worker_running_reqs = worker_running_reqs;
  offset += bytes;

  return Status();
}

ScheduleOutput* ScheduleOutputPool::GetScheduleOutput() {
  if (schedule_output_free_buffers_.Empty()) {
    ScheduleOutput* schedule_output = new ScheduleOutput();
    return schedule_output;
  }

  return schedule_output_free_buffers_.Get();
}

Status ScheduleOutputPool::FreeScheduleOutput(ScheduleOutput* schedule_output) {
  schedule_output->Clear();
  schedule_output_free_buffers_.Put(schedule_output);

  return Status();
}

Status ScheduleOutputPool::PutToRecvQueue(ScheduleOutput* schedule_output) {
  schedule_output_recv_buffers_.Put(schedule_output);
  return Status();
}

ScheduleOutput* ScheduleOutputPool::GetFromRecvQueue() { return schedule_output_recv_buffers_.Get(); }

Status ScheduleOutputPool::PutToSendQueue(ScheduleOutput* schedule_output) {
  size_t schedule_output_size = ScheduleOutputParser::GetSerializedSize(schedule_output);

  Packet* packet = GetRawPacket(schedule_output_size);
  if (packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessSendScheduleLoop allocate memory error.");
  }

  packet->type = PacketType::CONTROL_REQ_SCHEDULE;
  ScheduleOutputParser::SerializeScheduleOutput(schedule_output, packet->body);

  schedule_output_send_buffers_.Put(packet);
  return Status();
}

Packet* ScheduleOutputPool::GetFromSendQueue() { return schedule_output_send_buffers_.Get(); }

Status ScheduleOutputPool::Stop() {
  schedule_output_free_buffers_.Stop();
  schedule_output_send_buffers_.Stop();
  schedule_output_recv_buffers_.Stop();

  return Status();
}

}  // namespace ksana_llm
