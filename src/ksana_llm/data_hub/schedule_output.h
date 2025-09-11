/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/blocking_queue.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

struct WorkerInferRequest {
  // The req id of the user's request.
  int64_t req_id;

  // The name of model instance.
  std::string model_name;

  // forwarding tokens contains tokens used in forwarding step. There are two parts:
  // 1. tokens have kv-caches, kv_cached_token_num is the number
  // 2. tokens need to be processed, their kv-caches are generated during computation
  std::vector<int> forwarding_tokens;

  // context decode or decode stage.
  InferStage infer_stage;

  // The decode step, 0 for context decode, and then 1, 2, 3...
  int step = 0;

  // The kv cache blocks this request used, the index is used as device_id.
  // The key and value are stored in same blocks.
  std::vector<std::vector<int>> kv_cache_blocks;

  // The flag for tagging request prefix cache usage
  bool is_use_prefix_cache = false;

  // The prefix cache tokens number
  int prefix_cache_len = 0;

  // The number of tokens for which kv caches have been generated.
  int kv_cached_token_num = 0;

  // The offset for multimodal rotary position embedding, computed in prefill phase by Python plugin,
  // and used in decode phase.
  int64_t mrotary_embedding_pos_offset = 0;

  // The model instance pointer.
  std::shared_ptr<ModelInstance> model_instance = nullptr;

  // Different reqs may have different cache managers.
  std::shared_ptr<CacheManagerInterface> cache_manager;

  // Get addr ptr of blocks.
  std::vector<std::vector<void*>> GetBlockPtrs() {
    std::vector<std::vector<void*>> block_ptrs;
    block_ptrs.reserve(kv_cache_blocks.size());
    for (size_t rank = 0; rank < kv_cache_blocks.size(); ++rank) {
      std::vector<void*> block_ptr(kv_cache_blocks[rank].size());
      cache_manager->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(rank)->GetBlockPtrs(kv_cache_blocks[rank],
                                                                                           block_ptr);
      block_ptrs.emplace_back(std::move(block_ptr));
    }
    return block_ptrs;
  }

  // current froward request related attention data para group id
  // NOTE(karlluo): for example: machine has 4 GPUs, Attention Data Parallelism is 2, Tensor Parallelism is 2.
  // |----Attn DP Group id 0----|----Attn DP Group id 1----|
  // |     TP 0   |     TP1     |     TP0    |     TP1     |
  // |     GPU0   |     GPU1    |     GPU2   |     GPU3    |
  uint32_t attn_dp_group_id = 0;

  // Not used.
  std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;

  // Not used.
  EmbeddingSlice input_refit_embedding;

  // Not used.
  const std::map<std::string, TargetDescribe> request_target;

  // Not used.
  std::map<std::string, PythonTensor> response;
};

// The scheduler output of every step.
struct ScheduleOutput {
  // Make it empty again, but keep runing reqs, called only on master node.
  void Reset() {
    finish_req_ids.clear();
    merged_swapout_req_ids.clear();
    merged_swapin_req_ids.clear();
    swapout_req_block_ids.clear();
    swapin_req_block_ids.clear();
  }

  // Make it empty again, called only on worker node.
  void Clear() {
    Reset();
    running_reqs.clear();
    worker_running_reqs.clear();
  }

  std::string ToString();

  template <typename T>
  std::string InferRequestToString(std::vector<std::shared_ptr<T>> reqs) {
    std::string result;
    result += "    [\n";
    for (auto req : reqs) {
      result += "      {\n";
      result += "        req_id:" + std::to_string(req->req_id) + "\n";
      result += "        model_name:" + req->model_name + "\n";
      result += "        forwarding_tokens:" + Vector2Str(req->forwarding_tokens) + "\n";
      result += "        infer_stage:" + std::to_string(req->infer_stage) + "\n";
      result += "        step:" + std::to_string(req->step) + "\n";
      result += "        kv_cache_blocks:";
      for (auto v : req->kv_cache_blocks) {
        result += Vector2Str(v) + ", ";
      }
      result += "\n";
      result += "        is_use_prefix_cache:" + std::to_string(req->is_use_prefix_cache) + "\n";
      result += "        kv_cached_token_num:" + std::to_string(req->kv_cached_token_num) + "\n";
      result += "        mrotary_embedding_pos_offset:" + std::to_string(req->mrotary_embedding_pos_offset) + "\n";
      result += "        attn_dp_group_id:" + std::to_string(req->attn_dp_group_id) + "\n";
      result += "      }\n";
    }
    result += "    ]\n";

    return result;
  }

  // The unique id for one schedule step.
  size_t multi_batch_id = DEFAULT_MULTI_BATCH_ID;

  size_t hidden_token_num = 0;

  // NOTE(karlluo): finished req ids, outer vector is for attention data parallelism.
  std::vector<std::vector<int64_t>> finish_req_ids;

  // NOTE(karlluo): merged requests ids, outer vector is for attention data parallelism.
  std::vector<std::vector<int64_t>> merged_swapout_req_ids;
  std::vector<std::vector<int64_t>> merged_swapin_req_ids;

  // NOTE(karlluo): swapped requests ids, outer vector is for attention data parallelism.
  std::vector<std::unordered_map<int64_t, std::vector<int>>> swapout_req_block_ids;
  std::vector<std::unordered_map<int64_t, std::vector<int>>> swapin_req_block_ids;

  // running, for master node.
  std::vector<std::shared_ptr<InferRequest>> running_reqs;

  // running, for worker node.
  std::vector<std::shared_ptr<WorkerInferRequest>> worker_running_reqs;
};

struct ScheduleOutputGroup {
 public:
  size_t schedule_id = DEFAULT_SCHEDULE_ID;
  std::vector<ScheduleOutput*> outputs;

 public:
  explicit ScheduleOutputGroup(size_t dp_num = 1) : schedule_id(DEFAULT_SCHEDULE_ID) {
    outputs.resize(dp_num, nullptr);
  }

  size_t RunningSize() const {
    size_t size = 0;
    for (auto& output : outputs) {
      if (output == nullptr) {
        continue;
      }
      size += output->running_reqs.size();
    }
    return size;
  }
};

class ScheduleOutputParser {
 public:
  // We just assume the data memory is large enough, and do not check it.
  static Status SerializeScheduleOutput(const ScheduleOutput* schedule_output, void* data);
  static Status DeserializeScheduleOutput(void* data, ScheduleOutput* schedule_output);

  // Get the serialized byte of a ScheduleOutput object.
  static size_t GetSerializedSize(const ScheduleOutput* schedule_output);

 private:
  static Status SerializeAsWorkerInferRequest(const std::vector<std::shared_ptr<InferRequest>>& infer_reqs, void* data,
                                              size_t& bytes);

  static Status DeserializeWorkerInferRequest(void* data,
                                              std::vector<std::shared_ptr<WorkerInferRequest>>& worker_infer_reqs,
                                              size_t& bytes);

 private:
  template <typename T>
  static Status SerializeVector(const std::vector<T>& vec, void* data, size_t& bytes) {
    size_t offset = 0;

    // vec size
    int vec_size = vec.size();
    std::memcpy(data + offset, &vec_size, sizeof(int));
    offset += sizeof(int);

    // vec elements.
    for (T e : vec) {
      std::memcpy(data + offset, &e, sizeof(T));
      offset += sizeof(T);
    }
    bytes = offset;

    return Status();
  }

  template <typename T>
  static Status DeserializeVector(void* data, std::vector<T>& vec, size_t& bytes) {
    size_t offset = 0;

    // vec size
    int vec_size = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    for (int i = 0; i < vec_size; ++i) {
      T e = *reinterpret_cast<T*>(data + offset);
      vec.push_back(e);
      offset += sizeof(T);
    }
    bytes = offset;

    return Status();
  }

  template <typename K, typename V>
  static Status SerializeKeyToVector(const std::unordered_map<K, std::vector<V>>& dict, void* data, size_t& bytes) {
    size_t offset = 0;

    int dict_size = dict.size();
    std::memcpy(data + offset, &dict_size, sizeof(int));
    offset += sizeof(int);

    for (auto it = dict.begin(); it != dict.end(); ++it) {
      K key = it->first;
      std::vector<V> vec = it->second;

      std::memcpy(data + offset, &key, sizeof(K));
      offset += sizeof(K);

      size_t inner_bytes;
      SerializeVector(vec, data + offset, inner_bytes);
      offset += inner_bytes;
    }
    bytes = offset;

    return Status();
  }

  template <typename K, typename V>
  static Status DeserializeKeyToVector(void* data, std::unordered_map<K, std::vector<V>>& dict, size_t& bytes) {
    size_t offset = 0;

    int dict_size = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    for (int i = 0; i < dict_size; ++i) {
      K key = *reinterpret_cast<K*>(data + offset);
      offset += sizeof(K);

      size_t inner_bytes;
      std::vector<V> vals;
      DeserializeVector(data + offset, vals, inner_bytes);
      offset += inner_bytes;

      dict[key] = vals;
    }
    bytes = offset;

    return Status();
  }

  template <typename T>
  static Status SerializeVectorOfVector(const std::vector<std::vector<T>>& vecs, void* data, size_t& bytes) {
    size_t offset = 0;

    // vec size
    int vec_size = vecs.size();
    std::memcpy(data + offset, &vec_size, sizeof(int));
    offset += sizeof(int);

    size_t inner_bytes;

    // vec elements.
    for (const std::vector<T>& vec : vecs) {
      SerializeVector(vec, data + offset, inner_bytes);
      offset += inner_bytes;
    }
    bytes = offset;

    return Status();
  }

  template <typename T>
  static Status DeserializeVectorOfVector(void* data, std::vector<std::vector<T>>& vecs, size_t& bytes) {
    size_t offset = 0;

    int vec_size = *reinterpret_cast<int*>(data + offset);
    offset += sizeof(int);

    size_t inner_bytes;

    for (int i = 0; i < vec_size; ++i) {
      std::vector<T> vec;
      DeserializeVector(data + offset, vec, inner_bytes);
      offset += inner_bytes;

      vecs.push_back(vec);
    }
    bytes = offset;

    return Status();
  }
};

// An object pool for schedule output.
class ScheduleOutputPool {
 public:
  // Get a schedule output object.
  ScheduleOutput* GetScheduleOutput();

  // Free the schedule output to object pool.
  Status FreeScheduleOutput(ScheduleOutput* schedule_output);

  // Put to and get from received buffer.
  Status PutToRecvQueue(ScheduleOutput* schedule_output);
  ScheduleOutput* GetFromRecvQueue();

  // Put to and get from send buffer.
  Status PutToSendQueue(ScheduleOutput* schedule_output);
  Packet* GetFromSendQueue();

  // All blocked queue will be returned immediately.
  Status Stop();

 private:
  // all free buffer objects.
  BlockingQueue<ScheduleOutput*> schedule_output_free_buffers_;

  // Send buffer.
  BlockingQueue<Packet*> schedule_output_send_buffers_;

  // Recv buffer.
  BlockingQueue<ScheduleOutput*> schedule_output_recv_buffers_;
};

}  // namespace ksana_llm
