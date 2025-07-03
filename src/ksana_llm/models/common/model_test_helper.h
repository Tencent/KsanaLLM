/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/cache_manager/cache_manager_interface.h"
#include "ksana_llm/models/base/buffer_manager.h"
#include "ksana_llm/models/base/forwarding_context.h"
#include "ksana_llm/models/base/layer_creation_context.h"
#include "ksana_llm/profiler/profile_event.h"

#include "ksana_llm/layers/cast_layer.h"
#include "ksana_llm/models/base/fake_weight_for_test.h"
#include "ksana_llm/models/common/model_interface.h"
#include "ksana_llm/runtime/llm_runtime.h"
namespace ksana_llm {

template <typename T>
class FakeModel {
 public:
  FakeModel(std::shared_ptr<ModelInterface<T>> model, std::shared_ptr<Context> context, const int rank,
            ModelConfig& model_config, PipelineConfig pipeline_config, std::shared_ptr<BaseWeight> base_weight,
            bool reuse_prefix_config)  // TODO(robertyuan): reuse_prefix_config is a weird param
      : model_(model),
        context_(context),
        rank_(rank),
        head_num_(model_config.head_num),
        size_per_head_(model_config.size_per_head) {
    SetDevice(rank_);
    buffer_mgr_.SetRank(rank_);

    bool return_hidden_states = pipeline_config.lower_nextn_layer_idx >= static_cast<int>(model_config.num_layer);
    buffers_.Init(context_, rank_, model_config, return_hidden_states, &buffer_mgr_);
    forwarding_context_.Init(context, rank, model_config, pipeline_config, buffers_.buffers_.get(), &buffer_mgr_,
                             /*multi_batch_id*/ 0);
    // Initialize instances for each layer.
    layer_creation_context_.Init(base_weight, shared_matmul_workspace_buffer_, context, rank, pipeline_config,
                                 model_config, &buffer_mgr_);

    Tensor& residual_buffer_tensor = buffers_.local_residual_buffer_tensors_[0];
    host_residual_buffer_tensor_ =
        Tensor(MemoryLocation::LOCATION_HOST, residual_buffer_tensor.dtype, residual_buffer_tensor.shape, rank_);
    TensorBuffer* residual_fp32_buffer =
        buffer_mgr_.CreateBufferTensor("residual_fp32_buffer_", {residual_buffer_tensor.GetElementNumber()}, TYPE_FP32);
    residual_fp32_buffer_ = residual_fp32_buffer->GetTensors();
    ModelCreationConfig model_creation_config;
    // Flash Attention requires the input shape to match the actual token length.
    // When dealing with prefix_cache or speculative decoding, it is necessary to
    // first fill in the missing parts
    int layer_num_on_node = model_config.num_layer / model_config.tensor_para_size;
    ModelRunConfig model_run_config;
    model_->GetModelRunConfig(model_run_config, model_config);

    if (model_config.type == "qwen2_vl") {
      mrotary_section_tensor_ = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {3}, rank_);
      MemcpyAsync(mrotary_section_tensor_.GetPtr<void>(), model_config.rope_scaling_factor_config.mrope_section.data(),
                  3 * sizeof(int), MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    }
    model_creation_config.Init(model_config, buffers_.cos_sin_cache_tensor_, model_run_config.position_encoding,
                               reuse_prefix_config, layer_num_on_node, mrotary_section_tensor_.GetPtr<const int>());

    model_->CreateLayers(layer_creation_context_, model_creation_config);

    cast_layer_ = std::make_shared<CastLayer<T>>();
    cast_layer_->Init({}, context_, rank_);
  }

  ~FakeModel() {}

  Status Forward(std::vector<ForwardRequest>& forward_reqs, RunMode run_mode = RunMode::kMain) {
    forwarding_context_.UpdateBeforeForward(forward_reqs, run_mode);
    LookupEmbedding();
    model_->Forward(buffers_.local_residual_buffer_tensors_, forwarding_context_);
    forwarding_context_.UpdateAfterForward(forward_reqs);
    return Status();
  }

  Status GetOutputToCPU(std::vector<float>& cpu_fp32_data) {
    auto& residula_buffer_tensor = buffers_.local_residual_buffer_tensors_[0];
    cpu_fp32_data.resize(residula_buffer_tensor.GetElementNumber());
    STATUS_CHECK_RETURN(
        cast_layer_->Forward({residula_buffer_tensor, forwarding_context_.GetAttentionForwardContext().forward_shape},
                             residual_fp32_buffer_));
    DeviceSynchronize();
    Memcpy(cpu_fp32_data.data(), residual_fp32_buffer_[0].template GetPtr<void>(),
           sizeof(float) * residual_fp32_buffer_[0].GetElementNumber(), MEMCPY_DEVICE_TO_HOST);
    return Status();
  }

 protected:
  virtual void FakeEmbeddingResults(std::vector<int>& input_ids_cpu, std::vector<float>& residual_buffer_vector,
                                    size_t head_num, size_t size_per_head) {
    size_t input_ids_num = input_ids_cpu.size();
    // Fake embedding output
    float token_id_step = 0.001;
    float head_emb_step = 0.01;
    residual_buffer_vector.resize(input_ids_num * head_num * size_per_head);
    for (size_t idx = 0; idx < input_ids_num; idx++) {
      size_t idx_offset = idx * head_num * size_per_head;
      float token_id_value = input_ids_cpu[idx] * token_id_step;
      for (size_t head_idx = 0; head_idx < head_num; head_idx++) {
        size_t head_offset = head_idx * size_per_head;
        for (size_t emb_idx = 0; emb_idx < size_per_head; emb_idx++) {
          residual_buffer_vector[idx_offset + head_offset + emb_idx] = token_id_value + emb_idx * head_emb_step;
        }
      }
    }
  }

 private:
  Status LookupEmbedding() {
    std::vector<int> input_ids_cpu;
    std::vector<float> residual_buffer_vector;
    Tensor& input_ids = forwarding_context_.model_input_->input_ids;
    size_t input_ids_num = input_ids.GetElementNumber();

    // Fetch input ids from device
    input_ids_cpu.resize(input_ids_num);
    MemcpyAsync(input_ids_cpu.data(), input_ids.GetPtr<void>(), input_ids_cpu.size() * sizeof(int),
                MEMCPY_DEVICE_TO_HOST, context_->GetD2HStreams()[rank_]);
    DeviceSynchronize();

    FakeEmbeddingResults(input_ids_cpu, residual_buffer_vector, head_num_, size_per_head_);

    Tensor& gpu_tensor = buffers_.local_residual_buffer_tensors_[0];
    gpu_tensor.shape = {input_ids_num, head_num_ * size_per_head_};
    host_residual_buffer_tensor_.shape = gpu_tensor.shape;
    CopyVectorToHostTensor(host_residual_buffer_tensor_, host_residual_buffer_tensor_.dtype, residual_buffer_vector);
    Memcpy(gpu_tensor.GetPtr<void>(), host_residual_buffer_tensor_.GetPtr<void>(),
           host_residual_buffer_tensor_.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);
    return Status();
  }

 private:
  std::shared_ptr<ModelInterface<T>> model_;
  std::shared_ptr<Context> context_;
  int rank_;

  LayerCreationContext<T> layer_creation_context_;

  ModelBuffers buffers_;
  ForwardingContext<T> forwarding_context_;
  std::shared_ptr<Tensor> shared_matmul_workspace_buffer_ = nullptr;

  size_t head_num_;
  uint32_t size_per_head_;

  BufferManager buffer_mgr_;

  Tensor host_residual_buffer_tensor_;
  std::vector<Tensor> residual_fp32_buffer_{1};
  std::shared_ptr<BaseLayer> cast_layer_;

  // Only used for QWenVL
  Tensor mrotary_section_tensor_;
};

class ForwardRequestBuilderForTest {
 public:
  explicit ForwardRequestBuilderForTest(const ModelConfig& model_config,
                                        std::shared_ptr<CacheManagerInterface> cache_manager)
      : layer_num_(model_config.num_layer),
        block_token_num_(model_config.block_token_num),
        tensor_para_size_(model_config.tensor_para_size) {
    cache_manager_ = cache_manager;
  }
  ~ForwardRequestBuilderForTest() { Destroy(); }

  void CreateForwardRequest(int req_id, ForwardRequest& forward_req, const std::vector<int>& input_ids) {
    KLLM_CHECK_WITH_INFO(reqs_.find(req_id) == reqs_.end(), FormatStr("req_id: {} exists ", req_id));

    // fake KsanaPythonInput and req_ctx to create Request
    std::shared_ptr<KsanaPythonInput> ksana_python_input = std::make_shared<KsanaPythonInput>();
    ksana_python_input->sampling_config.num_beams = 0;
    ksana_python_input->sampling_config.num_return_sequences = 1;
    python_inputs_[req_id] = ksana_python_input;

    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>(
        std::unordered_map<std::string, std::string>{{"key1", "value1"}, {"key2", "value2"}});

    std::shared_ptr<Request> req = std::make_shared<Request>(ksana_python_input, req_ctx);
    req->req_id = req_id;
    req->input_tokens = input_ids;

    reqs_[req_id] = req;
    std::shared_ptr<InferRequest> infer_req = std::make_shared<InferRequest>(req, 0);
    infer_req->infer_stage = InferStage::STAGE_CONTEXT;  // This should be no need
    infer_req->cache_manager = cache_manager_;

    // Idealy, forward() related to these two variables.
    infer_req->forwarding_tokens = req->input_tokens;
    infer_req->kv_cached_token_num = 0;

    // alloc kv cache
    infer_req->kv_cache_blocks.resize(tensor_para_size_);
    int use_block_num = (input_ids.size() + block_token_num_ - 1) / block_token_num_;
    for (int rank = 0; rank < tensor_para_size_; ++rank) {
      // Alloc block without sharing
      // TODO(robertyuan): support prefix caching
      KLLM_CHECK_WITH_INFO(cache_manager_->GetBlockAllocatorGroup()
                               ->GetDeviceBlockAllocator(rank)
                               ->AllocateBlocks(use_block_num, infer_req->kv_cache_blocks[rank])
                               .OK(),
                           "faild to allocate blocks");
    }

    infer_reqs_[req_id] = infer_req;
    std::vector<float*> logits_buf;  // TODO(robertyuan): this buffer seems useless, but it refs to model_instance
    LlmRuntime::BuildForwardRequestFromInferRequest(forward_req, infer_req, layer_num_, logits_buf);
  }

 private:
  void Destroy() {
    for (const auto& pair : infer_reqs_) {
      auto& infer_req = pair.second;
      // release blocks
      for (int rank = 0; rank < tensor_para_size_; ++rank) {
        std::vector<int> block_ids;
        SetDevice(rank);
        KLLM_CHECK_WITH_INFO(cache_manager_->GetBlockAllocatorGroup()
                                 ->GetDeviceBlockAllocator(rank)
                                 ->FreeBlocks(infer_req->kv_cache_blocks[rank])
                                 .OK(),
                             "faild to free blocks");
      }
    }
  }

 private:
  uint32_t layer_num_;
  size_t block_token_num_;
  size_t tensor_para_size_;
  std::map<int, std::shared_ptr<KsanaPythonInput>> python_inputs_;
  std::map<int, std::shared_ptr<Request>> reqs_;
  std::map<int, std::shared_ptr<InferRequest>> infer_reqs_;

  std::shared_ptr<CacheManagerInterface> cache_manager_ = nullptr;
};
}  // namespace ksana_llm
