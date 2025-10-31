/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <algorithm>
#include <numeric>
#include <random>
#include <set>

#include "3rdparty/half/include/half.hpp"
#include "ksana_llm/layers/activation_layer.h"
#include "ksana_llm/layers/add_layer.h"
#include "ksana_llm/layers/assemble_tokens_hidden_layer.h"
#include "ksana_llm/layers/attention_layer.h"
#include "ksana_llm/layers/batched_matmul_layer.h"
#include "ksana_llm/layers/cutlass_matmul_layer.h"
#include "ksana_llm/layers/cutlass_moe_layer.h"
#include "ksana_llm/layers/emb_lookup_layer.h"
#include "ksana_llm/layers/flash_attention_layer.h"
#include "ksana_llm/layers/grouped_topk_layer.h"
#include "ksana_llm/layers/layer_workspace_manager.h"
#include "ksana_llm/layers/layernorm_layer.h"
#include "ksana_llm/layers/machete_matmul_layer.h"
#include "ksana_llm/layers/marlin_matmul_layer.h"
#include "ksana_llm/layers/marlin_moe_layer.h"
#include "ksana_llm/layers/matmul_layer.h"
#include "ksana_llm/layers/matmul_layer_factory.h"
#include "ksana_llm/layers/moe_layer.h"
#include "ksana_llm/layers/nccl_all_reduce_sum_layer.h"
#include "ksana_llm/layers/paged_attention_layer.h"
#include "ksana_llm/layers/silu_mul_layer.h"
#include "ksana_llm/models/common_moe/moe_config.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/search_status.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/cast/cast.h"
#  include "csrc/kernels/nvidia/permute/permute.h"
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#  ifdef ENABLE_FP8
#    include "ksana_llm/layers/fp8_moe_layer.h"
#  endif
#endif

namespace ksana_llm {

class LayerTest : public testing::Test {
 protected:
  // 在每个测试用例执行之前调用的函数
  void SetUp() override {
    model_config.path = "/model/llama-hf/7B/";
    model_config.weight_data_type = TYPE_FP16;
    model_config.head_num = 32;
    model_config.size_per_head = 128;
    model_config.inter_size = 11008;
    model_config.num_layer = 32;
    model_config.vocab_size = 32000;
    runtime_config.parallel_basic_config.tensor_parallel_size = 1;
    model_config.layernorm_eps = 1e-6;
    runtime_config.max_batch_size = 128;
    runtime_config.max_seq_len = 1024;
    model_config.rotary_embedding = 128;
    model_config.max_position_embeddings = 2048;
    model_config.rope_theta = 10000.0f;
    model_config.num_key_value_heads = model_config.head_num;

    BlockManagerConfig block_manager_config;
    block_manager_config.host_allocator_config.blocks_num = 2;
    block_manager_config.host_allocator_config.block_token_num = 16;
    block_manager_config.host_allocator_config.block_size = block_manager_config.host_allocator_config.block_token_num *
                                                            2 * model_config.head_num * model_config.size_per_head *
                                                            model_config.num_layer * sizeof(float16);
    block_manager_config.host_allocator_config.device = MEMORY_HOST;
    block_manager_config.device_allocator_config.blocks_num = 2;
    block_manager_config.device_allocator_config.block_token_num = 16;
    block_manager_config.device_allocator_config.block_size =
        block_manager_config.host_allocator_config.block_token_num * 2 * model_config.head_num *
        model_config.size_per_head * model_config.num_layer * sizeof(float16);
    KLLM_LOG_INFO << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);
    block_manager_config.device_allocator_config.device = MEMORY_DEVICE;

    runtime_config.attn_backend_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    runtime_config.attn_backend_config.block_size = block_manager_config.device_allocator_config.block_size;
    runtime_config.inter_data_type = model_config.weight_data_type;
    runtime_config.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::Default;

    Singleton<Environment>::GetInstance()->SetBlockManagerConfig(block_manager_config);
    context_ = std::make_shared<Context>(1, 1, 1);
  }

  // 在每个测试用例执行之后调用的函数
  void TearDown() override {}

  Status CreateHalfDataTypeTensor(Tensor& tensor, const std::vector<size_t>& shape, const DataType data_type,
                                  size_t dtype_size = 2) {
    tensor = Tensor(MemoryLocation::LOCATION_DEVICE, data_type, shape, 0);
    return Status();
  }

 protected:
  ModelConfig model_config;
  RuntimeConfig runtime_config;  // TODO(robertyuan): seems nobody use it
  std::shared_ptr<Context> context_{nullptr};
};

TEST_F(LayerTest, AttentionLayerTest) {
#ifndef ENABLE_CUDA
  GTEST_SKIP();
#endif

#ifdef ENABLE_CUDA
  std::shared_ptr<Context> context = std::make_shared<Context>(1, 1, 1);
  FlashAttentionLayer flash_attention_layer;
  QuantMode quant_mode = QUANT_NONE;
  int head_num = 32;
  int kv_head_num = 32;
  int size_per_head = 128;
  int rotary_embedding = 128;
  int max_position_embeddings = 2048;
  int max_batch_size = 1;
  size_t attn_temperature_tuning = 0;
  float attn_scale = 0;
  size_t floor_scale = 0;
  Tensor mrope_section_ptr;
  CreateHalfDataTypeTensor(mrope_section_ptr, {(size_t)rotary_embedding, (size_t)max_position_embeddings},
                           GetDataType<half>());
  bool enable_qk_pre_norm_before_rotary_pos = false;
  int stride_size = head_num * size_per_head;
  float k_scale = 1.0f;
  float v_scale = 1.0f;
  float rope_theta = 10000.0f;
  bool is_neox = true;
  uint32_t qk_nope_head_dim = 0;
  uint32_t v_head_dim = 0;
  uint32_t qk_rope_head_dim = 0;
  uint32_t q_lora_rank = 0;
  uint32_t kv_lora_rank = 0;
  Tensor cos_sin_cache_tensor;
  RoPEScalingFactor rope_scaling_factor;
  CreateHalfDataTypeTensor(cos_sin_cache_tensor, {(size_t)rotary_embedding, (size_t)max_position_embeddings},
                           GetDataType<half>());
  EXPECT_TRUE(flash_attention_layer
                  .Init({quant_mode,
                         static_cast<float>(0),
                         static_cast<bool>(false),
                         static_cast<int>(0),
                         static_cast<int>(1),
                         static_cast<int>(2048),
                         head_num,
                         kv_head_num,
                         size_per_head,
                         stride_size,
                         static_cast<size_t>(1),
                         TYPE_FP16,
                         k_scale,
                         v_scale,
                         rotary_embedding,
                         rope_theta,
                         v_head_dim,
                         qk_rope_head_dim,
                         qk_nope_head_dim,
                         q_lora_rank,
                         kv_lora_rank,
                         is_neox,
                         PositionEncoding::ROPE,
                         std::any(cos_sin_cache_tensor.GetPtr<void>()),
                         rope_scaling_factor,
                         max_batch_size,
                         attn_temperature_tuning,
                         attn_scale,
                         floor_scale,
                         true,
                         mrope_section_ptr,
                         enable_qk_pre_norm_before_rotary_pos},
                        runtime_config, context, 0)
                  .OK());

  Tensor qkv, input_len, prefix_offsets, pos, mask, forward_shape, flag_tensor, flexible_rotary_embedding_pos,
      flexible_rotary_embedding_mask, dst_flexible_kv_cache_tensor, src_flexible_kv_cache_tensor,
      dst_flexible_token_idx_tensor, src_flexible_token_idx_tensor;
  std::vector<size_t> input_shape = {2, 12288};
  CreateHalfDataTypeTensor(qkv, input_shape, GetDataType<half>());
  CreateHalfDataTypeTensor(input_len, {2}, GetDataType<uint64_t>(), sizeof(uint64_t));
  CreateHalfDataTypeTensor(prefix_offsets, {2}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(pos, {2}, GetDataType<uint64_t>(), /*dtype_size*/ sizeof(uint64_t));
  CreateHalfDataTypeTensor(mask, {2}, GetDataType<uint64_t>(), /*dtype_size*/ sizeof(uint64_t));

  flag_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_BOOL, {1}, 0);

  CreateHalfDataTypeTensor(flexible_rotary_embedding_pos, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(flexible_rotary_embedding_mask, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(dst_flexible_kv_cache_tensor, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(src_flexible_kv_cache_tensor, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(dst_flexible_token_idx_tensor, {0}, GetDataType<int>(), sizeof(int));
  CreateHalfDataTypeTensor(src_flexible_token_idx_tensor, {0}, GetDataType<int>(), sizeof(int));
  forward_shape.shape = {1, 2, 1, 0, 0, 0, 2, 1, 2, 0, 0, 0};
  void* pos_ptr = pos.GetPtr<void>();
  std::vector<uint64_t> pos_cpu({0, 1});
  Memcpy(pos_ptr, pos_cpu.data(), pos_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  void* mask_ptr = mask.GetPtr<void>();
  std::vector<uint64_t> mask_cpu({1, 1});
  Memcpy(mask_ptr, mask_cpu.data(), mask_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  void* input_len_ptr = input_len.GetPtr<void>();
  std::vector<uint64_t> input_len_cpu({0, 2});
  Memcpy(input_len_ptr, input_len_cpu.data(), input_len_cpu.size() * sizeof(uint64_t), MEMCPY_HOST_TO_DEVICE);
  Memset(prefix_offsets.GetPtr<void>(), 0, 2 * sizeof(int));
  flag_tensor.GetPtr<bool>()[0] = true;  // use_cache
  Tensor output_tensor;
  CreateHalfDataTypeTensor(output_tensor, input_shape, GetDataType<half>());
  std::vector<Tensor> output_tensors = {output_tensor};

  int block_size = runtime_config.attn_backend_config.block_size;
  std::vector<int> h_block_offsets = {0, 1};
  Tensor block_offsets;
  CreateHalfDataTypeTensor(block_offsets, {h_block_offsets.size()}, GetDataType<int>(), sizeof(int));
  Memcpy(block_offsets.GetPtr<void>(), h_block_offsets.data(), h_block_offsets.size() * sizeof(int),
         MEMCPY_HOST_TO_DEVICE);
  // 为 kv_list 分配内存并初始化
  Tensor kv_list;
  CreateHalfDataTypeTensor(kv_list, {static_cast<uint64_t>(h_block_offsets.back() * 20)}, GetDataType<uint64_t>());
  std::vector<void*> h_kv_list_ptrs(h_block_offsets.back() * 2);
  for (size_t i = 0; i < h_kv_list_ptrs.size(); i++) {
    Malloc(&h_kv_list_ptrs[i], block_size);
  }
  Memcpy(kv_list.GetPtr<void>(), h_kv_list_ptrs.data(), h_kv_list_ptrs.size() * sizeof(void*), MEMCPY_HOST_TO_DEVICE);

  // For blocke_prefill.
  Tensor layer_kv_cache_ptr_tensor;
  layer_kv_cache_ptr_tensor = Tensor(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<uint64_t>(1 + 2)}, 0);
  int64_t* kv_cache_block_num = layer_kv_cache_ptr_tensor.GetPtr<int64_t>();
  *kv_cache_block_num = static_cast<uint64_t>(1);
  void** layer_kv_cache_ptr = layer_kv_cache_ptr_tensor.GetPtr<void*>() + 1;
  layer_kv_cache_ptr[0] = h_kv_list_ptrs[0];
  layer_kv_cache_ptr[1] = h_kv_list_ptrs[1];

  std::vector<int32_t> multi_token_request_block_table_host = {0};
  Tensor multi_token_request_block_table;
  multi_token_request_block_table =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<uint64_t>(1), static_cast<uint64_t>(1)}, 0);
  Memcpy(multi_token_request_block_table.GetPtr<void>(), multi_token_request_block_table_host.data(),
         multi_token_request_block_table_host.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE);

  Tensor empty_q_norm_weight;
  Tensor empty_k_norm_weight;
  EXPECT_TRUE(flash_attention_layer
                  .Forward(
                      {
                          qkv,
                          input_len,
                          kv_list,
                          prefix_offsets,
                          block_offsets,
                          pos,
                          mask,
                          flexible_rotary_embedding_pos,
                          flexible_rotary_embedding_mask,
                          dst_flexible_kv_cache_tensor,
                          src_flexible_kv_cache_tensor,
                          dst_flexible_token_idx_tensor,
                          src_flexible_token_idx_tensor,
                          prefix_offsets,
                          forward_shape,
                          empty_q_norm_weight,
                          empty_k_norm_weight,
                          flag_tensor,
                          layer_kv_cache_ptr_tensor,
                          multi_token_request_block_table,
                          input_len,
                      },
                      output_tensors)
                  .OK());
  PagedAttentionLayer attention_layer;
  EXPECT_TRUE(attention_layer
                  .Init({quant_mode,
                         static_cast<float>(0),
                         static_cast<bool>(false),
                         static_cast<int>(1),
                         static_cast<int>(2),
                         static_cast<int>(2048),
                         static_cast<int>(head_num),
                         kv_head_num,
                         static_cast<int>(size_per_head),
                         stride_size,
                         static_cast<size_t>(1),
                         TYPE_FP16,
                         k_scale,
                         v_scale,
                         rotary_embedding,
                         rope_theta,
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         static_cast<uint32_t>(0),
                         is_neox,
                         PositionEncoding::ROPE,
                         std::any(cos_sin_cache_tensor.GetPtr<void>()),
                         rope_scaling_factor,
                         max_batch_size,
                         attn_temperature_tuning,
                         attn_scale,
                         floor_scale,
                         false,
                         nullptr,
                         enable_qk_pre_norm_before_rotary_pos},
                        runtime_config, context, 0)
                  .OK());
#endif
}

TEST_F(LayerTest, AddLayerTest) {
#ifndef ENABLE_TOPS

  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;

#  ifdef ENABLE_CUDA
  using device_type = half;
#  endif
#  ifdef ENABLE_ACL
  using device_type = aclFloat16;
#  endif

  // 初始化tensor
  Tensor input, bias_a, bias_b;
  std::vector<Tensor> output(1);
  input = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {32, 64}, kDeviceRank);
  bias_a = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {32, 64}, kDeviceRank);
  bias_b = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {1, 4, 16}, kDeviceRank);
  output[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {32, 64}, kDeviceRank);

  // 赋初值并拷贝到device
  std::vector<dtype> input_host(input.GetElementNumber());
  std::vector<dtype> bias_a_host(bias_a.GetElementNumber());
  std::vector<dtype> bias_b_host(bias_b.GetElementNumber());
  std::default_random_engine eng;
  std::uniform_real_distribution<float> random_range(-1, 1);
  for (size_t i = 0; i < input.GetElementNumber(); ++i) {
    input_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(input.GetPtr<void>(), input_host.data(), input.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->GetMemoryManageStreams()[kDeviceRank]);
  for (size_t i = 0; i < bias_a.GetElementNumber(); ++i) {
    bias_a_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(bias_a.GetPtr<void>(), bias_a_host.data(), bias_a.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->GetMemoryManageStreams()[kDeviceRank]);
  for (size_t i = 0; i < bias_b.GetElementNumber(); ++i) {
    bias_b_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(bias_b.GetPtr<void>(), bias_b_host.data(), bias_b.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->GetMemoryManageStreams()[kDeviceRank]);
  StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

  // 测试相同shape
  AddLayer add_layer = AddLayer();
  add_layer.Init({}, runtime_config, context_, kDeviceRank);
  add_layer.Forward({input, bias_a}, output);
  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
  // 验证结果
  std::vector<dtype> output_host(output[0].GetElementNumber());
  Memcpy(output_host.data(), output[0].GetPtr<void>(), output[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  for (size_t i = 0; i < output[0].GetElementNumber(); ++i) {
    EXPECT_FLOAT_EQ(output_host[i], input_host[i] + bias_a_host[i]);
  }

  // 测试broadcast shape
  add_layer.Forward({input, bias_b}, output);
  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
  // 验证结果
  Memcpy(output_host.data(), output[0].GetPtr<void>(), output[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  const size_t bias_b_elements = bias_b.GetElementNumber();
  for (size_t i = 0; i < output[0].GetElementNumber(); ++i) {
    EXPECT_FLOAT_EQ(output_host[i], input_host[i] + bias_b_host[i % bias_b_elements]);
  }
#endif
}

TEST_F(LayerTest, AssembleAcceptedTokensHiddenTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  using device_type = half;

  // 初始化tensor
  constexpr int batch_size = 2;
  constexpr size_t hidden_size = 1;
  Tensor input, accepted_tokens_idx;
  std::vector<Tensor> output(1);
  std::vector<int32_t> input_tokens_host = {2, 3};
  std::vector<int32_t> accepted_tokens_num_host = {2, 1};
  std::vector<size_t> accepted_tokens_idx_host;
  const size_t input_tokens_size = std::accumulate(input_tokens_host.begin(), input_tokens_host.end(), 0);
  const size_t accepted_tokens_size =
      std::accumulate(accepted_tokens_num_host.begin(), accepted_tokens_num_host.end(), 0);
  accepted_tokens_idx_host.reserve(accepted_tokens_size);
  int32_t input_token_offset = 0;
  int32_t output_token_offset = 0;
  std::vector<size_t> input_token_offsets;
  std::vector<size_t> output_token_offsets;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < accepted_tokens_num_host[i]; ++j) {
      accepted_tokens_idx_host.push_back(input_token_offset + j);
      printf("accepted_tokens_idx_host %d\n", input_token_offset + j);
    }
    input_token_offsets.push_back(input_token_offset);
    output_token_offsets.push_back(output_token_offset);
    input_token_offset += input_tokens_host[i];
    output_token_offset += accepted_tokens_num_host[i];
  }

  input = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {input_tokens_size, hidden_size}, kDeviceRank);
  accepted_tokens_idx = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {accepted_tokens_size}, kDeviceRank);
  output[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {accepted_tokens_size, hidden_size}, kDeviceRank);

  // 赋初值并拷贝到device
  std::vector<dtype> input_host(input.GetElementNumber());
  std::default_random_engine eng;
  std::uniform_real_distribution<float> random_range(-1, 1);
  for (size_t i = 0; i < input.GetElementNumber(); ++i) {
    input_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(input.GetPtr<void>(), input_host.data(), input.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->GetMemoryManageStreams()[kDeviceRank]);
  MemcpyAsync(accepted_tokens_idx.GetPtr<void>(), accepted_tokens_idx_host.data(), accepted_tokens_idx.GetTotalBytes(),
              MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[kDeviceRank]);
  StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

  AssembleTokensHiddenLayer test_layer = AssembleTokensHiddenLayer();
  test_layer.Init({}, runtime_config, context_, kDeviceRank);
  test_layer.Forward({input, accepted_tokens_idx}, output);
  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);

  // 验证结果
  std::vector<dtype> output_host(output[0].GetElementNumber());
  Memcpy(output_host.data(), output[0].GetPtr<void>(), output[0].GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);
  for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
    size_t accepted_num = accepted_tokens_num_host[sample_idx];
    for (size_t h = 0; h < hidden_size * accepted_num; ++h) {
      size_t in_start_offset = input_token_offsets[sample_idx] * hidden_size;
      size_t ou_start_offset = output_token_offsets[sample_idx] * hidden_size;
      EXPECT_FLOAT_EQ(input_host[in_start_offset + h], output_host[ou_start_offset + h]);
    }
  }

#endif
}

torch::Tensor GetRefGptq(torch::Tensor a, torch::Tensor pack_b, torch::Tensor b_scale) {
  int n = b_scale.size(1);
  int k = a.size(1);
  int groupsize = a.size(1) / b_scale.size(0);

  torch::Tensor w_packed_int4x2 = pack_b.t().contiguous().view(torch::kUInt8);
  torch::Tensor w_unpacked = torch::zeros({w_packed_int4x2.size(0), w_packed_int4x2.size(1) * 2},
                                          torch::TensorOptions().device(w_packed_int4x2.device()).dtype(torch::kInt8));
  w_unpacked.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                        w_packed_int4x2 % 16);
  w_unpacked.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                        w_packed_int4x2 / 16);
  w_unpacked = w_unpacked.t().contiguous();

  b_scale = b_scale.unsqueeze(1).repeat({1, groupsize, 1}).reshape({k, n}).contiguous();

  torch::Tensor b = b_scale * (w_unpacked - 8);

  return torch::matmul(a, b);
}

TEST_F(LayerTest, MacheteMatMulLayerTest) {
#ifdef ENABLE_CUDA

  if (context_->ext->GetComputeCapacity() != 90) {
    return;
  }

  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  using device_type = half;
  runtime_config.inter_data_type = TYPE_FP16;

  // 初始化参数
  const size_t max_m = 1024;
  const size_t max_n = 8192;
  const size_t max_k = 28672;
  const size_t groupsize = 128;
  const bool is_awq = false;
  const bool is_gptq_desc = false;
  const bool is_k_full = false;
  const bool cutlass_use_gemv_cuda_core = false;

  // 创建MacheteMatMulLayer实例
  MacheteMatMulLayer machete_matmul_layer;
  machete_matmul_layer.Init(
      {max_m, max_n, max_k, groupsize, is_awq, is_gptq_desc, is_k_full, cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
      runtime_config, context_, kDeviceRank);

  // 获取工作空间大小并分配
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  machete_matmul_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(machete_matmul_layer.GetWorkspaceSize()));

  // 准备输入张量
  const size_t m = 96;  // 实际使用的m值，小于max_m
  const size_t bits = 4;
  const size_t pack_factor = 32 / bits;

  // 创建输入数据
  Tensor input_activation = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_k}, kDeviceRank);
  Tensor weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_k / pack_factor, max_n}, kDeviceRank);
  Tensor scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {max_k / groupsize, max_n}, kDeviceRank);

  // 赋初值并拷贝到device，必要的步骤
  {
    std::default_random_engine eng;
    std::uniform_real_distribution<float> random_range(-1, 1);

    std::vector<dtype> input_activation_host(input_activation.GetElementNumber());
    for (size_t i = 0; i < input_activation.GetElementNumber(); ++i) {
      input_activation_host[i] = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(input_activation.GetPtr<void>(), input_activation_host.data(), input_activation.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[kDeviceRank]);

    std::vector<dtype> scales_host(scales.GetElementNumber());
    for (size_t i = 0; i < scales.GetElementNumber(); ++i) {
      scales_host[i] = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(scales.GetPtr<void>(), scales_host.data(), scales.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[kDeviceRank]);

    std::vector<int32_t> weight_host(weight.GetElementNumber());
    for (size_t i = 0; i < weight.GetElementNumber(); ++i) {
      weight_host[i] = static_cast<int32_t>(1000 * random_range(eng));
    }
    MemcpyAsync(weight.GetPtr<void>(), weight_host.data(), weight.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[kDeviceRank]);

    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
  }

  // 权重预处理
  Tensor weightT = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_k / pack_factor, max_n}, kDeviceRank);
  Tensor weightPrePack = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_k / pack_factor, max_n}, kDeviceRank);
  llm_kernels::nvidia::InvokePermute<2ul, sizeof(int32_t)>(weight.GetPtr<void>(), weightT.GetPtr<void>(),
                                                           {max_k / pack_factor, max_n}, {1, 0},
                                                           context_->GetComputeStreams()[kDeviceRank].Get());
  llm_kernels::nvidia::machete::machete_prepack_weight(
      weightT.GetPtr<void>(), {max_k / pack_factor, max_n}, weightPrePack.GetPtr<void>(),
      llm_kernels::nvidia::vllm_dtype::kHalf, llm_kernels::nvidia::vllm_dtype::kU4B8,
      llm_kernels::nvidia::vllm_dtype::kHalf, context_->GetComputeStreams()[kDeviceRank].Get());
  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);

  // 绑定scale
  weightPrePack.scales = &scales;

  // 执行默认矩阵计算
  Tensor output0 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_n}, kDeviceRank);
  int64_t default_duration = 0;
  {
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> output_tensors = {output0};
    for (size_t it = 0; it < 100; it++) {
      machete_matmul_layer.Forward({input_activation, weightPrePack}, output_tensors);
    }
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    auto end_time = std::chrono::high_resolution_clock::now();
    default_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
  }

  machete_matmul_layer.Preprocess(model_config, runtime_config);
  machete_matmul_layer.Preprocess(model_config, runtime_config);

  // 执行最优矩阵计算
  Tensor output1 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_n}, kDeviceRank);
  int64_t best_duration = 0;
  {
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> output_tensors = {output1};
    for (size_t it = 0; it < 100; it++) {
      machete_matmul_layer.Forward({input_activation, weightPrePack}, output_tensors);
    }
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    auto end_time = std::chrono::high_resolution_clock::now();
    best_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
  }

  printf("MacheteMatMulLayerTest default time: %ld ms, best time: %ld ms\n", default_duration, best_duration);
  EXPECT_TRUE((best_duration < default_duration) ||
              (std::abs(best_duration - default_duration) <
               0.01 * best_duration));  // 最优配置有可能刚好是默认配置，因此要允许1%的误差，不能强制小于

  // 按照GPTQ计算逻辑计算基准结果
  torch::Tensor t_weight =
      torch::from_blob(weight.GetPtr<void>(), {max_k / pack_factor, max_n}, torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor t_scale = torch::from_blob(scales.GetPtr<void>(), {max_k / groupsize, max_n},
                                           torch::TensorOptions().dtype(GetTorchDataType<device_type>()))
                              .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor t_a = torch::from_blob(input_activation.GetPtr<void>(), {m, max_k},
                                       torch::TensorOptions().dtype(GetTorchDataType<device_type>()))
                          .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor ref = GetRefGptq(t_a, t_weight, t_scale);

  // 验证输出形状
  EXPECT_EQ(output0.shape[0], m);
  EXPECT_EQ(output0.shape[1], max_n);
  EXPECT_EQ(output1.shape[0], m);
  EXPECT_EQ(output1.shape[1], max_n);

  // 验证结果
  std::vector<dtype> ref_host(m * max_n);
  std::vector<dtype> output0_host(m * max_n);
  std::vector<dtype> output1_host(m * max_n);
  Memcpy(ref_host.data(), ref.data_ptr(), ref_host.size() * sizeof(dtype), MEMCPY_DEVICE_TO_HOST);
  Memcpy(output0_host.data(), output0.GetPtr<void>(), output0_host.size() * sizeof(dtype), MEMCPY_DEVICE_TO_HOST);
  Memcpy(output1_host.data(), output1.GetPtr<void>(), output1_host.size() * sizeof(dtype), MEMCPY_DEVICE_TO_HOST);
  StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);
  for (size_t idx = 0; idx < m * max_n; idx++) {
    EXPECT_NEAR(ref_host[idx], output0_host[idx], 1.0);  // 算出来的结果数量级较大，不能用EXPECT_FLOAT_EQ完全比较
    EXPECT_NEAR(ref_host[idx], output1_host[idx], 1.0);
  }

#endif
}

TEST_F(LayerTest, CutlassMatMulLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  runtime_config.inter_data_type = TYPE_FP16;

  // 初始化参数
  const size_t max_m = 1024;
  const size_t max_n = 8192;
  const size_t max_k = 28672;
  const size_t groupsize = 128;
  const bool is_awq = false;
  const bool is_gptq_desc = false;
  const bool is_k_full = false;
  const bool cutlass_use_gemv_cuda_core = true;

  // 创建CutlassMatMulLayer实例
  CutlassMatMulLayer cutlass_matmul_layer;
  cutlass_matmul_layer.Init(
      {max_m, max_n, max_k, groupsize, is_awq, is_gptq_desc, is_k_full, cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
      runtime_config, context_, kDeviceRank);

  // 获取工作空间大小并分配
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  cutlass_matmul_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(cutlass_matmul_layer.GetWorkspaceSize()));

  // 准备输入张量
  const size_t m = 96;  // 实际使用的m值，小于max_m
  const size_t bits = 4;
  const size_t pack_factor = 32 / bits;

  Tensor input_activation = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_k}, kDeviceRank);
  Tensor output0 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_n}, kDeviceRank);
  Tensor output1 = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_n}, kDeviceRank);
  Tensor weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {max_k, max_n / 2}, kDeviceRank);
  Tensor scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {max_k / groupsize, max_n}, kDeviceRank);
  weight.scales = &scales;

  // 执行默认矩阵计算
  int64_t default_duration = 0;
  {
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> output_tensors = {output0};
    for (size_t it = 0; it < 100; it++) {
      cutlass_matmul_layer.Forward({input_activation, weight}, output_tensors);
    }
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    auto end_time = std::chrono::high_resolution_clock::now();
    default_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
  }

  cutlass_matmul_layer.Preprocess(model_config, runtime_config);
  cutlass_matmul_layer.Preprocess(model_config, runtime_config);

  // 执行最优矩阵计算
  int64_t best_duration = 0;
  {
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> output_tensors = {output1};
    for (size_t it = 0; it < 100; it++) {
      cutlass_matmul_layer.Forward({input_activation, weight}, output_tensors);
    }
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    auto end_time = std::chrono::high_resolution_clock::now();
    best_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
  }

  printf("CutlassMatMulLayerTest default time: %ld ms, best time: %ld ms\n", default_duration, best_duration);
  EXPECT_TRUE((best_duration < default_duration) ||
              (std::abs(best_duration - default_duration) <
               0.01 * best_duration));  // 最优配置有可能刚好只默认配置，因此要允许1%的误差，不能强制小于

  // 验证输出形状
  EXPECT_EQ(output0.shape[0], m);
  EXPECT_EQ(output0.shape[1], max_n);
  EXPECT_EQ(output1.shape[0], m);
  EXPECT_EQ(output1.shape[1], max_n);

  // 验证结果
  std::vector<dtype> output0_host(output0.GetElementNumber());
  std::vector<dtype> output1_host(output1.GetElementNumber());
  Memcpy(output0_host.data(), output0.GetPtr<void>(), output0.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  Memcpy(output1_host.data(), output1.GetPtr<void>(), output1.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);
  for (size_t idx = 0; idx < m * max_n; idx++) {
    EXPECT_FLOAT_EQ(output0_host[idx], output1_host[idx]);
  }
#endif
}

TEST_F(LayerTest, MarlinMatMulLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  using device_type = half;
  runtime_config.inter_data_type = TYPE_FP16;

  // 初始化参数
  const size_t max_m = 1024;
  const size_t max_n = 8192;
  const size_t max_k = 28672;
  const size_t groupsize = 128;
  const bool is_awq = false;
  const bool is_gptq_desc = false;
  const bool is_k_full = true;
  const bool cutlass_use_gemv_cuda_core = false;

  // 创建MarlinMatMulLayer实例
  MarlinMatMulLayer marlin_matmul_layer;
  marlin_matmul_layer.Init(
      {max_m, max_n, max_k, groupsize, is_awq, is_gptq_desc, is_k_full, cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
      runtime_config, context_, kDeviceRank);

  // 获取工作空间大小并分配
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  marlin_matmul_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(marlin_matmul_layer.GetWorkspaceSize()));

  // 准备输入张量
  const size_t m = 96;  // 实际使用的m值，小于max_m
  const size_t bits = 4;
  const size_t pack_factor = 32 / bits;

  Tensor input_activation = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_k}, kDeviceRank);
  Tensor weight = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_k / pack_factor, max_n}, kDeviceRank);
  Tensor scales = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {max_k / groupsize, max_n}, kDeviceRank);

  // 赋初值并拷贝到device，必要的步骤
  {
    std::default_random_engine eng;
    std::uniform_real_distribution<float> random_range(-1, 1);

    std::vector<dtype> input_activation_host(input_activation.GetElementNumber());
    for (size_t i = 0; i < input_activation.GetElementNumber(); ++i) {
      input_activation_host[i] = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(input_activation.GetPtr<void>(), input_activation_host.data(), input_activation.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[kDeviceRank]);

    std::vector<dtype> scales_host(scales.GetElementNumber());
    for (size_t i = 0; i < scales.GetElementNumber(); ++i) {
      scales_host[i] = static_cast<dtype>(random_range(eng));
    }
    MemcpyAsync(scales.GetPtr<void>(), scales_host.data(), scales.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[kDeviceRank]);

    std::vector<int32_t> weight_host(weight.GetElementNumber());
    for (size_t i = 0; i < weight.GetElementNumber(); ++i) {
      weight_host[i] = static_cast<int32_t>(1000 * random_range(eng));
    }
    MemcpyAsync(weight.GetPtr<void>(), weight_host.data(), weight.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[kDeviceRank]);

    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
  }

  // 权重预处理
  std::vector<int64_t> repack_shape = GetMarlinGptqRepackMeta(max_k, max_n, bits);
  Tensor weightPrePack =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {repack_shape[0], repack_shape[1]}, kDeviceRank);
  InvokeMarlinGptqRepack(weight.GetPtr<void>(), nullptr, weightPrePack.GetPtr<void>(), 1, max_k, max_n, bits, false,
                         kDeviceRank, context_->GetComputeStreams()[kDeviceRank].Get());
  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);

  // scale预处理
  Tensor scalesPrePack = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {max_k / groupsize, max_n}, kDeviceRank);
  InvokeMarlinPermuteScales<device_type>(context_->GetComputeStreams()[kDeviceRank].Get(), scales.GetPtr<void>(),
                                         scalesPrePack.GetPtr<void>(), max_k, max_n, groupsize);
  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);

  // 绑定scale
  weightPrePack.scales = &scalesPrePack;

  // 执行矩阵计算
  Tensor output = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, max_n}, kDeviceRank);
  int64_t default_duration = 0;
  {
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    auto begin_time = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> output_tensors = {output};
    for (size_t it = 0; it < 100; it++) {
      marlin_matmul_layer.Forward({input_activation, weightPrePack}, output_tensors);
    }
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    auto end_time = std::chrono::high_resolution_clock::now();
    default_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
  }

  printf("MarlinMatMulLayerTest time: %ld ms\n", default_duration);

  // 按照GPTQ计算逻辑计算基准结果
  torch::Tensor t_weight =
      torch::from_blob(weight.GetPtr<void>(), {max_k / pack_factor, max_n}, torch::TensorOptions().dtype(torch::kInt32))
          .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor t_scale = torch::from_blob(scales.GetPtr<void>(), {max_k / groupsize, max_n},
                                           torch::TensorOptions().dtype(GetTorchDataType<device_type>()))
                              .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor t_a = torch::from_blob(input_activation.GetPtr<void>(), {m, max_k},
                                       torch::TensorOptions().dtype(GetTorchDataType<device_type>()))
                          .to(torch::Device(torch::kCUDA, kDeviceRank));
  torch::Tensor ref = GetRefGptq(t_a, t_weight, t_scale);

  // 验证输出形状
  EXPECT_EQ(output.shape[0], m);
  EXPECT_EQ(output.shape[1], max_n);

  // 验证结果
  std::vector<dtype> ref_host(m * max_n);
  std::vector<dtype> output_host(m * max_n);
  Memcpy(ref_host.data(), ref.data_ptr(), ref_host.size() * sizeof(dtype), MEMCPY_DEVICE_TO_HOST);
  Memcpy(output_host.data(), output.GetPtr<void>(), output_host.size() * sizeof(dtype), MEMCPY_DEVICE_TO_HOST);
  StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);
  for (size_t idx = 0; idx < m * max_n; idx++) {
    EXPECT_NEAR(ref_host[idx], output_host[idx], 1.0);  // 算出来的结果数量级较大，不能用EXPECT_FLOAT_EQ完全比较
  }

#endif
}

TEST_F(LayerTest, BatchedMatMulLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;
  using device_type = half;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
  cublasCreate(&cublas_handle);
  cublasLtCreate(&cublaslt_handle);

  const int batch_size = 4;
  const int m = 32;
  const int n = 64;
  const int k = 128;

  BatchedMatMulLayer batched_matmul_layer;
  batched_matmul_layer.Init({}, runtime_config, context_, kDeviceRank);

  Tensor input_a = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, batch_size, k}, kDeviceRank);
  Tensor input_b = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {batch_size, k, n}, kDeviceRank);
  Tensor output = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, batch_size, n}, kDeviceRank);

  std::vector<dtype> input_a_host(input_a.GetElementNumber());
  std::vector<dtype> input_b_host(input_b.GetElementNumber());
  std::default_random_engine eng;
  std::uniform_real_distribution<float> random_range(-1, 1);

  for (size_t i = 0; i < input_a.GetElementNumber(); ++i) {
    input_a_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(input_a.GetPtr<void>(), input_a_host.data(), input_a.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->GetMemoryManageStreams()[kDeviceRank]);

  for (size_t i = 0; i < input_b.GetElementNumber(); ++i) {
    input_b_host[i] = static_cast<dtype>(random_range(eng));
  }
  MemcpyAsync(input_b.GetPtr<void>(), input_b_host.data(), input_b.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->GetMemoryManageStreams()[kDeviceRank]);

  StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

  std::vector<Tensor> output_tensors = {output};
  EXPECT_TRUE(batched_matmul_layer.Forward({input_a, input_b}, output_tensors).OK());
  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);

  // 验证输出形状
  EXPECT_EQ(output.shape.size(), 3);
  EXPECT_EQ(output.shape[0], m);
  EXPECT_EQ(output.shape[1], batch_size);
  EXPECT_EQ(output.shape[2], n);

  // 使用cublas计算参考结果进行验证
  Tensor ref_output = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {m, batch_size, n}, kDeviceRank);

  // 对每个batch单独计算矩阵乘法作为参考
  for (int b = 0; b < batch_size; ++b) {
    float alpha = 1.0f;
    float beta = 0.0f;

    void* a_ptr = static_cast<char*>(input_a.GetPtr<void>()) + b * k * sizeof(device_type);
    void* b_ptr = static_cast<char*>(input_b.GetPtr<void>()) + b * k * n * sizeof(device_type);
    void* c_ptr = static_cast<char*>(ref_output.GetPtr<void>()) + b * n * sizeof(device_type);

    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b_ptr, CUDA_R_16F, n, a_ptr, CUDA_R_16F,
                 batch_size * k, &beta, c_ptr, CUDA_R_16F, batch_size * n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
  }

  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);

  // 比较结果
  std::vector<dtype> output_host(output.GetElementNumber());
  std::vector<dtype> ref_output_host(ref_output.GetElementNumber());

  Memcpy(output_host.data(), output.GetPtr<void>(), output.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  Memcpy(ref_output_host.data(), ref_output.GetPtr<void>(), ref_output.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);

  for (size_t i = 0; i < output.GetElementNumber(); ++i) {
    EXPECT_NEAR(static_cast<float>(output_host[i]), static_cast<float>(ref_output_host[i]), 1e-2);
  }

  cublasDestroy(cublas_handle);
  cublasLtDestroy(cublaslt_handle);
#endif
}

TEST_F(LayerTest, MacheteSearchStatusTest) {
#ifdef ENABLE_CUDA
  if (context_->ext->GetComputeCapacity() != 90) {
    return;
  }
  constexpr int kDeviceRank = 0;

  // 初始化参数
  const size_t max_m = 32768;  // 32K
  std::vector<std::pair<const size_t, const size_t>> n_k_pairs = {
      {1024, 2048}, {2048, 1024}, {1024, 1024}, {2048, 2048}};
  const size_t groupsize = 128;
  const bool is_awq = false;
  const bool is_gptq_desc = false;
  const bool is_k_full = false;
  const bool cutlass_use_gemv_cuda_core = false;

  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  auto t1 = std::chrono::high_resolution_clock::now();
  {
    std::shared_ptr<MatMulLayerFactory> matmul_layer_factory =
        std::make_shared<MatMulLayerFactory>(model_config, runtime_config, kDeviceRank, context_);
    for (const auto& nk : n_k_pairs) {
      std::shared_ptr<BaseLayer> layer =
          matmul_layer_factory->CreateLayer(TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16,
                                            {max_m, nk.first, nk.second, groupsize, is_awq, is_gptq_desc, is_k_full,
                                             cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
                                            QUANT_GPTQ, MACHETE_LINEAR_BACKEND);
      layer->SetWorkspaceBuffer(workspace_mgr->GetWorkspace(layer->GetWorkspaceSize()));
      layer->Preprocess(model_config, runtime_config);
    }
  }

  Singleton<MacheteSearchStatus>::GetInstance()->ClearMacheteSchedule();
  Singleton<MacheteSearchStatus>::GetInstance()->ClearMacheteWorkspace();

  auto t2 = std::chrono::high_resolution_clock::now();
  {
    std::shared_ptr<MatMulLayerFactory> matmul_layer_factory =
        std::make_shared<MatMulLayerFactory>(model_config, runtime_config, kDeviceRank, context_);

    auto func = [&]() {
      for (size_t layer_idx = 0; layer_idx < model_config.num_layer; layer_idx++) {
        for (const auto& nk : n_k_pairs) {
          std::shared_ptr<BaseLayer> layer =
              matmul_layer_factory->CreateLayer(TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16,
                                                {max_m, nk.first, nk.second, groupsize, is_awq, is_gptq_desc, is_k_full,
                                                 cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
                                                QUANT_GPTQ, MACHETE_LINEAR_BACKEND);
          layer->SetWorkspaceBuffer(workspace_mgr->GetWorkspace(layer->GetWorkspaceSize()));
          layer->Preprocess(model_config, runtime_config);
        }
      }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
      threads.emplace_back(func);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  auto duration12 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  auto duration23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);

  printf("time1: %ld, time2: %ld\n", duration12.count(), duration23.count());

  // 有缓存，创建多次耗时不应该增加太多
  EXPECT_TRUE(2 * duration12.count() > duration23.count());

#endif
}

TEST_F(LayerTest, CutlassSearchStatusTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 初始化参数
  const size_t max_m = 8192;  // 8K
  std::vector<std::pair<const size_t, const size_t>> n_k_pairs = {
      {1024, 2048}, {2048, 1024}, {1024, 1024}, {2048, 2048}};
  const size_t groupsize = 128;
  const bool is_awq = false;
  const bool is_gptq_desc = false;
  const bool is_k_full = false;
  const bool cutlass_use_gemv_cuda_core = true;
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  auto t1 = std::chrono::high_resolution_clock::now();
  {
    std::shared_ptr<MatMulLayerFactory> matmul_layer_factory =
        std::make_shared<MatMulLayerFactory>(model_config, runtime_config, kDeviceRank, context_);
    for (const auto& nk : n_k_pairs) {
      std::shared_ptr<BaseLayer> layer =
          matmul_layer_factory->CreateLayer(TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16,
                                            {max_m, nk.first, nk.second, groupsize, is_awq, is_gptq_desc, is_k_full,
                                             cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
                                            QUANT_GPTQ, CUTLASS_LINEAR_BACKEND);
      layer->SetWorkspaceBuffer(workspace_mgr->GetWorkspace(layer->GetWorkspaceSize()));
      layer->Preprocess(model_config, runtime_config);
    }
  }

  Singleton<CutlassSearchStatus>::GetInstance()->ClearCutlassSchedule();
  Singleton<CutlassSearchStatus>::GetInstance()->ClearCutlassWorkspace();

  auto t2 = std::chrono::high_resolution_clock::now();
  {
    std::shared_ptr<MatMulLayerFactory> matmul_layer_factory =
        std::make_shared<MatMulLayerFactory>(model_config, runtime_config, kDeviceRank, context_);

    auto func = [&]() {
      for (size_t layer_idx = 0; layer_idx < model_config.num_layer; layer_idx++) {
        for (const auto& nk : n_k_pairs) {
          std::shared_ptr<BaseLayer> layer =
              matmul_layer_factory->CreateLayer(TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16,
                                                {max_m, nk.first, nk.second, groupsize, is_awq, is_gptq_desc, is_k_full,
                                                 cutlass_use_gemv_cuda_core, TYPE_I4_GROUP},
                                                QUANT_GPTQ, CUTLASS_LINEAR_BACKEND);
          layer->SetWorkspaceBuffer(workspace_mgr->GetWorkspace(layer->GetWorkspaceSize()));
          layer->Preprocess(model_config, runtime_config);
        }
      }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
      threads.emplace_back(func);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  auto duration12 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  auto duration23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);

  printf("time1: %ld, time2: %ld\n", duration12.count(), duration23.count());

  // 有缓存，创建多次耗时不应该增加太多
  EXPECT_TRUE(2 * duration12.count() > duration23.count());

#endif
}

TEST_F(LayerTest, Fp8MoeLayerTest) {
#ifdef ENABLE_CUDA
#  ifdef ENABLE_FP8
  if (!context_->IsGemmFp8Supported()) {
    return;
  }
  constexpr int kDeviceRank = 0;

  // params
  MoeScaleNormMode moe_scale_norm_mode = ksana_llm::MoeScaleNormMode::NO_NORM;
  size_t max_token_num = 4096;
  size_t expert_num = 4;
  size_t expert_hidden_size = 1024;
  size_t expert_inter_size = 2688;
  size_t expert_topk = 1;
  size_t tp_size = 1;
  bool use_vllm_moe = false;
  uint32_t num_expert_group = 1;
  uint32_t expert_groups_topk = 1;
  std::string scoring_func = "softmax";
  std::string topk_method = "greedy";
  bool norm_topk_prob = false;
  float routed_scaling_factor = 1.0f;
  bool use_e_score_correction_bias = false;
  bool enable_full_shared_expert = false;
  DataType fp8_weight_dtype = DataType::TYPE_INVALID;
  DataType int_weight_dtype = DataType::TYPE_INVALID;
  int group_size = 0;
  bool apply_weight = false;
  int layer_idx = 0;

  std::vector<std::any> params;
  params.push_back(moe_scale_norm_mode);
  params.push_back(max_token_num);
  params.push_back(layer_idx);
  params.push_back(expert_num);
  params.push_back(expert_hidden_size);
  params.push_back(expert_inter_size);
  params.push_back(expert_topk);
  params.push_back(tp_size);
  params.push_back(use_vllm_moe);
  params.push_back(num_expert_group);
  params.push_back(expert_groups_topk);
  params.push_back(scoring_func);
  params.push_back(topk_method);
  params.push_back(norm_topk_prob);
  params.push_back(routed_scaling_factor);
  params.push_back(use_e_score_correction_bias);
  params.push_back(enable_full_shared_expert);
  params.push_back(fp8_weight_dtype);
  params.push_back(int_weight_dtype);
  params.push_back(group_size);
  params.push_back(apply_weight);

  int num_tokens = 9;
  auto options = torch::TensorOptions().device(torch::kCUDA, kDeviceRank);

  // initialize input and weight tensor
  std::vector<Tensor> inputs(4);
  inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {num_tokens, expert_hidden_size}, kDeviceRank);
  inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {expert_num, expert_num}, kDeviceRank);
  inputs[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP8_E4M3,
                     {expert_num, expert_inter_size * 2, expert_hidden_size}, kDeviceRank);
  inputs[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP8_E4M3,
                     {expert_num, expert_hidden_size, expert_inter_size}, kDeviceRank);
  for (int i = 0; i < inputs.size(); ++i) {
    torch::Tensor tensor = torch::from_blob(inputs[i].GetPtr<void>(),
                                            {std::vector<int64_t>(inputs[i].shape.begin(), inputs[i].shape.end())},
                                            options.dtype(GetTorchTypeFromDataType(inputs[i].dtype)));
    if (i < 2) {
      tensor.fill_(0.8213);
    } else {
      tensor.fill_(89);
    }
  }
  inputs.push_back(Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {0}, kDeviceRank));

  // initialize scales tensor
  std::vector<Tensor> scales(4);
  scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1}, kDeviceRank);
  scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1}, kDeviceRank);
  scales[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1}, kDeviceRank);
  scales[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1}, kDeviceRank);
  for (int i = 0; i < scales.size(); ++i) {
    torch::Tensor tensor = torch::from_blob(scales[i].GetPtr<void>(),
                                            {std::vector<int64_t>(scales[i].shape.begin(), scales[i].shape.end())},
                                            options.dtype(GetTorchTypeFromDataType(scales[i].dtype)));
    if (i < 2) {
      tensor.fill_(0.01);
    } else {
      tensor.fill_(1.8601190e-06);
    }
  }

  // binding scales
  inputs[2].input_scales = &scales[0];
  inputs[3].input_scales = &scales[1];
  inputs[2].weight_scales = &scales[2];
  inputs[3].weight_scales = &scales[3];

  // initialize output tensor
  std::vector<Tensor> outputs(1);
  outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {num_tokens, expert_hidden_size}, kDeviceRank);

  // run moe_layer
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  Fp8MoeLayer moe_layer = Fp8MoeLayer();
  moe_layer.Init(params, runtime_config, context_, kDeviceRank);
  moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(moe_layer.GetWorkspaceSize()));
  moe_layer.Preprocess(model_config, runtime_config);
  EXPECT_TRUE(moe_layer.Forward(inputs, outputs).OK());

  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);

  // check output
  // output is a repetition of 0.1768
  torch::Tensor tensor = torch::from_blob(outputs[0].GetPtr<void>(),
                                          {std::vector<int64_t>(outputs[0].shape.begin(), outputs[0].shape.end())},
                                          options.dtype(GetTorchTypeFromDataType(outputs[0].dtype)))
                             .cpu();
  EXPECT_TRUE(torch::all(torch::eq(tensor[0], 0.1768)).item<bool>());

#  endif
#endif
}

TEST_F(LayerTest, MarlinMoeLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  int num_bits = 4;
  // params
  MoeScaleNormMode moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;
  size_t max_token_num = 4096;
  size_t expert_num = 4;
  size_t expert_hidden_size = 1024;
  size_t expert_inter_size = 2688;
  size_t expert_topk = 1;
  size_t tp_size = 1;
  bool use_vllm_moe = false;
  uint32_t num_expert_group = 1;
  uint32_t expert_groups_topk = 1;
  std::string scoring_func = "softmax";
  std::string topk_method = "greedy";
  bool norm_topk_prob = false;
  float routed_scaling_factor = 1.0f;
  bool use_e_score_correction_bias = false;
  bool enable_full_shared_expert = false;
  DataType fp8_weight_dtype = DataType::TYPE_INVALID;
  DataType int_weight_dtype = DataType::TYPE_I4_GROUP;
  int group_size = 128;
  bool apply_weight = false;

  std::vector<std::any> params;
  params.push_back(moe_scale_norm_mode);
  params.push_back(max_token_num);
  params.push_back(expert_num);
  params.push_back(expert_hidden_size);
  params.push_back(expert_inter_size);
  params.push_back(expert_topk);
  params.push_back(tp_size);
  params.push_back(use_vllm_moe);
  params.push_back(num_expert_group);
  params.push_back(expert_groups_topk);
  params.push_back(scoring_func);
  params.push_back(topk_method);
  params.push_back(norm_topk_prob);
  params.push_back(routed_scaling_factor);
  params.push_back(use_e_score_correction_bias);
  params.push_back(enable_full_shared_expert);
  params.push_back(fp8_weight_dtype);
  params.push_back(int_weight_dtype);
  params.push_back(group_size);
  params.push_back(apply_weight);

  int num_tokens = 9;
  auto options = torch::TensorOptions().device(torch::kCUDA, kDeviceRank);

  // initialize input and weight tensor
  std::vector<Tensor> inputs(4);
  inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {num_tokens, expert_hidden_size}, kDeviceRank);
  inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {num_tokens, expert_num}, kDeviceRank);
  inputs[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                     {expert_num, expert_hidden_size / 16, 2 * expert_inter_size * (num_bits / 2)}, kDeviceRank);
  inputs[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                     {expert_num, expert_inter_size / 16, expert_hidden_size * (num_bits / 2)}, kDeviceRank);
  for (int i = 0; i < inputs.size(); ++i) {
    torch::Tensor tensor = torch::from_blob(inputs[i].GetPtr<void>(),
                                            {std::vector<int64_t>(inputs[i].shape.begin(), inputs[i].shape.end())},
                                            options.dtype(GetTorchTypeFromDataType(inputs[i].dtype)));
    if (i < 2) {
      tensor.fill_(0.8213);
    } else {
      tensor.fill_(1754889370);
    }
  }
  inputs.push_back(Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {0}, kDeviceRank));

  // initialize scales tensor
  std::vector<Tensor> scales(2);
  scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                     {expert_num, expert_hidden_size / group_size, expert_inter_size * 2}, kDeviceRank);
  scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                     {expert_num, expert_inter_size / group_size, expert_hidden_size}, kDeviceRank);
  for (int i = 0; i < scales.size(); ++i) {
    torch::Tensor tensor = torch::from_blob(scales[i].GetPtr<void>(),
                                            {std::vector<int64_t>(scales[i].shape.begin(), scales[i].shape.end())},
                                            options.dtype(GetTorchTypeFromDataType(scales[i].dtype)));
    tensor.fill_(0.002735);
  }

  // binding scales
  inputs[2].scales = &scales[0];
  inputs[3].scales = &scales[1];

  // initialize output tensor
  std::vector<Tensor> outputs(1);
  outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {num_tokens, expert_hidden_size}, kDeviceRank);

  // run moe_layer
  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  MarlinMoeLayer moe_layer = MarlinMoeLayer();
  moe_layer.Init(params, runtime_config, context_, kDeviceRank);
  moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(moe_layer.GetWorkspaceSize()));
  moe_layer.Preprocess(model_config, runtime_config);
  EXPECT_TRUE(moe_layer.Forward(inputs, outputs).OK());

  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);

  // check output
  // output is a repetition of: [10.984, 10.984, 10.984, 10.984, 10.984, 10.984, 10.984, 10.984,
  // 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1, 14.1]
  torch::Tensor tensor = torch::from_blob(outputs[0].GetPtr<void>(),
                                          {std::vector<int64_t>(outputs[0].shape.begin(), outputs[0].shape.end())},
                                          options.dtype(GetTorchTypeFromDataType(outputs[0].dtype)))
                             .cpu()
                             .view({outputs[0].shape[0] * outputs[0].shape[1] / 16, 2, 8})
                             .permute({1, 0, 2});
  EXPECT_TRUE(torch::all(torch::eq(tensor[0], 10.984)).item<bool>());
  EXPECT_TRUE(torch::all(torch::eq(tensor[1], 14.1)).item<bool>());
#endif
}

TEST_F(LayerTest, GroupedTopkLayerTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = half_float::half;

  // 测试参数
  const int num_tokens = 4;
  const int num_experts = 8;
  const int topk = 2;
  const bool renormalize = true;
  const int num_expert_group = 4;
  const int topk_group = 2;
  const std::string scoring_func = "softmax";
  const float routed_scaling_factor = 1.0f;
  const bool use_e_score_correction_bias = false;

  // 创建 GroupedTopkLayer
  GroupedTopkLayer grouped_topk_layer;

  // 测试初始化
  std::vector<std::any> parameters = {topk,         renormalize,           num_expert_group,           topk_group,
                                      scoring_func, routed_scaling_factor, use_e_score_correction_bias};

  EXPECT_TRUE(grouped_topk_layer.Init(parameters, runtime_config, context_, kDeviceRank).OK());

  // 准备输入张量 - gating_output
  Tensor gating_output;
  gating_output = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                         {static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)}, kDeviceRank);

  // 初始化 gating_output 数据 - 使用不同的值来测试 topk 选择
  std::vector<dtype> gating_host(num_tokens * num_experts);
  for (int token = 0; token < num_tokens; ++token) {
    for (int expert = 0; expert < num_experts; ++expert) {
      // 为每个 token 设置不同的专家权重，确保 topk 选择有意义
      gating_host[token * num_experts + expert] = static_cast<dtype>((expert + token * 0.1f + 1.0f) / num_experts);
    }
  }
  MemcpyAsync(gating_output.GetPtr<void>(), gating_host.data(), gating_output.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->GetMemoryManageStreams()[kDeviceRank]);

  // 准备输出张量
  Tensor topk_weights, topk_ids;
  topk_weights = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                        {static_cast<size_t>(num_tokens), static_cast<size_t>(topk)}, kDeviceRank);
  topk_ids = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                    {static_cast<size_t>(num_tokens), static_cast<size_t>(topk)}, kDeviceRank);

  std::vector<Tensor> input_tensors = {gating_output};
  std::vector<Tensor> output_tensors = {topk_weights, topk_ids};

  StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

  // 测试前向传播
  EXPECT_TRUE(grouped_topk_layer.Forward(input_tensors, output_tensors).OK());
  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);

  // 验证输出
  std::vector<float> weights_host(num_tokens * topk);
  std::vector<int32_t> ids_host(num_tokens * topk);

  Memcpy(weights_host.data(), topk_weights.GetPtr<void>(), topk_weights.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);
  Memcpy(ids_host.data(), topk_ids.GetPtr<void>(), topk_ids.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST);

  // 验证 topk_ids 在有效范围内
  for (int i = 0; i < num_tokens * topk; ++i) {
    EXPECT_GE(ids_host[i], 0);
    EXPECT_LT(ids_host[i], num_experts);
  }

  // 验证权重为正数（softmax 输出）
  for (int i = 0; i < num_tokens * topk; ++i) {
    EXPECT_GT(weights_host[i], 0.0f);
    EXPECT_LE(weights_host[i], 1.0f);
  }

  // 验证每个 token 的权重和接近 1.0（如果 renormalize=true）
  if (renormalize) {
    for (int token = 0; token < num_tokens; ++token) {
      float weight_sum = 0.0f;
      for (int k = 0; k < topk; ++k) {
        weight_sum += weights_host[token * topk + k];
      }
      EXPECT_NEAR(weight_sum, 1.0f, 0.01f);  // 允许小的数值误差
    }
  }

  // 验证 topk 选择的正确性 - 检查选中的专家确实是权重最大的
  for (int token = 0; token < num_tokens; ++token) {
    std::vector<std::pair<float, int>> expert_weights;
    for (int expert = 0; expert < num_experts; ++expert) {
      expert_weights.push_back({static_cast<float>(gating_host[token * num_experts + expert]), expert});
    }
    std::sort(expert_weights.rbegin(), expert_weights.rend());  // 降序排列

    // 检查选中的专家是否在前 topk 中
    std::set<int> expected_experts;
    for (int k = 0; k < topk; ++k) {
      expected_experts.insert(expert_weights[k].second);
    }

    std::set<int> actual_experts;
    for (int k = 0; k < topk; ++k) {
      actual_experts.insert(ids_host[token * topk + k]);
    }

    EXPECT_EQ(expected_experts, actual_experts);
  }

  // 测试带 e_bias 的情况
  GroupedTopkLayer grouped_topk_layer_with_bias;
  std::vector<std::any> parameters_with_bias = {
      topk, renormalize, num_expert_group, topk_group, scoring_func, routed_scaling_factor,
      true  // use_e_score_correction_bias = true
  };

  EXPECT_TRUE(grouped_topk_layer_with_bias.Init(parameters_with_bias, runtime_config, context_, kDeviceRank).OK());

  // 准备 e_bias 张量
  Tensor e_bias(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(num_experts)}, kDeviceRank);
  std::vector<float> e_bias_host(num_experts);
  for (int i = 0; i < num_experts; ++i) {
    e_bias_host[i] = 0.1f * i;  // 简单的偏置值
  }
  MemcpyAsync(e_bias.GetPtr<void>(), e_bias_host.data(), e_bias.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context_->GetMemoryManageStreams()[kDeviceRank]);

  std::vector<Tensor> input_tensors_with_bias = {gating_output, e_bias};
  std::vector<Tensor> output_tensors_with_bias = {topk_weights, topk_ids};

  StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

  // 测试带偏置的前向传播
  EXPECT_TRUE(grouped_topk_layer_with_bias.Forward(input_tensors_with_bias, output_tensors_with_bias).OK());
  StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
#endif
}

TEST_F(LayerTest, MoeLayerTest) {
#ifdef ENABLE_CUDA
  if (context_->ext->GetComputeCapacity() != 90) {
    return;
  }
  constexpr int kDeviceRank = 0;
  using dtype = __nv_bfloat16;
  using alpha_dtype = float;

  ModelConfig new_model_config = model_config;
  RuntimeConfig new_runtime_config = runtime_config;
  new_model_config.weight_data_type = TYPE_BF16;
  new_runtime_config.inter_data_type = new_model_config.weight_data_type;

  torch::manual_seed(42);

  // params
  MoeScaleNormMode moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;  // 用不到
  size_t max_token_num = 4096;
  size_t expert_num = 32;
  size_t expert_hidden_size = 7168;
  size_t expert_inter_size = 2048;
  size_t expert_topk = 8;
  size_t tp_size = 1;
  bool use_vllm_moe = true;
  uint32_t num_expert_group = 8;
  uint32_t expert_groups_topk = 4;
  std::string scoring_func = "sigmoid";
  std::string topk_method = "";  // 用不到
  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;
  bool use_e_score_correction_bias = false;  // 关闭方便测试
  bool enable_full_shared_expert = false;
  DataType fp8_weight_dtype = DataType::TYPE_INVALID;
  DataType int_weight_dtype = DataType::TYPE_UINT4x2;
  int group_size = 128;
  bool apply_weight = false;  // 用不到
  int layer_idx = 0;

  std::vector<std::any> params;
  params.push_back(moe_scale_norm_mode);
  params.push_back(max_token_num);
  params.push_back(layer_idx);
  params.push_back(expert_num);
  params.push_back(expert_hidden_size);
  params.push_back(expert_inter_size);
  params.push_back(expert_topk);
  params.push_back(tp_size);
  params.push_back(use_vllm_moe);
  params.push_back(num_expert_group);
  params.push_back(expert_groups_topk);
  params.push_back(scoring_func);
  params.push_back(topk_method);
  params.push_back(norm_topk_prob);
  params.push_back(routed_scaling_factor);
  params.push_back(use_e_score_correction_bias);
  params.push_back(enable_full_shared_expert);
  params.push_back(fp8_weight_dtype);
  params.push_back(int_weight_dtype);
  params.push_back(group_size);
  params.push_back(apply_weight);

  auto int8_option = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
  auto int32_option = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto float_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto dtype_option = torch::TensorOptions().dtype(GetTorchDataType<dtype>()).device(torch::kCUDA);
  auto alpha_dtype_option = torch::TensorOptions().dtype(GetTorchDataType<alpha_dtype>()).device(torch::kCUDA);

  auto updateInterleave = [](size_t hidden_size, size_t intermediate_size_per_partition) {
    std::vector<size_t> interleave;
    std::vector<size_t> k_shapes = {hidden_size, intermediate_size_per_partition};
    for (const size_t& k_shape : k_shapes) {
      if (k_shape % 512 == 0) {
        interleave.push_back(4);
      } else if (k_shape % 256 == 0) {
        interleave.push_back(2);
      } else if (k_shape % 128 == 0) {
        interleave.push_back(1);
      } else {
        throw std::runtime_error(fmt::format("K shape is required to be multiple of 128, received {}.", k_shape));
      }
    }
    return interleave;
  };

  // NOTE(jinxcwu) 这里-8是uint4转int4。权重都是uint4的，但moe_layer那边需要的是int4，为了对齐需要额外实现
  auto unpack_int4_packed_tensor_to_int8 = [](torch::Tensor weight) {
    std::vector<int64_t> int8_tensor_size(weight.dim());
    for (int i = 0; i < weight.dim(); ++i) {
      int8_tensor_size[i] = weight.size(i);
    }
    int8_tensor_size[weight.dim() - 1] *= 2;

    torch::Tensor unpacked_weight =
        torch::zeros(int8_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    int8_t* packed_ptr = static_cast<int8_t*>(weight.data_ptr());
    int8_t* unpacked_ptr = static_cast<int8_t*>(unpacked_weight.data_ptr());

    for (int64_t packed_idx = 0; packed_idx < weight.numel(); ++packed_idx) {
      int8_t packed_data = packed_ptr[packed_idx];

      int8_t elt_0 = (int8_t(packed_data << 4) >> 4);  // The double shift here is to ensure sign extension
      int8_t elt_1 = packed_data >> 4;

      unpacked_ptr[2 * packed_idx + 0] = elt_0;
      unpacked_ptr[2 * packed_idx + 1] = elt_1;
    }

    return unpacked_weight;
  };

  auto ConvertPackUint4ToPackInt4 = [](const torch::Tensor& qweight) {
    auto origin_shape = qweight.sizes();
    auto weight = qweight.view(torch::kUInt8).view({-1});
    auto low = torch::bitwise_and(((weight % 16).to(torch::kInt8) - 8), 0x0F);
    auto high = torch::bitwise_and(((weight / 16).to(torch::kInt8) - 8), 0x0F);
    weight = torch::bitwise_or(torch::bitwise_left_shift(high, 4), low);
    weight = weight.view(origin_shape);
    return weight;
  };

  size_t num_tokens = 4;
  // 创建输入
  torch::Tensor hidden_states =
      torch::randn({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_hidden_size)}, dtype_option);
  torch::Tensor routing_out =
      torch::randn({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_num)}, dtype_option);
  // 创建输出
  torch::Tensor cutlass_output =
      torch::zeros({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_hidden_size)}, dtype_option);
  torch::Tensor triton_output =
      torch::zeros({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_hidden_size)}, dtype_option);

  // 创建权重
  float affine_coeff = 0.005f;
  std::map<std::string, torch::Tensor> weights;
  for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
    weights[fmt::format("{}.w1.weight", expert_id)] = torch::randint(
        -128, 127, {static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / 2)},
        int8_option);
    weights[fmt::format("{}.w2.weight", expert_id)] = torch::randint(
        -128, 127, {static_cast<int64_t>(expert_hidden_size), static_cast<int64_t>(expert_inter_size / 2)},
        int8_option);
    weights[fmt::format("{}.w3.weight", expert_id)] = torch::randint(
        -128, 127, {static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / 2)},
        int8_option);

    weights[fmt::format("{}.w1.weight_scale_inv", expert_id)] =
        torch::randn({static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / group_size)},
                     dtype_option) *
        affine_coeff;
    weights[fmt::format("{}.w2.weight_scale_inv", expert_id)] =
        torch::randn({static_cast<int64_t>(expert_hidden_size), static_cast<int64_t>(expert_inter_size / group_size)},
                     dtype_option) *
        affine_coeff;
    weights[fmt::format("{}.w3.weight_scale_inv", expert_id)] =
        torch::randn({static_cast<int64_t>(expert_inter_size), static_cast<int64_t>(expert_hidden_size / group_size)},
                     dtype_option) *
        affine_coeff;

    // NOTE(jinxcwu) input_scale用1结果会更稳定
    // torch::Tensor input_scale = torch::randn({static_cast<int64_t>(1)}, float_option) * 0.02;
    torch::Tensor input_scale = torch::ones({static_cast<int64_t>(1)}, float_option);
    weights[fmt::format("{}.w1.input_scale", expert_id)] = input_scale.clone();
    weights[fmt::format("{}.w2.input_scale", expert_id)] = input_scale.clone();
    weights[fmt::format("{}.w3.input_scale", expert_id)] = input_scale.clone();
  }

  {
    std::vector<size_t> interleave = updateInterleave(expert_hidden_size, expert_inter_size);

    // 权重加载
    // 步骤1: 创建一份空权重
    torch::Tensor w1_w3_weight =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_inter_size * 2),
                      static_cast<int64_t>(expert_hidden_size / 2)},
                     int8_option);
    torch::Tensor w2_weight = torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_hidden_size),
                                            static_cast<int64_t>(expert_inter_size / 2)},
                                           int8_option);
    torch::Tensor fc13_act_scale =
        torch::empty({static_cast<int64_t>(1), static_cast<int64_t>(expert_hidden_size)}, dtype_option);
    torch::Tensor fc2_act_scale =
        torch::empty({static_cast<int64_t>(1), static_cast<int64_t>(expert_inter_size)}, dtype_option);
    torch::Tensor fc13_weight_scale = torch::empty(
        {static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_hidden_size / (128 * interleave[0])),
         static_cast<int64_t>(expert_inter_size * 2 * interleave[0])},
        dtype_option);
    torch::Tensor fc2_weight_scale =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_inter_size / (128 * interleave[1])),
                      static_cast<int64_t>(expert_hidden_size * interleave[1])},
                     dtype_option);
    torch::Tensor fc13_alpha =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(1)}, alpha_dtype_option);
    torch::Tensor fc2_alpha =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(1)}, alpha_dtype_option);
    // 步骤2 处理weight
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      w1_w3_weight[expert_id].copy_(ConvertPackUint4ToPackInt4(torch::cat(
          {weights[fmt::format("{}.w1.weight", expert_id)], weights[fmt::format("{}.w3.weight", expert_id)]}, 0)));
      w2_weight[expert_id].copy_(ConvertPackUint4ToPackInt4(weights[fmt::format("{}.w2.weight", expert_id)]));
    }
    // 步骤3 处理act scale和alpha
    float all_w2_input_scales_max = std::numeric_limits<float>::min();
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w2_input_scales_max = std::max(
          all_w2_input_scales_max, torch::max(weights[fmt::format("{}.w2.input_scale", expert_id)]).item<float>());
    }
    fc2_act_scale.copy_((torch::ones_like(fc2_act_scale) * (1 / all_w2_input_scales_max)).to(fc2_act_scale.dtype()));
    fc2_alpha.copy_((torch::ones_like(fc2_alpha) * all_w2_input_scales_max).to(fc2_alpha.dtype()));
    // 步骤4 处理act scale和alpha
    float all_w1_w3_input_scales_max = std::numeric_limits<float>::min();
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w1_w3_input_scales_max = std::max(
          all_w1_w3_input_scales_max, torch::max(weights[fmt::format("{}.w1.input_scale", expert_id)]).item<float>());
      all_w1_w3_input_scales_max = std::max(
          all_w1_w3_input_scales_max, torch::max(weights[fmt::format("{}.w3.input_scale", expert_id)]).item<float>());
    }
    fc13_act_scale.copy_(
        (torch::ones_like(fc13_act_scale) * (1 / all_w1_w3_input_scales_max)).to(fc13_act_scale.dtype()));
    fc13_alpha.copy_((torch::ones_like(fc13_alpha) * all_w1_w3_input_scales_max).to(fc13_alpha.dtype()));
    // 步骤5
    std::vector<torch::Tensor> all_w2_scales;
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w2_scales.push_back(weights[fmt::format("{}.w2.weight_scale_inv", expert_id)]);
    }
    torch::Tensor w2_scales = torch::stack(all_w2_scales).to(torch::kBFloat16).view(GetTorchDataType<dtype>());
    auto w2_s_shape = w2_scales.sizes();
    torch::Tensor w2_scales_interleaved =
        w2_scales.reshape({w2_s_shape[0], w2_s_shape[1], w2_s_shape[2] / interleave[1], interleave[1]});
    w2_scales_interleaved = w2_scales_interleaved.permute({0, 2, 1, 3});
    w2_scales_interleaved =
        w2_scales_interleaved.reshape({w2_s_shape[0], w2_s_shape[2] / interleave[1], w2_s_shape[1] * interleave[1]});
    fc2_weight_scale.copy_(w2_scales_interleaved.contiguous());
    // 步骤6
    std::vector<torch::Tensor> all_w1_scales;
    std::vector<torch::Tensor> all_w3_scales;
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      all_w1_scales.push_back(weights[fmt::format("{}.w1.weight_scale_inv", expert_id)]);
      all_w3_scales.push_back(weights[fmt::format("{}.w3.weight_scale_inv", expert_id)]);
    }
    torch::Tensor all_w1_scales_stack = torch::stack(all_w1_scales);
    torch::Tensor all_w3_scales_stack = torch::stack(all_w3_scales);
    torch::Tensor all_w1_w3_scales = torch::cat({all_w1_scales_stack, all_w3_scales_stack}, -2);
    torch::Tensor w1_w3_scales = all_w1_w3_scales.to(torch::kBFloat16).view(GetTorchDataType<dtype>());
    auto w1_w3_s_shape = w1_w3_scales.sizes();
    torch::Tensor w1_w3_scales_interleaved =
        w1_w3_scales.reshape({w1_w3_s_shape[0], w1_w3_s_shape[1], w1_w3_s_shape[2] / interleave[0], interleave[0]});
    w1_w3_scales_interleaved = w1_w3_scales_interleaved.permute({0, 2, 1, 3});
    w1_w3_scales_interleaved = w1_w3_scales_interleaved.reshape(
        {w1_w3_s_shape[0], w1_w3_s_shape[2] / interleave[0], w1_w3_s_shape[1] * interleave[0]});
    fc13_weight_scale.copy_(w1_w3_scales_interleaved.contiguous());
    // 创建 cutlass moe layer
    std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
    CutlassMoeLayer cutlass_moe_layer = CutlassMoeLayer();
    cutlass_moe_layer.Init(params, new_runtime_config, context_, kDeviceRank);
    size_t workspace_size = cutlass_moe_layer.GetWorkSpaceSize();
    KLLM_LOG_INFO << fmt::format("CutlassMoeLayer WorkSpaceSize: {}", workspace_size);
    cutlass_moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(workspace_size));
    cutlass_moe_layer.Preprocess(new_model_config, new_runtime_config);
    // 构建输入 input_tensors: 0.hidden states 1.routing_out 2.up_gate_experts 3.down_experts 4.bias
    std::vector<Tensor> inputs(5);
    inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(hidden_states.size(0)), static_cast<size_t>(hidden_states.size(1))},
                       kDeviceRank, hidden_states.data_ptr());
    inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(routing_out.size(0)), static_cast<size_t>(routing_out.size(1))},
                       kDeviceRank, routing_out.data_ptr());
    inputs[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                       {static_cast<size_t>(w1_w3_weight.size(0)), static_cast<size_t>(w1_w3_weight.size(1)),
                        static_cast<size_t>(w1_w3_weight.size(2))},
                       kDeviceRank, w1_w3_weight.data_ptr());
    inputs[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                       {static_cast<size_t>(w2_weight.size(0)), static_cast<size_t>(w2_weight.size(1)),
                        static_cast<size_t>(w2_weight.size(2))},
                       kDeviceRank, w2_weight.data_ptr());
    inputs[4] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {0}, kDeviceRank);
    std::vector<Tensor> scales(2);
    scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(fc13_weight_scale.size(0)), static_cast<size_t>(fc13_weight_scale.size(1)),
                        static_cast<size_t>(fc13_weight_scale.size(2))},
                       kDeviceRank, fc13_weight_scale.data_ptr());
    scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(fc2_weight_scale.size(0)), static_cast<size_t>(fc2_weight_scale.size(1)),
                        static_cast<size_t>(fc2_weight_scale.size(2))},
                       kDeviceRank, fc2_weight_scale.data_ptr());
    inputs[2].scales = &scales[0];
    inputs[3].scales = &scales[1];
    std::vector<Tensor> input_scales(2);
    input_scales[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                             {static_cast<size_t>(fc13_act_scale.size(0)), static_cast<size_t>(fc13_act_scale.size(1))},
                             kDeviceRank, fc13_act_scale.data_ptr());
    input_scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                             {static_cast<size_t>(fc2_act_scale.size(0)), static_cast<size_t>(fc2_act_scale.size(1))},
                             kDeviceRank, fc2_act_scale.data_ptr());
    inputs[2].input_scales = &input_scales[0];
    inputs[3].input_scales = &input_scales[1];
    std::vector<Tensor> alpha(2);
    alpha[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                      {static_cast<size_t>(fc13_alpha.size(0)), static_cast<size_t>(fc13_alpha.size(1))}, kDeviceRank,
                      fc13_alpha.data_ptr());
    alpha[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                      {static_cast<size_t>(fc2_alpha.size(0)), static_cast<size_t>(fc2_alpha.size(1))}, kDeviceRank,
                      fc2_alpha.data_ptr());
    inputs[2].input_alpha = &alpha[0];
    inputs[3].input_alpha = &alpha[1];
    // 构建输出
    std::vector<Tensor> outputs(1);
    outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                        {static_cast<size_t>(cutlass_output.size(0)), static_cast<size_t>(cutlass_output.size(1))},
                        kDeviceRank, cutlass_output.data_ptr());
    // 推理
    EXPECT_TRUE(cutlass_moe_layer.Forward(inputs, outputs).OK());
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
  }

  {
    // 权重加载
    // 步骤1: 创建一份空权重
    torch::Tensor up_gate_weight =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_inter_size * 2),
                      static_cast<int64_t>(expert_hidden_size / 2)},
                     int8_option);
    torch::Tensor down_weight =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_hidden_size),
                      static_cast<int64_t>(expert_inter_size / 2)},
                     int8_option);
    torch::Tensor up_gate_weight_scale =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_inter_size * 2),
                      static_cast<int64_t>(expert_hidden_size / 128)},
                     dtype_option);
    torch::Tensor down_weight_scale =
        torch::empty({static_cast<int64_t>(expert_num), static_cast<int64_t>(expert_hidden_size),
                      static_cast<int64_t>(expert_inter_size / 128)},
                     dtype_option);
    // 步骤2 处理weight
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      up_gate_weight[expert_id].copy_(torch::cat(
          {weights[fmt::format("{}.w1.weight", expert_id)], weights[fmt::format("{}.w3.weight", expert_id)]}, 0));
      down_weight[expert_id].copy_(weights[fmt::format("{}.w2.weight", expert_id)]);
    }
    // 步骤3 处理scale
    for (size_t expert_id = 0; expert_id < expert_num; expert_id++) {
      up_gate_weight_scale[expert_id].copy_(torch::cat({weights[fmt::format("{}.w1.weight_scale_inv", expert_id)],
                                                        weights[fmt::format("{}.w3.weight_scale_inv", expert_id)]},
                                                       0));
      down_weight_scale[expert_id].copy_(weights[fmt::format("{}.w2.weight_scale_inv", expert_id)]);
    }
    // 创建 triton moe layer
    std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
    MoeLayer moe_layer = MoeLayer();
    moe_layer.Init(params, new_runtime_config, context_, kDeviceRank);
    size_t workspace_size = moe_layer.GetWorkSpaceSize();
    KLLM_LOG_INFO << fmt::format("MoeLayer WorkSpaceSize: {}", workspace_size);
    Tensor workspace_buffer = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {workspace_size}, kDeviceRank);
    std::shared_ptr<Tensor> workspace_buffer_ptr = std::make_shared<Tensor>(workspace_buffer);
    moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(workspace_size));
    moe_layer.Preprocess(new_model_config, new_runtime_config);
    // 构建输入 input_tensors: 0.hidden states 1.routing_out 2.up_gate_experts 3.down_experts 4.bias
    std::vector<Tensor> inputs(5);
    inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(hidden_states.size(0)), static_cast<size_t>(hidden_states.size(1))},
                       kDeviceRank, hidden_states.data_ptr());
    inputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(routing_out.size(0)), static_cast<size_t>(routing_out.size(1))},
                       kDeviceRank, routing_out.data_ptr());
    inputs[2] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                       {static_cast<size_t>(up_gate_weight.size(0)), static_cast<size_t>(up_gate_weight.size(1)),
                        static_cast<size_t>(up_gate_weight.size(2))},
                       kDeviceRank, up_gate_weight.data_ptr());
    inputs[3] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                       {static_cast<size_t>(down_weight.size(0)), static_cast<size_t>(down_weight.size(1)),
                        static_cast<size_t>(down_weight.size(2))},
                       kDeviceRank, down_weight.data_ptr());
    inputs[4] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {0}, kDeviceRank);
    std::vector<Tensor> scales(2);
    scales[0] =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
               {static_cast<size_t>(up_gate_weight_scale.size(0)), static_cast<size_t>(up_gate_weight_scale.size(1)),
                static_cast<size_t>(up_gate_weight_scale.size(2))},
               kDeviceRank, up_gate_weight_scale.data_ptr());
    scales[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(down_weight_scale.size(0)), static_cast<size_t>(down_weight_scale.size(1)),
                        static_cast<size_t>(down_weight_scale.size(2))},
                       kDeviceRank, down_weight_scale.data_ptr());
    inputs[2].scales = &scales[0];
    inputs[3].scales = &scales[1];
    // 构建输出
    std::vector<Tensor> outputs(1);
    outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                        {static_cast<size_t>(triton_output.size(0)), static_cast<size_t>(triton_output.size(1))},
                        kDeviceRank, triton_output.data_ptr());
    // 推理
    EXPECT_TRUE(moe_layer.Forward(inputs, outputs).OK());
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
  }

  torch::Tensor torch_output = torch::zeros_like(hidden_states);
  {
    GroupedTopkLayer grouped_topk_layer;
    std::vector<std::any> parameters = {
        static_cast<int>(expert_topk),        norm_topk_prob, static_cast<int>(num_expert_group),
        static_cast<int>(expert_groups_topk), scoring_func,   routed_scaling_factor,
        use_e_score_correction_bias};
    EXPECT_TRUE(grouped_topk_layer.Init(parameters, runtime_config, context_, kDeviceRank).OK());
    torch::Tensor topk_ids =
        torch::zeros({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_topk)}, int32_option);
    torch::Tensor topk_weights =
        torch::zeros({static_cast<int64_t>(num_tokens), static_cast<int64_t>(expert_topk)}, float_option);
    // 构建输入
    std::vector<Tensor> inputs(1);
    inputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(routing_out.size(0)), static_cast<size_t>(routing_out.size(1))},
                       kDeviceRank, routing_out.data_ptr());
    // 构建输出
    std::vector<Tensor> outputs(2);
    outputs[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                        {static_cast<size_t>(topk_weights.size(0)), static_cast<size_t>(topk_weights.size(1))},
                        kDeviceRank, topk_weights.data_ptr());
    outputs[1] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                        {static_cast<size_t>(topk_ids.size(0)), static_cast<size_t>(topk_ids.size(1))}, kDeviceRank,
                        topk_ids.data_ptr());
    // 推理
    EXPECT_TRUE(grouped_topk_layer.Forward(inputs, outputs).OK());
    StreamSynchronize(context_->GetComputeStreams()[kDeviceRank]);
    // 计算参考结果
    torch::Tensor selected_experts = topk_ids.clone();
    torch::Tensor final_scales = topk_weights.clone();
    for (size_t e_idx = 0; e_idx < expert_num; e_idx++) {
      torch::Tensor mask = selected_experts == e_idx;
      torch::Tensor activated_tokens = mask.sum(1).to(torch::kBool);
      torch::Tensor act = hidden_states.index_select(0, activated_tokens.nonzero().squeeze());
      if (act.size(0) == 0) {
        continue;
      }
      torch::Tensor final_scale = (final_scales * mask.to(torch::kFloat))
                                      .sum(1)
                                      .index_select(0, activated_tokens.nonzero().squeeze())
                                      .unsqueeze(1);

      torch::Tensor w1 = weights[fmt::format("{}.w1.weight", e_idx)];
      w1 = unpack_int4_packed_tensor_to_int8(ConvertPackUint4ToPackInt4(w1).cpu()).t().contiguous().cuda();
      torch::Tensor w2 = weights[fmt::format("{}.w2.weight", e_idx)];
      w2 = unpack_int4_packed_tensor_to_int8(ConvertPackUint4ToPackInt4(w2).cpu()).t().contiguous().cuda();
      torch::Tensor w3 = weights[fmt::format("{}.w3.weight", e_idx)];
      w3 = unpack_int4_packed_tensor_to_int8(ConvertPackUint4ToPackInt4(w3).cpu()).t().contiguous().cuda();
      torch::Tensor w1_w3 = torch::cat({w1, w3}, -1);

      torch::Tensor s1 = weights[fmt::format("{}.w1.weight_scale_inv", e_idx)].t().contiguous();
      torch::Tensor s2 = weights[fmt::format("{}.w2.weight_scale_inv", e_idx)].t().contiguous();
      torch::Tensor s3 = weights[fmt::format("{}.w3.weight_scale_inv", e_idx)].t().contiguous();
      torch::Tensor s1_s3 = torch::cat({s1, s3}, -1);

      torch::Tensor p1 = weights[fmt::format("{}.w1.input_scale", e_idx)];
      torch::Tensor p2 = weights[fmt::format("{}.w2.input_scale", e_idx)];
      torch::Tensor p3 = weights[fmt::format("{}.w3.input_scale", e_idx)];
      torch::Tensor p1_p3 = torch::max(p1, p3);

      act = torch::clamp((act / p1_p3), -448.0, 448.0).to(torch::kFloat8_e4m3fn).to(GetTorchDataType<dtype>());
      w1_w3 =
          (w1_w3.to(torch::kFloat) * s1_s3.repeat_interleave(128, 0).to(torch::kFloat)).to(GetTorchDataType<dtype>());
      torch::Tensor fc1_gate = torch::matmul(act, w1_w3) * p1_p3;
      auto chunks = fc1_gate.chunk(2, -1);
      torch::Tensor gate = chunks[0];
      torch::Tensor fc1 = chunks[1];
      fc1 = fc1 * torch::nn::functional::silu(gate);

      act = torch::clamp((fc1 / p2), -448.0, 448.0).to(torch::kFloat8_e4m3fn).to(GetTorchDataType<dtype>());
      w2 = (w2.to(torch::kFloat) * s2.repeat_interleave(128, 0).to(torch::kFloat)).to(GetTorchDataType<dtype>());
      torch::Tensor fc2 = torch::matmul(act, w2) * p2;
      torch_output.index_add_(0, activated_tokens.nonzero().squeeze(), (fc2 * final_scale).to(torch_output.dtype()));
    }
  }
#endif
}

TEST_F(LayerTest, CutlassMoeSearchStatusTest) {
#ifdef ENABLE_CUDA
  if (context_->ext->GetComputeCapacity() != 90) {
    return;
  }
  constexpr int kDeviceRank = 0;

  // params
  MoeScaleNormMode moe_scale_norm_mode = MoeScaleNormMode::NO_NORM;  // 用不到
  size_t max_token_num = 4096;
  size_t expert_num = 32;
  size_t expert_hidden_size = 7168;
  size_t expert_inter_size = 2048;
  size_t expert_topk = 8;
  size_t tp_size = 1;
  bool use_vllm_moe = true;
  uint32_t num_expert_group = 8;
  uint32_t expert_groups_topk = 4;
  std::string scoring_func = "sigmoid";
  std::string topk_method = "";  // 用不到
  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;
  bool use_e_score_correction_bias = false;  // 关闭方便测试
  bool enable_full_shared_expert = false;
  DataType fp8_weight_dtype = DataType::TYPE_INVALID;
  DataType int_weight_dtype = DataType::TYPE_UINT4x2;
  int group_size = 128;
  bool apply_weight = false;  // 用不到
  int layer_idx = 0;

  std::vector<std::any> params;
  params.push_back(moe_scale_norm_mode);
  params.push_back(max_token_num);
  params.push_back(layer_idx);
  params.push_back(expert_num);
  params.push_back(expert_hidden_size);
  params.push_back(expert_inter_size);
  params.push_back(expert_topk);
  params.push_back(tp_size);
  params.push_back(use_vllm_moe);
  params.push_back(num_expert_group);
  params.push_back(expert_groups_topk);
  params.push_back(scoring_func);
  params.push_back(topk_method);
  params.push_back(norm_topk_prob);
  params.push_back(routed_scaling_factor);
  params.push_back(use_e_score_correction_bias);
  params.push_back(enable_full_shared_expert);
  params.push_back(fp8_weight_dtype);
  params.push_back(int_weight_dtype);
  params.push_back(group_size);
  params.push_back(apply_weight);

  std::shared_ptr<LayerWorkspaceManager> workspace_mgr = std::make_shared<LayerWorkspaceManager>(kDeviceRank);
  auto t1 = std::chrono::high_resolution_clock::now();
  Singleton<CutlassMoeSearchStatus>::GetInstance()->ClearCutlassMoeSchedule();
  {
    CutlassMoeLayer cutlass_moe_layer = CutlassMoeLayer();
    cutlass_moe_layer.Init(params, runtime_config, context_, kDeviceRank);
    cutlass_moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(cutlass_moe_layer.GetWorkspaceSize()));
    cutlass_moe_layer.Preprocess(model_config, runtime_config);
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  Singleton<CutlassMoeSearchStatus>::GetInstance()->ClearCutlassMoeSchedule();
  {
    auto func = [&]() {
      for (size_t layer_idx = 0; layer_idx < model_config.num_layer; layer_idx++) {
        CutlassMoeLayer cutlass_moe_layer = CutlassMoeLayer();
        cutlass_moe_layer.Init(params, runtime_config, context_, kDeviceRank);
        cutlass_moe_layer.SetWorkspaceBuffer(workspace_mgr->GetWorkspace(cutlass_moe_layer.GetWorkspaceSize()));
        cutlass_moe_layer.Preprocess(model_config, runtime_config);
      }
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
      threads.emplace_back(func);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  auto duration12 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  auto duration23 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);

  printf("time1: %ld, time2: %ld\n", duration12.count(), duration23.count());

  // 有缓存，创建多次耗时不应该增加太多
  EXPECT_TRUE(2 * duration12.count() > duration23.count());
#endif
}

}  // namespace ksana_llm
