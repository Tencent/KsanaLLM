/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

#include <cstring>
#include <filesystem>
#include <random>

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#endif
#include "ksana_llm/models/base/model_input.h"
#include "ksana_llm/utils/singleton.h"
#include "tests/test.h"

namespace py = pybind11;

namespace ksana_llm {

class ModelInputTest : public testing::Test {
 protected:
  void SetUp() override {
    int rank = 0;
    auto context = std::make_shared<Context>(1, 1, 1);

    // Parse the yaml config file.
    const auto& env = Singleton<Environment>::GetInstance();
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();
    env->ParseConfig(config_path);

    // Initialize the model config
    ModelConfig model_config;
    env->GetModelConfig(model_config);

    // 修改kv_lora_rank为512
    model_config.mla_config.kv_lora_rank = 512;

    // Initialize the block manager.
    env->InitializeBlockManagerConfig();
    BlockManagerConfig block_manager_config;
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.block_host_memory_factor = 0.0;
    block_manager_config.device_allocator_config.blocks_num = 10;  // This test just need a few blocks;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;
    env->SetBlockManagerConfig(block_manager_config);

    RuntimeConfig runtime_config;
    env->GetRuntimeConfig(runtime_config);

    // Initialize the model input object.
    model_input = std::make_unique<ModelInput>(model_config, runtime_config, rank, context);

    // Initialize the random seed with 0.
    std::srand(0);
  }

  void TearDown() override {}

 protected:
  std::unique_ptr<ModelInput> model_input;
};

TEST_F(ModelInputTest, PrepareInputRefitTest) {
  std::vector<float*> input_refit_emb_ptr;
  std::vector<std::pair<int64_t, int64_t>> input_refit_pos_pair;

  auto VerifyPrepareInputRefit = [&]() {
    const size_t input_refit_size = input_refit_emb_ptr.size();
    EXPECT_EQ(model_input->cpu_input_refit_tensor.emb_fp32_ptr_tensor.shape.size(), 1);
    EXPECT_EQ(model_input->cpu_input_refit_tensor.emb_fp32_ptr_tensor.shape[0], input_refit_size);
    EXPECT_EQ(model_input->cpu_input_refit_tensor.pos_pair_tensor.shape.size(), 2);
    EXPECT_EQ(model_input->cpu_input_refit_tensor.pos_pair_tensor.shape[0], input_refit_size);
    EXPECT_EQ(model_input->cpu_input_refit_tensor.pos_pair_tensor.shape[1], 2);
    void** cpu_input_refit_emb_fp32_ptr =
        reinterpret_cast<void**>(model_input->cpu_input_refit_tensor.emb_fp32_ptr_tensor.GetPtr<void>());
    int64_t* cpu_input_refit_pos_pair =
        reinterpret_cast<int64_t*>(model_input->cpu_input_refit_tensor.pos_pair_tensor.GetPtr<void>());
    for (size_t i = 0; i < input_refit_size; i++) {
      EXPECT_EQ(cpu_input_refit_emb_fp32_ptr[i], input_refit_emb_ptr[i]);
      EXPECT_EQ(cpu_input_refit_pos_pair[i * 2], input_refit_pos_pair[i].first);
      EXPECT_EQ(cpu_input_refit_pos_pair[i * 2 + 1], input_refit_pos_pair[i].second);
    }
  };

  // Ensure that torch is imported, so that `THPVariableClass` is not nullptr.
  py::module torch = py::module::import("torch");

  // Test for each selected batch size.
  for (const int batch_size : {1, 3, 4}) {
    input_refit_emb_ptr.clear();
    input_refit_pos_pair.clear();

    std::vector<ForwardRequest> forward_reqs;

    // Reserve memory to avoid memory address being moved.
    std::vector<std::vector<int>> output_tokens;
    std::vector<EmbeddingSlice> embedding_slices;
    forward_reqs.reserve(batch_size);
    output_tokens.reserve(batch_size);
    embedding_slices.reserve(batch_size);

    model_input->multi_token_request_num = batch_size;
    size_t pos_offset = 0;

    // Construct input refit embeddings.
    for (int i = 0; i < batch_size; i++) {
      ForwardRequest forward_req;
      const size_t output_tokens_size = std::rand() % 4096 + 10;
      output_tokens.emplace_back(output_tokens_size);
      forward_req.forwarding_tokens = &output_tokens.back();
      EmbeddingSlice embedding_slice;
      const int input_refit_size = std::rand() % 3 + 1;
      for (int j = 0; j < input_refit_size; j++) {
        const size_t embedding_size = std::rand() % output_tokens_size + 1;
        const size_t embedding_start_pos = std::rand() % embedding_size;
        embedding_slice.embeddings.emplace_back(embedding_size);
        embedding_slice.pos.push_back(embedding_start_pos);
        input_refit_emb_ptr.emplace_back(embedding_slice.embeddings.back().data());
        input_refit_pos_pair.emplace_back(pos_offset + embedding_start_pos, embedding_size);
      }
      embedding_slices.push_back(std::move(embedding_slice));
      forward_req.input_refit_embedding = &embedding_slices.back();
      forward_reqs.push_back(std::move(forward_req));
      pos_offset += output_tokens_size;
    }

    // Parse and load the input refit embeddings.
    model_input->PrepareInputRefit(forward_reqs);

    // Check the result of PrepareInputRefit.
    VerifyPrepareInputRefit();

    // Construct input refit embedding tensors.
    input_refit_emb_ptr.clear();
    for (int i = 0; i < batch_size; i++) {
      ForwardRequest& forward_req = forward_reqs[i];
      auto& embedding_slice = forward_req.input_refit_embedding;
      embedding_slice->embedding_tensors.reserve(embedding_slice->embeddings.size());
      for (const auto& embedding : embedding_slice->embeddings) {
        torch::Tensor embedding_tensor = torch::randn(static_cast<int64_t>(embedding.size()), torch::kFloat32);
        input_refit_emb_ptr.push_back(reinterpret_cast<float*>(embedding_tensor.data_ptr()));
        {
          py::gil_scoped_acquire acquire;
          embedding_slice->embedding_tensors.push_back(
              py::reinterpret_steal<py::object>(THPVariable_Wrap(embedding_tensor)));
        }
      }
      embedding_slice->embeddings.clear();
    }

    // Parse and load the input refit embeddings.
    model_input->PrepareInputRefit(forward_reqs);

    // Check the result of PrepareInputRefit.
    VerifyPrepareInputRefit();

    // Construct bad input.
    forward_reqs[0].input_refit_embedding->embedding_tensors.clear();
    EXPECT_THROW(
        try { model_input->PrepareInputRefit(forward_reqs); } catch (const std::runtime_error& e) {
          EXPECT_NE(strstr(e.what(),
                           "`input_refit_pos.size()` should be equal to `input_refit_embeddings.size()` or "
                           "`input_refit_embedding_tensors.size()`."),
                    nullptr);
          throw;
        },
        std::runtime_error);
  }
}

TEST_F(ModelInputTest, CheckUseCacheTest) {
  // Construct forward requests as test input.
  SamplingConfig sampling_config1, sampling_config2;
  sampling_config1.max_new_tokens = 1;
  sampling_config2.max_new_tokens = 2;
  ForwardRequest forward_req1, forward_req2;
  forward_req1.sampling_config = &sampling_config1;
  forward_req2.sampling_config = &sampling_config2;
  std::vector<ForwardRequest> forward_reqs = {forward_req1, forward_req2};

  const auto& env = Singleton<Environment>::GetInstance();
  CacheManagerConfig cache_manager_config;
  env->GetCacheManagerConfig(cache_manager_config);

  // Test case 1: All the caching is disabled and all the requests only require the next token.
  RuntimeConfig runtime_config;
  env->GetRuntimeConfig(runtime_config);
  EXPECT_FALSE(runtime_config.enable_prefix_caching);
  EXPECT_FALSE(runtime_config.enable_flexible_caching);
  model_input->multi_token_request_num = 1;
  model_input->CheckUseCache(forward_reqs);
  EXPECT_FALSE(model_input->use_cache);

  // Test case 2: All the caching is disabled but some requests require more than one token.
  model_input->multi_token_request_num = 2;
  model_input->CheckUseCache(forward_reqs);
  EXPECT_TRUE(model_input->use_cache);

  // Test case 3: Prefix caching is enabled.
  cache_manager_config.enable_prefix_caching = true;
  env->SetCacheManagerConfig(cache_manager_config);
  env->UpdateModelConfig();
  env->GetRuntimeConfig(runtime_config);
  EXPECT_TRUE(runtime_config.enable_prefix_caching);
  EXPECT_FALSE(runtime_config.enable_flexible_caching);

  model_input->runtime_config_ = runtime_config;  // TODO(robertyuan): ugly, maybe bad test
  model_input->multi_token_request_num = 1;
  model_input->CheckUseCache(forward_reqs);
  EXPECT_TRUE(model_input->use_cache);

  // Test case 4: Flexible caching is enabled.
  cache_manager_config.enable_prefix_caching = false;
  cache_manager_config.min_flexible_cache_num = 256;
  env->SetCacheManagerConfig(cache_manager_config);
  env->UpdateModelConfig();
  env->GetRuntimeConfig(runtime_config);
  EXPECT_FALSE(runtime_config.enable_prefix_caching);
  EXPECT_TRUE(runtime_config.enable_flexible_caching);

  model_input->runtime_config_ = runtime_config;  // TODO(robertyuan): ugly, maybe bad test
  model_input->multi_token_request_num = 1;
  model_input->CheckUseCache(forward_reqs);
  EXPECT_TRUE(model_input->use_cache);
}

#ifdef ENABLE_CUDA
TEST_F(ModelInputTest, PrepareFlashMlaTest) {
  // 测试用例1: 当model_config_.mla_config.kv_lora_rank为0时，PrepareFlashMla应该直接返回
  model_input->model_config_.mla_config.kv_lora_rank = 0;
  model_input->single_token_request_num = 5;
  model_input->PrepareFlashMla(model_input->page_single_input);
  // 由于方法直接返回，没有明确的状态变化可以验证，这里我们只是确保方法不会崩溃

  // 测试用例2: 当single_token_request_num为0时，PrepareFlashMla应该直接返回
  model_input->model_config_.mla_config.kv_lora_rank = 10;
  model_input->single_token_request_num = 0;
  model_input->PrepareFlashMla(model_input->page_single_input);
  // 同样，这里我们只是确保方法不会崩溃

  // 测试用例3: 当所有条件满足时，PrepareFlashMla应该执行相应操作
  // 准备测试数据
  model_input->model_config_.mla_config.kv_lora_rank = 512;
  model_input->single_token_request_num = 4;
  model_input->model_config_.head_num = 16;
  model_input->runtime_config_.parallel_basic_config.tensor_parallel_size = 1;

  // 创建输入长度张量
  std::vector<int> input_lengths = {0, 20, 30, 40, 50};
  MemcpyAsync(model_input->page_single_input.input_length.GetPtr<void>(), input_lengths.data(),
              input_lengths.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              model_input->context_->GetH2DStreams()[model_input->rank_]);

  // 执行PrepareFlashMla
  model_input->PrepareFlashMla(model_input->page_single_input);

  // 从GPU复制数据回CPU并打印
  llm_kernels::nvidia::FlashMlaWorkspaceMap flash_mla_workspace_map;
  GetNumSmParts(flash_mla_workspace_map, 16, 1, 0, 0);
  if (flash_mla_workspace_map.num_sm_parts > 1) {
    int num_splits_cpu;

    MemcpyAsync(&num_splits_cpu, model_input->page_single_input.num_splits.GetPtr<void>(), sizeof(int),
                MEMCPY_DEVICE_TO_HOST, model_input->context_->GetH2DStreams()[model_input->rank_]);

    // 同步流以确保复制完成
    StreamSynchronize(model_input->context_->GetH2DStreams()[model_input->rank_]);
    EXPECT_EQ(num_splits_cpu, 0);
  }
}
#endif

#ifdef ENABLE_CUDA
TEST_F(ModelInputTest, PrepareNextNGatherIdxTest) {
  auto RandomNum = [](const size_t min, const size_t max) {
    static std::default_random_engine random_engine;
    return std::uniform_int_distribution<size_t>(min, max)(random_engine);
  };

  constexpr size_t kReqNum = 10;
  constexpr size_t kMaxReqLen = 1024;

  std::vector<ForwardRequest> forward_reqs(kReqNum);
  std::vector<std::vector<int>> req_tokens(kReqNum);
  for (size_t i = 0; i < kReqNum; ++i) {
    req_tokens[i].resize(RandomNum(0, kMaxReqLen));
    forward_reqs[i].forwarding_tokens = &req_tokens[i];
    forward_reqs[i].kv_cached_token_num = RandomNum(0, req_tokens[i].size());
    forward_reqs[i].req_id = i;
  }

  model_input->mtp_req_id_to_pos_.clear();
  model_input->PrepareNextNGatherIdx(forward_reqs, RunMode::kMain);

  EXPECT_EQ(model_input->mtp_req_id_to_pos_.size(), forward_reqs.size());
  size_t total_len = 0;
  for (size_t i = 0; i < forward_reqs.size(); ++i) {
    EXPECT_EQ(total_len, model_input->mtp_req_id_to_pos_[forward_reqs[i].req_id]);
    total_len += forward_reqs[i].forwarding_tokens->size() - forward_reqs[i].kv_cached_token_num;
  }

  model_input->PrepareNextNGatherIdx(forward_reqs, RunMode::kNextN);
  EXPECT_EQ(model_input->nextn_hidden_idx_uint64_tensor.shape.size(), 1);
  EXPECT_EQ(model_input->nextn_hidden_idx_uint64_tensor.shape[0], total_len);

  std::vector<size_t> host_idx_result(total_len);
  Memcpy(host_idx_result.data(), model_input->nextn_hidden_idx_uint64_tensor.GetPtr<void>(), total_len * sizeof(size_t),
         MEMCPY_DEVICE_TO_HOST);
  size_t result_i = 0, counter_i = 0;
  for (size_t i = 0; i < forward_reqs.size(); ++i) {
    const auto& req = forward_reqs[i];
    for (size_t token_i = 0; token_i < req.forwarding_tokens->size(); ++token_i) {
      if (token_i < req.kv_cached_token_num) {
        continue;
      }
      EXPECT_EQ(host_idx_result[result_i++], counter_i++);
    }
  }
  EXPECT_EQ(host_idx_result.size(), result_i);
}
#endif

TEST_F(ModelInputTest, PrepareMRopePosTest) {
  // Set the model type to qwen2_vl to enable MRoPE tensor creation
  model_input->model_config_.type = "qwen2_vl";
  model_input->model_config_.rope_scaling_factor_config.mrope_section = std::vector<int>{16, 24, 24};
  model_input->CreateVLTensors();

  auto VerifyPrepareMRopePos = [&](const std::vector<ForwardRequest>& forward_reqs,
                                   const std::vector<int64_t>& expected_mrotary_embedding_pos,
                                   const std::vector<int64_t>& expected_offsets) {
    EXPECT_EQ(model_input->dp_mrotary_embedding_pos.shape.size(), 2);
    EXPECT_GE(model_input->dp_mrotary_embedding_pos.shape[1], expected_mrotary_embedding_pos.size());

    // Verify the offsets.
    EXPECT_EQ(expected_offsets.size(), forward_reqs.size());
    for (size_t i = 0; i < forward_reqs.size(); i++) {
      EXPECT_EQ(*forward_reqs[i].mrotary_embedding_pos_offset, expected_offsets[i]);
    }

    // Verify the mrotary_embedding_pos tensor.
    const size_t total_pos_count = expected_mrotary_embedding_pos.size();
    std::vector<int64_t> actual_mrotary_embedding_pos(total_pos_count);
    Memcpy(actual_mrotary_embedding_pos.data(), model_input->dp_mrotary_embedding_pos.GetPtr<void>(),
           total_pos_count * sizeof(int64_t), MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < total_pos_count; i++) {
      EXPECT_EQ(actual_mrotary_embedding_pos[i], expected_mrotary_embedding_pos[i]);
    }
  };

  // Ensure that torch is imported for tensor handling
  py::module torch = py::module::import("torch");

  // Test for each selected batch size.
  for (const int batch_size : {1, 3, 4}) {
    std::vector<ForwardRequest> forward_reqs;
    std::vector<std::vector<int>> output_tokens;
    std::vector<EmbeddingSlice> embedding_slices;
    std::vector<int64_t> mrotary_offsets;

    // Reserve memory to avoid memory address being moved.
    forward_reqs.reserve(batch_size);
    output_tokens.reserve(batch_size);
    embedding_slices.reserve(batch_size);
    mrotary_offsets.reserve(batch_size);

    model_input->multi_token_request_num = batch_size;

    std::vector<int64_t> expected_mrotary_embedding_pos;
    std::vector<int64_t> expected_offsets;

    // Create a mix of plain text and visual inputs
    for (int i = 0; i < batch_size; i++) {
      ForwardRequest forward_req;
      const size_t token_size = 10 + i * 5;
      output_tokens.emplace_back(token_size);
      forward_req.forwarding_tokens = &output_tokens.back();
      EmbeddingSlice embedding_slice;
      embedding_slices.push_back(std::move(embedding_slice));
      forward_req.input_refit_embedding = &embedding_slices.back();
      mrotary_offsets.push_back(0);
      forward_req.mrotary_embedding_pos_offset = &mrotary_offsets.back();

      // Alternate between plain text and visual input
      if (i % 2 == 0) {
        // Plain text input (empty additional_tensors)
        // For plain text, the function creates positions where each triplet is [i, i, i]
        int64_t list_size = forward_req.forwarding_tokens->size() * 3;
        for (int64_t j = 0; j < list_size; j += 3) {
          expected_mrotary_embedding_pos.push_back(j);
          expected_mrotary_embedding_pos.push_back(j);
          expected_mrotary_embedding_pos.push_back(j);
        }
        expected_offsets.push_back(0);
      } else {
        // Visual input (non-empty additional_tensors)
        // Create position tensor with deterministic values for testing
        torch::Tensor pos_tensor = torch::randint(0, 100, {static_cast<int64_t>(token_size * 3)}, torch::kInt64);
        int64_t offset_value = i * 10;
        torch::Tensor offset_tensor = torch::tensor(offset_value, torch::kInt64);
        auto pos_accessor = pos_tensor.data_ptr<int64_t>();
        for (int64_t j = 0; j < pos_tensor.numel(); j++) {
          expected_mrotary_embedding_pos.push_back(pos_accessor[j]);
        }
        expected_offsets.push_back(offset_value);
        // Add tensors to additional_tensors
        {
          py::gil_scoped_acquire acquire;
          forward_req.input_refit_embedding->additional_tensors.push_back(
              py::reinterpret_steal<py::object>(THPVariable_Wrap(pos_tensor)));
          forward_req.input_refit_embedding->additional_tensors.push_back(
              py::reinterpret_steal<py::object>(THPVariable_Wrap(offset_tensor)));
        }
      }
      forward_reqs.push_back(std::move(forward_req));
    }

    // Parse MRopePos.
    model_input->PrepareMRopePos(forward_reqs);

    // Check the result of VerifyPrepareMRopePos.
    VerifyPrepareMRopePos(forward_reqs, expected_mrotary_embedding_pos, expected_offsets);
  }

  // Construct bad input.
  {
    std::vector<ForwardRequest> forward_reqs(1);
    std::vector<std::vector<int>> output_tokens(1, std::vector<int>(10));
    std::vector<EmbeddingSlice> embedding_slices(1);
    std::vector<int64_t> mrotary_offsets(1, 0);

    forward_reqs[0].forwarding_tokens = &output_tokens[0];
    forward_reqs[0].input_refit_embedding = &embedding_slices[0];
    forward_reqs[0].mrotary_embedding_pos_offset = &mrotary_offsets[0];

    model_input->multi_token_request_num = 1;
    {
      torch::Tensor pos_tensor = torch::randint(0, 100, {30}, torch::kInt64);
      {
        py::gil_scoped_acquire acquire;
        forward_reqs[0].input_refit_embedding->additional_tensors.clear();
        forward_reqs[0].input_refit_embedding->additional_tensors.push_back(
            py::reinterpret_steal<py::object>(THPVariable_Wrap(pos_tensor)));
      }

      EXPECT_THROW(
          try { model_input->PrepareMRopePos(forward_reqs); } catch (const std::runtime_error& e) {
            EXPECT_NE(strstr(e.what(), "additional_tensors should contain at least 2 tensors"), nullptr);
            throw;
          },
          std::runtime_error);
    }
  }
}
}  // namespace ksana_llm
