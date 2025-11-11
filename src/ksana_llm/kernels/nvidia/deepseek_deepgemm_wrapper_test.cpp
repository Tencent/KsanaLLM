/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <stdexcept>
#include <string>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/nvidia/deepseek_deepgemm_bridge.h"
#include "ksana_llm/kernels/nvidia/deepseek_deepgemm_wrapper.h"

namespace ksana_llm {
namespace nvidia {
namespace test {

using ::testing::HasSubstr;

#if ENABLE_DEEPSEEK_DEEPGEMM

class DeepSeekDeepGEMMWrapperValidationTest : public ::testing::Test {
 protected:
  DeepSeekDeepGEMMWrapperValidationTest() : device_(torch::kCPU) {}

  ~DeepSeekDeepGEMMWrapperValidationTest() {
  }

  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA device unavailable for DeepSeekDeepGEMMWrapper tests.";
    }
    device_ = torch::Device(torch::kCUDA, 0);
  }

  torch::Device device_;
};

TEST_F(DeepSeekDeepGEMMWrapperValidationTest, Fp8MqaLogitsRejectsMismatchedHeadDim) {
  auto wrapper = DeepSeekDeepGEMMWrapper::GetInstance(0);

  const auto fp8_opts = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device_);
  const auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  const auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_);

  auto q = torch::zeros({2, 2, 4}, fp8_opts).contiguous();
  auto kv = torch::zeros({2, 3}, fp8_opts).contiguous();  // head dim mismatch
  auto kv_scales = torch::ones({2}, f32_opts).contiguous();
  auto weights = torch::ones({2, 2}, f32_opts).contiguous();
  auto cur_seq_len_start = torch::zeros({2}, i32_opts).contiguous();
  auto cur_seq_len_end = torch::full({2}, 2, i32_opts).contiguous();
  auto logits = torch::zeros({4, 260}, f32_opts).contiguous();  // satisfy alignment check

  try {
    wrapper->Fp8MqaLogits(q, {kv, kv_scales}, weights, cur_seq_len_start, cur_seq_len_end, logits, false);
    FAIL() << "Expected Fp8MqaLogits to reject mismatched head dimensions";
  } catch (const std::runtime_error& err) {
    EXPECT_THAT(std::string(err.what()), HasSubstr("KV head dim must match query head dim"));
  }
}

TEST_F(DeepSeekDeepGEMMWrapperValidationTest, Fp8PagedMqaLogitsRejectsInsufficientLogitsWidth) {
  auto wrapper = DeepSeekDeepGEMMWrapper::GetInstance(0);

  const auto fp8_opts = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device_);
  const auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
  const auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  const auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_);

  constexpr int batch_size = 1;
  constexpr int next_tokens = 1;
  constexpr int num_heads = 2;
  constexpr int head_dim = 4;
  constexpr int block_kv = 256;
  constexpr int num_kv_blocks = 2;
  constexpr int max_context_len = 256;

  auto q = torch::zeros({batch_size, next_tokens, num_heads, head_dim}, fp8_opts).contiguous();
  auto fused_kv_cache =
      torch::zeros({num_kv_blocks, block_kv, 1, head_dim + static_cast<int>(sizeof(float))}, byte_opts).contiguous();
  auto weights = torch::ones({batch_size * next_tokens, num_heads}, f32_opts).contiguous();
  auto context_lens = torch::full({batch_size}, max_context_len, i32_opts).contiguous();
  auto block_table = torch::zeros({batch_size, 4}, i32_opts).contiguous();
  const int num_sms = llm_kernels::utils::GetSMCount();
  auto schedule_meta = torch::zeros({num_sms + 1, 2}, i32_opts).contiguous();
  auto logits = torch::zeros({batch_size * next_tokens, max_context_len}, f32_opts).contiguous();  // too narrow

  try {
    wrapper->Fp8PagedMqaLogits(q, fused_kv_cache, weights, context_lens, block_table, schedule_meta, logits,
                               max_context_len, false);
    FAIL() << "Expected Fp8PagedMqaLogits to enforce logits width alignment";
  } catch (const std::runtime_error& err) {
    EXPECT_THAT(std::string(err.what()), HasSubstr("logits cols must be at least the aligned max context length"));
  }
}

TEST_F(DeepSeekDeepGEMMWrapperValidationTest, PagedMqaLogitsMetadataValidatesScheduleRows) {
  auto wrapper = DeepSeekDeepGEMMWrapper::GetInstance(0);

  const auto i32_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_);

  constexpr int batch_size = 1;
  constexpr int block_kv = 256;

  auto context_lens = torch::full({batch_size}, block_kv, i32_opts).contiguous();
  auto schedule_meta = torch::zeros({1, 2}, i32_opts).contiguous();  // insufficient rows

  try {
    wrapper->PagedMqaLogitsMetadata(context_lens, schedule_meta, batch_size, block_kv);
    FAIL() << "Expected PagedMqaLogitsMetadata to enforce schedule metadata row count";
  } catch (const std::runtime_error& err) {
    EXPECT_THAT(std::string(err.what()), HasSubstr("schedule_metadata rows must equal num_sms + 1"));
  }
}

#endif  // ENABLE_DEEPSEEK_DEEPGEMM

}  // namespace test
}  // namespace nvidia
}  // namespace ksana_llm
