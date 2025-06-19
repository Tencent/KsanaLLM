/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <random>
#include <sstream>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaFlashMlaTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
};

template <typename T>
inline void* CreateFlashMlaTensor(std::vector<int> shape) {
  size_t size = sizeof(T);
  for (int dim : shape) {
    size *= dim;
  }

  void* data_ptr;
  cudaMalloc(&data_ptr, size);
  return data_ptr;
}

TEST_F(LlamaNvidiaFlashMlaTestSuit, FlashMlaKernelTest) {
  // 判断GPU是否是90以及以上的显卡
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);  // 获取设备0的属性

  int major = prop.major;
  int minor = prop.minor;

  std::cout << "当前GPU计算能力: " << major << "." << minor << std::endl;
  std::cout << "设备名称: " << prop.name << std::endl;

  if (major >= 9) {
    std::cout << "当前GPU是90或以上的显卡，支持Flash MLA操作" << std::endl;
  } else {
    std::cout << "当前GPU不是90或以上的显卡，可能不支持Flash MLA操作" << std::endl;
    GTEST_SKIP() << "跳过测试，因为当前GPU计算能力低于9.0";
  }

  int batch = 3;
  int num_heads = 16;
  int kv_lora_rank = 512;
  int qk_rope_head_dim = 64;
  int max_blocks_per_seq = 2;
  int block_num = 3;
  int num_kv_splits = 4;
  int page_size = 64;
  float sm_scale = 0.1147213867929261;

  void* q = CreateFlashMlaTensor<half>({batch, num_heads, kv_lora_rank + qk_rope_head_dim});
  void* k_buffer = CreateFlashMlaTensor<half>({block_num, page_size, 1, kv_lora_rank + qk_rope_head_dim});
  void* v_buffer = k_buffer;
  void* req_to_token = CreateFlashMlaTensor<int>({batch, max_blocks_per_seq});
  void* b_seqlen = CreateFlashMlaTensor<int>({batch});
  void* attn_out = CreateFlashMlaTensor<half>({batch, num_heads, num_kv_splits, kv_lora_rank + 1});

  // detail see ApplyWorkspaceBuffer
  void* workspace = CreateFlashMlaTensor<float>({4096});
  constexpr int q_seq_len = 1;
  InvokeFlashMla<half>(reinterpret_cast<half*>(q), reinterpret_cast<half*>(k_buffer), q_seq_len, sm_scale, req_to_token,
                       b_seqlen, nullptr, nullptr, workspace, attn_out, batch, num_heads, kv_lora_rank,
                       qk_rope_head_dim, page_size, max_blocks_per_seq, 0, block_num, stream);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
