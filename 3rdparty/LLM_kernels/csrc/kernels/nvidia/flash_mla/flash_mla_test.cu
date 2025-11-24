/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <random>
#include <sstream>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#include "csrc/kernels/nvidia/flash_mla/kernels/params.h"
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

  const int major = prop.major;
  const int minor = prop.minor;

  std::cout << "当前GPU计算能力: " << major << "." << minor << std::endl;
  std::cout << "设备名称: " << prop.name << std::endl;

  if (major >= 9) {
    std::cout << "当前GPU是90或以上的显卡，支持Flash MLA操作" << std::endl;
  } else {
    std::cout << "当前GPU不是90或以上的显卡，可能不支持Flash MLA操作" << std::endl;
    GTEST_SKIP() << "跳过测试，因为当前GPU计算能力低于9.0";
  }

  constexpr int batch = 3;
  constexpr int num_heads = 16;
  constexpr int kv_lora_rank = 512;
  constexpr int qk_rope_head_dim = 64;
  constexpr int max_blocks_per_seq = 2;
  constexpr int block_num = 3;
  constexpr int num_kv_splits = 4;
  constexpr int page_size = 64;
  constexpr float sm_scale = 0.1147213867929261;

  void* const q = CreateFlashMlaTensor<half>({batch, num_heads, kv_lora_rank + qk_rope_head_dim});
  void* const k_buffer = CreateFlashMlaTensor<half>({block_num, page_size, 1, kv_lora_rank + qk_rope_head_dim});
  void* const req_to_token = CreateFlashMlaTensor<int>({batch, max_blocks_per_seq});
  void* const b_seqlen = CreateFlashMlaTensor<int>({batch});
  void* const attn_out = CreateFlashMlaTensor<half>({batch, num_heads, num_kv_splits, kv_lora_rank + 1});

  FlashMlaWorkspaceMap flash_mla_workspace_map;
  GetNumSmParts(flash_mla_workspace_map, num_heads, 1, 0);
  flash_mla_workspace_map.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(
      CreateFlashMlaTensor<int>({flash_mla_workspace_map.num_sm_parts * TileSchedulerMetaDataSize}));
  flash_mla_workspace_map.num_splits_ptr = reinterpret_cast<int*>(CreateFlashMlaTensor<int>({batch + 1}));
  InvokeGetMlaMetadata(reinterpret_cast<int*>(b_seqlen), flash_mla_workspace_map, batch, stream);

  // detail see ApplyWorkspaceBuffer
  void* const workspace = CreateFlashMlaTensor<float>({4096});
  constexpr int q_seq_len = 1;
  float k_scale = 1.0f, v_scale = 1.0f;
  InvokeFlashMla<half, half, llm_kernels::utils::KVCacheType::kAuto>(
      reinterpret_cast<half*>(q), reinterpret_cast<half*>(k_buffer), q_seq_len, sm_scale, req_to_token, b_seqlen,
      flash_mla_workspace_map.tile_scheduler_metadata_ptr, flash_mla_workspace_map.num_splits_ptr, workspace, attn_out,
      batch, num_heads, kv_lora_rank, qk_rope_head_dim, page_size, k_scale, v_scale, max_blocks_per_seq, 0, block_num,
      stream);

  cudaFree(q);
  cudaFree(k_buffer);
  cudaFree(req_to_token);
  cudaFree(b_seqlen);
  cudaFree(attn_out);
  cudaFree(flash_mla_workspace_map.tile_scheduler_metadata_ptr);
  cudaFree(flash_mla_workspace_map.num_splits_ptr);
  cudaFree(workspace);
}

TEST_F(LlamaNvidiaFlashMlaTestSuit, FP8FlashMlaKernelTest) {
  // 判断GPU是否是90以及以上的显卡
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);  // 获取设备0的属性

  const int major = prop.major;
  const int minor = prop.minor;

  std::cout << "当前GPU计算能力: " << major << "." << minor << std::endl;
  std::cout << "设备名称: " << prop.name << std::endl;

  if (major >= 9) {
    std::cout << "当前GPU是90或以上的显卡，支持Flash MLA操作" << std::endl;
  } else {
    std::cout << "当前GPU不是90或以上的显卡，可能不支持Flash MLA操作" << std::endl;
    GTEST_SKIP() << "跳过测试，因为当前GPU计算能力低于9.0";
  }

  int batch = 3;
  int q_seq_len = 1;
  int num_heads = 16;
  int kv_lora_rank = 512;
  int qk_rope_head_dim = 64;
  int max_blocks_per_seq = 2;
  int block_num = 3;
  int num_kv_splits = 4;
  int page_size = 64;
  float sm_scale = 0.1147213867929261;

  void* const quant_q = CreateFlashMlaTensor<uint8_t>({batch * q_seq_len, num_heads, kv_lora_rank + qk_rope_head_dim});
  void* const k_buffer = CreateFlashMlaTensor<uint8_t>({block_num, page_size, 1, kv_lora_rank + qk_rope_head_dim});
  void* const req_to_token = CreateFlashMlaTensor<int>({batch, max_blocks_per_seq});
  void* const b_seqlen = CreateFlashMlaTensor<int>({batch});
  void* const attn_out =
      CreateFlashMlaTensor<__nv_bfloat16>({batch, q_seq_len * num_heads, num_kv_splits, kv_lora_rank + 1});

  FlashMlaWorkspaceMap flash_mla_workspace_map;
  GetNumSmParts(flash_mla_workspace_map, num_heads, 1, 0);
  flash_mla_workspace_map.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(
      CreateFlashMlaTensor<int>({flash_mla_workspace_map.num_sm_parts * TileSchedulerMetaDataSize}));
  flash_mla_workspace_map.num_splits_ptr = reinterpret_cast<int*>(CreateFlashMlaTensor<int>({batch + 1}));
  InvokeGetMlaMetadata(reinterpret_cast<int*>(b_seqlen), flash_mla_workspace_map, batch, stream);

  // detail see ApplyWorkspaceBuffer
  void* const workspace = CreateFlashMlaTensor<float>({4096});
  float k_scale = 1.0f;
  float v_scale = 1.0f;
  InvokeFlashMla<__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3>(
      reinterpret_cast<uint8_t*>(quant_q), reinterpret_cast<uint8_t*>(k_buffer), q_seq_len, sm_scale, req_to_token,
      b_seqlen, flash_mla_workspace_map.tile_scheduler_metadata_ptr, flash_mla_workspace_map.num_splits_ptr, workspace,
      attn_out, batch, num_heads, kv_lora_rank, qk_rope_head_dim, page_size, k_scale, v_scale, max_blocks_per_seq, 0,
      block_num, stream);

  cudaFree(quant_q);
  cudaFree(k_buffer);
  cudaFree(req_to_token);
  cudaFree(b_seqlen);
  cudaFree(attn_out);
  cudaFree(flash_mla_workspace_map.tile_scheduler_metadata_ptr);
  cudaFree(flash_mla_workspace_map.num_splits_ptr);
  cudaFree(workspace);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
