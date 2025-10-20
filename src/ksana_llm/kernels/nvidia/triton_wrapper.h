/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

struct TritonKernel {
  int shm_size;
  int grid_y;
  int grid_z;
  int block_x;
  int block_y;
  int block_z;
  int num_warps;
  int num_stages;
  std::string kernel_name;
  CUmodule module;
  CUfunction kernel;
};

static thread_local std::unordered_map<std::string, std::shared_ptr<TritonKernel>> kernel_map_;

class TritonWrapper {
 public:
  TritonWrapper();
  ~TritonWrapper() = default;

  void InitKernelsDir();

  void SetKernelsDir(const std::string& kernels_dir);

  template <typename T>
  void InvokeFusedMoeKernel(void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights,
                            void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded, int n, int k,
                            int max_num_tokens_padded, int numel, int a_stride0, int a_stride1, int b_stride0,
                            int b_stride2, int b_stride1, int c_stride1, int c_stride2, int a_scale_stride0,
                            int a_scale_stride1, int b_scale_stride0, int b_scale_stride2, int b_scale_stride1,
                            int block_shape0, int block_shape1, bool mul_routed_weight, int topk, bool use_fp8_w8a8,
                            bool use_int8_w8a16, std::unordered_map<std::string, int> config, cudaStream_t stream);

  /**
   * @brief 执行融合的MoE GPTQ AWQ内核计算
   *
   * @tparam T 模板类型参数
   * @param a 输入激活值，类型为T
   * @param b 量化权重数据，类型为uint8_t
   * @param c 输出结果，类型为T
   * @param b_scale 权重缩放因子，类型为T
   * @param b_zp 权重零点值，类型为uint8_t
   * @param topk_weights 顶部k个专家的路由权重，类型为float32
   * @param sorted_token_ids 排序后的token ID，类型为int32_t
   * @param expert_ids 专家ID，类型为int32_t
   * @param num_tokens_post_padded 填充后的token数量，类型为int32_t
   */
  template <typename T>
  void InvokeFusedMoeGptqAwqKernel(void* a, void* b, void* c, void* b_scale, void* b_zp, void* topk_weights,
                                   void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded, int n, int k,
                                   int max_num_tokens_padded, int numel, int a_stride0, int a_stride1, int b_stride0,
                                   int b_stride2, int b_stride1, int c_stride1, int c_stride2, int b_scale_stride0,
                                   int b_scale_stride2, int b_scale_stride1, int b_zp_stride0, int b_zp_stride2,
                                   int b_zp_stride1, bool mul_routed_weight, int top_k, bool has_zp, int weight_bits,
                                   int group_size, std::unordered_map<std::string, int> config, cudaStream_t stream);

  template <typename T>
  void InvokeFusedMoeGptqInt4Fp8Kernel(void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights,
                                       void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded, int n,
                                       int k, int max_num_tokens_padded, int numel, bool mul_routed_weight, int top_k,
                                       int group_size, bool quant_a_per_tensor,
                                       std::unordered_map<std::string, int> config, cudaStream_t stream);

 private:
  std::optional<std::string> ConstructKernelName(const std::string& kernel_base_name,
                                                 const std::unordered_map<std::string, std::string>& map);

  std::optional<std::shared_ptr<TritonKernel>> LoadTritonKernelJsonFile(const std::string& file_path,
                                                                        const std::string& kernel_name);

  CUresult LoadTritonKernelFromFile(const std::string& ptx_file_path, std::shared_ptr<TritonKernel>& triton_kernel);

  void InvokeTritonKernel(const std::string& kernel_base_name, void* args[],
                          const std::unordered_map<std::string, std::string>& map, size_t grid_x, size_t grid_y,
                          size_t grid_z, cudaStream_t stream);

  std::unordered_map<std::string, std::vector<std::string>> kernel_key_map_;

  std::string kernels_dir_;
};

}  // namespace ksana_llm
