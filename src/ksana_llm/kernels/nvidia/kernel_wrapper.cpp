/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

#include <fstream>
#include <iostream>

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"

#include "ksana_llm/utils/singleton.h"
#if defined(ENABLE_FLASH_ATTN_2) || defined(ENABLE_VLLM_FLASH_ATTN_2)
#  include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#else
#  include "flash_api.h"
#endif

#include "ksana_llm/kernels/nvidia/triton_wrapper.h"

#include "csrc/kernels/nvidia/activation/activation.h"
#include "csrc/kernels/nvidia/add/add.h"
#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "csrc/kernels/nvidia/assemble_tokens_hidden/assemble_tokens_hidden.h"
#include "csrc/kernels/nvidia/blockwise_gemm/blockwise_gemm.h"
#include "csrc/kernels/nvidia/cast/cast.h"
#include "csrc/kernels/nvidia/concat/concat.h"
#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/kernels/nvidia/expand/expand.h"
#include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#include "csrc/kernels/nvidia/fused_add_norm/fused_add_norm.h"
#include "csrc/kernels/nvidia/fused_moe_gptq_int4_fp8_kernel/dequant.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/grouped_topk/grouped_topk.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"
#include "csrc/kernels/nvidia/moe/moe.h"
#include "csrc/kernels/nvidia/moe_wna16/moe_wna16.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy_flash_attn_layout.h"
#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/paged_attention.h"
#include "csrc/kernels/nvidia/per_token_group_quant/per_token_group_quant_8bit.h"
#include "csrc/kernels/nvidia/permute/permute.h"
#include "csrc/kernels/nvidia/samplers/greedy.h"
#include "csrc/utils/nvidia/cuda_fp8_utils.h"

#include "ksana_llm/kernels/argmax.h"
#include "ksana_llm/kernels/cast.h"

#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/search_status.h"

namespace ksana_llm {

void DequantInt4Fp8(cudaStream_t stream, void* output, const void* input, const size_t datasize) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::dequant::dequant_int4_fp8(stream, output, input, datasize));
}

// Adapted from https://github.com/vllm-project/vllm/blob/v0.8.4/vllm/model_executor/layers/fused_moe/fused_moe.py#L676
void UpdateMoeWna16BlockConfig(std::unordered_map<std::string, int>& config, bool use_moe_wna16_cuda,
                               bool use_fp8_compute, int num_valid_tokens, int size_k, int size_n, int num_experts,
                               int group_size, int real_top_k, int block_size_m) {
  if (config.find("block_size_n") != config.end() && config.find("block_size_k") != config.end()) {
    // optimal block config is set
    return;
  }

  if (!use_moe_wna16_cuda) {
    if (use_fp8_compute) {
      config["block_size_n"] = 128;
      config["block_size_k"] = 128;
    } else {
      // triton moe wna16 kernel
      if (num_valid_tokens / real_top_k == 1) {
        // if bs=1, use a smaller block_size_n
        config["block_size_n"] = 32;
        config["block_size_k"] = 64;
      } else {
        config["block_size_n"] = 64;
        config["block_size_k"] = 32;
      }
    }
  } else {
    // cuda moe wna16 kernel
    // set default block_size 128, and increase them when num_blocks is too large.
    int block_size_n = 128;
    int block_size_k = 128;
    if (block_size_k <= group_size) {
      block_size_k = group_size;
    }

    int num_n_blocks = size_k / block_size_k;
    int num_k_blocks = size_n / block_size_k;
    int num_m_blocks = (num_valid_tokens + block_size_m - 1) / block_size_m + num_experts;
    if (num_valid_tokens / real_top_k <= block_size_m) {
      num_m_blocks = std::min(num_m_blocks, num_valid_tokens);
    }
    int num_blocks = num_m_blocks * num_n_blocks * num_k_blocks;

    if (size_k % 256 == 0 && num_blocks >= 256 && block_size_k < 256) {
      block_size_k = 256;
      num_blocks = num_blocks / (256 / block_size_k);
    }

    if (num_m_blocks <= 16 && size_k % (block_size_k * 2) == 0 && block_size_k <= 512 && num_blocks >= 512) {
      block_size_k *= 2;
      num_blocks /= 2;
    }

    if (num_blocks > 1024) {
      block_size_n = 256;
      num_n_blocks /= 2;
      num_blocks /= 2;
    }

    if (size_n <= 1024 && num_blocks >= 1024) {
      // The kernel performance got much better with block_size_n=1024
      // when num_blocks is large, even when N is small.
      block_size_n = 1024;
    }

    config["block_size_n"] = block_size_n;
    config["block_size_k"] = block_size_k;
  }
}

// Adapted from https://github.com/vllm-project/vllm/blob/v0.8.4/vllm/model_executor/layers/fused_moe/fused_moe.py#L734
bool ShouldMoeWna16UseCuda(int num_valid_tokens, int group_size, int num_experts, int bit) {
  return bit == 4 && (group_size == 32 || group_size == 64 || group_size == 128) &&
         static_cast<double>(num_valid_tokens) / num_experts <= 6.0;
}

template <typename T>
void InvokeMoeWna16Gemm(cudaStream_t stream, void* output, const void* input, const void* b_qweight,
                        const void* b_scales, const void* b_qzeros, const void* topk_weights,
                        const void* sorted_token_ids, const void* expert_ids, const void* num_tokens_post_pad,
                        int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit, int num_experts,
                        int size_m, int size_n, int size_k, int group_size, int num_token_blocks) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::moe_wna16::moe_wna16_gemm<T>(
      stream, output, input, b_qweight, b_scales, b_qzeros, topk_weights, sorted_token_ids, expert_ids,
      num_tokens_post_pad, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, bit, num_experts, size_m, size_n, size_k,
      group_size, num_token_blocks));
}
#define INVOKE_MOE_WNA16_GEMM(T)                                                                                 \
  template void InvokeMoeWna16Gemm<T>(                                                                           \
      cudaStream_t stream, void* output, const void* input, const void* b_qweight, const void* b_scales,         \
      const void* b_qzeros, const void* topk_weights, const void* sorted_token_ids, const void* expert_ids,      \
      const void* num_tokens_post_pad, int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit, \
      int num_experts, int size_m, int size_n, int size_k, int group_size, int num_token_blocks)
INVOKE_MOE_WNA16_GEMM(float);
INVOKE_MOE_WNA16_GEMM(half);
#ifdef ENABLE_BFLOAT16
INVOKE_MOE_WNA16_GEMM(__nv_bfloat16);
#endif
#undef INVOKE_MOE_WNA16_GEMM

#define GET_MACHETE_DATA_TYPE(T, MACHETE_TYPE)                          \
  template <>                                                           \
  llm_kernels::nvidia::vllm_dtype::ScalarType GetMacheteDataType<T>() { \
    return MACHETE_TYPE;                                                \
  }
GET_MACHETE_DATA_TYPE(float, llm_kernels::nvidia::vllm_dtype::kFloat);
GET_MACHETE_DATA_TYPE(half, llm_kernels::nvidia::vllm_dtype::kHalf);
#ifdef ENABLE_BFLOAT16
GET_MACHETE_DATA_TYPE(__nv_bfloat16, llm_kernels::nvidia::vllm_dtype::kBFloat16);
#endif
#undef GET_MACHETE_DATA_TYPE

std::vector<std::string> GetMacheteSupportedSchedules(
    llm_kernels::nvidia::vllm_dtype::ScalarType a_type, llm_kernels::nvidia::vllm_dtype::ScalarType b_type,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_group_scales_type,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_group_zeros_type) {
  return llm_kernels::nvidia::machete::machete_supported_schedules(a_type, b_type, maybe_group_scales_type,
                                                                   maybe_group_zeros_type);
}

void InvokeMacheteGemm(int64_t& workspace_size, void* workspace, cudaStream_t stream, int M, int N, int K,
                       const void* Aptr, const void* Bptr, void* Dptr,
                       llm_kernels::nvidia::vllm_dtype::ScalarType const& a_type,
                       llm_kernels::nvidia::vllm_dtype::ScalarType const& b_type,
                       std::optional<void*> const& maybe_group_scales_ptr,
                       std::optional<std::vector<size_t>> const& maybe_group_scales_shape,
                       std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_scales_type,
                       std::optional<void*> const& maybe_group_zeros_ptr,
                       std::optional<std::vector<size_t>> const& maybe_group_zeros_shape,
                       std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_zeros_type,
                       std::optional<int64_t> maybe_group_size, std::optional<std::string> maybe_schedule) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::machete::machete_gemm(
      workspace_size, workspace, stream, M, N, K, Aptr, Bptr, Dptr, a_type, b_type, maybe_group_scales_ptr,
      maybe_group_scales_shape, maybe_group_scales_type, maybe_group_zeros_ptr, maybe_group_zeros_shape,
      maybe_group_zeros_type, maybe_group_size, maybe_schedule));
}

void InvokeMachetePrepackWeight(
    const void* B_ptr, const std::vector<size_t>& B_shape, void* out_ptr,
    llm_kernels::nvidia::vllm_dtype::ScalarType const& a_type,
    llm_kernels::nvidia::vllm_dtype::ScalarType const& b_type,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_scales_type, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::machete::machete_prepack_weight(B_ptr, B_shape, out_ptr, a_type, b_type,
                                                                             maybe_group_scales_type, stream));
}

std::string GetMacheteBestSchedule(
    size_t warmup_iters, size_t record_iters, void* workspace, cudaStream_t stream, int M, int N, int K,
    const void* Aptr, const void* Bptr, void* Dptr, llm_kernels::nvidia::vllm_dtype::ScalarType const& a_type,
    llm_kernels::nvidia::vllm_dtype::ScalarType const& b_type, std::optional<void*> const& maybe_group_scales_ptr,
    std::optional<std::vector<size_t>> const& maybe_group_scales_shape,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_scales_type,
    std::optional<void*> const& maybe_group_zeros_ptr,
    std::optional<std::vector<size_t>> const& maybe_group_zeros_shape,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_zeros_type,
    std::optional<int64_t> maybe_group_size) {
  return llm_kernels::nvidia::machete::machete_best_schedule(
      warmup_iters, record_iters, workspace, stream, M, N, K, Aptr, Bptr, Dptr, a_type, b_type, maybe_group_scales_ptr,
      maybe_group_scales_shape, maybe_group_scales_type, maybe_group_zeros_ptr, maybe_group_zeros_shape,
      maybe_group_zeros_type, maybe_group_size);
}

void InvokeMarlinAwqRepack(const void* b_q_weight_ptr, void* out_ptr, int64_t size_k, int64_t size_n, int64_t num_bits,
                           int rank, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::marlin::awq_marlin_repack(
      reinterpret_cast<const uint32_t*>(b_q_weight_ptr), reinterpret_cast<uint32_t*>(out_ptr), size_k, size_n, num_bits,
      rank, stream));
}

std::vector<int64_t> GetMarlinAwqRepackMeta(int64_t size_k, int64_t size_n, int64_t num_bits) {
  return llm_kernels::nvidia::marlin::awq_marlin_repack_meta(size_k, size_n, num_bits);
}

void InvokeMarlinGptqRepack(const void* b_q_weight_ptr, const void* perm_ptr, void* out_ptr, int64_t size_k,
                            int64_t size_n, int64_t num_bits, bool has_perm, int rank, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::marlin::gptq_marlin_repack(
      reinterpret_cast<const uint32_t*>(b_q_weight_ptr), reinterpret_cast<const uint32_t*>(perm_ptr),
      reinterpret_cast<uint32_t*>(out_ptr), size_k, size_n, num_bits, has_perm, rank, stream));
}

std::vector<int64_t> GetMarlinGptqRepackMeta(int64_t size_k, int64_t size_n, int64_t num_bits) {
  return llm_kernels::nvidia::marlin::gptq_marlin_repack_meta(size_k, size_n, num_bits);
}

template <typename T>
llm_kernels::nvidia::marlin::WorkspaceInfo GetMarlinWorkspace(bool use_fp32_reduce, bool has_act_order, int rank,
                                                              int64_t size_m, int64_t size_k) {
  return llm_kernels::nvidia::marlin::get_workspace<T>(use_fp32_reduce, has_act_order, rank, size_m, size_k);
}
#define GET_MARLIN_WORKSPACE(T)                                                                                       \
  template llm_kernels::nvidia::marlin::WorkspaceInfo GetMarlinWorkspace<T>(bool use_fp32_reduce, bool has_act_order, \
                                                                            int rank, int64_t size_m, int64_t size_k)
GET_MARLIN_WORKSPACE(float);
GET_MARLIN_WORKSPACE(half);
#ifdef ENABLE_BFLOAT16
GET_MARLIN_WORKSPACE(__nv_bfloat16);
#endif
#undef GET_MARLIN_WORKSPACE

template <typename T>
void InvokeMarlinPermuteScales(cudaStream_t stream, const void* input, void* output, const size_t k, const size_t n,
                               const int64_t groupsize) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::marlin::permute_scales<T>(stream, reinterpret_cast<const T*>(input),
                                                                       reinterpret_cast<T*>(output), k, n, groupsize));
}
#define INVOKE_MARLIN_PERMUTE_SCALES(T)                                                                            \
  template void InvokeMarlinPermuteScales<T>(cudaStream_t stream, const void* input, void* output, const size_t k, \
                                             const size_t n, const int64_t groupsize)
INVOKE_MARLIN_PERMUTE_SCALES(float);
INVOKE_MARLIN_PERMUTE_SCALES(half);
#ifdef ENABLE_BFLOAT16
INVOKE_MARLIN_PERMUTE_SCALES(__nv_bfloat16);
#endif
#undef INVOKE_MARLIN_PERMUTE_SCALES

template <typename T>
void InvokeMarlinGemm(void* a, void* a_tmp, void* b_q_weight, void* b_scales, void* b_zeros, void* g_idx, void* perm,
                      void* workspace, void* c, void* c_tmp, int64_t size_m, int64_t size_n, int64_t size_k,
                      int64_t num_groups, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float,
                      bool has_zp, bool has_act_order, bool is_awq, int rank, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::marlin::gptq_marlin_gemm<T>(
      a, a_tmp, b_q_weight, b_scales, b_zeros, g_idx, perm, workspace, c, c_tmp, size_m, size_n, size_k, num_groups,
      is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float, has_zp, has_act_order, is_awq, rank, stream));
}
#define INVOKE_MARLIN_GEMM(T)                                                                                       \
  template void InvokeMarlinGemm<T>(void* a, void* a_tmp, void* b_q_weight, void* b_scales, void* b_zeros,          \
                                    void* g_idx, void* perm, void* workspace, void* c, void* c_tmp, int64_t size_m, \
                                    int64_t size_n, int64_t size_k, int64_t num_groups, bool is_k_full,             \
                                    bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float, bool has_zp,       \
                                    bool has_act_order, bool is_awq, int rank, cudaStream_t stream)
INVOKE_MARLIN_GEMM(float);
INVOKE_MARLIN_GEMM(half);
#ifdef ENABLE_BFLOAT16
INVOKE_MARLIN_GEMM(__nv_bfloat16);
#endif
#undef INVOKE_MARLIN_GEMM

template <typename T>
torch::ScalarType GetTorchDataType();
#define GET_TORCH_DATA_TYPE(T, TORCH_TYPE)  \
  template <>                               \
  torch::ScalarType GetTorchDataType<T>() { \
    return TORCH_TYPE;                      \
  }
GET_TORCH_DATA_TYPE(int32_t, torch::kInt32);
GET_TORCH_DATA_TYPE(float, torch::kFloat32);
GET_TORCH_DATA_TYPE(half, torch::kFloat16);
#ifdef ENABLE_BFLOAT16
GET_TORCH_DATA_TYPE(__nv_bfloat16, torch::kBFloat16);
#endif
#undef GET_TORCH_DATA_TYPE

DataType GetDataTypeFromTorchType(const c10::ScalarType& torch_type) {
  DataType data_type = TYPE_INVALID;
  switch (torch_type) {
    case c10::kBFloat16:
      data_type = TYPE_BF16;
      break;
    case torch::kFloat16:
      data_type = TYPE_FP16;
      break;
    case torch::kFloat32:
      data_type = TYPE_FP32;
      break;
    case torch::kInt32:
      data_type = TYPE_INT32;
      break;
    case torch::kInt8:
      data_type = TYPE_INT8;
      break;
    case torch::kUInt8:
      data_type = TYPE_UINT8;
      break;
    default:
      break;
  }
  return data_type;
}

template <typename T, typename WT, typename OT>
void GetMoeGemmWorkspaceSize(size_t token_num, size_t expert_num, size_t expert_hidden_size, size_t expert_inter_size,
                             size_t expert_topk, int tp_size, int rank, bool use_lora, size_t& ws_bytes) {
  auto moe_gemm = llm_kernels::nvidia::MoeGemmWrapper<T, WT, OT>();
  moe_gemm.GetWorkspaceSize(token_num, expert_num, expert_hidden_size, expert_inter_size, expert_topk, tp_size, rank,
                            use_lora, ws_bytes);
}
#define GET_MOE_GEMM_WORKSPACE_SIZE(T, WT, OT)                                                                     \
  template void GetMoeGemmWorkspaceSize<T, WT, OT>(size_t token_num, size_t expert_num, size_t expert_hidden_size, \
                                                   size_t expert_inter_size, size_t expert_topk, int tp_size,      \
                                                   int rank, bool use_lora, size_t& ws_bytes)
GET_MOE_GEMM_WORKSPACE_SIZE(float, float, float);
GET_MOE_GEMM_WORKSPACE_SIZE(half, half, half);
#ifdef ENABLE_FP8
GET_MOE_GEMM_WORKSPACE_SIZE(__nv_fp8_e4m3, __nv_fp8_e4m3, half);
#endif
#ifdef ENABLE_BFLOAT16
GET_MOE_GEMM_WORKSPACE_SIZE(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
#  ifdef ENABLE_FP8
GET_MOE_GEMM_WORKSPACE_SIZE(__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16);
#  endif
#endif
#undef GET_MOE_GEMM_WORKSPACE_SIZE

template <typename T, typename WT, typename OT>
size_t InvokeMoeGemmConfigProfile(bool is_fp8) {
  auto moe_gemm = llm_kernels::nvidia::MoeGemmWrapper<T, WT, OT>();
  return moe_gemm.GetBestConfigIndex(is_fp8);
}
#define INVOKE_MOE_GEMM_CONFIG_PROFILE(T, WT, OT) template size_t InvokeMoeGemmConfigProfile<T, WT, OT>(bool is_fp8)
INVOKE_MOE_GEMM_CONFIG_PROFILE(float, float, float);
INVOKE_MOE_GEMM_CONFIG_PROFILE(half, half, half);
#ifdef ENABLE_FP8
INVOKE_MOE_GEMM_CONFIG_PROFILE(__nv_fp8_e4m3, __nv_fp8_e4m3, half);
#endif
#ifdef ENABLE_BFLOAT16
INVOKE_MOE_GEMM_CONFIG_PROFILE(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
#  ifdef ENABLE_FP8
INVOKE_MOE_GEMM_CONFIG_PROFILE(__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16);
#  endif
#endif
#undef INVOKE_MOE_GEMM_CONFIG_PROFILE

void SaveTorchTensor(const torch::Tensor& tensor, std::string file) {
  auto c_tensor = tensor.contiguous();

  std::vector<size_t> shape;
  std::transform(c_tensor.sizes().vec().begin(), c_tensor.sizes().vec().end(), shape.begin(),
                 [](int64_t val) { return static_cast<size_t>(val); });

  int device_id;
  GetDevice(&device_id);
  Tensor(MemoryLocation::LOCATION_DEVICE, GetDataTypeFromTorchType(c_tensor.scalar_type()), shape, device_id,
         c_tensor.data_ptr())
      .SaveToNpyFile(file);
}

void SaveVoidPtr(void* data_ptr, std::vector<size_t> shape, std::string file, const DataType& data_type) {
  int device_id;
  GetDevice(&device_id);
  Tensor(MemoryLocation::LOCATION_DEVICE, data_type, shape, device_id, data_ptr).SaveToNpyFile(file);
}

// Adapted from
// [DeepSeek-V3 Project] https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L532
torch::Tensor NoauxTc(const std::string& scoring_func_, bool norm_topk_prob_, bool use_e_score_correction_bias_,
                      torch::Tensor logits, torch::Tensor bias_tensor, int64_t top_k, int64_t device_limited_n_group,
                      int64_t device_limited_topk_group) {
  // Step 1: 计算scores和scores_with_bias
  torch::Tensor scores;
  if (scoring_func_ == "sigmoid") {
    scores = torch::sigmoid(logits);
  } else {
    scores = torch::softmax(logits, -1);
    device_limited_n_group = 1;
    device_limited_topk_group = 1;
  }

  torch::Tensor scores_with_bias;
  if (use_e_score_correction_bias_) {
    scores_with_bias = scores + bias_tensor;
  } else {
    scores_with_bias = scores;
  }

  // 获取原始形状
  auto scores_shape = scores_with_bias.sizes();
  std::vector<int64_t> scores_shape_vec(scores_shape.begin(), scores_shape.end());
  int64_t last_dim = scores_shape_vec.back();

  // Step 2: 重塑张量并计算group_scores
  std::vector<int64_t> new_shape(scores_shape_vec.begin(), scores_shape_vec.end() - 1);
  new_shape.push_back(device_limited_n_group);
  new_shape.push_back(last_dim / device_limited_n_group);

  auto reshaped = scores_with_bias.view(new_shape);
  auto topk2 = torch::topk(reshaped, 2, -1);
  torch::Tensor group_scores = std::get<0>(topk2).sum(-1);

  // Step 3: 选择topk组
  auto topk_groups = torch::topk(group_scores, device_limited_topk_group, -1);
  torch::Tensor group_idx = std::get<1>(topk_groups);

  // Step 4: 创建group_mask
  torch::Tensor group_mask = torch::zeros_like(group_scores);
  group_mask.scatter_(-1, group_idx, torch::ones_like(group_idx, group_mask.options()));

  // Step 5: 扩展mask并应用
  auto unsqueezed_mask = group_mask.unsqueeze(-1);
  auto expanded_mask = unsqueezed_mask.expand(new_shape).contiguous();
  torch::Tensor score_mask = expanded_mask.view(scores_shape);

  scores_with_bias = scores_with_bias * score_mask;

  // Step 6: 最终topk选择
  auto final_topk = torch::topk(scores_with_bias, top_k, -1);
  torch::Tensor topk_idx = std::get<1>(final_topk);

  // Step 7: 创建最终mask并归一化
  torch::Tensor new_mask = torch::zeros_like(scores);
  new_mask.scatter_(-1, topk_idx, torch::ones_like(topk_idx, new_mask.options()));

  scores = scores * new_mask;
  torch::Tensor score_sum = scores.sum(-1, /*keepdim=*/true) + 1e-20;
  if (norm_topk_prob_) {
    scores = scores / score_sum;
  } else {
  }
  return scores;
}

torch::Tensor compute_moe(torch::Tensor& input, torch::Tensor& moe_gate, torch::Tensor& gate, torch::Tensor& up,
                          torch::Tensor& down, int64_t topk, int64_t num_experts) {
  // 后续计算使用 GPU 张量
  auto scores = moe_gate.squeeze();
  auto [values, selected_indices] = torch::topk(scores, topk, 0, true);
  selected_indices = selected_indices.to(torch::kLong);
  // 假设selected_indices是通过topk获得的GPU张量
  auto cpu_indices = selected_indices.cpu().contiguous();  // 转移到CPU并确保连续

  // 获取数据指针和形状信息
  int64_t* indices_ptr = cpu_indices.data_ptr<int64_t>();

  auto total = torch::zeros_like(input, torch::kDouble);
  for (int i = 0; i < topk; i++) {
    int64_t expert_idx = indices_ptr[i];

    auto mg = moe_gate.slice(1, expert_idx, expert_idx + 1).squeeze();
    auto g = gate.slice(0, expert_idx, expert_idx + 1);
    auto u = up.slice(0, expert_idx, expert_idx + 1);
    auto d = down.slice(0, expert_idx, expert_idx + 1);

    auto part1 = torch::silu(torch::matmul(input, g));
    auto part2 = torch::matmul(input, u);
    auto temp = part1.mul(part2);
    auto res = temp.matmul(d);
    res = res.mul(mg);
    total = total.add(res);
  }
  return total;
}

template <typename T, typename WT, typename OT, llm_kernels::nvidia::MOEExpertScaleNormalizationMode NT>
void InvokeMoeCutlassGemm(void const* input_activations, void* gating_output, void const* fc1_expert_weights,
                          void const* fc2_expert_weights, void* e_score_correction_bias, int64_t const num_rows,
                          int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const topk,
                          char* workspace_ptr, void* final_output, void* token_topk_final_scales,
                          int* expanded_source_row_to_expanded_dest_row, int* expert_for_source_row, int tp_size,
                          int rank, bool use_lora, size_t best_config_index, bool use_vllm_moe_,
                          uint32_t num_expert_group_, uint32_t expert_groups_topk_, const std::string& scoring_func_,
                          const std::string& topk_method_, bool norm_topk_prob_, float routed_scaling_factor_,
                          bool use_e_score_correction_bias_, cudaStream_t stream, bool is_fp8, void const* scale1,
                          void const* scale2, void const* scale3, bool apply_weight) {
  auto origin_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<OT>());
  torch::Tensor gating_tensor = torch::from_blob(
      gating_output, {static_cast<int64_t>(num_rows), static_cast<int64_t>(num_experts)}, origin_options);
  gating_tensor = gating_tensor.to(torch::kFloat32);

  llm_kernels::nvidia::RoutingFunctionType custom_routing_function;
  if (topk_method_ == "greedy" && scoring_func_ == "softmax") {
    custom_routing_function = llm_kernels::nvidia::RoutingFunctionType::GREEDY_TOPK_SOFTMAX_SCORE;
  } else if (topk_method_ == "fast" && scoring_func_ == "sigmoid") {
    custom_routing_function = llm_kernels::nvidia::RoutingFunctionType::FAST_TOPK_SIGMOID_SCORE;
  } else {
    KLLM_THROW(fmt::format("topk_method_ {} with scoring_func_ {} is not supported", topk_method_, scoring_func_));
  }

  auto moe_gemm = llm_kernels::nvidia::MoeGemmWrapper<T, WT, OT>();
  moe_gemm.Gemm(input_activations, gating_tensor.data_ptr(), fc1_expert_weights, fc2_expert_weights, num_rows,
                hidden_size, inter_size, num_experts, topk, workspace_ptr, final_output, token_topk_final_scales,
                expanded_source_row_to_expanded_dest_row, expert_for_source_row, tp_size, rank, use_lora,
                best_config_index, NT, stream, is_fp8, scale1, scale2, scale3, custom_routing_function, apply_weight);
}

#define INVOKE_MOE_CUTLASS_GEMM(T, WT, OT, NT)                                                                         \
  template void InvokeMoeCutlassGemm<T, WT, OT, NT>(                                                                   \
      void const* input_activations, void* gating_output, void const* fc1_expert_weights,                              \
      void const* fc2_expert_weights, void* e_score_correction_bias, int64_t const num_rows,                           \
      int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const topk, char* workspace_ptr, \
      void* final_output, void* token_topk_final_scales, int* expanded_source_row_to_expanded_dest_row,                \
      int* expert_for_source_row, int tp_size, int rank, bool use_lora, size_t best_config_index, bool use_vllm_moe_,  \
      uint32_t num_expert_group_, uint32_t expert_groups_topk_, const std::string& scoring_func_,                      \
      const std::string& topk_method_, bool norm_topk_prob_, float routed_scaling_factor_,                             \
      bool use_e_score_correction_bias_, cudaStream_t stream, bool is_fp8, void const* scale1, void const* scale2,     \
      void const* scale3, bool apply_weight)

INVOKE_MOE_CUTLASS_GEMM(float, float, float, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE);
INVOKE_MOE_CUTLASS_GEMM(float, float, float, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE);
INVOKE_MOE_CUTLASS_GEMM(half, half, half, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE);
INVOKE_MOE_CUTLASS_GEMM(half, half, half, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE);
#ifdef ENABLE_FP8
INVOKE_MOE_CUTLASS_GEMM(__nv_fp8_e4m3, __nv_fp8_e4m3, half, llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE);
INVOKE_MOE_CUTLASS_GEMM(__nv_fp8_e4m3, __nv_fp8_e4m3, half,
                        llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE);
#endif

#ifdef ENABLE_BFLOAT16
INVOKE_MOE_CUTLASS_GEMM(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                        llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE);
INVOKE_MOE_CUTLASS_GEMM(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                        llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE);
#  ifdef ENABLE_FP8
INVOKE_MOE_CUTLASS_GEMM(__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                        llm_kernels::nvidia::MOEExpertScaleNormalizationMode::NONE);
INVOKE_MOE_CUTLASS_GEMM(__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                        llm_kernels::nvidia::MOEExpertScaleNormalizationMode::RENORMALIZE);
#  endif
#endif
#undef INVOKE_MOE_CUTLASS_GEMM

template <typename T, llm_kernels::nvidia::WeightType WT>
void GetFpAIntBGroupCutlassGemmWorkspaceSize(size_t m, size_t n, size_t k, size_t& ws_bytes) {
  auto gemm = llm_kernels::nvidia::FpAIntBGroupCutlassGemmWrapper<T, WT>();
  gemm.GetWorkspaceSize(m, n, k, ws_bytes);
}
#define GET_FPA_INTB_GROUP_CUTLASS_GEMM_WORKSPACE_SIZE(T, WT) \
  template void GetFpAIntBGroupCutlassGemmWorkspaceSize<T, WT>(size_t m, size_t n, size_t k, size_t& ws_bytes)
GET_FPA_INTB_GROUP_CUTLASS_GEMM_WORKSPACE_SIZE(float, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_GROUP_CUTLASS_GEMM_WORKSPACE_SIZE(float, llm_kernels::nvidia::WeightType::INT8);
GET_FPA_INTB_GROUP_CUTLASS_GEMM_WORKSPACE_SIZE(half, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_GROUP_CUTLASS_GEMM_WORKSPACE_SIZE(half, llm_kernels::nvidia::WeightType::INT8);
#ifdef ENABLE_BFLOAT16
GET_FPA_INTB_GROUP_CUTLASS_GEMM_WORKSPACE_SIZE(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_GROUP_CUTLASS_GEMM_WORKSPACE_SIZE(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
#endif
#undef GET_FPA_INTB_GROUP_CUTLASS_GEMM_WORKSPACE_SIZE

template <typename T, llm_kernels::nvidia::WeightType WT>
void InvokeFpAIntBGroupCutlassGemm(void* output, const void* input, const void* weight, const void* scales,
                                   const void* zeros, void* ws, size_t m, size_t n, size_t k, size_t groupsize,
                                   size_t config_index, cudaStream_t stream) {
  auto gemm = llm_kernels::nvidia::FpAIntBGroupCutlassGemmWrapper<T, WT>();
  CUDA_CHECK_LAST_ERROR(gemm.Gemm(output, input, weight, scales, zeros, ws, m, n, k, groupsize, config_index, stream));
}
#define INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM(T, WT)                                                                     \
  template void InvokeFpAIntBGroupCutlassGemm<T, WT>(                                                                 \
      void* output, const void* input, const void* weight, const void* scales, const void* zeros, void* ws, size_t m, \
      size_t n, size_t k, size_t groupsize, size_t config_index, cudaStream_t stream)
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM(float, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM(float, llm_kernels::nvidia::WeightType::INT8);
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM(half, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM(half, llm_kernels::nvidia::WeightType::INT8);
#ifdef ENABLE_BFLOAT16
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
#endif
#undef INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM

template <typename T, llm_kernels::nvidia::WeightType WT>
size_t InvokeFpAIntBGroupCutlassGemmConfigProfile(size_t warmup, size_t iter, void* output, const void* input,
                                                  const void* weight, const void* scales, const void* zeros, void* ws,
                                                  size_t m, size_t n, size_t k, size_t groupsize, cudaStream_t stream) {
  auto gemm = llm_kernels::nvidia::FpAIntBGroupCutlassGemmWrapper<T, WT>();
  return gemm.GetBestConfigIndex(warmup, iter, output, input, weight, scales, zeros, ws, m, n, k, groupsize, stream);
}
#define INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM_CONFIG_PROGILE(T, WT)                                           \
  template size_t InvokeFpAIntBGroupCutlassGemmConfigProfile<T, WT>(                                       \
      size_t warmup, size_t iter, void* output, const void* input, const void* weight, const void* scales, \
      const void* zeros, void* ws, size_t m, size_t n, size_t k, size_t groupsize, cudaStream_t stream)
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM_CONFIG_PROGILE(float, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM_CONFIG_PROGILE(float, llm_kernels::nvidia::WeightType::INT8);
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM_CONFIG_PROGILE(half, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM_CONFIG_PROGILE(half, llm_kernels::nvidia::WeightType::INT8);
#ifdef ENABLE_BFLOAT16
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM_CONFIG_PROGILE(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM_CONFIG_PROGILE(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
#endif
#undef INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM_CONFIG_PROGILE

template <typename T, llm_kernels::nvidia::WeightType WT>
bool GetFpAIntBGroupCudaGemmSupported() {
  auto gemm = llm_kernels::nvidia::FpAIntBGroupCudaGemmWrapper<T, WT>();
  return gemm.IsSupport();
}
#define GET_FPA_INTB_GROUP_CUDA_GEMM_SUPPORTED(T, WT) template bool GetFpAIntBGroupCudaGemmSupported<T, WT>()
GET_FPA_INTB_GROUP_CUDA_GEMM_SUPPORTED(float, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_GROUP_CUDA_GEMM_SUPPORTED(float, llm_kernels::nvidia::WeightType::INT8);
GET_FPA_INTB_GROUP_CUDA_GEMM_SUPPORTED(half, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_GROUP_CUDA_GEMM_SUPPORTED(half, llm_kernels::nvidia::WeightType::INT8);
#ifdef ENABLE_BFLOAT16
GET_FPA_INTB_GROUP_CUDA_GEMM_SUPPORTED(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_GROUP_CUDA_GEMM_SUPPORTED(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
#endif
#undef GET_FPA_INTB_GROUP_CUDA_GEMM_SUPPORTED

template <typename T, llm_kernels::nvidia::WeightType WT>
void InvokeFpAIntBGroupCudaGemm(void* output, const void* input, const void* weight, const void* scales,
                                const void* zeros, size_t m, size_t n, size_t k, size_t groupsize,
                                cudaStream_t stream) {
  auto gemm = llm_kernels::nvidia::FpAIntBGroupCudaGemmWrapper<T, WT>();
  CUDA_CHECK_LAST_ERROR(gemm.Gemm(output, input, weight, scales, zeros, m, n, k, groupsize, stream));
}
#define INVOKE_FPA_INTB_GROUP_CUDA_GEMM(T, WT)                                                                         \
  template void InvokeFpAIntBGroupCudaGemm<T, WT>(void* output, const void* input, const void* weight,                 \
                                                  const void* scales, const void* zeros, size_t m, size_t n, size_t k, \
                                                  size_t groupsize, cudaStream_t stream)
INVOKE_FPA_INTB_GROUP_CUDA_GEMM(float, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUDA_GEMM(float, llm_kernels::nvidia::WeightType::INT8);
INVOKE_FPA_INTB_GROUP_CUDA_GEMM(half, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUDA_GEMM(half, llm_kernels::nvidia::WeightType::INT8);
#ifdef ENABLE_BFLOAT16
INVOKE_FPA_INTB_GROUP_CUDA_GEMM(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUDA_GEMM(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
#endif
#undef INVOKE_FPA_INTB_GROUP_CUDA_GEMM

template <typename T>
void LookupEmbedding(const void* input_ids, const void* ids_offsets, const void* prefix_offsets, const void* emb,
                     const void* pos, const void* steps, void* output, const T emb_scale, int vocab_size,
                     int hidden_size, int bs, int vocab_id, cudaStream_t stream, void* workspace_ptr) {
  const bool do_position_encoding = (pos != nullptr) && (steps != nullptr);
  if (do_position_encoding) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<T, true>(
        reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), reinterpret_cast<const T*>(pos), emb_scale, {},
        reinterpret_cast<const int32_t*>(input_ids), reinterpret_cast<const size_t*>(steps),
        reinterpret_cast<const size_t*>(ids_offsets), reinterpret_cast<const size_t*>(prefix_offsets), bs, hidden_size,
        vocab_size, vocab_id, stream));
  } else {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<T, false>(
        reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), /* pos */ nullptr, emb_scale, {},
        reinterpret_cast<const int32_t*>(input_ids), /* steps */ nullptr, reinterpret_cast<const size_t*>(ids_offsets),
        reinterpret_cast<const size_t*>(prefix_offsets), bs, hidden_size, vocab_size, vocab_id, stream));
  }
}
#define LOOKUP_EMBEDDING(T)                                                                                    \
  template void LookupEmbedding<T>(const void* input_ids, const void* ids_offsets, const void* prefix_offsets, \
                                   const void* emb, const void* pos, const void* steps, void* output,          \
                                   const T emb_scale, int vocab_size, int hidden_size, int bs, int vocab_id,   \
                                   cudaStream_t stream, void* workspace_ptr)
LOOKUP_EMBEDDING(float);
LOOKUP_EMBEDDING(half);
#ifdef ENABLE_BFLOAT16
LOOKUP_EMBEDDING(__nv_bfloat16);
#endif
#undef LOOKUP_EMBEDDING

template <typename T>
void InvokeLayerNorm(const void* input, const void* weight, const void* bias, const float layernorm_eps, const int m,
                     const int n, void* output, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeLayerNorm<T>(
      reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input), reinterpret_cast<const T*>(weight),
      reinterpret_cast<const T*>(bias), layernorm_eps, m, n, stream));
}
#define INVOKE_LAYER_NORM(T)                                                                                           \
  template void InvokeLayerNorm<T>(const void* input, const void* weight, const void* bias, const float layernorm_eps, \
                                   const int m, const int n, void* output, cudaStream_t stream)
INVOKE_LAYER_NORM(float);
INVOKE_LAYER_NORM(half);
#ifdef ENABLE_BFLOAT16
INVOKE_LAYER_NORM(__nv_bfloat16);
#endif
#undef INVOKE_LAYER_NORM

#define INVOKE_MATMUL(T, CUDA_TYPE)                                                                                    \
  template <>                                                                                                          \
  void InvokeMatMul<T>(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,            \
                       const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream, void* workspace_ptr,   \
                       cublasLtMatmulAlgo_t* cublaslt_algo) {                                                          \
    CUDA_CHECK(llm_kernels::nvidia::InvokeCublasGemm(cublas_handle, cublaslt_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m,   \
                                                     k, b_ptr, n, CUDA_TYPE, a_ptr, k, CUDA_TYPE, c_ptr, n, CUDA_TYPE, \
                                                     CUDA_R_32F, stream, workspace_ptr, 0, cublaslt_algo));            \
  }
INVOKE_MATMUL(float, CUDA_R_32F);
INVOKE_MATMUL(half, CUDA_R_16F);
#ifdef ENABLE_BFLOAT16
INVOKE_MATMUL(__nv_bfloat16, CUDA_R_16BF);
#endif
#undef INVOKE_MATMUL

#define INVOKE_BATCHED_GEMM(T, CUDA_TYPE)                                                                            \
  template <>                                                                                                        \
  void InvokeBatchedMatMul<T>(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int batch_size, int m, \
                              int n, int k, const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream, \
                              void* workspace_ptr, size_t workspace_size, cublasLtMatmulAlgo_t* cublaslt_algo) {     \
    CUDA_CHECK(llm_kernels::nvidia::InvokeCublasGemm(                                                                \
        cublas_handle, cublaslt_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, b_ptr, n, CUDA_TYPE, a_ptr, k, CUDA_TYPE, \
        c_ptr, n, CUDA_TYPE, CUDA_R_32F, batch_size, stream, workspace_ptr, workspace_size, cublaslt_algo));         \
  }
INVOKE_BATCHED_GEMM(float, CUDA_R_32F);
INVOKE_BATCHED_GEMM(half, CUDA_R_16F);
#ifdef ENABLE_BFLOAT16
INVOKE_BATCHED_GEMM(__nv_bfloat16, CUDA_R_16BF);
#endif
#undef INVOKE_BATCHED_GEMM

template <typename T>
void InvokeAddBiasResidual(const void* input_a, const void* input_b, const void* bias, const int m, const int n,
                           void* output, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeAddBiasResidual<T>(
      reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input_a), reinterpret_cast<const T*>(input_b), nullptr,
      reinterpret_cast<const T*>(bias), nullptr, nullptr, m, n, stream));
}

#define INVOKE_ADD_BIAS_RESIDUAL(T)                                                                               \
  template void InvokeAddBiasResidual<T>(const void* input_a, const void* input_b, const void* bias, const int m, \
                                         const int n, void* output, cudaStream_t stream)
INVOKE_ADD_BIAS_RESIDUAL(float);
INVOKE_ADD_BIAS_RESIDUAL(half);
#ifdef ENABLE_BFLOAT16
INVOKE_ADD_BIAS_RESIDUAL(__nv_bfloat16);
#endif
#undef INVOKE_ADD_BIAS_RESIDUAL

template <template <typename T> class Activation, typename T>
void InvokeGatedActivation(const void* input, const void* bias, const void* gated_weights, const void* gated_bias,
                           const int m, const int n, void* output, cudaStream_t stream) {
  if (output != input) {
    KLLM_THROW("Activation is an in-place operation, `output` must be the same as `input`.");
  }
  const int* ia3_tasks = nullptr;
  const T* ia3_weights = nullptr;
  const int int8_mode = 0;
  const int* padding_offsets = nullptr;
  const int seq_len = 0;
  const float* activation_in = nullptr;
  const float* activation_out = nullptr;
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeGenericActivation<Activation, T, T>(
      reinterpret_cast<T*>(output), reinterpret_cast<const T*>(bias), reinterpret_cast<const T*>(gated_weights),
      reinterpret_cast<const T*>(gated_bias), ia3_tasks, ia3_weights, m, n, int8_mode, activation_in, activation_out,
      padding_offsets, seq_len, stream));
}

template <template <typename T> class Activation, typename T>
void InvokeRowBasedGatedActivation(const void* input, const int m, const int n, void* output, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeRowBasedActivation<Activation, T>(
      reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input), m, n, stream));
}

#define INVOKE_ROW_BASED_GATED_ACTIVATION(Activation, T)                                                  \
  template void InvokeRowBasedGatedActivation<Activation, T>(const void* input, const int m, const int n, \
                                                             void* output, cudaStream_t stream)
INVOKE_ROW_BASED_GATED_ACTIVATION(llm_kernels::nvidia::SiluActivation, float);
INVOKE_ROW_BASED_GATED_ACTIVATION(llm_kernels::nvidia::SiluActivation, half);
#ifdef ENABLE_BFLOAT16
INVOKE_ROW_BASED_GATED_ACTIVATION(llm_kernels::nvidia::SiluActivation, __nv_bfloat16);
#endif

#define INVOKE_GATED_ACTIVATION(Activation, T)                                                                       \
  template void InvokeGatedActivation<Activation, T>(const void* input, const void* bias, const void* gated_weights, \
                                                     const void* gated_bias, const int m, const int n, void* output, \
                                                     cudaStream_t stream)
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::GeluActivation, float);
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::GeluActivation, half);
#ifdef ENABLE_BFLOAT16
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::GeluActivation, __nv_bfloat16);
#endif

INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::SiluActivation, float);
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::SiluActivation, half);
#ifdef ENABLE_BFLOAT16
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::SiluActivation, __nv_bfloat16);
#endif

INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::ReluActivation, float);
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::ReluActivation, half);
#ifdef ENABLE_BFLOAT16
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::ReluActivation, __nv_bfloat16);
#endif

// Enables kContextDecodeUseFP8Cache to simulate the effect of KV cache quantization on flash attention,
// intended for use in testing accuracy outcomes only.
static bool kContextDecodeUseFP8Cache = []() -> bool {
  const char* val = std::getenv("ContextDecodeUseFP8Cache");
  if (val != nullptr) {
    return true;
  }
  return false;
}();

template <typename T>
void InvokeQKRmsNorm(void* qkv_ptr, const void* q_gamma, const void* k_gamma, const float layernorm_eps,
                     const int32_t total_tokens, const int32_t num_heads, const int32_t num_kv_heads,
                     const int32_t head_size, const int64_t* mask, cudaStream_t stream) {
  // q 3D norm
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeRmsNorm3D<T>(
      reinterpret_cast<T*>(qkv_ptr), reinterpret_cast<const T*>(qkv_ptr), reinterpret_cast<const T*>(q_gamma),
      layernorm_eps, total_tokens, (num_heads + num_kv_heads * 2), head_size, 0, num_heads, mask, stream));
  // k 3D norm
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeRmsNorm3D<T>(
      reinterpret_cast<T*>(qkv_ptr), reinterpret_cast<const T*>(qkv_ptr), reinterpret_cast<const T*>(k_gamma),
      layernorm_eps, total_tokens, (num_heads + num_kv_heads * 2), head_size, num_heads, (num_heads + num_kv_heads),
      mask, stream));
}
#define INVOKE_QK_LAYER_NORM(T)                                                                                        \
  template void InvokeQKRmsNorm<T>(void* qkv_ptr, const void* q_gamma, const void* k_gamma, const float layernorm_eps, \
                                   const int32_t total_tokens, const int32_t num_heads, const int32_t num_kv_heads,    \
                                   const int32_t head_size, const int64_t* mask, cudaStream_t stream)
INVOKE_QK_LAYER_NORM(float);
INVOKE_QK_LAYER_NORM(half);
#ifdef ENABLE_BFLOAT16
INVOKE_QK_LAYER_NORM(__nv_bfloat16);
#endif
#undef INVOKE_QK_LAYER_NORM

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void AttenVarlen(void* qkv_ptr, void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen,
                 std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                 int total_tokens, int max_tokens, int batch, int num_heads, int num_kv_heads, int head_size,
                 int stride_size, float k_scale, float v_scale, size_t tensor_para_size, bool is_causal, int rank,
                 int block_size, void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,
                 const std::optional<void*>& alibi_slopes, int layer_index, void* flexible_rotary_embedding_pos_ptr,
                 void* flexible_rotary_embedding_mask_ptr, void* dst_flexible_kv_cache_ptr,
                 void* src_flexible_kv_cache_ptr, void* dst_flexible_token_idx_ptr, void* src_flexible_token_idx_ptr,
                 void* flexible_offset_uint64_ptr, int flexible_len, float layernorm_eps, bool use_qk_norm,
                 void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream, void* k_cache_ptr,
                 void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq,
                 size_t* without_prefix_offsets, int max_forwarding_tokens, bool enable_qk_pre_norm_before_rotary_pos,
                 bool no_rope, bool attn_temperature_tuning, float attn_scale, size_t floor_scale) {
  // qk norm before rotary position embedding
  if (enable_qk_pre_norm_before_rotary_pos && q_norm_weight != nullptr && k_norm_weight != nullptr) {
    InvokeQKRmsNorm<SCALAR_T>(qkv_ptr, q_norm_weight, k_norm_weight, layernorm_eps, total_tokens, num_heads,
                              num_kv_heads, head_size, reinterpret_cast<int64_t*>(rotary_embedding_mask), stream);
  }
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
  torch::Tensor qkv_tensor =
      torch::from_blob(qkv_ptr, {total_tokens, (num_heads + num_kv_heads * 2) * head_size}, options);
  auto tt = qkv_tensor.split({num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size}, -1);
  auto int_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt64);
  torch::Tensor seqlen_tensor = torch::from_blob(seqlen, {batch + 1}, int_options);

  // attn_backend config.
  AttnBackendConfig attn_backend_config;
  Singleton<Environment>::GetInstance()->GetAttnBackendConfig(attn_backend_config);
  c10::optional<at::Tensor> null_tensor = c10::nullopt;
  c10::optional<const at::Tensor> const_null_tensor = c10::nullopt;

  // rotary embedding
  torch::Tensor q_tensor = tt[0];
  torch::Tensor k_tensor = tt[1];
  torch::Tensor v_tensor = tt[2];
  if (flexible_len != 0) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FlexibleReverseCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<CACHE_T**>(src_flexible_kv_cache_ptr), reinterpret_cast<CACHE_T**>(dst_flexible_kv_cache_ptr),
        reinterpret_cast<int*>(src_flexible_token_idx_ptr), reinterpret_cast<int*>(dst_flexible_token_idx_ptr),
        block_size, layer_index, flexible_len, num_kv_heads, head_size, stride_size, stream));
  }

  if (use_cache && !attn_backend_config.enable_blocked_multi_token_forwarding_kv) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ReverseCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
        v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(prefix_offsets),
        reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
        k_scale, v_scale, stream));
  }

  if (!no_rope && rotary_embedding_cuda.has_value()) {
    if (flexible_len != 0) {
      rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(flexible_rotary_embedding_pos_ptr),
                                      reinterpret_cast<int64_t*>(flexible_rotary_embedding_mask_ptr), nullptr,
                                      reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), total_tokens, stream);
      CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
    }

    rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                    reinterpret_cast<int64_t*>(rotary_embedding_mask),
                                    reinterpret_cast<SCALAR_T*>(q_tensor.data_ptr()),
                                    reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
  }

  if (!no_rope && use_qk_norm) {
    InvokeQKRmsNorm<SCALAR_T>(qkv_ptr, q_norm_weight, k_norm_weight, layernorm_eps, total_tokens, num_heads,
                              num_kv_heads, head_size, reinterpret_cast<int64_t*>(rotary_embedding_mask), stream);
  }

  if (attn_temperature_tuning) {
    torch::Tensor positions_tensor =
        torch::from_blob(rotary_embedding_pos, {total_tokens}, int_options).to(torch::kFloat32);
    torch::Tensor attn_scale_tensor =
        torch::log(torch::floor((positions_tensor + 1.0f) / static_cast<float>(floor_scale)) + 1.0f) * attn_scale +
        1.0f;
    attn_scale_tensor = attn_scale_tensor.unsqueeze(-1).to(q_tensor.dtype());
    // Notice: attn_scale_tensor and q_tensor's multiplication is fp32 in transformers
    torch::mul_out(q_tensor, q_tensor, attn_scale_tensor);
  }

  if (use_cache) {
    if (attn_backend_config.enable_blocked_multi_token_forwarding_kv)
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CacheCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
          v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(prefix_offsets), without_prefix_offsets,
          reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
          k_scale, v_scale, stream));
    else
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
          v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(flexible_offset_uint64_ptr),
          reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
          k_scale, v_scale, stream));
  }

// flash attention 2 or flash attention 1
#if defined(ENABLE_FLASH_ATTN_2) || defined(ENABLE_VLLM_FLASH_ATTN_2)
  // refer to github Dao-AILab/flash-attention csrc/flash_attn/flash_api.cpp#L374
  // When the flag is set to True and the output is not nullptr, calling the function mha_varlen_fwd
  // leads to a core dump.
  bool seqlenq_ngroups_swapped =
      max_tokens == 1 && num_heads > num_kv_heads && head_size % 8 == 0 && !alibi_slopes.has_value();
  c10::optional<at::Tensor> out_tensor = c10::nullopt;
  if (!seqlenq_ngroups_swapped) {
    out_tensor = torch::from_blob(out, {total_tokens, num_heads, head_size}, options);
  }
  at::Tensor q_tmp_tensor = torch::reshape(q_tensor, {total_tokens, num_heads, head_size});
  c10::optional<at::Tensor> seqused_k = c10::nullopt;
  c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
  if (alibi_slopes.has_value()) {
    alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
  }
  // Enables kContextDecodeUseFP8Cache to simulate the effect of KV cache quantization on flash attention,
  // intended for use in testing accuracy outcomes only.
  if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kAuto) {
    if (kContextDecodeUseFP8Cache) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), k_tensor.size(0), k_tensor.size(1), stride_size, k_scale,
          stream));
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), v_tensor.size(0), v_tensor.size(1), stride_size, v_scale,
          stream));
    }
  }
  // vllm-attn apis.
#  ifdef ENABLE_VLLM_FLASH_ATTN_MINOR_6
  std::vector<at::Tensor> mha_output;
  if (attn_backend_config.enable_blocked_multi_token_forwarding_kv) {
    torch::Tensor seqlen_q_tensor = torch::from_blob(without_prefix_offsets, {batch + 1}, int_options);
    auto cache_options = options;
    if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 ||
        KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      // cache_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt8);
      KLLM_THROW("FlashAttention not support fp8 kv cache");
    }
    // kv_cache[num_blocks, block_size, num_kv_heads, head_size]
    torch::Tensor k_cache_tensor =
        torch::from_blob(k_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    torch::Tensor v_cache_tensor =
        torch::from_blob(v_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>());
    c10::optional<at::Tensor> block_table =
        torch::from_blob(block_table_ptr, {batch, max_blocks_per_seq}, int32_options);
    mha_output = mha_varlen_fwd(q_tmp_tensor, k_cache_tensor, v_cache_tensor, out_tensor,
                                seqlen_q_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), seqused_k,
                                block_table, alibi_slopes_tensor, max_forwarding_tokens, max_tokens, 0.f,
                                1.0 / sqrt(head_size), false, is_causal, -1, -1, 0.f, false, c10::nullopt);
  } else {
    c10::optional<at::Tensor> block_table = c10::nullopt;  // batch_size x max_num_blocks_per_seq
    mha_output = mha_varlen_fwd(q_tmp_tensor, torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                                torch::reshape(tt[2], {total_tokens, num_kv_heads, head_size}), out_tensor,
                                seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), seqused_k,
                                block_table, alibi_slopes_tensor, max_tokens, max_tokens, 0.f, 1.0 / sqrt(head_size),
                                false, is_causal, -1, -1, 0.f, false, c10::nullopt);
  }
#  endif

  // flash_attn v.2.4, 2.5.6. and later versions.
#  if defined(ENABLE_FLASH_ATTN_2)
#    if defined(ENABLE_FLASH_ATTN_MINNOR_4) || defined(ENABLE_FLASH_ATTN_MINOR_5)
  std::vector<at::Tensor> mha_output =
      mha_varlen_fwd(q_tmp_tensor, torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                     torch::reshape(tt[2], {total_tokens, num_kv_heads, head_size}), out_tensor,
                     seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), seqused_k, alibi_slopes_tensor,
                     max_tokens, max_tokens, 0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, false, c10::nullopt);

#    else  // Since v2.7.2.post1, add two more parms, such as block_table_,leftpad_t_.
  std::vector<at::Tensor> mha_output = mha_varlen_fwd(
      q_tmp_tensor, torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
      torch::reshape(tt[2], {total_tokens, num_kv_heads, head_size}), out_tensor, seqlen_tensor.to(torch::kInt32),
      seqlen_tensor.to(torch::kInt32), seqused_k, const_null_tensor, /* leftpad_k_  */
      null_tensor,                                                   /* block_table */
      alibi_slopes_tensor, max_tokens, max_tokens, 0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, 0.0, false,
      c10::nullopt);
#    endif
#  endif
  if (seqlenq_ngroups_swapped) {
    KLLM_LOG_DEBUG << "To prevent a core dump when seqlenq_ngroups_swapped is True, set the output tensor to nullptr.";
    at::Tensor& out_data = mha_output[0];
    size_t total_size = out_data.numel() * out_data.element_size();
    CUDA_CHECK(cudaMemcpyAsync(out, out_data.data_ptr(), total_size, cudaMemcpyDeviceToDevice, stream));
  }

#else  // flash_attn v1.x?
  c10::optional<at::Tensor> out_tensor = torch::from_blob(out, {total_tokens, num_heads, head_size}, options);
  flash_attn::mha_varlen_fwd(torch::reshape(q_tensor, {total_tokens, num_heads, head_size}),
                             torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                             torch::reshape(tt[2], {total_tokens, num_kv_heads, head_size}), out_tensor,
                             seqlen_tensor.to(torch::kInt32), seqlen_tensor.to(torch::kInt32), max_tokens, max_tokens,
                             0.f, 1.0 / sqrt(head_size), false, is_causal, -1, -1, false, c10::nullopt);
#endif
}

#define ATTEN_VARLEN(SCALAR_T, CACHE_T, KV_DTYPE)                                                                     \
  template void AttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(                                                             \
      void* qkv_ptr, void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen,                \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, int total_tokens,     \
      int max_tokens, int batch, int num_heads, int num_kv_heads, int head_size, int stride_size, float k_scale,      \
      float v_scale, size_t tensor_para_size, bool is_causal, int rank, int block_size, void** k_list, void** v_list, \
      void* prefix_offsets, void* block_offsets, const std::optional<void*>& alibi_slopes, int layer_index,           \
      void* flexible_rotary_embedding_pos_ptr, void* flexible_rotary_embedding_mask_ptr,                              \
      void* dst_flexible_kv_cache_ptr, void* src_flexible_kv_cache_ptr, void* dst_flexible_token_idx_ptr,             \
      void* src_flexible_token_idx_ptr, void* flexible_offset_uint64_ptr, int flexible_len, float layernorm_eps,      \
      bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream,                \
      void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num,                     \
      int max_blocks_per_seq, size_t* without_prefix_offsets, int max_forwarding_tokens,                              \
      bool enable_qk_pre_norm_before_rotary_pos, bool no_rope, bool attn_temperature_tuning, float attn_scale,        \
      size_t floor_scale)
ATTEN_VARLEN(float, float, llm_kernels::utils::KVCacheType::kAuto);
ATTEN_VARLEN(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
ATTEN_VARLEN(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
ATTEN_VARLEN(half, half, llm_kernels::utils::KVCacheType::kAuto);
ATTEN_VARLEN(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
ATTEN_VARLEN(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#ifdef ENABLE_BFLOAT16
ATTEN_VARLEN(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
ATTEN_VARLEN(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
ATTEN_VARLEN(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef ATTEN_VARLEN

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void PagedAttention(int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size, float k_scale,
                    float v_scale, void* out, void* q_tensor_ptr, void* key_cache_ptrs, void* value_cache_ptrs,
                    void* cache_offsets_ptr, void* context_lens_ptr, int max_context_len, int num_seqs,
                    cudaStream_t& stream, void* workspace, size_t work_size, const float* alibi_slopes_ptr);

#define PAGED_ATTENTION(T1, T2, CACHE_T1, CACHE_T2, KV_DTYPE)                                                        \
  template <>                                                                                                        \
  void PagedAttention<T1, CACHE_T1, KV_DTYPE>(                                                                       \
      int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size, float k_scale, float v_scale, \
      void* out, void* q_tensor_ptr, void* key_cache_ptrs, void* value_cache_ptrs, void* cache_offsets_ptr,          \
      void* context_lens_ptr, int max_context_len, int num_seqs, cudaStream_t& stream, void* workspace,              \
      size_t work_size, const float* alibi_slopes_ptr) {                                                             \
    llm_kernels::nvidia::PagedAttentionCuda<T2, CACHE_T2, KV_DTYPE> op;                                              \
    op.SetConfig(num_kv_heads, num_heads, head_size, block_size, stride_size, k_scale, v_scale);                     \
    op.SetInput(reinterpret_cast<T2*>(out), reinterpret_cast<const T2*>(q_tensor_ptr),                               \
                reinterpret_cast<CACHE_T2**>(key_cache_ptrs), reinterpret_cast<CACHE_T2**>(value_cache_ptrs),        \
                reinterpret_cast<const int*>(cache_offsets_ptr), reinterpret_cast<const int*>(context_lens_ptr),     \
                max_context_len, num_seqs, stream, workspace, work_size, alibi_slopes_ptr);                          \
    CUDA_CHECK_LAST_ERROR(op.Forward());                                                                             \
  }
PAGED_ATTENTION(float, float, float, float, llm_kernels::utils::KVCacheType::kAuto);
PAGED_ATTENTION(float, float, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
PAGED_ATTENTION(float, float, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
PAGED_ATTENTION(half, uint16_t, half, uint16_t, llm_kernels::utils::KVCacheType::kAuto);
PAGED_ATTENTION(half, uint16_t, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
PAGED_ATTENTION(half, uint16_t, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#ifdef ENABLE_BFLOAT16
PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, uint8_t, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef PAGED_ATTENTION

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokePagedAttention(void* output_ptr, void* query_ptr, void** key_cache_ptrs, void** value_cache_ptrs,
                          void* context_lens_ptr, int max_context_len, cudaStream_t stream, void* cache_offsets_ptr,
                          int seqs_num, int num_heads, int head_size, int num_kv_heads, int stride_size, int block_size,
                          float k_scale, float v_scale, int batch, void* rotary_embedding_pos,
                          void* rotary_embedding_mask, int total_tokens,
                          std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                          void* workspace_ptr, float layernorm_eps, bool use_qk_norm, void* q_norm_weight,
                          void* k_norm_weight, size_t work_size, int rank, const std::optional<void*>& alibi_slopes,
                          void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr,
                          int64_t kv_cache_block_num, int max_blocks_per_seq, bool enable_qk_pre_norm_before_rotary_pos,
                          bool no_rope, bool attn_temperature_tuning, float attn_scale, size_t floor_scale) {
  // qk norm before rotary position embedding for paged attention
  if (enable_qk_pre_norm_before_rotary_pos && q_norm_weight != nullptr && k_norm_weight != nullptr) {
    InvokeQKRmsNorm<SCALAR_T>(query_ptr, q_norm_weight, k_norm_weight, layernorm_eps, total_tokens, num_heads,
                              num_kv_heads, head_size, reinterpret_cast<int64_t*>(rotary_embedding_mask), stream);
  }
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
  auto int_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt64);
  torch::Tensor qkv_tensor =
      torch::from_blob(query_ptr, {total_tokens, (num_heads + num_kv_heads * 2) * head_size}, options);
  auto tt = qkv_tensor.split({num_heads * head_size, num_kv_heads * head_size, num_kv_heads * head_size}, -1);

  torch::Tensor q_tensor = tt[0];
  torch::Tensor k_tensor = tt[1];
  torch::Tensor v_tensor = tt[2];
  void* q_tensor_ptr = q_tensor.data_ptr();
  void* k_tensor_ptr = k_tensor.data_ptr();
  void* v_tensor_ptr = v_tensor.data_ptr();

  AttnBackendConfig attn_backend_config;
  Singleton<Environment>::GetInstance()->GetAttnBackendConfig(attn_backend_config);

  if (!no_rope && rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(
        reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<int64_t*>(rotary_embedding_mask),
        reinterpret_cast<SCALAR_T*>(q_tensor_ptr), reinterpret_cast<SCALAR_T*>(k_tensor_ptr), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
  }
  if (!no_rope && use_qk_norm) {
    InvokeQKRmsNorm<SCALAR_T>(query_ptr, q_norm_weight, k_norm_weight, layernorm_eps, total_tokens, num_heads,
                              num_kv_heads, head_size, reinterpret_cast<int64_t*>(rotary_embedding_mask), stream);
  }

  if (attn_temperature_tuning) {
    torch::Tensor positions_tensor =
        torch::from_blob(rotary_embedding_pos, {total_tokens}, int_options).to(torch::kFloat32);
    torch::Tensor attn_scale_tensor =
        torch::log(torch::floor((positions_tensor + 1.0f) / static_cast<float>(floor_scale)) + 1.0f) * attn_scale +
        1.0f;
    attn_scale_tensor = attn_scale_tensor.unsqueeze(-1).to(q_tensor.dtype());
    // Notice: attn_scale_tensor and q_tensor's multiplication is fp32 in transformers
    torch::mul_out(q_tensor, q_tensor, attn_scale_tensor);
  }

  if (attn_backend_config.enable_blocked_multi_token_forwarding_kv) {
    constexpr size_t kReqQLen = 1;
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(v_tensor_ptr), key_cache_ptrs,
        value_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr),
        block_size, batch, kReqQLen, num_kv_heads, head_size, stride_size, k_scale, v_scale, stream));
  } else {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(v_tensor_ptr), key_cache_ptrs,
        value_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr),
        block_size, batch, total_tokens, num_kv_heads, head_size, stride_size, k_scale, v_scale, stream));
  }

  if (attn_backend_config.enable_blocked_multi_token_forwarding_kv) {
    auto cache_options = options;
    if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 ||
        KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      KLLM_THROW("FlashAttention not support fp8 kv cache");
    }
    // kv_cache[num_blocks, 2, block_size, num_kv_heads, head_size]
    torch::Tensor k_cache_tensor =
        torch::from_blob(k_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    torch::Tensor v_cache_tensor =
        torch::from_blob(v_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>());
    c10::optional<at::Tensor> block_table_tensor =
        torch::from_blob(block_table_ptr, {batch, max_blocks_per_seq}, int32_options);
    c10::optional<const at::Tensor> seqlens_k_tensor =
        c10::optional<const at::Tensor>(torch::from_blob(context_lens_ptr, {batch}, int32_options));
    q_tensor = q_tensor.reshape({batch, 1, num_heads, head_size});
    c10::optional<at::Tensor> out_tensor = torch::from_blob(output_ptr, {batch, 1, num_heads, head_size}, options);
    float softmax_scale = 1.0 / sqrt(head_size);
    c10::optional<at::Tensor> null_tensor = c10::nullopt;
    c10::optional<const at::Tensor> const_null_tensor = c10::nullopt;
    c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
    if (alibi_slopes.has_value()) {
      alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
    }

    //  Not support flash-attn < 2.6.0 && != 2.5.6
#if defined(ENABLE_FLASH_ATTN_2) && defined(ENABLE_FLASH_ATTN_MINOR_5)
    mha_fwd_kvcache(q_tensor,             // batch_size x seqlen_q x num_heads x head_size
                    k_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    v_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    const_null_tensor,    // k_
                    const_null_tensor,    // v_
                    seqlens_k_tensor,     // batch_size
                    const_null_tensor,    // rotary_cos_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // rotary_sin_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // cache_batch_idx_: indices to index into the KV cache
                    block_table_tensor,   // batch_size x max_num_blocks_per_seq
                    alibi_slopes_tensor,  // num_heads or batch_size x num_heads
                    out_tensor,           // batch_size x seqlen_q x num_heads x head_size
                    softmax_scale, true, -1, -1, true, 0);
#elif defined(ENABLE_FLASH_ATTN_2) && defined(ENABLE_FLASH_ATTN_MINOR_7)
    // add leftpad_k_ param since flash-attn 2.7.2
    mha_fwd_kvcache(q_tensor,             // batch_size x seqlen_q x num_heads x head_size
                    k_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    v_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    const_null_tensor,    // k_
                    const_null_tensor,    // v_
                    seqlens_k_tensor,     // batch_size
                    const_null_tensor,    // rotary_cos_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // rotary_sin_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // cache_batch_idx_: indices to index into the KV cache
                    const_null_tensor,    // indices that the KV cache starts. [batch_size,], nullptr, default 0
                    block_table_tensor,   // batch_size x max_num_blocks_per_seq
                    alibi_slopes_tensor,  // num_heads or batch_size x num_heads
                    out_tensor,           // batch_size x seqlen_q x num_heads x head_size
                    softmax_scale, true, -1, -1, 0.0, true, 0);

#else  // It works for vllm-flash-attn and flash-attn >= v2.5.6
    mha_fwd_kvcache(q_tensor,             // batch_size x seqlen_q x num_heads x head_size
                    k_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    v_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                    const_null_tensor,    // k_
                    const_null_tensor,    // v_
                    seqlens_k_tensor,     // batch_size
                    const_null_tensor,    // rotary_cos_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // rotary_sin_: seqlen_ro x (rotary_dim / 2)
                    const_null_tensor,    // cache_batch_idx_: indices to index into the KV cache
                    block_table_tensor,   // batch_size x max_num_blocks_per_seq
                    alibi_slopes_tensor,  // num_heads or batch_size x num_heads
                    out_tensor,           // batch_size x seqlen_q x num_heads x head_size
                    softmax_scale, true, -1, -1, 0.0, true, 0);
#endif

  } else {
    const float* alibi_slopes_ptr =
        reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);
    PagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(num_heads, head_size, num_kv_heads, stride_size, block_size, k_scale,
                                                v_scale, output_ptr, q_tensor_ptr, key_cache_ptrs, value_cache_ptrs,
                                                cache_offsets_ptr, context_lens_ptr, max_context_len, seqs_num, stream,
                                                workspace_ptr, work_size, alibi_slopes_ptr);
  }
}

#define RUN_PAGED_ATTENTION(SCALAR_T, CACHE_T, KV_DTYPE)                                                             \
  template void InvokePagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(                                                   \
      void* output_ptr, void* query_ptr, void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr,     \
      int max_context_len, cudaStream_t stream, void* cache_offsets_ptr, int seqs_num, int num_heads, int head_size, \
      int num_kv_heads, int stride_size, int block_size, float k_scale, float v_scale, int batch,                    \
      void* rotary_embedding_pos, void* rotary_embedding_mask, int total_tokens,                                     \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, void* workspace_ptr, \
      float layernorm_eps, bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, size_t work_size, int rank,   \
      const std::optional<void*>& alibi_slopes, void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr,           \
      int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq,                                  \
      bool enable_qk_pre_norm_before_rotary_pos, bool no_rope, bool attn_temperature_tuning, float attn_scale,       \
      size_t floor_scale)
RUN_PAGED_ATTENTION(float, float, llm_kernels::utils::KVCacheType::kAuto);
RUN_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_PAGED_ATTENTION(half, half, llm_kernels::utils::KVCacheType::kAuto);
RUN_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#ifdef ENABLE_BFLOAT16
RUN_PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
RUN_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef RUN_PAGED_ATTENTION

template <typename T>
void AssembleTokensHidden(const void* inputs, const void* logits_idx, const int batch_size, const int hidden_units_num,
                          void* output, cudaStream_t& stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::AssembleTokensHidden<T>(
      reinterpret_cast<const T*>(inputs), reinterpret_cast<const size_t*>(logits_idx), batch_size, hidden_units_num,
      reinterpret_cast<T*>(output), stream));
}

#define ASSEMBEL_LAST_TOKEN(T)                                                                            \
  template void AssembleTokensHidden<T>(const void* inputs, const void* logits_idx, const int batch_size, \
                                        const int hidden_units_num, void* output, cudaStream_t& stream);
ASSEMBEL_LAST_TOKEN(float);
ASSEMBEL_LAST_TOKEN(half);
#ifdef ENABLE_BFLOAT16
ASSEMBEL_LAST_TOKEN(__nv_bfloat16);
#endif
#undef ASSEMBEL_LAST_TOKEN

template <typename T>
void Concat(const void* input_a, const void* input_b, size_t concat_size_a, size_t concat_size_b, size_t outer_dim_size,
            size_t inner_dim_size, void* output, cudaStream_t& stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::Concat<T>(
      reinterpret_cast<const T*>(input_a), reinterpret_cast<const T*>(input_b), concat_size_a, concat_size_b,
      outer_dim_size, inner_dim_size, reinterpret_cast<T*>(output), stream));
}

#define CONCAT(T)                                                                                               \
  template void Concat<T>(const void* input_a, const void* input_b, size_t concat_size_a, size_t concat_size_b, \
                          size_t outer_dim_size, size_t inner_dim_size, void* output, cudaStream_t& stream);
CONCAT(float);
CONCAT(half);
#ifdef ENABLE_BFLOAT16
CONCAT(__nv_bfloat16);
#endif
#undef CONCAT

template <typename T>
void Expand(void* input, void* output, const int m, const int expand_size, const int n, const size_t stride,
            cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeExpand<T>(
      reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output), m, expand_size, n, stride, stream));
}
#define INVOKE_EXPAND(T)                                                                              \
  template void Expand<T>(void* input, void* output, const int m, const int expand_size, const int n, \
                          const size_t stride, cudaStream_t stream)
INVOKE_EXPAND(float);
INVOKE_EXPAND(half);
#ifdef ENABLE_BFLOAT16
INVOKE_EXPAND(__nv_bfloat16);
#endif
#undef INVOKE_EXPAND

// ptr: 指向所维护的CustomAllreduce算子的指针
// signals: 所有GPU共享的中间结果的指针数组
// rank_data & rank_data_sz: 当前GPU的中间结果数据
template <typename T>
void CustomAllReduceInit(void** ptr, void** signals, void* rank_data, size_t rank_data_sz, int cur_rank,
                         int total_ranks, bool is_full_nvlink, uint32_t root_rank) {
  *ptr = new llm_kernels::nvidia::CustomAllreduce((llm_kernels::nvidia::Signal**)signals, rank_data, rank_data_sz,
                                                  cur_rank, total_ranks, is_full_nvlink, root_rank);
}

#define CUSTOM_ALL_REDUCE_INIT(T)                                                                                      \
  template void CustomAllReduceInit<T>(void** ptr, void** signals, void* rank_data, size_t rank_data_sz, int cur_rank, \
                                       int total_ranks, bool is_full_nvlink, uint32_t root_rank);
CUSTOM_ALL_REDUCE_INIT(float);
CUSTOM_ALL_REDUCE_INIT(half);
#ifdef ENABLE_BFLOAT16
CUSTOM_ALL_REDUCE_INIT(__nv_bfloat16);
#endif
#undef CUSTOM_ALL_REDUCE_INIT

template <typename T>
void CustomAllReduceRegisterBuffer(void* ptr, void** input_handles, cudaStream_t& stream) {
  llm_kernels::nvidia::CustomAllreduce* reduce_op = static_cast<llm_kernels::nvidia::CustomAllreduce*>(ptr);
  reduce_op->RegisterBuffer(input_handles, stream);
}

#define CUSTOM_ALL_REDUCE_REGISTER_BUFFER(T) \
  template void CustomAllReduceRegisterBuffer<T>(void* ptr, void** input_handles, cudaStream_t& stream);
CUSTOM_ALL_REDUCE_REGISTER_BUFFER(float);
CUSTOM_ALL_REDUCE_REGISTER_BUFFER(half);
#ifdef ENABLE_BFLOAT16
CUSTOM_ALL_REDUCE_REGISTER_BUFFER(__nv_bfloat16);
#endif
#undef CUSTOM_ALL_REDUCE_REGISTER_BUFFER

template <typename T>
void CustomAllReduceRun(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream) {
  llm_kernels::nvidia::CustomAllreduce* reduce_op = static_cast<llm_kernels::nvidia::CustomAllreduce*>(ptr);
  reduce_op->AllReduce<T>(stream, static_cast<T*>(input), static_cast<T*>(result), data_size);
}

template void CustomAllReduceRun<float>(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream);
template void CustomAllReduceRun<half>(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream);
#ifdef ENABLE_BFLOAT16
template void CustomAllReduceRun<__nv_bfloat16>(void* ptr, void* input, void* result, int data_size,
                                                cudaStream_t& stream);
#endif

template <typename T>
void InvokeSigmoidActivation(void* input, const size_t size, const float scale, cudaStream_t& stream) {
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokeSigmoid<T>(reinterpret_cast<T*>(input), static_cast<int32_t>(size), scale, stream));
}

template void InvokeSigmoidActivation<float>(void* input, const size_t size, const float scale, cudaStream_t& stream);
template void InvokeSigmoidActivation<half>(void* input, const size_t size, const float scale, cudaStream_t& stream);
#ifdef ENABLE_BFLOAT16
template void InvokeSigmoidActivation<__nv_bfloat16>(void* input, const size_t size, const float scale,
                                                     cudaStream_t& stream);
#endif

template <>
ncclDataType_t GetNcclDataType<float>() {
  return ncclFloat;
}
template <>
ncclDataType_t GetNcclDataType<half>() {
  return ncclHalf;
}
#ifdef ENABLE_BFLOAT16
template <>
ncclDataType_t GetNcclDataType<__nv_bfloat16>() {
  return ncclBfloat16;
}
#endif

template <typename T>
void InvokePermute(void* input, void* output, std::vector<size_t> input_shape, std::vector<size_t> permutation,
                   cudaStream_t& stream) {
  KLLM_CHECK_WITH_INFO(input_shape.size() <= 4ul,
                       fmt::format("input shape dims number {} > 4 is not supported", input_shape.size()));
  if (input_shape.empty()) {
    return;
  }

  // Extend to num_dims = 4
  input_shape.resize(4, 1);
  for (size_t i = permutation.size(); i < 4; ++i) {
    permutation.push_back(i);
  }
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokePermute<4ul, sizeof(T)>(input, output, input_shape, permutation, stream));
}
#define INVOKE_PERMUTE(T)                                                                    \
  template void InvokePermute<T>(void* input, void* output, std::vector<size_t> input_shape, \
                                 std::vector<size_t> permutation, cudaStream_t& stream);
INVOKE_PERMUTE(float);
INVOKE_PERMUTE(half);
#ifdef ENABLE_BFLOAT16
INVOKE_PERMUTE(__nv_bfloat16);
#endif
#undef INVOKE_PERMUTE

template <>
void DataToFloat<float>(const void* input, const int data_size, const size_t vocab_size, const size_t vocab_size_pad,
                        void* output, cudaStream_t& stream) {
  if (input != output) {
    if (vocab_size != vocab_size_pad) {
      // It should be implemented when supporting float inference.
      KLLM_LOG_ERROR << "Float to float does not support Stride.";
    }
    CUDA_CHECK(cudaMemcpyAsync(output, input, data_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  }
}
template <>
void DataToFloat<half>(const void* input, const int data_size, const size_t vocab_size, const size_t vocab_size_pad,
                       void* output, cudaStream_t& stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToFloat(reinterpret_cast<const half*>(input), data_size,
                                                         reinterpret_cast<float*>(output), stream, vocab_size_pad,
                                                         vocab_size));
}
#ifdef ENABLE_BFLOAT16
template <>
void DataToFloat<__nv_bfloat16>(const void* input, const int data_size, const size_t vocab_size,
                                const size_t vocab_size_pad, void* output, cudaStream_t& stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToFloat(reinterpret_cast<const __nv_bfloat16*>(input), data_size,
                                                             reinterpret_cast<float*>(output), stream, vocab_size_pad,
                                                             vocab_size));
}
#endif

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  if (tensor.dtype == DataType::TYPE_BF16 && target_dtype == DataType::TYPE_FP16) {
#ifdef ENABLE_BFLOAT16
    CUDA_CHECK_LAST_ERROR(
        llm_kernels::nvidia::BFloat16ToHalf(tensor.GetPtr<void>(), tensor.GetElementNumber(), stream.Get()));
#endif
  } else if (tensor.dtype == DataType::TYPE_FP16 && target_dtype == DataType::TYPE_BF16) {
#ifdef ENABLE_BFLOAT16
    CUDA_CHECK_LAST_ERROR(
        llm_kernels::nvidia::HalfToBFloat16(tensor.GetPtr<void>(), tensor.GetElementNumber(), stream.Get()));
#endif
  } else if (tensor.dtype == target_dtype) {
    // No need to convert
  } else {
    KLLM_THROW(fmt::format("CastInplace from type {} to {} is not yet implement", tensor.dtype, target_dtype));
  }
  tensor.dtype = target_dtype;
  return Status();
}

Status Permute(Tensor& input_tensor, Tensor& output_tensor, const std::vector<size_t>& permutation, Stream& stream,
               void* workspace_ptr) {
  if (input_tensor.dtype == TYPE_INT32) {
    InvokePermute<int32_t>(input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), input_tensor.shape, permutation,
                           stream.Get());
  } else if (input_tensor.dtype == TYPE_FP32) {
    InvokePermute<float>(input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), input_tensor.shape, permutation,
                         stream.Get());
  } else if (input_tensor.dtype == TYPE_FP16) {
    InvokePermute<half>(input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), input_tensor.shape, permutation,
                        stream.Get());
  } else if (input_tensor.dtype == TYPE_BF16) {
#ifdef ENABLE_BFLOAT16
    InvokePermute<__nv_bfloat16>(input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), input_tensor.shape,
                                 permutation, stream.Get());
#else
    KLLM_THROW(fmt::format("Permute of type {} is not yet implement", input_tensor.dtype));
#endif
  } else if (input_tensor.dtype == TYPE_FP8_E4M3) {
#ifdef ENABLE_FP8
    InvokePermute<__nv_fp8_e4m3>(input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), input_tensor.shape,
                                 permutation, stream.Get());
#else
    KLLM_THROW(fmt::format("Permute of type {} is not yet implement", GetTypeString(input_tensor.dtype)));
#endif
  } else {
    KLLM_THROW(fmt::format("Permute of type {} is not yet implement", GetTypeString(input_tensor.dtype)));
  }
  return Status();
}

template <typename T>
void InvokeMul(void* a, void* b, void* c, int m1, int n1, int m2, int n2, int device_rank) {
  auto options = torch::TensorOptions().device(torch::kCUDA, device_rank).dtype(GetTorchDataType<T>());
  auto a_tensor = torch::from_blob(a, {m1, n1}, options);
  auto b_tensor = torch::from_blob(b, {m2, n2}, options);
  auto c_tensor = torch::from_blob(c, {m1 >= m2 ? m1 : m2, n1 >= n2 ? n1 : n2}, options);
  mul_out(c_tensor, a_tensor, b_tensor);
  c = c_tensor.data_ptr();
}
#define InvokeMUL(T) \
  template void InvokeMul<T>(void* a, void* b, void* c, int m1, int n1, int m2, int n2, int device_rank);
InvokeMUL(float);
InvokeMUL(half);
#ifdef ENABLE_BFLOAT16
InvokeMUL(__nv_bfloat16);
#endif
#undef InvokeMUL

// c = InvokeMul(a, b)
void InvokeMul(float* a, float* b, float* c, int n, int device_rank) {
  auto options = torch::TensorOptions().device(torch::kCUDA, device_rank).dtype(torch::kFloat32);
  torch::Tensor a_tensor = torch::from_blob(a, {n}, options);
  torch::Tensor b_tensor = torch::from_blob(b, {n}, options);
  torch::Tensor c_tensor = torch::from_blob(c, {n}, options);
  torch::mul_out(c_tensor, a_tensor, b_tensor);
}

// out = div(1, in)
void Reciprocal(float* out, float* in, int n, int device_rank) {
  auto options = torch::TensorOptions().device(torch::kCUDA, device_rank).dtype(torch::kFloat32);
  torch::Tensor ones = torch::ones({n}, options);
  torch::Tensor in_tensor = torch::from_blob(in, {n}, options);
  torch::Tensor out_tensor = torch::from_blob(out, {n}, options);
  torch::div_out(out_tensor, ones, in_tensor);
}

// out = max(a, b)
void Max(float* out, float* a, float* b, int n, int device_rank) {
  auto options = torch::TensorOptions().device(torch::kCUDA, device_rank).dtype(torch::kFloat32);
  torch::Tensor a_tensor = torch::from_blob(a, {n}, options);
  torch::Tensor out_tensor = torch::from_blob(out, {n}, options);
  torch::Tensor b_tensor = torch::from_blob(b, {n}, options);
  torch::max_out(out_tensor, a_tensor, b_tensor);
}

void CalcLogprobs(float* logits, float* temperatures, int vocab_size, int bs, int logprobs_num, float* logprobs,
                  int64_t* token_ids) {
  auto options = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat32);
  auto logits_tensor = torch::from_blob(logits, {bs, vocab_size}, options);

  torch::Tensor logits_sort, logits_idx;
  std::tie(logits_sort, logits_idx) = logits_tensor.sort(-1, true);

  logits_sort = logits_sort.narrow(1, 0, logprobs_num);
  if (temperatures != nullptr) {
    auto temperatures_tensor = torch::from_blob(temperatures, {bs}, options);
    logits_sort = logits_sort.div_(temperatures_tensor.unsqueeze_(1));
  }
  logits_sort = logits_sort.log_softmax(-1).to(torch::kCPU).view({-1});
  logits_idx = logits_idx.narrow(1, 0, logprobs_num).to(torch::kCPU).view({-1});

  memcpy(logprobs, logits_sort.data_ptr<float>(), logprobs_num * bs * sizeof(float));
  memcpy(token_ids, logits_idx.data_ptr<int64_t>(), logprobs_num * bs * sizeof(int64_t));
}

template <typename T>
Status ArgMax(const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result, Stream& stream,
              void* buffer_ptr) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeArgMaxReduce(input, batch_size, vocab_size, result, stream.Get()));
  return Status();
}

#define INSTANTIATE_ARG_MAX(T)                                                                                    \
  template Status ArgMax<T>(const T* input, const int32_t batch_size, const int32_t vocab_size, uint32_t* result, \
                            Stream& stream, void* buffer_ptr);

INSTANTIATE_ARG_MAX(float);
INSTANTIATE_ARG_MAX(half);
#ifdef ENABLE_BF16
INSTANTIATE_ARG_MAX(__nv_bfloat16);
#endif

#undef INSTANTIATE_ARG_MAX

#ifdef ENABLE_FP8
#  define INSTANTIATE_FP8_E4M3_QUANTIZE(T)                                                                             \
    template <>                                                                                                        \
    void Fp8E4m3Quantize<T>(int num_channels, int channel_size, const T* input_ptr, void* quant_ptr, float* scale_ptr, \
                            bool is_static, cudaStream_t& stream) {                                                    \
      if (!is_static) {                                                                                                \
        CUDA_CHECK_LAST_ERROR(llm_kernels::utils::InvokeComputeFP8QuantizeScale<T>(scale_ptr, input_ptr, num_channels, \
                                                                                   channel_size, stream));             \
      }                                                                                                                \
      CUDA_CHECK_LAST_ERROR(llm_kernels::utils::InvokeQuantizeMatrix<__nv_fp8_e4m3, T>(                                \
          static_cast<__nv_fp8_e4m3*>(quant_ptr), scale_ptr, input_ptr, num_channels, channel_size, stream));          \
    }
INSTANTIATE_FP8_E4M3_QUANTIZE(float);
INSTANTIATE_FP8_E4M3_QUANTIZE(half);
#  ifdef ENABLE_BFLOAT16
INSTANTIATE_FP8_E4M3_QUANTIZE(__nv_bfloat16);
#  endif
#  undef INSTANTIATE_FP8_E4M3_QUANTIZE

#  define INVOKE_FP8_QUANTIZED_MATMUL(T, CUDA_TYPE)                                                                    \
    template <>                                                                                                        \
    void Fp8QuantizedMatMul<T>(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,    \
                               const void* a_ptr, const void* a_scale, const void* b_ptr, const void* b_scale,         \
                               T* c_ptr, cudaStream_t& stream, cublasLtMatmulAlgo_t* cublaslt_algo, void* workspace) { \
      CUDA_CHECK(llm_kernels::nvidia::InvokeCublasGemm(cublas_handle, cublaslt_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, \
                                                       k, b_ptr, k, CUDA_R_8F_E4M3, a_ptr, k, CUDA_R_8F_E4M3, c_ptr,   \
                                                       n, CUDA_TYPE, 1, 1.0f, 0.f, CUDA_R_32F, stream, workspace, 0,   \
                                                       nullptr, a_scale, b_scale));                                    \
    }

INVOKE_FP8_QUANTIZED_MATMUL(float, CUDA_R_32F);
INVOKE_FP8_QUANTIZED_MATMUL(half, CUDA_R_16F);
#  ifdef ENABLE_BFLOAT16
INVOKE_FP8_QUANTIZED_MATMUL(__nv_bfloat16, CUDA_R_16BF);
#  endif
#  undef INVOKE_FP8_QUANTIZED_MATMUL

void RescaleFp8E4m3(void* input, void* output, size_t n, const float* input_scale, const float* output_scale,
                    cudaStream_t& stream) {
  llm_kernels::utils::InvokeRescaleFp8E4m3(input, output, n, input_scale, output_scale, stream);
}
#endif

size_t InvokeGetCublasWorkspaceSize() { return llm_kernels::nvidia::GetCublasWorkspaceSize(); }

#ifdef ENABLE_VLLM_FLASH_ATTN_2
cudaStream_t InvokeSetTorchStream(cudaStream_t& stream, int rank) {
  cudaStream_t old_stream = c10::cuda::getCurrentCUDAStream(rank).stream();
  // set compute stream as torch stream
  c10::cuda::CUDAStream new_stream = c10::cuda::getStreamFromExternal(stream, rank);
  c10::cuda::setCurrentCUDAStream(new_stream);
  return old_stream;
}
#endif

// Adapted from
// [DeepSeek-V3 Project] https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L393
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaAttenVarlen(void* hidden_buffer_0, void* q_nope_ptr, void* q_pe_ptr, void* k_pe_ptr, void* compressed_kv_ptr,
                    void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* o_proj_weight,
                    void* kv_b_nope_weight_scale, void* v_head_weight_scale, void* o_weight_scale, size_t o_proj_dim,
                    void* workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles,
                    void* rotary_embedding_pos, void* rotary_embedding_mask, void* out, void* seqlen, float attn_scale,
                    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                    int total_tokens, int max_tokens, int batch, int num_heads, int qk_rope_head_dim,
                    int qk_nope_head_dim, int kv_lora_rank, int v_head_dim, int num_kv_heads, int head_size,
                    int stride_size, float k_scale, float v_scale, size_t tensor_para_size, bool is_causal, int rank,
                    int block_size, void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,
                    const std::optional<void*>& alibi_slopes, int layer_index, void* flexible_rotary_embedding_pos_ptr,
                    void* flexible_rotary_embedding_mask_ptr, void* dst_flexible_kv_cache_ptr,
                    void* src_flexible_kv_cache_ptr, void* dst_flexible_token_idx_ptr, void* src_flexible_token_idx_ptr,
                    void* flexible_offset_uint64_ptr, int flexible_len, float layernorm_eps, bool use_qk_norm,
                    void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream, void* k_cache_ptr,
                    void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq,
                    size_t* without_prefix_offsets, int max_forwarding_tokens, int total_prefix_len,
                    void* seqlens_q_ptr, void* prefix_k_buffer, void* prefix_v_buffer, void* prefix_o_buffer,
                    void* prefix_kv_buffer, void* prefix_k_up_buffer, void* prefix_v_up_buffer,
                    QuantMode mm_quant_mode) {
  // 修改stride_size 和 head_size
  stride_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim);
  head_size = qk_nope_head_dim + qk_rope_head_dim;

  // k & v have different size in MLA condition.
  int k_stride_size = qk_rope_head_dim;
  int v_stride_size = kv_lora_rank;

  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  auto int_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt64);
  auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt32);

  static thread_local void* seqlen_int32 = nullptr;
  static thread_local void* seqlens_q_ptr_int32 = nullptr;
  static thread_local void* without_prefix_offsets_int32 = nullptr;
  if (seqlen_int32 == nullptr) {
    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

    cudaMallocAsync(&seqlen_int32, batch_scheduler_config.max_batch_size * sizeof(int), stream);
    cudaMallocAsync(&seqlens_q_ptr_int32, batch_scheduler_config.max_batch_size * sizeof(int), stream);
    cudaMallocAsync(&without_prefix_offsets_int32, batch_scheduler_config.max_batch_size * sizeof(int), stream);
  }
  llm_kernels::nvidia::Int64ToInt(reinterpret_cast<const int64_t*>(seqlen), batch + 1,
                                  reinterpret_cast<int*>(seqlen_int32), stream);
  llm_kernels::nvidia::Int64ToInt(reinterpret_cast<const int64_t*>(seqlens_q_ptr), batch + 1,
                                  reinterpret_cast<int*>(seqlens_q_ptr_int32), stream);

  torch::Tensor seqlen_tensor = torch::from_blob(seqlen_int32, {batch + 1}, int32_options);
  torch::Tensor prefill_seq_q_tensor = torch::from_blob(seqlens_q_ptr_int32, {batch + 1}, int32_options);

  // kv_b_proj升维矩阵乘
  // 1. kv_nope_proj
  if (kv_b_nope_weight_scale != nullptr) {
    if (workspace == nullptr) {
      KLLM_THROW("Quantized matmul has not workspace");
    }
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      SCALAR_T* a = static_cast<SCALAR_T*>(compressed_kv_ptr);
      void* a_q = workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * kv_lora_rank);
      InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, kv_lora_rank, true, stream);
      float* b_scale = static_cast<float*>(kv_b_nope_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, kv_b_nope_proj_weight, b_scale, hidden_buffer_0, total_tokens, kv_lora_rank,
                                num_heads * qk_nope_head_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * qk_nope_head_dim, kv_lora_rank);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, workspace, stream, total_tokens, num_heads * qk_nope_head_dim, kv_lora_rank,
                        compressed_kv_ptr, kv_b_nope_proj_weight, hidden_buffer_0, GetMacheteDataType<SCALAR_T>(),
                        llm_kernels::nvidia::vllm_dtype::kU4B8, kv_b_nope_weight_scale,
                        std::optional<std::vector<size_t>>({static_cast<size_t>(kv_lora_rank / 128),
                                                            static_cast<size_t>(num_heads * qk_nope_head_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    }
  } else {
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * qk_nope_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(kv_b_nope_proj_weight), hidden_buffer_0, stream, nullptr,
                           nullptr);
  }
  // 2. v_head_proj
  if (v_head_weight_scale != nullptr) {
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      void* a_q = workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * kv_lora_rank);
      float* b_scale = static_cast<float*>(v_head_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, v_head_proj_weight, b_scale, out, total_tokens, kv_lora_rank,
                                num_heads * v_head_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * v_head_dim, kv_lora_rank);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, workspace, stream, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                        compressed_kv_ptr, v_head_proj_weight, out, GetMacheteDataType<SCALAR_T>(),
                        llm_kernels::nvidia::vllm_dtype::kU4B8, v_head_weight_scale,
                        std::optional<std::vector<size_t>>(
                            {static_cast<size_t>(kv_lora_rank / 128), static_cast<size_t>(num_heads * v_head_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    }
  } else {
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(v_head_proj_weight), out, stream, nullptr, nullptr);
  }

  if (flexible_len != 0) {
    // 暂时先不支持灵活cache，先不做处理
    KLLM_THROW("Not support flexible length cache.");
  }
#ifndef ENABLE_FLASH_ATTN_WITH_CACHE
  // MLA使用vllm_flash_attn,暂时先不做处理
  KLLM_LOG_INFO << "Only support flash attn with cache, skip";
#endif
  if (rotary_embedding_cuda.has_value()) {
    if (flexible_len != 0) {
      KLLM_THROW("Not support flexible length cache.");
    }

    rotary_embedding_cuda->SetInput(
        reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<int64_t*>(rotary_embedding_mask),
        reinterpret_cast<SCALAR_T*>(q_pe_ptr), reinterpret_cast<SCALAR_T*>(k_pe_ptr), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
  }

  /*
  Input Parameters:
    q_nope : q_nope_ptr       [total_tokens, num_heads, qk_nope_head_dim]
    k_nope : hidden_buffer_0  [total_tokens, num_heads, qk_nope_head_dim]
    q_pe   : q_pe_ptr         [total_tokens, num_heads, qk_rope_head_dim]
    k_pe   : k_pe_ptr         [total_tokens, 1, qk_rope_head_dim]
    v_pe   : out              [total_tokens, num_heads, v_head_dim]

  Intermediate Tensors:
    k_pe_expanded : k_pe_expanded_ptr  [total_tokens, num_heads, qk_rope_head_dim]
    v_pad         : v_pad_ptr          [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim - v_head_dim]
    q_tensor      : q_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
    k_tensor      : k_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
    v_tensor      : v_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]

  Memory Buffer Allocation:
    hidden_buffer_0 : [k_nope][q_tensor][k_tensor]
    out             : [v_pe][k_pe_expanded][v_pad][v_tensor]
  */

  size_t qk_combined_size = total_tokens * num_heads * (qk_nope_head_dim + qk_rope_head_dim) * sizeof(SCALAR_T);
  size_t v_tensor_size = total_tokens * num_heads * v_head_dim * sizeof(SCALAR_T);
  size_t k_pe_expanded_size = total_tokens * num_heads * qk_rope_head_dim * sizeof(SCALAR_T);
  size_t v_pad_size = total_tokens * num_heads * (qk_nope_head_dim + qk_rope_head_dim - v_head_dim) * sizeof(SCALAR_T);
  size_t k_nope_size = total_tokens * num_heads * qk_nope_head_dim * sizeof(SCALAR_T);

  size_t q_tensor_offset = (k_nope_size + 1023) & ~(1023);
  size_t k_tensor_offset = (q_tensor_offset + qk_combined_size + 1023) & ~(1023);
  size_t k_pe_expanded_offset = (v_tensor_size + 1023) & ~(1023);
  size_t v_pad_offset = (k_pe_expanded_offset + k_pe_expanded_size + 1023) & ~(1023);
  size_t v_tensor_offset = (v_pad_offset + v_pad_size + 1023) & ~(1023);

  void* q_tensor_ptr = static_cast<char*>(hidden_buffer_0) + q_tensor_offset;
  void* k_tensor_ptr = static_cast<char*>(hidden_buffer_0) + k_tensor_offset;
  void* k_pe_expanded_ptr = static_cast<char*>(out) + k_pe_expanded_offset;
  void* v_pad_ptr = static_cast<char*>(out) + v_pad_offset;
  void* v_tensor_ptr = static_cast<char*>(out) + v_tensor_offset;

  torch::Tensor q_tensor =
      torch::from_blob(q_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);
  torch::Tensor k_tensor =
      torch::from_blob(k_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);
  torch::Tensor v_tensor =
      torch::from_blob(v_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);

  const size_t outer_dim_size = total_tokens * num_heads;
  const size_t inner_dim_size = 1;

  // cat(q_nope, q_pe)
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   q_tensor.data_ptr(), stream);

  // cat(k_nope, k_pe)
  Expand<SCALAR_T>(k_pe_ptr, k_pe_expanded_ptr, total_tokens, num_heads, qk_rope_head_dim, 0, stream);
  Concat<SCALAR_T>(hidden_buffer_0, k_pe_expanded_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size,
                   inner_dim_size, k_tensor.data_ptr(), stream);

  // pad v
  CUDA_CHECK(cudaMemsetAsync(v_pad_ptr, 0, v_pad_size, stream));
  Concat<SCALAR_T>(out, v_pad_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   v_tensor.data_ptr(), stream);

  if (use_cache) {
#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
    // Use compressed kvcache, k is [num_token, qk_rope_head_dim], v is  [num_token, kv_lora_rank]
    if (IsAbsorbWeightsEnabled()) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaKVCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_pe_ptr), reinterpret_cast<SCALAR_T*>(compressed_kv_ptr), k_list, v_list,
          reinterpret_cast<size_t*>(prefix_offsets), without_prefix_offsets, reinterpret_cast<int*>(block_offsets),
          block_size, batch, total_tokens, k_stride_size, v_stride_size, k_scale, v_scale, stream));
    } else {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CacheCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
          v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(prefix_offsets), without_prefix_offsets,
          reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
          k_scale, v_scale, stream));
    }
#else
    KLLM_THROW("Only support ENABLE_FLASH_ATTN_WITH_CACHE");
#endif
  }

  // flash attention 2 or flash attention 1
#if defined(ENABLE_FLASH_ATTN_2) || defined(ENABLE_VLLM_FLASH_ATTN_2)

  // refer to github Dao-AILab/flash-attention csrc/flash_attn/flash_api.cpp#L374
  // When the flag is set to True and the output is not nullptr, calling the function mha_varlen_fwd
  // leads to a core dump.
  bool seqlenq_ngroups_swapped =
      max_tokens == 1 && num_heads > num_kv_heads && head_size % 8 == 0 && !alibi_slopes.has_value();
  c10::optional<at::Tensor> out_tensor = c10::nullopt;
  if (!seqlenq_ngroups_swapped) {
    out_tensor = torch::from_blob(out, {total_tokens, num_heads, head_size}, options);
  }
  at::Tensor q_tmp_tensor = torch::reshape(q_tensor, {total_tokens, num_heads, head_size});
  c10::optional<at::Tensor> seqused_k = c10::nullopt;
  auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
  c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
  if (alibi_slopes.has_value()) {
    alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
  }
  // Enables kContextDecodeUseFP8Cache to simulate the effect of KV cache quantization on flash attention,
  // intended for use in testing accuracy outcomes only.
  if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kAuto) {
    if (kContextDecodeUseFP8Cache) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), k_tensor.size(0), k_tensor.size(1), stride_size, k_scale,
          stream));
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), v_tensor.size(0), v_tensor.size(1), stride_size, v_scale,
          stream));
    }
  }
  std::vector<at::Tensor> mha_output;
#  if defined(ENABLE_VLLM_FLASH_ATTN_MINOR_6) || defined(ENABLE_VLLM_FLASH_ATTN_MINOR_7)
#    ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  if (IsAbsorbWeightsEnabled()) {
    if (total_prefix_len > 0) {
      KLLM_LOG_DEBUG << "Prefix caching triggered, unique len:" << total_tokens << ", prefix len:" << total_prefix_len;

      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaCopyValueBlockToBuffer<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(prefix_kv_buffer), v_list, kv_lora_rank + qk_rope_head_dim, 0, kv_lora_rank,
          kv_lora_rank, reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<int*>(block_offsets), block_size,
          total_prefix_len, stream));

      if (kv_b_nope_weight_scale != nullptr) {
        if (workspace == nullptr) {
          KLLM_THROW("FP8 quantized matmul has not workspace");
        }
        if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
          SCALAR_T* a = static_cast<SCALAR_T*>(prefix_kv_buffer);
          void* a_q = workspace;
          float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_prefix_len * kv_lora_rank);
          InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_prefix_len, kv_lora_rank, true, stream);

          float* b_scale = static_cast<float*>(kv_b_nope_weight_scale);
          InvokeBlockGemm<SCALAR_T>(a_q, a_s, kv_b_nope_proj_weight, b_scale, prefix_k_up_buffer, total_prefix_len,
                                    kv_lora_rank, num_heads * qk_nope_head_dim, stream);
        } else if (mm_quant_mode == QUANT_GPTQ) {
          int64_t workspace_size = 0;
          std::vector<std::string> machete_schedule_map =
              Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * qk_nope_head_dim,
                                                                                kv_lora_rank);
          std::optional<std::string> best_schedule = std::nullopt;
          if (static_cast<size_t>(total_prefix_len) < machete_schedule_map.size()) {
            best_schedule = std::optional<std::string>(machete_schedule_map[total_prefix_len]);
          }
          InvokeMacheteGemm(
              workspace_size, workspace, stream, total_prefix_len, num_heads * qk_nope_head_dim, kv_lora_rank,
              prefix_kv_buffer, kv_b_nope_proj_weight, prefix_k_up_buffer, GetMacheteDataType<SCALAR_T>(),
              llm_kernels::nvidia::vllm_dtype::kU4B8, kv_b_nope_weight_scale,
              std::optional<std::vector<size_t>>(
                  {static_cast<size_t>(kv_lora_rank / 128), static_cast<size_t>(num_heads * qk_nope_head_dim)}),
              GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
        } else {
          KLLM_THROW(fmt::format("MLA not support quant mode: {}", mm_quant_mode));
        }
      } else {
        InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_prefix_len, num_heads * qk_nope_head_dim,
                               kv_lora_rank, prefix_kv_buffer, kv_b_nope_proj_weight, prefix_k_up_buffer, stream,
                               nullptr, nullptr);
      }
      if (v_head_weight_scale != nullptr) {
        if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
          void* a_q = workspace;
          float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_prefix_len * kv_lora_rank);

          float* b_scale = static_cast<float*>(v_head_weight_scale);
          InvokeBlockGemm<SCALAR_T>(a_q, a_s, v_head_proj_weight, b_scale, prefix_v_up_buffer, total_prefix_len,
                                    kv_lora_rank, num_heads * v_head_dim, stream);
        } else if (mm_quant_mode == QUANT_GPTQ) {
          int64_t workspace_size = 0;
          std::vector<std::string> machete_schedule_map =
              Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * v_head_dim, kv_lora_rank);
          std::optional<std::string> best_schedule = std::nullopt;
          if (static_cast<size_t>(total_prefix_len) < machete_schedule_map.size()) {
            best_schedule = std::optional<std::string>(machete_schedule_map[total_prefix_len]);
          }
          InvokeMacheteGemm(workspace_size, workspace, stream, total_prefix_len, num_heads * v_head_dim, kv_lora_rank,
                            prefix_kv_buffer, v_head_proj_weight, prefix_v_up_buffer, GetMacheteDataType<SCALAR_T>(),
                            llm_kernels::nvidia::vllm_dtype::kU4B8, v_head_weight_scale,
                            std::optional<std::vector<size_t>>(
                                {static_cast<size_t>(kv_lora_rank / 128), static_cast<size_t>(num_heads * v_head_dim)}),
                            GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128,
                            best_schedule);
        }
      } else {
        InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_prefix_len, num_heads * v_head_dim, kv_lora_rank,
                               prefix_kv_buffer, v_head_proj_weight, prefix_v_up_buffer, stream, nullptr, nullptr);
      }

      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaExtendKVPrefixWithEmpty<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()),
          reinterpret_cast<SCALAR_T*>(prefix_k_buffer), reinterpret_cast<SCALAR_T*>(prefix_v_buffer),
          reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<size_t*>(without_prefix_offsets), num_heads,
          qk_nope_head_dim + qk_rope_head_dim, total_tokens, stream));

      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaCopyKeyBlockWithReplication<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(prefix_k_buffer), k_list, kv_lora_rank + qk_rope_head_dim, kv_lora_rank,
          qk_rope_head_dim, num_heads, qk_nope_head_dim + qk_rope_head_dim, qk_nope_head_dim,
          reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<size_t*>(without_prefix_offsets),
          reinterpret_cast<int*>(block_offsets), block_size, total_prefix_len, stream));

      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaFillKVPrefix<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(prefix_k_buffer), reinterpret_cast<SCALAR_T*>(prefix_v_buffer),
          reinterpret_cast<SCALAR_T*>(prefix_k_up_buffer), reinterpret_cast<SCALAR_T*>(prefix_v_up_buffer),
          reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<size_t*>(without_prefix_offsets), num_heads,
          qk_nope_head_dim, qk_nope_head_dim + qk_rope_head_dim, total_prefix_len, stream));

      const int total_len_with_prefix = total_tokens + total_prefix_len;
      torch::Tensor prefix_k_tensor =
          torch::from_blob(prefix_k_buffer, {total_len_with_prefix, num_heads, head_size}, options);
      torch::Tensor prefix_v_tensor =
          torch::from_blob(prefix_v_buffer, {total_len_with_prefix, num_heads, head_size}, options);
      c10::optional<at::Tensor> prefix_out_tensor =
          torch::from_blob(out, {total_tokens, num_heads, head_size}, options);

      c10::optional<at::Tensor> block_table = c10::nullopt;
      mha_output = mha_varlen_fwd(q_tensor, prefix_k_tensor, prefix_v_tensor, prefix_out_tensor, prefill_seq_q_tensor,
                                  seqlen_tensor, seqused_k, block_table, alibi_slopes_tensor, max_tokens, max_tokens,
                                  0.f, attn_scale, false, is_causal, -1, -1, 0.f, false, c10::nullopt);
    } else {
      c10::optional<at::Tensor> block_table = c10::nullopt;  // batch_size x max_num_blocks_per_seq
      mha_output = mha_varlen_fwd(q_tmp_tensor, torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                                  torch::reshape(v_tensor, {total_tokens, num_kv_heads, head_size}), out_tensor,
                                  seqlen_tensor, seqlen_tensor, seqused_k, block_table, alibi_slopes_tensor, max_tokens,
                                  max_tokens, 0.f, attn_scale, false, is_causal, -1, -1, 0.f, false, c10::nullopt);
    }
  } else {
    llm_kernels::nvidia::Int64ToInt(reinterpret_cast<const int64_t*>(without_prefix_offsets), batch + 1,
                                    reinterpret_cast<int*>(without_prefix_offsets_int32), stream);
    torch::Tensor seqlen_q_tensor = torch::from_blob(without_prefix_offsets_int32, {batch + 1}, int32_options);

    auto cache_options = options;
    if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 ||
        KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      KLLM_THROW("FlashAttention not support fp8 kv cache");
    }
    // kv_cache[num_blocks, block_size, num_kv_heads, head_size]
    torch::Tensor k_cache_tensor =
        torch::from_blob(k_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    torch::Tensor v_cache_tensor =
        torch::from_blob(v_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>());
    c10::optional<at::Tensor> block_table =
        torch::from_blob(block_table_ptr, {batch, max_blocks_per_seq}, int32_options);

    mha_output = mha_varlen_fwd(q_tmp_tensor, k_cache_tensor, v_cache_tensor, out_tensor, seqlen_q_tensor,
                                seqlen_tensor, seqused_k, block_table, alibi_slopes_tensor, max_forwarding_tokens,
                                max_tokens, 0.f, attn_scale, false, is_causal, -1, -1, 0.f, false, c10::nullopt);
  }
#    else
  KLLM_THROW("Only support ENABLE_FLASH_ATTN_WITH_CACHE");
#    endif
#  endif

#  if defined(ENABLE_FLASH_ATTN_MINOR_4) || defined(ENABLE_FLASH_ATTN_MINOR_5)
  KLLM_THROW("Only support ENABLE_FLASH_ATTN_MINOR_6 or ENABLE_FLASH_ATTN_MINOR_7");
#  endif
  if (seqlenq_ngroups_swapped) {
    KLLM_LOG_DEBUG << "To prevent a core dump when seqlenq_ngroups_swapped is True, "
                      "set the output tensor to nullptr.";
    at::Tensor& out_data = mha_output[0];
    size_t total_size = out_data.numel() * out_data.element_size();
    CUDA_CHECK(cudaMemcpyAsync(hidden_buffer_0, out_data.data_ptr(), total_size, cudaMemcpyDeviceToDevice, stream));
  }

#else
  KLLM_THROW("Only support ENABLE_FLASH_ATTN_2 or ENABLE_VLLM_FLASH_ATTN_2");
#endif

  //  当 v_tensor 被 pad 时调用, 取out_tensor 的 v_head_dim 大小
  size_t dst_pitch = v_head_dim * sizeof(SCALAR_T);
  size_t src_pitch = head_size * sizeof(SCALAR_T);
  // Tensor(MEMORY_DEVICE, TYPE_FP16, {total_tokens, num_heads, qk_nope_head_dim +
  // qk_rope_head_dim}
  CUDA_CHECK(cudaMemcpy2DAsync(hidden_buffer_0, dst_pitch, out, src_pitch, dst_pitch, total_tokens * num_heads,
                               cudaMemcpyDeviceToDevice, stream));
  torch::Tensor out_ttt = torch::from_blob(hidden_buffer_0, {total_tokens, num_heads, v_head_dim}, options);
  // 3. o_proj
  if (o_weight_scale != nullptr) {
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      SCALAR_T* a = static_cast<SCALAR_T*>(hidden_buffer_0);
      void* a_q = workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * num_heads * v_head_dim);
      InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, num_heads * v_head_dim, true, stream);
      float* b_scale = static_cast<float*>(o_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, o_proj_weight, b_scale, out, total_tokens, num_heads * v_head_dim, o_proj_dim,
                                stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(o_proj_dim, num_heads * v_head_dim);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, workspace, stream, total_tokens, o_proj_dim, num_heads * v_head_dim,
                        hidden_buffer_0, o_proj_weight, out, GetMacheteDataType<SCALAR_T>(),
                        llm_kernels::nvidia::vllm_dtype::kU4B8, o_weight_scale,
                        std::optional<std::vector<size_t>>(
                            {static_cast<size_t>(num_heads * v_head_dim / 128), static_cast<size_t>(o_proj_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    } else {
      KLLM_THROW(fmt::format("MLA Prefill not support quant mode: {}", mm_quant_mode));
    }
  } else {
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, o_proj_dim, num_heads * v_head_dim,
                           reinterpret_cast<const void*>(hidden_buffer_0), reinterpret_cast<const void*>(o_proj_weight),
                           out, stream, nullptr, nullptr);
  }
}

#define MLA_ATTEN_VARLEN(SCALAR_T, CACHE_T, KV_DTYPE)                                                               \
  template void MlaAttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(                                                        \
      void* hidden_buffer_0, void* q_nope_ptr, void* q_pe_ptr, void* k_pe_ptr, void* compressed_kv_ptr,             \
      void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* o_proj_weight, void* kv_b_nope_weight_scale,     \
      void* v_head_weight_scale, void* o_weight_scale, size_t o_proj_dim, void* workspace,                          \
      cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles, void* rotary_embedding_pos,               \
      void* rotary_embedding_mask, void* out, void* seqlen, float attn_scale,                                       \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, int total_tokens,   \
      int max_tokens, int batch, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank,       \
      int v_head_dim, int num_kv_heads, int head_size, int stride_size, float k_scale, float v_scale,               \
      size_t tensor_para_size, bool is_causal, int rank, int block_size, void** k_list, void** v_list,              \
      void* prefix_offsets, void* block_offsets, const std::optional<void*>& alibi_slopes, int layer_index,         \
      void* flexible_rotary_embedding_pos_ptr, void* flexible_rotary_embedding_mask_ptr,                            \
      void* dst_flexible_kv_cache_ptr, void* src_flexible_kv_cache_ptr, void* dst_flexible_token_idx_ptr,           \
      void* src_flexible_token_idx_ptr, void* flexible_offset_uint64_ptr, int flexible_len, float layernorm_eps,    \
      bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream,              \
      void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num,                   \
      int max_blocks_per_seq, size_t* without_prefix_offsets, int max_forwarding_tokens, int total_len_with_prefix, \
      void* seqlens_q_ptr, void* prefix_k_buffer, void* prefix_v_buffer, void* prefix_o_buffer,                     \
      void* prefix_kv_buffer, void* prefix_k_up_buffer, void* prefix_v_up_buffer, QuantMode mm_quant_mode)
MLA_ATTEN_VARLEN(float, float, llm_kernels::utils::KVCacheType::kAuto);
MLA_ATTEN_VARLEN(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_ATTEN_VARLEN(half, half, llm_kernels::utils::KVCacheType::kAuto);
MLA_ATTEN_VARLEN(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#ifdef ENABLE_BFLOAT16
MLA_ATTEN_VARLEN(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
MLA_ATTEN_VARLEN(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef MLA_ATTEN_VARLEN

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeMlaPagedAttention(
    void* output_ptr, void* hidden_buffer_0, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr, void* k_pe_ptr,
    void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* o_proj_weight, void* kv_b_nope_weight_scale,
    void* v_head_weight_scale, void* o_weight_scale, size_t o_proj_dim, void* workspace, cublasHandle_t& cublas_handles,
    cublasLtHandle_t& cublaslt_handles, void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr,
    int max_context_len, cudaStream_t stream, void* cache_offsets_ptr, int seqs_num, int num_heads,
    int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank, int v_head_dim, int head_size, int num_kv_heads,
    int stride_size, int block_size, float k_scale, float v_scale, int batch, void* rotary_embedding_pos,
    void* rotary_embedding_mask, int total_tokens, float attn_scale,
    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, void* workspace_ptr,
    float layernorm_eps, bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, size_t work_size, int rank,
    const std::optional<void*>& alibi_slopes, void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr,
    int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, QuantMode mm_quant_mode,
    int q_seq_len) {
  // 修改stride_size 和 head_size
  stride_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim);
  head_size = qk_nope_head_dim + qk_rope_head_dim;
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  // kv_b_proj升维矩阵乘
  // 1. kv_nope_proj
  if (kv_b_nope_weight_scale != nullptr) {
    if (workspace == nullptr) {
      KLLM_THROW("FP8 quantized matmul has not workspace");
    }
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      SCALAR_T* a = static_cast<SCALAR_T*>(compressed_kv_ptr);
      void* a_q = workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * kv_lora_rank);
      InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, kv_lora_rank, true, stream);

      float* b_scale = static_cast<float*>(kv_b_nope_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, kv_b_nope_proj_weight, b_scale, hidden_buffer_0, total_tokens, kv_lora_rank,
                                num_heads * qk_nope_head_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * qk_nope_head_dim, kv_lora_rank);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, workspace, stream, total_tokens, num_heads * qk_nope_head_dim, kv_lora_rank,
                        compressed_kv_ptr, kv_b_nope_proj_weight, hidden_buffer_0, GetMacheteDataType<SCALAR_T>(),
                        llm_kernels::nvidia::vllm_dtype::kU4B8, kv_b_nope_weight_scale,
                        std::optional<std::vector<size_t>>({static_cast<size_t>(kv_lora_rank / 128),
                                                            static_cast<size_t>(num_heads * qk_nope_head_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    } else {
      KLLM_THROW(fmt::format("MLA Decode not support quant mode: {}", mm_quant_mode));
    }
  } else {
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * qk_nope_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(kv_b_nope_proj_weight), hidden_buffer_0, stream, nullptr,
                           nullptr);
  }
  // 2. v_head_proj
  if (v_head_weight_scale != nullptr) {
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      SCALAR_T* a = static_cast<SCALAR_T*>(compressed_kv_ptr);
      void* a_q = workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * kv_lora_rank);
      // TODO(winminkong): 两个矩阵乘的输入一致，应该可以只求一次input_scale, 后期验证后改成只求一次
      InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, kv_lora_rank, true, stream);
      float* b_scale = static_cast<float*>(v_head_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, v_head_proj_weight, b_scale, output_ptr, total_tokens, kv_lora_rank,
                                num_heads * v_head_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * v_head_dim, kv_lora_rank);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, workspace, stream, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                        compressed_kv_ptr, v_head_proj_weight, output_ptr, GetMacheteDataType<SCALAR_T>(),
                        llm_kernels::nvidia::vllm_dtype::kU4B8, v_head_weight_scale,
                        std::optional<std::vector<size_t>>(
                            {static_cast<size_t>(kv_lora_rank / 128), static_cast<size_t>(num_heads * v_head_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    }
  } else {
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(v_head_proj_weight), output_ptr, stream, nullptr, nullptr);
  }

  if (rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(
        reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<int64_t*>(rotary_embedding_mask),
        reinterpret_cast<SCALAR_T*>(q_pe_ptr), reinterpret_cast<SCALAR_T*>(k_pe_ptr), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
  }
  /*
  Input Parameters:
    q_nope : q_nope_ptr       [total_tokens, num_heads, qk_nope_head_dim]
    k_nope : hidden_buffer_0  [total_tokens, num_heads, qk_nope_head_dim]
    q_pe   : q_pe_ptr         [total_tokens, num_heads, qk_rope_head_dim]
    k_pe   : k_pe_ptr         [total_tokens, 1, qk_rope_head_dim]
    v_pe   : output_ptr       [total_tokens, num_heads, v_head_dim]

  Intermediate Tensors:
    k_pe_expanded : k_pe_expanded_ptr  [total_tokens, num_heads, qk_rope_head_dim]
    v_pad         : v_pad_ptr          [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim - v_head_dim]
    q_tensor      : q_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
    k_tensor      : k_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
    v_tensor      : v_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]

  Memory Buffer Allocation:
    hidden_buffer_0 : [k_nope][q_tensor][k_tensor]
    out             : [v_pe][k_pe_expanded][v_pad][v_tensor]
  */
  size_t qk_combined_size = total_tokens * num_heads * (qk_nope_head_dim + qk_rope_head_dim) * sizeof(SCALAR_T);
  size_t v_tensor_size = total_tokens * num_heads * v_head_dim * sizeof(SCALAR_T);
  size_t k_pe_expanded_size = total_tokens * num_heads * qk_rope_head_dim * sizeof(SCALAR_T);
  size_t v_pad_size = total_tokens * num_heads * (qk_nope_head_dim + qk_rope_head_dim - v_head_dim) * sizeof(SCALAR_T);
  size_t k_nope_size = total_tokens * num_heads * qk_nope_head_dim * sizeof(SCALAR_T);

  size_t q_tensor_offset = (k_nope_size + 1023) & ~(1023);
  size_t k_tensor_offset = (q_tensor_offset + qk_combined_size + 1023) & ~(1023);
  size_t k_pe_expanded_offset = (v_tensor_size + 1023) & ~(1023);
  size_t v_pad_offset = (k_pe_expanded_offset + k_pe_expanded_size + 1023) & ~(1023);
  size_t v_tensor_offset = (v_pad_offset + v_pad_size + 1023) & ~(1023);

  void* q_tensor_ptr = static_cast<char*>(hidden_buffer_0) + q_tensor_offset;
  void* k_tensor_ptr = static_cast<char*>(hidden_buffer_0) + k_tensor_offset;
  void* k_pe_expanded_ptr = static_cast<char*>(output_ptr) + k_pe_expanded_offset;
  void* v_pad_ptr = static_cast<char*>(output_ptr) + v_pad_offset;
  void* v_tensor_ptr = static_cast<char*>(output_ptr) + v_tensor_offset;

  const size_t outer_dim_size = total_tokens * num_heads;
  const size_t inner_dim_size = 1;

  // cat(q_nope, q_pe)
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   q_tensor_ptr, stream);

  // cat(k_nope, k_pe)
  Expand<SCALAR_T>(k_pe_ptr, k_pe_expanded_ptr, total_tokens, num_heads, qk_rope_head_dim, 0, stream);
  Concat<SCALAR_T>(hidden_buffer_0, k_pe_expanded_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size,
                   inner_dim_size, k_tensor_ptr, stream);

  // pad v
  CUDA_CHECK(cudaMemsetAsync(v_pad_ptr, 0, v_pad_size, stream));
  Concat<SCALAR_T>(output_ptr, v_pad_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   v_tensor_ptr, stream);

#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(v_tensor_ptr), key_cache_ptrs,
      value_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr), block_size,
      batch, q_seq_len, num_kv_heads, head_size, stride_size, k_scale, v_scale, stream));
#else
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(v_tensor_ptr), key_cache_ptrs,
      value_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr), block_size,
      batch, total_tokens, num_kv_heads, head_size, stride_size, k_scale, v_scale, stream));
#endif
#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  auto cache_options = options;
  if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 || KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    // cache_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt8);
    KLLM_THROW("FlashAttention not support fp8 kv cache");
  }
  // kv_cache[num_blocks, 2, block_size, num_kv_heads, head_size]
  torch::Tensor k_cache_tensor =
      torch::from_blob(k_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
  torch::Tensor v_cache_tensor =
      torch::from_blob(v_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
  auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>());
  c10::optional<at::Tensor> block_table_tensor =
      torch::from_blob(block_table_ptr, {batch, max_blocks_per_seq}, int32_options);
  c10::optional<const at::Tensor> seqlens_k_tensor =
      c10::optional<const at::Tensor>(torch::from_blob(context_lens_ptr, {batch}, int32_options));
  torch::Tensor q_tensor =
      torch::from_blob(q_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);
  q_tensor = q_tensor.reshape({batch, 1, num_heads, head_size});
  c10::optional<at::Tensor> out_tensor = torch::from_blob(output_ptr, {batch, 1, num_heads, head_size}, options);
  float softmax_scale = attn_scale;
  c10::optional<at::Tensor> null_tensor = c10::nullopt;
  c10::optional<const at::Tensor> const_null_tensor = c10::nullopt;
  c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
  if (alibi_slopes.has_value()) {
    auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
    alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
  }

  mha_fwd_kvcache(q_tensor,             // batch_size x seqlen_q x num_heads x head_size
                  k_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                  v_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                  const_null_tensor,    // k_
                  const_null_tensor,    // v_
                  seqlens_k_tensor,     // batch_size
                  const_null_tensor,    // rotary_cos_: seqlen_ro x (rotary_dim / 2)
                  const_null_tensor,    // rotary_sin_: seqlen_ro x (rotary_dim / 2)
                  const_null_tensor,    // cache_batch_idx_: indices to index into the KV cache
                  block_table_tensor,   // batch_size x max_num_blocks_per_seq
                  alibi_slopes_tensor,  // num_heads or batch_size x num_heads
                  out_tensor,           // batch_size x seqlen_q x num_heads x head_size
                  softmax_scale, true, -1, -1, 0.0, true, 0);
#else
  const float* alibi_slopes_ptr =
      reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);
  // 可能会有softmax_scale的问题（也不调用）
  PagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(num_heads, head_size, num_kv_heads, stride_size, block_size, k_scale,
                                              v_scale, output_ptr, q_tensor_ptr, key_cache_ptrs, value_cache_ptrs,
                                              cache_offsets_ptr, context_lens_ptr, max_context_len, seqs_num, stream,
                                              workspace_ptr, work_size, alibi_slopes_ptr);
#endif
  //  当 v_tensor 被 pad 时调用, 取out_tensor 的 v_head_dim 大小
  size_t dst_pitch = v_head_dim * sizeof(SCALAR_T);
  size_t src_pitch = head_size * sizeof(SCALAR_T);
  CUDA_CHECK(cudaMemcpy2DAsync(hidden_buffer_0, dst_pitch, output_ptr, src_pitch, dst_pitch, total_tokens * num_heads,
                               cudaMemcpyDeviceToDevice, stream));
  // 3. o_proj
  if (o_weight_scale != nullptr) {
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      SCALAR_T* a = static_cast<SCALAR_T*>(hidden_buffer_0);
      void* a_q = workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * num_heads * v_head_dim);
      InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, num_heads * v_head_dim, true, stream);
      float* b_scale = static_cast<float*>(o_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, o_proj_weight, b_scale, output_ptr, total_tokens, num_heads * v_head_dim,
                                o_proj_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(o_proj_dim, num_heads * v_head_dim);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, workspace, stream, total_tokens, o_proj_dim, num_heads * v_head_dim,
                        hidden_buffer_0, o_proj_weight, output_ptr, GetMacheteDataType<SCALAR_T>(),
                        llm_kernels::nvidia::vllm_dtype::kU4B8, o_weight_scale,
                        std::optional<std::vector<size_t>>(
                            {static_cast<size_t>(num_heads * v_head_dim / 128), static_cast<size_t>(o_proj_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    } else {
      KLLM_THROW(fmt::format("MLA not support quant mode: {}", mm_quant_mode));
    }
  } else {
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, o_proj_dim, num_heads * v_head_dim,
                           reinterpret_cast<const void*>(hidden_buffer_0), reinterpret_cast<const void*>(o_proj_weight),
                           output_ptr, stream, nullptr, nullptr);
  }
}

#define RUN_MLA_PAGED_ATTENTION(SCALAR_T, CACHE_T, KV_DTYPE)                                                         \
  template void InvokeMlaPagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(                                                \
      void* output_ptr, void* hidden_buffer_0, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr,            \
      void* k_pe_ptr, void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* o_proj_weight,                    \
      void* kv_b_nope_weight_scale, void* v_head_weight_scale, void* o_weight_scale, size_t o_proj_dim,              \
      void* workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles, void** key_cache_ptrs,    \
      void** value_cache_ptrs, void* context_lens_ptr, int max_context_len, cudaStream_t stream,                     \
      void* cache_offsets_ptr, int seqs_num, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim,              \
      int kv_lora_rank, int v_head_dim, int head_size, int num_kv_heads, int stride_size, int block_size,            \
      float k_scale, float v_scale, int batch, void* rotary_embedding_pos, void* rotary_embedding_mask,              \
      int total_tokens, float attn_scale,                                                                            \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, void* workspace_ptr, \
      float layernorm_eps, bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, size_t work_size, int rank,   \
      const std::optional<void*>& alibi_slopes, void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr,           \
      int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, QuantMode mm_quant_mode,         \
      int q_seq_len)
RUN_MLA_PAGED_ATTENTION(float, float, llm_kernels::utils::KVCacheType::kAuto);
RUN_MLA_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_MLA_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_MLA_PAGED_ATTENTION(half, half, llm_kernels::utils::KVCacheType::kAuto);
RUN_MLA_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_MLA_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#ifdef ENABLE_BFLOAT16
RUN_MLA_PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
RUN_MLA_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_MLA_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef RUN_MLA_PAGED_ATTENTION

// Adapted from
// [DeepSeek-V3 Project]
// https://github.com/vllm-project/vllm/blob/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm/attention/backends/triton_mla.py#L698
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeAbsorbMlaPagedAttention(
    void* output_ptr, void* hidden_buffer_0, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr, void* k_pe_ptr,
    void* w_q_uk_weight, void* w_q_r_weight, void* w_uv_weight, void* w_o_weight, void* o_weight_scale,
    void* w_uv_o_weight, void* w_q_uk_weight_scale, void* w_q_r_weight_scale, void* w_uv_o_weight_scale,
    size_t w_uv_o_dim, void* workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles,
    void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr, cudaStream_t stream,
    void* cache_offsets_ptr, int seqs_num, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank,
    int v_head_dim, int head_size, int num_kv_heads, int stride_size, int block_size, float k_scale, float v_scale,
    int batch, void* rotary_embedding_pos, void* rotary_embedding_mask, int total_tokens, float attn_scale,
    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
    void* tile_scheduler_metadata_ptr, void* num_splits_ptr, void* workspace_ptr, float layernorm_eps, bool use_qk_norm,
    void* q_norm_weight, void* k_norm_weight, size_t work_size, int rank, const std::optional<void*>& alibi_slopes,
    void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num,
    int max_blocks_per_seq, int q_seq_len, int skiped_decode_q_token, int tail_offset_token, QuantMode mm_quant_mode) {
  // 修改stride_size 和 head_size
  stride_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim);
  head_size = qk_nope_head_dim + qk_rope_head_dim;
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());

  if (rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(
        reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<int64_t*>(rotary_embedding_mask),
        reinterpret_cast<SCALAR_T*>(q_pe_ptr), reinterpret_cast<SCALAR_T*>(k_pe_ptr), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
  }
  /*
  Input Parameters:
    v_tensor : compressed_kv_ptr  [total_tokens, kv_lora_rank]
    q_nope   : q_nope_ptr         [total_tokens, num_heads, kv_lora_rank]
    q_pe     : q_pe_ptr           [total_tokens, num_heads, qk_rope_head_dim]
    k_pe     : k_pe_ptr           [total_tokens, qk_rope_head_dim]

  Intermediate Tensors:
    q_tensor      : q_tensor_ptr       [total_tokens, num_heads, kv_lora_rank + qk_rope_head_dim]
    k_tensor      : k_tensor_ptr       [total_tokens, kv_lora_rank + qk_rope_head_dim]
    o_states      : hidden_buffer_0    [total_tokens, num_heads, kv_lora_rank]

  Memory Buffer Allocation(Take the maximum value):
    hidden_buffer_0 : [o_states][q_tensor][k_tensor]
  */
  const size_t origin_size =
      (tail_offset_token + total_tokens) * num_heads * (kv_lora_rank + qk_rope_head_dim) * sizeof(SCALAR_T);
  const size_t q_tensor_offset = (origin_size + 1023) & ~(1023);
  const size_t q_tensor_size = (total_tokens)*num_heads * (kv_lora_rank + qk_rope_head_dim) * sizeof(SCALAR_T);
  const size_t k_tensor_offset = (q_tensor_offset + q_tensor_size + 1023) & ~(1023);

  void* const q_tensor_ptr = hidden_buffer_0 + q_tensor_offset;
  void* const k_tensor_ptr = hidden_buffer_0 + k_tensor_offset;

  const size_t outer_q_dim_size = total_tokens * num_heads;
  const size_t outer_k_dim_size = total_tokens;
  const size_t inner_dim_size = 1;

  const AbsorbWeightsType absorb_type = GetAbsorbWeightsType();
  // cat(q_nope, q_pe)
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, kv_lora_rank, qk_rope_head_dim, outer_q_dim_size, inner_dim_size, q_tensor_ptr,
                   stream);

  // cat(v, k_pe)
  Concat<SCALAR_T>(compressed_kv_ptr, k_pe_ptr, kv_lora_rank, qk_rope_head_dim, outer_k_dim_size, inner_dim_size,
                   k_tensor_ptr, stream);

  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(k_tensor_ptr), key_cache_ptrs,
      key_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr), block_size,
      batch, q_seq_len, 1, kv_lora_rank + qk_rope_head_dim, kv_lora_rank + qk_rope_head_dim, k_scale, v_scale, stream));
  auto cache_options = options;
  if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 || KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    int32_t* dst_block_table_ptr = block_table_ptr + batch * max_blocks_per_seq;
    llm_kernels::nvidia::ConvertToScalar<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<CACHE_T*>(k_cache_ptr), reinterpret_cast<SCALAR_T*>(v_cache_ptr), block_table_ptr,
        dst_block_table_ptr, batch * max_blocks_per_seq, block_size * (kv_lora_rank + qk_rope_head_dim), k_scale,
        v_scale, stream);
    k_cache_ptr = v_cache_ptr;
    block_table_ptr = dst_block_table_ptr;
    KLLM_LOG_DEBUG << "ConvertToScalar num:" << batch * max_blocks_per_seq;
  }

  static const char* enable_flash_mla = std::getenv("ENABLE_FLASH_MLA");
  if (enable_flash_mla != nullptr && strcmp(enable_flash_mla, "1") == 0) {
    // Absorb has two versions
    if (absorb_type == AbsorbWeightsType::kAbsorbTypeBMM) {
      llm_kernels::nvidia::InvokeFlashMla<SCALAR_T>(
          static_cast<SCALAR_T*>(q_tensor_ptr), static_cast<SCALAR_T*>(k_cache_ptr), q_seq_len, attn_scale,
          block_table_ptr, context_lens_ptr, tile_scheduler_metadata_ptr, num_splits_ptr, qkv_workspace /*workspace*/,
          hidden_buffer_0, batch, num_heads, kv_lora_rank, qk_rope_head_dim, block_size, max_blocks_per_seq, rank,
          kv_cache_block_num, stream);
      // tp8: num_heads:16, total_tokens:256,kv_lora_rank:512,qk_rope_head_dim:64,w_uv_o_dim:7168,v_head_dim:128
      // [256, 16, 512] => [16, 256, 512]
      InvokePermute<SCALAR_T>(hidden_buffer_0, qkv_workspace, {total_tokens, num_heads, kv_lora_rank}, {1, 0, 2},
                              stream);
      // [16, 256, 512] * [16, 512, 128] => [16, 256, 128]
      InvokeBatchedMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, num_heads, total_tokens, v_head_dim, kv_lora_rank,
                                    qkv_workspace, w_uv_weight, output_ptr, stream, nullptr, 0, nullptr);
      // [16, 256, 128] => [256, 16, 128];
      InvokePermute<SCALAR_T>(output_ptr, hidden_buffer_0, {num_heads, total_tokens, v_head_dim}, {1, 0, 2}, stream);
    } else {
      llm_kernels::nvidia::InvokeFlashMla<SCALAR_T>(
          static_cast<SCALAR_T*>(q_tensor_ptr), static_cast<SCALAR_T*>(k_cache_ptr), q_seq_len, attn_scale,
          block_table_ptr, context_lens_ptr, tile_scheduler_metadata_ptr, num_splits_ptr, qkv_workspace /*workspace*/,
          hidden_buffer_0, batch, num_heads, kv_lora_rank, qk_rope_head_dim, block_size, max_blocks_per_seq, rank,
          kv_cache_block_num, stream);
    }

  } else {
    float softmax_scale = attn_scale;
    if (absorb_type == AbsorbWeightsType::kAbsorbTypeBMM) {
      Singleton<TritonWrapper>::GetInstance()->InvokeMlaAttenStage1<SCALAR_T>(
          q_tensor_ptr, k_cache_ptr, k_cache_ptr, softmax_scale, block_table_ptr, context_lens_ptr, output_ptr,
          total_tokens, num_heads, kv_lora_rank, qk_rope_head_dim, block_size, max_blocks_per_seq, stream);
      Singleton<TritonWrapper>::GetInstance()->InvokeMlaAttenStage2<SCALAR_T>(
          output_ptr, context_lens_ptr, qkv_workspace, total_tokens, num_heads, kv_lora_rank, stream);

      InvokePermute<SCALAR_T>(qkv_workspace, hidden_buffer_0, {total_tokens, num_heads, kv_lora_rank}, {1, 0, 2},
                              stream);
      InvokeBatchedMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, num_heads, total_tokens, v_head_dim, kv_lora_rank,
                                    hidden_buffer_0, w_uv_weight, output_ptr, stream, nullptr, 0, nullptr);
      InvokePermute<SCALAR_T>(output_ptr, hidden_buffer_0, {num_heads, total_tokens, v_head_dim}, {1, 0, 2}, stream);
    } else {
      Singleton<TritonWrapper>::GetInstance()->InvokeMlaAttenStage1<SCALAR_T>(
          q_tensor_ptr, k_cache_ptr, k_cache_ptr, softmax_scale, block_table_ptr, context_lens_ptr, output_ptr,
          total_tokens, num_heads, kv_lora_rank, qk_rope_head_dim, block_size, max_blocks_per_seq, stream);
      Singleton<TritonWrapper>::GetInstance()->InvokeMlaAttenStage2<SCALAR_T>(
          output_ptr, context_lens_ptr, hidden_buffer_0, total_tokens, num_heads, kv_lora_rank, stream);
    }
  }

  if (absorb_type == AbsorbWeightsType::kAbsorbTypeBMM) {
    KLLM_CHECK_WITH_INFO(w_o_weight != nullptr, "w_o_weight is nullptr");
    if (o_weight_scale != nullptr) {
      if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
        // [256, 16*128] * [16*128, 7168] => [256, 7168]
        SCALAR_T* a = static_cast<SCALAR_T*>(hidden_buffer_0);
        void* a_q = workspace;
        float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * num_heads * v_head_dim);
        InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, num_heads * v_head_dim, true, stream);
        float* b_scale = static_cast<float*>(o_weight_scale);
        InvokeBlockGemm<SCALAR_T>(a_q, a_s, w_o_weight, b_scale, output_ptr, total_tokens, num_heads * v_head_dim,
                                  w_uv_o_dim, stream);
      } else if (mm_quant_mode == QUANT_GPTQ) {
        int64_t workspace_size = 0;
        std::vector<std::string> machete_schedule_map =
            Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(w_uv_o_dim, num_heads * v_head_dim);
        std::optional<std::string> best_schedule = std::nullopt;
        if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
          best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
        }
        InvokeMacheteGemm(workspace_size, workspace, stream, total_tokens, w_uv_o_dim, num_heads * v_head_dim,
                          hidden_buffer_0, w_o_weight, output_ptr, GetMacheteDataType<SCALAR_T>(),
                          llm_kernels::nvidia::vllm_dtype::kU4B8, o_weight_scale,
                          std::optional<std::vector<size_t>>(
                              {static_cast<size_t>(num_heads * v_head_dim / 128), static_cast<size_t>(w_uv_o_dim)}),
                          GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
      } else {
        KLLM_THROW(fmt::format("MLA not support quant mode: {}", mm_quant_mode));
      }
    } else {
      InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, w_uv_o_dim, num_heads * v_head_dim,
                             reinterpret_cast<const void*>(hidden_buffer_0), reinterpret_cast<const void*>(w_o_weight),
                             output_ptr, stream, nullptr, nullptr);
    }
    return;
  }

  if (w_uv_o_weight_scale != nullptr) {
    SCALAR_T* a = static_cast<SCALAR_T*>(hidden_buffer_0);
    void* a_q = workspace;
    float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * num_heads * kv_lora_rank);
    InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, num_heads * kv_lora_rank, true, stream);
    float* b_scale = static_cast<float*>(w_uv_o_weight_scale);
    InvokeBlockGemm<SCALAR_T>(a_q, a_s, w_uv_o_weight, b_scale, output_ptr, total_tokens, num_heads * kv_lora_rank,
                              w_uv_o_dim, stream);
  } else {
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, w_uv_o_dim, num_heads * kv_lora_rank,
                           reinterpret_cast<const void*>(hidden_buffer_0), reinterpret_cast<const void*>(w_uv_o_weight),
                           output_ptr, stream, nullptr, nullptr);
  }
}

#define RUN_ABSORB_MLA_PAGED_ATTENTION(SCALAR_T, CACHE_T, KV_DTYPE)                                               \
  template void InvokeAbsorbMlaPagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(                                       \
      void* output_ptr, void* hidden_buffer_0, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr,         \
      void* k_pe_ptr, void* w_q_uk_weight, void* w_q_r_weight, void* w_uv_weight, void* w_o_weight,               \
      void* o_weight_scale, void* w_uv_o_weight, void* w_q_uk_weight_scale, void* w_q_r_weight_scale,             \
      void* w_uv_o_weight_scale, size_t w_uv_o_dim, void* workspace, cublasHandle_t& cublas_handles,              \
      cublasLtHandle_t& cublaslt_handles, void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr, \
      cudaStream_t stream, void* cache_offsets_ptr, int seqs_num, int num_heads, int qk_rope_head_dim,            \
      int qk_nope_head_dim, int kv_lora_rank, int v_head_dim, int head_size, int num_kv_heads, int stride_size,   \
      int block_size, float k_scale, float v_scale, int batch, void* rotary_embedding_pos,                        \
      void* rotary_embedding_mask, int total_tokens, float attn_scale,                                            \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,                   \
      void* tile_scheduler_metadata_ptr, void* num_splits_ptr, void* workspace_ptr, float layernorm_eps,          \
      bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, size_t work_size, int rank,                     \
      const std::optional<void*>& alibi_slopes, void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr,        \
      int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, int q_seq_len,                \
      int skiped_decode_q_token, int total_q_len, QuantMode mm_quant_mode)
RUN_ABSORB_MLA_PAGED_ATTENTION(float, float, llm_kernels::utils::KVCacheType::kAuto);
RUN_ABSORB_MLA_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_ABSORB_MLA_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_ABSORB_MLA_PAGED_ATTENTION(half, half, llm_kernels::utils::KVCacheType::kAuto);
RUN_ABSORB_MLA_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_ABSORB_MLA_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#ifdef ENABLE_BFLOAT16
RUN_ABSORB_MLA_PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
RUN_ABSORB_MLA_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_ABSORB_MLA_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef RUN_ABSORB_MLA_PAGED_ATTENTION

// Adapted from
// [DeepSeek-V3 Project]
// https://github.com/vllm-project/vllm/blob/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm/attention/backends/mla/utils.py#L345
template <typename T>
void MlaAbsorbWeight(void* w_q, void* w_uk, void* w_uv, void* w_o, void* w_q_uk, void* w_uv_o, size_t q, size_t n,
                     size_t d, size_t l, size_t h, bool transpose_matrix, int rank, cudaStream_t& stream) {
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  cudaStream_t torch_stream = InvokeSetTorchStream(stream, rank);
#endif
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>());

  if (transpose_matrix) {
    // For transposed matrices
    torch::Tensor w_q_tensor =
        torch::from_blob(w_q, {static_cast<int64_t>(n), static_cast<int64_t>(d), static_cast<int64_t>(q)}, options);
    torch::Tensor w_uk_tensor =
        torch::from_blob(w_uk, {static_cast<int64_t>(n), static_cast<int64_t>(d), static_cast<int64_t>(l)}, options);
    torch::Tensor w_uv_tensor =
        torch::from_blob(w_uv, {static_cast<int64_t>(n), static_cast<int64_t>(d), static_cast<int64_t>(l)}, options);
    torch::Tensor w_o_tensor =
        torch::from_blob(w_o, {static_cast<int64_t>(h), static_cast<int64_t>(n), static_cast<int64_t>(d)}, options);

    // Perform einsum with reshaped tensors
    auto w_q_uk_tmp = torch::einsum("ndq,ndl -> nlq", {w_q_tensor, w_uk_tensor}).contiguous();
    auto w_uv_o_tmp = torch::einsum("ndl,hnd -> hnl", {w_uv_tensor, w_o_tensor}).contiguous();
    CUDA_CHECK(cudaMemcpyAsync(w_q_uk, w_q_uk_tmp.data_ptr(), q * n * l * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(w_uv_o, w_uv_o_tmp.data_ptr(), n * l * h * sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    // Original logic for non-transposed matrices
    torch::Tensor w_q_tensor =
        torch::from_blob(w_q, {static_cast<int64_t>(q), static_cast<int64_t>(n), static_cast<int64_t>(d)}, options);
    torch::Tensor w_uk_tensor =
        torch::from_blob(w_uk, {static_cast<int64_t>(l), static_cast<int64_t>(n), static_cast<int64_t>(d)}, options);
    torch::Tensor w_uv_tensor =
        torch::from_blob(w_uv, {static_cast<int64_t>(l), static_cast<int64_t>(n), static_cast<int64_t>(d)}, options);
    torch::Tensor w_o_tensor =
        torch::from_blob(w_o, {static_cast<int64_t>(n), static_cast<int64_t>(d), static_cast<int64_t>(h)}, options);
    auto w_q_uk_tmp = torch::einsum("qnd,lnd -> qnl", {w_q_tensor, w_uk_tensor}).contiguous();
    CUDA_CHECK(cudaMemcpyAsync(w_q_uk, w_q_uk_tmp.data_ptr(), q * n * l * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    auto w_uv_o_tmp = torch::einsum("lnd,ndh -> nlh", {w_uv_tensor, w_o_tensor}).contiguous();
    CUDA_CHECK(cudaMemcpyAsync(w_uv_o, w_uv_o_tmp.data_ptr(), n * l * h * sizeof(T), cudaMemcpyDeviceToDevice, stream));
  }
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  InvokeSetTorchStream(torch_stream, rank);
#endif
}

#define MLA_ABSORB_WEIGHT(T)                                                                                           \
  template void MlaAbsorbWeight<T>(void* w_q, void* w_uk, void* w_uv, void* w_o, void* w_q_uk, void* w_uv_o, size_t q, \
                                   size_t n, size_t d, size_t l, size_t h, bool transpose_matrix, int rank,            \
                                   cudaStream_t& stream);
MLA_ABSORB_WEIGHT(float);
MLA_ABSORB_WEIGHT(half);
#ifdef ENABLE_BFLOAT16
MLA_ABSORB_WEIGHT(__nv_bfloat16);
#endif
#undef MLA_ABSORB_WEIGHT

// Adapted from
// [vLLM Project]
// https://github.com/Chen-XiaoBing/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L923
template <typename T>
void InvokeGroupedTopk(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows, int num_experts,
                       int topk, bool renormalize, int num_expert_group, int topk_group, std::string scoring_func,
                       void* e_bias, float routed_scaling_factor, int rank, cudaStream_t stream) {
  bool is_enable_fused_grouped_moe =
      (scoring_func == "sigmoid" && e_bias != nullptr && renormalize && topk < num_experts / num_expert_group);
  if (is_enable_fused_grouped_moe) {
    llm_kernels::nvidia::InvokeDeepSeekV3GroupedTopk<T>(gating_output, e_bias, routed_scaling_factor, topk_weights_ptr,
                                                        topk_ids_ptr, num_rows, num_experts, topk, num_expert_group,
                                                        topk_group, stream);
    return;
  }

  auto origin_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>());
  torch::Tensor gating_tensor = torch::from_blob(
      gating_output, {static_cast<int64_t>(num_rows), static_cast<int64_t>(num_experts)}, origin_options);
  torch::Tensor scores;
  if (scoring_func == "softmax") {
    scores = torch::softmax(gating_tensor, -1);
  } else if (scoring_func == "sigmoid") {
    scores = gating_tensor.sigmoid();
  } else {
    KLLM_LOG_ERROR << fmt::format("Unsupported scoring function: {}", scoring_func);
  }

  torch::Tensor original_scores;
  torch::Tensor group_scores;
  int num_token = scores.size(0);
  if (e_bias != nullptr) {
    original_scores = scores.clone();
    auto fp32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
    torch::Tensor e_score_correction_bias_tensor = torch::from_blob(e_bias, {num_experts}, fp32_options);
    scores = scores + e_score_correction_bias_tensor;
    group_scores = std::get<0>(scores.view({num_token, num_expert_group, -1}).topk(2, -1)).sum(-1);
  } else {
    group_scores = std::get<0>(scores.view({num_token, num_expert_group, -1}).max(-1));
  }
  // KLLM_LOG_DEBUG << "group_scores " << group_scores;
  torch::Tensor group_idx = std::get<1>(torch::topk(group_scores, topk_group, -1, true));  // [n, top_k_group]
  torch::Tensor group_mask = torch::zeros_like(group_scores);                              // [n, n_group]
  group_mask.scatter_(1, group_idx, 1);                                                    // [n, n_group]
  torch::Tensor score_mask = group_mask.unsqueeze(-1)
                                 .expand({num_token, num_expert_group, scores.size(-1) / num_expert_group})
                                 .reshape({num_token, -1});  // [n, e]
  auto tmp_scores = scores.masked_fill(~score_mask.to(torch::kBool),
                                       -std::numeric_limits<float>::infinity());  // [n, e]

  torch::Tensor topk_weights, topk_ids;
  if (e_bias != nullptr) {
    topk_ids = std::get<1>(torch::topk(tmp_scores, topk, -1, true));
    // Use original unbiased scores for the routing weights
    topk_weights = original_scores.gather(1, topk_ids);
  } else {
    std::tie(topk_weights, topk_ids) = torch::topk(tmp_scores, topk, -1, true);
  }

  if (renormalize) {
    topk_weights = topk_weights / topk_weights.sum(-1, true);
  }
  // 这里做了提前乘routed_scaling_factor，
  // 如果要与vllm对比moe内部计算结果，
  // 需要将routed_scaling_factor设置为1.0。
  // routed_scaling_factor = 1.0;
  topk_weights = topk_weights * routed_scaling_factor;
  torch::Tensor output_topk_weights = topk_weights.cuda().to(torch::kFloat32).clone();
  torch::Tensor output_topk_ids = topk_ids.cuda().to(torch::kInt32).clone();

  CUDA_CHECK(cudaMemcpyAsync(topk_weights_ptr, output_topk_weights.data_ptr(), topk_weights.numel() * sizeof(float),
                             cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(topk_ids_ptr, output_topk_ids.data_ptr(), topk_ids.numel() * sizeof(int32_t),
                             cudaMemcpyDeviceToDevice, stream));
}
#define INVOKE_GROUPED_TOPK(T)                                                                                      \
  template void InvokeGroupedTopk<T>(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows, \
                                     int num_experts, int topk, bool renormalize, int num_expert_group,             \
                                     int topk_group, std::string scoring_func, void* e_bias,                        \
                                     float routed_scaling_factor, int rank, cudaStream_t stream)
INVOKE_GROUPED_TOPK(float);
INVOKE_GROUPED_TOPK(half);
#ifdef ENABLE_BFLOAT16
INVOKE_GROUPED_TOPK(__nv_bfloat16);
#endif
#undef INVOKE_GROUPED_TOPK

template <typename T>
void InvokeBlockGemm(void* a, float* a_scales, void* b, float* b_scales, void* output, int m, int k, int n,
                     cudaStream_t& stream, void* cutlass_buf, size_t cutlass_buf_size) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BlockwiseGemmKernel<T>(a, a_scales, b, b_scales, output, m, k, n, stream,
                                                                    cutlass_buf, cutlass_buf_size));
}

#define BLOCKWISE_GEMM(T)                                                                                          \
  template void InvokeBlockGemm<T>(void* a, float* a_scales, void* b, float* b_scales, void* output, int m, int k, \
                                   int n, cudaStream_t& stream, void* cutlass_buf, size_t cutlass_buf_size)
BLOCKWISE_GEMM(float);
BLOCKWISE_GEMM(half);
#ifdef ENABLE_BFLOAT16
BLOCKWISE_GEMM(__nv_bfloat16);
#endif
#undef BLOCKWISE_GEMM

// Adapted from
// [vLLM Project]
// https://github.com/Chen-XiaoBing/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L624
template <typename T>
void InvokeFusedMoeKernelFunc(void* a, void* b, void* c, void* a_q, void* a_scale, void* b_scale, void* b_zp,
                              void* topk_weights, void* topk_ids, void* sorted_token_ids, void* expert_ids,
                              void* num_tokens_post_padded, bool mul_routed_weight, int topk, int chunk_num_tokens,
                              int numel, int m, int k, int n, int max_num_tokens_padded,
                              const std::unordered_map<std::string, int>& config, DataType weight_dtype,
                              DataType compute_dtype, bool use_moe_wna16_cuda, void* dequant_workspace,
                              std::vector<int>& block_shape, int num_experts, const cudaStream_t& stream) {
  if (chunk_num_tokens < config.at("block_size_m")) {
    max_num_tokens_padded = std::min(max_num_tokens_padded, chunk_num_tokens * topk * config.at("block_size_m"));
  }

  if (weight_dtype == DataType::TYPE_I4_GROUP && block_shape.size() > 1 && block_shape[1] > 0) {
    int group_size = block_shape[1];
    bool has_zp = !(b_zp == nullptr);
    int weight_bits = 4;

    std::unordered_map<std::string, int> moe_wna16_config = config;  // shallow copy
    UpdateMoeWna16BlockConfig(moe_wna16_config, use_moe_wna16_cuda,
                              compute_dtype == DataType::TYPE_BLOCK_FP8_E4M3 && !has_zp, numel, k, n, n, group_size,
                              topk, moe_wna16_config.at("block_size_m"));

    if (use_moe_wna16_cuda) {
      void* topk_weights_input = mul_routed_weight ? topk_weights : nullptr;
      int num_token_blocks =
          (max_num_tokens_padded + moe_wna16_config.at("block_size_m") - 1) / moe_wna16_config.at("block_size_m");
      InvokeMoeWna16Gemm<T>(stream, c, a, b, b_scale, b_zp, topk_weights_input, sorted_token_ids, expert_ids,
                            num_tokens_post_padded, topk, moe_wna16_config.at("block_size_m"),
                            moe_wna16_config.at("block_size_n"), moe_wna16_config.at("block_size_k"), weight_bits,
                            num_experts, m, n, k, group_size, num_token_blocks);
    } else if (compute_dtype == DataType::TYPE_BLOCK_FP8_E4M3 && !has_zp) {  // TODO(jinxcwu) 目前只支持无zp的gptq
      // https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L688
      // TODO(zakwang) :这里有个假定,即 block_shape 不为 None
      InvokePerTokenGroupQuantFp8E4m3<T>(a, a_q, a_scale, chunk_num_tokens, k, false, stream);
      DequantInt4Fp8(stream, dequant_workspace, b, (size_t)num_experts * n * k / 2);
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeGptqInt4Fp8Kernel<T>(
          a_q, dequant_workspace, c, a_scale, b_scale, topk_weights, sorted_token_ids, expert_ids,
          num_tokens_post_padded, n, k, max_num_tokens_padded, numel, mul_routed_weight, topk, group_size,
          moe_wna16_config, stream);
    } else {
      int pack_factor = 8 / weight_bits;
      Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeGptqAwqKernel<T>(
          a, b, c, b_scale, b_zp, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, n, k,
          max_num_tokens_padded, numel, k, 1, n * k / pack_factor, 1, k / pack_factor, n, 1, n * k / group_size, 1,
          k / group_size, n / pack_factor * k / group_size, 1, k / group_size, mul_routed_weight, topk, has_zp,
          weight_bits, group_size, moe_wna16_config, stream);
    }
  } else {
    if (compute_dtype == DataType::TYPE_BLOCK_FP8_E4M3) {
      // https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L688
      // TODO(zakwang) :这里有个假定,即 block_shape 不为 None
      InvokePerTokenGroupQuantFp8E4m3<T>(a, a_q, a_scale, chunk_num_tokens, k, false, stream);
      a = a_q;
    }
    // A [m, k]
    // B [exprts_num, n, k]
    // C [m, topk, n]
    // A_scale [m, k / 128]
    // B_scale [experts_num, n / 128, k / 128]
    Singleton<TritonWrapper>::GetInstance()->InvokeFusedMoeKernel<T>(
        a, b, c, a_scale, b_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, n, k,
        max_num_tokens_padded, numel, k, 1, n * k, 1, k, n, 1, k / 128, 1, n / 128 * k / 128, 1, k / 128,
        block_shape[0], block_shape[1], mul_routed_weight, topk, compute_dtype == DataType::TYPE_BLOCK_FP8_E4M3, false,
        config, stream);
  }
}

// Adapted from
// [vLLM Project]
// https://github.com/Chen-XiaoBing/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L1271
template <typename T, bool UseExpertParallel>
void InvokeFusedMoe(void* hidden_states, void* w1, void* w2, void* gating_output, int* expert_map, int topk,
                    bool renormalize, const std::string& scoring_func_, void* e_bias, bool inplace,
                    bool use_grouped_topk, int num_expert_group, int topk_group, DataType weight_dtype,
                    DataType compute_dtype, bool is_marlin, bool use_triton, void* w1_scale, void* w2_scale,
                    void* w1_zp, void* w2_zp, void* a1_q, void* a2_q, void* a1_scale, void* a2_scale,
                    std::vector<int> block_shape, void* topk_weights_ptr, void* topk_ids_ptr,
                    float routed_scaling_factor, void* output_hidden_states, void* intermediate_cache1,
                    void* intermediate_cache2, void* intermediate_cache3, void* fused_id_buffer, int num_tokens,
                    int num_experts, int hidden_size, int inter_size, void* dequant_workspace, int rank,
                    cudaStream_t stream) {
  // hidden_states [num_tokens, hidden_size] dtype = T
  // gating_output [num_tokens, num_experts]
  // w1 [num_experts, inter_size * 2, hidden_size]
  // w2 [num_experts, hidden_size, inter_size]
  // topk_ids [num_tokens, topk]
  // topk_weights [num_tokens, topk]
  // M = min(num_tokens, CHUNK_SIZE)
  // N = inter_size * 2
  // intermediate_cache1 [M, topk, N]
  // intermediate_cache2 [M * topk, N // 2]
  // intermediate_cache3 [M, topk, hidden_size]
  // expert_map [num_experts]

  // a1_q, a2_q: FP8  需要将 FP16 输入量化成 FP8 + FLOAT32,这两个是两个对应空间
  // TODO(zezhao): 使用 num_experts / expert_para_size 来替换 total_num_experts. 不再维护 ExpertParallelSize
  size_t expert_para_size = Singleton<Environment>::GetInstance()->GetExpertParallelSize();
  size_t expert_world_size = Singleton<Environment>::GetInstance()->GetExpertWorldSize();
  int total_num_experts = num_experts * expert_para_size * expert_world_size;
  if (use_grouped_topk) {
    InvokeGroupedTopk<T>(gating_output, topk_weights_ptr, topk_ids_ptr, num_tokens, total_num_experts, topk,
                         renormalize, num_expert_group, topk_group, scoring_func_, e_bias, routed_scaling_factor, rank,
                         stream);
  } else {
    // 需要对非group的做优化，目前直接复用
    InvokeGroupedTopk<T>(gating_output, topk_weights_ptr, topk_ids_ptr, num_tokens, total_num_experts, topk,
                         renormalize, num_expert_group, topk_group, scoring_func_, e_bias, routed_scaling_factor, rank,
                         stream);
  }

  // Expert parallel.
  // hidden_state[num_tokens][hidden_dim]
  // topk_ids_ptr[num_tokens][topk]
  // topk_weights_ptr[num_tokens][topk]
  // token_node_map[node_ids][tokens_idx]

  // fused_experts_impl
  const int chunk_size = 32 * 1024;
  int M = std::min(num_tokens, chunk_size);
  int chunk_times = (num_tokens + chunk_size - 1) / chunk_size;

  std::unordered_map<std::string, int> config;
  bool use_moe_wna16_cuda = false;
  block_shape.resize(2);
  if (weight_dtype == DataType::TYPE_BLOCK_FP8_E4M3) {
    if (block_shape.size() < 2) {
      KLLM_LOG_ERROR << fmt::format("Config fp8_w8a8 need block_shape.shape = [2,]");
      return;
    }
    // TODO(rockcao): 256是参考sglang的阈值，后续需要具备搜索最优config的能力
    const int config_selection_threshold = 256;
    if (num_tokens > config_selection_threshold) {
      config = {{"block_size_m", 64},
                {"block_size_n", block_shape[0]},
                {"block_size_k", block_shape[1]},
                {"group_size_m", 1},
                {"num_warps", 4},
                {"num_stages", 3}};
    } else {
      config = {{"block_size_m", 16},
                {"block_size_n", block_shape[0]},
                {"block_size_k", block_shape[1]},
                {"group_size_m", 32},
                {"num_warps", 4},
                {"num_stages", 3}};
    }
  } else if (weight_dtype == DataType::TYPE_I4_GROUP && (!block_shape.empty())) {
    use_moe_wna16_cuda = ShouldMoeWna16UseCuda(M * topk, block_shape[1], num_experts, 4);
    if (use_moe_wna16_cuda) {
      config = {{"block_size_m", std::min(16, M)}};
    } else if (M <= 20) {
      config = {{"block_size_m", 16}, {"group_size_m", 1}};
    } else if (M <= 40) {
      config = {{"block_size_m", 32}, {"group_size_m", 1}};
    } else {
      config = {{"block_size_m", 64}, {"group_size_m", 1}};
    }
  } else {
    config = {{"block_size_m", 64}, {"block_size_n", 64}, {"block_size_k", 32}, {"group_size_m", 8}};
    if (M <= num_experts || (M <= 32 && is_marlin)) {
      config = {{"block_size_m", 16}, {"block_size_n", 32}, {"block_size_k", 64}, {"group_size_m", 1}};
    }
  }
  auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>());
  // [start_offset0, length0, value0, ..., start_offsetN, lengthN, valueN]
  std::vector<int> fill_info;
  // reserve for fill_info
  const size_t fill_info_reserved_size = 24;
  fill_info.reserve(fill_info_reserved_size);
  for (int i = 0; i < chunk_times; ++i) {
    // curr_hidden_states [tokens_in_chunk, hidden_size]
    // curr_interm_cache1 [tokens_in_chunk, topk, inter_size * 2]
    // curr_interm_cache2 [tokens_in_chunk * topk, inter_size]
    // curr_interm_cache3 [tokens_in_chunk, topk, hidden_size]
    // curr_topk_ids [tokens_in_chunk, topk]
    // curr_topk_weights [tokens_in_chunk, topk]
    void* curr_hidden_states = hidden_states + sizeof(T) * chunk_size * i * hidden_size;
    int tokens_in_chunk = std::min(chunk_size, num_tokens - chunk_size * i);
    void* curr_topk_ids = topk_ids_ptr + sizeof(int32_t) * chunk_size * i * topk;
    void* curr_topk_weights = topk_weights_ptr + sizeof(float) * chunk_size * i * topk;
    void* curr_intermediate_cache1 = intermediate_cache1;
    void* curr_intermediate_cache2 = intermediate_cache2;
    void* curr_intermediate_cache3 = intermediate_cache3;

    void* fill_info_buffer_on_device = fused_id_buffer;
    void* curr_fused_id_buffer =
        fill_info_buffer_on_device + fill_info_reserved_size * sizeof(decltype(fill_info)::value_type);
    int id_buffer_offset = 0;

    // moe_align_block_size
    int block_size = config["block_size_m"];
    int numel = tokens_in_chunk * topk;
    int max_num_tokens_padded = numel + num_experts * (block_size - 1);
    // torch::Tensor sorted_ids = torch::empty({max_num_tokens_padded}, int32_options);
    // sorted_ids.fill_(numel);
    fill_info.insert(fill_info.end(), {id_buffer_offset, max_num_tokens_padded, numel});
    void* sorted_ids_ptr = curr_fused_id_buffer + sizeof(int32_t) * id_buffer_offset;
    id_buffer_offset += max_num_tokens_padded;

    int max_num_m_blocks = (max_num_tokens_padded + block_size - 1) / block_size;
    // torch::Tensor expert_ids = torch::empty({max_num_m_blocks}, int32_options);
    // expert_ids.fill_(-1);
    fill_info.insert(fill_info.end(), {id_buffer_offset, max_num_m_blocks, -1});
    void* expert_ids_ptr = curr_fused_id_buffer + sizeof(int32_t) * id_buffer_offset;
    id_buffer_offset += max_num_m_blocks;

    // torch::Tensor num_tokens_post_pad = torch::empty({1}, int32_options);
    fill_info.insert(fill_info.end(), {id_buffer_offset, 1, 0});
    void* num_tokens_post_pad_ptr = curr_fused_id_buffer + sizeof(int32_t) * id_buffer_offset;
    id_buffer_offset += 1;
    // TODO(zezhao): SglMoe need support Expert-Parallel
    if (num_experts >= 224) {
      // torch::Tensor cumsum = torch::zeros({num_experts + 1}, int32_options);
      fill_info.insert(fill_info.end(), {id_buffer_offset, num_experts + 1, 0});
      void* cumsum_ptr = curr_fused_id_buffer + sizeof(int32_t) * id_buffer_offset;
      id_buffer_offset += num_experts + 1;
      llm_kernels::nvidia::InvokeFillIntToBuffer(reinterpret_cast<int32_t*>(curr_fused_id_buffer),
                                                 fill_info_buffer_on_device, reinterpret_cast<int*>(fill_info.data()),
                                                 fill_info.size(), stream);
      llm_kernels::nvidia::InvokeSglMoeAlignBlockSize<int32_t>(
          reinterpret_cast<int32_t*>(curr_topk_ids), reinterpret_cast<int32_t*>(sorted_ids_ptr),
          reinterpret_cast<int32_t*>(expert_ids_ptr), reinterpret_cast<int32_t*>(num_tokens_post_pad_ptr), num_experts,
          block_size, numel, reinterpret_cast<int32_t*>(cumsum_ptr), stream);
    } else {
      // https://github.com/vllm-project/vllm/blob/185cc19f922a29868bf62e4f2674c763df36c8b5/csrc/moe/moe_align_sum_kernels.cu#L296
      llm_kernels::nvidia::InvokeFillIntToBuffer(reinterpret_cast<int32_t*>(curr_fused_id_buffer),
                                                 fill_info_buffer_on_device, reinterpret_cast<int*>(fill_info.data()),
                                                 fill_info.size(), stream);
      llm_kernels::nvidia::InvokeMoeAlignBlockSize<int32_t, uint16_t, UseExpertParallel>(
          reinterpret_cast<int32_t*>(curr_topk_ids), reinterpret_cast<int32_t*>(sorted_ids_ptr),
          reinterpret_cast<int32_t*>(expert_ids_ptr), reinterpret_cast<int32_t*>(num_tokens_post_pad_ptr), expert_map,
          topk, num_experts, expert_para_size, block_size, numel, rank, stream);
    }

    // invoke_fused_moe_kernel
    InvokeFusedMoeKernelFunc<T>(
        curr_hidden_states, w1, curr_intermediate_cache1, a1_q, a1_scale, w1_scale, w1_zp, curr_topk_weights,
        curr_topk_ids, reinterpret_cast<int32_t*>(sorted_ids_ptr), reinterpret_cast<int32_t*>(expert_ids_ptr),
        reinterpret_cast<int32_t*>(num_tokens_post_pad_ptr), false, topk, tokens_in_chunk, numel, tokens_in_chunk,
        hidden_size, inter_size * 2, max_num_tokens_padded, config, weight_dtype, compute_dtype, use_moe_wna16_cuda,
        dequant_workspace, block_shape, num_experts, stream);
    size_t elements_num = static_cast<size_t>(tokens_in_chunk) * topk * inter_size * 2;
    llm_kernels::nvidia::InvokeSiluAndMul<T, UseExpertParallel>(
        reinterpret_cast<const T*>(curr_intermediate_cache1), reinterpret_cast<T*>(curr_intermediate_cache2),
        reinterpret_cast<const int*>(curr_topk_ids), reinterpret_cast<const int*>(expert_map), num_experts,
        elements_num, inter_size, stream);

    InvokeFusedMoeKernelFunc<T>(
        curr_intermediate_cache2, w2, curr_intermediate_cache3, a2_q, a2_scale, w2_scale, w2_zp, curr_topk_weights,
        curr_topk_ids, reinterpret_cast<int32_t*>(sorted_ids_ptr), reinterpret_cast<int32_t*>(expert_ids_ptr),
        reinterpret_cast<int32_t*>(num_tokens_post_pad_ptr), true, 1, tokens_in_chunk * topk, numel,
        tokens_in_chunk * topk, inter_size, hidden_size, max_num_tokens_padded, config, weight_dtype, compute_dtype,
        use_moe_wna16_cuda, dequant_workspace, block_shape, num_experts, stream);

    void* curr_out_hidden_states = output_hidden_states + sizeof(T) * chunk_size * hidden_size * i;
    llm_kernels::nvidia::InvokeMoeSum<T, UseExpertParallel>(intermediate_cache3, curr_out_hidden_states, curr_topk_ids,
                                                            expert_map, tokens_in_chunk, num_experts, topk, hidden_size,
                                                            stream);
    fill_info.clear();
  }
}

#define FUSEDMOE(T)                                                                                                   \
  template void InvokeFusedMoe<T, true>(                                                                              \
      void* hidden_states, void* w1, void* w2, void* gating_output, int* expert_map, int topk, bool renormalize,      \
      const std::string& scoring_func_, void* e_bias, bool inplace, bool use_grouped_topk, int num_expert_group,      \
      int topk_group, DataType weight_dtype, DataType compute_dtype, bool is_marlin, bool use_triton, void* w1_scale, \
      void* w2_scale, void* w1_zp, void* w2_zp, void* a1_q, void* a2_q, void* a1_scale, void* a2_scale,               \
      std::vector<int> block_shape, void* topk_weights_ptr, void* topk_ids_ptr, float routed_scaling_factor,          \
      void* output_hidden_states, void* intermediate_cache1, void* intermediate_cache2, void* intermediate_cache3,    \
      void* fused_id_buffer, int num_tokens, int num_experts, int hidden_size, int inter_size,                        \
      void* dequant_workspace, int rank, cudaStream_t stream);                                                        \
  template void InvokeFusedMoe<T, false>(                                                                             \
      void* hidden_states, void* w1, void* w2, void* gating_output, int* expert_map, int topk, bool renormalize,      \
      const std::string& scoring_func_, void* e_bias, bool inplace, bool use_grouped_topk, int num_expert_group,      \
      int topk_group, DataType weight_dtype, DataType compute_dtype, bool is_marlin, bool use_triton, void* w1_scale, \
      void* w2_scale, void* w1_zp, void* w2_zp, void* a1_q, void* a2_q, void* a1_scale, void* a2_scale,               \
      std::vector<int> block_shape, void* topk_weights_ptr, void* topk_ids_ptr, float routed_scaling_factor,          \
      void* output_hidden_states, void* intermediate_cache1, void* intermediate_cache2, void* intermediate_cache3,    \
      void* fused_id_buffer, int num_tokens, int num_experts, int hidden_size, int inter_size,                        \
      void* dequant_workspace, int rank, cudaStream_t stream)
FUSEDMOE(float);
FUSEDMOE(half);
#ifdef ENABLE_BFLOAT16
FUSEDMOE(__nv_bfloat16);
#endif
#undef FUSEDMOE

// Adapted from
// [vLLM Project]
// https://github.com/vllm-project/vllm/blob/v0.7.1/vllm/model_executor/layers/quantization/utils/quant_utils.py#L63
template <typename T>
void ScaledQuantize(void* x, void* output, float* scale, std::vector<int> group_shape, int m, int n, int rank) {
  int block_m = m / group_shape[0];
  int block_n = n / group_shape[1];
  auto origin_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>());
  torch::Tensor x_tensor = torch::from_blob(x, {m, n}, origin_options);

  // Reshape and permute
  x_tensor = x_tensor.reshape({block_m, group_shape[0], block_n, group_shape[1]});
  x_tensor = x_tensor.permute({0, 2, 1, 3}).flatten(2);

  // Compute scales
  auto [min_val, max_val] = x_tensor.aminmax(-1);
  auto amax = torch::max(min_val.abs(), max_val.abs()).clamp_min(1e-12);
  auto finfo = std::numeric_limits<T>();
  *scale = finfo.max() / amax.item<double>();

  // Apply scale and clamp
  torch::Tensor x_scl_sat = (x_tensor * (*scale)).clamp(finfo.min(), finfo.max());
  x_scl_sat = x_scl_sat.reshape({block_m, block_n, group_shape[0], group_shape[1]});
  x_scl_sat = x_scl_sat.permute({0, 2, 1, 3}).reshape({m, n});

  // Copy the result to output
  torch::Tensor output_tensor = torch::from_blob(output, {m, n}, origin_options);
  output_tensor.copy_(x_scl_sat.to(GetTorchDataType<T>()).contiguous());
}

#ifdef ENABLE_FP8
// Adapted from
// [vLLM Project]
// https://github.com/vllm-project/vllm/blob/v0.7.1/vllm/model_executor/layers/quantization/utils/quant_utils.py#L63
// Quantize assuming once scale per group of elements with shape group_shape,
// example group shapes:
// * (-1, -1)   for per-tensor quantization
// * (1, -1)    for per-row quantization
// * (-1, 1)    for per-column quantization
// * (128, 128) for 128x128 deepseek style block quantization
// * (1, 128)   for deepseek style activation quantization
//               (i.e. per-token-per-group)
//  shape of x is: (m, n)
//  T: type of X.
//  quant_type: fp8
#  ifdef ENABLE_FP8_TORCH
template <typename T>
void ScaledQuantizeFp8E4m3(T* x, void* output, float* scale2, std::vector<size_t> group_shape, int m, int n, int rank) {
  // goup_shape only support (128, 128).
  if (group_shape.size() != 2 || group_shape[0] < 1 || group_shape[1] < 1) {
    KLLM_LOG_ERROR << "group_shape's dims != 2 or not supported";
    return;
  }
  if (m % group_shape[0] != 0 || n % group_shape[1] != 0) {
    KLLM_LOG_ERROR << "Shape of x cannot be divisible by group shape";
    return;
  }

  int block_m = m / group_shape[0];
  int block_n = n / group_shape[1];

  auto origin_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<T>());
  torch::Tensor x_tensor = torch::from_blob(x, {m, n}, origin_options).to(torch::kFloat32);

  // Reshape and permute
  // (block_m,  BLOCK_SIZE_M, block_n, BLOCK_SIZE_N)
  x_tensor = x_tensor.reshape({block_m, group_shape[0], block_n, group_shape[1]});
  // (block_m, block_n, BLOCK_SIZE_M, BLOCK_SIZE_N)
  x_tensor = x_tensor.permute({0, 2, 1, 3}).flatten(2);

  // (block_m, block_n, BLOCK_SIZE_M * BLOCK_SIZE_N)
  x_tensor = x_tensor.flatten(2);

  // Compute scales
  auto [min_val, max_val] = x_tensor.aminmax(-1);
  auto amax = torch::max(min_val.abs(), max_val.abs()).clamp_min(1e-12);
  auto finfo_max = llm_kernels::utils::FP8_E4M3_MAX;
  auto finfo_min = llm_kernels::utils::FP8_E4M3_MIN;

  torch::Tensor finfo_max_tensor = torch::full_like(amax, finfo_max, torch::kCUDA);
  torch::Tensor scale = amax / finfo_max_tensor;

  // Apply scale and clamp
  scale = scale.unsqueeze(-1);
  torch::Tensor x_tensor_scaled = (x_tensor / (scale)).clamp(finfo_min, finfo_max);
  x_tensor_scaled = x_tensor_scaled.reshape({block_m, block_n, group_shape[0], group_shape[1]});
  x_tensor_scaled = x_tensor_scaled.permute({0, 2, 1, 3}).reshape({m, n});

  // Copy the result to output
  x_tensor_scaled = x_tensor_scaled.to(torch::kFloat8_e4m3fn).contiguous();
  CUDA_CHECK(cudaMemcpy(output, x_tensor_scaled.data_ptr(), m * n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(scale2, scale.data_ptr(), block_m * block_n * sizeof(float), cudaMemcpyDeviceToHost));
}

#    define SCALED_QUANTIZE_FP8_E4M3(T)                                                                          \
      template void ScaledQuantizeFp8E4m3<T>(T * x, void* output, float* scale, std::vector<size_t> group_shape, \
                                             int m, int n, int rank)

SCALED_QUANTIZE_FP8_E4M3(half);
#    ifdef ENABLE_BFLOAT16
SCALED_QUANTIZE_FP8_E4M3(__nv_bfloat16);
#    endif
SCALED_QUANTIZE_FP8_E4M3(float);
#  endif

// Dequant fp8_e4m3 block-wise
template <typename T>
void DequantFp8E4m3BlockWise(const void* d_data, const void* d_s, void* d_output, int m, int n, int block_size,
                             cudaStream_t& stream) {
  llm_kernels::nvidia::InvokeWeightDequant<T>(reinterpret_cast<const uint8_t*>(d_data),
                                              reinterpret_cast<const float*>(d_s), reinterpret_cast<T*>(d_output), m, n,
                                              block_size, stream);
}

#  define DEQUANTIZE_FP8_E4M3_BLOCKWISE(T)                                                                      \
    template void DequantFp8E4m3BlockWise<T>(const void* d_data, const void* d_s, void* d_output, int m, int n, \
                                             int block_size, cudaStream_t& stream)

DEQUANTIZE_FP8_E4M3_BLOCKWISE(half);
DEQUANTIZE_FP8_E4M3_BLOCKWISE(float);
#  ifdef ENABLE_BFLOAT16
DEQUANTIZE_FP8_E4M3_BLOCKWISE(__nv_bfloat16);
#  endif
#endif

template <typename T>
void InvokePerTokenGroupQuantFp8E4m3(void* input, void* output_q, void* output_s, int m, int n, bool is_column_major,
                                     cudaStream_t stream, int64_t group_size, float eps, float min_fp8, float max_fp8) {
#ifdef ENABLE_FP8
  llm_kernels::nvidia::per_token_group_quant_fp8<T>(input, output_q, output_s, m, n, group_size, is_column_major,
                                                    stream, eps, min_fp8, max_fp8);
#else
  KLLM_THROW("FP8 is not supported in this build. Please enable FP8 support.");
#endif
}
#define INVOKE_PER_TOKEN_GROUP_QUANT_FP8E4M3(T)                                                                   \
  template void InvokePerTokenGroupQuantFp8E4m3<T>(void* input, void* output_q, void* output_s, int m, int n,     \
                                                   bool is_column_major, cudaStream_t stream, int64_t group_size, \
                                                   float eps, float min_fp8, float max_fp8);
INVOKE_PER_TOKEN_GROUP_QUANT_FP8E4M3(float);
INVOKE_PER_TOKEN_GROUP_QUANT_FP8E4M3(half);
#ifdef ENABLE_BFLOAT16
INVOKE_PER_TOKEN_GROUP_QUANT_FP8E4M3(__nv_bfloat16);
#endif
#undef INVOKE_PER_TOKEN_GROUP_QUANT_FP8E4M3

template <typename T>
void InvokeFusedAddRmsNorm(void* input, void* residual, void* weight, double eps, int m, int n, cudaStream_t stream) {
  llm_kernels::nvidia::InvokeFusedAddRMSNorm<T>(input, residual, weight, eps, /*enable_pdl*/ false, m, n, stream);
}
#define INVOKE_FUSED_ADD_RMS_NORM(T)                                                                          \
  template void InvokeFusedAddRmsNorm<T>(void* input, void* residual, void* weight, double eps, int m, int n, \
                                         cudaStream_t stream);
INVOKE_FUSED_ADD_RMS_NORM(float);
INVOKE_FUSED_ADD_RMS_NORM(half);
#ifdef ENABLE_BFLOAT16
INVOKE_FUSED_ADD_RMS_NORM(__nv_bfloat16);
#endif
#undef INVOKE_FUSED_ADD_RMS_NORM

}  // namespace ksana_llm
