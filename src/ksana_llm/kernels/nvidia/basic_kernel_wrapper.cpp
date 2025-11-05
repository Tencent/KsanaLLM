/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/nvidia/basic_kernel_wrapper.h"

#include <fstream>
#include <iostream>

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"

#include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#include "ksana_llm/utils/singleton.h"

#include "ksana_llm/kernels/nvidia/triton_wrapper.h"

#include "csrc/kernels/nvidia/activation/activation.h"
#include "csrc/kernels/nvidia/add/add.h"
#include "csrc/kernels/nvidia/add_mul/add_mul.h"
#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "csrc/kernels/nvidia/assemble_tokens_hidden/assemble_tokens_hidden.h"
#include "csrc/kernels/nvidia/blockwise_gemm/blockwise_gemm.h"
#include "csrc/kernels/nvidia/cast/cast.h"
#include "csrc/kernels/nvidia/concat/concat.h"
#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/kernels/nvidia/expand/expand.h"
#include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#include "csrc/kernels/nvidia/fused_add_norm/fused_add_norm.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/grouped_topk/grouped_topk.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"
#include "csrc/kernels/nvidia/moe_utils/moe_utils.h"
#include "csrc/kernels/nvidia/others/sglang/main/quantization/fp8/per_token_group_quant.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/main/communication_kernels/trtllm_all_reduce.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy_flash_attn_layout.h"
#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/paged_attention.h"
#include "csrc/kernels/nvidia/permute/permute.h"
#include "csrc/kernels/nvidia/samplers/apply_token_bitmask_inplace.h"
#include "csrc/kernels/nvidia/samplers/greedy.h"
#include "csrc/utils/nvidia/cuda_fp8_utils.h"

#include "ksana_llm/kernels/argmax.h"
#include "ksana_llm/kernels/cast.h"

#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/search_status.h"

namespace ksana_llm {

#define GET_MACHETE_DATA_TYPE(T, MACHETE_TYPE)                          \
  template <>                                                           \
  llm_kernels::nvidia::vllm_dtype::ScalarType GetMacheteDataType<T>() { \
    return MACHETE_TYPE;                                                \
  }
GET_MACHETE_DATA_TYPE(float, llm_kernels::nvidia::vllm_dtype::kFloat);
GET_MACHETE_DATA_TYPE(half, llm_kernels::nvidia::vllm_dtype::kHalf);
GET_MACHETE_DATA_TYPE(__nv_bfloat16, llm_kernels::nvidia::vllm_dtype::kBFloat16);
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

void InvokeMarlinGptqRepack(const void* b_q_weight_ptr, const void* perm_ptr, void* out_ptr, int64_t num_experts,
                            int64_t size_k, int64_t size_n, int64_t num_bits, bool has_perm, int rank,
                            cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::marlin::gptq_marlin_repack(
      reinterpret_cast<const uint32_t*>(b_q_weight_ptr), reinterpret_cast<const uint32_t*>(perm_ptr),
      reinterpret_cast<uint32_t*>(out_ptr), num_experts, size_k, size_n, num_bits, has_perm, rank, stream));
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
GET_MARLIN_WORKSPACE(__nv_bfloat16);
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
INVOKE_MARLIN_PERMUTE_SCALES(__nv_bfloat16);
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
INVOKE_MARLIN_GEMM(__nv_bfloat16);
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
GET_TORCH_DATA_TYPE(__nv_bfloat16, torch::kBFloat16);
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
GET_FPA_INTB_GROUP_CUTLASS_GEMM_WORKSPACE_SIZE(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_GROUP_CUTLASS_GEMM_WORKSPACE_SIZE(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
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
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
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
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM_CONFIG_PROGILE(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUTLASS_GEMM_CONFIG_PROGILE(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
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
GET_FPA_INTB_GROUP_CUDA_GEMM_SUPPORTED(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
GET_FPA_INTB_GROUP_CUDA_GEMM_SUPPORTED(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
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
INVOKE_FPA_INTB_GROUP_CUDA_GEMM(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT4);
INVOKE_FPA_INTB_GROUP_CUDA_GEMM(__nv_bfloat16, llm_kernels::nvidia::WeightType::INT8);
#undef INVOKE_FPA_INTB_GROUP_CUDA_GEMM

template <typename T>
void LookupEmbedding(const void* input_ids, const void* ids_offsets, const void* prefix_offsets, const void* emb,
                     const void* pos, const void* steps, void* output, bool use_emb_scale, const T emb_scale,
                     int vocab_size, int hidden_size, int bs, int vocab_id, cudaStream_t stream, void* workspace_ptr) {
  const bool do_position_encoding = (pos != nullptr) && (steps != nullptr);
  if (do_position_encoding) {
    if (use_emb_scale) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<T, true, true>(
          reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), reinterpret_cast<const T*>(pos), emb_scale, {},
          reinterpret_cast<const int32_t*>(input_ids), reinterpret_cast<const size_t*>(steps),
          reinterpret_cast<const size_t*>(ids_offsets), reinterpret_cast<const size_t*>(prefix_offsets), bs,
          hidden_size, vocab_size, vocab_id, stream));
    } else {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<T, true, false>(
          reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), reinterpret_cast<const T*>(pos), emb_scale, {},
          reinterpret_cast<const int32_t*>(input_ids), reinterpret_cast<const size_t*>(steps),
          reinterpret_cast<const size_t*>(ids_offsets), reinterpret_cast<const size_t*>(prefix_offsets), bs,
          hidden_size, vocab_size, vocab_id, stream));
    }
  } else {
    if (use_emb_scale) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<T, false, true>(
          reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), /* pos */ nullptr, emb_scale, {},
          reinterpret_cast<const int32_t*>(input_ids), /* steps */ nullptr,
          reinterpret_cast<const size_t*>(ids_offsets), reinterpret_cast<const size_t*>(prefix_offsets), bs,
          hidden_size, vocab_size, vocab_id, stream));
    } else {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::LookupFusedEmbeddingWithCSRInputs<T, false, false>(
          reinterpret_cast<T*>(output), reinterpret_cast<const T*>(emb), /* pos */ nullptr, emb_scale, {},
          reinterpret_cast<const int32_t*>(input_ids), /* steps */ nullptr,
          reinterpret_cast<const size_t*>(ids_offsets), reinterpret_cast<const size_t*>(prefix_offsets), bs,
          hidden_size, vocab_size, vocab_id, stream));
    }
  }
}
#define LOOKUP_EMBEDDING(T)                                                                                        \
  template void LookupEmbedding<T>(const void* input_ids, const void* ids_offsets, const void* prefix_offsets,     \
                                   const void* emb, const void* pos, const void* steps, void* output,              \
                                   bool use_emb_scale, const T emb_scale, int vocab_size, int hidden_size, int bs, \
                                   int vocab_id, cudaStream_t stream, void* workspace_ptr)
LOOKUP_EMBEDDING(float);
LOOKUP_EMBEDDING(half);
LOOKUP_EMBEDDING(__nv_bfloat16);
#undef LOOKUP_EMBEDDING

template <typename T>
void InvokeLayerNorm(const void* input, const void* weight, const void* bias, float layernorm_eps, int m, int n,
                     void* output, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeLayerNorm<T>(
      reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input), reinterpret_cast<const T*>(weight),
      reinterpret_cast<const T*>(bias), layernorm_eps, m, n, stream));
}
#define INVOKE_LAYER_NORM(T)                                                                                     \
  template void InvokeLayerNorm<T>(const void* input, const void* weight, const void* bias, float layernorm_eps, \
                                   int m, int n, void* output, cudaStream_t stream)
INVOKE_LAYER_NORM(float);
INVOKE_LAYER_NORM(half);
INVOKE_LAYER_NORM(__nv_bfloat16);
#undef INVOKE_LAYER_NORM

template <typename T>
void InvokeRMSNorm(void* input, void* weight, float layernorm_eps, int m, int n, void* output, bool enable_pdl,
                   cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeRMSNorm<T>(reinterpret_cast<T*>(output), reinterpret_cast<T*>(input),
                                                              reinterpret_cast<T*>(weight), layernorm_eps, m, n,
                                                              enable_pdl, stream));
}
#define INVOKE_RMS_NORM(T)                                                                                   \
  template void InvokeRMSNorm<T>(void* input, void* weight, float layernorm_eps, int m, int n, void* output, \
                                 bool enable_pdl, cudaStream_t stream)
INVOKE_RMS_NORM(float);
INVOKE_RMS_NORM(half);
INVOKE_RMS_NORM(__nv_bfloat16);
#undef INVOKE_RMS_NORM

#define INVOKE_MATMUL(T, CUDA_TYPE)                                                                                  \
  template <>                                                                                                        \
  void InvokeMatMul<T>(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,          \
                       const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream, void* workspace_ptr, \
                       cublasLtMatmulAlgo_t* cublaslt_algo, size_t workspace_size) {                                 \
    CUDA_CHECK(llm_kernels::nvidia::InvokeCublasGemm(                                                                \
        cublas_handle, cublaslt_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, b_ptr, n, CUDA_TYPE, a_ptr, k, CUDA_TYPE, \
        c_ptr, n, CUDA_TYPE, CUDA_R_32F, stream, workspace_ptr, workspace_size, cublaslt_algo));                     \
  }
INVOKE_MATMUL(float, CUDA_R_32F);
INVOKE_MATMUL(half, CUDA_R_16F);
INVOKE_MATMUL(__nv_bfloat16, CUDA_R_16BF);
#undef INVOKE_MATMUL

#define INVOKE_BATCHED_GEMM(T, CUDA_TYPE)                                                                              \
  template <>                                                                                                          \
  void InvokeBatchedMatMul<T>(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int batch_size, int m,   \
                              int n, int k, int lda, int ldb, int ldc, int64_t batch_offset_a, int64_t batch_offset_b, \
                              int64_t batch_offset_c, const void* a_ptr, const void* b_ptr, void* c_ptr,               \
                              cudaStream_t& stream, void* workspace_ptr, size_t workspace_size,                        \
                              cublasLtMatmulAlgo_t* cublaslt_algo) {                                                   \
    CUDA_CHECK(llm_kernels::nvidia::InvokeCublasGemm(                                                                  \
        cublas_handle, cublaslt_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, b_ptr, ldb, CUDA_TYPE, a_ptr, lda,          \
        CUDA_TYPE, c_ptr, ldc, CUDA_TYPE, batch_size, /*f_alpha*/ 1.0f, /*f_beta*/ 0.0f, CUDA_R_32F, stream,           \
        workspace_ptr, workspace_size, cublaslt_algo, /*a_scale*/ nullptr, /*b_scale*/ nullptr, batch_offset_b,        \
        batch_offset_a, batch_offset_c));                                                                              \
  }
INVOKE_BATCHED_GEMM(float, CUDA_R_32F);
INVOKE_BATCHED_GEMM(half, CUDA_R_16F);
INVOKE_BATCHED_GEMM(__nv_bfloat16, CUDA_R_16BF);
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
INVOKE_ADD_BIAS_RESIDUAL(__nv_bfloat16);
#undef INVOKE_ADD_BIAS_RESIDUAL

// Add-Multiply fused operations implementations
template <typename T>
void InvokeAddThenMul(const void* input1, const void* input2, const T scale, const int m, const int n, void* output,
                      cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokeAddThenMul<T>(reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input1),
                                               reinterpret_cast<const T*>(input2), scale, m, n, stream));
}

template <typename T>
void InvokeAddMulSecond(const void* input1, const void* input2, const T scale, const int m, const int n, void* output,
                        cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokeAddMulSecond<T>(reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input1),
                                                 reinterpret_cast<const T*>(input2), scale, m, n, stream));
}

template <typename T>
void InvokeAddBiasThenMul(const void* input1, const void* input2, const void* bias, const T scale, const int m,
                          const int n, void* output, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeAddBiasThenMul<T>(
      reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input1), reinterpret_cast<const T*>(input2),
      reinterpret_cast<const T*>(bias), scale, m, n, stream));
}

template <typename T>
void InvokeMulThenAdd(const void* input1, const void* input2, const T scale1, const T scale2, const int m, const int n,
                      void* output, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokeMulThenAdd<T>(reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input1),
                                               reinterpret_cast<const T*>(input2), scale1, scale2, m, n, stream));
}

template <typename T>
void InvokeAddResidualsBiasThenMul(const void* input1, const void* residual1, const void* residual2, const void* bias,
                                   const T scale, const int m, const int n, void* output, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::InvokeAddResidualsBiasThenMul<T>(
      reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input1), reinterpret_cast<const T*>(residual1),
      reinterpret_cast<const T*>(residual2), reinterpret_cast<const T*>(bias), scale, m, n, stream));
}

// Explicit template instantiations for Add-Multiply operations
#define INVOKE_ADD_THEN_MUL(T)                                                                                       \
  template void InvokeAddThenMul<T>(const void* input1, const void* input2, const T scale, const int m, const int n, \
                                    void* output, cudaStream_t stream)
INVOKE_ADD_THEN_MUL(float);
INVOKE_ADD_THEN_MUL(half);
INVOKE_ADD_THEN_MUL(__nv_bfloat16);
#undef INVOKE_ADD_THEN_MUL

#define INVOKE_ADD_MUL_SECOND(T)                                                                                       \
  template void InvokeAddMulSecond<T>(const void* input1, const void* input2, const T scale, const int m, const int n, \
                                      void* output, cudaStream_t stream)
INVOKE_ADD_MUL_SECOND(float);
INVOKE_ADD_MUL_SECOND(half);
INVOKE_ADD_MUL_SECOND(__nv_bfloat16);
#undef INVOKE_ADD_MUL_SECOND

#define INVOKE_ADD_BIAS_THEN_MUL(T)                                                                              \
  template void InvokeAddBiasThenMul<T>(const void* input1, const void* input2, const void* bias, const T scale, \
                                        const int m, const int n, void* output, cudaStream_t stream)
INVOKE_ADD_BIAS_THEN_MUL(float);
INVOKE_ADD_BIAS_THEN_MUL(half);
INVOKE_ADD_BIAS_THEN_MUL(__nv_bfloat16);
#undef INVOKE_ADD_BIAS_THEN_MUL

#define INVOKE_MUL_THEN_ADD(T)                                                                              \
  template void InvokeMulThenAdd<T>(const void* input1, const void* input2, const T scale1, const T scale2, \
                                    const int m, const int n, void* output, cudaStream_t stream)
INVOKE_MUL_THEN_ADD(float);
INVOKE_MUL_THEN_ADD(half);
INVOKE_MUL_THEN_ADD(__nv_bfloat16);
#undef INVOKE_MUL_THEN_ADD

#define INVOKE_ADD_RESIDUALS_BIAS_THEN_MUL(T)                                                                      \
  template void InvokeAddResidualsBiasThenMul<T>(const void* input1, const void* residual1, const void* residual2, \
                                                 const void* bias, const T scale, const int m, const int n,        \
                                                 void* output, cudaStream_t stream)
INVOKE_ADD_RESIDUALS_BIAS_THEN_MUL(float);
INVOKE_ADD_RESIDUALS_BIAS_THEN_MUL(half);
INVOKE_ADD_RESIDUALS_BIAS_THEN_MUL(__nv_bfloat16);
#undef INVOKE_ADD_RESIDUALS_BIAS_THEN_MUL

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
INVOKE_ROW_BASED_GATED_ACTIVATION(llm_kernels::nvidia::SiluActivation, __nv_bfloat16);

#define INVOKE_GATED_ACTIVATION(Activation, T)                                                                       \
  template void InvokeGatedActivation<Activation, T>(const void* input, const void* bias, const void* gated_weights, \
                                                     const void* gated_bias, const int m, const int n, void* output, \
                                                     cudaStream_t stream)
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::GeluActivation, float);
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::GeluActivation, half);
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::GeluActivation, __nv_bfloat16);

INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::SiluActivation, float);
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::SiluActivation, half);
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::SiluActivation, __nv_bfloat16);

INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::ReluActivation, float);
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::ReluActivation, half);
INVOKE_GATED_ACTIVATION(llm_kernels::nvidia::ReluActivation, __nv_bfloat16);

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
ASSEMBEL_LAST_TOKEN(__nv_bfloat16);
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
CONCAT(__nv_bfloat16);
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
INVOKE_EXPAND(__nv_bfloat16);
#undef INVOKE_EXPAND

// ptr: 指向所维护的CustomAllreduce算子的指针
// signals: 所有GPU共享的中间结果的指针数组
// rank_data & rank_data_sz: 当前GPU的中间结果数据
template <typename T>
void CustomAllReduceInit(void** ptr, void* rank_data, size_t rank_data_sz, int cur_rank, int total_ranks,
                         bool is_full_nvlink, uint32_t root_rank) {
  *ptr = new llm_kernels::nvidia::CustomAllreduce(rank_data, rank_data_sz, cur_rank, total_ranks, is_full_nvlink,
                                                  root_rank);
}

#define CUSTOM_ALL_REDUCE_INIT(T)                                                                      \
  template void CustomAllReduceInit<T>(void** ptr, void* rank_data, size_t rank_data_sz, int cur_rank, \
                                       int total_ranks, bool is_full_nvlink, uint32_t root_rank);
CUSTOM_ALL_REDUCE_INIT(float);
CUSTOM_ALL_REDUCE_INIT(half);
CUSTOM_ALL_REDUCE_INIT(__nv_bfloat16);
#undef CUSTOM_ALL_REDUCE_INIT

template <typename T>
void CustomAllReduceRegisterBuffer(void* ptr, void** input_handles, cudaStream_t& stream) {
  llm_kernels::nvidia::CustomAllreduce* reduce_op = static_cast<llm_kernels::nvidia::CustomAllreduce*>(ptr);
  reduce_op->RegisterBuffer(input_handles, stream);
}

template <typename T>
void CustomAllReduceRegisterSignalBuffer(void* ptr, void** signals) {
  llm_kernels::nvidia::CustomAllreduce* reduce_op = static_cast<llm_kernels::nvidia::CustomAllreduce*>(ptr);
  reduce_op->RegisterSignalBuffer((llm_kernels::nvidia::Signal**)signals);
}

#define CUSTOM_ALL_REDUCE_REGISTER_BUFFER(T)                                                             \
  template void CustomAllReduceRegisterBuffer<T>(void* ptr, void** input_handles, cudaStream_t& stream); \
  template void CustomAllReduceRegisterSignalBuffer<T>(void* ptr, void** signals);
CUSTOM_ALL_REDUCE_REGISTER_BUFFER(float);
CUSTOM_ALL_REDUCE_REGISTER_BUFFER(half);
CUSTOM_ALL_REDUCE_REGISTER_BUFFER(__nv_bfloat16);
#undef CUSTOM_ALL_REDUCE_REGISTER_BUFFER

template <typename T>
void CustomAllReduceRun(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream) {
  llm_kernels::nvidia::CustomAllreduce* reduce_op = static_cast<llm_kernels::nvidia::CustomAllreduce*>(ptr);
  reduce_op->AllReduce<T>(stream, static_cast<T*>(input), static_cast<T*>(result), data_size);
}

template void CustomAllReduceRun<float>(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream);
template void CustomAllReduceRun<half>(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream);
template void CustomAllReduceRun<__nv_bfloat16>(void* ptr, void* input, void* result, int data_size,
                                                cudaStream_t& stream);

// Allocate three workspaces required for trt allreduce
// three buffers: data buffer: [nranks * max_token_num * hidden_dim * data_type_size]
//                flag buffer: [nranks * (kBarrierFlagCount = 256) * sizeof(int)]
//                lamport buffer: [nranks * max_token_num * hidden_dim * data_type_size * 3]
//                The factor of 3 is due to the flag value having three different states
// flag pointers buffer: [5 * sizeof(void*)]
//                       [0]: atomic flag read counter
//                       [1]: non-lamport flag
//                       [2]: lamport flag
//                       [3]: lamport triple buffer offset
//                       [4]: lamport clear size
// workspace pointers buffer: [(3 * nranks + 1) * sizeof(void*)]
//                            [0: 3 * nranks): addresses of three buffers for each rank
//                            [3 * nranks]: address of flag pointers buffer for the current rank
void AllocTrtAllReduceWorkspace(const int rank, const int max_token_num, const int hidden_dim, const int data_type_size,
                                std::vector<void*>& buffer_d_ptrs, std::vector<void*>& flag_d_ptrs,
                                std::vector<void*>& workspace_d_ptrs, cudaStream_t stream) {
  const int nranks = static_cast<int>(flag_d_ptrs.size());
  llm_kernels::nvidia::AllocTrtAllReduceWorkspace(nranks, rank, max_token_num, hidden_dim, data_type_size,
                                                  buffer_d_ptrs, flag_d_ptrs, workspace_d_ptrs, stream);
}

// Initialize the three workspaces used in trt allreduce after allocation is finished
void InitTrtAllReduceWorkspace(const int rank, const std::vector<void*>& buffer_d_ptrs,
                               const std::vector<void*>& flag_d_ptrs, const std::vector<void*>& workspace_d_ptrs,
                               cudaStream_t stream) {
  const int nranks = static_cast<int>(flag_d_ptrs.size());
  llm_kernels::nvidia::InitTrtAllReduceWorkspace(nranks, rank, buffer_d_ptrs, flag_d_ptrs, workspace_d_ptrs, stream);
}

// Free the three workspaces used in trt allreduce
void FreeTrtAllReduceWorkspace(const int rank, const std::vector<void*>& buffer_d_ptrs,
                               const std::vector<void*>& flag_d_ptrs, const std::vector<void*>& workspace_d_ptrs,
                               cudaStream_t stream) {
  const int nranks = static_cast<int>(flag_d_ptrs.size());
  llm_kernels::nvidia::FreeTrtAllReduceWorkspace(nranks, rank, buffer_d_ptrs, flag_d_ptrs, workspace_d_ptrs, stream);
}

template <typename T>
void RunTrtAllReduce(void* input, const int rank, const int token_num, const int hidden_dim,
                     const std::vector<void*>& workspace_d_ptrs, void* output, cudaStream_t stream) {
  llm_kernels::nvidia::AllReduceFusionParams<T> params;
  params.nranks = static_cast<int>(workspace_d_ptrs.size());
  params.rank = rank;
  params.size = token_num * hidden_dim;
  params.hidden_dim = hidden_dim;
  params.workspace = reinterpret_cast<void**>(workspace_d_ptrs[rank]);
  params.allreduce_in = input;
  params.allreduce_out = output;
  // Always use oneshot, since twoshot has poor performance
  params.use_oneshot = true;
  params.stream = stream;
  params.pattern = llm_kernels::nvidia::AllReduceFusionPattern::kAllReduce;
  llm_kernels::nvidia::allreduce_fusion_op(params);
}
#define RUN_TRT_ALLREDUCE(T)                                                                                 \
  template void RunTrtAllReduce<T>(void*, const int, const int, const int, const std::vector<void*>&, void*, \
                                   cudaStream_t)
RUN_TRT_ALLREDUCE(float);
RUN_TRT_ALLREDUCE(half);
RUN_TRT_ALLREDUCE(__nv_bfloat16);
#undef RUN_TRT_ALLREDUCE

template <typename T>
void RunTrtFusedAllReduceResidualNorm(void* input, const int rank, const int token_num, const int hidden_dim,
                                      const std::vector<void*>& workspace_d_ptrs, void* d_rms_gamma_ptr, float rms_eps,
                                      void* residual_in_ptr, void* residual_out_ptr, void* norm_out_ptr,
                                      cudaStream_t stream) {
  llm_kernels::nvidia::AllReduceFusionParams<T> params;
  params.nranks = static_cast<int>(workspace_d_ptrs.size());
  params.rank = rank;
  params.size = token_num * hidden_dim;
  params.hidden_dim = hidden_dim;
  params.workspace = reinterpret_cast<void**>(workspace_d_ptrs[rank]);
  params.allreduce_in = input;
  params.rms_gamma = d_rms_gamma_ptr;
  params.rms_eps = rms_eps;
  params.use_oneshot = true;
  params.residual_in = residual_in_ptr;
  params.residual_out = residual_out_ptr;  // inplace
  params.norm_out = norm_out_ptr;
  params.stream = stream;
  params.pattern = llm_kernels::nvidia::AllReduceFusionPattern::kARResidualRMSNorm;
  llm_kernels::nvidia::allreduce_fusion_op(params);
}
#define RUN_TRT_FUSED_ALLREDUCE_RESIDUAL_NORM(T)                                                                       \
  template void RunTrtFusedAllReduceResidualNorm<T>(void*, const int, const int, const int, const std::vector<void*>&, \
                                                    void*, float, void*, void*, void*, cudaStream_t)
RUN_TRT_FUSED_ALLREDUCE_RESIDUAL_NORM(float);
RUN_TRT_FUSED_ALLREDUCE_RESIDUAL_NORM(half);
RUN_TRT_FUSED_ALLREDUCE_RESIDUAL_NORM(__nv_bfloat16);
#undef RUN_TRT_FUSED_ALLREDUCE_RESIDUAL_NORM

template <typename T>
void InvokeSigmoidActivation(void* input, const size_t size, const float scale, cudaStream_t& stream) {
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokeSigmoid<T>(reinterpret_cast<T*>(input), static_cast<int32_t>(size), scale, stream));
}

template void InvokeSigmoidActivation<float>(void* input, const size_t size, const float scale, cudaStream_t& stream);
template void InvokeSigmoidActivation<half>(void* input, const size_t size, const float scale, cudaStream_t& stream);
template void InvokeSigmoidActivation<__nv_bfloat16>(void* input, const size_t size, const float scale,
                                                     cudaStream_t& stream);

template <>
ncclDataType_t GetNcclDataType<float>() {
  return ncclFloat;
}
template <>
ncclDataType_t GetNcclDataType<half>() {
  return ncclHalf;
}
template <>
ncclDataType_t GetNcclDataType<__nv_bfloat16>() {
  return ncclBfloat16;
}

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
INVOKE_PERMUTE(__nv_bfloat16);
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

template <>
void DataToFloat<__nv_bfloat16>(const void* input, const int data_size, const size_t vocab_size,
                                const size_t vocab_size_pad, void* output, cudaStream_t& stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToFloat(reinterpret_cast<const __nv_bfloat16*>(input), data_size,
                                                             reinterpret_cast<float*>(output), stream, vocab_size_pad,
                                                             vocab_size));
}

Status CastInplace(Tensor& tensor, const DataType target_dtype, Stream& stream, void* workspace_ptr) {
  if (tensor.dtype == DataType::TYPE_BF16 && target_dtype == DataType::TYPE_FP16) {
    CUDA_CHECK_LAST_ERROR(
        llm_kernels::nvidia::BFloat16ToHalf(tensor.GetPtr<void>(), tensor.GetElementNumber(), stream.Get()));
  } else if (tensor.dtype == DataType::TYPE_FP16 && target_dtype == DataType::TYPE_BF16) {
    CUDA_CHECK_LAST_ERROR(
        llm_kernels::nvidia::HalfToBFloat16(tensor.GetPtr<void>(), tensor.GetElementNumber(), stream.Get()));
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
    InvokePermute<__nv_bfloat16>(input_tensor.GetPtr<void>(), output_tensor.GetPtr<void>(), input_tensor.shape,
                                 permutation, stream.Get());
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
InvokeMUL(__nv_bfloat16);
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

  if (temperatures != nullptr) {
    auto temperatures_tensor = torch::from_blob(temperatures, {bs}, options);
    logits_tensor = logits_tensor.div_(temperatures_tensor.unsqueeze_(1));
  }
  logits_tensor = logits_tensor.log_softmax(-1);

  torch::Tensor logits_sort, logits_idx;
  // TODO(winminkong): Modify top-k to use a calculation method consistent with the generated tokens. Avoid the
  // situation where the index of the maximum logprobs does not match the generated tokens.
  std::tie(logits_sort, logits_idx) = logits_tensor.topk(logprobs_num, 1, true, true);
  logits_sort = logits_sort.to(torch::kCPU).view({-1});
  logits_idx = logits_idx.to(torch::kCPU).view({-1});
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
INSTANTIATE_ARG_MAX(__nv_bfloat16);

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
INSTANTIATE_FP8_E4M3_QUANTIZE(__nv_bfloat16);
#  undef INSTANTIATE_FP8_E4M3_QUANTIZE

#  define INVOKE_FP8_QUANTIZED_MATMUL(T, CUDA_TYPE)                                                                    \
    template <>                                                                                                        \
    void Fp8QuantizedMatMul<T>(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,    \
                               const void* a_ptr, const void* a_scale, const void* b_ptr, const void* b_scale,         \
                               T* c_ptr, cudaStream_t& stream, cublasLtMatmulAlgo_t* cublaslt_algo, void* workspace,   \
                               size_t workspace_size) {                                                                \
      CUDA_CHECK(llm_kernels::nvidia::InvokeCublasGemm(cublas_handle, cublaslt_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, \
                                                       k, b_ptr, k, CUDA_R_8F_E4M3, a_ptr, k, CUDA_R_8F_E4M3, c_ptr,   \
                                                       n, CUDA_TYPE, 1, 1.0f, 0.f, CUDA_R_32F, stream, workspace,      \
                                                       workspace_size, nullptr, a_scale, b_scale));                    \
    }

INVOKE_FP8_QUANTIZED_MATMUL(float, CUDA_R_32F);
INVOKE_FP8_QUANTIZED_MATMUL(half, CUDA_R_16F);
INVOKE_FP8_QUANTIZED_MATMUL(__nv_bfloat16, CUDA_R_16BF);
#  undef INVOKE_FP8_QUANTIZED_MATMUL

void RescaleFp8E4m3(void* input, void* output, size_t n, const float* input_scale, const float* output_scale,
                    cudaStream_t& stream) {
  llm_kernels::utils::InvokeRescaleFp8E4m3(input, output, n, input_scale, output_scale, stream);
}
#endif

size_t InvokeGetCublasWorkspaceSize() { return llm_kernels::nvidia::GetCublasWorkspaceSize(); }

cudaStream_t InvokeSetTorchStream(cudaStream_t& stream, int rank) {
  cudaStream_t old_stream = c10::cuda::getCurrentCUDAStream(rank).stream();
  // set compute stream as torch stream
  c10::cuda::CUDAStream new_stream = c10::cuda::getStreamFromExternal(stream, rank);
  c10::cuda::setCurrentCUDAStream(new_stream);
  return old_stream;
}

template <typename T>
size_t InvokeGetBlockGemmWorkspaceSize(int m, int k, int n) {
  return llm_kernels::nvidia::GetBlockwiseGemmWorkspaceSize<T>(m, k, n);
}

#define GET_BLOCKWISE_GEMM_WORKSPACE_SIZE(T) template size_t InvokeGetBlockGemmWorkspaceSize<T>(int m, int k, int n)
GET_BLOCKWISE_GEMM_WORKSPACE_SIZE(float);
GET_BLOCKWISE_GEMM_WORKSPACE_SIZE(half);
GET_BLOCKWISE_GEMM_WORKSPACE_SIZE(__nv_bfloat16);
#undef GET_BLOCKWISE_GEMM_WORKSPACE_SIZE

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
BLOCKWISE_GEMM(__nv_bfloat16);
#undef BLOCKWISE_GEMM

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
SCALED_QUANTIZE_FP8_E4M3(__nv_bfloat16);
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
DEQUANTIZE_FP8_E4M3_BLOCKWISE(__nv_bfloat16);
#endif

template <typename T>
void InvokePerTokenGroupQuantFp8E4m3(const void* input, void* output_q, void* output_s, int m, int n,
                                     bool is_column_major, cudaStream_t stream, int64_t group_size,
                                     const PerTokenGroupQuantFusionParams& params) {
#ifdef ENABLE_FP8
  llm_kernels::nvidia::per_token_group_quant_fp8<T>(input, output_q, output_s, m, n, group_size, is_column_major,
                                                    params.fuse_silu_mul, stream);
#else
  KLLM_THROW("FP8 is not supported in this build. Please enable FP8 support.");
#endif
}
#define INVOKE_PER_TOKEN_GROUP_QUANT_FP8E4M3(T)                                                                     \
  template void InvokePerTokenGroupQuantFp8E4m3<T>(const void* input, void* output_q, void* output_s, int m, int n, \
                                                   bool is_column_major, cudaStream_t stream, int64_t group_size,   \
                                                   const PerTokenGroupQuantFusionParams& params)
INVOKE_PER_TOKEN_GROUP_QUANT_FP8E4M3(float);
INVOKE_PER_TOKEN_GROUP_QUANT_FP8E4M3(half);
INVOKE_PER_TOKEN_GROUP_QUANT_FP8E4M3(__nv_bfloat16);
#undef INVOKE_PER_TOKEN_GROUP_QUANT_FP8E4M3

template <typename T>
void InvokeFusedAddRmsNorm(void* input, void* residual, void* weight, double eps, int m, int n, bool enable_pdl,
                           cudaStream_t stream) {
  llm_kernels::nvidia::InvokeFusedAddRMSNorm<T>(input, residual, weight, eps, enable_pdl, m, n, stream);
}
#define INVOKE_FUSED_ADD_RMS_NORM(T)                                                                          \
  template void InvokeFusedAddRmsNorm<T>(void* input, void* residual, void* weight, double eps, int m, int n, \
                                         bool enable_pdl, cudaStream_t stream);
INVOKE_FUSED_ADD_RMS_NORM(float);
INVOKE_FUSED_ADD_RMS_NORM(half);
INVOKE_FUSED_ADD_RMS_NORM(__nv_bfloat16);
#undef INVOKE_FUSED_ADD_RMS_NORM

template <typename T>
void InvokeSplit(const T* __restrict__ input, const std::vector<T*>& output_ptrs,
                 std::vector<int>& col_offsets,  // [0, col1, col1+col2, ...]
                 int rows, int cols, int num_outputs, cudaStream_t& stream) {
  if (output_ptrs.size() != num_outputs || col_offsets.size() != num_outputs + 1) {
    KLLM_THROW("Invalid input for InvokeSplit");
  }
  llm_kernels::nvidia::InvokeSplit<T>(input, output_ptrs, col_offsets, rows, cols, num_outputs, stream);
}
#define INVOKE_SPLIT(T)                                                                            \
  template void InvokeSplit<T>(const T* __restrict__ input, const std::vector<T*>& output_ptrs,    \
                               std::vector<int>& col_offsets, int rows, int cols, int num_outputs, \
                               cudaStream_t& stream);
INVOKE_SPLIT(float);
INVOKE_SPLIT(half);
INVOKE_SPLIT(__nv_bfloat16);
#undef INVOKE_SPLIT

template <typename T>
void InvokeExtractSubMatrix(const T* __restrict__ input, T* __restrict__ output, size_t m, size_t input_n,
                            size_t output_n, cudaStream_t& stream) {
  llm_kernels::nvidia::InvokeExtractSubMatrix<T>(input, output, m, input_n, output_n, stream);
}

#define INVOKE_GATHER_SUBMATRIX(T)                                                                                    \
  template void InvokeExtractSubMatrix(const T* __restrict__ input, T* __restrict__ output, size_t m, size_t input_n, \
                                       size_t output_n, cudaStream_t& stream);

INVOKE_GATHER_SUBMATRIX(float);
INVOKE_GATHER_SUBMATRIX(half);
INVOKE_GATHER_SUBMATRIX(__nv_bfloat16);

#undef INVOKE_GATHER_SUBMATRIX

void InvokeProcessKvList(void** kv_list, size_t layer_num, size_t block_num, size_t block_size, cudaStream_t stream) {
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ProcessKvList(kv_list, layer_num, block_num, block_size, stream));
}

// Apply token bitmask to logits for grammar-constrained sampling
template <typename T>
void InvokeApplyTokenBitmaskInplace(void* logits, const void* bitmask, const void* indices, int32_t vocab_size,
                                    int32_t logits_stride, int32_t bitmask_stride, int32_t num_rows,
                                    cudaStream_t stream) {
  if (logits == nullptr || bitmask == nullptr) {
    KLLM_LOG_ERROR << "logits or bitmask pointer is null, cannot apply token bitmask";
    return;
  }

  const int32_t* bitmask_ptr = static_cast<const int32_t*>(bitmask);
  const int32_t* indices_ptr = (indices != nullptr) ? static_cast<const int32_t*>(indices) : nullptr;

  // Call the underlying CUDA kernel implementation
  llm_kernels::nvidia::ApplyTokenBitmaskInplace<T>(static_cast<T*>(logits), bitmask_ptr, indices_ptr, vocab_size,
                                                   logits_stride, bitmask_stride, num_rows, stream);
}
#define INVOKE_APPLY_TOKEN_BITMASK_INPLACE(T)                                                                        \
  template void InvokeApplyTokenBitmaskInplace<T>(void* logits, const void* bitmask, const void* indices,            \
                                                  int32_t vocab_size, int32_t logits_stride, int32_t bitmask_stride, \
                                                  int32_t num_rows, cudaStream_t stream);
INVOKE_APPLY_TOKEN_BITMASK_INPLACE(float);
INVOKE_APPLY_TOKEN_BITMASK_INPLACE(half);
INVOKE_APPLY_TOKEN_BITMASK_INPLACE(__nv_bfloat16);
#undef INVOKE_APPLY_TOKEN_BITMASK_INPLACE

void InvokeCalculateChecksum(void** d_ptrs, size_t* d_results, int num_ptrs, size_t data_size_in_bytes,
                             cudaStream_t stream) {
  llm_kernels::nvidia::CalculateChecksum(d_ptrs, d_results, num_ptrs, data_size_in_bytes, stream);
}

}  // namespace ksana_llm
