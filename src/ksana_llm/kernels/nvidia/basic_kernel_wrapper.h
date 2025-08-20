/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <optional>
#include <vector>

#include "csrc/kernels/nvidia/adjust_mem/adjust_mem.h"
#include "csrc/kernels/nvidia/asymmetric_gemm/asymmetric_gemm_wrapper.h"
#include "csrc/kernels/nvidia/cast/cast.h"
#include "csrc/kernels/nvidia/gptq_marlin/marlin_wrapper.h"
#include "csrc/kernels/nvidia/identity/identity.h"
#include "csrc/kernels/nvidia/machete/machete_wrapper.h"
#include "csrc/kernels/nvidia/mixture_of_experts/moe_wrapper.h"
#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "csrc/kernels/nvidia/split/split.h"
#include "csrc/kernels/nvidia/weight_only_batched_gemv/weight_only_gemv_wrapper.h"
#include "csrc/utils/nvidia/scalar_type.hpp"
#include "csrc/utils/quant_type.h"

#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/nvidia/nccl_utils.h"

namespace ksana_llm {

void DequantInt4Fp8(cudaStream_t stream, void* output, const void* input, const size_t datasize);
template <typename T>
llm_kernels::nvidia::vllm_dtype::ScalarType GetMacheteDataType();

std::vector<std::string> GetMacheteSupportedSchedules(
    llm_kernels::nvidia::vllm_dtype::ScalarType a_type, llm_kernels::nvidia::vllm_dtype::ScalarType b_type,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_group_scales_type,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> maybe_group_zeros_type);

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
                       std::optional<int64_t> maybe_group_size, std::optional<std::string> maybe_schedule);

void InvokeMachetePrepackWeight(
    const void* B_ptr, const std::vector<size_t>& B_shape, void* out_ptr,
    llm_kernels::nvidia::vllm_dtype::ScalarType const& a_type,
    llm_kernels::nvidia::vllm_dtype::ScalarType const& b_type,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_scales_type, cudaStream_t stream);

std::string GetMacheteBestSchedule(
    size_t warmup_iters, size_t record_iters, void* workspace, cudaStream_t stream, int M, int N, int K,
    const void* Aptr, const void* Bptr, void* Dptr, llm_kernels::nvidia::vllm_dtype::ScalarType const& a_type,
    llm_kernels::nvidia::vllm_dtype::ScalarType const& b_type, std::optional<void*> const& maybe_group_scales_ptr,
    std::optional<std::vector<size_t>> const& maybe_group_scales_shape,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_scales_type,
    std::optional<void*> const& maybe_group_zeros_ptr,
    std::optional<std::vector<size_t>> const& maybe_group_zeros_shape,
    std::optional<llm_kernels::nvidia::vllm_dtype::ScalarType> const& maybe_group_zeros_type,
    std::optional<int64_t> maybe_group_size);

void InvokeMarlinAwqRepack(const void* b_q_weight_ptr, void* out_ptr, int64_t size_k, int64_t size_n, int64_t num_bits,
                           int rank, cudaStream_t stream);

std::vector<int64_t> GetMarlinAwqRepackMeta(int64_t size_k, int64_t size_n, int64_t num_bits);

void InvokeMarlinGptqRepack(const void* b_q_weight_ptr, const void* perm_ptr, void* out_ptr, int64_t num_experts,
                            int64_t size_k, int64_t size_n, int64_t num_bits, bool has_perm, int rank,
                            cudaStream_t stream);

std::vector<int64_t> GetMarlinGptqRepackMeta(int64_t size_k, int64_t size_n, int64_t num_bits);

template <typename T>
llm_kernels::nvidia::marlin::WorkspaceInfo GetMarlinWorkspace(bool use_fp32_reduce, bool has_act_order, int rank,
                                                              int64_t size_m, int64_t size_k);

template <typename T>
void InvokeMarlinPermuteScales(cudaStream_t stream, const void* input, void* output, const size_t k, const size_t n,
                               const int64_t groupsize);

template <typename T>
void InvokeMarlinGemm(void* a, void* a_tmp, void* b_q_weight, void* b_scales, void* b_zeros, void* g_idx, void* perm,
                      void* workspace, void* c, void* c_tmp, int64_t size_m, int64_t size_n, int64_t size_k,
                      int64_t num_groups, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float,
                      bool has_zp, bool has_act_order, bool is_awq, int rank, cudaStream_t stream);

template <typename T>
torch::ScalarType GetTorchDataType();

DataType GetDataTypeFromTorchType(const c10::ScalarType& torch_type);

template <typename T, llm_kernels::nvidia::WeightType WT>
void GetFpAIntBGroupCutlassGemmWorkspaceSize(size_t m, size_t n, size_t k, size_t& ws_bytes);

template <typename T, llm_kernels::nvidia::WeightType WT>
void InvokeFpAIntBGroupCutlassGemm(void* output, const void* input, const void* weight, const void* scales,
                                   const void* zeros, void* ws, size_t m, size_t n, size_t k, size_t groupsize,
                                   size_t config_index, cudaStream_t stream);

template <typename T, llm_kernels::nvidia::WeightType WT>
size_t InvokeFpAIntBGroupCutlassGemmConfigProfile(size_t warmup, size_t iter, void* output, const void* input,
                                                  const void* weight, const void* scales, const void* zeros, void* ws,
                                                  size_t m, size_t n, size_t k, size_t groupsize, cudaStream_t stream);

template <typename T, llm_kernels::nvidia::WeightType WT>
bool GetFpAIntBGroupCudaGemmSupported();

template <typename T, llm_kernels::nvidia::WeightType WT>
void InvokeFpAIntBGroupCudaGemm(void* output, const void* input, const void* weight, const void* scales,
                                const void* zeros, size_t m, size_t n, size_t k, size_t groupsize, cudaStream_t stream);

// Invoke the lookup embedding.
template <typename T>
void LookupEmbedding(const void* input_ids, const void* ids_offsets, const void* prefix_offsets, const void* emb,
                     const void* pos, const void* steps, void* output, bool use_emb_scale, const T emb_scale,
                     int vocab_size, int hidden_size, int bs, int vocab_id, cudaStream_t stream,
                     void* workspace_ptr = nullptr);

// Layernorm without bias computes rmsnorm.
template <typename T>
void InvokeLayerNorm(const void* input, const void* weight, const void* bias, const float layernorm_eps, const int m,
                     const int n, void* output, cudaStream_t stream);

template <typename T>
void InvokeMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,
                  const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream, void* workspace_ptr,
                  cublasLtMatmulAlgo_t* cublaslt_algo, size_t workspace_size = 0);

template <typename T>
void InvokeBatchedMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int batch_size, int m, int n,
                         int k, const void* a_ptr, const void* b_ptr, void* c_ptr, cudaStream_t& stream,
                         void* workspace_ptr, size_t workspace_size, cublasLtMatmulAlgo_t* cublaslt_algo);

template <typename T>
void InvokeAddBiasResidual(const void* input_a, const void* input_b, const void* bias, const int m, const int n,
                           void* output, cudaStream_t stream);

// Add-Multiply fused operations
template <typename T>
void InvokeAddThenMul(const void* input1, const void* input2, const T scale, const int m, const int n,
                      void* output, cudaStream_t stream);

template <typename T>
void InvokeAddMulSecond(const void* input1, const void* input2, const T scale, const int m, const int n,
                        void* output, cudaStream_t stream);

template <typename T>
void InvokeAddBiasThenMul(const void* input1, const void* input2, const void* bias, const T scale,
                          const int m, const int n, void* output, cudaStream_t stream);

template <typename T>
void InvokeMulThenAdd(const void* input1, const void* input2, const T scale1, const T scale2,
                      const int m, const int n, void* output, cudaStream_t stream);

template <typename T>
void InvokeAddResidualsBiasThenMul(const void* input1, const void* residual1, const void* residual2,
                                   const void* bias, const T scale, const int m, const int n,
                                   void* output, cudaStream_t stream);

// for the tensor input with shape [m, n]
//  out = act(input[:, :n/2]) * input[:, n/2:]
template <template <typename T> class Activation, typename T>
void InvokeRowBasedGatedActivation(const void* input, const int m, const int n, void* output, cudaStream_t stream);

// Invoke activation in-place, `output` must be the same as `input`.
template <template <typename T> class Activation, typename T>
void InvokeGatedActivation(const void* input, const void* bias, const void* gated_weights, const void* gated_bias,
                           const int m, const int n, void* output, cudaStream_t stream);

template <typename T>
void AssembleTokensHidden(const void* inputs, const void* logits_idx, const int batch_size, const int hidden_units_num,
                          void* output, cudaStream_t& stream);

template <typename T>
void Concat(const void* input_a, const void* input_b, size_t concat_size_a, size_t concat_size_b, size_t outer_dim_size,
            size_t inner_dim_size, void* output, cudaStream_t& stream);

template <typename T>
void InvokeQKRmsNorm(void* qkv_ptr, const void* q_gamma, const void* k_gamma, const float layernorm_eps,
                     const int32_t total_tokens, const int32_t num_heads, const int32_t num_kv_heads,
                     const int32_t head_size, const int64_t* mask, cudaStream_t stream);

template <typename T>
void Expand(void* input, void* output, const int m, const int expand_size, const int n, const size_t stride,
            cudaStream_t stream);

template <typename T>
void CustomAllReduceInit(void** ptr, void* rank_data, size_t rank_data_sz, int cur_rank, int total_ranks,
                         bool is_full_nvlink, uint32_t root_rank);

template <typename T>
void CustomAllReduceRegisterSignalBuffer(void* ptr, void** signals);

template <typename T>
void CustomAllReduceRegisterBuffer(void* ptr, void** input_handles, cudaStream_t& stream);

template <typename T>
void CustomAllReduceRun(void* ptr, void* input, void* result, int data_size, cudaStream_t& stream);

template <typename T>
void InvokeSigmoidActivation(void* input, const size_t size, const float scale, cudaStream_t& stream);

template <typename T>
ncclDataType_t GetNcclDataType();

template <typename T>
void DataToFloat(const void* input, const int data_size, const size_t vocab_size, const size_t vocab_size_pad,
                 void* output, cudaStream_t& stream);

template <typename T>
void InvokePermute(void* input, void* output, std::vector<size_t> input_shape, std::vector<size_t> permutation,
                   cudaStream_t& stream);

template <typename T>
void InvokeMul(void* a, void* b, void* c, int m1, int n1, int m2, int n2, int device_rank);
// c = InvokeMul(a, b)
void InvokeMul(float* a, float* b, float* c, int n, int device_rank);
void Reciprocal(float* out, float* in, int n, int device_rank);
void Max(float* out, float* a, float* b, int n, int device_rank);

void CalcLogprobs(float* logits, float* temperatures, int vocab_size, int bs, int logprobs_num, float* logprobs,
                  int64_t* token_ids);

#ifdef ENABLE_FP8
template <typename T>
void Fp8E4m3Quantize(int num_channels, int channel_size, const T* input_ptr, void* quant_ptr, float* scale_ptr,
                     bool is_static, cudaStream_t& stream);

template <typename T>
void Fp8QuantizedMatMul(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle, int m, int n, int k,
                        const void* a_ptr, const void* a_scale, const void* b_ptr, const void* b_scale, T* c_ptr,
                        cudaStream_t& stream, cublasLtMatmulAlgo_t* cublaslt_algo, void* workspace,
                        size_t workspace_size = 0);

void RescaleFp8E4m3(void* input, void* output, size_t n, const float* input_scale, const float* output_scale,
                    cudaStream_t& stream);
template <typename T>
void ScaledQuantizeFp8E4m3(T* x, void* output, float* scale, std::vector<size_t> group_shape, int m, int n, int rank);

template <typename T>
void DequantFp8E4m3BlockWise(const void* d_data, const void* d_s, void* d_output, int m, int n, int block_size,
                             cudaStream_t& stream);
#endif

size_t InvokeGetCublasWorkspaceSize();

#if defined(ENABLE_VLLM_FLASH_ATTN_2) || defined(ENABLE_FLASH_ATTN_3)
cudaStream_t InvokeSetTorchStream(cudaStream_t& stream, int rank);
#endif

template <typename T>
void InvokeBlockGemm(void* a, float* a_scales, void* b, float* b_scales, void* output, int m, int k, int n,
                     cudaStream_t& stream, void* cutlass_buffer = nullptr, size_t cutlass_buffer_size = 0ul);

template <typename T>
void InvokePerTokenGroupQuantFp8E4m3(void* input, void* output_q, void* output_s, int m, int n, bool is_column_major,
                                     cudaStream_t stream, int64_t group_size = 128, float eps = 1e-10,
                                     float min_fp8 = -448.0, float max_fp8 = 448.0);

template <typename T>
void InvokeFusedAddRmsNorm(void* input, void* residual, void* weight, double eps, int m, int n, cudaStream_t stream);

template <typename T>
void InvokeSplit(const T* __restrict__ input, const std::vector<T*>& output_ptrs, std::vector<int>& col_offsets,
                 int rows, int cols, int num_outputs, cudaStream_t& stream);

template <typename T>
void InvokeGatherSubmatrix(const T* __restrict__ input, T* __restrict__ output, std::vector<size_t>& m_num_per_group,
                           size_t group_size, size_t group_num, size_t n_start, size_t n_end, size_t m, size_t n,
                           void* workspace, cudaStream_t& stream);

template <typename T>
void InvokeDpMapCopy(const T* __restrict__ input, T* __restrict__ output, const std::vector<size_t>& group_info,
                     size_t m, size_t n, void* workspace, cudaStream_t& stream);
void InvokeProcessKvList(void** kv_list, size_t layer_num, size_t block_num, size_t block_size, cudaStream_t stream);

// Apply token bitmask to logits for grammar-constrained sampling, setting masked tokens to negative infinity
// logits-输入输出的logits数据, bitmask-位掩码数组, indices-可选索引数组
// vocab_size-词汇表大小, logits_stride-logits行步长, bitmask_stride-掩码行步长, num_rows-批次大小
template <typename T>
void InvokeApplyTokenBitmaskInplace(void* logits, const void* bitmask, const void* indices, int32_t vocab_size,
                                    int32_t logits_stride, int32_t bitmask_stride, int32_t num_rows,
                                    cudaStream_t stream);

}  // namespace ksana_llm
