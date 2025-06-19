/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "csrc/utils/nvidia/scalar_type.hpp"

namespace llm_kernels {
namespace nvidia {
namespace machete {

template <typename T>
void unpackInt4x2(cudaStream_t stream, const uint8_t* packed, T* unpacked, const size_t len);

void unpackScale(cudaStream_t stream, const float* packed, float* unpacked, size_t k, size_t n, size_t groupsize);

void elementwiseMul(cudaStream_t stream, const float* A, const float* B, float* C, size_t len);

/**
 * @brief 获取给定数据类型组合支持的调度策略列表
 *
 * 该函数根据提供的激活类型、权重类型、缩放因子类型和零点类型，
 * 返回machete库支持的所有调度策略名称列表。
 *
 * 示例：对于典型的float16+4bitGPTQ量化模型：
 * - a_type: half (激活为半精度浮点数)
 * - b_type: kU4B8 (权重为4bit量化)
 * - maybe_group_scales_type: half (缩放因子为半精度浮点数)
 * - maybe_group_zeros_type: std::nullopt (无零点)
 *
 * @param a_type 激活张量的数据类型
 * @param b_type 权重张量的数据类型
 * @param maybe_group_scales_type 量化缩放因子的数据类型，可选参数
 * @param maybe_group_zeros_type 量化零点的数据类型，可选参数
 * @return std::vector<std::string> 支持的调度策略名称列表
 */
std::vector<std::string> machete_supported_schedules(vllm_dtype::ScalarType a_type, vllm_dtype::ScalarType b_type,
                                                     std::optional<vllm_dtype::ScalarType> maybe_group_scales_type,
                                                     std::optional<vllm_dtype::ScalarType> maybe_group_zeros_type);

/**
 * @brief 执行machete的通用矩阵乘法(GEMM)操作
 *
 * 该函数执行高性能的矩阵乘法操作，支持各种数据类型和量化格式。
 * 函数有两种工作模式：
 * 1. 当workspace_size为-1时，函数不执行实际计算，仅计算所需的workspace大小并返回
 * 2. 当workspace_size不为0时，函数执行实际的矩阵乘法计算
 *
 * 注意：Bptr指向的权重数据必须先通过machete_prepack_weight函数进行预处理转换
 *
 * @param workspace_size 输入/输出参数，当为-1时计算所需workspace大小并返回；不为0时表示提供的workspace大小
 * @param workspace 指向工作空间内存的指针，用于临时计算
 * @param stream CUDA流，用于异步执行计算操作
 * @param M GEMM操作的M维度大小（输出矩阵的行数）
 * @param N GEMM操作的N维度大小（输出矩阵的列数）
 * @param K GEMM操作的K维度大小（内部乘法维度）
 * @param Aptr 指向激活数据的指针
 * @param Bptr 指向预处理后权重数据的指针（必须先使用machete_prepack_weight处理）
 * @param Dptr 指向输出结果的指针
 * @param a_type 激活张量的数据类型
 * @param b_type 权重张量的数据类型
 * @param maybe_group_scales_ptr 可选的量化缩放因子指针
 * @param maybe_group_scales_shape 可选的量化缩放因子张量形状
 * @param maybe_group_scales_type 可选的量化缩放因子数据类型
 * @param maybe_group_zeros_ptr 可选的量化零点指针
 * @param maybe_group_zeros_shape 可选的量化零点张量形状
 * @param maybe_group_zeros_type 可选的量化零点数据类型
 * @param maybe_group_size 可选的量化分组大小，用于分组量化算法
 * @param maybe_schedule 可选的调度策略，可以为空或使用machete_supported_schedules返回的策略之一
 */
void machete_gemm(int64_t& workspace_size, void* workspace, cudaStream_t stream, int M, int N, int K, const void* Aptr,
                  const void* Bptr, void* Dptr, vllm_dtype::ScalarType const& a_type,
                  vllm_dtype::ScalarType const& b_type, std::optional<void*> const& maybe_group_scales_ptr,
                  std::optional<std::vector<size_t>> const& maybe_group_scales_shape,
                  std::optional<vllm_dtype::ScalarType> const& maybe_group_scales_type,
                  std::optional<void*> const& maybe_group_zeros_ptr,
                  std::optional<std::vector<size_t>> const& maybe_group_zeros_shape,
                  std::optional<vllm_dtype::ScalarType> const& maybe_group_zeros_type,
                  std::optional<int64_t> maybe_group_size, std::optional<std::string> maybe_schedule);

/**
 * @brief 将权重转换为machete需要的内存布局
 *
 * 该函数将输入权重转换为machete库所需的特定内存布局格式，以优化计算性能。
 * 函数不支持原地操作，输入和输出必须使用不同的内存空间。
 *
 * @param B_ptr 指向输入权重数据的指针
 * @param B_shape 输入权重张量的形状
 * @param out_ptr 指向输出预处理权重的指针（必须与输入指针不同）
 * @param a_type 激活张量的数据类型
 * @param b_type 权重张量的数据类型
 * @param maybe_group_scales_type 量化缩放因子的数据类型，可选参数
 * @param stream CUDA流，用于异步执行预处理操作
 */
void machete_prepack_weight(const void* B_ptr, const std::vector<size_t>& B_shape, void* out_ptr,
                            vllm_dtype::ScalarType const& a_type, vllm_dtype::ScalarType const& b_type,
                            std::optional<vllm_dtype::ScalarType> const& maybe_group_scales_type, cudaStream_t stream);

std::string machete_best_schedule(size_t warmup_iters, size_t record_iters, void* workspace, cudaStream_t stream, int M,
                                  int N, int K, const void* Aptr, const void* Bptr, void* Dptr,
                                  vllm_dtype::ScalarType const& a_type, vllm_dtype::ScalarType const& b_type,
                                  std::optional<void*> const& maybe_group_scales_ptr,
                                  std::optional<std::vector<size_t>> const& maybe_group_scales_shape,
                                  std::optional<vllm_dtype::ScalarType> const& maybe_group_scales_type,
                                  std::optional<void*> const& maybe_group_zeros_ptr,
                                  std::optional<std::vector<size_t>> const& maybe_group_zeros_shape,
                                  std::optional<vllm_dtype::ScalarType> const& maybe_group_zeros_type,
                                  std::optional<int64_t> maybe_group_size);

}  // namespace machete
}  // namespace nvidia
}  // namespace llm_kernels
