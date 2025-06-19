/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"

#include "csrc/kernels/nvidia/asymmetric_gemm/asymmetric_gemm_wrapper.h"

#include "csrc/kernels/nvidia/weight_only_batched_gemv/weight_only_gemv_wrapper.h"

#include "csrc/kernels/nvidia/machete/machete_wrapper.h"

#include "csrc/kernels/nvidia/cast/cast.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/permute/permute.h"

using namespace llm_kernels::nvidia;

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaAsymmetricGemmTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
  }

  void TearDown() override {
    NvidiaTestSuitBase::TearDown();
    cublasDestroy(cublas_handle);
    cublasLtDestroy(cublaslt_handle);
  }

 protected:
  using NvidiaTestSuitBase::stream;
  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;

 protected:
  void getGPTQInt4Ref(cublasHandle_t cublas_handle, cudaStream_t stream, void* d_a, void* d_w, void* d_s, void* d_o,
                      const size_t m, const size_t n, const size_t k) {
    const size_t pack_factor = 32 / 4;
    const size_t groupsize = 128;

    float alpha = 1.0f;
    float beta = 0.0f;

    BufferMeta d_WT = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {k / pack_factor, n}, false);
    BufferMeta d_unpackedWT = CreateBuffer<float>(MemoryType::MEMORY_GPU, {k * n}, false);
    BufferMeta d_unpackedW = CreateBuffer<float>(MemoryType::MEMORY_GPU, {k * n}, false);
    BufferMeta d_S = CreateBuffer<float>(MemoryType::MEMORY_GPU, {k / groupsize, n}, false);
    BufferMeta d_unpacked_S = CreateBuffer<float>(MemoryType::MEMORY_GPU, {k, n}, false);
    BufferMeta d_WS = CreateBuffer<float>(MemoryType::MEMORY_GPU, {k, n}, false);
    BufferMeta d_A = CreateBuffer<float>(MemoryType::MEMORY_GPU, {m, k}, false);

    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    InvokePermute<2ul, sizeof(int32_t)>(d_w, d_WT.data_ptr, {k / pack_factor, n}, {1, 0}, stream);
    machete::unpackInt4x2<float>(stream, reinterpret_cast<const uint8_t*>(d_WT.data_ptr),
                                 reinterpret_cast<float*>(d_unpackedWT.data_ptr), k * n / 2);
    InvokePermute<2ul, sizeof(float)>(d_unpackedWT.data_ptr, d_unpackedW.data_ptr, {n, k}, {1, 0}, stream);
    HalfToFloat(reinterpret_cast<half*>(d_s), k / groupsize * n, reinterpret_cast<float*>(d_S.data_ptr), stream);
    machete::unpackScale(stream, reinterpret_cast<const float*>(d_S.data_ptr),
                         reinterpret_cast<float*>(d_unpacked_S.data_ptr), k, n, groupsize);
    machete::elementwiseMul(stream, reinterpret_cast<float*>(d_unpackedW.data_ptr),
                            reinterpret_cast<float*>(d_unpacked_S.data_ptr), reinterpret_cast<float*>(d_WS.data_ptr),
                            k * n);
    HalfToFloat(reinterpret_cast<half*>(d_a), m * k, reinterpret_cast<float*>(d_A.data_ptr), stream);

    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_NVIDIA_CUDA_ERROR(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, (const void*)&alpha,
                                         d_WS.data_ptr, CUDA_R_32F, n, d_A.data_ptr, CUDA_R_32F, k, (const void*)&beta,
                                         d_o, CUDA_R_32F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    DeleteBuffer(d_WT);
    DeleteBuffer(d_unpackedWT);
    DeleteBuffer(d_unpackedW);
    DeleteBuffer(d_S);
    DeleteBuffer(d_unpacked_S);
    DeleteBuffer(d_WS);
    DeleteBuffer(d_A);
  }

  void TestAsymmetricGemmCudaCutlass(const size_t m, const size_t n, const size_t k, const size_t groupsize) {
    BufferMeta buffer_cutlass_output = CreateBuffer<half>(MemoryType::MEMORY_GPU, {m, n}, false);
    BufferMeta buffer_cuda_output = CreateBuffer<half>(MemoryType::MEMORY_GPU, {m, n}, false);

    BufferMeta buffer_input = CreateBuffer<half>(MemoryType::MEMORY_GPU, {m, k}, true);
    BufferMeta buffer_qweight = CreateBuffer<char>(MemoryType::MEMORY_GPU, {k, n / 2}, true);
    BufferMeta buffer_weight_scales = CreateBuffer<half>(MemoryType::MEMORY_GPU, {k / groupsize, n}, true);

    // 执行cutlass gemm
    auto cutlass_gemm = std::make_shared<FpAIntBGroupCutlassGemmWrapper<half, WeightType::INT4>>();

    size_t ws_bytes;
    cutlass_gemm->GetWorkspaceSize(m, n, k, ws_bytes);
    BufferMeta buffer_ws = CreateBuffer<char>(MemoryType::MEMORY_GPU, {ws_bytes}, false);

    int best_config_index = cutlass_gemm->GetBestConfigIndex(
        10, 100, buffer_cutlass_output.data_ptr, buffer_input.data_ptr, buffer_qweight.data_ptr,
        buffer_weight_scales.data_ptr, nullptr, buffer_ws.data_ptr, m, n, k, groupsize, stream);

    auto cutlass_run = [&]() {
      cutlass_gemm->Gemm(buffer_cutlass_output.data_ptr, buffer_input.data_ptr, buffer_qweight.data_ptr,
                         buffer_weight_scales.data_ptr,
                         nullptr,  // no zeros
                         buffer_ws.data_ptr, m, n, k, groupsize, best_config_index, stream);
    };
    float cutlass_time = MeasureCudaExecutionTime(cutlass_run, stream);

    // 执行cuda gemm
    auto cuda_gemm = std::make_shared<FpAIntBGroupCudaGemmWrapper<half, WeightType::INT4>>();
    if (!cuda_gemm->IsSupport()) {
      throw std::runtime_error("Not support cuda kernel for type: FP16Int4Groupwise in current arch.");
    }

    auto cuda_run = [&]() {
      cuda_gemm->Gemm(buffer_cuda_output.data_ptr, buffer_input.data_ptr, buffer_qweight.data_ptr,
                      buffer_weight_scales.data_ptr,
                      nullptr,  // no zeros
                      m, n, k, groupsize, stream);
    };
    float cuda_time = MeasureCudaExecutionTime(cuda_run, stream);

    // 校验
    printf("TestAsymmetricGemmCudaCutlass[%zu,%zu,%zu] cuda time: %fms, cutlass time: %fms\n", m, n, k, cuda_time,
           cutlass_time);

    EXPECT_TRUE(cuda_time < cutlass_time);

    EXPECT_TRUE(CheckResult<half>("TestAsymmetricGemmCudaCutlass_m" + std::to_string(m) + "_n" + std::to_string(n) +
                                      "_k" + std::to_string(k) + "_g" + std::to_string(groupsize),
                                  buffer_cuda_output, buffer_cutlass_output, 1e-1f, 1e-5f));

    DeleteBuffer(buffer_ws);
    DeleteBuffer(buffer_cutlass_output);
    DeleteBuffer(buffer_cuda_output);
    DeleteBuffer(buffer_input);
    DeleteBuffer(buffer_qweight);
    DeleteBuffer(buffer_weight_scales);
  }

  void TestAsymmetricGemm(const size_t m, const size_t n, const size_t k, const size_t groupsize) {
    const size_t bits = 4;
    const size_t pack_factor = 32 / bits;
    QuantType quant_type = QuantType::W4_A16;

    BufferMeta activation = CreateBuffer<half>(MemoryType::MEMORY_GPU, {m, k}, true);
    BufferMeta qweight_int32 = CreateBuffer<uint32_t>(MemoryType::MEMORY_GPU, {k / pack_factor, n}, true);
    BufferMeta scales = CreateBuffer<half>(MemoryType::MEMORY_GPU, {k / groupsize, n}, true);

    // cutlass算子权重layout转换
    BufferMeta qweightT_int32 = CreateBuffer<uint32_t>(MemoryType::MEMORY_GPU, {n, k / pack_factor}, false);
    BufferMeta qweightT_int4 = CreateBuffer<char>(MemoryType::MEMORY_GPU, {n, k}, false);
    BufferMeta qweight_int4 = CreateBuffer<char>(MemoryType::MEMORY_GPU, {k, n}, false);
    BufferMeta qweight_int8 = CreateBuffer<char>(MemoryType::MEMORY_GPU, {k, n / 2}, false);
    InvokePermute<2ul, sizeof(int32_t)>(qweight_int32.data_ptr, qweightT_int32.data_ptr, {k / pack_factor, n}, {1, 0},
                                        stream);
    machete::unpackInt4x2<int8_t>(stream, reinterpret_cast<const uint8_t*>(qweightT_int32.data_ptr),
                                  reinterpret_cast<int8_t*>(qweightT_int4.data_ptr), k * n / 2);
    InvokePermute<2ul, sizeof(int8_t)>(qweightT_int4.data_ptr, qweight_int4.data_ptr, {n, k}, {1, 0}, stream);
    packInt4x2ToInt8(stream, reinterpret_cast<const int8_t*>(qweight_int4.data_ptr),
                     reinterpret_cast<int8_t*>(qweight_int8.data_ptr), k * n / 2);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    std::vector<int32_t> row_permutation = get_permutation_map(quant_type);
    BufferMeta device_row_permutation = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {row_permutation.size()}, false);
    cudaMemcpy(device_row_permutation.data_ptr, row_permutation.data(), device_row_permutation.buf_size,
               cudaMemcpyHostToDevice);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta qweight_prepeocess = CreateBuffer<char>(MemoryType::MEMORY_GPU, {k, n / 2}, false);
    fast_preprocess_weights_for_mixed_gemm(reinterpret_cast<int8_t*>(qweight_prepeocess.data_ptr),
                                           reinterpret_cast<int8_t*>(qweight_int8.data_ptr),
                                           reinterpret_cast<const int32_t*>(device_row_permutation.data_ptr),
                                           row_permutation.size(), {1, k, n}, quant_type, false, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 执行cutlass算子计算
    auto cutlass_gemm = std::make_shared<FpAIntBGroupCutlassGemmWrapper<half, WeightType::INT4>>();
    size_t ws_bytes;
    cutlass_gemm->GetWorkspaceSize(m, n, k, ws_bytes);
    BufferMeta buffer_ws = CreateBuffer<char>(MemoryType::MEMORY_GPU, {ws_bytes}, false);
    BufferMeta cutlass_output = CreateBuffer<half>(MemoryType::MEMORY_GPU, {m, n}, false);
    cutlass_gemm->Gemm(cutlass_output.data_ptr, activation.data_ptr, qweight_prepeocess.data_ptr, scales.data_ptr,
                       nullptr,  // no zeros
                       buffer_ws.data_ptr, m, n, k, groupsize, 0, stream);

    BufferMeta d_R = CreateBuffer<float>(MemoryType::MEMORY_GPU, {m, n}, false);
    getGPTQInt4Ref(cublas_handle, stream, activation.data_ptr, qweight_int32.data_ptr, scales.data_ptr, d_R.data_ptr, m,
                   n, k);

    // 统一到float精度上对比结果
    BufferMeta cutlass_output_float = CreateBuffer<float>(MemoryType::MEMORY_GPU, {m, n}, false);
    HalfToFloat(reinterpret_cast<half*>(cutlass_output.data_ptr), m * n,
                reinterpret_cast<float*>(cutlass_output_float.data_ptr), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    EXPECT_TRUE(CheckResult<float>("TestAsymmetricGemm", cutlass_output_float, d_R, 1e-2f, 1e-5f, 0.01f, true));
  }
};

TEST_F(NvidiaAsymmetricGemmTestSuit, AsymmetricGemmTest) {
  TestAsymmetricGemm(1, 5120, 5120, 128);
  TestAsymmetricGemm(1, 5120, 13824, 128);
  TestAsymmetricGemm(1, 13824, 5120, 128);
  TestAsymmetricGemm(1, 15360, 5120, 128);

  TestAsymmetricGemm(1, 5120, 2560, 128);
  TestAsymmetricGemm(1, 5120, 6912, 128);
  TestAsymmetricGemm(1, 6912, 5120, 128);
  TestAsymmetricGemm(1, 7680, 5120, 128);

  TestAsymmetricGemmCudaCutlass(1, 5120, 5120, 128);
  TestAsymmetricGemmCudaCutlass(1, 5120, 13824, 128);
  TestAsymmetricGemmCudaCutlass(1, 13824, 5120, 128);
  TestAsymmetricGemmCudaCutlass(1, 15360, 5120, 128);

  TestAsymmetricGemmCudaCutlass(1, 5120, 2560, 128);
  TestAsymmetricGemmCudaCutlass(1, 5120, 6912, 128);
  TestAsymmetricGemmCudaCutlass(1, 6912, 5120, 128);
  TestAsymmetricGemmCudaCutlass(1, 7680, 5120, 128);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels