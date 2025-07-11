/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/gptq_marlin/marlin_wrapper.h"

#include "csrc/kernels/nvidia/cast/cast.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/machete/machete_wrapper.h"
#include "csrc/kernels/nvidia/permute/permute.h"

using namespace llm_kernels::nvidia;

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaMarlinTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    cublasCreate(&cublas_handle);
  }

  void TearDown() override {
    NvidiaTestSuitBase::TearDown();
    cublasDestroy(cublas_handle);
  }

 protected:
  using NvidiaTestSuitBase::stream;
  cublasHandle_t cublas_handle;
  const size_t warmup = 100;
  const size_t iters = 1000;

 protected:
  template <typename T>
  void GetGPTQInt4Ref(cublasHandle_t cublas_handle, cudaStream_t stream, void* d_a, void* d_w, void* d_s, void* d_o,
                      const size_t m, const size_t n, const size_t k, const size_t groupsize) {
    const size_t pack_factor = 32 / 4;

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
    if constexpr (std::is_same_v<T, half>) {
      HalfToFloat(reinterpret_cast<T*>(d_s), k / groupsize * n, reinterpret_cast<float*>(d_S.data_ptr), stream);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      BFloat16ToFloat(reinterpret_cast<T*>(d_s), k / groupsize * n, reinterpret_cast<float*>(d_S.data_ptr), stream);
    } else {
      KLLM_KERNEL_THROW("Not support type!");
    }
    machete::unpackScale(stream, reinterpret_cast<const float*>(d_S.data_ptr),
                         reinterpret_cast<float*>(d_unpacked_S.data_ptr), k, n, groupsize);
    machete::elementwiseMul(stream, reinterpret_cast<float*>(d_unpackedW.data_ptr),
                            reinterpret_cast<float*>(d_unpacked_S.data_ptr), reinterpret_cast<float*>(d_WS.data_ptr),
                            k * n);
    if constexpr (std::is_same_v<T, half>) {
      HalfToFloat(reinterpret_cast<T*>(d_a), m * k, reinterpret_cast<float*>(d_A.data_ptr), stream);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      BFloat16ToFloat(reinterpret_cast<T*>(d_a), m * k, reinterpret_cast<float*>(d_A.data_ptr), stream);
    } else {
      KLLM_KERNEL_THROW("Not support type!");
    }
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

  template <typename T>
  void TestMarlinInt4GPTQ(const size_t m, const size_t n, const size_t k, const int64_t groupsize) {
    // TODO:目前只做了4bit GPTQ的测试
    const size_t bits = 4;
    const size_t pack_factor = 32 / bits;

    // 生成随机数据
    BufferMeta a = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, k}, true);
    BufferMeta w_q = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {k / pack_factor, n}, true);
    BufferMeta w_s = CreateBuffer<T>(MemoryType::MEMORY_GPU, {k / groupsize, n}, true);

    // qweight预处理
    std::vector<int64_t> new_qweight_shape = llm_kernels::nvidia::marlin::gptq_marlin_repack_meta(k, n, bits);
    BufferMeta w_q_repack = CreateBuffer<int32_t>(
        MemoryType::MEMORY_GPU, {static_cast<size_t>(new_qweight_shape[0]), static_cast<size_t>(new_qweight_shape[1])},
        false);
    llm_kernels::nvidia::marlin::gptq_marlin_repack(reinterpret_cast<const uint32_t*>(w_q.data_ptr), nullptr,
                                                    reinterpret_cast<uint32_t*>(w_q_repack.data_ptr), 1, k, n, bits,
                                                    false, 0, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // scale预处理
    BufferMeta w_s_permute = CreateBuffer<T>(MemoryType::MEMORY_GPU, {k / groupsize, n}, false);
    llm_kernels::nvidia::marlin::permute_scales(stream, reinterpret_cast<const T*>(w_s.data_ptr),
                                                reinterpret_cast<T*>(w_s_permute.data_ptr), k, n, groupsize);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 开辟workspace
    llm_kernels::nvidia::marlin::WorkspaceInfo info =
        llm_kernels::nvidia::marlin::get_workspace<T>(true, false, 0, m, k);
    BufferMeta workspace = CreateBuffer<char>(MemoryType::MEMORY_GPU, {info.workspace_size}, false);
    BufferMeta a_tmp = CreateBuffer<char>(MemoryType::MEMORY_GPU, {info.a_tmp_size}, false);
    BufferMeta c_tmp = CreateBuffer<char>(MemoryType::MEMORY_GPU, {info.c_tmp_size}, false);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta device_D = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, false);
    auto cuda_run = [&]() {
      cudaMemsetAsync(workspace.data_ptr, 0, info.workspace_size, stream);
      llm_kernels::nvidia::marlin::gptq_marlin_gemm<T>(
          reinterpret_cast<void*>(a.data_ptr), reinterpret_cast<void*>(a_tmp.data_ptr),
          reinterpret_cast<void*>(w_q_repack.data_ptr), reinterpret_cast<void*>(w_s_permute.data_ptr), nullptr, nullptr,
          nullptr, reinterpret_cast<void*>(workspace.data_ptr), reinterpret_cast<void*>(device_D.data_ptr),
          reinterpret_cast<void*>(c_tmp.data_ptr), m, n, k, k / groupsize, true, true, true, false, false, false, false,
          0, stream);
    };
    float cost_time = MeasureCudaExecutionTime(cuda_run, stream, warmup, iters);

    printf("mnk=(%zu,%zu,%zu), gs:%zu, time:%f ms\n", m, n, k, groupsize, cost_time);

    BufferMeta d_R = CreateBuffer<float>(MemoryType::MEMORY_GPU, {m, n}, false);
    GetGPTQInt4Ref<T>(cublas_handle, stream, a.data_ptr, w_q.data_ptr, w_s.data_ptr, d_R.data_ptr, m, n, k, groupsize);

    // 全部统一到float精度上结果对比
    BufferMeta device_D_float = CreateBuffer<float>(MemoryType::MEMORY_GPU, {m, n}, false);
    if constexpr (std::is_same_v<T, half>) {
      HalfToFloat(reinterpret_cast<T*>(device_D.data_ptr), m * n, reinterpret_cast<float*>(device_D_float.data_ptr),
                  stream);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      BFloat16ToFloat(reinterpret_cast<T*>(device_D.data_ptr), m * n, reinterpret_cast<float*>(device_D_float.data_ptr),
                      stream);
    } else {
      KLLM_KERNEL_THROW("Not support type!");
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    if constexpr (std::is_same_v<T, half>) {
      EXPECT_TRUE(CheckResult<float>("TestMarlinInt4GPTQ", device_D_float, d_R, 1e-2f, 1e-5f, 0.01f, true));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      EXPECT_TRUE(CheckResult<float>("TestMarlinInt4GPTQ", device_D_float, d_R, 5e-2f, 1e-5f, 0.01f, true));
    } else {
      KLLM_KERNEL_THROW("Not support type!");
    }

    DeleteBuffer(a);
    DeleteBuffer(w_q);
    DeleteBuffer(w_s);
    DeleteBuffer(w_q_repack);
    DeleteBuffer(w_s_permute);
    DeleteBuffer(workspace);
    DeleteBuffer(device_D);
    DeleteBuffer(d_R);
    DeleteBuffer(device_D_float);
  }
};

TEST_F(NvidiaMarlinTestSuit, MarlinFP16Int4GPTQTest) {
  TestMarlinInt4GPTQ<half>(1, 4096, 4096, 128);
  TestMarlinInt4GPTQ<half>(16, 8192, 4096, 128);
  TestMarlinInt4GPTQ<half>(32, 8192, 28672, 128);

  TestMarlinInt4GPTQ<half>(1, 4096, 4096, 64);
  TestMarlinInt4GPTQ<half>(16, 8192, 4096, 64);
  TestMarlinInt4GPTQ<half>(32, 8192, 28672, 64);
}

TEST_F(NvidiaMarlinTestSuit, MarlinBF16Int4GPTQTest) {
  TestMarlinInt4GPTQ<__nv_bfloat16>(1, 4096, 4096, 128);
  TestMarlinInt4GPTQ<__nv_bfloat16>(16, 8192, 4096, 128);
  TestMarlinInt4GPTQ<__nv_bfloat16>(32, 8192, 28672, 128);

  TestMarlinInt4GPTQ<__nv_bfloat16>(1, 4096, 4096, 64);
  TestMarlinInt4GPTQ<__nv_bfloat16>(16, 8192, 4096, 64);
  TestMarlinInt4GPTQ<__nv_bfloat16>(32, 8192, 28672, 64);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
