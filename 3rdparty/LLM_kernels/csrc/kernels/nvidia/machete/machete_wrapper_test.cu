/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/machete/machete_wrapper.h"

#include "csrc/kernels/nvidia/cast/cast.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/permute/permute.h"

using namespace llm_kernels::nvidia;

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaMacheteTestSuit : public NvidiaTestSuitBase {
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
  void TestMacheteInt4GPTQ(const size_t m, const size_t n, const size_t k, const size_t groupsize) {
    vllm_dtype::ScalarType::Id activation_type_id;
    if constexpr (std::is_same_v<T, half>) {
      activation_type_id = vllm_dtype::kHalf.id();
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      activation_type_id = vllm_dtype::kBFloat16.id();
    } else {
      KLLM_KERNEL_THROW("Not support type!");
    }

    // TODO:目前只做了4bit GPTQ的测试
    const size_t bits = 4;
    const size_t pack_factor = 32 / bits;

    // 生成随机数据
    BufferMeta a = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, k}, true);
    BufferMeta w_q_src = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {k / pack_factor, n}, true);
    BufferMeta w_g_s = CreateBuffer<T>(MemoryType::MEMORY_GPU, {k / groupsize, n}, true);

    // 测试schedules
    std::vector<std::string> schedules =
        machete::machete_supported_schedules(vllm_dtype::ScalarType::from_id(activation_type_id), vllm_dtype::kU4B8,
                                             vllm_dtype::ScalarType::from_id(activation_type_id), std::nullopt);
    printf("supported_schedules: %zu\n", schedules.size());

    // 权重预处理
    // preprack需要的B是列主序的，所以用转置处理一下
    BufferMeta w_q_col = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {k / pack_factor, n}, false);
    BufferMeta w_q = CreateBuffer<int32_t>(MemoryType::MEMORY_GPU, {k / pack_factor, n}, false);
    InvokePermute<2ul, sizeof(int32_t)>(w_q_src.data_ptr, w_q_col.data_ptr, {k / pack_factor, n}, {1, 0}, stream);
    machete::machete_prepack_weight(reinterpret_cast<const void*>(w_q_col.data_ptr), {k / pack_factor, n},
                                    reinterpret_cast<void*>(w_q.data_ptr),
                                    vllm_dtype::ScalarType::from_id(activation_type_id), vllm_dtype::kU4B8,
                                    vllm_dtype::ScalarType::from_id(activation_type_id), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta device_D = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, false);
    {
      // 第一次跑确定workspace，workspace_size==-1时，不会真实计算mm
      int64_t workspace_size = -1;
      machete::machete_gemm(
          workspace_size, nullptr, stream, m, n, k, reinterpret_cast<const void*>(a.data_ptr),
          reinterpret_cast<const void*>(w_q.data_ptr), reinterpret_cast<void*>(device_D.data_ptr),
          vllm_dtype::ScalarType::from_id(activation_type_id), vllm_dtype::kU4B8, reinterpret_cast<void*>(w_g_s.data_ptr),
          std::optional<std::vector<size_t>>({k / groupsize, n}), vllm_dtype::ScalarType::from_id(activation_type_id),
          std::nullopt, std::nullopt, std::nullopt, groupsize, std::nullopt);
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      // 开辟workspace
      BufferMeta device_workspace =
          CreateBuffer<char>(MemoryType::MEMORY_GPU, {static_cast<size_t>(workspace_size)}, false);

      // 真实mm计算
      auto cuda_run = [&]() {
        machete::machete_gemm(
            workspace_size, reinterpret_cast<void*>(device_workspace.data_ptr), stream, m, n, k,
            reinterpret_cast<const void*>(a.data_ptr), reinterpret_cast<const void*>(w_q.data_ptr),
            reinterpret_cast<void*>(device_D.data_ptr), vllm_dtype::ScalarType::from_id(activation_type_id),
            vllm_dtype::kU4B8, reinterpret_cast<void*>(w_g_s.data_ptr),
            std::optional<std::vector<size_t>>({k / groupsize, n}), vllm_dtype::ScalarType::from_id(activation_type_id),
            std::nullopt, std::nullopt, std::nullopt, groupsize, std::nullopt);
      };
      float schedule_time = MeasureCudaExecutionTime(cuda_run, stream, warmup, iters);

      // 卸载workspace
      DeleteBuffer(device_workspace);

      printf("mnk=(%zu,%zu,%zu), default schedule, workspace_size:%ld, time:%f ms\n", m, n, k, workspace_size,
             schedule_time);
    }
    for (std::string schedule : schedules) {
      // 第一次跑确定workspace，workspace_size==-1时，不会真实计算mm
      int64_t workspace_size = -1;
      machete::machete_gemm(
          workspace_size, nullptr, stream, m, n, k, reinterpret_cast<const void*>(a.data_ptr),
          reinterpret_cast<const void*>(w_q.data_ptr), reinterpret_cast<void*>(device_D.data_ptr),
          vllm_dtype::ScalarType::from_id(activation_type_id), vllm_dtype::kU4B8, reinterpret_cast<void*>(w_g_s.data_ptr),
          std::optional<std::vector<size_t>>({k / groupsize, n}), vllm_dtype::ScalarType::from_id(activation_type_id),
          std::nullopt, std::nullopt, std::nullopt, groupsize, schedule);
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      // 开辟workspace
      BufferMeta device_workspace =
          CreateBuffer<char>(MemoryType::MEMORY_GPU, {static_cast<size_t>(workspace_size)}, false);

      // 真实mm计算
      auto cuda_run = [&]() {
        machete::machete_gemm(
            workspace_size, reinterpret_cast<void*>(device_workspace.data_ptr), stream, m, n, k,
            reinterpret_cast<const void*>(a.data_ptr), reinterpret_cast<const void*>(w_q.data_ptr),
            reinterpret_cast<void*>(device_D.data_ptr), vllm_dtype::ScalarType::from_id(activation_type_id),
            vllm_dtype::kU4B8, reinterpret_cast<void*>(w_g_s.data_ptr),
            std::optional<std::vector<size_t>>({k / groupsize, n}), vllm_dtype::ScalarType::from_id(activation_type_id),
            std::nullopt, std::nullopt, std::nullopt, groupsize, schedule);
      };
      float schedule_time = MeasureCudaExecutionTime(cuda_run, stream, warmup, iters);

      // 卸载workspace
      DeleteBuffer(device_workspace);

      printf("mnk=(%zu,%zu,%zu), gs:%zu, schedule:(%s), workspace_size:%ld, time:%f ms\n", m, n, k, groupsize,
             schedule.c_str(), workspace_size, schedule_time);
    }

    BufferMeta d_R = CreateBuffer<float>(MemoryType::MEMORY_GPU, {m, n}, false);
    GetGPTQInt4Ref<T>(cublas_handle, stream, a.data_ptr, w_q_src.data_ptr, w_g_s.data_ptr, d_R.data_ptr, m, n, k,
                      groupsize);

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
      EXPECT_TRUE(CheckResult<float>("TestMacheteInt4GPTQ", device_D_float, d_R, 1e-2f, 1e-5f));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      EXPECT_TRUE(CheckResult<float>("TestMacheteInt4GPTQ", device_D_float, d_R, 5e-2f, 1e-5f));
    } else {
      KLLM_KERNEL_THROW("Not support type!");
    }

    DeleteBuffer(a);
    DeleteBuffer(w_q_src);
    DeleteBuffer(w_g_s);
    DeleteBuffer(w_q_col);
    DeleteBuffer(w_q);
    DeleteBuffer(device_D);
    DeleteBuffer(d_R);
    DeleteBuffer(device_D_float);
  }
};

TEST_F(NvidiaMacheteTestSuit, MacheteFP16Int4GPTQTest) {
#if defined(ENABLE_MACHETE)
  TestMacheteInt4GPTQ<half>(1, 4096, 4096, 128);
  TestMacheteInt4GPTQ<half>(16, 8192, 4096, 128);
  TestMacheteInt4GPTQ<half>(32, 8192, 28672, 128);

  TestMacheteInt4GPTQ<half>(1, 4096, 4096, 64);
  TestMacheteInt4GPTQ<half>(16, 8192, 4096, 64);
  TestMacheteInt4GPTQ<half>(32, 8192, 28672, 64);
#else
  std::cerr << "SM version is lower than 90. skipping Machete Kernel." << std::endl;
#endif
}

TEST_F(NvidiaMacheteTestSuit, MacheteBF16Int4GPTQTest) {
#if defined(ENABLE_MACHETE)
  TestMacheteInt4GPTQ<__nv_bfloat16>(1, 4096, 4096, 128);
  TestMacheteInt4GPTQ<__nv_bfloat16>(16, 8192, 4096, 128);
  TestMacheteInt4GPTQ<__nv_bfloat16>(32, 8192, 28672, 128);

  TestMacheteInt4GPTQ<__nv_bfloat16>(1, 4096, 4096, 64);
  TestMacheteInt4GPTQ<__nv_bfloat16>(16, 8192, 4096, 64);
  TestMacheteInt4GPTQ<__nv_bfloat16>(32, 8192, 28672, 64);
#else
  std::cerr << "SM version is lower than 90. skipping Machete Kernel." << std::endl;
#endif
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels