/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/fused_add_norm/fused_add_norm.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaFusedAddNormTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    cudaDriverGetVersion(&cuda_driver_version_);
  }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<std::pair<int, int>> m_n_pairs = {{1, 2048}, {2, 4096}, {64000, 8}};;
  int cuda_driver_version_;
  const int MIN_CUDA_DRIVER_VERSION = 12000;  // Minimum required CUDA version for fused_add_norm

  template <typename T>
  void RunAddNormRef(const float variance_epsilon = 1e-6f) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    std::stringstream ss;
    ss << "python fused_add_norm_test.py --type=" << type_str
       << " --variance_epsilon=" << std::to_string(variance_epsilon);
    system(ss.str().c_str());
  }

  template <typename T>
  void TestAddNorm(const size_t m, const size_t n) {
    if (cuda_driver_version_ <= MIN_CUDA_DRIVER_VERSION) {
      GTEST_SKIP() << "Skipping fused_add_norm test for CUDA driver version " << cuda_driver_version_
                   << ", requires at least " << MIN_CUDA_DRIVER_VERSION;
    }
    std::string type_str = "float";
    float tol = 1e-5f;
    if (std::is_same<T, half>::value) {
      type_str = "half";
      tol = 1e-3f;  // half precision has higher tolerance
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
      tol = 1e-3f;  // __nv_bfloat16 precision has higher tolerance
    }

    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                            /*is_random_init*/ true);
    BufferMeta residual_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                            /*is_random_init*/ true);
    BufferMeta weight_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {n},
                                             /*is_random_init*/ true);
    BufferMeta rmsnorm_output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                         /*is_random_init*/ false);
    BufferMeta rmsnorm_residual_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                           /*is_random_init*/ false);
    float norm_eps = 1e-6;

    input_meta.SaveToNpy<T>("add_norm_test_input.npy");
    weight_meta.SaveToNpy<T>("add_norm_test_weight.npy");
    residual_meta.SaveToNpy<T>("add_norm_test_residual.npy");

    RunAddNormRef<T>(norm_eps);
    rmsnorm_output_ref_meta.LoadNpy<T>("add_norm_test_output.npy", MemoryType::MEMORY_GPU);
    rmsnorm_residual_ref_meta.LoadNpy<T>("add_norm_test_residual.npy", MemoryType::MEMORY_GPU);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    InvokeFusedAddRMSNorm<T>(input_meta.data_ptr, residual_meta.data_ptr, weight_meta.data_ptr, norm_eps,
                             /*enable_pdl*/ false, m, n, stream);

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_TRUE(CheckResult<T>("rmsnorm_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               rmsnorm_output_ref_meta, input_meta, tol, tol));
    EXPECT_TRUE(CheckResult<T>("rmsnorm_residual" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               rmsnorm_residual_ref_meta, residual_meta, tol, tol));

    auto cuda_run = [&]() {
      InvokeFusedAddRMSNorm<T>(input_meta.data_ptr, residual_meta.data_ptr, weight_meta.data_ptr, norm_eps,
                               /*enable_pdl*/ true, m, n, stream);
    };
    float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 100);
    std::cout << std::left << "Fused Add RMSNorm " << type_str << " m=" << std::setw(6) << m << " n=" << std::setw(6)
              << n << " execution 1 times " << std::setw(10) << milliseconds << " ms" << std::endl;
    DeleteBuffer(rmsnorm_output_ref_meta);
    DeleteBuffer(weight_meta);
    DeleteBuffer(input_meta);
    DeleteBuffer(residual_meta);
  }
};

TEST_F(LlamaNvidiaFusedAddNormTestSuit, HalfAddNormCommonTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestAddNorm<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(LlamaNvidiaFusedAddNormTestSuit, FloatAddNormCommonTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestAddNorm<float>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(LlamaNvidiaFusedAddNormTestSuit, Bf16AddNormCommonTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestAddNorm<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
