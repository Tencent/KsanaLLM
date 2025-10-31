/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/layernorm/layernorm.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaLayernormTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<std::pair<int, int>> m_n_pairs = {{1, 2048}, {2, 4096}, {64000, 8}};
  ;

  template <typename T>
  void RunLayerNormRef(const float variance_epsilon = 1e-6f, const std::string test_mode = "default") {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    std::stringstream ss;
    ss << "python layernorm_test.py --type=" << type_str << " --variance_epsilon=" << std::to_string(variance_epsilon)
       << " --test_mode=" << test_mode;
    system(ss.str().c_str());
  }

  template <typename T>
  void TestLayerNorm(const size_t m, const size_t n) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                            /*is_random_init*/ true);
    BufferMeta weight_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {n},
                                             /*is_random_init*/ true);
    BufferMeta bias_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {n},
                                           /*is_random_init*/ true);
    BufferMeta layernorm_output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                       /*is_random_init*/ false);
    BufferMeta rmsnorm_output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                     /*is_random_init*/ false);
    BufferMeta layernorm_output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                           /*is_random_init*/ false);
    BufferMeta rmsnorm_output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                         /*is_random_init*/ false);
    float layernorm_eps = 1e-6;

    input_meta.SaveToNpy<T>("layernorm_test_input.npy");
    weight_meta.SaveToNpy<T>("layernorm_test_weight.npy");
    bias_meta.SaveToNpy<T>("layernorm_test_bias.npy");
    RunLayerNormRef<T>(layernorm_eps, "default");
    layernorm_output_ref_meta.LoadNpy<T>("layernorm_test_output.npy", MemoryType::MEMORY_GPU);
    rmsnorm_output_ref_meta.LoadNpy<T>("rmsnorm_test_output.npy", MemoryType::MEMORY_GPU);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    InvokeLayerNorm<T>(reinterpret_cast<T*>(layernorm_output_meta.data_ptr),
                       reinterpret_cast<const T*>(input_meta.data_ptr),
                       reinterpret_cast<const T*>(weight_meta.data_ptr),
                       reinterpret_cast<const T*>(bias_meta.data_ptr),
                       layernorm_eps, m, n, stream);
    // compute rmsnorm if bias is none
    InvokeRMSNorm<T>(reinterpret_cast<T*>(rmsnorm_output_meta.data_ptr), reinterpret_cast<T*>(input_meta.data_ptr),
                     reinterpret_cast<T*>(weight_meta.data_ptr), layernorm_eps, m, n, GetEnablePDL(), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    float tolerance = 1e-5f;
    if (std::is_same<T, half>::value) {
      tolerance = 1e-3f;
    }
    EXPECT_TRUE(CheckResult<T>("layernorm_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               layernorm_output_ref_meta, layernorm_output_meta, tolerance, tolerance));
    EXPECT_TRUE(CheckResult<T>("rmsnorm_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               rmsnorm_output_ref_meta, rmsnorm_output_meta, tolerance, tolerance));

    auto cuda_run = [&]() {
      InvokeLayerNorm<T>(reinterpret_cast<T*>(layernorm_output_meta.data_ptr),
                         reinterpret_cast<const T*>(input_meta.data_ptr),
                         reinterpret_cast<const T*>(weight_meta.data_ptr),
                         reinterpret_cast<const T*>(bias_meta.data_ptr), layernorm_eps, m, n, stream);
    };
    float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 100);
    auto cuda_run_rmsnorm = [&]() {
      InvokeRMSNorm<T>(reinterpret_cast<T*>(rmsnorm_output_meta.data_ptr), reinterpret_cast<T*>(input_meta.data_ptr),
                       reinterpret_cast<T*>(weight_meta.data_ptr), layernorm_eps, m, n, GetEnablePDL(), stream);
    };
    float milliseconds_rmsnorm = MeasureCudaExecutionTime(cuda_run_rmsnorm, stream, 10, 100);
    std::cout << std::left << type_str << " m=" << std::setw(6) << m << " n=" << std::setw(6) << n
              << " execution 1 times : " << "layernorm " << std::setw(10) << milliseconds << " ms " << "| rmsnorm "
              << milliseconds_rmsnorm << " ms" << std::endl;
    DeleteBuffer(rmsnorm_output_ref_meta);
    DeleteBuffer(layernorm_output_ref_meta);
    DeleteBuffer(rmsnorm_output_meta);
    DeleteBuffer(layernorm_output_meta);
    DeleteBuffer(bias_meta);
    DeleteBuffer(weight_meta);
    DeleteBuffer(input_meta);
  }

  template <typename T>
  void TestLayerNorm3D(const size_t l, const size_t m, const size_t n) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {l, m, n},
                                            /*is_random_init*/ true);
    BufferMeta weight_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {n},
                                             /*is_random_init*/ true);
    BufferMeta rmsnorm_3d_output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {l, m, n},
                                                            /*is_random_init*/ false);
    BufferMeta d_mask = CreateBuffer<int64_t>(MemoryType::MEMORY_GPU, {l}, /*is_random_init*/ false);
    std::vector<int64_t> h_mask(l, 1);
    std::fill(h_mask.begin(), h_mask.begin() + 2, 0);
    cudaMemcpy(d_mask.data_ptr, h_mask.data(), l * sizeof(int64_t), cudaMemcpyHostToDevice);

    float layernorm_eps = 1e-6;

    input_meta.SaveToNpy<T>("rmsnorm_3d_test_input.npy");
    weight_meta.SaveToNpy<T>("rmsnorm_3d_test_weight.npy");
    d_mask.SaveNpy<int64_t>("rmsnorm_3d_test_mask.npy");
    RunLayerNormRef<T>(layernorm_eps, "use_layernorm_3d");
    rmsnorm_3d_output_ref_meta.LoadNpy<T>("rmsnorm_3d_test_torch_output.npy", MemoryType::MEMORY_GPU);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    // compute 3d rmsnorm
    InvokeRmsNorm3D<T>(reinterpret_cast<T*>(input_meta.data_ptr), reinterpret_cast<const T*>(input_meta.data_ptr),
                       reinterpret_cast<const T*>(weight_meta.data_ptr), layernorm_eps, l, m, n, 0, 2,
                       reinterpret_cast<int64_t*>(d_mask.data_ptr), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_TRUE(CheckResult<T>(
        "rmsnorm_3d_" + type_str + "_l_" + std::to_string(l) + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
        rmsnorm_3d_output_ref_meta, input_meta, 1e-3f, 1e-3f));
    input_meta.SaveToNpy<T>("rmsnorm_3d_test_ksana_output.npy");

    auto cuda_run = [&]() {
      InvokeRmsNorm3D<T>(reinterpret_cast<T*>(input_meta.data_ptr), reinterpret_cast<const T*>(input_meta.data_ptr),
                         reinterpret_cast<const T*>(weight_meta.data_ptr), layernorm_eps, l, m, n, 0, 2,
                         reinterpret_cast<int64_t*>(d_mask.data_ptr), stream);
    };
    float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 100);
    std::cout << std::left << type_str << " l=" << std::setw(6) << l << " m=" << std::setw(6) << m
              << " n=" << std::setw(6) << n << " execution 1 times : "
              << "rmsnorm_3d " << std::setw(10) << milliseconds << " ms" << std::endl;
    DeleteBuffer(rmsnorm_3d_output_ref_meta);
    DeleteBuffer(d_mask);
    DeleteBuffer(weight_meta);
    DeleteBuffer(input_meta);
  }

  template <typename T>
  void TestLayerNormFusedQKV(const size_t l, const size_t num_heads, const size_t num_kv_heads, const size_t n) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {l, num_heads + 2 * num_kv_heads, n},
                                            /*is_random_init*/ true);
    BufferMeta weight_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {n},
                                             /*is_random_init*/ true);
    BufferMeta rmsnorm_fused_qkv_output_ref_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {l, num_heads + 2 * num_kv_heads, n},
                        /*is_random_init*/ false);
    BufferMeta d_mask = CreateBuffer<int64_t>(MemoryType::MEMORY_GPU, {l}, /*is_random_init*/ false);
    std::vector<int64_t> h_mask(l, 1);
    std::fill(h_mask.begin(), h_mask.begin() + 2, 0);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_mask.data_ptr, h_mask.data(), l * sizeof(int64_t), cudaMemcpyHostToDevice));

    float layernorm_eps = 1e-6;

    input_meta.SaveToNpy<T>("rmsnorm_fused_qkv_test_input.npy");
    weight_meta.SaveToNpy<T>("rmsnorm_fused_qkv_test_weight.npy");
    d_mask.SaveNpy<int64_t>("rmsnorm_fused_qkv_test_mask.npy");
    RunLayerNormRef<T>(layernorm_eps, "use_layernorm_fused_qkv");
    rmsnorm_fused_qkv_output_ref_meta.LoadNpy<T>("rmsnorm_fused_qkv_test_torch_output.npy", MemoryType::MEMORY_GPU);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    // compute rmsnorm fused qkv
    InvokeFusedQKVRmsNorm<T>(reinterpret_cast<T*>(input_meta.data_ptr), reinterpret_cast<const T*>(input_meta.data_ptr),
                             reinterpret_cast<const T*>(weight_meta.data_ptr),
                             reinterpret_cast<const T*>(weight_meta.data_ptr), layernorm_eps, l, num_heads,
                             num_kv_heads, n, reinterpret_cast<int64_t*>(d_mask.data_ptr), stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    std::string check_msg_str = "rmsnorm_3d_" + type_str + "_l_" + std::to_string(l) + "_m_" +
                                std::to_string(num_heads + 2 * num_kv_heads) + "_n_" + std::to_string(n);
    EXPECT_TRUE(CheckResult<T>(check_msg_str, rmsnorm_fused_qkv_output_ref_meta, input_meta, 1e-3f, 1e-3f));
    input_meta.SaveToNpy<T>("rmsnorm_fused_qkv_test_ksana_output.npy");

    auto cuda_run = [&]() {
      InvokeFusedQKVRmsNorm<T>(
          reinterpret_cast<T*>(input_meta.data_ptr), reinterpret_cast<const T*>(input_meta.data_ptr),
          reinterpret_cast<const T*>(weight_meta.data_ptr), reinterpret_cast<const T*>(weight_meta.data_ptr),
          layernorm_eps, l, num_heads, num_kv_heads, n, reinterpret_cast<int64_t*>(d_mask.data_ptr), stream);
    };
    float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 100);
    std::cout << std::left << type_str << " l=" << std::setw(6) << l << " m=" << std::setw(6)
              << num_heads + 2 * num_kv_heads << " n=" << std::setw(6) << n << " execution 1 times : "
              << "rmsnorm_fused_qkv " << std::setw(10) << milliseconds << " ms" << std::endl;
    DeleteBuffer(rmsnorm_fused_qkv_output_ref_meta);
    DeleteBuffer(d_mask);
    DeleteBuffer(weight_meta);
    DeleteBuffer(input_meta);
  }
};

TEST_F(LlamaNvidiaLayernormTestSuit, HalfLayernormCommonTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestLayerNorm<half>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(LlamaNvidiaLayernormTestSuit, FloatLayernormCommonTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestLayerNorm<float>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(LlamaNvidiaLayernormTestSuit, Bf16LayernormCommonTest) {
  for (const auto& m_n_pair : m_n_pairs) {
    TestLayerNorm<__nv_bfloat16>(static_cast<size_t>(m_n_pair.first), static_cast<size_t>(m_n_pair.second));
  }
}

TEST_F(LlamaNvidiaLayernormTestSuit, HalfLayernorm3DTest) { TestLayerNorm3D<half>(5, 5, 6); }

TEST_F(LlamaNvidiaLayernormTestSuit, FloatLayernorm3DTest) { TestLayerNorm3D<float>(5, 5, 6); }

TEST_F(LlamaNvidiaLayernormTestSuit, Bf16Layernorm3DTest) { TestLayerNorm3D<__nv_bfloat16>(5, 5, 6); }

TEST_F(LlamaNvidiaLayernormTestSuit, HalfLayernormFusedQKVTest) { TestLayerNormFusedQKV<half>(5, 2, 2, 6); }

TEST_F(LlamaNvidiaLayernormTestSuit, FloatLayernormFusedQKVTest) { TestLayerNormFusedQKV<float>(5, 2, 2, 6); }

TEST_F(LlamaNvidiaLayernormTestSuit, Bf16LayernormFusedQKVTest) { TestLayerNormFusedQKV<__nv_bfloat16>(5, 2, 2, 6); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
