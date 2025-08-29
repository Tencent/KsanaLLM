/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>
#include <algorithm>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/adjust_mem/adjust_mem.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class NvidiaExtractMatrixTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  size_t m = 1024;
  size_t n = 128 * 128;
  size_t input_offset = 32 * 128;
  size_t output_n = 16 * 128;


 protected:
  template <typename T>
  void RunRef(size_t m, size_t n, size_t input_offset, size_t output_n, const std::string& type_str) {
    std::stringstream ss;
    ss << "python ./adjust_mem_test.py --type=" << type_str << " --m=" << m
       << " --n=" << n << " --input_offset=" << input_offset << " --output_n=" << output_n
       << " --test_func=InvokeExtractSubMatrix ";
    system(ss.str().c_str());
  }

  template <typename T>
  void TestGahterSubmatirx(cudaStream_t stream) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                            /*is_random_init*/ true);
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, output_n},
                                             /*is_random_init*/ false);

    InvokeExtractSubMatrix<T>(reinterpret_cast<T*>(input_meta.data_ptr) + input_offset, reinterpret_cast<T*>(output_meta.data_ptr),
                             m, n, output_n, stream);

    input_meta.SaveToNpy<T>("extract_submatrix_input.npy");
    RunRef<T>(m, n, input_offset, output_n, type_str);
    BufferMeta output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, output_n},
                                                 /*is_random_init*/ false);
    output_ref_meta.LoadNpy<T>("extract_submatrix_output.npy");
    EXPECT_TRUE(CheckResult<T>("extract_submatrix_test_" + type_str, output_ref_meta, output_meta, 1e-5f, 1e-5f, 0.0f));
    DeleteBuffer(output_ref_meta);

    auto cuda_run = [&]() {
      InvokeExtractSubMatrix<T>(reinterpret_cast<T*>(input_meta.data_ptr) + input_offset,
                               reinterpret_cast<T*>(output_meta.data_ptr), m, n, output_n, stream);
    };
    float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 30);
    std::cout << std::left << "InvokeExtractSubMatrix  m=" << std::setw(6) << m << " n=" << std::setw(6) << n
              << " execution 1 times " << std::setw(10) << milliseconds << " ms " << std::endl;

    DeleteBuffer(input_meta);
    DeleteBuffer(output_meta);  }
};

TEST_F(NvidiaExtractMatrixTestSuit, FloatNvidiaExtractMatrixTestSuit) { TestGahterSubmatirx<float>(stream); }
TEST_F(NvidiaExtractMatrixTestSuit, HalfNvidiaExtractMatrixTestSuit) { TestGahterSubmatirx<half>(stream); }
TEST_F(NvidiaExtractMatrixTestSuit, BF16NvidiaExtractMatrixTestSuit) { TestGahterSubmatirx<__nv_bfloat16>(stream); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels