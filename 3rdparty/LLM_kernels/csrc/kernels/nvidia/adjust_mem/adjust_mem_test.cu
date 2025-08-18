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

class NvidiaGatherMatrixTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  size_t n_start_ = 1024;
  size_t n_end_ = 2048;
  size_t n_ = 4096;
  std::vector<size_t> m_num_per_group_ = {2, 5, 0, 8};

 protected:
  template <typename T>
  void RunRef(size_t m, size_t group_num, size_t group_size, const std::string& type_str) {
    std::stringstream ss;
    ss << "python ./adjust_mem_test.py --type=" << type_str
       << " --n_start=" << n_start_ << " --n_end=" << n_end_ << " --m=" << m << " --n=" << n_
       << " --group_num=" << group_num << " --group_size=" << group_size << " --test_func=InvokeGatherSubmatrix "
       << " --m_num_per_group ";
    for (size_t m_num : m_num_per_group_) {
      ss << m_num << " ";
    }
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
    size_t group_num = m_num_per_group_.size();
    size_t group_size = *std::max_element(m_num_per_group_.begin(), m_num_per_group_.end());
    size_t m = group_num * group_size;
    size_t output_m = std::accumulate(m_num_per_group_.begin(), m_num_per_group_.end(), 0);

    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n_},
                                            /*is_random_init*/ true);
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {output_m, n_end_ - n_start_},
                                             /*is_random_init*/ false);
    BufferMeta workspace_meta = CreateBuffer<size_t>(MemoryType::MEMORY_GPU, {2 * group_num + 1},
                                                     /*is_random_init*/ false);
    InvokeGatherSubmatrix<T>(reinterpret_cast<T*>(input_meta.data_ptr), reinterpret_cast<T*>(output_meta.data_ptr),
                             m_num_per_group_, group_size, group_num, n_start_, n_end_, m, n_,
                             reinterpret_cast<void*>(workspace_meta.data_ptr), stream);

    input_meta.SaveToNpy<T>("gather_submatrix_input.npy");
    RunRef<T>(m, group_num, group_size, type_str);
    BufferMeta output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {output_m, n_end_ - n_start_},
                                                 /*is_random_init*/ false);
    output_ref_meta.LoadNpy<T>("gather_submatrix_output.npy");
    EXPECT_TRUE(CheckResult<T>("gather_submatrix_test_" + type_str, output_ref_meta, output_meta, 1e-5f, 1e-5f, 0.0f));
    DeleteBuffer(output_ref_meta);

    auto cuda_run = [&]() {
      InvokeGatherSubmatrix<T>(reinterpret_cast<T*>(input_meta.data_ptr), reinterpret_cast<T*>(output_meta.data_ptr),
                               m_num_per_group_, group_size, group_num, n_start_, n_end_, m, n_,
                               reinterpret_cast<void*>(workspace_meta.data_ptr), stream);
    };
    float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 30);
    std::cout << std::left << "InvokeGatherSubmatrix  m=" << std::setw(6) << m << " n=" << std::setw(6) << n_
              << " execution 1 times " << std::setw(10) << milliseconds << " ms " << std::endl;

    DeleteBuffer(input_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(workspace_meta);
  }
};

class NvidiaDpMapCopyTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  // row_offset,  prefill_num, prefill_start, decode_start
  std::vector<size_t> group_info_ = {11, 7, 0, 14, 13, 0, 7, 18, 13, 0, 7, 20, 20, 7, 7, 20};
  size_t n = 7168;

 protected:
  template <typename T>
  void RunRef(const std::string& type_str) {
    std::stringstream ss;
    ss << "python ./adjust_mem_test.py --type=" << type_str
       << " --test_func=InvokeDpMapCopy " << " --group_info ";
    for (size_t info : group_info_) {
      ss << info << " ";
    }
    system(ss.str().c_str());
  }

  template <typename T>
  void TestDpMapCopy(cudaStream_t stream) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }
    size_t group_num = group_info_.size() / 4;
    size_t m = group_info_[4 * (group_num - 1)];

    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                            /*is_random_init*/ true);
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                             /*is_random_init*/ false);

    BufferMeta workspace_meta = CreateBuffer<size_t>(MemoryType::MEMORY_GPU, {group_info_.size()},
                                                     /*is_random_init*/ false);

    InvokeDpMapCopy<T>(reinterpret_cast<T*>(input_meta.data_ptr), reinterpret_cast<T*>(output_meta.data_ptr),
                       group_info_, m, n, reinterpret_cast<void*>(workspace_meta.data_ptr), stream);

    input_meta.SaveToNpy<T>("map_copy_input.npy");
    RunRef<T>(type_str);
    BufferMeta output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                 /*is_random_init*/ false);
    output_ref_meta.LoadNpy<T>("map_copy_output.npy");
    EXPECT_TRUE(CheckResult<T>("map_copy_test_" + type_str, output_ref_meta, output_meta, 1e-5f, 1e-5f, 0.0f));
    DeleteBuffer(output_ref_meta);

    auto cuda_run = [&]() {
      InvokeDpMapCopy<T>(reinterpret_cast<T*>(input_meta.data_ptr), reinterpret_cast<T*>(output_meta.data_ptr),
                         group_info_, m, n, reinterpret_cast<void*>(workspace_meta.data_ptr), stream);
    };
    float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 30);
    std::cout << std::left << "InvokeDpMapCopy  m=" << std::setw(6) << m << " n=" << std::setw(6) << n
              << " execution 1 times " << std::setw(10) << milliseconds << " ms " << std::endl;

    DeleteBuffer(input_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(workspace_meta);
  }
};

TEST_F(NvidiaGatherMatrixTestSuit, FloatNvidiaGatherMatrixTestSuit) { TestGahterSubmatirx<float>(stream); }
TEST_F(NvidiaGatherMatrixTestSuit, HalfNvidiaGatherMatrixTestSuit) { TestGahterSubmatirx<half>(stream); }
TEST_F(NvidiaGatherMatrixTestSuit, BF16NvidiaGatherMatrixTestSuit) { TestGahterSubmatirx<__nv_bfloat16>(stream); }

TEST_F(NvidiaDpMapCopyTestSuit, FloatNvidiaDpMapCopyTestSuit) { TestDpMapCopy<float>(stream); }
TEST_F(NvidiaDpMapCopyTestSuit, HalfNvidiaDpMapCopyTestSuit) { TestDpMapCopy<half>(stream); }
TEST_F(NvidiaDpMapCopyTestSuit, Bf16NvidiaDpMapCopyTestSuit) { TestDpMapCopy<__nv_bfloat16>(stream); }

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels