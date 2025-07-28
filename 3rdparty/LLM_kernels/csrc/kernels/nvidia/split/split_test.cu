/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

 #include <gtest/gtest.h>

 #include "csrc/utils/nvidia/cuda_utils.h"
 #include "tests/kernels/nvidia/utils/testsuit_base.h"
 
 #include "csrc/kernels/nvidia/split/split.h"
 
 namespace llm_kernels {
 namespace nvidia {
 namespace test {
 
 class NvidiaSplitTestSuit : public NvidiaTestSuitBase {
  public:
   void SetUp() override { NvidiaTestSuitBase::SetUp(); }
 
   void TearDown() override { NvidiaTestSuitBase::TearDown(); }

  protected:
   using NvidiaTestSuitBase::stream;
   using InputTuple = std::tuple<size_t, size_t, std::vector<size_t>>;
   const std::vector<InputTuple> input_tuple = {
       {30, 2112, {1536, 576}},     {30, 2112, {1539, 573}},         {30, 2112, {1536, 512, 64}},
       {30, 2112, {1539, 512, 61}}, {30, 2112, {1536, 512, 48, 16}}, {30, 2112, {1539, 512, 48, 13}},
   };

  protected:
   template <typename T>
   void RunSplitRef(const std::vector<size_t>& output_n_dim, const std::string& type_str) {
     std::stringstream ss;
     ss << "python split_test.py --type=" << type_str << " --output_n ";
     for (const auto& n : output_n_dim) {
       ss << n << " ";
     }
     system(ss.str().c_str());
 
   }
 
   template <typename T>
   void TestSplit(cudaStream_t stream) {
     std::string type_str = "float";
     if (std::is_same<T, half>::value) {
       type_str = "half";
     } else if (std::is_same<T, __nv_bfloat16>::value) {
       type_str = "bfloat16";
     }
     for (const auto& [m, n, output_n_dim] : input_tuple) {
       BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                               /*is_random_init*/ true);
       std::vector<T*> outputs;
       std::vector<int> col_offset = {0};
       std::vector<BufferMeta> output_metas;
       for (size_t output_n : output_n_dim) {
         output_metas.emplace_back(CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, output_n},
                                                   /*is_random_init*/ false));
         outputs.push_back(reinterpret_cast<T*>(output_metas.back().data_ptr));
         col_offset.push_back(col_offset.back() + output_n);
       }
 
       input_meta.SaveToNpy<T>("split_test_input.npy");
       RunSplitRef<T>(output_n_dim, type_str);
       InvokeSplit<T>(reinterpret_cast<T*>(input_meta.data_ptr), outputs, col_offset, m, n, output_n_dim.size(), stream);
       for (size_t i = 0; i < outputs.size(); ++i) {
         BufferMeta output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, output_n_dim[i]},
                                                      /*is_random_init*/ false);
         output_ref_meta.LoadNpy<T>("split_test_output_" + std::to_string(i) + ".npy");
         EXPECT_TRUE(CheckResult<T>("split_test_" + type_str + "_m_" + std::to_string(m), output_ref_meta,
                                    output_metas[i], 1e-5f, 1e-5f));
         output_metas[i].SaveToNpy<T>("kernel_split_test_output_" + std::to_string(i) + ".npy");
 
         DeleteBuffer(output_ref_meta);
       }
 
       auto cuda_run = [&]() {
         InvokeSplit<T>(reinterpret_cast<T*>(input_meta.data_ptr), outputs, col_offset, m, n, output_n_dim.size(),
                        stream);
       };
       float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 30);
       std::cout << "Split Matrix [" << m << "," << n << "] -> ";
       std::stringstream output_dims_ss;
       for (size_t i = 0; i < output_n_dim.size(); ++i) {
         output_dims_ss << "[" << m << "," << output_n_dim[i] << "]";
         if (i < output_n_dim.size() - 1) output_dims_ss << " ";
       }
       std::cout << std::left << std::setw(40) << output_dims_ss.str();
       std::cout << " | Execution Time: " << std::fixed << milliseconds << "ms" << std::endl;
       for (size_t i = 0; i < outputs.size(); ++i) {
         DeleteBuffer(output_metas[i]);
       }
       DeleteBuffer(input_meta);
     }
   }
 };
 
 TEST_F(NvidiaSplitTestSuit, HalfNvidiaSplitTestSuit) { TestSplit<half>(stream); }
 
 TEST_F(NvidiaSplitTestSuit, FloatNvidiaSplitTestSuit) { TestSplit<float>(stream); }
 
 TEST_F(NvidiaSplitTestSuit, Bf16NvidiaSplitTestSuit) { TestSplit<__nv_bfloat16>(stream); }
 
 }  // namespace test
 }  // namespace nvidia
 }  // namespace llm_kernels