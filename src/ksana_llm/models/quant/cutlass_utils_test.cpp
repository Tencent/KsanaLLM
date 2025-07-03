/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/quant/cutlass_utils.h"
#include "tests/test.h"

using namespace ksana_llm;

class CutlassUtilsTest : public testing::Test {
 protected:
  // 在每个测试用例执行之前调用的函数
  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1, 1);
    cutlass_helper_ = std::make_shared<CutlassUtils>(context_, rank_, bits_);
  }

  // 在每个测试用例执行之后调用的函数
  void TearDown() override {}

  torch::Tensor RefCutlassPackInt8ToPackedInt4(torch::Tensor weight) {
    std::vector<int64_t> packed_tensor_size(weight.dim());
    for (int i = 0; i < weight.dim(); ++i) {
      packed_tensor_size[i] = weight.size(i);
    }
    packed_tensor_size[weight.dim() - 1] = (packed_tensor_size[weight.dim() - 1] + 1) / 2;

    torch::Tensor packed_weight = torch::zeros(packed_tensor_size, torch::dtype(torch::kInt8).device(torch::kCPU));

    int8_t* unpacked_ptr = reinterpret_cast<int8_t*>(weight.data_ptr());
    int8_t* packed_ptr = reinterpret_cast<int8_t*>(packed_weight.data_ptr());

    for (int64_t packed_idx = 0; packed_idx < packed_weight.numel(); ++packed_idx) {
      int8_t packed_int4s = 0;
      int8_t elt_0 = unpacked_ptr[2 * packed_idx + 0];
      int8_t elt_1 = unpacked_ptr[2 * packed_idx + 1];

      packed_int4s |= ((elt_0 & 0x0F));
      packed_int4s |= int8_t(elt_1 << 4);

      packed_ptr[packed_idx] = packed_int4s;
    }
    return packed_weight;
  }

  torch::Tensor RefCutlassPreprocessWeightsForMixedGemmWarpper(torch::Tensor row_major_quantized_weight,
                                                               llm_kernels::nvidia::QuantType quant_type) {
    const size_t bits_in_quant_type = get_weight_quant_bits(quant_type);
    const size_t num_experts = row_major_quantized_weight.dim() == 2 ? 1 : row_major_quantized_weight.size(0);
    const size_t num_rows = row_major_quantized_weight.size(-2);
    const size_t num_cols = (8 / bits_in_quant_type) * row_major_quantized_weight.size(-1);

    torch::Tensor processed_tensor = torch::zeros_like(row_major_quantized_weight);
    preprocess_weights_for_mixed_gemm(reinterpret_cast<int8_t*>(processed_tensor.data_ptr()),
                                      reinterpret_cast<int8_t*>(row_major_quantized_weight.data_ptr()),
                                      {num_experts, num_rows, num_cols}, quant_type);
    return processed_tensor;
  }

 protected:
  std::shared_ptr<CutlassUtils> cutlass_helper_{nullptr};
  std::shared_ptr<Context> context_{nullptr};
  int rank_{0};
  int bits_{4};
};

TEST_F(CutlassUtilsTest, TestCutlassPackInt8ToPackedInt4) {
#ifdef ENABLE_CUDA
  int8_t int8_min = std::numeric_limits<int8_t>::min();
  int8_t int8_max = std::numeric_limits<int8_t>::max();
  auto options = torch::TensorOptions().dtype(torch::kInt8);
  torch::Tensor tensor = torch::randint(int8_min, int8_max, {1024, 1024}, options);

  torch::Tensor ref = RefCutlassPackInt8ToPackedInt4(tensor);
  torch::Tensor dst = cutlass_helper_->CutlassPackInt8ToPackedInt4(tensor);
  EXPECT_TRUE(torch::allclose(ref, dst));
#endif
}

TEST_F(CutlassUtilsTest, TestCutlassUnPackGPTQ) {
#ifdef ENABLE_CUDA
  torch::Tensor tensor =
      torch::full({1024, 2048}, 1717986918, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank_));
  tensor = tensor.view(torch::kInt32);

  tensor = cutlass_helper_->CutlassUnpackGPTQ(tensor);
  int8_t zero = std::pow(2, bits_ - 1);
  tensor = (tensor - zero).contiguous();
  tensor = cutlass_helper_->CutlassPackInt8ToPackedInt4(tensor);

  EXPECT_TRUE(torch::all(tensor == -18).item<bool>());
#endif
}

TEST_F(CutlassUtilsTest, TestCutlassUnPackAWQ) {
#ifdef ENABLE_CUDA
  torch::Tensor tensor =
      torch::full({1024, 2048}, 1717986918, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank_));
  tensor = tensor.view(torch::kInt32);

  tensor = cutlass_helper_->CutlassUnpackAWQ(tensor);
  int8_t zero = std::pow(2, bits_ - 1);
  tensor = (tensor - zero).contiguous();
  tensor = cutlass_helper_->CutlassPackInt8ToPackedInt4(tensor);

  EXPECT_TRUE(torch::all(tensor == -18).item<bool>());
#endif
}

TEST_F(CutlassUtilsTest, TestCutlassPrepack) {
#ifdef ENABLE_CUDA
  size_t num_rows = 1024, num_cols = 1024;

  int8_t int8_min = std::numeric_limits<int8_t>::min();
  int8_t int8_max = std::numeric_limits<int8_t>::max();
  auto options = torch::TensorOptions().dtype(torch::kInt8);
  torch::Tensor random_cpu = torch::randint(int8_min, int8_max, {num_rows, num_cols / 2}, options);

  torch::Tensor processed_tensor_cpu =
      RefCutlassPreprocessWeightsForMixedGemmWarpper(random_cpu, llm_kernels::nvidia::QuantType::W4_A16);

  torch::Tensor random_gpu = random_cpu.to(torch::Device(torch::kCUDA, rank_));
  torch::Tensor processed_tensor_gpu =
      cutlass_helper_->CutlassPreprocessWeightsForMixedGemmWarpper(random_gpu, llm_kernels::nvidia::QuantType::W4_A16);

  EXPECT_TRUE(torch::allclose(processed_tensor_gpu, processed_tensor_cpu.to(torch::Device(torch::kCUDA, rank_))));
#endif
}