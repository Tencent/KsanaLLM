/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/common_model_weight_loader.h"

#include <memory>
#include <random>
#include <vector>

#include "3rdparty/half/include/half.hpp"
#include "ksana_llm/utils/common_device.h"
#include "tests/test.h"

namespace ksana_llm {

class CommonModelWeightLoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    model_config_ = std::make_shared<BaseModelConfig>();
    model_config_->model_dir = "/model/test";

    BlockManagerConfig block_manager_config;
    block_manager_config.host_allocator_config.blocks_num = 2;
    block_manager_config.host_allocator_config.block_token_num = 16;
    block_manager_config.host_allocator_config.block_size = 1024;
    block_manager_config.host_allocator_config.device = MEMORY_HOST;
    block_manager_config.device_allocator_config.blocks_num = 2;
    block_manager_config.device_allocator_config.block_token_num = 16;
    block_manager_config.device_allocator_config.block_size = 1024;
    block_manager_config.device_allocator_config.device = MEMORY_DEVICE;

    env_ = std::make_shared<Environment>();
    env_->SetBlockManagerConfig(block_manager_config);

    context_ = std::make_shared<Context>(1, 1, 1);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<BaseModelConfig> model_config_;
  std::shared_ptr<Environment> env_;
  std::shared_ptr<Context> context_;
};

TEST_F(CommonModelWeightLoaderTest, Permute2DTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  using dtype = uint8_t;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: 1024x2048 张量转置
  {
    const size_t rows = 1024;
    const size_t cols = 2048;

    // 创建输入张量
    Tensor input_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {rows, cols}, kDeviceRank);

    // 准备测试数据
    std::vector<dtype> input_host(rows * cols);
    std::default_random_engine eng;
    std::uniform_int_distribution<int> random_range(0, 255);
    for (size_t i = 0; i < rows * cols; ++i) {
      input_host[i] = static_cast<dtype>(random_range(eng));
    }

    // 拷贝到设备
    MemcpyAsync(input_tensor.GetPtr<void>(), input_host.data(), input_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[kDeviceRank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 执行 Permute2D
    Status status = weight_loader.Permute2D(input_tensor, kDeviceRank);
    EXPECT_TRUE(status.OK());

    // 同步 stream
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 验证形状已经转置
    EXPECT_EQ(input_tensor.shape[0], cols);
    EXPECT_EQ(input_tensor.shape[1], rows);

    // 验证数据正确性
    std::vector<dtype> output_host(rows * cols);
    MemcpyAsync(output_host.data(), input_tensor.GetPtr<void>(), input_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context_->GetMemoryManageStreams()[kDeviceRank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 完全遍历验证转置结果：output[j][i] 应该等于 input[i][j]
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        size_t original_idx = i * cols + j;
        size_t transposed_idx = j * rows + i;
        EXPECT_EQ(output_host[transposed_idx], input_host[original_idx]);
      }
    }
  }

  // 测试用例 2: 错误情况 - 非2D张量应该抛出异常
  {
    // 1D 张量
    Tensor tensor_1d(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {100}, kDeviceRank);
    EXPECT_THROW(weight_loader.Permute2D(tensor_1d, kDeviceRank), std::exception);

    // 3D 张量
    Tensor tensor_3d(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {10, 20, 30}, kDeviceRank);
    EXPECT_THROW(weight_loader.Permute2D(tensor_3d, kDeviceRank), std::exception);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, GetTorchTensorFromTensorTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: HOST 上的 FP32 张量
  {
    const size_t rows = 10;
    const size_t cols = 20;

    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {rows, cols}, kDeviceRank);
    torch::Tensor torch_tensor = weight_loader.GetTorchTensorFromTensor(host_tensor);

    EXPECT_TRUE(torch_tensor.device().is_cpu());
    EXPECT_EQ(torch_tensor.scalar_type(), torch::kFloat32);
    EXPECT_EQ(torch_tensor.size(0), rows);
    EXPECT_EQ(torch_tensor.size(1), cols);
  }

  // 测试用例 2: DEVICE 上的 FP16 张量
  {
    const size_t rows = 8;
    const size_t cols = 16;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows, cols}, kDeviceRank);
    torch::Tensor torch_tensor = weight_loader.GetTorchTensorFromTensor(device_tensor);

    EXPECT_TRUE(torch_tensor.device().is_cuda());
    EXPECT_EQ(torch_tensor.device().index(), kDeviceRank);
    EXPECT_EQ(torch_tensor.scalar_type(), torch::kFloat16);
    EXPECT_EQ(torch_tensor.size(0), rows);
    EXPECT_EQ(torch_tensor.size(1), cols);
  }

  // 测试用例 3: 1D 张量
  {
    const size_t size = 100;

    Tensor tensor_1d(MemoryLocation::LOCATION_HOST, TYPE_FP32, {size}, kDeviceRank);
    torch::Tensor torch_tensor = weight_loader.GetTorchTensorFromTensor(tensor_1d);

    EXPECT_EQ(torch_tensor.dim(), 1);
    EXPECT_EQ(torch_tensor.size(0), size);
    EXPECT_TRUE(torch_tensor.device().is_cpu());
    EXPECT_EQ(torch_tensor.scalar_type(), torch::kFloat32);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, GetTensorFromTorchTensorTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: CPU 上的 FP32 torch::Tensor
  {
    const size_t rows = 10;
    const size_t cols = 20;

    torch::Tensor torch_tensor = torch::randn({static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    Tensor tensor = weight_loader.GetTensorFromTorchTensor(torch_tensor);

    EXPECT_EQ(tensor.location, MemoryLocation::LOCATION_HOST);
    EXPECT_EQ(tensor.dtype, TYPE_FP32);
    EXPECT_EQ(tensor.shape.size(), 2);
    EXPECT_EQ(tensor.shape[0], rows);
    EXPECT_EQ(tensor.shape[1], cols);
  }

  // 测试用例 2: CUDA 上的 FP16 torch::Tensor
  {
    const size_t rows = 8;
    const size_t cols = 16;

    torch::Tensor torch_tensor =
        torch::randn({static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                     torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, kDeviceRank));
    Tensor tensor = weight_loader.GetTensorFromTorchTensor(torch_tensor);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    EXPECT_EQ(tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(tensor.device_id, kDeviceRank);
    EXPECT_EQ(tensor.dtype, TYPE_FP16);
    EXPECT_EQ(tensor.shape.size(), 2);
    EXPECT_EQ(tensor.shape[0], rows);
    EXPECT_EQ(tensor.shape[1], cols);
  }

  // 测试用例 3: 1D torch::Tensor
  {
    const size_t size = 100;

    torch::Tensor torch_tensor =
        torch::randn({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    Tensor tensor = weight_loader.GetTensorFromTorchTensor(torch_tensor);

    EXPECT_EQ(tensor.shape.size(), 1);
    EXPECT_EQ(tensor.shape[0], size);
    EXPECT_EQ(tensor.location, MemoryLocation::LOCATION_HOST);
    EXPECT_EQ(tensor.dtype, TYPE_FP32);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, GetMemcpyKindTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: HOST → HOST
  {
    Tensor host_tensor1(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20}, kDeviceRank);
    Tensor host_tensor2(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20}, kDeviceRank);
    MemcpyKind kind = weight_loader.GetMemcpyKind(host_tensor1, host_tensor2);
    EXPECT_EQ(kind, MEMCPY_HOST_TO_HOST);
  }

  // 测试用例 2: HOST → DEVICE
  {
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20}, kDeviceRank);
    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20}, kDeviceRank);
    MemcpyKind kind = weight_loader.GetMemcpyKind(host_tensor, device_tensor);
    EXPECT_EQ(kind, MEMCPY_HOST_TO_DEVICE);
  }

  // 测试用例 3: DEVICE → HOST
  {
    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20}, kDeviceRank);
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20}, kDeviceRank);
    MemcpyKind kind = weight_loader.GetMemcpyKind(device_tensor, host_tensor);
    EXPECT_EQ(kind, MEMCPY_DEVICE_TO_HOST);
  }

  // 测试用例 4: DEVICE → DEVICE
  {
    Tensor device_tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20}, kDeviceRank);
    Tensor device_tensor2(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20}, kDeviceRank);
    MemcpyKind kind = weight_loader.GetMemcpyKind(device_tensor1, device_tensor2);
    EXPECT_EQ(kind, MEMCPY_DEVICE_TO_DEVICE);
  }

  // 测试用例 5: 不同数据类型的张量（应该仍然正常工作，因为只检查 location）
  {
    Tensor host_fp32(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20}, kDeviceRank);
    Tensor device_fp16(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {10, 20}, kDeviceRank);
    MemcpyKind kind = weight_loader.GetMemcpyKind(host_fp32, device_fp16);
    EXPECT_EQ(kind, MEMCPY_HOST_TO_DEVICE);
  }

  // 测试用例 6: 不同形状的张量（应该仍然正常工作，因为只检查 location）
  {
    Tensor device_small(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {5, 10}, kDeviceRank);
    Tensor host_large(MemoryLocation::LOCATION_HOST, TYPE_FP32, {100, 200}, kDeviceRank);
    MemcpyKind kind = weight_loader.GetMemcpyKind(device_small, host_large);
    EXPECT_EQ(kind, MEMCPY_DEVICE_TO_HOST);
  }
#endif
}

TEST_F(CommonModelWeightLoaderTest, MoveToDeviceTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例: FP16 类型的 2D 张量
  {
    const size_t rows = 10;
    const size_t cols = 20;

    // 创建 HOST 张量并填充数据
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {rows, cols}, kDeviceRank);
    std::vector<half_float::half> host_data(rows * cols);
    std::default_random_engine eng;
    std::uniform_real_distribution<float> random_range(0.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = half_float::half(random_range(eng));
    }
    std::memcpy(host_tensor.GetPtr<void>(), host_data.data(), host_tensor.GetTotalBytes());

    // 执行 MoveToDevice
    Tensor device_tensor = weight_loader.MoveToDevice(host_tensor, kDeviceRank);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.device_id, kDeviceRank);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(device_tensor.shape.size(), 2);
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);

    // 验证数据正确性：将 device 数据拷贝回 host 并比较
    std::vector<half_float::half> device_data(rows * cols);
    MemcpyAsync(device_data.data(), device_tensor.GetPtr<void>(), device_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context_->GetMemoryManageStreams()[kDeviceRank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    for (size_t i = 0; i < rows * cols; ++i) {
      EXPECT_EQ(device_data[i], host_data[i]);
    }
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, CastDeviceTensorTypeTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: FP32 → FP16
  {
    const size_t rows = 10;
    const size_t cols = 20;

    // 创建 DEVICE 上的 FP32 张量并填充数据
    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {rows, cols}, kDeviceRank);
    std::vector<float> host_data(rows * cols);
    std::default_random_engine eng(42);
    std::uniform_real_distribution<float> random_range(-1.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = random_range(eng);
    }
    MemcpyAsync(device_tensor.GetPtr<void>(), host_data.data(), device_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[kDeviceRank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 执行类型转换
    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_FP16, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(device_tensor.shape.size(), 2);
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);

    // 验证数据正确性：将转换后的数据拷贝回 host 并比较
    std::vector<half_float::half> result_data(rows * cols);
    MemcpyAsync(result_data.data(), device_tensor.GetPtr<void>(), device_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context_->GetMemoryManageStreams()[kDeviceRank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    for (size_t i = 0; i < rows * cols; ++i) {
      float expected = host_data[i];
      float actual = static_cast<float>(result_data[i]);
      EXPECT_NEAR(actual, expected, 1e-3);  // FP16 精度损失
    }
  }

  // 测试用例 2: FP32 → BF16
  {
    const size_t rows = 8;
    const size_t cols = 16;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {rows, cols}, kDeviceRank);
    std::vector<float> host_data(rows * cols);
    std::default_random_engine eng(123);
    std::uniform_real_distribution<float> random_range(-1.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = random_range(eng);
    }
    MemcpyAsync(device_tensor.GetPtr<void>(), host_data.data(), device_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[kDeviceRank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_BF16, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    EXPECT_EQ(device_tensor.dtype, TYPE_BF16);
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);
  }

  // 测试用例 3: FP16 → FP32
  {
    const size_t rows = 12;
    const size_t cols = 24;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows, cols}, kDeviceRank);
    std::vector<half_float::half> host_data(rows * cols);
    std::default_random_engine eng(456);
    std::uniform_real_distribution<float> random_range(-1.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; ++i) {
      host_data[i] = half_float::half(random_range(eng));
    }
    MemcpyAsync(device_tensor.GetPtr<void>(), host_data.data(), device_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[kDeviceRank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_FP32, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);

    // 验证数据正确性
    std::vector<float> result_data(rows * cols);
    MemcpyAsync(result_data.data(), device_tensor.GetPtr<void>(), device_tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context_->GetMemoryManageStreams()[kDeviceRank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    for (size_t i = 0; i < rows * cols; ++i) {
      float expected = static_cast<float>(host_data[i]);
      float actual = result_data[i];
      EXPECT_NEAR(actual, expected, 1e-3);
    }
  }

  // 测试用例 4: BF16 → FP32
  {
    const size_t size = 100;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {size}, kDeviceRank);

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_FP32, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(device_tensor.shape[0], size);
  }

  // 测试用例 5: FP16 → BF16 (原地转换)
  {
    const size_t rows = 16;
    const size_t cols = 32;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows, cols}, kDeviceRank);
    void* original_ptr = device_tensor.GetPtr<void>();

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_BF16, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    EXPECT_EQ(device_tensor.dtype, TYPE_BF16);
    EXPECT_EQ(device_tensor.GetPtr<void>(), original_ptr);  // 原地转换，指针不变
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);
  }

  // 测试用例 6: BF16 → FP16 (原地转换)
  {
    const size_t rows = 20;
    const size_t cols = 40;

    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {rows, cols}, kDeviceRank);
    void* original_ptr = device_tensor.GetPtr<void>();

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_FP16, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    EXPECT_EQ(device_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(device_tensor.GetPtr<void>(), original_ptr);  // 原地转换，指针不变
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols);
  }

  // 测试用例 7: 相同类型转换（应该直接返回成功）
  {
    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20}, kDeviceRank);

    Status status = weight_loader.CastDeviceTensorType(device_tensor, TYPE_FP32, kDeviceRank);
    EXPECT_TRUE(status.OK());

    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
  }

  // 测试用例 8: 不支持的类型转换（应该抛出异常）
  {
    Tensor device_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {10, 20}, kDeviceRank);

    // 假设 TYPE_UINT8 到 TYPE_FP16 的转换不被支持
    EXPECT_THROW(weight_loader.CastDeviceTensorType(device_tensor, TYPE_UINT8, kDeviceRank), std::exception);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, TensorParallelSplitTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;
  constexpr size_t kParaSize = 2;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: 1D 张量 RowPara 切分
  {
    const size_t size = 100;

    // 创建 HOST 上的 1D 张量
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {size}, kDeviceRank);

    // 执行 TensorParallelSplit
    Tensor device_tensor =
        weight_loader.TensorParallelSplit(host_tensor, kDeviceRank, kParaSize, TensorParallelMode::RowPara);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.device_id, kDeviceRank);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(device_tensor.shape.size(), 1);
    EXPECT_EQ(device_tensor.shape[0], size / kParaSize);
  }

  // 测试用例 2: 2D 张量 RowPara 切分
  {
    const size_t rows = 100;
    const size_t cols = 50;

    // 创建 HOST 上的 2D 张量
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {rows, cols}, kDeviceRank);

    // 执行 TensorParallelSplit
    Tensor device_tensor =
        weight_loader.TensorParallelSplit(host_tensor, kDeviceRank, kParaSize, TensorParallelMode::RowPara);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.device_id, kDeviceRank);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(device_tensor.shape.size(), 2);
    EXPECT_EQ(device_tensor.shape[0], rows / kParaSize);
    EXPECT_EQ(device_tensor.shape[1], cols);
  }

  // 测试用例 3: 2D 张量 ColPara 切分
  {
    const size_t rows = 80;
    const size_t cols = 60;

    // 创建 HOST 上的 2D 张量
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {rows, cols}, kDeviceRank);

    // 执行 TensorParallelSplit
    Tensor device_tensor =
        weight_loader.TensorParallelSplit(host_tensor, kDeviceRank, kParaSize, TensorParallelMode::ColPara);
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 验证输出张量的属性
    EXPECT_EQ(device_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(device_tensor.device_id, kDeviceRank);
    EXPECT_EQ(device_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(device_tensor.shape.size(), 2);
    EXPECT_EQ(device_tensor.shape[0], rows);
    EXPECT_EQ(device_tensor.shape[1], cols / kParaSize);
  }

  // 测试用例 4: 错误情况 - 1D 张量使用 ColPara 模式（应该抛出异常）
  {
    Tensor host_tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {100}, kDeviceRank);
    EXPECT_THROW(weight_loader.TensorParallelSplit(host_tensor, kDeviceRank, kParaSize, TensorParallelMode::ColPara),
                 std::exception);
  }

  // 测试用例 5: 错误情况 - 张量大小不能被 para_size 整除（应该抛出异常）
  {
    // 1D 张量
    Tensor host_tensor_1d(MemoryLocation::LOCATION_HOST, TYPE_FP32, {101}, kDeviceRank);
    EXPECT_THROW(weight_loader.TensorParallelSplit(host_tensor_1d, kDeviceRank, kParaSize, TensorParallelMode::RowPara),
                 std::exception);

    // 2D 张量 RowPara
    Tensor host_tensor_2d_row(MemoryLocation::LOCATION_HOST, TYPE_FP32, {101, 50}, kDeviceRank);
    EXPECT_THROW(
        weight_loader.TensorParallelSplit(host_tensor_2d_row, kDeviceRank, kParaSize, TensorParallelMode::RowPara),
        std::exception);

    // 2D 张量 ColPara
    Tensor host_tensor_2d_col(MemoryLocation::LOCATION_HOST, TYPE_FP32, {100, 51}, kDeviceRank);
    EXPECT_THROW(
        weight_loader.TensorParallelSplit(host_tensor_2d_col, kDeviceRank, kParaSize, TensorParallelMode::ColPara),
        std::exception);
  }

  // 测试用例 6: 错误情况 - 3D 张量（应该抛出异常）
  {
    Tensor host_tensor_3d(MemoryLocation::LOCATION_HOST, TYPE_FP32, {10, 20, 30}, kDeviceRank);
    EXPECT_THROW(weight_loader.TensorParallelSplit(host_tensor_3d, kDeviceRank, kParaSize, TensorParallelMode::RowPara),
                 std::exception);
  }

#endif
}

TEST_F(CommonModelWeightLoaderTest, AutoMergeWeightTest) {
#ifdef ENABLE_CUDA
  constexpr int kDeviceRank = 0;

  // 创建 CommonModelWeightLoader 实例
  CommonModelWeightLoader weight_loader(model_config_, env_, context_);

  // 测试用例 1: 合并两个 2D 张量
  {
    const size_t rows1 = 100;
    const size_t rows2 = 200;
    const size_t cols = 50;

    // 创建两个 DEVICE 张量
    Tensor tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {rows1, cols}, kDeviceRank);
    Tensor tensor2(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {rows2, cols}, kDeviceRank);

    // 创建 dev_weights_map
    std::unordered_map<std::string, Tensor> dev_weights_map;
    dev_weights_map["weight1"] = tensor1;
    dev_weights_map["weight2"] = tensor2;

    // 执行 AutoMergeWeight
    std::vector<std::string> input_names = {"weight1", "weight2"};
    Status status = weight_loader.AutoMergeWeight(input_names, "merged_weight", dev_weights_map, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 验证合并后的张量存在
    EXPECT_TRUE(dev_weights_map.find("merged_weight") != dev_weights_map.end());

    // 验证原始张量已被删除
    EXPECT_TRUE(dev_weights_map.find("weight1") == dev_weights_map.end());
    EXPECT_TRUE(dev_weights_map.find("weight2") == dev_weights_map.end());

    // 验证合并后的张量 shape
    const Tensor& merged_tensor = dev_weights_map.at("merged_weight");
    EXPECT_EQ(merged_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(merged_tensor.dtype, TYPE_FP32);
    EXPECT_EQ(merged_tensor.shape.size(), 2);
    EXPECT_EQ(merged_tensor.shape[0], rows1 + rows2);
    EXPECT_EQ(merged_tensor.shape[1], cols);
  }

  // 测试用例 2: 合并三个 2D 张量
  {
    const size_t rows1 = 50;
    const size_t rows2 = 75;
    const size_t rows3 = 100;
    const size_t cols = 64;

    // 创建三个 DEVICE 张量
    Tensor tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows1, cols}, kDeviceRank);
    Tensor tensor2(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows2, cols}, kDeviceRank);
    Tensor tensor3(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {rows3, cols}, kDeviceRank);

    // 创建 dev_weights_map
    std::unordered_map<std::string, Tensor> dev_weights_map;
    dev_weights_map["q_proj"] = tensor1;
    dev_weights_map["k_proj"] = tensor2;
    dev_weights_map["v_proj"] = tensor3;

    // 执行 AutoMergeWeight
    std::vector<std::string> input_names = {"q_proj", "k_proj", "v_proj"};
    Status status = weight_loader.AutoMergeWeight(input_names, "qkv_proj", dev_weights_map, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 验证合并后的张量存在
    EXPECT_TRUE(dev_weights_map.find("qkv_proj") != dev_weights_map.end());

    // 验证原始张量已被删除
    EXPECT_TRUE(dev_weights_map.find("q_proj") == dev_weights_map.end());
    EXPECT_TRUE(dev_weights_map.find("k_proj") == dev_weights_map.end());
    EXPECT_TRUE(dev_weights_map.find("v_proj") == dev_weights_map.end());

    // 验证合并后的张量 shape
    const Tensor& merged_tensor = dev_weights_map.at("qkv_proj");
    EXPECT_EQ(merged_tensor.location, MemoryLocation::LOCATION_DEVICE);
    EXPECT_EQ(merged_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(merged_tensor.shape.size(), 2);
    EXPECT_EQ(merged_tensor.shape[0], rows1 + rows2 + rows3);
    EXPECT_EQ(merged_tensor.shape[1], cols);
  }

  // 测试用例 3: 错误情况 - 输入权重不存在
  {
    std::unordered_map<std::string, Tensor> dev_weights_map;
    Tensor tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {100, 50}, kDeviceRank);
    dev_weights_map["weight1"] = tensor1;

    // 尝试合并不存在的权重
    std::vector<std::string> input_names = {"weight1", "non_existent_weight"};
    Status status = weight_loader.AutoMergeWeight(input_names, "merged_weight", dev_weights_map, kDeviceRank);
    EXPECT_FALSE(status.OK());
    EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
  }

  // 测试用例 4: 合并不同数据类型的张量（应该使用第一个张量的类型）
  {
    const size_t rows1 = 80;
    const size_t rows2 = 120;
    const size_t cols = 32;

    // 创建两个不同类型的 DEVICE 张量
    Tensor tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {rows1, cols}, kDeviceRank);
    Tensor tensor2(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {rows2, cols}, kDeviceRank);

    // 创建 dev_weights_map
    std::unordered_map<std::string, Tensor> dev_weights_map;
    dev_weights_map["gate_proj"] = tensor1;
    dev_weights_map["up_proj"] = tensor2;

    // 执行 AutoMergeWeight
    std::vector<std::string> input_names = {"gate_proj", "up_proj"};
    Status status = weight_loader.AutoMergeWeight(input_names, "gate_up_proj", dev_weights_map, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 验证合并后的张量 shape 和类型
    const Tensor& merged_tensor = dev_weights_map.at("gate_up_proj");
    EXPECT_EQ(merged_tensor.dtype, TYPE_BF16);
    EXPECT_EQ(merged_tensor.shape[0], rows1 + rows2);
    EXPECT_EQ(merged_tensor.shape[1], cols);
  }

  // 测试用例 5: 合并单个张量（边界情况）
  {
    const size_t rows = 100;
    const size_t cols = 50;

    Tensor tensor1(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {rows, cols}, kDeviceRank);

    std::unordered_map<std::string, Tensor> dev_weights_map;
    dev_weights_map["single_weight"] = tensor1;

    // 执行 AutoMergeWeight
    std::vector<std::string> input_names = {"single_weight"};
    Status status = weight_loader.AutoMergeWeight(input_names, "merged_single", dev_weights_map, kDeviceRank);
    EXPECT_TRUE(status.OK());
    StreamSynchronize(context_->GetMemoryManageStreams()[kDeviceRank]);

    // 验证合并后的张量 shape（应该与原始张量相同）
    const Tensor& merged_tensor = dev_weights_map.at("merged_single");
    EXPECT_EQ(merged_tensor.shape[0], rows);
    EXPECT_EQ(merged_tensor.shape[1], cols);
    EXPECT_TRUE(dev_weights_map.find("single_weight") == dev_weights_map.end());
  }

#endif
}

}  // namespace ksana_llm