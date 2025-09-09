/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include "ksana_llm/data_hub/expert_parallel_deepep_wrapper.h"
#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

using namespace ksana_llm;

std::mutex deepep_wrapper_test_mutex_;
static int deepep_wrapper_test_count_ = 0;

// TODO(zezhao): 更新环境，使用真实的 DeepEP 完成单测
class MockDeepEP {
 public:
  MockDeepEP(size_t num_ranks, size_t num_experts, size_t max_token_num, size_t hidden_size, size_t expert_topk) {
    num_ranks_ = num_ranks;
    num_experts_ = num_experts;
    hidden_size_ = hidden_size;
    expert_topk_ = expert_topk;
    // 创建一片临时空间用于后续数据分发
    for (size_t rank = 0; rank < num_ranks; rank++) {
      SetDevice(rank);
      void* x_ptr;
      Malloc(&x_ptr, max_token_num * hidden_size * sizeof(uint16_t));
      buffer_x.push_back(x_ptr);
      void* topk_ids_ptr;
      Malloc(&topk_ids_ptr, max_token_num * expert_topk * sizeof(int));
      buffer_topk_ids.push_back(topk_ids_ptr);
      void* topk_weights_ptr;
      Malloc(&topk_weights_ptr, max_token_num * expert_topk * sizeof(int));
      buffer_topk_weights.push_back(topk_weights_ptr);
    }
  }

  void MockInit(int node_rank) {
    // 打开共享内存
    std::string shm_name = fmt::format("/nvshmem_ipc_data_{}", node_rank);
    int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
    ASSERT_NE(shm_fd, -1);
    struct stat shm_stat;
    ASSERT_NE(fstat(shm_fd, &shm_stat), -1);
    ASSERT_GE(shm_stat.st_size, sizeof(IPCData));

    void* addr = mmap(nullptr, sizeof(IPCData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    close(shm_fd);  // 映射后就可以关闭文件描述符
    ASSERT_NE(addr, MAP_FAILED);
    IPCData* data = static_cast<IPCData*>(addr);
    volatile IPCData* volatile_data = data;

    // 检查有效性
    try {
      __sync_synchronize();
      ASSERT_TRUE(volatile_data->ready);
    } catch (...) {
      // 访问异常，说明还没初始化完成
      FAIL() << "access safe Failed" << std::endl;
    }

    shared_data_ = const_cast<ksana_llm::IPCData*>(volatile_data);

    // 将共享显存绑定到 void* 上
    x_ptrs.resize(num_ranks_);
    topk_ids_ptrs.resize(num_ranks_);
    topk_weights_ptrs.resize(num_ranks_);
#ifdef ENABLE_CUDA
    for (size_t rank = 0; rank < num_ranks_; rank++) {
      cudaIpcOpenMemHandle(&x_ptrs[rank], shared_data_->x[rank], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&topk_ids_ptrs[rank], shared_data_->topk_ids[rank], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&topk_weights_ptrs[rank], shared_data_->topk_weights[rank], cudaIpcMemLazyEnablePeerAccess);
    }
#endif
  }

  void MockUploadDeepEPMeta(size_t num_ranks_per_node, size_t node_rank) {
    // 模拟上传 DeepEP Meta 信息
    for (size_t rank = node_rank * num_ranks_per_node;
         rank < std::min(num_ranks_, (node_rank + 1) * num_ranks_per_node); rank++) {
      shared_data_->ipc_handle_ready[rank] = true;
      shared_data_->unique_id_ready[rank] = true;
    }
  }

  void MockSetUnready() {
    for (size_t rank = 0; rank < num_ranks_; rank++) {
      shared_data_->ipc_handle_ready[rank] = false;
      shared_data_->unique_id_ready[rank] = false;
    }
  }

  void MockDispatch() {
    bool ksana_dispatch_flag = false;
    while (!ksana_dispatch_flag) {
      ksana_dispatch_flag = true;
      for (size_t rank = 0; rank < num_ranks_; rank++) {
        ksana_dispatch_flag &= shared_data_->trigger_dispatch[rank];
      }
      usleep(1);
    }
    // 收到了全部信号，根据 topk_ids 做数据分发
    // ... 此处应当由 DeepEP 完成单机/跨机数据交换
    // 交换完成，唤醒一念进程
    for (size_t rank = 0; rank < num_ranks_; rank++) {
      shared_data_->trigger_dispatch[rank] = false;
    }
  }

  void MockCombine() {
    bool ksana_combine_flag = false;
    while (!ksana_combine_flag) {
      ksana_combine_flag = true;
      for (size_t rank = 0; rank < num_ranks_; rank++) {
        ksana_combine_flag &= shared_data_->trigger_combine[rank];
      }
      usleep(1);
    }
    // 收到了全部信号，根据暂存的 send_head 数据做数据还原
    // ... 此处应当由 DeepEP 完成单机/跨机数据交换
    // 交换完成，唤醒一念进程
    for (size_t rank = 0; rank < num_ranks_; rank++) {
      shared_data_->trigger_combine[rank] = false;
    }
  }

  IPCData* shared_data_ = nullptr;
  size_t num_ranks_;
  size_t num_experts_;
  size_t hidden_size_;
  size_t expert_topk_;
  std::vector<void*> x_ptrs, topk_ids_ptrs, topk_weights_ptrs;
  std::vector<void*> buffer_x, buffer_topk_ids, buffer_topk_weights;
};

class ExpertParallelDeepepWrapperTest : public testing::Test {
 protected:
  void SetUp() override {
    num_ranks_ = 2;
    num_experts_ = 32;
    max_token_num_ = 128;
    hidden_size_ = 20;
    expert_topk_ = 6;
    // 创建上下文对象
    context_ = std::make_shared<Context>(num_ranks_, 1, 1);
    mock_deepep_ = std::make_shared<MockDeepEP>(num_ranks_, num_experts_, max_token_num_, hidden_size_, expert_topk_);
  }

  void TearDown() override {}

  Tensor CreateDeviceTensor(const std::vector<size_t>& shape, const DataType& dtype, const int rank) {
    void* data_ptr;
    size_t element_size = GetTypeSize(dtype);
    for (auto& dim : shape) {
      element_size *= dim;
    }
    Malloc(&data_ptr, element_size);
    Tensor tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, shape, rank, data_ptr);
    return tensor;
  }

 protected:
  size_t num_ranks_;
  size_t num_experts_;
  size_t max_token_num_;
  size_t hidden_size_;
  size_t expert_topk_;
  std::shared_ptr<Context> context_;
  std::shared_ptr<ExpertParallelDeepepWrapper> deepep_wrapper_;
  std::shared_ptr<MockDeepEP> mock_deepep_;
};

TEST_F(ExpertParallelDeepepWrapperTest, DeepepWrapperTest) {
#ifndef ENABLE_CUDA
    GTEST_SKIP_("Only Nvidia support this test temporary.");
#endif

  KLLM_LOG_INFO << fmt::format("ExpertParallelDeepepWrapperTest.DeepepWrapperTest begin");
  // 设置测试参数
  size_t num_ranks_per_node = num_ranks_;
  size_t node_rank = 0;

  // 测试 DeepEP Wrapper 的初始化
  // 另起一个线程，控制 deepep_wrapper 启动
  std::thread deepep_wrapper_thread([&]() {
    std::lock_guard<std::mutex> lock(deepep_wrapper_test_mutex_);
    deepep_wrapper_ =
        std::make_shared<ExpertParallelDeepepWrapper>(num_ranks_, num_ranks_per_node, node_rank, max_token_num_,
                                                      hidden_size_, expert_topk_, num_experts_, context_);
    deepep_wrapper_->Init();
    KLLM_LOG_INFO << fmt::format("deepep_wrapper_ Init Success.");
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // 模拟 DeepEP 进程启动
  mock_deepep_->MockInit(node_rank);
  ASSERT_NE(mock_deepep_->shared_data_, nullptr);
  // 模拟上报（仅标记上报成功，不实际上报）
  mock_deepep_->MockUploadDeepEPMeta(num_ranks_per_node, node_rank);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ASSERT_NE(deepep_wrapper_, nullptr);

  // 等待线程完成
  deepep_wrapper_thread.join();

  // 测试双方连通性
  mock_deepep_->MockSetUnready();
  deepep_wrapper_->SetReady();
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    ASSERT_TRUE(mock_deepep_->shared_data_->ipc_handle_ready[rank]);
    ASSERT_TRUE(mock_deepep_->shared_data_->unique_id_ready[rank]);
  }

  // 测试 Dispatch 和 Combine 功能
  // 准备输入数据
  std::vector<size_t> num_tokens = {4, 2};
  // 创建输入输出 tensor
  std::vector<Tensor> x_tensors, topk_ids_tensors, topk_weights_tensors, out_x_tensors;
  for (int rank = 0; rank < static_cast<int>(num_ranks_); rank++) {
    SetDevice(rank);
    x_tensors.push_back(CreateDeviceTensor({max_token_num_, hidden_size_}, TYPE_FP16, rank));
    topk_ids_tensors.push_back(CreateDeviceTensor({max_token_num_, expert_topk_}, TYPE_INT32, rank));
    topk_weights_tensors.push_back(CreateDeviceTensor({max_token_num_, expert_topk_}, TYPE_FP32, rank));
    out_x_tensors.push_back(CreateDeviceTensor({max_token_num_, hidden_size_}, TYPE_FP16, rank));
    x_tensors[rank].shape[0] = num_tokens[rank];
    topk_ids_tensors[rank].shape[0] = num_tokens[rank];
    topk_weights_tensors[rank].shape[0] = num_tokens[rank];
  }

  // Test Dispatch
  // Step 1: 由一念发起 Dispatch 信号
  std::vector<std::thread> threads;
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    std::vector<Tensor> input_tensors = {x_tensors[rank], topk_ids_tensors[rank], topk_weights_tensors[rank]};
    std::vector<Tensor> output_tensors = {out_x_tensors[rank], out_x_tensors[rank]};
    std::thread dispatch_thread([this, rank, in = std::move(input_tensors), out = std::move(output_tensors)]() mutable {
      deepep_wrapper_->Dispatch(in, out, rank);
    });
    threads.push_back(std::move(dispatch_thread));
  }

  // Step 2: 由 MockDeepEP 汇总并处理 Dispatch
  mock_deepep_->MockDispatch();
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    threads[rank].join();
  }
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    ASSERT_FALSE(mock_deepep_->shared_data_->trigger_dispatch[rank]);
  }
  threads.clear();

  // Step 3: 由一念发起 Combine 信号
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    std::vector<Tensor> input_tensors = {out_x_tensors[rank]};
    // 第一个项用于存储 MOE 计算结果, 第二项用于存储 Dispatch 输出
    std::vector<Tensor> output_tensors = {out_x_tensors[rank], out_x_tensors[rank]};
    std::thread combine_thread([this, rank, in = std::move(input_tensors), out = std::move(output_tensors)]() mutable {
      deepep_wrapper_->Combine(in, out, rank);
    });
    threads.push_back(std::move(combine_thread));
  }

  // Step 4: 由 MockDeepEP 汇总并处理 Combine
  mock_deepep_->MockCombine();
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    threads[rank].join();
  }
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    ASSERT_FALSE(mock_deepep_->shared_data_->trigger_combine[rank]);
  }
}

TEST_F(ExpertParallelDeepepWrapperTest, TestWithUseScalesEnabled) {
#ifndef ENABLE_CUDA
  GTEST_SKIP_("Only Nvidia support this test temporary.");
#endif

  KLLM_LOG_INFO << fmt::format("ExpertParallelDeepepWrapperTest.TestWithUseScalesEnabled begin");

  // 设置测试参数
  size_t num_ranks_per_node = num_ranks_;
  size_t node_rank = 0;

  // 测试 DeepEP Wrapper 的初始化
  std::thread deepep_wrapper_thread([&]() {
    std::lock_guard<std::mutex> lock(deepep_wrapper_test_mutex_);
    deepep_wrapper_ = std::make_shared<ExpertParallelDeepepWrapper>(
        num_ranks_, num_ranks_per_node, node_rank, max_token_num_, hidden_size_, expert_topk_, num_experts_, context_);
    deepep_wrapper_->Init();
    KLLM_LOG_INFO << fmt::format("deepep_wrapper_ Init Success for use_scales test.");
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // 模拟 DeepEP 进程启动
  mock_deepep_->MockInit(node_rank);
  ASSERT_NE(mock_deepep_->shared_data_, nullptr);

  // 设置 use_scales 为 true（这是测试的重点）
  mock_deepep_->shared_data_->use_scales = true;

  // 模拟上报（仅标记上报成功，不实际上报）
  mock_deepep_->MockUploadDeepEPMeta(num_ranks_per_node, node_rank);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ASSERT_NE(deepep_wrapper_, nullptr);

  // 等待线程完成
  deepep_wrapper_thread.join();

  // 验证 use_scales 设置
  ASSERT_TRUE(mock_deepep_->shared_data_->use_scales);

  // 测试双方连通性
  mock_deepep_->MockSetUnready();
  deepep_wrapper_->SetReady();
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    ASSERT_TRUE(mock_deepep_->shared_data_->ipc_handle_ready[rank]);
    ASSERT_TRUE(mock_deepep_->shared_data_->unique_id_ready[rank]);
  }

  // 测试带有 scales 的 Dispatch 和 Combine 功能
  // 准备输入数据（包含 scales tensor）
  std::vector<size_t> num_tokens = {4, 2};
  // 创建输入输出 tensor，包括 scales tensor
  std::vector<Tensor> x_tensors, topk_ids_tensors, topk_weights_tensors, out_x_tensors, scales_tensors;
  for (int rank = 0; rank < static_cast<int>(num_ranks_); rank++) {
    SetDevice(rank);
    x_tensors.push_back(CreateDeviceTensor({max_token_num_, hidden_size_}, TYPE_FP16, rank));
    topk_ids_tensors.push_back(CreateDeviceTensor({max_token_num_, expert_topk_}, TYPE_INT32, rank));
    topk_weights_tensors.push_back(CreateDeviceTensor({max_token_num_, expert_topk_}, TYPE_FP32, rank));
    out_x_tensors.push_back(CreateDeviceTensor({max_token_num_, hidden_size_}, TYPE_FP16, rank));
    // 为 use_scales=true 的情况添加 scales tensor
    scales_tensors.push_back(CreateDeviceTensor({max_token_num_, hidden_size_}, TYPE_FP32, rank));

    x_tensors[rank].shape[0] = num_tokens[rank];
    topk_ids_tensors[rank].shape[0] = num_tokens[rank];
    topk_weights_tensors[rank].shape[0] = num_tokens[rank];
    scales_tensors[rank].shape[0] = num_tokens[rank];
  }

  // Test Dispatch with scales
  std::vector<std::thread> threads;
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    // 包含 scales tensor 的输入
    std::vector<Tensor> input_tensors = {x_tensors[rank], topk_ids_tensors[rank], topk_weights_tensors[rank],
                                         scales_tensors[rank]};
    std::vector<Tensor> output_tensors = {out_x_tensors[rank], out_x_tensors[rank]};
    std::thread dispatch_thread([this, rank, in = std::move(input_tensors), out = std::move(output_tensors)]() mutable {
      deepep_wrapper_->Dispatch(in, out, rank);
    });
    threads.push_back(std::move(dispatch_thread));
  }

  // 由 MockDeepEP 汇总并处理 Dispatch
  mock_deepep_->MockDispatch();
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    threads[rank].join();
  }
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    ASSERT_FALSE(mock_deepep_->shared_data_->trigger_dispatch[rank]);
  }
  threads.clear();

  // Test Combine with scales
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    std::vector<Tensor> input_tensors = {out_x_tensors[rank]};
    std::vector<Tensor> output_tensors = {out_x_tensors[rank], out_x_tensors[rank]};
    std::thread combine_thread([this, rank, in = std::move(input_tensors), out = std::move(output_tensors)]() mutable {
      deepep_wrapper_->Combine(in, out, rank);
    });
    threads.push_back(std::move(combine_thread));
  }

  // 由 MockDeepEP 汇总并处理 Combine
  mock_deepep_->MockCombine();
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    threads[rank].join();
  }
  for (size_t rank = 0; rank < num_ranks_; rank++) {
    ASSERT_FALSE(mock_deepep_->shared_data_->trigger_combine[rank]);
  }

  KLLM_LOG_INFO << fmt::format("ExpertParallelDeepepWrapperTest.TestWithUseScalesEnabled completed successfully");
}
