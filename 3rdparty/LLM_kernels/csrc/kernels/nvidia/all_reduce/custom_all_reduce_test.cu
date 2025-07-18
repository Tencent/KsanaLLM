/*
 * Adapted from
 * https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce_test.cu
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <curand_kernel.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <atomic>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_profiler_api.h>
#include <nccl.h>

#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

#define NCCLCHECK(cmd)                                                                      \
  do {                                                                                      \
    ncclResult_t r = cmd;                                                                   \
    if (r != ncclSuccess) {                                                                 \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

__global__ void dummy_kernel() {
  for (int i = 0; i < 100; i++) __nanosleep(1000000);  // 100ms
}

class LlamaNvidiaCustomAllReduceTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    // 判断GPU是否是90以及以上的显卡
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // 获取设备0的属性
    int major = prop.major;
    skip_test = major < 9;
  }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  bool skip_test = false;

 protected:
  template <typename T>
  void RunCustomAllReduce(int cur_rank, int total_ranks, ncclComm_t &comm, size_t data_size, void **signals,
                          void **data_handles, void **input_handles, std::atomic<int> &counter, bool is_full_nvlink) {
    std::string type_str = "float";
    ncclDataType_t ncclDtype = ncclFloat;
    if (std::is_same<T, half>::value) {
      type_str = "half";
      ncclDtype = ncclFloat16;
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
      ncclDtype = ncclBfloat16;
    }
    cudaStream_t stream;
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    BufferMeta result_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, false);
    T *result = static_cast<T *>(result_meta.data_ptr);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(result, 0, data_size * sizeof(T)));

    size_t buffer_size = data_size * sizeof(T);
    BufferMeta buffer_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, false);
    data_handles[cur_rank] = (char *)buffer_meta.data_ptr;

    size_t largest_part = (data_size * sizeof(T)) / total_ranks + (data_size * sizeof(T)) % total_ranks;
    BufferMeta meta_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {(sizeof(Signal) + largest_part) / sizeof(T)}, false);
    signals[cur_rank] = meta_meta.data_ptr;

    BufferMeta self_data_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, true);
    T *self_data = static_cast<T *>(self_data_meta.data_ptr);
    input_handles[cur_rank] = self_data;

    // sync all threads
    counter++;
    while (counter != total_ranks);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(data_handles[cur_rank], 0, buffer_size));

    BufferMeta refer_result_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, false);
    T *refer_result = static_cast<T *>(refer_result_meta.data_ptr);

    size_t rank_data_sz = 8 * 1024 * 1024;
    BufferMeta rank_data_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(rank_data_sz / sizeof(T))}, false);
    void *rank_data = rank_data_meta.data_ptr;

    std::vector<int64_t> offsets(total_ranks, 0);
    CustomAllreduce custom_all_reduce((Signal **)signals, rank_data, rank_data_sz, cur_rank, total_ranks,
                                      is_full_nvlink);
    // hack buffer registration
    void *data[8];
    for (int i = 0; i < total_ranks; i++) {
      data[i] = input_handles[i];
    }
    custom_all_reduce.RegisterBuffer(data, stream);

    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpyAsync(refer_result, self_data, data_size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    constexpr int warmup_iters = 10;
    constexpr int num_iters = 25;

    dummy_kernel<<<1, 1, 0, stream>>>();
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    auto nccl_run = [&]() {
      NCCLCHECK(ncclAllReduce(self_data, refer_result, data_size, ncclDtype, ncclSum, comm, stream));
    };
    float allreduce_ms = MeasureCudaExecutionTime(nccl_run, stream, warmup_iters, num_iters);

    dummy_kernel<<<1, 1, 0, stream>>>();
    auto custom_allreduce_run = [&]() {
      custom_all_reduce.AllReduce<T>(stream, self_data, result, data_size);
    };
    float duration_ms = MeasureCudaExecutionTime(custom_allreduce_run, stream, warmup_iters, num_iters);

    if (cur_rank == 0) {
      printf(
          "Rank %d done, nGPUs:%d, sz (kb), %ld, my time,%.2f,us, nccl "
          "time,%.2f,us\n",
          cur_rank, total_ranks, data_size * sizeof(T) / 1024, duration_ms * 1e3, allreduce_ms * 1e3);
    }

    // And wait for all the queued up work to complete
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    NCCLCHECK(ncclAllReduce(self_data, refer_result, data_size, ncclDtype, ncclSum, comm, stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    constexpr float atol = std::is_same<T, half>::value ? 1e-3 : std::is_same<T, __nv_bfloat16>::value ? 5e-3 : 1e-5;
    constexpr float rtol = std::is_same<T, half>::value ? 1e-4 : std::is_same<T, __nv_bfloat16>::value ? 5e-3 : 1e-5;
    EXPECT_TRUE(CheckResult<T>("custom_all_reduce_" + type_str + "_size_" + std::to_string(data_size * sizeof(T)),
                               refer_result_meta, result_meta, atol, rtol));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));

    // sync
    counter--;
    while (counter != 0);
  }

  template <typename T>
  void RunCustomAllReduceThread(int cur_rank, int total_ranks, ncclUniqueId nccl_id, void **signals,
                                void **data_handles, void **input_handles, std::atomic<int> &counter,
                                bool is_full_nvlink) {
    CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));
    for (int i = 0; i < total_ranks; i++) {
      if (i != cur_rank) {
        auto err = cudaDeviceEnablePeerAccess(i, 0);
        if (err != cudaErrorPeerAccessAlreadyEnabled) {
          CHECK_NVIDIA_CUDA_ERROR(err);
        }
      }
    }
    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, total_ranks, nccl_id, cur_rank));
    std::vector<size_t> tokens = {8, 16, 32, 128};
    size_t hidden_size = 7168;
    for (int token : tokens) {
      RunCustomAllReduce<T>(cur_rank, total_ranks, comm, token * hidden_size, signals, data_handles, input_handles,
                            counter, is_full_nvlink);
    }
    for (int i = 0; i < total_ranks; ++i) {
      if (i != cur_rank) {
        CHECK_NVIDIA_CUDA_ERROR(cudaDeviceDisablePeerAccess(i));
      }
    }
  }

  template <typename T>
  void TestCustomAllReduce() {
    int device_count = -1;
    CHECK_NVIDIA_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (device_count < 2 || device_count > 8 || device_count % 2 != 0) {
      GTEST_SKIP_("This test is just for 2,4,6,8 GPUs");
    }

    int total_ranks = device_count;
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    CHECK_NVIDIA_CUDA_ERROR(cudaProfilerStart());
    std::vector<std::shared_ptr<std::thread>> run_threads;
    std::atomic<int> counter(0);
    std::vector<void *> signals(8);
    std::vector<void *> data_handles(8);
    std::vector<void *> input_handles(8);

    bool is_full_nvlink = true;
    for (size_t i = 0; i < static_cast<size_t>(device_count); ++i) {
      if (GetNvLinkVersion(0, i) == 0) {
        is_full_nvlink = false;
        break;
      }
    }

    if (total_ranks > 2 && is_full_nvlink == false) {
      return;
    }

    for (int cur_rank = 0; cur_rank < total_ranks; ++cur_rank) {
      run_threads.emplace_back(std::shared_ptr<std::thread>(new std::thread(
          &LlamaNvidiaCustomAllReduceTestSuit::RunCustomAllReduceThread<T>, this, cur_rank, total_ranks, nccl_id,
          static_cast<void **>(signals.data()), static_cast<void **>(data_handles.data()),
          static_cast<void **>(input_handles.data()), std::ref<std::atomic<int>>(counter), is_full_nvlink)));
    }
    for (int cur_rank = 0; cur_rank < total_ranks; ++cur_rank) {
      run_threads[cur_rank]->join();
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaProfilerStop());
  }

  template <typename T>
  void RunCustomGroupAllReduce(int cur_rank, int total_ranks, uint32_t group_size, size_t data_size, void **signals,
                               void **data_handles, void **input_handles, void **input_handles_cpu_ptrs,
                               std::atomic<int> &counter, bool is_full_nvlink) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }
    uint32_t root_rank = cur_rank / group_size * group_size;
    cudaStream_t stream;
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    BufferMeta result_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, false);
    T *result = static_cast<T *>(result_meta.data_ptr);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(result, 0, data_size * sizeof(T)));

    size_t buffer_size = data_size * sizeof(T);
    BufferMeta buffer_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, false);
    data_handles[cur_rank] = (char *)buffer_meta.data_ptr;

    size_t largest_part = (data_size * sizeof(T)) / total_ranks + (data_size * sizeof(T)) % total_ranks;
    BufferMeta meta_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {(sizeof(Signal) + largest_part) / sizeof(T)}, false);
    signals[cur_rank] = meta_meta.data_ptr;

    BufferMeta self_data_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, true);
    T *self_data = static_cast<T *>(self_data_meta.data_ptr);
    input_handles[cur_rank] = self_data;
    BufferMeta self_data_meta_cpu = CopyToHost<T>(self_data_meta);
    input_handles_cpu_ptrs[cur_rank] = self_data_meta_cpu.data_ptr;

    BufferMeta refer_result_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(data_size)}, false);
    BufferMeta refer_result_meta_cpu = CopyToHost<T>(refer_result_meta);

    // sync all threads
    counter++;
    while (counter != total_ranks);
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(data_handles[cur_rank], 0, buffer_size));

    size_t rank_data_sz = 8 * 1024 * 1024;
    BufferMeta rank_data_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(rank_data_sz / sizeof(T))}, false);
    void *rank_data = rank_data_meta.data_ptr;

    CustomAllreduce custom_all_reduce((Signal **)signals, rank_data, rank_data_sz, cur_rank, group_size, is_full_nvlink,
                                      root_rank);
    // hack buffer registration
    void *data[8];
    for (int i = 0; i < total_ranks; i++) {
      data[i] = input_handles[i];
    }
    custom_all_reduce.RegisterBuffer(data, stream);

    constexpr int num_iters = 25;
    dummy_kernel<<<1, 1, 0, stream>>>();
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    custom_all_reduce.AllReduce<T>(stream, self_data, result, data_size);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta result_meta_cpu = CopyToHost<T>(result_meta);

    constexpr float atol = std::is_same<T, half>::value ? 1e-3 : std::is_same<T, __nv_bfloat16>::value ? 5e-3 : 1e-5;
    constexpr float rtol = std::is_same<T, half>::value ? 1e-4 : std::is_same<T, __nv_bfloat16>::value ? 5e-3 : 1e-5;

    for (size_t i = root_rank; i < root_rank + group_size; i++) {
      T *input_ptr = static_cast<T *>(input_handles_cpu_ptrs[i]);
      for (size_t j = 0; j < data_size; j++) {
        T input_val = static_cast<T *>(input_ptr)[j];
        (static_cast<T *>(refer_result_meta_cpu.data_ptr))[j] += input_val;
      }
    }

    EXPECT_TRUE(CheckResult<T>("custom_group_all_reduce_" + type_str + "_size_" + std::to_string(data_size * sizeof(T)),
                               refer_result_meta_cpu, result_meta_cpu, atol, rtol));

    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));

    // sync
    counter--;
    while (counter != 0);
  }

  template <typename T>
  void RunCustomGroupAllReduceThread(int cur_rank, int total_ranks, uint32_t group_size, void **signals,
                                     void **data_handles, void **input_handles, void **input_handles_cpu_ptrs,
                                     std::atomic<int> &counter, bool is_full_nvlink) {
    CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));
    for (int i = 0; i < total_ranks; i++) {
      if (i != cur_rank) {
        auto err = cudaDeviceEnablePeerAccess(i, 0);
        if (err != cudaErrorPeerAccessAlreadyEnabled) {
          CHECK_NVIDIA_CUDA_ERROR(err);
        }
      }
    }
    std::vector<size_t> tokens = {8, 16, 32, 128};
    size_t hidden_size = 7168;
    for (int token : tokens) {
      RunCustomGroupAllReduce<T>(cur_rank, total_ranks, group_size, token * hidden_size, signals, data_handles,
                                 input_handles, input_handles_cpu_ptrs, counter, is_full_nvlink);
    }
    for (int i = 0; i < total_ranks; ++i) {
      if (i != cur_rank) {
        CHECK_NVIDIA_CUDA_ERROR(cudaDeviceDisablePeerAccess(i));
      }
    }
  }

  template <typename T>
  void TestCustomGroupAllReduce() {
    // NOTE(karlluo): for attention data parallel, we need to do allreduce inner attention dp with group all reduce, for
    // example, if we have 4 GPUs(tensor_para_size = 4), and each 2 GPUs relate to 1 attention dp(attn_data_para_size =
    // 2).
    // |-----------------TP 4------------------|
    // |  GPU 0  |  GPU 1  |  GPU 2  |  GPU 3  |
    // |  attn dp size = 2 |  attn dp size = 2 |
    // |  dp group id = 0  |  dp group id = 1  |
    // |    embedding tensor para forward      |
    // |              all gather               |
    // |----------------barrier----------------|
    // |    layernorm tensor para forward      |
    // |---------------------------------------|
    // |   attn data para  |   attn data para  |
    // |---------------------------------------|
    // | group all reduce  | group all reduce  |
    // |---------------------------------------|
    // |              all gather               |
    // |----------------barrier----------------|
    // |       MOE tensor para forward         |
    // |---------------------------------------|
    int device_count = -1;
    CHECK_NVIDIA_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (device_count < 4 || device_count > 8 || device_count % 2 != 0) {
      GTEST_SKIP_("Custom Group AllReduce is just for 4,6,8 GPUs");
    }

    int total_ranks = device_count;
    uint32_t group_size = 2;
    CHECK_NVIDIA_CUDA_ERROR(cudaProfilerStart());
    std::vector<std::shared_ptr<std::thread>> run_threads;
    std::atomic<int> counter(0);
    std::vector<void *> signals(8);
    std::vector<void *> data_handles(8);
    std::vector<void *> input_handles(8);
    std::vector<void *> input_handles_cpu_ptrs(8);

    bool is_full_nvlink = true;
    for (size_t i = 0; i < static_cast<size_t>(device_count); ++i) {
      if (GetNvLinkVersion(0, i) == 0) {
        is_full_nvlink = false;
        break;
      }
    }

    if (total_ranks > 2 && is_full_nvlink == false) {
      return;
    }

    for (int cur_rank = 0; cur_rank < total_ranks; ++cur_rank) {
      run_threads.emplace_back(std::shared_ptr<std::thread>(new std::thread(
          &LlamaNvidiaCustomAllReduceTestSuit::RunCustomGroupAllReduceThread<T>, this, cur_rank, total_ranks,
          group_size, static_cast<void **>(signals.data()), static_cast<void **>(data_handles.data()),
          static_cast<void **>(input_handles.data()), static_cast<void **>(input_handles_cpu_ptrs.data()),
          std::ref<std::atomic<int>>(counter), is_full_nvlink)));
    }
    for (int cur_rank = 0; cur_rank < total_ranks; ++cur_rank) {
      run_threads[cur_rank]->join();
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaProfilerStop());
  }
};

TEST_F(LlamaNvidiaCustomAllReduceTestSuit, FloatCustomAllReduceTest) {
  if (!skip_test) {
    TestCustomAllReduce<float>();
  }
}
TEST_F(LlamaNvidiaCustomAllReduceTestSuit, HalfCustomAllReduceTest) {
  if (!skip_test) {
    TestCustomAllReduce<half>();
  }
}
TEST_F(LlamaNvidiaCustomAllReduceTestSuit, BFloat16CustomAllReduceTest) {
  if (!skip_test) {
    TestCustomAllReduce<__nv_bfloat16>();
  }
}

TEST_F(LlamaNvidiaCustomAllReduceTestSuit, FloatCustomGroupAllReduceTest) {
  if (!skip_test) {
    TestCustomGroupAllReduce<float>();
  }
}
TEST_F(LlamaNvidiaCustomAllReduceTestSuit, HalfCustomGroupAllReduceTest) {
  if (!skip_test) {
    TestCustomGroupAllReduce<half>();
  }
}
TEST_F(LlamaNvidiaCustomAllReduceTestSuit, BFloat16CustomGroupAllReduceTest) {
  if (!skip_test) {
    TestCustomGroupAllReduce<__nv_bfloat16>();
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
