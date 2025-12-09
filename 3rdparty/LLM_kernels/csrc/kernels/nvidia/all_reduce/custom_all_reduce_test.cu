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

  struct PerformanceShape {
    size_t micro_batch;
    size_t seq_len;
    size_t hidden;

    size_t NumElements() const { return micro_batch * seq_len * hidden; }

    std::string Description() const {
      return "mb=" + std::to_string(micro_batch) + ",seq=" + std::to_string(seq_len) + ",hidden=" +
             std::to_string(hidden);
    }
  };

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
    CustomAllreduce custom_all_reduce(rank_data, rank_data_sz, cur_rank, total_ranks, is_full_nvlink);
    custom_all_reduce.RegisterSignalBuffer((Signal **)signals);
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

    CustomAllreduce custom_all_reduce(rank_data, rank_data_sz, cur_rank, group_size, is_full_nvlink, root_rank, true);
    custom_all_reduce.RegisterSignalBuffer((Signal **)signals);
    // hack buffer registration
    void *data[8];
    for (int i = 0; i < total_ranks; i++) {
      data[i] = input_handles[i];
    }
    custom_all_reduce.RegisterBuffer(data, stream);

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

  // Simple barrier using sense-reversing flag to avoid race conditions
  struct Barrier {
    std::atomic<int> counter{0};
    std::atomic<int> sense{0};
    int total_ranks;
    
    Barrier(int n) : total_ranks(n) {}
    
    void Wait() {
      int local_sense = sense.load(std::memory_order_acquire);
      int arrived = counter.fetch_add(1, std::memory_order_acq_rel) + 1;
      
      if (arrived == total_ranks) {
        // Last thread resets counter and flips sense
        counter.store(0, std::memory_order_release);
        sense.store(1 - local_sense, std::memory_order_release);
      } else {
        // Wait for sense to flip
        while (sense.load(std::memory_order_acquire) == local_sense) {
          // Busy wait
        }
      }
    }
  };

  template <typename T>
  float RunCustomAllReducePerformanceOnce(int cur_rank, int total_ranks, size_t data_size, void **signals, void **data_handles,
                           void **input_handles, Barrier &barrier, bool is_full_nvlink, bool use_group_mode,
                           uint32_t group_size) {
    cudaStream_t stream;
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    BufferMeta result_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {data_size}, false);
    T *result = static_cast<T *>(result_meta.data_ptr);

    BufferMeta buffer_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {data_size}, false);
    data_handles[cur_rank] = buffer_meta.data_ptr;

    size_t largest_part = (data_size * sizeof(T)) / total_ranks + (data_size * sizeof(T)) % total_ranks;
    BufferMeta signal_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {(sizeof(Signal) + largest_part) / sizeof(T)}, false);
    signals[cur_rank] = signal_meta.data_ptr;

    BufferMeta self_data_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {data_size}, true);
    T *self_data = static_cast<T *>(self_data_meta.data_ptr);
    input_handles[cur_rank] = self_data;

    barrier.Wait();
    CHECK_NVIDIA_CUDA_ERROR(cudaMemset(data_handles[cur_rank], 0, data_size * sizeof(T)));

    size_t rank_data_sz = 8 * 1024 * 1024;
    BufferMeta rank_data_meta =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(rank_data_sz / sizeof(T))}, false);
    void *rank_data = rank_data_meta.data_ptr;

    uint32_t root_rank = 0;
    bool is_group_custom_all_reduce = false;
    int effective_world = total_ranks;
    if (use_group_mode) {
      is_group_custom_all_reduce = true;
      root_rank = (cur_rank / group_size) * group_size;
      effective_world = group_size;
    }

    CustomAllreduce custom_all_reduce(rank_data, rank_data_sz, cur_rank, effective_world, is_full_nvlink, root_rank,
                                      is_group_custom_all_reduce);
    custom_all_reduce.RegisterSignalBuffer(reinterpret_cast<Signal **>(signals));

    void *data_ptrs[maxDeviceCount] = {nullptr};
    for (int i = 0; i < total_ranks; ++i) {
      data_ptrs[i] = input_handles[i];
    }
    custom_all_reduce.RegisterBuffer(data_ptrs, stream);

    constexpr int warmup_iters = 10;
    constexpr int run_iters = 25;
    auto custom_run = [&]() { custom_all_reduce.AllReduce<T>(stream, self_data, result, data_size); };
    float duration_ms = MeasureCudaExecutionTime(custom_run, stream, warmup_iters, run_iters);

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    barrier.Wait();
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));
    return duration_ms;
  }

  template <typename T>
  void RunCustomAllReducePerformance() {
    int device_count = -1;
    CHECK_NVIDIA_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (device_count < 2 || device_count > 8 || device_count % 2 != 0) {
      GTEST_SKIP_("Performance benchmark is just for 2,4,6,8 GPUs");
    }

    bool is_full_nvlink = true;
    for (int i = 0; i < device_count; ++i) {
      if (GetNvLinkVersion(0, i) == 0) {
        is_full_nvlink = false;
        break;
      }
    }
    if (device_count > 2 && !is_full_nvlink) {
      GTEST_SKIP_("Skipping performance benchmark because NVLink is not fully connected.");
    }

    uint32_t group_size = 2;
    if (device_count % static_cast<int>(group_size) != 0) {
      GTEST_SKIP_("Device count must be divisible by group size.");
    }

    std::vector<PerformanceShape> shapes = {
        {1, 16, 4096},
        {1, 32, 7168},
        {2, 64, 7168},
        {4, 128, 8192},
        {8, 128, 8192},
        {16, 256, 8192},
        {32, 128, 8192},
        {32, 256, 8192},
        {64, 512, 8192},
        {128, 512, 8192},
    };

    Barrier barrier(device_count);
    std::vector<void *> signals(static_cast<size_t>(device_count), nullptr);
    std::vector<void *> data_handles(static_cast<size_t>(device_count), nullptr);
    std::vector<void *> input_handles(static_cast<size_t>(device_count), nullptr);
    std::vector<float> non_group_timings(shapes.size() * device_count, 0.0f);
    std::vector<float> group_timings(shapes.size() * device_count, 0.0f);

    std::vector<std::shared_ptr<std::thread>> run_threads;
    for (int rank = 0; rank < device_count; ++rank) {
      run_threads.emplace_back(std::shared_ptr<std::thread>(new std::thread([&, rank]() {
        CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(rank));
        for (int peer = 0; peer < device_count; ++peer) {
          if (peer == rank) {
            continue;
          }
          auto err = cudaDeviceEnablePeerAccess(peer, 0);
          if (err != cudaErrorPeerAccessAlreadyEnabled) {
            CHECK_NVIDIA_CUDA_ERROR(err);
          }
        }

        for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
          const auto &shape = shapes[shape_idx];
          size_t data_size = shape.NumElements();
          float non_group_ms = RunCustomAllReducePerformanceOnce<T>(rank, 
            device_count, data_size, signals.data(), data_handles.data(),
          input_handles.data(), barrier, is_full_nvlink, false, group_size);
          float group_ms = RunCustomAllReducePerformanceOnce<T>(rank, device_count, data_size,
            signals.data(), data_handles.data(),
     input_handles.data(), barrier, is_full_nvlink, true, group_size);
          non_group_timings[shape_idx * device_count + rank] = non_group_ms;
          group_timings[shape_idx * device_count + rank] = group_ms;
        }

        for (int peer = 0; peer < device_count; ++peer) {
          if (peer == rank) {
            continue;
          }
          auto err = cudaDeviceDisablePeerAccess(peer);
          if (err != cudaErrorPeerAccessNotEnabled) {
            CHECK_NVIDIA_CUDA_ERROR(err);
          }
        }
      })));
    }

    for (auto &thread : run_threads) {
      thread->join();
    }

    for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
      double non_group_avg = 0.0;
      double group_avg = 0.0;
      for (int rank = 0; rank < device_count; ++rank) {
        non_group_avg += non_group_timings[shape_idx * device_count + rank];
        group_avg += group_timings[shape_idx * device_count + rank];
      }
      non_group_avg /= device_count;
      group_avg /= device_count;
      double speedup = (group_avg > 1e-6) ? (non_group_avg / group_avg) : 0.0;
      printf("[CustomAllReducePerf] shape(%s) is_group=false: %.3f ms | is_group=true: %.3f ms | speedup: %.2fx\n",
             shapes[shape_idx].Description().c_str(), non_group_avg, group_avg, speedup);
    }
  }

  template <typename T>
  void TestCustomGroupVsNonGroupAccuracy() {
    if (skip_test) {
      GTEST_SKIP();
    }

    int device_count = -1;
    CHECK_NVIDIA_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (device_count < 2 || device_count > 8 || device_count % 2 != 0) {
      GTEST_SKIP_("This comparison test is just for 2,4,6,8 GPUs");
    }

    bool is_full_nvlink = true;
    for (int i = 0; i < device_count; ++i) {
      if (GetNvLinkVersion(0, i) == 0) {
        is_full_nvlink = false;
        break;
      }
    }
    if (device_count > 2 && !is_full_nvlink) {
      GTEST_SKIP_("Skipping because NVLink is not fully connected.");
    }

    const size_t hidden_size = 7168;
    const std::vector<size_t> tokens = {16, 32};

    Barrier barrier(device_count);
    std::vector<void *> signals(static_cast<size_t>(device_count), nullptr);
    std::vector<void *> input_handles(static_cast<size_t>(device_count), nullptr);

    auto run_compare_thread = [&](int rank) {
      CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(rank));
      for (int peer = 0; peer < device_count; ++peer) {
        if (peer == rank) continue;
        auto err = cudaDeviceEnablePeerAccess(peer, 0);
        if (err != cudaErrorPeerAccessAlreadyEnabled) {
          CHECK_NVIDIA_CUDA_ERROR(err);
        }
      }

      for (size_t t : tokens) {
        size_t data_size = t * hidden_size;

        cudaStream_t stream;
        CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        BufferMeta result_non_group = CreateBuffer<T>(MemoryType::MEMORY_GPU, {data_size}, false);
        BufferMeta result_group = CreateBuffer<T>(MemoryType::MEMORY_GPU, {data_size}, false);

        size_t largest_part = (data_size * sizeof(T)) / device_count + (data_size * sizeof(T)) % device_count;
        BufferMeta signal_meta =
            CreateBuffer<T>(MemoryType::MEMORY_GPU, {(sizeof(Signal) + largest_part) / sizeof(T)}, false);
        signals[rank] = signal_meta.data_ptr;

        BufferMeta self_data_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {data_size}, true);
        T *self_data = static_cast<T *>(self_data_meta.data_ptr);
        input_handles[rank] = self_data;

        barrier.Wait();

        size_t rank_data_sz = 8 * 1024 * 1024;
        BufferMeta rank_data_meta =
            CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(rank_data_sz / sizeof(T))}, false);
        void *rank_data = rank_data_meta.data_ptr;

        {
          uint32_t root_rank = 0;
          bool is_group = false;
          int effective_world = device_count;
          CustomAllreduce car(rank_data, rank_data_sz, rank, effective_world, is_full_nvlink, root_rank, is_group);
          car.RegisterSignalBuffer(reinterpret_cast<Signal **>(signals.data()));
          std::vector<void *> ptrs(static_cast<size_t>(device_count), nullptr);
          for (int i = 0; i < device_count; ++i) ptrs[static_cast<size_t>(i)] = input_handles[i];
          car.RegisterBuffer(ptrs.data(), stream);
          car.AllReduce<T>(stream, self_data, static_cast<T *>(result_non_group.data_ptr), data_size);
          CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
        }

        {
          uint32_t root_rank = (rank / device_count) * device_count;
          bool is_group = true;
          int effective_world = device_count;  // group_size == world
          CustomAllreduce car(rank_data, rank_data_sz, rank, effective_world, is_full_nvlink, root_rank, is_group);
          car.RegisterSignalBuffer(reinterpret_cast<Signal **>(signals.data()));
          std::vector<void *> ptrs(static_cast<size_t>(device_count), nullptr);
          for (int i = 0; i < device_count; ++i) ptrs[static_cast<size_t>(i)] = input_handles[i];
          car.RegisterBuffer(ptrs.data(), stream);
          car.AllReduce<T>(stream, self_data, static_cast<T *>(result_group.data_ptr), data_size);
          CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
        }

        BufferMeta host_non_group = CopyToHost<T>(result_non_group);
        BufferMeta host_group = CopyToHost<T>(result_group);

        constexpr float atol = std::is_same<T, half>::value ? 1e-5f : std::is_same<T, __nv_bfloat16>::value ? 5e-3f : 1e-5f;
        constexpr float rtol = std::is_same<T, half>::value ? 1e-5f : std::is_same<T, __nv_bfloat16>::value ? 5e-3f : 1e-5f;
        EXPECT_TRUE(CheckResult<T>(
            "CustomAllReduce_GroupVsNonGroup_" + std::to_string(data_size * sizeof(T)), host_group, host_non_group,
            atol, rtol));

        CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));
      }

      for (int peer = 0; peer < device_count; ++peer) {
        if (peer == rank) continue;
        auto err = cudaDeviceDisablePeerAccess(peer);
        if (err != cudaErrorPeerAccessNotEnabled) {
          CHECK_NVIDIA_CUDA_ERROR(err);
        }
      }
    };

    std::vector<std::shared_ptr<std::thread>> threads;
    for (int rank = 0; rank < device_count; ++rank) {
      threads.emplace_back(std::shared_ptr<std::thread>(new std::thread(run_compare_thread, rank)));
    }
    for (auto &th : threads) {
      th->join();
    }
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

TEST_F(LlamaNvidiaCustomAllReduceTestSuit, DISABLED_FloatCustomAllReducePerformanceTest) {
  if (!skip_test) {
    RunCustomAllReducePerformance<float>();
  }
}

TEST_F(LlamaNvidiaCustomAllReduceTestSuit, HalfCustomAllReduceGroupVsNonGroupAccuracy) {
  if (!skip_test) {
    TestCustomGroupVsNonGroupAccuracy<half>();
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
