/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <nccl.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <type_traits>

#include "csrc/kernels/nvidia/fused_add_norm/fused_add_norm.h"
#include "csrc/kernels/nvidia/others/tensorrt-llm/main/communication_kernels/trtllm_all_reduce.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"


using namespace llm_kernels::utils;

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

class LlamaNvidiaTrtllmAllReduceTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    skip_float_test = GetSMVersion() < 90;
    skip_test = GetSMVersion() < 89;
    device_count = GetDeviceCount();
    if (device_count < 2 || device_count > 8 || device_count % 2 != 0 || !EnableGpuP2PAccess(device_count)) {
      skip_test = true;
    }
    if (skip_test) {
      return;
    }

    NvidiaTestSuitBase::SetUp();

    // Init nccl
    ncclGetUniqueId(&nccl_id);
    nccl_comms.resize(device_count);

    // Init stream
    streams.resize(device_count);

    // Init trt allreduce buffer
    buffer_d_ptrs.resize(3 * device_count);
    flag_d_ptrs.resize(device_count);
    workspace_d_ptrs.resize(device_count);

    std::atomic<int> counter{0};
    auto Initialize = [&](int cur_rank) {
      CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));

      // Init nccl
      NCCLCHECK(ncclCommInitRank(&nccl_comms[cur_rank], device_count, nccl_id, cur_rank));

      cudaStream_t& stream = streams[cur_rank];
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

      // Alloc allreduce fusion workspace
      // To test float, we always use `T = float`
      AllocTrtAllReduceWorkspace(device_count, cur_rank, max_token_num, hidden_dim, sizeof(float), buffer_d_ptrs,
                                 flag_d_ptrs, workspace_d_ptrs, stream);

      // Synchronize across all GPUs to ensure the previous step has completed
      for (++counter; counter != device_count;)
        ;

      // Init allreduce fusion workspace
      InitTrtAllReduceWorkspace(device_count, cur_rank, buffer_d_ptrs, flag_d_ptrs, workspace_d_ptrs, stream);
    };
    std::vector<std::unique_ptr<std::thread>> run_threads;
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads.emplace_back(std::make_unique<std::thread>(Initialize, cur_rank));
    }
    for (int cur_rank = 0; cur_rank < device_count; ++cur_rank) {
      run_threads[cur_rank]->join();
    }
  }

  void TearDown() override {
    if (skip_test) {
      return;
    }

    // Free
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));
      NCCLCHECK(ncclCommDestroy(nccl_comms[cur_rank]));
      FreeTrtAllReduceWorkspace(device_count, cur_rank, buffer_d_ptrs, flag_d_ptrs, workspace_d_ptrs,
                                streams[cur_rank]);
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(streams[cur_rank]));
    }

    NvidiaTestSuitBase::TearDown();
  }

 protected:
  int device_count = 0;
  bool skip_test = false;
  bool skip_float_test = false;
  
  ncclUniqueId nccl_id;
  std::vector<ncclComm_t> nccl_comms;

  std::vector<cudaStream_t> streams;

  const int max_token_num = 2048;
  const int hidden_dim = 7168;  // Config of DeepSeek-V3
  std::vector<void*> buffer_d_ptrs;
  std::vector<void*> flag_d_ptrs;
  std::vector<void*> workspace_d_ptrs;

 protected:
  template <typename T>
  void RunTrtllmAllReduceThread(int cur_rank, int token_num, bool perf) {
    CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));

    std::string type_str = "float";
    ncclDataType_t ncclDtype = ncclFloat;
    if constexpr (std::is_same_v<T, half>) {
      type_str = "half";
      ncclDtype = ncclFloat16;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      type_str = "bfloat16";
      ncclDtype = ncclBfloat16;
    }

    // Prepare device data
    BufferMeta d_input =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    BufferMeta d_output =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ false);
    BufferMeta d_output_ref =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ false);

    cudaStream_t& stream = streams[cur_rank];

    const int warmups = perf ? 5 : 0;
    const int iterations = perf ? 10 : 1;

    // Run nccl version
    auto nccl_run = [&]() {
      NCCLCHECK(ncclAllReduce(reinterpret_cast<T*>(d_input.data_ptr), reinterpret_cast<T*>(d_output_ref.data_ptr),
                              token_num * hidden_dim, ncclDtype, ncclSum, nccl_comms[cur_rank], stream));
    };
    const float elapsed_ms_nccl = MeasureCudaExecutionTime(nccl_run, stream, warmups, iterations);
    if (cur_rank == 0 && perf) {
      std::cout << "Token num: " << token_num << ", Execution time of nccl allreduce " << type_str << ": "
                << elapsed_ms_nccl << " ms" << std::endl;
    }

    // Run trt version
    AllReduceFusionParams<T> params;
    params.nranks = device_count;
    params.rank = cur_rank;
    params.size = token_num * hidden_dim;
    params.hidden_dim = hidden_dim;
    params.workspace = reinterpret_cast<void**>(workspace_d_ptrs[cur_rank]);
    params.allreduce_in = d_input.data_ptr;
    params.allreduce_out = d_output.data_ptr;
    params.stream = stream;
    params.pattern = AllReduceFusionPattern::kAllReduce;
    auto trt_run = [&]() { allreduce_fusion_op(params); };
    // one shot
    params.use_oneshot = true;
    const float elapsed_ms_trt_oneshot = MeasureCudaExecutionTime(trt_run, stream, warmups, iterations);
    if (!perf) {
      EXPECT_TRUE(CheckResult<T>("trt_oneshot_allreduce" + type_str + "_token_num_" + std::to_string(token_num),
                                 d_output, d_output_ref, 1e-4, 1e-4));
    }
    if (cur_rank == 0 && perf) {
      std::cout << "Token num: " << token_num << ", Execution time of trt oneshot allreduce " << type_str << ": "
                << elapsed_ms_trt_oneshot << " ms" << std::endl;
    }
    // two shot
    params.use_oneshot = false;
    const float elapsed_ms_trt_twoshot = MeasureCudaExecutionTime(trt_run, stream, warmups, iterations);
    if (!perf) {
      EXPECT_TRUE(CheckResult<T>("trt_twoshot_allreduce" + type_str + "_token_num_" + std::to_string(token_num),
                                 d_output, d_output_ref, 1e-4, 1e-4));
    }
    if (cur_rank == 0 && perf) {
      std::cout << "Token num: " << token_num << ", Execution time of trt twoshot allreduce " << type_str << ": "
                << elapsed_ms_trt_twoshot << " ms" << std::endl;
    }

    // Free device data
    DeleteBuffer(d_input);
    DeleteBuffer(d_output);
    DeleteBuffer(d_output_ref);
  }

  template <typename T>
  void RunTrtllmAllReduceResidualNormThread(int cur_rank, int token_num, bool perf) {
    CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));

    std::string type_str = "float";
    ncclDataType_t ncclDtype = ncclFloat;
    if constexpr (std::is_same_v<T, half>) {
      type_str = "half";
      ncclDtype = ncclFloat16;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      type_str = "bfloat16";
      ncclDtype = ncclBfloat16;
    }

    // Prepare device data
    BufferMeta d_input =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    BufferMeta d_output_ref =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    BufferMeta d_output1 =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    BufferMeta d_output2 =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    BufferMeta d_residual_ref =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    BufferMeta d_residual1 =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    BufferMeta d_residual2 =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    BufferMeta d_rms_gamma =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    constexpr float rms_eps = 1e-6;

    cudaStream_t& stream = streams[cur_rank];

    const int warmups = perf ? 5 : 0;
    const int iterations = perf ? 10 : 1;

    // Run fused residual norm + nccl version
    auto nccl_run = [&]() {
      NCCLCHECK(ncclAllReduce(reinterpret_cast<T*>(d_input.data_ptr), reinterpret_cast<T*>(d_output_ref.data_ptr),
                              token_num * hidden_dim, ncclDtype, ncclSum, nccl_comms[cur_rank], stream));
      InvokeFusedAddRMSNorm<T>(d_output_ref.data_ptr, d_residual_ref.data_ptr, d_rms_gamma.data_ptr, rms_eps, false,
                               token_num, hidden_dim, stream);
    };
    const float elapsed_ms_nccl = MeasureCudaExecutionTime(nccl_run, stream, warmups, iterations);
    if (cur_rank == 0 && perf) {
      std::cout << "Token num: " << token_num << ", Execution time of nccl allreduce residual norm " << type_str << ": "
                << elapsed_ms_nccl << " ms" << std::endl;
    }

    // Run trt fusion version
    AllReduceFusionParams<T> params;
    params.nranks = device_count;
    params.rank = cur_rank;
    params.size = token_num * hidden_dim;
    params.hidden_dim = hidden_dim;
    params.workspace = reinterpret_cast<void**>(workspace_d_ptrs[cur_rank]);
    params.allreduce_in = d_input.data_ptr;
    params.rms_gamma = d_rms_gamma.data_ptr;
    params.rms_eps = rms_eps;
    params.stream = stream;
    params.pattern = AllReduceFusionPattern::kARResidualRMSNorm;
    auto trt_run = [&]() { allreduce_fusion_op(params); };
    // one shot
    params.use_oneshot = true;
    params.residual_in = d_residual1.data_ptr;
    params.residual_out = d_residual1.data_ptr;  // inplace
    params.norm_out = d_output1.data_ptr;
    const float elapsed_ms_trt_oneshot = MeasureCudaExecutionTime(trt_run, stream, warmups, iterations);
    if (!perf) {
      EXPECT_TRUE(
          CheckResult<T>("trt_oneshot_allreduce_residual_norm" + type_str + "_token_num_" + std::to_string(token_num),
                         d_output1, d_output_ref, 5e-4, 5e-4));
      EXPECT_TRUE(
          CheckResult<T>("trt_oneshot_allreduce_residual_norm" + type_str + "_token_num_" + std::to_string(token_num),
                         d_residual1, d_residual_ref, 5e-4, 5e-4));
    }
    if (cur_rank == 0 && perf) {
      std::cout << "Token num: " << token_num << ", Execution time of trt oneshot allreduce residual norm" << type_str
                << ": " << elapsed_ms_trt_oneshot << " ms" << std::endl;
    }
    // two shot
    params.use_oneshot = false;
    params.residual_in = d_residual2.data_ptr;
    params.residual_out = d_residual2.data_ptr;  // inplace
    params.norm_out = d_output2.data_ptr;
    const float elapsed_ms_trt_twoshot = MeasureCudaExecutionTime(trt_run, stream, warmups, iterations);
    if (!perf) {
      EXPECT_TRUE(
          CheckResult<T>("trt_oneshot_allreduce_residual_norm" + type_str + "_token_num_" + std::to_string(token_num),
                         d_output2, d_output_ref, 5e-4, 5e-4));
      EXPECT_TRUE(
          CheckResult<T>("trt_oneshot_allreduce_residual_norm" + type_str + "_token_num_" + std::to_string(token_num),
                         d_residual2, d_residual_ref, 5e-4, 5e-4));
    }
    if (cur_rank == 0 && perf) {
      std::cout << "Token num: " << token_num << ", Execution time of trt twoshot allreduce residual norm" << type_str
                << ": " << elapsed_ms_trt_twoshot << " ms" << std::endl;
    }

    // Free device data
    DeleteBuffer(d_input);
    DeleteBuffer(d_output_ref);
    DeleteBuffer(d_output1);
    DeleteBuffer(d_output2);
    DeleteBuffer(d_residual_ref);
    DeleteBuffer(d_residual1);
    DeleteBuffer(d_residual2);
    DeleteBuffer(d_rms_gamma);
  }

  template <typename T>
  void RunTrtllmAllReduce(int token_num, bool perf = false) {
    std::vector<std::unique_ptr<std::thread>> run_threads;
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads.emplace_back(std::make_unique<std::thread>(
          &LlamaNvidiaTrtllmAllReduceTestSuit::RunTrtllmAllReduceThread<T>, this, cur_rank, token_num, perf));
    }
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads[cur_rank]->join();
    }
  }

  template <typename T>
  void RunTrtllmAllReduceResidualNorm(int token_num, bool perf = false) {
    std::vector<std::unique_ptr<std::thread>> run_threads;
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads.emplace_back(
          std::make_unique<std::thread>(&LlamaNvidiaTrtllmAllReduceTestSuit::RunTrtllmAllReduceResidualNormThread<T>,
                                        this, cur_rank, token_num, perf));
    }
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads[cur_rank]->join();
    }
  }
};

TEST_F(LlamaNvidiaTrtllmAllReduceTestSuit, TrtllmAllReduceAccTest) {
  if (skip_test) {
    return;
  }
  for (const int token_num : {32, 256}) {
    if (!skip_float_test) {
      RunTrtllmAllReduce<float>(token_num);
    }
    RunTrtllmAllReduce<half>(token_num);
    RunTrtllmAllReduce<__nv_bfloat16>(token_num);
  }
}

TEST_F(LlamaNvidiaTrtllmAllReduceTestSuit, DISABLED_TrtllmAllReducePerfTest) {
  if (skip_test) {
    return;
  }
  const std::vector<int> token_nums = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
  if (!skip_float_test) {
    for (const int token_num : token_nums) {
      RunTrtllmAllReduce<float>(token_num, /*perf*/ true);
    }
  }
  for (const int token_num : token_nums) {
    RunTrtllmAllReduce<half>(token_num, /*perf*/ true);
  }
  for (const int token_num : token_nums) {
    RunTrtllmAllReduce<__nv_bfloat16>(token_num, /*perf*/ true);
  }
}

TEST_F(LlamaNvidiaTrtllmAllReduceTestSuit, TrtllmAllReduceResidualNormAccTest) {
  if (skip_test) {
    return;
  }
  for (const int token_num : {32, 256}) {
    if (!skip_float_test) {
      RunTrtllmAllReduceResidualNorm<float>(token_num);
    }
    RunTrtllmAllReduceResidualNorm<half>(token_num);
    RunTrtllmAllReduceResidualNorm<__nv_bfloat16>(token_num);
  }
}

TEST_F(LlamaNvidiaTrtllmAllReduceTestSuit, DISABLED_TrtllmAllReduceResidualNormPerfTest) {
  if (skip_test) {
    return;
  }
  const std::vector<int> token_nums = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
  if (!skip_float_test) {
    for (const int token_num : token_nums) {
      RunTrtllmAllReduceResidualNorm<float>(token_num, /*perf*/ true);
    }
  }
  for (const int token_num : token_nums) {
    RunTrtllmAllReduceResidualNorm<half>(token_num, /*perf*/ true);
  }
  for (const int token_num : token_nums) {
    RunTrtllmAllReduceResidualNorm<__nv_bfloat16>(token_num, /*perf*/ true);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
