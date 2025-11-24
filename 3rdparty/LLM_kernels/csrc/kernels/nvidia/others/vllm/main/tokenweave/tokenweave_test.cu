/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <type_traits>

#include "csrc/kernels/nvidia/fused_add_norm/fused_add_norm.h"
#include "csrc/kernels/nvidia/others/vllm/main/tokenweave/tokenweave_fused_kernels.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

using namespace llm_kernels::utils;

#define ROUND_UP(n, multiple) (((n + multiple - 1) / multiple) * multiple)

union IpcMemHandle {
  uint64_t fd;
  CUmemFabricHandle fh;
};

struct NvlsHandle {
  size_t size = 0;
  // Device pointers used by kernels
  uintptr_t uc_ptr = 0;
  uintptr_t mc_ptr = 0;
  // Device pointers
  CUdeviceptr uc_va;
  CUdeviceptr mc_va;
  std::vector<CUdeviceptr> peer_uc_vas;
  // Device allocation handles
  std::vector<CUmemGenericAllocationHandle> peer_uc_handles;
  CUmemGenericAllocationHandle uc_handle;
  CUmemGenericAllocationHandle mc_handle;
};

class LlamaNvidiaTokenWeaveTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    skip_test = GetSMVersion() < 90;
    device_count = GetDeviceCount();
    if (device_count < 2 || device_count > 8 || device_count % 2 != 0 || !EnableGpuP2PAccess(device_count)) {
      skip_test = true;
    }
    for (int cur_rank = 0; cur_rank < device_count && !skip_test; cur_rank++) {
      int multicast_supported = 0;
      CUCHECK(cuDeviceGetAttribute(&multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cur_rank));
      if (!multicast_supported) {
        skip_test = true;
      }
    }
    if (skip_test) {
      return;
    }

    NvidiaTestSuitBase::SetUp();

    // Init nccl
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    nccl_comms.resize(device_count);
    auto InitNccl = [&](int cur_rank) {
      CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));
      NCCLCHECK(ncclCommInitRank(&nccl_comms[cur_rank], device_count, nccl_id, cur_rank));
    };
    std::vector<std::unique_ptr<std::thread>> run_threads;
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads.emplace_back(std::make_unique<std::thread>(InitNccl, cur_rank));
    }
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads[cur_rank]->join();
    }
    run_threads.clear();

    // Init stream
    streams.resize(device_count);
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));
      cudaStream_t& stream = streams[cur_rank];
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    const testing::TestInfo* test_info = testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = test_info->name();
    const size_t max_token_num = test_name.find("Perf") != std::string::npos ? 65536 : 32;
    const size_t max_buffer_size = max_token_num * hidden_dim * sizeof(__nv_bfloat16);
    constexpr size_t kSignalPadAlignment = 16UL;
    const size_t signal_pad_offset = (max_buffer_size + kSignalPadAlignment - 1) & ~(kSignalPadAlignment - 1);

    // Init multimem
    nvls_handles.resize(device_count);
    ipc_handles.resize(device_count);
    signal_pads_dev.resize(device_count);
    counter = 0;
    auto InitMultimem = [&](int cur_rank) {
      CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));
      AllocNvlsMcastMem(cur_rank, signal_pad_offset + kSignalPadSize);
    };
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads.emplace_back(std::make_unique<std::thread>(InitMultimem, cur_rank));
    }
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads[cur_rank]->join();
    }

    // Initialize signal pads
    std::vector<std::vector<CUdeviceptr>> signal_pads(device_count);
    signal_pads_dev.resize(device_count);
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      CHECK_NVIDIA_CUDA_ERROR(cudaSetDevice(cur_rank));
      signal_pads[cur_rank].resize(device_count);
      for (int oth_rank = 0; oth_rank < device_count; oth_rank++) {
        signal_pads[cur_rank][oth_rank] = nvls_handles[cur_rank]->peer_uc_vas[oth_rank] + signal_pad_offset;
        if (oth_rank == cur_rank) {
          CUCHECK(cuMemsetD8(signal_pads[cur_rank][oth_rank], 0, kSignalPadSize));
        }
      }
      // Copy host array of pointers to device array
      CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(signal_pads_dev.data() + cur_rank, device_count * sizeof(CUdeviceptr)));
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(signal_pads_dev[cur_rank], signal_pads[cur_rank].data(),
                                         device_count * sizeof(CUdeviceptr), cudaMemcpyHostToDevice));
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
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(streams[cur_rank]));
      if (NvlsHandle* nvls_handle = nvls_handles[cur_rank]; nvls_handle != nullptr) {
        CUCHECK(cuMemUnmap(nvls_handle->mc_va, nvls_handle->size));
        CUCHECK(cuMemRelease(nvls_handle->mc_handle));
        CUCHECK(cuMemAddressFree(nvls_handle->mc_va, nvls_handle->size));
        for (size_t i = 0; i < nvls_handle->peer_uc_vas.size(); ++i) {
          CUCHECK(cuMemUnmap(nvls_handle->peer_uc_vas[i], nvls_handle->size));
          CUCHECK(cuMemRelease(nvls_handle->peer_uc_handles[i]));
          CUCHECK(cuMemAddressFree(nvls_handle->peer_uc_vas[i], nvls_handle->size));
        }
        delete nvls_handle;
      }
      if (void** signal_pad_dev = signal_pads_dev[cur_rank]; signal_pad_dev != nullptr) {
        CHECK_NVIDIA_CUDA_ERROR(cudaFree(signal_pad_dev));
      }
    }

    NvidiaTestSuitBase::TearDown();
  }

 protected:
  int device_count = 0;
  bool skip_test = false;

  std::vector<ncclComm_t> nccl_comms;

  std::vector<cudaStream_t> streams;

  const int hidden_dim = 7168;  // Config of DeepSeek-V3

  // For intra-node communication handle
  std::vector<NvlsHandle*> nvls_handles;
  std::vector<IpcMemHandle> ipc_handles;
  // Device array of signal pad pointers
  std::vector<void**> signal_pads_dev;

  // For synchronization
  std::atomic<int> counter;

  static constexpr size_t kSignalPadSize = 2048;

 protected:
  static CUmemAllocationHandleType GetMemHandleType(const int device_id) {
    // Check if fabric handle support is available.
    int fabric_supported = 0;
    CUCHECK(cuDeviceGetAttribute(&fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device_id));
    if (!fabric_supported) {
      return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    nvmlDevice_t nvml_device;
    nvmlGpuFabricInfo_t fabric_info;
    NVMLCHECK(nvmlInit_v2());
    NVMLCHECK(nvmlDeviceGetHandleByIndex(device_id, &nvml_device));
    NVMLCHECK(nvmlDeviceGetGpuFabricInfo(nvml_device, &fabric_info));
    NVMLCHECK(nvmlShutdown());

    // Check if the fabric is fully initialized.
    if (fabric_info.state != NVML_GPU_FABRIC_STATE_COMPLETED || fabric_info.status != NVML_SUCCESS) {
      return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    // Check that fabric handles can be created.
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(CUmemAllocationProp));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    size_t alloc_size = 1024;  // anything > 0
    size_t min_gran = 0;
    CUCHECK(cuMemGetAllocationGranularity(&min_gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    alloc_size = ROUND_UP(alloc_size, min_gran);

    CUmemGenericAllocationHandle handle;
    CUresult err = cuMemCreate(&handle, alloc_size, &prop, 0);
    if (err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED) {
      return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    // Check if fabric handles can be exported & imported by IMEX (Internode Memory Exchange)
    CUmemFabricHandle fh;
    err = cuMemExportToShareableHandle(&fh, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
    if (err != CUDA_SUCCESS ||
        (err = cuMemImportFromShareableHandle(&handle, &fh, CU_MEM_HANDLE_TYPE_FABRIC)) != CUDA_SUCCESS) {
      CUCHECK(cuMemRelease(handle));
      return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    CUCHECK(cuMemRelease(handle));
    // If we get here, fabric handles are supported.
    return CU_MEM_HANDLE_TYPE_FABRIC;
  }

  void AllocNvlsMcastMem(const int cur_rank, const size_t size) {
    auto& nvls_handle = nvls_handles[cur_rank];
    nvls_handle = new NvlsHandle();
    nvls_handle->size = size;

    int CU_dev;
    CUCHECK(cuDeviceGet(&CU_dev, cur_rank));

    // Get handle type used to share memory handles between devices.
    auto handle_type = GetMemHandleType(cur_rank);

    // Define allocation access permissions (same for unicast and multicast).
    CUmemAccessDesc access_desc;
    memset(&access_desc, 0, sizeof(CUmemAccessDesc));
    access_desc.location.id = cur_rank;
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Define unicast allocation properties.
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(CUmemAllocationProp));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cur_rank;
    prop.requestedHandleTypes = handle_type;

    // Define multicast allocation properties.
    CUmulticastObjectProp mcprop;
    memset(&mcprop, 0, sizeof(CUmulticastObjectProp));
    mcprop.numDevices = device_count;
    mcprop.handleTypes = handle_type;
    mcprop.flags = 0;

    // Round up allocation size to the nearest multiple of the unicast allocation granularity.
    size_t granularity = 0;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    nvls_handle->size = ROUND_UP(nvls_handle->size, granularity);

    // Round up allocation size to the nearest multiple of the multicast allocation granularity.
    size_t mc_granularity = 0;
    CUCHECK(cuMulticastGetGranularity(&mc_granularity, &mcprop, CU_MULTICAST_GRANULARITY_MINIMUM));
    nvls_handle->size = ROUND_UP(nvls_handle->size, mc_granularity);
    mcprop.size = nvls_handle->size;

    // Allocate physical pages of memory on GPU.
    CUCHECK(cuMemCreate(&nvls_handle->uc_handle, nvls_handle->size, &prop, 0));
    // Reserve unicast virtual address space for the memory.
    CUCHECK(cuMemAddressReserve(&nvls_handle->uc_va, nvls_handle->size, granularity, 0U, 0));
    // Map the unicast virtual address space to the physical pages.
    CUCHECK(cuMemMap(nvls_handle->uc_va, nvls_handle->size, 0, nvls_handle->uc_handle, 0));
    // Set the access permissions for the unicast memory.
    CUCHECK(cuMemSetAccess(nvls_handle->uc_va, nvls_handle->size, &access_desc, 1));
    nvls_handle->uc_ptr = reinterpret_cast<uintptr_t>((void*)nvls_handle->uc_va);

    // Unicast pointer exchange between ranks.
    IpcMemHandle& ipc_handle = ipc_handles[cur_rank];
    CUCHECK(cuMemExportToShareableHandle((void*)&ipc_handle, nvls_handle->uc_handle, handle_type, /*flags*/ 0));

    nvls_handle->peer_uc_vas.resize(device_count);
    nvls_handle->peer_uc_handles.resize(device_count);

    // Synchronize across all GPUs to ensure the previous step has completed
    int step = 1;
    for (++counter; counter < step * device_count;)
      ;
    for (int i = 0; i < device_count; i++) {
      if (i != cur_rank) {
        IpcMemHandle peer_ipc_handle = ipc_handles[i];
        void* os_handle =
            handle_type == CU_MEM_HANDLE_TYPE_FABRIC ? (void*)&peer_ipc_handle : (void*)peer_ipc_handle.fd;
        CUCHECK(cuMemImportFromShareableHandle(&nvls_handle->peer_uc_handles[i], os_handle, handle_type));
        // Reserve peer unicast virtual address space for the memory.
        CUCHECK(cuMemAddressReserve(&nvls_handle->peer_uc_vas[i], nvls_handle->size, granularity, 0U, 0));
        // Map the peer unicast virtual address space to the physical pages.
        CUCHECK(cuMemMap(nvls_handle->peer_uc_vas[i], nvls_handle->size, 0, nvls_handle->peer_uc_handles[i], 0));
        // Set the access permissions for the peer unicast memory.
        CUCHECK(cuMemSetAccess(nvls_handle->peer_uc_vas[i], nvls_handle->size, &access_desc, 1));
      } else {
        nvls_handle->peer_uc_vas[i] = nvls_handle->uc_va;
        nvls_handle->peer_uc_handles[i] = nvls_handle->uc_handle;
      }
    }

    // Initialize multicast object for all ranks.
    if (cur_rank == 0) {
      CUCHECK(cuMulticastCreate(&nvls_handle->mc_handle, &mcprop));
      // Export the allocation for the importing process.
      CUCHECK(cuMemExportToShareableHandle(&ipc_handle, nvls_handle->mc_handle, handle_type, /*flags*/ 0));
      // Synchronize across all GPUs to ensure the previous step has completed
      ++step;
      for (++counter; counter < step * device_count;)
        ;
    } else {
      // Synchronize across all GPUs to ensure the previous step has completed
      ++step;
      for (++counter; counter < step * device_count;)
        ;
      IpcMemHandle ipc_handle = ipc_handles[0];
      void* os_handle = handle_type == CU_MEM_HANDLE_TYPE_FABRIC ? (void*)&ipc_handle : (void*)ipc_handle.fd;
      CUCHECK(cuMemImportFromShareableHandle(&nvls_handle->mc_handle, os_handle, handle_type));
    }

    // Add device to multicast object
    CUCHECK(cuMulticastAddDevice(nvls_handle->mc_handle, CU_dev));
    // Bind physical memory to the Multicast group.
    // Note: It will block until all ranks have been added to the group.
    CUCHECK(cuMulticastBindMem(nvls_handle->mc_handle, 0, nvls_handle->uc_handle, 0, nvls_handle->size, 0));
    // Reserve multicast virtual address space for the memory.
    CUCHECK(cuMemAddressReserve(&nvls_handle->mc_va, nvls_handle->size, mc_granularity, 0U, 0));
    // Map the multicast virtual address space to the physical pages.
    CUCHECK(cuMemMap(nvls_handle->mc_va, nvls_handle->size, 0, nvls_handle->mc_handle, 0));
    // Set the access permissions for the multicast memory.
    CUCHECK(cuMemSetAccess(nvls_handle->mc_va, nvls_handle->size, &access_desc, 1 /* count */));
    nvls_handle->mc_ptr = reinterpret_cast<uintptr_t>((void*)nvls_handle->mc_va);
  }

  template <typename T>
  void RunTokenWeaveThread(int cur_rank, int token_num, bool perf) {
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
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(nvls_handles[cur_rank]->uc_ptr), d_input.data_ptr,
                                       token_num * hidden_dim * sizeof(T), cudaMemcpyDeviceToDevice));
    BufferMeta d_output_ref =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)});
    BufferMeta d_output =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)});
    BufferMeta d_residual_ref =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    BufferMeta d_residual =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)});
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_residual.data_ptr, d_residual_ref.data_ptr, token_num * hidden_dim * sizeof(T),
                                       cudaMemcpyDeviceToDevice));
    BufferMeta d_residual_2 =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)});
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_residual_2.data_ptr, d_residual_ref.data_ptr,
                                       token_num * hidden_dim * sizeof(T), cudaMemcpyDeviceToDevice));
    BufferMeta d_rms_gamma =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(token_num), static_cast<size_t>(hidden_dim)},
                        /*is_random_init*/ true);
    constexpr float rms_eps = 1e-6;

    cudaStream_t& stream = streams[cur_rank];

    const int warmups = perf ? 5 : 0;
    const int iterations = perf ? 10 : 1;

    // Run tokenweave version
    auto tokenweave_run = [&]() {
      assert(token_num % device_count == 0);
      const int token_num_per_rank = token_num / device_count;
      const int offset = token_num_per_rank * cur_rank * hidden_dim;
      FusedRsLmAgCta<T>(nvls_handles[cur_rank]->mc_ptr + offset * sizeof(T),
                        reinterpret_cast<T*>(d_residual.data_ptr) + offset,
                        reinterpret_cast<const T*>(d_rms_gamma.data_ptr), signal_pads_dev[cur_rank], cur_rank,
                        device_count, rms_eps, token_num_per_rank, hidden_dim, stream);
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(d_output.data_ptr,
                                              reinterpret_cast<void*>(nvls_handles[cur_rank]->uc_ptr),
                                              token_num * hidden_dim * sizeof(T), cudaMemcpyDeviceToDevice, stream));
      // NOTE: When fusing norm, each rank's residual is only correct for its own computed portion, as no allgather is
      // performed on the residual
      if (!perf) {
        CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(reinterpret_cast<void*>(nvls_handles[cur_rank]->uc_ptr),
                                                d_input.data_ptr, token_num * hidden_dim * sizeof(T),
                                                cudaMemcpyDeviceToDevice, stream));
        FusedRsAgCta<T>(nvls_handles[cur_rank]->mc_ptr + offset * sizeof(T),
                        reinterpret_cast<T*>(d_residual_2.data_ptr) + offset, signal_pads_dev[cur_rank], cur_rank,
                        device_count, token_num_per_rank, hidden_dim, stream);
        CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(d_residual_2.data_ptr,
                                                reinterpret_cast<void*>(nvls_handles[cur_rank]->uc_ptr),
                                                token_num * hidden_dim * sizeof(T), cudaMemcpyDeviceToDevice, stream));
      }
    };
    const float elapsed_ms_tokenweave = MeasureCudaExecutionTime(tokenweave_run, stream, warmups, iterations);

    // Run fused residual norm + nccl version
    auto nccl_run = [&]() {
      NCCLCHECK(ncclAllReduce(reinterpret_cast<T*>(d_input.data_ptr), reinterpret_cast<T*>(d_output_ref.data_ptr),
                              token_num * hidden_dim, ncclDtype, ncclSum, nccl_comms[cur_rank], stream));
      InvokeFusedAddRMSNorm<T>(d_output_ref.data_ptr, d_residual_ref.data_ptr, d_rms_gamma.data_ptr, rms_eps, false,
                               token_num, hidden_dim, stream);
    };
    const float elapsed_ms_nccl = MeasureCudaExecutionTime(nccl_run, stream, warmups, iterations);

    if (!perf) {
      EXPECT_TRUE(CheckResult<T>("tokenweave_output_" + type_str + "_token_num_" + std::to_string(token_num), d_output,
                                 d_output_ref, 1e-3, 1e-3));
      EXPECT_TRUE(CheckResult<T>("tokenweave_residual_" + type_str + "_token_num_" + std::to_string(token_num),
                                 d_residual_2, d_residual_ref, 1e-3, 1e-3));
    } else if (cur_rank == 0) {
      std::cout << "Token num: " << token_num << ", Execution time of tokenweave allreduce residual norm " << type_str
                << ": " << elapsed_ms_tokenweave << " ms" << std::endl;
      std::cout << "Token num: " << token_num << ", Execution time of nccl allreduce residual norm " << type_str << ": "
                << elapsed_ms_nccl << " ms" << std::endl;
    }

    // Free device data
    DeleteBuffer(d_input);
    DeleteBuffer(d_output_ref);
    DeleteBuffer(d_output);
    DeleteBuffer(d_residual_ref);
    DeleteBuffer(d_residual);
    DeleteBuffer(d_residual_2);
    DeleteBuffer(d_rms_gamma);
  }

  template <typename T>
  void RunTokenWeave(int token_num, bool perf = false) {
    std::vector<std::unique_ptr<std::thread>> run_threads;
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads.emplace_back(std::make_unique<std::thread>(&LlamaNvidiaTokenWeaveTestSuit::RunTokenWeaveThread<T>,
                                                             this, cur_rank, token_num, perf));
    }
    for (int cur_rank = 0; cur_rank < device_count; cur_rank++) {
      run_threads[cur_rank]->join();
    }
  }
};

TEST_F(LlamaNvidiaTokenWeaveTestSuit, TokenWeaveAccTest) {
  if (skip_test) {
    return;
  }
  for (const int token_num : {8, 32}) {
    RunTokenWeave<__nv_bfloat16>(token_num);
    RunTokenWeave<half>(token_num);
  }
}

// Performance test is disabled by default
TEST_F(LlamaNvidiaTokenWeaveTestSuit, DISABLED_TokenWeavePerfTest) {
  if (skip_test) {
    return;
  }
  const std::vector<int> token_nums = {8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536};
  for (const int token_num : token_nums) {
    RunTokenWeave<__nv_bfloat16>(token_num, /*perf*/ true);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
