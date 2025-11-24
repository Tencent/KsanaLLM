/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "src/ksana_llm/utils/nvidia/cuda_utils.h"

#include <atomic>
#include <thread>

#include "src/ksana_llm/utils/nvidia/nccl_utils.h"

namespace ksana_llm {

NvlsMcastMemory* NvlsMcastMemory::GetInstance() {
  static NvlsMcastMemory instance;
  return &instance;
}

void NvlsMcastMemory::Initialize(const size_t device_count) {
  nvls_handles_.resize(device_count);
  ipc_handles_.resize(device_count);
  signal_pads_dev_.resize(device_count);
}

std::vector<NvlsHandle*>& NvlsMcastMemory::GetNvlsHandles() { return nvls_handles_; }

std::vector<IpcMemHandle>& NvlsMcastMemory::GetIpcHandles() { return ipc_handles_; }

// Get the raw array of signal pad pointers to all ranks (including self)
// Used for synchronization across devices
std::vector<void**>& NvlsMcastMemory::GetSignalPadsDev() { return signal_pads_dev_; }

#define ROUND_UP(n, multiple) (((n + multiple - 1) / multiple) * multiple)

// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/runtime/ipcNvlsMemory.cu
CUmemAllocationHandleType NvlsMcastMemory::GetMemHandleType(const size_t device_id) {
  // Check if fabric handle support is available.
  int fabric_supported = 0;
  CU_CHECK(cuDeviceGetAttribute(&fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device_id));
  if (!fabric_supported) {
    return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }

  nvmlDevice_t nvml_device;
  nvmlGpuFabricInfo_t fabric_info;
  NVML_CHECK(nvmlInit_v2());
  NVML_CHECK(nvmlDeviceGetHandleByIndex(device_id, &nvml_device));
  NVML_CHECK(nvmlDeviceGetGpuFabricInfo(nvml_device, &fabric_info));
  NVML_CHECK(nvmlShutdown());

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
  CU_CHECK(cuMemGetAllocationGranularity(&min_gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
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
    CU_CHECK(cuMemRelease(handle));
    return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }

  CU_CHECK(cuMemRelease(handle));
  // If we get here, fabric handles are supported.
  return CU_MEM_HANDLE_TYPE_FABRIC;
}

void NvlsMcastMemory::AllocMcastMemory(const size_t device_id, const size_t size) {
  NvlsHandle*& nvls_handle = nvls_handles_[device_id];
  // Prevent duplicate execution
  if (nvls_handle != nullptr) {
    return;
  }

  const size_t device_count = nvls_handles_.size();

  // Get runtime and driver device IDs.
  int CU_dev;
  CU_CHECK(cuDeviceGet(&CU_dev, device_id));

  nvls_handle = new NvlsHandle();

  constexpr size_t kSignalPadAlignment = 16UL;
  nvls_handle->signal_pad_offset = (size + kSignalPadAlignment - 1) & ~(kSignalPadAlignment - 1);
  nvls_handle->size = nvls_handle->signal_pad_offset + kSignalPadSize;

  // Get handle type used to share memory handles between devices.
  nvls_handle->mem_handle_type = GetMemHandleType(device_id);

  // Define allocation access permissions (same for unicast and multicast).
  memset(&nvls_handle->access_desc, 0, sizeof(CUmemAccessDesc));
  nvls_handle->access_desc.location.id = device_id;
  nvls_handle->access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  nvls_handle->access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  // Define unicast allocation properties.
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(CUmemAllocationProp));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.requestedHandleTypes = nvls_handle->mem_handle_type;

  // Define multicast allocation properties.
  memset(&nvls_handle->mcprop, 0, sizeof(CUmulticastObjectProp));
  nvls_handle->mcprop.numDevices = device_count;
  nvls_handle->mcprop.handleTypes = nvls_handle->mem_handle_type;
  nvls_handle->mcprop.flags = 0;

  // Round up allocation size to the nearest multiple of the unicast allocation granularity.
  CU_CHECK(cuMemGetAllocationGranularity(&nvls_handle->granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  nvls_handle->size = ROUND_UP(nvls_handle->size, nvls_handle->granularity);

  // Round up allocation size to the nearest multiple of the multicast allocation granularity.
  CU_CHECK(
      cuMulticastGetGranularity(&nvls_handle->mc_granularity, &nvls_handle->mcprop, CU_MULTICAST_GRANULARITY_MINIMUM));
  nvls_handle->size = ROUND_UP(nvls_handle->size, nvls_handle->mc_granularity);
  nvls_handle->mcprop.size = nvls_handle->size;

  // Allocate physical pages of memory on GPU.
  CU_CHECK(cuMemCreate(&nvls_handle->uc_handle, nvls_handle->size, &prop, 0));
  // Reserve unicast virtual address space for the memory.
  CU_CHECK(cuMemAddressReserve(&nvls_handle->uc_va, nvls_handle->size, nvls_handle->granularity, 0U, 0));
  // Map the unicast virtual address space to the physical pages.
  CU_CHECK(cuMemMap(nvls_handle->uc_va, nvls_handle->size, 0, nvls_handle->uc_handle, 0));
  // Set the access permissions for the unicast memory.
  CU_CHECK(cuMemSetAccess(nvls_handle->uc_va, nvls_handle->size, &nvls_handle->access_desc, 1));
  nvls_handle->uc_ptr = reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(nvls_handle->uc_va));
}

void NvlsMcastMemory::InitMcastMemory(const size_t device_id) {
  NvlsHandle* nvls_handle = nvls_handles_[device_id];
  // Prevent duplicate execution
  if (!nvls_handle->peer_uc_vas.empty()) {
    return;
  }

  const size_t device_count = nvls_handles_.size();

  // Get runtime and driver device IDs.
  int CU_dev;
  CU_CHECK(cuDeviceGet(&CU_dev, device_id));

  // Unicast pointer exchange between ranks.
  IpcMemHandle& ipc_handle = ipc_handles_[device_id];
  CU_CHECK(cuMemExportToShareableHandle(&ipc_handle, nvls_handle->uc_handle, nvls_handle->mem_handle_type,
                                        /*flags*/ 0));

  nvls_handle->peer_uc_vas.resize(device_count);
  nvls_handle->peer_uc_handles.resize(device_count);

  // Sync all ranks for uc exchanging
  static std::atomic<size_t> wg_uc{0};
  for (wg_uc.fetch_add(1, std::memory_order_relaxed); wg_uc.load(std::memory_order_relaxed) % device_count > 0;) {
    std::this_thread::yield();
  }

  for (size_t i = 0; i < device_count; i++) {
    if (i == device_id) {
      nvls_handle->peer_uc_vas[i] = nvls_handle->uc_va;
      nvls_handle->peer_uc_handles[i] = nvls_handle->uc_handle;
    } else {
      IpcMemHandle peer_ipc_handle = ipc_handles_[i];
      void* os_handle = nvls_handle->mem_handle_type == CU_MEM_HANDLE_TYPE_FABRIC
                            ? &peer_ipc_handle
                            : reinterpret_cast<void*>(peer_ipc_handle.fd);
      CU_CHECK(
          cuMemImportFromShareableHandle(&nvls_handle->peer_uc_handles[i], os_handle, nvls_handle->mem_handle_type));
      // Reserve peer unicast virtual address space for the memory.
      CU_CHECK(cuMemAddressReserve(&nvls_handle->peer_uc_vas[i], nvls_handle->size, nvls_handle->granularity, 0U, 0));
      // Map the peer unicast virtual address space to the physical pages.
      CU_CHECK(cuMemMap(nvls_handle->peer_uc_vas[i], nvls_handle->size, 0, nvls_handle->peer_uc_handles[i], 0));
      // Set the access permissions for the peer unicast memory.
      CU_CHECK(cuMemSetAccess(nvls_handle->peer_uc_vas[i], nvls_handle->size, &nvls_handle->access_desc, 1));
    }
  }

  // Initialize multicast object for all ranks.
  if (device_id == 0) {
    CU_CHECK(cuMulticastCreate(&nvls_handle->mc_handle, &nvls_handle->mcprop));
    // Export the allocation for the importing process.
    CU_CHECK(
        cuMemExportToShareableHandle(&ipc_handle, nvls_handle->mc_handle, nvls_handle->mem_handle_type, /*flags*/ 0));
  }

  // Sync all ranks for mc exchanging
  static std::atomic<size_t> wg_mc{0};
  for (wg_mc.fetch_add(1, std::memory_order_relaxed); wg_mc.load(std::memory_order_relaxed) % device_count > 0;) {
    std::this_thread::yield();
  }

  if (device_id != 0) {
    IpcMemHandle ipc_handle = ipc_handles_[0];
    void* os_handle = nvls_handle->mem_handle_type == CU_MEM_HANDLE_TYPE_FABRIC
                          ? &ipc_handle
                          : reinterpret_cast<void*>(ipc_handle.fd);
    CU_CHECK(cuMemImportFromShareableHandle(&nvls_handle->mc_handle, os_handle, nvls_handle->mem_handle_type));
  }

  // Add device to multicast object
  CU_CHECK(cuMulticastAddDevice(nvls_handle->mc_handle, CU_dev));
  // Bind physical memory to the Multicast group.
  // Note: It will block until all ranks have been added to the group.
  CU_CHECK(cuMulticastBindMem(nvls_handle->mc_handle, 0, nvls_handle->uc_handle, 0, nvls_handle->size, 0));
  // Reserve multicast virtual address space for the memory.
  CU_CHECK(cuMemAddressReserve(&nvls_handle->mc_va, nvls_handle->size, nvls_handle->mc_granularity, 0U, 0));
  // Map the multicast virtual address space to the physical pages.
  CU_CHECK(cuMemMap(nvls_handle->mc_va, nvls_handle->size, 0, nvls_handle->mc_handle, 0));
  // Set the access permissions for the multicast memory.
  CU_CHECK(cuMemSetAccess(nvls_handle->mc_va, nvls_handle->size, &nvls_handle->access_desc, /*count*/ 1));
  nvls_handle->mc_ptr = reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(nvls_handle->mc_va));

  // Initialize signal pads
  std::vector<CUdeviceptr> signal_pads(device_count);
  for (size_t i = 0; i < device_count; i++) {
    signal_pads[i] = nvls_handle->peer_uc_vas[i] + nvls_handle->signal_pad_offset;
    if (i == device_id) {
      CU_CHECK(cuMemsetD8(signal_pads[i], 0, kSignalPadSize));
    }
  }
  // Copy host array of pointers to device array
  void**& signal_pad_dev = signal_pads_dev_[device_id];
  CUDA_CHECK(cudaMalloc(&signal_pad_dev, device_count * sizeof(CUdeviceptr)));
  CUDA_CHECK(
      cudaMemcpy(signal_pad_dev, signal_pads.data(), device_count * sizeof(CUdeviceptr), cudaMemcpyHostToDevice));

  // Sync all ranks for signal
  static std::atomic<size_t> wg_signal{0};
  for (wg_signal.fetch_add(1, std::memory_order_relaxed);
       wg_signal.load(std::memory_order_relaxed) % device_count > 0;) {
    std::this_thread::yield();
  }
}

void NvlsMcastMemory::FreeMcastMemory(const size_t device_id) {
  if (NvlsHandle* nvls_handle = nvls_handles_[device_id]; nvls_handle != nullptr) {
    CU_CHECK(cuMemUnmap(nvls_handle->mc_va, nvls_handle->size));
    CU_CHECK(cuMemRelease(nvls_handle->mc_handle));
    CU_CHECK(cuMemAddressFree(nvls_handle->mc_va, nvls_handle->size));
    for (size_t i = 0; i < nvls_handle->peer_uc_vas.size(); ++i) {
      CU_CHECK(cuMemUnmap(nvls_handle->peer_uc_vas[i], nvls_handle->size));
      CU_CHECK(cuMemRelease(nvls_handle->peer_uc_handles[i]));
      CU_CHECK(cuMemAddressFree(nvls_handle->peer_uc_vas[i], nvls_handle->size));
    }
    delete nvls_handle;
  }
  if (void** signal_pad_dev = signal_pads_dev_[device_id]; signal_pad_dev != nullptr) {
    CUDA_CHECK(cudaFree(signal_pad_dev));
  }
}

}  // namespace ksana_llm
