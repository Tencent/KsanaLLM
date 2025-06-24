/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/nvidia/nvidia_context.h"

#include <cuda_profiler_api.h>

#include <csignal>
#include "3rdparty/LLM_kernels/csrc/utils/nvidia/cuda_utils.h"

namespace ksana_llm {

// The minimum cuda version that support mempool.
constexpr int CUDA_MEMPOOL_MIN_DRIVER_VERSION = 11030;
constexpr int CUDA_GEMM_SUPPORT_FP8_MIN_CUBLASLT_VERSION = 120103;

template <int T>
void NvidiaContextExtension<T>::InitGpuMemoryPool(const int worker_id) {
  int device_supports_memory_pools = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&device_supports_memory_pools, cudaDevAttrMemoryPoolsSupported, worker_id));
  // NOTE(karlluo): 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, and 0 otherwise
  if (device_supports_memory_pools == 0) {
    KLLM_LOG_WARNING << fmt::format("GPU {} is not support GPU mempool, skip init.", worker_id);
    return;
  }

  KLLM_LOG_DEBUG << "Init nvidia memroy pool on GPU " << worker_id;
  CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version_));
  if (cuda_driver_version_ >= CUDA_MEMPOOL_MIN_DRIVER_VERSION) {
    int pool_supported_handle_types = 0;
    cudaMemPool_t mempool;
    CUDA_CHECK(
        cudaDeviceGetAttribute(&pool_supported_handle_types, cudaDevAttrMemoryPoolSupportedHandleTypes, worker_id));
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, worker_id));
    uint64_t threshold = UINT64_MAX;
    CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    int enable = 1;
    CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseFollowEventDependencies, &enable));
    CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseAllowOpportunistic, &enable));
    CUDA_CHECK(cudaMemPoolSetAttribute(mempool, cudaMemPoolReuseAllowInternalDependencies, &enable));

    // Set access_id's accessing to the worker_id's mempool.
    for (int access_id = 0; access_id < base_ptr_->tensor_parallel_size_; ++access_id) {
      if (access_id != worker_id) {
        cudaMemAccessDesc desc = {};
        desc.location.type = cudaMemLocationTypeDevice;
        desc.location.id = access_id;
        desc.flags = cudaMemAccessFlagsProtReadWrite;
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, access_id, worker_id));
        if (can_access == 0) {
          KLLM_THROW(
              fmt::format("GPU {} is not capable of directly accessing memory of peer GPU {}.", access_id, worker_id));
        }
        CUDA_CHECK(cudaMemPoolSetAccess(mempool, &desc, 1));
      }
    }
  }
}

template <int T>
void NvidiaContextExtension<T>::InitCublasHandle(const int worker_id) {
  KLLM_LOG_DEBUG << "Init nvidia cublas/cublasLt on worker " << worker_id;
  CUDA_CHECK(cublasCreate(&cublas_handles_[worker_id]));
  CUDA_CHECK(cublasLtCreate(&cublaslt_handles_[worker_id]));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, worker_id);
  // FP8 GEMM is supported when arch >= 89 and cublaslt >= 12.1.3.
  base_ptr_->is_gemm_fp8_supported_ = (((prop.major >= 8 && prop.minor >= 9) || prop.major >= 9) &&
                                       (cublasLtGetVersion() >= CUDA_GEMM_SUPPORT_FP8_MIN_CUBLASLT_VERSION));

  // binding compute stream to cublas
  CUDA_CHECK(cublasSetStream(cublas_handles_[worker_id], base_ptr_->compute_streams_[worker_id].Get()));
}

template <int T>
void NvidiaContextExtension<T>::InitNcclParam() {
  KLLM_LOG_DEBUG << "Init nvidia nccl param.";
  reduce_signals_.resize(max_reduce_inputs_num_);
  reduce_inputs_.resize(max_reduce_inputs_num_);

  const size_t world_size = base_ptr_->tensor_parallel_size_;

  for (int worker_id = 0; worker_id < world_size; ++worker_id) {
    CUDA_CHECK(cudaSetDevice(worker_id));
    for (int i = 0; i < world_size; i++) {
      if (i != worker_id) {
        auto err = cudaDeviceEnablePeerAccess(i, 0);
        if (err != cudaErrorPeerAccessAlreadyEnabled) {
          CUDA_CHECK(err);
        }
      }
    }
  }

  nccl_params_.resize(world_size);
  if (world_size > 1) {
    nccl_uid_ = GenerateNCCLUniqueID();
    NCCL_CHECK(ncclGroupStart());
    // TODO(karlluo): for single machine multiple xpus, device_num is the world_size
    // for multiple machine, world size should change in future, and the same situation of rank_id
    for (int worker_id = 0; worker_id < world_size; ++worker_id) {
      CUDA_CHECK(cudaSetDevice(worker_id));
      NCCL_CHECK(ncclCommInitRank(&(nccl_params_[worker_id].nccl_comm), world_size, nccl_uid_, worker_id));
    }
    NCCL_CHECK(ncclGroupEnd());
  }

  init_done_.store(true, std::memory_order_relaxed);
}

template <int T>
void NvidiaContextExtension<T>::Initialize() {
  CUDA_CHECK(cudaDriverGetVersion(&base_ptr_->driver_version_));
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    KLLM_THROW("There is not GPU on you machine.");
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  sm_ = prop.major * 10 + prop.minor;
  int cuda_ver_tmp;
  CUDA_CHECK(cudaRuntimeGetVersion(&cuda_ver_tmp));
  cuda_ver_ = static_cast<uint32_t>(cuda_ver_tmp);
  KLLM_LOG_INFO << fmt::format("Get sm: {}, cuda version: {}", sm_, cuda_ver_);

  memory_pool_.resize(base_ptr_->tensor_parallel_size_);
  cublas_handles_.resize(base_ptr_->tensor_parallel_size_);
  cublaslt_handles_.resize(base_ptr_->tensor_parallel_size_);
  std::vector<std::thread> init_threads;
  for (int worker_id = 0; worker_id < base_ptr_->tensor_parallel_size_; ++worker_id) {
    init_threads.emplace_back([worker_id, this]() {
      KLLM_LOG_DEBUG << "Init nvidia gpu relate handler on worker " << worker_id;
      CUDA_CHECK(cudaSetDevice(worker_id));
      InitGpuMemoryPool(worker_id);
      InitCublasHandle(worker_id);
      if (worker_id != 0 && llm_kernels::utils::GetNvLinkVersion(0, worker_id) == 0) {
        is_full_nvlink_ = false;
      }
    });
  }
  for (auto& thread : init_threads) {
    thread.join();
  }

  // init nccl async
  std::thread([this]() { InitNcclParam(); }).detach();

  // reset device id
  CUDA_CHECK(cudaSetDevice(base_ptr_->defalt_device_id_));
}

template <int T>
void NvidiaContextExtension<T>::Destroy() {
  // wait nccl async init finish
  GetNCCLParam();

  for (int worker_id = 0; worker_id < base_ptr_->tensor_parallel_size_; ++worker_id) {
    CUDA_CHECK(cudaSetDevice(worker_id));
    CUDA_CHECK(cublasDestroy(cublas_handles_[worker_id]));
    CUDA_CHECK(cublasLtDestroy(cublaslt_handles_[worker_id]));
    NCCL_CHECK(DestroyNCCLParam(nccl_params_[worker_id]));
  }
}

template <>
void ContextT<DEVICE_TYPE_NVIDIA>::InitializeExtension() {
  ext = new NvidiaContextExtension<DEVICE_TYPE_NVIDIA>(this);
  ext->Initialize();
}

template <>
void ContextT<DEVICE_TYPE_NVIDIA>::DestroyExtension() {
  ext->Destroy();
  delete ext;
}

}  // namespace ksana_llm
