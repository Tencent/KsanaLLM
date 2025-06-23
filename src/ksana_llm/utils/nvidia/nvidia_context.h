/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "ksana_llm/utils/common_context.h"

#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/nvidia/nccl_utils.h"
#include "ksana_llm/utils/waiter.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/gemm_wrapper/gemm_algo_map.h"

namespace ksana_llm {

// The class used for nvidia extension.
template <int T>
class NvidiaContextExtension {
 public:
  explicit NvidiaContextExtension(ContextT<T>* base_ptr) { base_ptr_ = base_ptr; }

  std::vector<cudaMemPool_t>& GetMemoryPools() { return memory_pool_; }

  ncclUniqueId& GetNCCLUniqueID() { return nccl_uid_; }

  std::vector<NCCLParam>& GetNCCLParam() {
    while (!init_done_.load(std::memory_order_relaxed)) {
      std::this_thread::yield();
    }
    return nccl_params_;
  }

  std::vector<cublasHandle_t>& GetCublasHandles() { return cublas_handles_; }

  std::vector<cublasLtHandle_t>& GetCublasLtHandles() { return cublaslt_handles_; }

  void** GetCustomAllReduceSignals() { return static_cast<void**>(reduce_signals_.data()); }

  void** GetCustomAllReduceInputs() { return static_cast<void**>(reduce_inputs_.data()); }

  bool IsFullNvLink() { return is_full_nvlink_; }

  uint32_t GetComputeCapacity() { return sm_; }
  uint32_t GetCudaVersion() { return cuda_ver_; }

  llm_kernels::nvidia::GPUGemmAlgoHelper& GetGPUGemmAlgoHelper() { return gpu_gemm_algo_helper_; }

  // Initialize and destroy extension.
  void Initialize();
  void Destroy();

 private:
  // init gpu memory pool
  void InitGpuMemoryPool(const int worker_id);

  // init cublas handle
  void InitCublasHandle(const int worker_id);

  // init nccl handle
  void InitNcclParam();

 private:
  ContextT<T>* base_ptr_ = nullptr;

  bool is_full_nvlink_ = true;

  // The cuda driver version.
  int cuda_driver_version_;

  // Nvidia GPU memory pool
  std::vector<cudaMemPoolProps> memory_pool_props_;
  std::vector<cudaMemPool_t> memory_pool_;

  // nccl comms
  ncclUniqueId nccl_uid_;
  std::vector<NCCLParam> nccl_params_;

  // cublas handles
  std::vector<cublasHandle_t> cublas_handles_;
  std::vector<cublasLtHandle_t> cublaslt_handles_;

  // The max reduce inputs num for custom reduce.
  int max_reduce_inputs_num_{8};

  // Stores the intermediate results for each gpu of custom all reduce.
  std::vector<void*> reduce_signals_;

  // Maintain the input tensor required by each gpu of custom all reduce.
  std::vector<void*> reduce_inputs_;

  uint32_t sm_{0};
  uint32_t cuda_ver_{0};
  std::vector<void*> cublaslt_workspace_ptrs_;
  // The helper for providing the best performance gemm algo
  llm_kernels::nvidia::GPUGemmAlgoHelper gpu_gemm_algo_helper_;

  std::atomic_bool init_done_{false};
};

template <>
struct ExtensionTypeTraits<DEVICE_TYPE_NVIDIA> {
  typedef NvidiaContextExtension<DEVICE_TYPE_NVIDIA> value_type;
};

// 构造扩展类对象
template <>
void ContextT<DEVICE_TYPE_NVIDIA>::InitializeExtension();

// 销毁扩展类对象
template <>
void ContextT<DEVICE_TYPE_NVIDIA>::DestroyExtension();

}  // namespace ksana_llm
