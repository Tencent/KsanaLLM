/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu
 *
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "csrc/kernels/nvidia/others/tensorrt-llm/main/communication_kernels/trtllm_all_reduce.h"

#include <cooperative_groups.h>
#include <vector>

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

#define FINAL_MASK 0xffffffff

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

template <int NRanks>
struct SyncComm {
  __device__ __forceinline__ SyncComm(void** workspace) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[1];
    flag_value = *flag_ptr;
    for (int r = 0; r < NRanks; ++r) {
      comm_bufs[r] = workspace[r];
      barrier_flags[r] = workspace[NRanks + r];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_flag_value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = new_flag_value;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  void* comm_bufs[NRanks];
  void* barrier_flags[NRanks];
  int flag_value;
};

template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[4];
    flag_value = *flag_ptr;
    int comm_size = reinterpret_cast<int*>(workspace[NRanks * 3])[3];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_offset * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int clear_size;
  int flag_value;
};

template <int NRanks>
class Barrier {
 public:
  __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const& comm) {
    if (threadIdx.x < NRanks) {
      m_flag_value = comm.flag_value;
      int current_rank = rank;
      int target_rank = threadIdx.x;
      m_target_flag = reinterpret_cast<int*>(comm.barrier_flags[target_rank]) + current_rank;
      m_current_flag = reinterpret_cast<int*>(comm.barrier_flags[current_rank]) + blockIdx.x * NRanks + target_rank;
    }
  }

  __device__ __forceinline__ void sync() {
    __syncthreads();
    if (threadIdx.x < NRanks) {
      m_flag_value = next_flag(m_flag_value);
      // To avoid the ABA problem, we need to synchronize the correct flag value to all barrier_flags, even if the
      // corresponding CTA has not been launched.
      for (int flag_idx = blockIdx.x; flag_idx < kBarrierFlagCount; flag_idx += gridDim.x) {
        st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
      }
      while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
      }
    }
    __syncthreads();
  }

 protected:
  __device__ __forceinline__ void st_flag(int* addr, int flag) {
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
  }

  __device__ __forceinline__ int ld_flag(int* addr) {
    int flag;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(addr));
    return flag;
  }

  __device__ __forceinline__ int next_flag(int flag) { return flag == 2 ? 0 : flag + 1; }

  __device__ __forceinline__ int prev_flag(int flag) { return flag == 0 ? 2 : flag - 1; }

 public:
  int m_flag_value;

 private:
  int* m_target_flag;
  int* m_current_flag;
};

template <typename T, typename PackedType>
__device__ __forceinline__ PackedType add128(PackedType const& a, PackedType const& b) {
  static constexpr int kMathCount = sizeof(PackedType) / sizeof(T);
  PackedType c;
#pragma unroll
  for (int i = 0; i < kMathCount; ++i) {
    reinterpret_cast<T*>(&c)[i] = reinterpret_cast<T const*>(&a)[i] + reinterpret_cast<T const*>(&b)[i];
  }
  return c;
}

template <AllReduceFusionPattern Pattern, typename T>
class FusedOp {
  static constexpr int kMathCount = sizeof(float4) / sizeof(T);

 public:
  __device__ __forceinline__ FusedOp(AllReduceFusionParams<T> const& params, int access_id, int access_id_in_token)
      : m_params(params), m_access_id(access_id), m_access_id_in_token(access_id_in_token) {
    if constexpr (HasRMSNorm<Pattern>) {
      m_gamma_val = reinterpret_cast<float4*>(params.rms_gamma)[m_access_id_in_token];
    }
    if constexpr (HasResidual<Pattern>) {
      m_residual_val = reinterpret_cast<float4*>(params.residual_in)[m_access_id];
    }
  }

  __device__ __forceinline__ void update(int access_id) {
    if (m_access_id != access_id) {
      m_access_id = access_id;
      if constexpr (HasResidual<Pattern>) {
        m_residual_val = reinterpret_cast<float4*>(m_params.residual_in)[m_access_id];
      }
    }
  }

  __device__ __forceinline__ void operator()(float4 val, int token_id) {
    if constexpr (HasAllReduceOut<Pattern>) {
      reinterpret_cast<float4*>(m_params.allreduce_out)[m_access_id] = val;
    }
    if constexpr (HasResidual<Pattern>) {
      val = add128<T>(val, m_residual_val);
      if constexpr (HasResidualOut<Pattern>) {
        reinterpret_cast<float4*>(m_params.residual_out)[m_access_id] = val;
      }
    }
    if constexpr (HasRMSNorm<Pattern>) {
      val = rms_norm(val, m_gamma_val);
      if constexpr (HasNormOut<Pattern>) {
        reinterpret_cast<float4*>(m_params.norm_out)[m_access_id] = val;
      }
    }
    static_assert(GetQuantType<Pattern> == QuantType::kNone, "Invalid quant type");
  }

 protected:
  __device__ __forceinline__ float4 rms_norm(float4 const& residual, float4 const& gamma) {
    __shared__ float s_val;
    float4 norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kMathCount; ++i) {
      float v = static_cast<float>(reinterpret_cast<T const*>(&residual)[i]);
      acc += v * v;
    }
    blockReduceSumV2<float, 1>(&acc);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    if (cluster.num_blocks() > 1) {
      if (threadIdx.x == 0) {
        s_val = acc;
        acc = 0.f;
      }
      cluster.sync();
      if (threadIdx.x == 0) {
        for (int i = 0; i < cluster.num_blocks(); ++i) {
          acc += *cluster.map_shared_rank(&s_val, i);
        }
      }
      cluster.sync();
    }
#endif
    if (threadIdx.x == 0) {
      s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kMathCount; ++i) {
      reinterpret_cast<T*>(&norm_out)[i] =
          static_cast<T>(static_cast<float>(reinterpret_cast<T const*>(&residual)[i]) * s_val *
                         static_cast<float>(reinterpret_cast<T const*>(&gamma)[i]));
    }
    return norm_out;
  }

 private:
  AllReduceFusionParams<T> const& m_params;
  int m_access_id;
  int m_access_id_in_token;
  float m_scale_factor;
  float4 m_residual_val;
  float4 m_gamma_val;
};

__device__ __forceinline__ bool is_neg_zero(float v) { return *reinterpret_cast<uint32_t*>(&v) == 0x80000000; }

__device__ __forceinline__ bool is_neg_zero(float4 v) {
  return is_neg_zero(v.x) || is_neg_zero(v.y) || is_neg_zero(v.z) || is_neg_zero(v.w);
}

__device__ __forceinline__ float4 get_neg_zero() {
  float4 vec;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000;
  }
  return vec;
}

__device__ __forceinline__ float4 ld_global_volatile(float4* addr) {
  float4 val;
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
               : "l"(addr));
  return val;
}

template <typename T, int NRanks, bool Fp32Acc>
__device__ __forceinline__ float4 allreduce_sum(float4* vals) {
  if constexpr (Fp32Acc) {
    static_assert(!std::is_same_v<T, float>);
    float acc_f32[kElemsPerAccess<T>];
#pragma unroll
    for (int i = 0; i < kElemsPerAccess<T>; ++i) {
      acc_f32[i] = static_cast<float>(reinterpret_cast<T*>(&vals[0])[i]);
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
#pragma unroll
      for (int i = 0; i < kElemsPerAccess<T>; ++i) {
        acc_f32[i] += static_cast<float>(reinterpret_cast<T*>(&vals[r])[i]);
      }
    }
    float4 acc;
#pragma unroll
    for (int i = 0; i < kElemsPerAccess<T>; ++i) {
      reinterpret_cast<T*>(&acc)[i] = static_cast<T>(acc_f32[i]);
    }
    return acc;
  } else {
    float4 acc = vals[0];
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
      acc = add128<T>(acc, vals[r]);
    }
    return acc;
  }
}

template <typename T>
class IndexHelper {
 public:
  __device__ __forceinline__ IndexHelper(AllReduceFusionParams<T> const& params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    token_id = grid.cluster_rank();
    access_id_in_token = cluster.thread_rank();
    token_stride = grid.num_clusters();
#else
    token_id = blockIdx.x;
    access_id_in_token = threadIdx.x;
    token_stride = gridDim.x;
#endif
    access_id = token_id * params.hidden_dim / kElemsPerAccess<T> + access_id_in_token;
    access_stride = token_stride * params.hidden_dim / kElemsPerAccess<T>;
    tot_access = params.size / kElemsPerAccess<T>;
  }

  int token_id;
  int access_id_in_token;
  int token_stride;
  int access_id;
  int access_stride;
  int tot_access;
};

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024) allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams<T> params) {
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  float4 clear_vec = get_neg_zero();
  FusedOp<Pattern, T> fused_op(params, access_id, access_id_in_token);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  if constexpr (!TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / kElemsPerAccess<T>;

  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    alignas(16) float val[4];
    *reinterpret_cast<float4*>(val) = reinterpret_cast<float4*>(params.allreduce_in)[idx];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (is_neg_zero(val[i])) {
        val[i] = 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      // Push data to other ranks
      reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * tot_access + idx] = *reinterpret_cast<float4*>(val);
    }
  }
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    // Clear comm buffer that previous kernel used
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }

  for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride) {
    fused_op.update(idx);
    float4 vals[NRanks];
    bool done = false;
    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        // LDG.128 from local rank
        vals[r] = ld_global_volatile(&reinterpret_cast<float4*>(comm.data_bufs[params.rank])[r * tot_access + idx]);
        done &= !is_neg_zero(vals[r]);
      }
    }
    float4 sum_val = allreduce_sum<T, NRanks, Fp32Acc>(vals);
    fused_op(sum_val, tidx);
  }

  comm.update(params.size * NRanks);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (TriggerCompletionAtEnd) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
__global__ void __launch_bounds__(1024)
    allreduce_fusion_kernel_twoshot_sync(AllReduceFusionParams<T> params, std::array<int, NRanks> begin_tokens,
                                         std::array<int, NRanks> token_num_per_ranks) {
  IndexHelper<T> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  FusedOp<Pattern, T> fused_op(params, access_id, access_id_in_token);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  SyncComm<NRanks> comm(params.workspace);
#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / kElemsPerAccess<T>;
    int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / kElemsPerAccess<T>;
    for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
      reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[idx] = reinterpret_cast<float4*>(params.allreduce_in)[idx];
    }
  }
  Barrier<NRanks> barrier(params.rank, comm);
  barrier.sync();
  int comm_access_id = access_id + begin_tokens[params.rank] * params.hidden_dim / kElemsPerAccess<T>;
  int comm_tot_access =
      (begin_tokens[params.rank] + token_num_per_ranks[params.rank]) * params.hidden_dim / kElemsPerAccess<T>;
  for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride) {
    float4 vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      vals[r] = reinterpret_cast<float4*>(comm.comm_bufs[r])[idx];
    }
    float4 sum_val = allreduce_sum<T, NRanks, Fp32Acc>(vals);
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      reinterpret_cast<float4*>(comm.comm_bufs[r])[tot_access + idx] = sum_val;
    }
  }
  barrier.sync();
#pragma unroll
  for (int r = 0; r < NRanks; ++r) {
    int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / kElemsPerAccess<T>;
    int comm_token_id = token_id + begin_tokens[r];
    int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / kElemsPerAccess<T>;
    for (int idx = comm_access_id, tidx = comm_token_id; idx < comm_tot_access;
         idx += access_stride, tidx += token_stride) {
      fused_op.update(idx);
      float4 sum_val = reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[tot_access + idx];
      fused_op(sum_val, tidx);
    }
  }
  comm.update(barrier.m_flag_value);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd = true>
int get_registers_per_thread_oneshot() {
  auto kernel = allreduce_fusion_kernel_oneshot_lamport<Pattern, T, NRanks, Fp32Acc, TriggerCompletionAtEnd>;
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel);
  return attr.numRegs;
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd = true>
void launch_oneshot_lamport(AllReduceFusionParams<T> const& params, cudaLaunchConfig_t& cfg) {
  CHECK_NVIDIA_CUDA_ERROR(cudaLaunchKernelEx(
      &cfg, allreduce_fusion_kernel_oneshot_lamport<Pattern, T, NRanks, Fp32Acc, TriggerCompletionAtEnd>, params));
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
int get_registers_per_thread_twoshot() {
  auto kernel = allreduce_fusion_kernel_twoshot_sync<Pattern, T, NRanks, Fp32Acc>;
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kernel);
  return attr.numRegs;
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
void launch_twoshot_sync(AllReduceFusionParams<T> const& params, cudaLaunchConfig_t& cfg,
                         std::array<int, NRanks> begin_tokens, std::array<int, NRanks> token_num_per_ranks) {
  CHECK_NVIDIA_CUDA_ERROR(cudaLaunchKernelEx(&cfg, allreduce_fusion_kernel_twoshot_sync<Pattern, T, NRanks, Fp32Acc>,
                                             params, begin_tokens, token_num_per_ranks));
}

template <AllReduceFusionPattern Pattern, typename T, int NRanks, bool Fp32Acc>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const& params) {
  KLLM_KERNEL_CHECK(params.size % params.hidden_dim == 0);
  KLLM_KERNEL_CHECK(params.hidden_dim % kElemsPerAccess<T> == 0);
  static int SM = GetSMVersion();
  int token_num = params.size / params.hidden_dim;
  bool oneshot = params.use_oneshot;
  int cluster_num = token_num;
  std::array<int, NRanks> begin_tokens, token_num_per_ranks;
  if (!oneshot) {
    int remaining_token = token_num % NRanks;
    int token_num_per_rank = token_num / NRanks;
    cluster_num = token_num_per_rank;
    if (remaining_token) {
      cluster_num++;
    }
    for (int r = 0; r < NRanks; ++r) {
      begin_tokens[r] = r * token_num_per_rank + (remaining_token > r ? r : remaining_token);
      token_num_per_ranks[r] = token_num_per_rank + (remaining_token > r ? 1 : 0);
    }
  }
  int threads_per_token = params.hidden_dim / kElemsPerAccess<T>;
  int cluster_size;
  if (SM >= 90) {
    cluster_size = 8;
  } else {
    cluster_size = 1;
  }
  while (threads_per_token % cluster_size != 0 && cluster_size > 1) {
    cluster_size /= 2;
  }
  int threads_per_block = threads_per_token / cluster_size;
  while (threads_per_block < 128 && cluster_size >= 2) {
    threads_per_block *= 2;
    cluster_size /= 2;
  }
  int sm_count = GetSMCount();
  int registers_per_thread;
  if (oneshot) {
    if (params.trigger_completion_at_end) {
      registers_per_thread = get_registers_per_thread_oneshot<Pattern, T, NRanks, Fp32Acc, true>();
    } else {
      registers_per_thread = get_registers_per_thread_oneshot<Pattern, T, NRanks, Fp32Acc, false>();
    }
  } else {
    registers_per_thread = get_registers_per_thread_twoshot<Pattern, T, NRanks, Fp32Acc>();
  }
  int max_registers = GetPerBlockRegisterCount();
  int max_threads_per_block = min(max_registers / registers_per_thread, 1024);
  while (cluster_num * cluster_size > sm_count && cluster_size > 1 && threads_per_block <= max_threads_per_block / 2) {
    threads_per_block *= 2;
    cluster_size /= 2;
  }
  KLLM_KERNEL_CHECK(oneshot || threads_per_block >= params.nranks);
  int block_size = threads_per_block;
  KLLM_KERNEL_CHECK(block_size <= 1024 && cluster_size > 0);

  int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;
  cudaLaunchConfig_t cfg;
  cudaLaunchAttribute attribute[2];
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = GetEnablePDL() ? 1 : 0;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;
  if (oneshot) {
    bool trigger_completion_at_end = params.trigger_completion_at_end;
    if (trigger_completion_at_end) {
      launch_oneshot_lamport<Pattern, T, NRanks, Fp32Acc, true>(params, cfg);
    } else {
      launch_oneshot_lamport<Pattern, T, NRanks, Fp32Acc, false>(params, cfg);
    }
  } else {
    launch_twoshot_sync<Pattern, T, NRanks, Fp32Acc>(params, cfg, begin_tokens, token_num_per_ranks);
  }
}

template <typename T>
void allreduce_fusion_op(AllReduceFusionParams<T> const& params) {
#define DISPATCH_ACC_TYPE(T, Pattern, NRanks) \
  return allreduce_fusion_kernel_launcher<Pattern, T, NRanks, false>(params);

#define DISPATCH_PATTERN(T, NRanks)                                                      \
  switch (params.pattern) {                                                              \
    case AllReduceFusionPattern::kAllReduce:                                             \
      DISPATCH_ACC_TYPE(T, AllReduceFusionPattern::kAllReduce, NRanks);                  \
      break;                                                                             \
    case AllReduceFusionPattern::kARResidualRMSNorm:                                     \
      DISPATCH_ACC_TYPE(T, AllReduceFusionPattern::kARResidualRMSNorm, NRanks);          \
      break;                                                                             \
    default:                                                                             \
      KLLM_KERNEL_CHECK_WITH_INFO(false, "allreduce_fusion_kernel: unsupported dtype!"); \
  }

  switch (params.nranks) {
    case 2:
      DISPATCH_PATTERN(T, 2);
      break;
    case 4:
      DISPATCH_PATTERN(T, 4);
      break;
    case 8:
      DISPATCH_PATTERN(T, 8);
      break;
    case 16:
      DISPATCH_PATTERN(T, 16);
      break;
    default:
      KLLM_KERNEL_CHECK_WITH_INFO(false, "allreduce_fusion_kernel: unsupported ranks number!");
  }
}

#define INSTANTIATE_ALLREDUCE_FUSION_OP(T) template void allreduce_fusion_op<T>(AllReduceFusionParams<T> const&);
INSTANTIATE_ALLREDUCE_FUSION_OP(float);
INSTANTIATE_ALLREDUCE_FUSION_OP(half);
INSTANTIATE_ALLREDUCE_FUSION_OP(__nv_bfloat16);
#undef INSTANTIATE_ALLREDUCE_FUSION_OP

__global__ void lamport_initialize_kernel(float* ptr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  ptr[idx] = -0.f;
}

void lamport_initialize(void* ptr, int bytes, cudaStream_t stream) {
  int grid_size = (bytes + 127) / 128;
  lamport_initialize_kernel<<<grid_size, 128, 0, stream>>>(reinterpret_cast<float*>(ptr), bytes / sizeof(float));
}

int RoundUp(int x, int y) { return (x + y - 1) / y * y; }

// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.cu
void AllocTrtAllReduceWorkspace(const int nranks, const int rank, const int max_token_num, const int hidden_dim,
                                const int data_type_size, std::vector<void*>& buffer_d_ptrs,
                                std::vector<void*>& flag_d_ptrs, std::vector<void*>& workspace_d_ptrs,
                                cudaStream_t stream) {
  KLLM_KERNEL_CHECK(static_cast<int>(buffer_d_ptrs.size()) == 3 * nranks);
  KLLM_KERNEL_CHECK(static_cast<int>(flag_d_ptrs.size()) == nranks);
  KLLM_KERNEL_CHECK(static_cast<int>(workspace_d_ptrs.size()) == nranks);

  // We assume that int overflow will not occur here
  const int buffer_size = RoundUp(nranks * max_token_num * hidden_dim * data_type_size, kLamportAlignSize);
  const int flag_size = RoundUp(nranks * kBarrierFlagCount * sizeof(int), kLamportAlignSize);
  // This is used only in oneshot
  const int lamport_comm_size = RoundUp(nranks * max_token_num * hidden_dim * data_type_size, kLamportAlignSize);
  // The factor of 3 is due to the flag value having three different states
  const int lamport_buffer_size = 3 * lamport_comm_size;
  KLLM_KERNEL_CHECK_WITH_INFO(lamport_buffer_size <= kMaxCommSize,
                              "`lamport_buffer_size` should not exceed `kMaxCommSize`, please reduce the "
                              "`max_token_num` for lamport allreduce");

  int i = 0;
  for (const int size : {buffer_size, flag_size, lamport_buffer_size}) {
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&buffer_d_ptrs[i * nranks + rank], size));
    i++;
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  lamport_initialize(buffer_d_ptrs[2 * nranks + rank], lamport_buffer_size, stream);

  // Content of flag_d_ptrs:
  // atomic flag read counter
  // kernel_flag_ptr[0] = 0;
  // non-lamport flag
  // kernel_flag_ptr[1] = 0;
  // lamport flag
  // kernel_flag_ptr[2] = 0;
  // lamport triple buffer offset
  // kernel_flag_ptr[3] = lamport_comm_size;
  // lamport clear size
  // kernel_flag_ptr[4] = 0;
  CHECK_NVIDIA_CUDA_ERROR(cudaMallocAsync(&flag_d_ptrs[rank], 5 * sizeof(int), stream));
  std::vector<int> h_data{0, 0, 0, lamport_comm_size, 0};
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpyAsync(flag_d_ptrs[rank], h_data.data(), 5 * sizeof(int), cudaMemcpyHostToDevice, stream));

  CHECK_NVIDIA_CUDA_ERROR(cudaMallocAsync(&workspace_d_ptrs[rank], (3 * nranks + 1) * sizeof(void*), stream));

  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
}

void InitTrtAllReduceWorkspace(const int nranks, const int rank, const std::vector<void*>& buffer_d_ptrs,
                               const std::vector<void*>& flag_d_ptrs, const std::vector<void*>& workspace_d_ptrs,
                               cudaStream_t stream) {
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(workspace_d_ptrs[rank], buffer_d_ptrs.data(), 3 * nranks * sizeof(void*),
                                          cudaMemcpyHostToDevice, stream));
  CHECK_NVIDIA_CUDA_ERROR(
      cudaMemcpyAsync(reinterpret_cast<uint8_t*>(workspace_d_ptrs[rank]) + 3 * nranks * sizeof(void*),
                      flag_d_ptrs.data() + rank, sizeof(void*), cudaMemcpyHostToDevice, stream));

  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
}

void FreeTrtAllReduceWorkspace(const int nranks, const int rank, const std::vector<void*>& buffer_d_ptrs,
                               const std::vector<void*>& flag_d_ptrs, const std::vector<void*>& workspace_d_ptrs,
                               cudaStream_t stream) {
  for (int i = 0; i < 3; i++) {
    CHECK_NVIDIA_CUDA_ERROR(cudaFreeAsync(buffer_d_ptrs[i * nranks + rank], stream));
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaFreeAsync(flag_d_ptrs[rank], stream));
  CHECK_NVIDIA_CUDA_ERROR(cudaFreeAsync(workspace_d_ptrs[rank], stream));

  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
}

}  // namespace nvidia
}  // namespace llm_kernels
