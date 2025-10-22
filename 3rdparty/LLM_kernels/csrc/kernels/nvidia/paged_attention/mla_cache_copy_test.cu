/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <gtest/gtest.h>
#include <random>

#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class MlaPagedAttentionTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();

    // Calc offsets.
    for (size_t i = 0; i < batch_size_; i++) {
      host_input_offsets_.push_back(input_token_num_[i] + host_input_offsets_.back());
      host_block_offsets_.push_back((input_token_num_[i] + block_size_ - 1) / block_size_ + host_block_offsets_.back());

      host_prefix_offset_.push_back(input_prefix_len_[i] + host_prefix_offset_.back());
      host_without_prefix_offset_.push_back(host_input_offsets_[i + 1] - host_prefix_offset_[i + 1]);
    }

    total_len_without_prefix_ = host_without_prefix_offset_.back();
    total_prefix_len_ = host_prefix_offset_.back();
    total_len_with_prefix_ = total_len_without_prefix_ + total_prefix_len_;

    // k_src & v_src.
    int v_value_num = kv_lora_rank_;
    for (size_t i = 0; i < batch_size_; ++i) {
      size_t prefix_len = input_prefix_len_[i];
      for (int j = 0; j < input_token_num_[i]; ++j) {
        // Skip shared prefix.
        if (static_cast<size_t>(j) < prefix_len) {
          continue;
        }

        for (int k = 0; k < qk_rope_head_dim_; ++k) {
          host_k_src_.push_back((i * 100 + j) * 100 + k);
        }

        for (int v = 0; v < v_value_num; ++v) {
          host_v_src_.push_back((i * 100 + j) * 100 + v);
        }
      }
    }

    // Malloc device buffers.
    cudaMallocAsync(&dev_k_src_, host_k_src_.size() * sizeof(float), stream);
    cudaMallocAsync(&dev_v_src_, host_v_src_.size() * sizeof(float), stream);
    cudaMallocAsync(&dev_k_list_, host_block_offsets_.back() * sizeof(void*), stream);
    cudaMallocAsync(&dev_v_list_, host_block_offsets_.back() * sizeof(void*), stream);
    cudaMallocAsync(&dev_input_offsets_, host_input_offsets_.size() * sizeof(size_t), stream);
    cudaMallocAsync(&dev_input_lengths_, input_token_num_.size() * sizeof(int), stream);
    cudaMallocAsync(&dev_prefix_offsets_, host_input_offsets_.size() * sizeof(size_t), stream);
    cudaMallocAsync(&dev_without_prefix_offsets_, host_input_offsets_.size() * sizeof(size_t), stream);
    cudaMallocAsync(&dev_block_offsets_, host_block_offsets_.size() * sizeof(int), stream);

    // Copy host to device.
    cudaMemcpyAsync(dev_k_src_, host_k_src_.data(), host_k_src_.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_v_src_, host_v_src_.data(), host_v_src_.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(dev_input_offsets_, host_input_offsets_.data(), host_input_offsets_.size() * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_input_lengths_, input_token_num_.data(), input_token_num_.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_prefix_offsets_, host_prefix_offset_.data(), host_prefix_offset_.size() * sizeof(size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_without_prefix_offsets_, host_without_prefix_offset_.data(),
                    host_without_prefix_offset_.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_block_offsets_, host_block_offsets_.data(), host_block_offsets_.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Malloc k blocks and v blocks.
    // Make k & v as same pointers.
    host_k_list_ptrs_.resize(host_block_offsets_.back());
    for (int i = 0; i < host_block_offsets_.back(); i++) {
      cudaMallocAsync(&host_k_list_ptrs_[i], block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) * sizeof(float), stream);
      cudaMemsetAsync(host_k_list_ptrs_[i], 0, block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) * sizeof(float),
                      stream);
    }
    host_v_list_ptrs_ = host_k_list_ptrs_;
    cudaMemcpyAsync(dev_k_list_, host_k_list_ptrs_.data(), host_k_list_ptrs_.size() * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_v_list_, host_v_list_ptrs_.data(), host_v_list_ptrs_.size() * sizeof(float*),
                    cudaMemcpyHostToDevice, stream);

    // resize and set initial value.
    host_q_states_.resize(total_len_without_prefix_ * num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_));
    host_k_states_.resize(total_len_without_prefix_ * num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_));
    host_v_states_.resize(total_len_without_prefix_ * num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_));
    for (int i = 0; i < total_len_without_prefix_; ++i) {
      for (int j = 0; j < num_heads_; ++j) {
        for (int k = 0; k < qk_nope_head_dim_ + qk_rope_head_dim_; ++k) {
          int index = i * num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_) +
                      j * (qk_nope_head_dim_ + qk_rope_head_dim_) + k;
          host_q_states_[index] = (i * 100 + j) * 100 + k;
          host_k_states_[index] = (i * 200 + j) * 200 + k;
          host_v_states_[index] = (i * 300 + j) * 300 + k;
        }
      }
    }

    // resize, but no value.
    host_attn_q_states_.resize(total_len_with_prefix_ * num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_), 0);
    host_attn_k_states_.resize(total_len_with_prefix_ * num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_), 0);
    host_attn_v_states_.resize(total_len_with_prefix_ * num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_), 0);
    host_kv_buffer_.resize(total_prefix_len_ * kv_lora_rank_, 0);
    host_k_up_buffer_.resize(total_prefix_len_ * num_heads_ * qk_nope_head_dim_, 0);
    host_v_up_buffer_.resize(total_prefix_len_ * num_heads_ * v_head_dim_, 0);

    cudaMallocAsync(&dev_q_states_, host_q_states_.size() * sizeof(float), stream);
    cudaMallocAsync(&dev_k_states_, host_k_states_.size() * sizeof(float), stream);
    cudaMallocAsync(&dev_v_states_, host_v_states_.size() * sizeof(float), stream);
    cudaMallocAsync(&dev_attn_q_states_, host_attn_q_states_.size() * sizeof(float), stream);
    cudaMallocAsync(&dev_attn_k_states_, host_attn_k_states_.size() * sizeof(float), stream);
    cudaMallocAsync(&dev_attn_v_states_, host_attn_v_states_.size() * sizeof(float), stream);
    cudaMallocAsync(&dev_kv_buffer_, host_kv_buffer_.size() * sizeof(float), stream);
    cudaMallocAsync(&dev_k_up_buffer_, host_k_up_buffer_.size() * sizeof(float), stream);
    cudaMallocAsync(&dev_v_up_buffer_, host_v_up_buffer_.size() * sizeof(float), stream);

    cudaMemcpyAsync(dev_q_states_, host_q_states_.data(), host_q_states_.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(dev_k_states_, host_k_states_.data(), host_k_states_.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(dev_v_states_, host_v_states_.data(), host_v_states_.size() * sizeof(float), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(dev_attn_q_states_, host_attn_q_states_.data(), host_attn_q_states_.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_attn_k_states_, host_attn_k_states_.data(), host_attn_k_states_.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_attn_v_states_, host_attn_v_states_.data(), host_attn_v_states_.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_kv_buffer_, host_kv_buffer_.data(), host_kv_buffer_.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_k_up_buffer_, host_k_up_buffer_.data(), host_k_up_buffer_.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_v_up_buffer_, host_v_up_buffer_.data(), host_v_up_buffer_.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);
  }

  void TearDown() override {
    // Free device memory.
    cudaFreeAsync(dev_k_src_, stream);
    cudaFreeAsync(dev_v_src_, stream);
    cudaFreeAsync(dev_k_list_, stream);
    cudaFreeAsync(dev_v_list_, stream);
    cudaFreeAsync(dev_input_offsets_, stream);
    cudaFreeAsync(dev_input_lengths_, stream);
    cudaFreeAsync(dev_prefix_offsets_, stream);
    cudaFreeAsync(dev_block_offsets_, stream);
    for (auto ptr : host_k_list_ptrs_) {
      cudaFreeAsync(ptr, stream);
    }

    // Free memory buffer
    cudaFreeAsync(dev_q_states_, stream);
    cudaFreeAsync(dev_k_states_, stream);
    cudaFreeAsync(dev_v_states_, stream);
    cudaFreeAsync(dev_attn_q_states_, stream);
    cudaFreeAsync(dev_attn_k_states_, stream);
    cudaFreeAsync(dev_attn_v_states_, stream);
    cudaFreeAsync(dev_kv_buffer_, stream);
    cudaFreeAsync(dev_k_up_buffer_, stream);
    cudaFreeAsync(dev_v_up_buffer_, stream);

    cudaStreamSynchronize(stream);

    NvidiaTestSuitBase::TearDown();
  }

  void CopyDeviceBlocksToHost(std::vector<float>& host_k_dst) {
    // Copy result to host, include shared part, to checking the correction.
    host_k_dst.resize(host_block_offsets_.back() * (kv_lora_rank_ + qk_rope_head_dim_) * block_size_);
    for (int i = 0; i < host_block_offsets_.back(); i++) {
      cudaMemcpyAsync(host_k_dst.data() + i * block_size_ * (kv_lora_rank_ + qk_rope_head_dim_), host_k_list_ptrs_[i],
                      block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) * sizeof(float), cudaMemcpyDeviceToHost,
                      stream);
    }
    cudaStreamSynchronize(stream);
  }

 protected:
  using NvidiaTestSuitBase::stream;

 protected:
  // Assume we have two seq, their seq lengths are 17 and 41.
  // their prefix length are 16 and 32, with token_block_size 16.
  //
  // bs = 2
  // block_size = 16
  // k_stride_size = 64
  // v_stride_size = 512
  // k_scale = 1.0
  // v_scale = 1.0
  // input_offsets:  [0, 17, 58]
  // prefix_offsets: [0, 16, 48]
  // without_prefix_offsets: [0, 1, 10]
  // block_offsets: [0, 2, 5]

  int kv_lora_rank_ = 512;
  int qk_rope_head_dim_ = 64;
  int qk_nope_head_dim_ = 128;
  int v_head_dim_ = qk_nope_head_dim_;

  int num_heads_ = 8;

  size_t batch_size_ = 2;
  size_t block_size_ = 64;

  float k_scale_ = 1.0;
  float v_scale_ = 1.0;

  int total_prefix_len_;
  int total_len_with_prefix_;
  int total_len_without_prefix_;

  std::vector<int> input_prefix_len_ = {1000, 1000};
  std::vector<int> input_token_num_ = {1002, 1002};

  std::vector<size_t> host_input_offsets_ = {0};
  std::vector<int> host_block_offsets_ = {0};
  std::vector<size_t> host_prefix_offset_ = {0};
  std::vector<size_t> host_without_prefix_offset_ = {0};

  // The k & v that not contain prefix part.
  std::vector<float> host_k_src_;
  std::vector<float> host_v_src_;

  std::vector<float*> host_k_list_ptrs_;
  std::vector<float*> host_v_list_ptrs_;

  // device buffer.
  float* dev_k_src_;
  float* dev_v_src_;
  void** dev_k_list_;
  void** dev_v_list_;
  size_t* dev_input_offsets_;
  int* dev_input_lengths_;
  size_t* dev_prefix_offsets_;
  size_t* dev_without_prefix_offsets_;
  int* dev_block_offsets_;

  // contiguous memory buffer.
  std::vector<float> host_q_states_;
  std::vector<float> host_k_states_;
  std::vector<float> host_v_states_;

  std::vector<float> host_attn_q_states_;
  std::vector<float> host_attn_k_states_;
  std::vector<float> host_attn_v_states_;

  std::vector<float> host_kv_buffer_;
  std::vector<float> host_k_up_buffer_;
  std::vector<float> host_v_up_buffer_;

  float* dev_q_states_;
  float* dev_k_states_;
  float* dev_v_states_;

  float* dev_attn_q_states_;
  float* dev_attn_k_states_;
  float* dev_attn_v_states_;

  float* dev_kv_buffer_;
  float* dev_k_up_buffer_;
  float* dev_v_up_buffer_;
};

TEST_F(MlaPagedAttentionTestSuit, MlaFlashKVCacheCopyKernelAccTest) {
  // Lanch kernel wrapper.
  MlaFlashKVCacheCopy<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_k_src_, dev_v_src_, dev_k_list_, dev_v_list_, dev_prefix_offsets_, dev_without_prefix_offsets_,
      dev_block_offsets_, block_size_, batch_size_, total_len_without_prefix_, qk_rope_head_dim_, kv_lora_rank_,
      k_scale_, v_scale_, nullptr);
  cudaDeviceSynchronize();

  // Copy result to host, include shared part, to check the correction.
  std::vector<float> host_k_dst;
  CopyDeviceBlocksToHost(host_k_dst);

  // Verify result, should skip prefix part.
  size_t k_total_idx = 0;
  size_t v_total_idx = 0;
  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    size_t prefix_len = input_prefix_len_[batch_idx];
    for (int token_idx = 0; token_idx < input_token_num_[batch_idx]; ++token_idx) {
      // Check k.
      for (int k_idx = 0; k_idx < qk_rope_head_dim_; ++k_idx) {
        if (static_cast<size_t>(token_idx) >= prefix_len) {
          size_t k_total_dst_idx = host_block_offsets_[batch_idx] * block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) +
                                   token_idx * (kv_lora_rank_ + qk_rope_head_dim_) + (kv_lora_rank_ + k_idx);
          EXPECT_FLOAT_EQ(host_k_src_[k_total_idx], host_k_dst[k_total_dst_idx]);
          ++k_total_idx;
        }
      }

      // Check v.
      for (int v_idx = 0; v_idx < kv_lora_rank_; ++v_idx) {
        if (static_cast<size_t>(token_idx) >= prefix_len) {
          size_t v_total_dst_idx = host_block_offsets_[batch_idx] * block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) +
                                   token_idx * (kv_lora_rank_ + qk_rope_head_dim_) + v_idx;
          EXPECT_FLOAT_EQ(host_v_src_[v_total_idx], host_k_dst[v_total_dst_idx]);
          ++v_total_idx;
        }
      }
    }
  }
}

TEST_F(MlaPagedAttentionTestSuit, MlaPagedKVCacheCopyKernelAccTest) {
  for (int req_q_len = 1; req_q_len <= 2; ++req_q_len) {
    // Lanch kernel wrapper.
    MlaPagedKVCacheCopy<float, float, llm_kernels::utils::KVCacheType::kAuto>(
        dev_v_src_, dev_k_src_, dev_k_list_, dev_input_lengths_, dev_block_offsets_, block_size_, batch_size_,
        req_q_len, kv_lora_rank_, qk_rope_head_dim_, kv_lora_rank_, qk_rope_head_dim_, k_scale_, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy result to host, to check the correction.
    std::vector<float> host_k_dst;
    CopyDeviceBlocksToHost(host_k_dst);

    // Verify result.
    size_t kv_c_total_idx = 0;
    size_t k_pe_total_idx = 0;
    for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
      for (int token_idx = 0; token_idx < req_q_len; ++token_idx) {
        // Check kv_c.
        for (int i = 0; i < kv_lora_rank_; ++i) {
          size_t kv_c_total_dst_idx =
              host_block_offsets_[batch_idx] * block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) +
              (input_token_num_[batch_idx] - req_q_len + token_idx) * (kv_lora_rank_ + qk_rope_head_dim_) + i;
          EXPECT_FLOAT_EQ(host_v_src_[kv_c_total_idx], host_k_dst[kv_c_total_dst_idx]);
          ++kv_c_total_idx;
        }
        // Check k_pe.
        for (int i = 0; i < qk_rope_head_dim_; ++i) {
          size_t k_pe_total_dst_idx =
              host_block_offsets_[batch_idx] * block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) +
              (input_token_num_[batch_idx] - req_q_len + token_idx) * (kv_lora_rank_ + qk_rope_head_dim_) +
              (kv_lora_rank_ + i);
          EXPECT_FLOAT_EQ(host_k_src_[k_pe_total_idx], host_k_dst[k_pe_total_dst_idx]);
          ++k_pe_total_idx;
        }
      }
    }
  }
}

// Performance test is disabled by default
TEST_F(MlaPagedAttentionTestSuit, DISABLED_MlaPagedKVCacheCopyKernelPerfTest) {
  const std::vector<size_t> batch_sizes = {1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  const int req_q_len = 1;

  for (const size_t batch_size : batch_sizes) {
    const size_t token_num = batch_size * req_q_len;
    // Prepare device data
    BufferMeta d_kv_c =
        CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {token_num, static_cast<size_t>(kv_lora_rank_)},
                                    /*is_random_init*/ true);
    BufferMeta d_kv_pe =
        CreateBuffer<__nv_bfloat16>(MemoryType::MEMORY_GPU, {token_num, static_cast<size_t>(qk_rope_head_dim_)},
                                    /*is_random_init*/ true);
    BufferMeta h_input_lengths = CreateBuffer<int>(MemoryType::MEMORY_CPU, {batch_size}, /*is_random_init*/ false);
    BufferMeta h_block_offsets = CreateBuffer<int>(MemoryType::MEMORY_CPU, {batch_size}, /*is_random_init*/ false);
    BufferMeta h_kv_list = CreateBuffer<size_t>(MemoryType::MEMORY_CPU, {batch_size}, /*is_random_init*/ false);
    // Construct arbitrary input length and block offset
    for (size_t i = 0; i < batch_size; i++) {
      reinterpret_cast<int*>(h_input_lengths.data_ptr)[i] = 30;
      reinterpret_cast<int*>(h_block_offsets.data_ptr)[i] = i;
      cudaMallocAsync(&(reinterpret_cast<void**>(h_kv_list.data_ptr)[i]),
                      block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) * sizeof(__nv_bfloat16), stream);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    BufferMeta d_input_lengths = CopyToDevice<int>(h_input_lengths);
    BufferMeta d_block_offsets = CopyToDevice<int>(h_block_offsets);
    BufferMeta d_kv_list = CopyToDevice<size_t>(h_kv_list);

    const int warmups = 5;
    const int iterations = 10;
    // Run kernel
    auto cuda_run = [&]() {
      MlaPagedKVCacheCopy<__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto>(
          reinterpret_cast<__nv_bfloat16*>(d_kv_c.data_ptr), reinterpret_cast<__nv_bfloat16*>(d_kv_pe.data_ptr),
          reinterpret_cast<void**>(d_kv_list.data_ptr), reinterpret_cast<int*>(d_input_lengths.data_ptr),
          reinterpret_cast<int*>(d_block_offsets.data_ptr), block_size_, batch_size, req_q_len, kv_lora_rank_,
          qk_rope_head_dim_, kv_lora_rank_, qk_rope_head_dim_, k_scale_, stream);
    };
    const float elapsed_ms = MeasureCudaExecutionTime(cuda_run, stream, warmups, iterations);
    std::cout << "Token num: " << token_num << ", Execution time: " << elapsed_ms << " ms" << std::endl;

    // Free data
    DeleteBuffer(d_kv_c);
    DeleteBuffer(d_kv_pe);
  }
}

TEST_F(MlaPagedAttentionTestSuit, MlaGetFromCompressedCacheTest) {
  // 初始化缓存块数据
  const size_t kv_stride_size = kv_lora_rank_ + qk_rope_head_dim_;
  std::vector<std::vector<float>> host_block_list(host_block_offsets_.back());
  for (size_t i = 0; i < static_cast<size_t>(host_block_offsets_.back()); ++i) {
    host_block_list[i].resize(block_size_ * kv_stride_size);
  }

  // 为每个块填充测试数据
  std::default_random_engine random_engine;
  std::uniform_real_distribution<float> random_range(0, 1);
  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    const size_t base_block_offset = host_block_offsets_[batch_idx];
    const size_t token_num = input_token_num_[batch_idx];

    const size_t block_num = (token_num + block_size_ - 1) / block_size_;
    for (size_t block_idx = 0; block_idx < block_num; ++block_idx) {
      const size_t total_block_idx = base_block_offset + block_idx;
      for (size_t i = 0; i < block_size_ * kv_stride_size; ++i) {
        host_block_list[total_block_idx][i] = random_range(random_engine);
      }
      cudaMemcpyAsync(host_k_list_ptrs_[total_block_idx], host_block_list[total_block_idx].data(),
                      host_block_list[total_block_idx].size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
  }
  cudaStreamSynchronize(stream);

  // 准备输出缓冲区
  const int total_len = total_len_with_prefix_;
  std::vector<float> host_latent_buffer(total_len * kv_lora_rank_);
  std::vector<float> host_rope_buffer(total_len * qk_rope_head_dim_);

  float *dev_latent_buffer, *dev_rope_buffer;
  cudaMalloc(&dev_latent_buffer, host_latent_buffer.size() * sizeof(float));
  cudaMalloc(&dev_rope_buffer, host_rope_buffer.size() * sizeof(float));

  // 执行kernel
  MlaGetFromCompressedCache<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_rope_buffer, dev_latent_buffer, dev_k_list_, total_len, dev_input_offsets_, dev_block_offsets_, block_size_,
      qk_rope_head_dim_, kv_lora_rank_, stream);
  cudaStreamSynchronize(stream);

  // 将结果复制回主机
  cudaMemcpy(host_latent_buffer.data(), dev_latent_buffer, host_latent_buffer.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(host_rope_buffer.data(), dev_rope_buffer, host_rope_buffer.size() * sizeof(float), cudaMemcpyDeviceToHost);

  // 验证结果
  const float* host_latent_ptr = host_latent_buffer.data();
  const float* host_rope_ptr = host_rope_buffer.data();

  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    size_t token_offset = host_input_offsets_[batch_idx];
    size_t token_num = input_token_num_[batch_idx];
    size_t base_block_offset = host_block_offsets_[batch_idx];

    for (size_t token_idx = 0; token_idx < token_num; ++token_idx) {
      size_t block_idx = token_idx / block_size_;
      size_t token_offset_in_block = token_idx % block_size_;
      size_t total_block_idx = base_block_offset + block_idx;

      for (int i = 0; i < kv_lora_rank_; ++i) {
        const size_t src_offset = token_offset_in_block * kv_stride_size + i;
        EXPECT_FLOAT_EQ(*host_latent_ptr++, host_block_list[total_block_idx][src_offset]);
      }
      for (int i = 0; i < qk_rope_head_dim_; ++i) {
        const size_t src_offset = token_offset_in_block * kv_stride_size + kv_lora_rank_ + i;
        EXPECT_FLOAT_EQ(*host_rope_ptr++, host_block_list[total_block_idx][src_offset]);
      }
    }
  }

  // 释放临时分配的内存
  cudaFree(dev_latent_buffer);
  cudaFree(dev_rope_buffer);
}

TEST_F(MlaPagedAttentionTestSuit, MlaReverseFlashKVCacheCopyTest) {
  // 初始化缓存块数据
  const size_t kv_stride_size = kv_lora_rank_ + qk_rope_head_dim_;
  std::vector<std::vector<float>> host_block_list(host_block_offsets_.back());
  for (size_t i = 0; i < static_cast<size_t>(host_block_offsets_.back()); ++i) {
    host_block_list[i].resize(block_size_ * kv_stride_size);
  }

  // 为每个块填充测试数据
  std::default_random_engine random_engine;
  std::uniform_real_distribution<float> random_range(0, 1);
  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    const size_t base_block_offset = host_block_offsets_[batch_idx];
    const size_t token_num = input_token_num_[batch_idx];

    const size_t block_num = (token_num + block_size_ - 1) / block_size_;
    for (size_t block_idx = 0; block_idx < block_num; ++block_idx) {
      const size_t total_block_idx = base_block_offset + block_idx;
      for (size_t i = 0; i < block_size_ * kv_stride_size; ++i) {
        host_block_list[total_block_idx][i] = random_range(random_engine);
      }
      cudaMemcpyAsync(host_k_list_ptrs_[total_block_idx], host_block_list[total_block_idx].data(),
                      host_block_list[total_block_idx].size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
  }
  cudaStreamSynchronize(stream);

  // 准备输出缓冲区
  const int total_len = total_len_with_prefix_;
  const int total_q_len = total_len_without_prefix_;
  std::vector<float> host_latent_buffer(total_len * kv_lora_rank_);
  std::vector<float> host_rope_buffer(total_len * qk_rope_head_dim_);

  float *dev_latent_buffer, *dev_rope_buffer;
  cudaMalloc(&dev_latent_buffer, host_latent_buffer.size() * sizeof(float));
  cudaMalloc(&dev_rope_buffer, host_rope_buffer.size() * sizeof(float));

  // 执行kernel
  MlaFlashPrefixKVReverseCacheCopy<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_rope_buffer, dev_latent_buffer, dev_k_list_, dev_prefix_offsets_, dev_input_offsets_, dev_block_offsets_,
      block_size_, total_len, qk_rope_head_dim_, kv_lora_rank_, k_scale_, v_scale_, stream);
  MlaFlashWithoutPrefixKVCopy<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_rope_buffer, dev_latent_buffer, dev_k_src_, dev_v_src_, dev_prefix_offsets_, dev_without_prefix_offsets_,
      total_q_len, qk_rope_head_dim_, kv_lora_rank_, stream);

  cudaStreamSynchronize(stream);

  // 将结果复制回主机
  cudaMemcpy(host_latent_buffer.data(), dev_latent_buffer, host_latent_buffer.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(host_rope_buffer.data(), dev_rope_buffer, host_rope_buffer.size() * sizeof(float), cudaMemcpyDeviceToHost);

  // 验证结果
  const float* host_latent_ptr = host_latent_buffer.data();
  const float* host_rope_ptr = host_rope_buffer.data();
  const float* host_k_src_ptr = host_k_src_.data();
  const float* host_v_src_ptr = host_v_src_.data();

  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    size_t token_offset = host_input_offsets_[batch_idx];
    size_t prefix_num = input_prefix_len_[batch_idx];
    size_t token_num = input_token_num_[batch_idx];
    size_t base_block_offset = host_block_offsets_[batch_idx];

    size_t token_idx = 0;
    for (; token_idx < prefix_num; ++token_idx) {
      size_t block_idx = token_idx / block_size_;
      size_t token_offset_in_block = token_idx % block_size_;
      size_t total_block_idx = base_block_offset + block_idx;

      for (int i = 0; i < kv_lora_rank_; ++i) {
        const size_t src_offset = token_offset_in_block * kv_stride_size + i;
        EXPECT_FLOAT_EQ(*host_latent_ptr++, host_block_list[total_block_idx][src_offset]);
      }
      for (int i = 0; i < qk_rope_head_dim_; ++i) {
        const size_t src_offset = token_offset_in_block * kv_stride_size + kv_lora_rank_ + i;
        EXPECT_FLOAT_EQ(*host_rope_ptr++, host_block_list[total_block_idx][src_offset]);
      }
    }

    for (; token_idx < token_num; ++token_idx) {
      for (int i = 0; i < kv_lora_rank_; ++i) {
        EXPECT_FLOAT_EQ(*host_latent_ptr++, *host_v_src_ptr++);
      }
      for (int i = 0; i < qk_rope_head_dim_; ++i) {
        EXPECT_FLOAT_EQ(*host_rope_ptr++, *host_k_src_ptr++);
      }
    }
  }

  // 释放临时分配的内存
  cudaFree(dev_latent_buffer);
  cudaFree(dev_rope_buffer);
}

TEST_F(MlaPagedAttentionTestSuit, MlaFlexibleTokenCacheCopyTest) {
  // 初始化缓存块数据
  const size_t kv_stride_size = kv_lora_rank_ + qk_rope_head_dim_;
  std::vector<std::vector<float>> host_block_list(host_block_offsets_.back());
  for (size_t i = 0; i < static_cast<size_t>(host_block_offsets_.back()); ++i) {
    host_block_list[i].resize(block_size_ * kv_stride_size);
  }

  // 为每个块填充测试数据
  std::default_random_engine random_engine;
  std::uniform_real_distribution<float> random_range(0, 1);
  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    const size_t base_block_offset = host_block_offsets_[batch_idx];
    const size_t token_num = input_token_num_[batch_idx];

    const size_t block_num = (token_num + block_size_ - 1) / block_size_;
    for (size_t block_idx = 0; block_idx < block_num; ++block_idx) {
      const size_t total_block_idx = base_block_offset + block_idx;
      for (size_t i = 0; i < block_size_ * kv_stride_size; ++i) {
        host_block_list[total_block_idx][i] = random_range(random_engine);
      }
      cudaMemcpyAsync(host_k_list_ptrs_[total_block_idx], host_block_list[total_block_idx].data(),
                      host_block_list[total_block_idx].size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
  }
  cudaStreamSynchronize(stream);

  // 准备flexible传输信息
  const int layer_idx = 0;
  const int flexible_len = block_size_ * 2;
  std::vector<int> host_src_flexible_token_idx(flexible_len);
  std::vector<int> host_dst_flexible_token_idx(flexible_len);
  std::vector<float*> host_src_flexible_kv_cache_ptrs(flexible_len);
  std::vector<float*> host_dst_flexible_kv_cache_ptrs(flexible_len);

  const int src_start_token_idx = 62;
  const int dst_start_token_idx = 60;
  assert(batch_size_ >= 2);
  assert(flexible_len + src_start_token_idx <= input_token_num_[0] &&
         flexible_len + dst_start_token_idx <= input_token_num_[1]);
  for (size_t i = 0; i < flexible_len; ++i) {
    int src_token_idx = src_start_token_idx + i;
    int dst_token_idx = dst_start_token_idx + i;
    int src_block_idx = host_block_offsets_[0] + src_token_idx / block_size_;
    int dst_block_idx = host_block_offsets_[1] + dst_token_idx / block_size_;
    host_src_flexible_token_idx[i] = src_token_idx % block_size_;
    host_dst_flexible_token_idx[i] = dst_token_idx % block_size_;
    host_src_flexible_kv_cache_ptrs[i] = host_k_list_ptrs_[src_block_idx];
    host_dst_flexible_kv_cache_ptrs[i] = host_k_list_ptrs_[dst_block_idx];
  }

  int *dev_src_flexible_token_idx, *dev_dst_flexible_token_idx;
  cudaMalloc(&dev_src_flexible_token_idx, flexible_len * sizeof(int));
  cudaMalloc(&dev_dst_flexible_token_idx, flexible_len * sizeof(int));

  float** dev_src_flexible_kv_cache_ptrs;
  float** dev_dst_flexible_kv_cache_ptrs;
  cudaMalloc(&dev_src_flexible_kv_cache_ptrs, flexible_len * sizeof(float*));
  cudaMalloc(&dev_dst_flexible_kv_cache_ptrs, flexible_len * sizeof(float*));

  cudaMemcpyAsync(dev_src_flexible_token_idx, host_src_flexible_token_idx.data(), flexible_len * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dev_dst_flexible_token_idx, host_dst_flexible_token_idx.data(), flexible_len * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dev_src_flexible_kv_cache_ptrs, host_src_flexible_kv_cache_ptrs.data(), flexible_len * sizeof(float*),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dev_dst_flexible_kv_cache_ptrs, host_dst_flexible_kv_cache_ptrs.data(), flexible_len * sizeof(float*),
                  cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  // 执行kernel
  MlaFlexibleTokenCacheCopy<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_src_flexible_kv_cache_ptrs, dev_dst_flexible_kv_cache_ptrs, dev_src_flexible_token_idx,
      dev_dst_flexible_token_idx, block_size_, layer_idx, flexible_len, qk_rope_head_dim_ + kv_lora_rank_, stream);

  // 将结果复制回主机并验证结果
  std::vector<std::vector<float>> host_flexible_cache(flexible_len);
  for (size_t i = 0; i < flexible_len; ++i) {
    // 只需验证一个token
    host_flexible_cache[i].resize(kv_stride_size);
    int src_token_idx = src_start_token_idx + i;
    int dst_token_idx = dst_start_token_idx + i;
    int src_block_idx = host_block_offsets_[0] + src_token_idx / block_size_;
    int dst_block_idx = host_block_offsets_[1] + dst_token_idx / block_size_;
    std::vector<float> host_src_k_list_vec = host_block_list[src_block_idx];
    float* host_dst_k_list_ptr = host_k_list_ptrs_[dst_block_idx] + host_dst_flexible_token_idx[i] * kv_stride_size;
    cudaMemcpy(host_flexible_cache[i].data(), host_dst_k_list_ptr, kv_stride_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (size_t j = 0; j < kv_stride_size; ++j) {
      EXPECT_FLOAT_EQ(host_flexible_cache[i][j],
                      host_src_k_list_vec[host_src_flexible_token_idx[i] * kv_stride_size + j]);
    }
  }

  // 释放临时分配的内存
  cudaFree(dev_src_flexible_token_idx);
  cudaFree(dev_dst_flexible_token_idx);
  cudaFree(dev_src_flexible_kv_cache_ptrs);
  cudaFree(dev_dst_flexible_kv_cache_ptrs);
}

TEST_F(MlaPagedAttentionTestSuit, MlaFlashFlexibleKCacheCopyTest) {
  // 准备flexible cache信息
  assert(batch_size_ >= 2);
  std::vector<size_t> flexible_length(batch_size_);
  flexible_length[1] = block_size_ * 2;
  std::vector<size_t> host_flexible_offset{0};
  std::vector<size_t> input_flexible_len(batch_size_);
  for (size_t i = 0; i < batch_size_; ++i) {
    size_t flexible_offset = input_prefix_len_[i] - flexible_length[i];
    host_flexible_offset.push_back(host_flexible_offset.back() + flexible_offset);
    input_flexible_len[i] = host_flexible_offset.back();
  }

  size_t* dev_flexible_offset;
  cudaMalloc(&dev_flexible_offset, host_flexible_offset.size() * sizeof(size_t));
  cudaMemcpy(dev_flexible_offset, host_flexible_offset.data(), host_flexible_offset.size() * sizeof(size_t),
             cudaMemcpyHostToDevice);

  // 准备输入缓冲区
  const int total_len = total_len_with_prefix_;
  std::vector<float> host_rope_buffer(0);
  for (size_t i = 0; i < batch_size_; ++i) {
    for (int j = 0; j < input_token_num_[i]; ++j) {
      for (int k = 0; k < qk_rope_head_dim_; ++k) {
        host_rope_buffer.push_back((i * 100 + j) * 100 + k);
      }
    }
  }
  assert(host_rope_buffer.size() == total_len * qk_rope_head_dim_);

  float* dev_rope_buffer;
  cudaMalloc(&dev_rope_buffer, host_rope_buffer.size() * sizeof(float));
  cudaMemcpy(dev_rope_buffer, host_rope_buffer.data(), host_rope_buffer.size() * sizeof(float), cudaMemcpyHostToDevice);

  // 执行kernel
  MlaFlashFlexibleKCacheCopy<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_rope_buffer, dev_k_list_, dev_flexible_offset, dev_prefix_offsets_, dev_input_offsets_, dev_block_offsets_,
      block_size_, batch_size_, total_len, qk_rope_head_dim_, kv_lora_rank_, k_scale_, stream);

  cudaStreamSynchronize(stream);

  // 将结果复制回主机
  std::vector<float> host_k_dst;
  CopyDeviceBlocksToHost(host_k_dst);

  // 验证结果，只验证flexible部分
  size_t k_total_idx = 0;
  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    size_t prefix_len = input_prefix_len_[batch_idx];
    size_t flexible_len = input_flexible_len[batch_idx];
    for (int token_idx = 0; token_idx < input_token_num_[batch_idx]; ++token_idx) {
      // Check k.
      for (int k_idx = 0; k_idx < qk_rope_head_dim_; ++k_idx) {
        if (static_cast<size_t>(token_idx) >= flexible_len && static_cast<size_t>(token_idx) < prefix_len) {
          size_t k_total_dst_idx = host_block_offsets_[batch_idx] * block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) +
                                   token_idx * (kv_lora_rank_ + qk_rope_head_dim_) + (kv_lora_rank_ + k_idx);
          EXPECT_FLOAT_EQ(host_rope_buffer[k_total_idx], host_k_dst[k_total_dst_idx]);
        }
        ++k_total_idx;
      }
    }
  }

  // 释放临时分配的内存
  cudaFree(dev_flexible_offset);
  cudaFree(dev_rope_buffer);
}

TEST_F(MlaPagedAttentionTestSuit, FillKVScaleIntoBufferTest) {
  // 初始化测试数据
  const size_t num_heads = 128;
  float k_scale = 0.5f;
  float v_scale = 1.5f;

  // 准备输出缓冲区
  void* dev_k_scale_buffer;
  void* dev_v_scale_buffer;
  cudaMalloc(&dev_k_scale_buffer, num_heads * sizeof(float));
  cudaMalloc(&dev_v_scale_buffer, num_heads * sizeof(float));

  // 执行kernel
  InvokeFillKVScaleIntoBuffer(dev_k_scale_buffer, dev_v_scale_buffer, &k_scale, &v_scale, num_heads, stream);
  cudaStreamSynchronize(stream);

  // 将结果复制回主机
  std::vector<float> host_k_scale_buffer(num_heads);
  std::vector<float> host_v_scale_buffer(num_heads);
  cudaMemcpy(host_k_scale_buffer.data(), dev_k_scale_buffer, num_heads * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_v_scale_buffer.data(), dev_v_scale_buffer, num_heads * sizeof(float), cudaMemcpyDeviceToHost);

  // 验证结果
  for (int i = 0; i < num_heads; ++i) {
    EXPECT_FLOAT_EQ(host_k_scale_buffer[i], k_scale);
    EXPECT_FLOAT_EQ(host_v_scale_buffer[i], v_scale);
  }

  // 释放临时分配的内存
  cudaFree(dev_k_scale_buffer);
  cudaFree(dev_v_scale_buffer);
}

// 使用 CPU 校验 CalculateChecksumKernel 的计算结果
TEST_F(MlaPagedAttentionTestSuit, CalculateChecksumKernelCpuVerify) {
  // 测试指针数量与每个缓冲区的数据大小（字节）
  const int num_ptrs = 3;
  const size_t data_size_in_bytes = 1024;  // 必须是 sizeof(size_t) 的整数倍
  const size_t num_elements = data_size_in_bytes / sizeof(size_t);

  // 准备主机侧数据与设备缓冲区
  std::vector<std::vector<size_t>> host_datas(num_ptrs);
  std::vector<void*> d_buffers(num_ptrs, nullptr);
  std::vector<size_t> host_cpu_checksums(num_ptrs, 0);

  for (int p = 0; p < num_ptrs; ++p) {
    host_datas[p].resize(num_elements);
    // 填充可预测数据，避免溢出
    size_t base = static_cast<size_t>((p + 1) * 1000);
    size_t sum = 0;
    for (size_t i = 0; i < num_elements; ++i) {
      host_datas[p][i] = base + i;
      sum += host_datas[p][i];
    }
    host_cpu_checksums[p] = sum;

    // 为每个指针分配设备缓冲区并拷贝数据
    cudaMallocAsync(&d_buffers[p], data_size_in_bytes, stream);
    cudaMemcpyAsync(d_buffers[p], host_datas[p].data(), data_size_in_bytes, cudaMemcpyHostToDevice, stream);
  }
  cudaStreamSynchronize(stream);

  // 将设备缓冲区指针数组拷贝至设备
  void** d_ptrs = nullptr;
  cudaMallocAsync(&d_ptrs, num_ptrs * sizeof(void*), stream);
  cudaMemcpyAsync(d_ptrs, d_buffers.data(), num_ptrs * sizeof(void*), cudaMemcpyHostToDevice, stream);

  // 结果缓冲区
  size_t* d_results = nullptr;
  cudaMallocAsync(&d_results, num_ptrs * sizeof(size_t), stream);

  // 启动 kernel 计算校验和
  llm_kernels::nvidia::CalculateChecksum(d_ptrs, d_results, num_ptrs, data_size_in_bytes, stream);
  cudaStreamSynchronize(stream);

  // 拷回结果并与 CPU 结果比对
  std::vector<size_t> host_results(num_ptrs, 0);
  cudaMemcpy(host_results.data(), d_results, num_ptrs * sizeof(size_t), cudaMemcpyDeviceToHost);

  for (int p = 0; p < num_ptrs; ++p) {
    EXPECT_EQ(host_results[p], host_cpu_checksums[p]) << "Checksum mismatch at index " << p;
  }

  // 释放资源
  cudaFreeAsync(d_results, stream);
  cudaFreeAsync(d_ptrs, stream);
  for (int p = 0; p < num_ptrs; ++p) {
    cudaFreeAsync(d_buffers[p], stream);
  }
  cudaStreamSynchronize(stream);
}
 
}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
