/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <random>

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"

#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class MlaPagedAttentionTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();

    // Calc offsets.
    for (int i = 0; i < batch_size_; i++) {
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
    for (int i = 0; i < batch_size_; ++i) {
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
    cudaMallocAsync(&dev_prefix_offsets_, host_input_offsets_.size() * sizeof(size_t), stream);
    cudaMallocAsync(&dev_without_prefix_offsets_, host_input_offsets_.size() * sizeof(size_t), stream);
    cudaMallocAsync(&dev_block_offsets_, host_block_offsets_.size() * sizeof(int), stream);

    // Copy host to device.
    cudaMemcpyAsync(dev_k_src_, host_k_src_.data(), host_k_src_.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_v_src_, host_v_src_.data(), host_v_src_.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(dev_input_offsets_, host_input_offsets_.data(), host_input_offsets_.size() * sizeof(size_t),
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

TEST_F(MlaPagedAttentionTestSuit, MlaKVCacheCopyKernel) {
  // Lanch kernel wrapper.
  MlaKVCacheCopy<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_k_src_, dev_v_src_, dev_k_list_, dev_v_list_, dev_prefix_offsets_, dev_without_prefix_offsets_,
      dev_block_offsets_, block_size_, batch_size_, total_len_without_prefix_, qk_rope_head_dim_, kv_lora_rank_,
      k_scale_, v_scale_, nullptr);
  cudaDeviceSynchronize();

  // Copy result to host, include shared part, to checking the correction.
  std::vector<float> host_k_dst;
  CopyDeviceBlocksToHost(host_k_dst);

  // Verify result, should skip prefix part.
  size_t k_total_idx = 0;
  size_t v_total_idx = 0;
  for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
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

      for (size_t i = 0; i < kv_lora_rank_; ++i) {
        const size_t src_offset = token_offset_in_block * kv_stride_size + i;
        EXPECT_FLOAT_EQ(*host_latent_ptr++, host_block_list[total_block_idx][src_offset]);
      }
      for (size_t i = 0; i < qk_rope_head_dim_; ++i) {
        const size_t src_offset = token_offset_in_block * kv_stride_size + kv_lora_rank_ + i;
        EXPECT_FLOAT_EQ(*host_rope_ptr++, host_block_list[total_block_idx][src_offset]);
      }
    }
  }

  // 释放临时分配的内存
  cudaFree(dev_latent_buffer);
  cudaFree(dev_rope_buffer);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
