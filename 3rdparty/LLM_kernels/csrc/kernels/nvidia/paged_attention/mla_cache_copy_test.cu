/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

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
    cudaMalloc(&dev_k_src_, host_k_src_.size() * sizeof(float));
    cudaMalloc(&dev_v_src_, host_v_src_.size() * sizeof(float));
    cudaMalloc(&dev_k_list_, host_block_offsets_.back() * sizeof(void*));
    cudaMalloc(&dev_v_list_, host_block_offsets_.back() * sizeof(void*));
    cudaMalloc(&dev_input_offsets_, host_input_offsets_.size() * sizeof(size_t));
    cudaMalloc(&dev_prefix_offsets_, host_input_offsets_.size() * sizeof(size_t));
    cudaMalloc(&dev_without_prefix_offsets_, host_input_offsets_.size() * sizeof(size_t));
    cudaMalloc(&dev_block_offsets_, host_block_offsets_.size() * sizeof(int));

    // Copy host to device.
    cudaMemcpy(dev_k_src_, host_k_src_.data(), host_k_src_.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v_src_, host_v_src_.data(), host_v_src_.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_input_offsets_, host_input_offsets_.data(), host_input_offsets_.size() * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_prefix_offsets_, host_prefix_offset_.data(), host_prefix_offset_.size() * sizeof(size_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_without_prefix_offsets_, host_without_prefix_offset_.data(),
               host_without_prefix_offset_.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_block_offsets_, host_block_offsets_.data(), host_block_offsets_.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    // Malloc k blocks and v blocks.
    // Make k & v as same pointers.
    host_k_list_ptrs_.resize(host_block_offsets_.back());
    for (int i = 0; i < host_block_offsets_.back(); i++) {
      cudaMalloc(&host_k_list_ptrs_[i], block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) * sizeof(float));
      cudaMemset(host_k_list_ptrs_[i], 0, block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) * sizeof(float));
    }
    host_v_list_ptrs_ = host_k_list_ptrs_;
    cudaMemcpy(dev_k_list_, host_k_list_ptrs_.data(), host_k_list_ptrs_.size() * sizeof(float*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v_list_, host_v_list_ptrs_.data(), host_v_list_ptrs_.size() * sizeof(float*),
               cudaMemcpyHostToDevice);

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

    cudaMalloc(&dev_q_states_, host_q_states_.size() * sizeof(float));
    cudaMalloc(&dev_k_states_, host_k_states_.size() * sizeof(float));
    cudaMalloc(&dev_v_states_, host_v_states_.size() * sizeof(float));
    cudaMalloc(&dev_attn_q_states_, host_attn_q_states_.size() * sizeof(float));
    cudaMalloc(&dev_attn_k_states_, host_attn_k_states_.size() * sizeof(float));
    cudaMalloc(&dev_attn_v_states_, host_attn_v_states_.size() * sizeof(float));
    cudaMalloc(&dev_kv_buffer_, host_kv_buffer_.size() * sizeof(float));
    cudaMalloc(&dev_k_up_buffer_, host_k_up_buffer_.size() * sizeof(float));
    cudaMalloc(&dev_v_up_buffer_, host_v_up_buffer_.size() * sizeof(float));

    cudaMemcpy(dev_q_states_, host_q_states_.data(), host_q_states_.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k_states_, host_k_states_.data(), host_k_states_.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v_states_, host_v_states_.data(), host_v_states_.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_attn_q_states_, host_attn_q_states_.data(), host_attn_q_states_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_attn_k_states_, host_attn_k_states_.data(), host_attn_k_states_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_attn_v_states_, host_attn_v_states_.data(), host_attn_v_states_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kv_buffer_, host_kv_buffer_.data(), host_kv_buffer_.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_k_up_buffer_, host_k_up_buffer_.data(), host_k_up_buffer_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v_up_buffer_, host_v_up_buffer_.data(), host_v_up_buffer_.size() * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  void TearDown() override {
    NvidiaTestSuitBase::TearDown();
    // Free device memory.
    cudaFree(dev_k_src_);
    cudaFree(dev_v_src_);
    cudaFree(dev_k_list_);
    cudaFree(dev_v_list_);
    cudaFree(dev_input_offsets_);
    cudaFree(dev_prefix_offsets_);
    cudaFree(dev_block_offsets_);
    for (auto ptr : host_k_list_ptrs_) {
      cudaFree(ptr);
    }

    // Free memory buffer
    cudaFree(dev_q_states_);
    cudaFree(dev_k_states_);
    cudaFree(dev_v_states_);
    cudaFree(dev_attn_q_states_);
    cudaFree(dev_attn_k_states_);
    cudaFree(dev_attn_v_states_);
    cudaFree(dev_kv_buffer_);
    cudaFree(dev_k_up_buffer_);
    cudaFree(dev_v_up_buffer_);
  }

  void CopyDeviceBlocksToHost(std::vector<float>& host_k_dst) {
    // Copy result to host, include shared part, to checking the correction.
    host_k_dst.resize(host_block_offsets_.back() * (kv_lora_rank_ + qk_rope_head_dim_) * block_size_);
    for (int i = 0; i < host_block_offsets_.back(); i++) {
      cudaMemcpy(host_k_dst.data() + i * block_size_ * (kv_lora_rank_ + qk_rope_head_dim_), host_k_list_ptrs_[i],
                 block_size_ * (kv_lora_rank_ + qk_rope_head_dim_) * sizeof(float), cudaMemcpyDeviceToHost);
    }
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

TEST_F(MlaPagedAttentionTestSuit, MlaExtendKVPrefixWithEmptyTest) {
  MlaExtendKVPrefixWithEmpty<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_k_states_, dev_v_states_, dev_attn_k_states_, dev_attn_v_states_, dev_prefix_offsets_,
      dev_without_prefix_offsets_, num_heads_, qk_nope_head_dim_ + qk_rope_head_dim_, total_len_without_prefix_,
      nullptr);
  cudaDeviceSynchronize();

  // Copy result to host.
  std::vector<float> host_attn_q_dst(host_attn_q_states_.size());
  cudaMemcpy(host_attn_q_dst.data(), dev_attn_q_states_, host_attn_q_dst.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Verify result.
  size_t stride_size = num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_);
  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    // All prefix should stay zero.
    size_t prefix_len = host_prefix_offset_[batch_idx + 1] - host_prefix_offset_[batch_idx];
    size_t dst_token_offset = host_prefix_offset_[batch_idx] + host_without_prefix_offset_[batch_idx];
    for (size_t i = 0; i < prefix_len; ++i) {
      for (size_t j = 0; j < stride_size; ++j) {
        size_t dst_offset = (dst_token_offset + i) * stride_size + j;
        EXPECT_FLOAT_EQ(host_attn_q_dst[dst_offset], 0);
      }
    }
  }
}

TEST_F(MlaPagedAttentionTestSuit, MlaCopyKeyBlockWithReplicationTest) {
  // Initialize prefix cache block.
  // kv_lora_rank_: The dimension of the downward projection matrix for keys and values, with a default value of 512.
  size_t kv_stride_size = kv_lora_rank_ + qk_rope_head_dim_;
  std::vector<std::vector<float>> host_block_list;
  for (size_t i = 0; i < static_cast<size_t>(host_block_offsets_.back()); ++i) {
    std::vector<float> block(block_size_ * kv_stride_size, 0);
    // Each cache block stores the key data of block_size_ tokens.
    // The size of each cache block is: block_size_ * kv_stride_size
    host_block_list.push_back(block);
  }

  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    size_t base_block_offset = host_block_offsets_[batch_idx];
    size_t prefix_len = host_prefix_offset_[batch_idx + 1] - host_prefix_offset_[batch_idx];

    size_t prefix_block_num = prefix_len / block_size_;
    for (size_t prefix_block_idx = 0; prefix_block_idx < prefix_block_num; ++prefix_block_idx) {
      size_t total_block_idx = base_block_offset + prefix_block_idx;
      for (size_t i = 0; i < block_size_ * kv_stride_size; ++i) {
        host_block_list[total_block_idx][i] = (total_block_idx * 100) + i;
        cudaMemcpy(host_k_list_ptrs_[total_block_idx], host_block_list[total_block_idx].data(),
                   host_block_list[total_block_idx].size() * sizeof(float), cudaMemcpyHostToDevice);
      }
    }
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  MlaCopyKeyBlockWithReplication<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_attn_k_states_ /*k_out*/, dev_k_list_ /*k_list*/, kv_lora_rank_ + qk_rope_head_dim_ /*src_stride_size*/,
      kv_lora_rank_ /*src_copy_offset*/, qk_rope_head_dim_ /*src_copy_len*/, num_heads_ /*dst_num_heads*/,
      qk_nope_head_dim_ + qk_rope_head_dim_ /*dst_head_size*/, qk_nope_head_dim_ /*dst_copy_offset*/,
      dev_prefix_offsets_ /*prefix_offsets*/, dev_without_prefix_offsets_ /*without_prefix_offsets*/,
      dev_block_offsets_ /*block_offsets*/, block_size_ /*block_size*/, total_prefix_len_ /*total_prefix_len*/,
      nullptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  const int RUNS = 10;
  float total_time = 0;
  for (int i = 0; i < RUNS; ++i) {
    cudaEventRecord(start);
    MlaCopyKeyBlockWithReplication<float, float, llm_kernels::utils::KVCacheType::kAuto>(
        dev_attn_k_states_ /*k_out*/, dev_k_list_ /*k_list*/, kv_lora_rank_ + qk_rope_head_dim_ /*src_stride_size*/,
        kv_lora_rank_ /*src_copy_offset*/, qk_rope_head_dim_ /*src_copy_len*/, num_heads_ /*dst_num_heads*/,
        qk_nope_head_dim_ + qk_rope_head_dim_ /*dst_head_size*/, qk_nope_head_dim_ /*dst_copy_offset*/,
        dev_prefix_offsets_ /*prefix_offsets*/, dev_without_prefix_offsets_ /*without_prefix_offsets*/,
        dev_block_offsets_ /*block_offsets*/, block_size_ /*block_size*/, total_prefix_len_ /*total_prefix_len*/,
        nullptr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t;
    cudaEventElapsedTime(&t, start, stop);
    total_time += t;
  }
  printf("MlaCopyKeyBlockWithReplication Average time: %.3f ms\n", total_time / RUNS);

  // Copy result to host.
  std::vector<float> host_attn_k_dst(host_attn_k_states_.size());
  cudaMemcpy(host_attn_k_dst.data(), dev_attn_k_states_, host_attn_k_dst.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Verify result.
  size_t stride_size = num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_);
  size_t head_stride_size = qk_nope_head_dim_ + qk_rope_head_dim_;
  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    // All prefix should be equal with block.
    size_t prefix_len = host_prefix_offset_[batch_idx + 1] - host_prefix_offset_[batch_idx];
    size_t dst_token_offset = host_prefix_offset_[batch_idx] + host_without_prefix_offset_[batch_idx];
    size_t base_block_offset = host_block_offsets_[batch_idx];

    for (size_t i = 0; i < prefix_len; ++i) {
      size_t total_block_idx = base_block_offset + (i / block_size_);
      for (size_t head_idx = 0; head_idx < num_heads_; ++head_idx) {
        for (size_t dim_idx = 0; dim_idx < head_stride_size; ++dim_idx) {
          size_t dst_offset = (dst_token_offset + i) * stride_size + (head_idx)*head_stride_size + dim_idx;
          // For every head, qk_nope_head_dim should be stay zero,
          if (dim_idx < qk_nope_head_dim_) {
            EXPECT_FLOAT_EQ(host_attn_k_dst[dst_offset], 0);
          }
          // qk_rope_head_dim should be equal with block.
          else {
            size_t block_offset = (i % block_size_) * kv_stride_size + (kv_lora_rank_ + dim_idx - qk_nope_head_dim_);
            EXPECT_FLOAT_EQ(host_attn_k_dst[dst_offset], host_block_list[total_block_idx][block_offset]);
          }
        }
      }
    }

    // All unique should be stay zero.
    size_t unique_len = host_without_prefix_offset_[batch_idx + 1] - host_without_prefix_offset_[batch_idx];
    for (size_t i = 0; i < unique_len; ++i) {
      for (size_t j = 0; j < stride_size; ++j) {
        size_t dst_offset = (dst_token_offset + prefix_len + i) * stride_size + j;
        EXPECT_FLOAT_EQ(host_attn_k_dst[dst_offset], 0);
      }
    }
  }
}

TEST_F(MlaPagedAttentionTestSuit, MlaCopyValueBlockToBufferTest) {
  // Initialize prefix cache block.
  size_t kv_stride_size = kv_lora_rank_ + qk_rope_head_dim_;
  std::vector<std::vector<float>> host_block_list;
  for (size_t i = 0; i < host_block_offsets_.back(); ++i) {
    std::vector<float> block(block_size_ * kv_stride_size, 0);
    host_block_list.push_back(block);
  }

  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    size_t base_block_offset = host_block_offsets_[batch_idx];
    size_t prefix_len = host_prefix_offset_[batch_idx + 1] - host_prefix_offset_[batch_idx];

    size_t prefix_block_num = prefix_len / block_size_;
    for (size_t prefix_block_idx = 0; prefix_block_idx < prefix_block_num; ++prefix_block_idx) {
      size_t total_block_idx = base_block_offset + prefix_block_idx;
      for (size_t i = 0; i < block_size_ * kv_stride_size; ++i) {
        host_block_list[total_block_idx][i] = (total_block_idx * 100) + i;
        cudaMemcpy(host_k_list_ptrs_[total_block_idx], host_block_list[total_block_idx].data(),
                   host_block_list[total_block_idx].size() * sizeof(float), cudaMemcpyHostToDevice);
      }
    }
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  MlaCopyValueBlockToBuffer<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_kv_buffer_, dev_v_list_, kv_stride_size, 0, kv_lora_rank_, kv_lora_rank_, dev_prefix_offsets_,
      dev_block_offsets_, block_size_, total_prefix_len_, nullptr);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  const int RUNS = 10;
  float total_time = 0;
  for (int i = 0; i < RUNS; ++i) {
    cudaEventRecord(start);
    MlaCopyValueBlockToBuffer<float, float, llm_kernels::utils::KVCacheType::kAuto>(
        dev_kv_buffer_, dev_v_list_, kv_stride_size, 0, kv_lora_rank_, kv_lora_rank_, dev_prefix_offsets_,
        dev_block_offsets_, block_size_, total_prefix_len_, nullptr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float t;
    cudaEventElapsedTime(&t, start, stop);
    total_time += t;
  }
  printf("MlaCopyValueBlockToBufferTest Average time: %.3f ms\n", total_time / RUNS);

  // Copy result to host.
  std::vector<float> host_kv_buffer_dst(host_kv_buffer_.size());
  cudaMemcpy(host_kv_buffer_dst.data(), dev_kv_buffer_, host_kv_buffer_dst.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Verify result.
  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    // All prefix should be equal with block.
    size_t prefix_len = host_prefix_offset_[batch_idx + 1] - host_prefix_offset_[batch_idx];
    size_t dst_token_offset = host_prefix_offset_[batch_idx];
    size_t base_block_offset = host_block_offsets_[batch_idx];

    for (size_t i = 0; i < prefix_len; ++i) {
      size_t total_block_idx = base_block_offset + (i / block_size_);
      for (size_t dim_idx = 0; dim_idx < kv_lora_rank_; ++dim_idx) {
        size_t dst_offset = (dst_token_offset + i) * kv_lora_rank_ + dim_idx;
        // For every head, only kv_lora_rank_ part is copied.
        if (dim_idx < kv_lora_rank_) {
          size_t block_offset = (i % block_size_) * kv_stride_size + dim_idx;
          EXPECT_FLOAT_EQ(host_kv_buffer_dst[dst_offset], host_block_list[total_block_idx][block_offset]);
        }
      }
    }
  }
}

TEST_F(MlaPagedAttentionTestSuit, MlaFillKVPrefixTest) {
  // Initialize k_up_buffer and v_up_buffer.
  for (size_t i = 0; i < total_prefix_len_; ++i) {
    for (size_t j = 0; j < num_heads_; ++j) {
      for (size_t k = 0; k < qk_nope_head_dim_; ++k) {
        int index = i * num_heads_ * (qk_nope_head_dim_) + j * (qk_nope_head_dim_) + k;
        host_k_up_buffer_[index] = (i * 100 + j) * 100 + k;
        host_v_up_buffer_[index] = (i * 200 + j) * 200 + k;
      }
    }
  }

  cudaMemcpy(dev_k_up_buffer_, host_k_up_buffer_.data(), host_k_up_buffer_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_v_up_buffer_, host_v_up_buffer_.data(), host_v_up_buffer_.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  MlaFillKVPrefix<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_attn_k_states_, dev_attn_v_states_, dev_k_up_buffer_, dev_v_up_buffer_, dev_prefix_offsets_,
      dev_without_prefix_offsets_, num_heads_, qk_nope_head_dim_, qk_nope_head_dim_ + qk_rope_head_dim_,
      total_prefix_len_, nullptr);
  cudaDeviceSynchronize();

  // Copy result to host.
  std::vector<float> host_attn_k_states_dst(host_attn_k_states_.size());
  cudaMemcpy(host_attn_k_states_dst.data(), dev_attn_k_states_, host_attn_k_states_.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Verify result.
  size_t dst_head_stride_size = qk_nope_head_dim_ + qk_rope_head_dim_;
  size_t dst_stride_size = num_heads_ * dst_head_stride_size;

  size_t src_head_stride_size = qk_nope_head_dim_;
  size_t src_stride_size = num_heads_ * src_head_stride_size;

  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    size_t prefix_len = host_prefix_offset_[batch_idx + 1] - host_prefix_offset_[batch_idx];
    size_t dst_token_offset = host_prefix_offset_[batch_idx] + host_without_prefix_offset_[batch_idx];
    size_t src_token_offset = host_prefix_offset_[batch_idx];

    for (size_t i = 0; i < prefix_len; ++i) {
      for (size_t head_idx = 0; head_idx < num_heads_; ++head_idx) {
        for (size_t dim_idx = 0; dim_idx < dst_head_stride_size; ++dim_idx) {
          if (dim_idx < src_head_stride_size) {
            size_t dst_offset = (dst_token_offset + i) * dst_stride_size + head_idx * dst_head_stride_size + dim_idx;
            size_t src_offset = (src_token_offset + i) * src_stride_size + head_idx * src_head_stride_size + dim_idx;
            EXPECT_FLOAT_EQ(host_attn_k_states_dst[dst_offset], host_k_up_buffer_[src_offset]);
          }
        }
      }
    }
  }
}

TEST_F(MlaPagedAttentionTestSuit, MlaDrainAttnOutPrefixTest) {
  // Initialize dev_attn_q_states as input, dev_q_states as output.
  for (int i = 0; i < total_len_with_prefix_; ++i) {
    for (int j = 0; j < num_heads_; ++j) {
      for (int k = 0; k < qk_nope_head_dim_ + qk_rope_head_dim_; ++k) {
        int index =
            i * num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_) + j * (qk_nope_head_dim_ + qk_rope_head_dim_) + k;
        host_attn_q_states_[index] = (i * 100 + j) * 100 + k;
      }
    }
  }
  cudaMemcpy(dev_attn_q_states_, host_attn_q_states_.data(), host_attn_q_states_.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  MlaDrainAttnOutPrefix<float, float, llm_kernels::utils::KVCacheType::kAuto>(
      dev_q_states_, dev_attn_q_states_, dev_prefix_offsets_, dev_without_prefix_offsets_, num_heads_,
      qk_nope_head_dim_ + qk_rope_head_dim_, batch_size_, total_len_without_prefix_, nullptr);
  cudaDeviceSynchronize();

  // Copy result to host.
  std::vector<float> host_q_states_dst(host_q_states_.size());
  cudaMemcpy(host_q_states_dst.data(), dev_q_states_, host_q_states_dst.size() * sizeof(float), cudaMemcpyDeviceToHost);

  // Verify result.
  size_t stride_size = num_heads_ * (qk_nope_head_dim_ + qk_rope_head_dim_);
  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    // All unique should be equal.
    size_t prefix_len = host_prefix_offset_[batch_idx + 1] - host_prefix_offset_[batch_idx];
    size_t unique_len = host_without_prefix_offset_[batch_idx + 1] - host_without_prefix_offset_[batch_idx];

    size_t src_token_offset = host_prefix_offset_[batch_idx] + host_without_prefix_offset_[batch_idx];
    size_t dst_token_offset = host_without_prefix_offset_[batch_idx];

    for (size_t i = 0; i < unique_len; ++i) {
      for (size_t j = 0; j < stride_size; ++j) {
        size_t src_offset = (src_token_offset + prefix_len + i) * stride_size + j;
        size_t dst_offset = (dst_token_offset + i) * stride_size + j;
        EXPECT_FLOAT_EQ(host_q_states_dst[dst_offset], host_attn_q_states_[src_offset]);
      }
    }
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
