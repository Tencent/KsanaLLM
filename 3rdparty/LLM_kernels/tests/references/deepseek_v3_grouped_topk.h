/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <vector>

namespace llm_kernels {
namespace nvidia {
namespace test {

static constexpr int32_t kMaxCandidateExpertNum = 256;

template <int MAX_K>
void GetTopKIndices(const float* data, int* indices, int n, int k) {
  // Local arrays used as heap storage
  float heap_values[MAX_K];
  int heap_indices[MAX_K];

  // Initialize the heap
  for (int i = 0; i < k; i++) {
    heap_values[i] = data[i];
    heap_indices[i] = i;
  }

  // Build min heap
  for (int i = k / 2 - 1; i >= 0; i--) {
    int parent = i;
    while (true) {
      int leftChild = 2 * parent + 1;
      int rightChild = 2 * parent + 2;
      int smallest = parent;

      if (leftChild < k && heap_values[leftChild] < heap_values[smallest]) {
        smallest = leftChild;
      }
      if (rightChild < k && heap_values[rightChild] < heap_values[smallest]) {
        smallest = rightChild;
      }

      if (smallest == parent) break;

      // Swap elements
      float temp_val = heap_values[parent];
      heap_values[parent] = heap_values[smallest];
      heap_values[smallest] = temp_val;

      int temp_idx = heap_indices[parent];
      heap_indices[parent] = heap_indices[smallest];
      heap_indices[smallest] = temp_idx;

      parent = smallest;
    }
  }

  // Process remaining elements
  for (int i = k; i < n; i++) {
    if (data[i] > heap_values[0]) {
      heap_values[0] = data[i];
      heap_indices[0] = i;

      // Sift down
      int parent = 0;
      while (true) {
        int leftChild = 2 * parent + 1;
        int rightChild = 2 * parent + 2;
        int smallest = parent;

        if (leftChild < k && heap_values[leftChild] < heap_values[smallest]) {
          smallest = leftChild;
        }
        if (rightChild < k && heap_values[rightChild] < heap_values[smallest]) {
          smallest = rightChild;
        }

        if (smallest == parent) break;

        float temp_val = heap_values[parent];
        heap_values[parent] = heap_values[smallest];
        heap_values[smallest] = temp_val;

        int temp_idx = heap_indices[parent];
        heap_indices[parent] = heap_indices[smallest];
        heap_indices[smallest] = temp_idx;

        parent = smallest;
      }
    }
  }

  // Extract results
  for (int i = k - 1; i > 0; i--) {
    indices[k - 1 - i] = heap_indices[0];

    heap_values[0] = heap_values[i];
    heap_indices[0] = heap_indices[i];

    int parent = 0;
    int heapSize = i;

    while (true) {
      int leftChild = 2 * parent + 1;
      int rightChild = 2 * parent + 2;
      int smallest = parent;

      if (leftChild < heapSize && heap_values[leftChild] < heap_values[smallest]) {
        smallest = leftChild;
      }
      if (rightChild < heapSize && heap_values[rightChild] < heap_values[smallest]) {
        smallest = rightChild;
      }

      if (smallest == parent) break;

      float temp_val = heap_values[parent];
      heap_values[parent] = heap_values[smallest];
      heap_values[smallest] = temp_val;

      int temp_idx = heap_indices[parent];
      heap_indices[parent] = heap_indices[smallest];
      heap_indices[smallest] = temp_idx;

      parent = smallest;
    }
  }

  indices[k - 1] = heap_indices[0];
}

template <typename T>
void RunDeepSeekV3GroupedTopkRef(void* gating_output, void* e_bias, float routed_scaling_factor, void* topk_weights_ptr,
                                 void* topk_ids_ptr, int tokens_num, int num_experts, int topk, int num_expert_group,
                                 int topk_group) {
  std::vector<float> original_scores((tokens_num * num_experts), 0.0f);
  float* gating_output_f_ptr = reinterpret_cast<float*>(gating_output);
  float* e_bias_f_ptr = reinterpret_cast<float*>(e_bias);

  for (int token_idx = 0; token_idx < tokens_num; ++token_idx) {
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      float val = static_cast<float>(static_cast<T>(gating_output_f_ptr[token_idx * num_experts + expert_idx]));
      original_scores[token_idx * num_experts + expert_idx] = 1.0f / (1.0f + expf(-val));
    }
  }

  std::vector<float> scores((tokens_num * num_experts), 0.0f);
  for (int token_idx = 0; token_idx < tokens_num; ++token_idx) {
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      scores[token_idx * num_experts + expert_idx] =
          original_scores[token_idx * num_experts + expert_idx] + e_bias_f_ptr[expert_idx];
    }
  }

  // groups_scores = scores.topk(2, -1)).sum(-1);
  int experts_per_group = (num_experts + num_expert_group - 1) / num_expert_group;
  assert(experts_per_group >= 2);
  std::vector<float> groups_scores((tokens_num * num_expert_group), 0.0f);
  for (int token_idx = 0; token_idx < tokens_num; ++token_idx) {
    for (int g_expert_idx = 0; g_expert_idx < num_expert_group; ++g_expert_idx) {
      float max1 = -INFINITY;
      float max2 = -INFINITY;
      for (int expert_idx = 0; expert_idx < experts_per_group; ++expert_idx) {
        float cur_score = scores[token_idx * num_experts + g_expert_idx * experts_per_group + expert_idx];
        if (cur_score > max1) {
          max2 = max1;
          max1 = cur_score;
        } else if (cur_score > max2) {
          max2 = cur_score;
        }
      }
      groups_scores[token_idx * num_expert_group + g_expert_idx] = max1 + max2;
    }
  }

  std::vector<int32_t> group_idx((tokens_num * topk_group), 0);
  for (int token_idx = 0; token_idx < tokens_num; ++token_idx) {
    GetTopKIndices<kMaxCandidateExpertNum>(groups_scores.data() + token_idx * num_expert_group,
                                           group_idx.data() + token_idx * topk_group, num_expert_group, topk_group);
  }

  std::vector<float> tmp_scores((tokens_num * num_experts), 0.0f);
  for (int token_idx = 0; token_idx < tokens_num; ++token_idx) {
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      bool is_masked = false;
      int group_id = expert_idx / experts_per_group;
      for (int topk_g_idx = 0; topk_g_idx < topk_group; ++topk_g_idx) {
        if (group_id == group_idx[token_idx * topk_group + topk_g_idx]) {
          is_masked = true;
        }
      }

      if (is_masked) {
        tmp_scores[token_idx * num_experts + expert_idx] = scores[token_idx * num_experts + expert_idx];
      } else {
        tmp_scores[token_idx * num_experts + expert_idx] = 0.0f;
      }
    }
  }

  int32_t* topk_ids_int_ptr = reinterpret_cast<int32_t*>(topk_ids_ptr);
  for (int token_idx = 0; token_idx < tokens_num; ++token_idx) {
    GetTopKIndices<kMaxCandidateExpertNum>(tmp_scores.data() + token_idx * num_experts,
                                           topk_ids_int_ptr + token_idx * topk, num_experts, topk);
  }

  float* topk_weights_f_ptr = reinterpret_cast<float*>(topk_weights_ptr);
  for (int token_idx = 0; token_idx < tokens_num; ++token_idx) {
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
        if (expert_idx == topk_ids_int_ptr[token_idx * topk + topk_idx]) {
          topk_weights_f_ptr[token_idx * topk + topk_idx] = original_scores[token_idx * num_experts + expert_idx];
        }
      }
    }
  }

  for (int token_idx = 0; token_idx < tokens_num; ++token_idx) {
    float expert_sum_val = 0.0f;
    for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
      expert_sum_val += topk_weights_f_ptr[token_idx * topk + topk_idx];
    }
    for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
      topk_weights_f_ptr[token_idx * topk + topk_idx] =
          topk_weights_f_ptr[token_idx * topk + topk_idx] / expert_sum_val * routed_scaling_factor;
    }
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels