/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/attention_layer.h"  // For PositionEncoding enum
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/utils/config/model_config_parser.h"  // For RoPEScalingFactor

#ifdef ENABLE_CUDA
#  include <optional>
#  include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#endif

namespace ksana_llm {

class FlashSparseMlaIndexerLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;
  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

  // Get workspace size needed for this layer
  virtual size_t GetWorkspaceSize() override;

 private:
  // Forward 主函数模板
  template <typename SCALAR_T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  // Model parameters
  int dim_;                      // hidden dimension
  int n_heads_;                  // number of index heads
  int head_dim_;                 // index head dimension
  int rope_head_dim_;            // rope head dimension
  int index_topk_;               // top-k value
  int block_size_;               // kv block size: number of tokens in one block
  int quant_block_size_;         // quantization block size: number of tokens sharing one quantization parameter
  DataType kv_cache_dtype_;      // kv cache data type
  float softmax_scale_;          // softmax scale = head_dim ** -0.5
  int max_batch_size_;           // max batch size
  int max_seq_len_;              // max sequence length
  int max_position_embeddings_;  // max position embeddings
  int layer_index_;              // layer index
  size_t max_step_token_num_;    // max step token num from runtime_config

#ifdef ENABLE_CUDA
  // RoPE embedding
  std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda> rotary_embedding_cuda_;
#endif
};

}  // namespace ksana_llm
