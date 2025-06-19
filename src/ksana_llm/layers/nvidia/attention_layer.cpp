/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/attention_layer.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
Status AttentionLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  int parameter_index = 0;
  mm_quant_mode_ = std::any_cast<const QuantMode>(parameters[parameter_index++]);
  layernorm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  use_qk_norm_ = std::any_cast<const bool>(parameters[parameter_index++]);
  layer_index_ = std::any_cast<const int>(parameters[parameter_index++]);
  layer_num_ = std::any_cast<const int>(parameters[parameter_index++]);
  int max_position_embeddings = std::any_cast<const int>(parameters[parameter_index++]);
  num_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_kv_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  head_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  stride_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  tensor_para_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  kv_cache_dtype_ = std::any_cast<DataType>(parameters[parameter_index++]);
  k_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  v_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  uint32_t rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  float base = std::any_cast<const float>(parameters[parameter_index++]);
  qk_rope_head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  qk_nope_head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  q_lora_rank_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  kv_lora_rank_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  v_head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  bool is_neox = std::any_cast<const bool>(parameters[parameter_index++]);
  PositionEncoding position_encoding = std::any_cast<const PositionEncoding>(parameters[parameter_index++]);
  void* cos_sin_cache_ptr = std::any_cast<void*>(parameters[parameter_index++]);

  block_size_ = Singleton<Environment>::GetInstance()->GetBlockSize();
  block_token_num_ = Singleton<Environment>::GetInstance()->GetBlockTokenNum();
  // +7 is because there are 6 more elements down:
  // attn_temperature_tuning_, attn_scale_, floor_scale_, RoPEScalingFactor, is_multi_token_forward, mrope_section_ptr
  enable_qk_pre_norm_before_rotary_pos_ = std::any_cast<const bool>(parameters[parameter_index + 7]);

  // setting scaling factor and mode
  if (position_encoding == PositionEncoding::ROPE) {
    RoPEScalingFactor rope_scaling_factor_config = std::any_cast<const RoPEScalingFactor>(parameters[parameter_index]);
    // parameters[parameter_index + 1] is max_batch_size
    llm_kernels::nvidia::RotaryEmbeddingType rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::DEFAULT;
    float scaling_factor = 1.0f;
    float low_freq_factor = 1.0f;
    float high_freq_factor = 4.0f;
    int original_max_position_embeddings = 8192;
    float scaling_alpha = 1.0f;
    // deepseek_yarn params
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    float mscale = 1.0f;
    float mscale_all_dim = 1.0f;
    const int* mrope_section_ptr = nullptr;
    if (rope_scaling_factor_config.type == "dynamic") {
      if (rope_scaling_factor_config.has_alpha) {
        rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::DYNAMIC_NTK_ALPHA;
        scaling_alpha = rope_scaling_factor_config.scaling_alpha;
      } else {
        rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::DYNAMIC_NTK_SCALING;
      }
      scaling_factor = rope_scaling_factor_config.factor;
    } else if (rope_scaling_factor_config.type == "linear") {
      rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::LINEAR_SCALING;
      scaling_factor = rope_scaling_factor_config.factor;
    } else if (rope_scaling_factor_config.type == "llama3") {
      rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::MULTIFREQ_SCALING;
      scaling_factor = rope_scaling_factor_config.factor;
      low_freq_factor = rope_scaling_factor_config.low_freq_factor;
      high_freq_factor = rope_scaling_factor_config.high_freq_factor;
      original_max_position_embeddings = rope_scaling_factor_config.original_max_position_embeddings;
    } else if (rope_scaling_factor_config.type == "yarn") {
      scaling_factor = rope_scaling_factor_config.factor;
      float attn_factor = 1.0f;
      if (rope_scaling_factor_config.use_deepseek_yarn) {
        rotary_dim = qk_rope_head_dim_;  // mla rope dim
        beta_fast = rope_scaling_factor_config.beta_fast;
        beta_slow = rope_scaling_factor_config.beta_slow;
        mscale_all_dim = rope_scaling_factor_config.mscale_all_dim;
        mscale = deepseek_yarn_get_mscale(scaling_factor, rope_scaling_factor_config.mscale) /
                 deepseek_yarn_get_mscale(scaling_factor, mscale_all_dim) * attn_factor;
        // Get attn scale
        float qk_head_dim_scale = 1.0f / sqrt(qk_rope_head_dim_ + qk_nope_head_dim_);
        float attn_mscale = deepseek_yarn_get_mscale(scaling_factor, mscale_all_dim);
        attn_scale_ = qk_head_dim_scale * attn_mscale * attn_mscale;
      } else {
        mscale = common_yarn_get_mscale(scaling_factor) * attn_factor;
      }
      rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::YARN_SCALING;
      original_max_position_embeddings = rope_scaling_factor_config.original_max_position_embeddings;
    } else if (rope_scaling_factor_config.type == "internlm2_dynamic") {
      rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::INTERNLM2_DYNAMIC_NTK_SCALING;
      scaling_factor = rope_scaling_factor_config.factor;
    } else if (rope_scaling_factor_config.type == "mrope") {
      if (/* is_multi_token_forward */ std::any_cast<const bool>(parameters[parameter_index + 5])) {
        rotary_embedding_type = llm_kernels::nvidia::RotaryEmbeddingType::MROPE;
        mrope_section_ptr = std::any_cast<const int*>(parameters[parameter_index + 6]);
      } else {
        // At the decoding stage, the effect of mrope and rope (default) are the same,
        // so we just adopt rope.
        rope_scaling_factor_config.type = "default";
      }
    } else if (rope_scaling_factor_config.type != "default") {
      KLLM_THROW(fmt::format("Unsupport rope scaling type: {}.", rope_scaling_factor_config.type));
    }
    rotary_embedding_cuda_.emplace();
    rotary_embedding_cuda_->SetConfig(
        static_cast<T*>(cos_sin_cache_ptr), rotary_dim, max_position_embeddings, base, head_size_, num_heads_,
        num_kv_heads_, stride_size_, is_neox, context_->GetComputeStreams()[rank_].Get(), rotary_embedding_type,
        scaling_factor, low_freq_factor, high_freq_factor, original_max_position_embeddings, scaling_alpha,
        mrope_section_ptr, beta_fast, beta_slow, mscale, mscale_all_dim, rope_scaling_factor_config.use_deepseek_yarn);
  } else if (position_encoding == PositionEncoding::ALIBI) {
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::GetAlibiSlopesCuda(reinterpret_cast<float*>(cos_sin_cache_ptr),
                                                                  num_heads_ * tensor_para_size_,
                                                                  context_->GetComputeStreams()[rank_].Get()));
    alibi_slopes_ = reinterpret_cast<void*>(cos_sin_cache_ptr) + num_heads_ * rank_ * sizeof(float);
  } else if (position_encoding == PositionEncoding::NO_ROPE) {
    no_rope_ = true;
    attn_temperature_tuning_ = std::any_cast<const size_t>(parameters[parameter_index + 2]) > 0;
    attn_scale_ = std::any_cast<const float>(parameters[parameter_index + 3]);
    floor_scale_ = std::any_cast<const size_t>(parameters[parameter_index + 4]);
  }
  StreamSynchronize(context_->GetComputeStreams()[rank_]);
  return Status();
}

template <typename T>
float AttentionLayer<T>::deepseek_yarn_get_mscale(const float scale, const float mscale) {
  if (scale <= 1.0f) {
    return 1.0f;
  }
  return 0.1f * mscale * std::log(scale) + 1.0f;
}

template <typename T>
float AttentionLayer<T>::common_yarn_get_mscale(const float scale) {
  if (scale <= 1.0f) {
    return 1.0f;
  }
  return 0.1f * std::log(scale) + 1.0f;
}

template class AttentionLayer<float>;
template class AttentionLayer<half>;
#ifdef ENABLE_BFLOAT16
template class AttentionLayer<__nv_bfloat16>;
#endif

}  // namespace ksana_llm
