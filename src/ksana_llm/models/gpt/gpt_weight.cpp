/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/gpt/gpt_weight.h"

namespace ksana_llm {

template <typename T>
GPTWeight<T>::GPTWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, rank, context) {}

template <typename T>
void GPTWeight<T>::ProcessWeights() {
  CommonWeight<T>::ProcessWeights();

  // Fairseq transformer uses unlearnable sinusoidal position encoding.
  // Refer to
  // https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py#L43
  if (model_config_.vocab_size == 7000) {
    size_t max_token_num = model_config_.max_token_num;
    size_t hidden_units = model_config_.hidden_units;
    size_t half_hidden_units = hidden_units / 2;
    size_t partition = hidden_units / context_->GetTensorParallelSize();
    // The position embedding is 1-indexed and adjusted by pad id.
    // We adopt this hack to get consistent results.
    // See https://github.com/huggingface/transformers/issues/15292
    const size_t offset = 1 + model_config_.pad_id;
    const float scale = -std::log(10000.f) / (half_hidden_units - 1);
    std::vector<T> position_weight_cpu(max_token_num * partition, 0.f);
    for (size_t pos = 0, index = 0; pos < max_token_num; pos++) {
      for (size_t i = rank_ * partition; i < (rank_ + 1) * partition; i++) {
        if (i < half_hidden_units) {
          position_weight_cpu[index++] = std::sin((pos + offset) * std::exp(i * scale));
        } else {
          position_weight_cpu[index++] = std::cos((pos + offset) * std::exp((i - half_hidden_units) * scale));
        }
      }
    }

    const std::string layer_name = "model.embed_positions.weight";
    tensor_manager_->AddWeightTensor(layer_name, {max_token_num, partition}, model_config_.weight_data_type);
    Tensor& position_weight_tensor = weights_map_[layer_name];
    weights_data_type_map_[layer_name] = model_config_.weight_data_type;
    size_t pitch = partition * sizeof(T);
    Memcpy2DAsync(position_weight_tensor.GetPtr<void>(), pitch, position_weight_cpu.data(), pitch, pitch, max_token_num,
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[rank_]);
    StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
  }
}

template class GPTWeight<float>;
template class GPTWeight<float16>;
template class GPTWeight<bfloat16>;

}  // namespace ksana_llm
