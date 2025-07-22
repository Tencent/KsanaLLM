/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "ksana_llm/kernels/permute.h"
#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"
#include "ksana_llm/models/quant/machete_utils.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/context.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {
class NewDeepSeekV3WeightImplBase {
 public:
  virtual ~NewDeepSeekV3WeightImplBase() = default;

  // Permutation with buffer
  virtual Status PermuteWeight(Tensor& input_tensor, const std::vector<size_t>& permutation, int dev_rank) = 0;

  // Transpose and split weight along axis = 0, then with param `transpose` to decide whether to transpose back
  virtual Status TransSplitOptTrans(const Tensor& host_weight_tensor, Tensor& output_tensor, int dev_rank,
                                    std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config, size_t para_size,
                                    bool transpose) = 0;

  // Split weight along axis = 0, then with param `skip_transpose` to decide whether skip transpose
  virtual Status SplitOptTrans(const Tensor& host_weight_tensor, Tensor& output_tensor, int dev_rank,
                               std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config, size_t para_size,
                               bool transpose) = 0;

  virtual Status GetExpertsIdx(const std::string& expert_name, int32_t& layer_idx_, int32_t& expert_idx_) = 0;

  virtual Status ProcessGateUpProjWeight(std::string& file_weight_name_, const Tensor& dev_tensor,
                                         std::unordered_map<std::string, Tensor>& device_model_weights, int dev_rank,
                                         bool is_quant_weight) = 0;
#ifdef ENABLE_FP8
  virtual bool LoadMoeFp8E4m3BlockWiseScale(const std::string& host_weight_name, const Tensor& host_weight_tensor,
                                            int dev_rank, std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
                                            std::unordered_map<std::string, Tensor>& device_model_weights,
                                            int32_t expert_idx) = 0;

  virtual bool LoadMlaFp8E4m3BlockWiseScale(const std::string& host_weight_name, const Tensor& host_weight_tensor,
                                            int dev_rank, std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
                                            std::unordered_map<std::string, Tensor>& device_model_weights) = 0;

  virtual Tensor DequantFp8E4m3BlockWiseTensor(const Tensor& weight_tensor, const Tensor& weight_scale_tensor,
                                               int dev_rank,
                                               const std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) = 0;
  virtual std::pair<Tensor, Tensor> QuantFp8E4m3BlockWiseTensor(
      Tensor& weight_tensor, int dev_rank, const std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) = 0;
#  ifdef ENABLE_FP8_TORCH
  virtual Status ProcessMlaFp8E4m3BlockWiseScaleOfWeight(
      std::unordered_set<std::string>& dequant_weights, int dev_rank,
      const std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
      std::unordered_map<std::string, Tensor>& device_model_weights) = 0;
#  endif
#endif

  virtual Status LoadInt4QuantWeight(std::unordered_map<std::string, Tensor>& host_gptq_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                     std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) = 0;

  virtual Status PostProcessInt4QuantWeights(std::unordered_map<std::string, Tensor>& device_model_weights,
                                             int dev_rank,
                                             std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) = 0;
};

template <typename T>
class NewDeepSeekV3WeightImpl : public NewDeepSeekV3WeightImplBase {
 public:
  explicit NewDeepSeekV3WeightImpl(const std::shared_ptr<Context>& context);
  virtual ~NewDeepSeekV3WeightImpl() = default;

  // TODO(huicongyao): reuse all temp allocated memory as buffers
  Status PermuteWeight(Tensor& input_tensor, const std::vector<size_t>& permutation, int dev_rank) override;
  Status TransSplitOptTrans(const Tensor& host_weight_tensor, Tensor& output_tensor, int dev_rank,
                            std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config, size_t para_size,
                            bool transpose = false) override;

  Status SplitOptTrans(const Tensor& host_weight_tensor, Tensor& output_tensor, int dev_rank,
                       std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config, size_t para_size,
                       bool transpose = false) override;

  Status GetExpertsIdx(const std::string& expert_name, int32_t& layer_idx_, int32_t& expert_idx_) override;

  Status ProcessGateUpProjWeight(std::string& file_weight_name_, const Tensor& dev_tensor,
                                 std::unordered_map<std::string, Tensor>& device_model_weights, int dev_rank,
                                 bool is_quant_weight = false) override;

#ifdef ENABLE_FP8
  Tensor DequantFp8E4m3BlockWiseTensor(const Tensor& weight_tensor, const Tensor& weight_scale_tensor, int dev_rank,
                                       const std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) override;

  std::pair<Tensor, Tensor> QuantFp8E4m3BlockWiseTensor(
      Tensor& weight_tensor, int dev_rank, const std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) override;
#  ifdef ENABLE_FP8_TORCH
  Status ProcessMlaFp8E4m3BlockWiseScaleOfWeight(
      std::unordered_set<std::string>& dequant_weights, int dev_rank,
      const std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
      std::unordered_map<std::string, Tensor>& device_model_weights) override;
#  endif

  bool LoadMoeFp8E4m3BlockWiseScale(const std::string& host_weight_name, const Tensor& host_weight_tensor, int dev_rank,
                                    std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
                                    std::unordered_map<std::string, Tensor>& device_model_weights,
                                    int32_t expert_idx) override;

  bool LoadMlaFp8E4m3BlockWiseScale(const std::string& host_weight_name, const Tensor& host_weight_tensor, int dev_rank,
                                    std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config,
                                    std::unordered_map<std::string, Tensor>& device_model_weights) override;
#endif

  Status LoadInt4QuantWeight(std::unordered_map<std::string, Tensor>& host_gptq_weights, int dev_rank,
                             std::unordered_map<std::string, Tensor>& device_model_weights,
                             std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) override;

  Status PostProcessInt4QuantWeights(std::unordered_map<std::string, Tensor>& device_model_weights, int dev_rank,
                                     std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config) override;

// TODO(huicongyao): support all quant methods for new ds weight loader
// TODO(huicongyao): reformat and unify all quant pack methods
#ifdef ENABLE_CUDA
  // For machete quant utils
  Tensor MachetePackWeight(Tensor& weight, int dev_rank, QuantMode quant_method);

  // For marlin quant utils
  Tensor MarlinPackGptqWeight(Tensor& qweight, Tensor& perm, int dev_rank, int bits, int pack_factor);

  Tensor MarlinPermuteScales(Tensor& s, int dev_rank, int k, int n, int group_size);

  Tensor DequantGptqWeight(Tensor& qweight, int dev_rank, std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config);
#endif

 private:
  std::shared_ptr<Context> context_;
  std::vector<std::unordered_map<std::string, Tensor>> permute_buffers_;
};

}  // namespace ksana_llm
