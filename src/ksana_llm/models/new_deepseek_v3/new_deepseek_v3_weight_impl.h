/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <string>
#include <utility>
#include <regex>

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/kernels/permute.h"

#ifdef ENABLE_CUDA
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {
class NewDeepSeekV3WeightImplBase {
 public:
  virtual ~NewDeepSeekV3WeightImplBase() = default;

  // Permutation with buffer
  virtual Status PermuteWeight(Tensor & input_tensor, const std::vector<size_t> & permutation, int dev_rank) = 0;
  // Transpose and split weight along axis = 0, then with param `transpose` to decide whether to transpose back
  virtual Status TransSplitOptTrans(const Tensor & host_weight_tensor,
                            Tensor & output_tensor,
                            int dev_rank,
                            std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                            size_t para_size,
                            bool transpose) = 0;

  // Split weight along axis = 0, then with param `skip_transpose` to decide whether skip transpose
  virtual Status SplitOptTrans(const Tensor & host_weight_tensor,
                       Tensor & output_tensor,
                       int dev_rank,
                       std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                       size_t para_size,
                       bool skip_transpose) = 0;

  virtual Status GetExpertsIdx(const std::string& expert_name,
                        int32_t & layer_idx_,
                        int32_t & expert_idx_) = 0;

  virtual Status ProcessGateUpProjWeight(std::string& file_weight_name_,
                                 const Tensor& dev_tensor,
                                 std::unordered_map<std::string, Tensor>& device_model_weights,
                                 int dev_rank,
                                 bool is_quant_weight) = 0;
#ifdef ENABLE_CUDA
  virtual Status ProcessAbsorbWeightsTypeUKV(std::unordered_map<std::string, Tensor>& dev_weights_map_,
                                             int dev_rank,
                                             const std::shared_ptr<NewDeepSeekV3Config> &
                                                 new_deepseek_v3_config) = 0;
#endif
#ifdef ENABLE_FP8
  virtual bool LoadMoeFp8E4m3BlockWiseScale(const std::string & host_weight_name,
                                    const Tensor & host_weight_tensor,
                                    int dev_rank,
                                    std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                    std::unordered_map<std::string, Tensor> & device_model_weights,
                                    int32_t expert_idx) = 0;

  virtual bool LoadMlaFp8E4m3BlockWiseScale(const std::string & host_weight_name,
                                    const Tensor & host_weight_tensor,
                                    int dev_rank,
                                    std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                    std::unordered_map<std::string, Tensor> & device_model_weights) = 0;

  virtual Tensor DequantFp8E4m3BlockWiseTensor(const Tensor & weight_tensor,
                                                  const Tensor & weight_scale_tensor,
                                                  int dev_rank,
                                                  const std::shared_ptr<NewDeepSeekV3Config> &
                                                    new_deepseek_v3_config) = 0;
  virtual std::pair<Tensor, Tensor> QuantFp8E4m3BlockWiseTensor(Tensor & weight_tensor,
                                                                int dev_rank,
                                                                const std::shared_ptr<NewDeepSeekV3Config> &
                                                                new_deepseek_v3_config) = 0;
#ifdef ENABLE_FP8_TORCH
  virtual Status ProcessMlaFp8E4m3BlockWiseScaleOfWeight(std::unordered_set<std::string> & dequant_weights,
                                        int dev_rank,
                                        const std::shared_ptr<NewDeepSeekV3Config> &
                                          new_deepseek_v3_config,
                                        std::unordered_map<std::string, Tensor> &
                                          device_model_weights) = 0;
#endif
#endif
};

// TODO(huicongyao): invoke permutation buffer to avoid creating temporary tensor
template <typename T>
class NewDeepSeekV3WeightImpl : public NewDeepSeekV3WeightImplBase {
 public:
  explicit NewDeepSeekV3WeightImpl(const std::shared_ptr<Context> & context);
  virtual ~NewDeepSeekV3WeightImpl() = default;

  Status PermuteWeight(Tensor & input_tensor, const std::vector<size_t> & permutation, int dev_rank) override;
  Status TransSplitOptTrans(const Tensor & host_weight_tensor,
                            Tensor & output_tensor,
                            int dev_rank,
                            std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                            size_t para_size,
                            bool transpose = false) override;

  Status SplitOptTrans(const Tensor & host_weight_tensor,
                       Tensor & output_tensor,
                       int dev_rank,
                       std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                       size_t para_size,
                       bool skip_transpose = false) override;

  Status GetExpertsIdx(const std::string& expert_name,
                        int32_t & layer_idx_,
                        int32_t & expert_idx_) override;

  Status ProcessGateUpProjWeight(std::string& file_weight_name_,
                                 const Tensor& dev_tensor,
                                 std::unordered_map<std::string, Tensor>& device_model_weights,
                                 int dev_rank,
                                 bool is_quant_weight = false) override;
#ifdef ENABLE_CUDA
  Status ProcessAbsorbWeightsTypeUKV(std::unordered_map<std::string, Tensor>& dev_weights_map_,
                                     int dev_rank,
                                     const std::shared_ptr<NewDeepSeekV3Config> &
                                         new_deepseek_v3_config) override;
#endif

#ifdef ENABLE_FP8
  Tensor DequantFp8E4m3BlockWiseTensor(const Tensor & weight_tensor,
                                          const Tensor & weight_scale_tensor,
                                          int dev_rank,
                                          const std::shared_ptr<NewDeepSeekV3Config> &
                                              new_deepseek_v3_config) override;

  std::pair<Tensor, Tensor> QuantFp8E4m3BlockWiseTensor(Tensor & weight_tensor,
                                                        int dev_rank,
                                                        const std::shared_ptr<NewDeepSeekV3Config> &
                                                          new_deepseek_v3_config) override;
#ifdef ENABLE_FP8_TORCH
  Status ProcessMlaFp8E4m3BlockWiseScaleOfWeight(std::unordered_set<std::string> & dequant_weights,
                                          int dev_rank,
                                          const std::shared_ptr<NewDeepSeekV3Config> &
                                            new_deepseek_v3_config,
                                          std::unordered_map<std::string, Tensor> &
                                            device_model_weights) override;
#endif

  bool LoadMoeFp8E4m3BlockWiseScale(const std::string & host_weight_name,
                                    const Tensor & host_weight_tensor,
                                    int dev_rank,
                                    std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                    std::unordered_map<std::string, Tensor> & device_model_weights,
                                    int32_t expert_idx) override;

  bool LoadMlaFp8E4m3BlockWiseScale(const std::string & host_weight_name,
                                    const Tensor & host_weight_tensor,
                                    int dev_rank,
                                    std::shared_ptr<NewDeepSeekV3Config> & new_deepseek_v3_config,
                                    std::unordered_map<std::string, Tensor> & device_model_weights) override;
#endif

 private:
  std::shared_ptr<Context> context_;
  std::vector<std::unordered_map<std::string, Tensor>> permute_buffers_;
};

}  // namespace ksana_llm
