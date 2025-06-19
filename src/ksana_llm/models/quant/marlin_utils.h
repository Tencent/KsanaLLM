/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/nn/functional/normalization.h>

#include "ksana_llm/utils/context.h"

namespace ksana_llm {

class MarlinUtils {
 public:
  MarlinUtils(std::shared_ptr<Context> context, int rank, int bits, int groupsize);

#ifdef ENABLE_CUDA

  template <typename T>
  torch::Tensor MarlinPermuteScales(torch::Tensor s, int k, int n);

  torch::Tensor MarlinUnpackCols(const torch::Tensor& packed_q_w, int k, int n);

  torch::Tensor MarlinPackCols(const torch::Tensor& q_w, int k, int n);

  torch::Tensor MarlinZeroPoints(const torch::Tensor& zp_, int k, int n);

  torch::Tensor MarlinAwqToMarlinZeroPoints(const torch::Tensor& q_zp_packed, int k, int n);

  torch::Tensor MarlinSortGIdx(torch::Tensor& g_idx);

  torch::Tensor PackGptqWeight(torch::Tensor& qweight, std::optional<torch::Tensor> perm);

  torch::Tensor PackAwqWeight(torch::Tensor& qweight);

#endif

 private:
  std::shared_ptr<Context> context_{nullptr};
  int rank_;
  int bits_;
  int pack_factor_;
  int groupsize_;
};

}  // namespace ksana_llm
