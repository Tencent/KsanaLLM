/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/nn/functional/normalization.h>

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

class MacheteUtils {
 public:
  MacheteUtils(std::shared_ptr<Context> context, int rank, int bits) : context_(context), rank_(rank), bits_(bits) {}

#ifdef ENABLE_CUDA

  template <typename T>
  torch::Tensor PackWeight(torch::Tensor& weight, QuantMode quant_method);

#endif

 private:
  std::shared_ptr<Context> context_{nullptr};
  int rank_;
  int bits_;
};

}  // namespace ksana_llm
