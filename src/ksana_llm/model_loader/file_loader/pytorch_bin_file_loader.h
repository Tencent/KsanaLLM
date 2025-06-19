/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <Python.h>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

#include "ksana_llm/model_loader/file_loader/base_file_loader.h"

namespace py = pybind11;

namespace ksana_llm {

// Used to load pytorch bin files.
// .bin is a serialized of python's dict.
class PytorchBinFileLoader : public BaseFileLoader {
 public:
  explicit PytorchBinFileLoader(const std::string& filename);
  virtual ~PytorchBinFileLoader();

  // Load weight names from file, but not load it.
  virtual Status LoadWeightNames(std::vector<std::string>& weight_names) override;

  // Load weights in weight_names.
  virtual Status LoadModelWeights(const std::vector<std::string>& weight_names,
                                  std::unordered_map<std::string, Tensor>& result) override;

 private:
  Status LoadPytorchModelModelState();

 private:
  std::string filename_;

  py::object py_model_;
  py::dict state_dict_;

  // Whether the file have been loaded.
  bool loaded_ = false;
};

}  // namespace ksana_llm
