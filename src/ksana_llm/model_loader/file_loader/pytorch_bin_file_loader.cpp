/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/file_loader/pytorch_bin_file_loader.h"

#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/script.h>

#include <Python.h>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <filesystem>
#include <unordered_set>

#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

PytorchBinFileLoader::PytorchBinFileLoader(const std::string& filename) : filename_(filename) {}

PytorchBinFileLoader::~PytorchBinFileLoader() { loaded_ = false; }

// Convert torch dtype to ksana dtype.
DataType GetTensorDataType(c10::ScalarType dtype) {
  DataType data_type = TYPE_INVALID;
  switch (dtype) {
    case c10::kBFloat16:
      data_type = TYPE_BF16;
      break;
    case torch::kFloat16:
      data_type = TYPE_FP16;
      break;
    case torch::kFloat32:
      data_type = TYPE_FP32;
      break;
    case torch::kInt32:
      data_type = TYPE_INT32;
      break;
    default:
      break;
  }
  return data_type;
}

Status PytorchBinFileLoader::LoadPytorchModelModelState() {
  if (!loaded_) {
    py::module torch = py::module::import("torch");

    try {
      py_model_ = torch.attr("load")(filename_, "cpu");
    } catch (const py::error_already_set& e) {
      PyErr_Clear();
      py::finalize_interpreter();
      return Status(RET_RUNTIME_FAILED, FormatStr("Failed to load file %s.", filename_.c_str()));
    }

    if (py::hasattr(py_model_, "state_dict")) {
      state_dict_ = py_model_.attr("state_dict")();
    } else {
      state_dict_ = py_model_;
    }

    loaded_ = true;
  }
  return Status();
}

Status PytorchBinFileLoader::LoadWeightNames(std::vector<std::string>& weight_names) {
  Status status = LoadPytorchModelModelState();
  if (!status.OK()) {
    return status;
  }

  for (auto& item : state_dict_) {
    std::string weight_name = py::str(item.first);
    weight_names.push_back(weight_name);
  }

  return Status();
}

Status PytorchBinFileLoader::LoadModelWeights(const std::vector<std::string>& weight_names,
                                              std::unordered_map<std::string, Tensor>& result) {
  Status status = LoadPytorchModelModelState();
  if (!status.OK()) {
    return status;
  }

  std::unordered_set<std::string> weight_name_set(weight_names.begin(), weight_names.end());
  for (auto& item : state_dict_) {
    std::string weight_name = py::str(item.first);
    if (weight_name_set.find(weight_name) == weight_name_set.end()) {
      KLLM_LOG_DEBUG << "Skip weight tensor name " << weight_name;
      continue;
    }
    KLLM_LOG_DEBUG << "Load weight tensor name " << weight_name;

    py::object value_obj = py::reinterpret_borrow<py::object>(item.second);
    const torch::Tensor& torch_tensor = THPVariable_Unpack(value_obj.ptr());

    // Get shape.
    std::vector<std::size_t> tensor_shape;
    tensor_shape.insert(tensor_shape.end(), torch_tensor.sizes().begin(), torch_tensor.sizes().end());

    // Get dtype
    DataType tensor_dtype = GetTensorDataType(torch_tensor.scalar_type());

    Tensor weight_tensor = Tensor(MemoryLocation::LOCATION_HOST, tensor_dtype, tensor_shape);

    memcpy(weight_tensor.GetPtr<void>(), torch_tensor.data_ptr(), weight_tensor.GetTotalBytes());
    result[weight_name] = weight_tensor;
  }

  return Status();
}

}  // namespace ksana_llm
