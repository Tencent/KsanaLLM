// Copyright 2024 Tencent Inc.  All rights reserved.
#include "pytorch_file_tensor_loader.h"
#include "logger.h"
#include "torch/csrc/autograd/python_variable.h"
// #include "ksana_llm/utils/nvidia/cuda_utils.h"

namespace ksana_llm {
// Constructor of PytorchFileTensorLoader that takes a file name as input
PytorchFileTensorLoader::PytorchFileTensorLoader(const std::string& file_name, const bool load_bias)
    : BaseFileTensorLoader(file_name, load_bias) {
  // Check if the file name has a ".bin" extension
  if (file_name_.length() > 4) {
    if (file_name_.substr(file_name_.length() - 4) == ".bin") {
      LoadPytorchBin();
    }
  }
}

// Function to load the PyTorch binary file
void PytorchFileTensorLoader::LoadPytorchBin() {
  py::module torch = py::module::import("torch");
  try {
    model_ = torch.attr("load")(file_name_, "cpu");
  } catch (const py::error_already_set& e) {
    PyErr_Clear();
    KLLM_LOG_ERROR << fmt::format("Failed to load file {}", file_name_);
    return;
  }
  py::dict state_dict;
  if (py::hasattr(model_, "state_dict")) {
    state_dict = model_.attr("state_dict")();
  } else {
    state_dict = model_;
  }

  for (auto& item : state_dict) {
    std::string tensor_name = py::str(item.first);
    if (!load_bias_ && tensor_name.find(".bias") != std::string::npos) {
      continue;
    }
    tensor_name_list_.push_back(tensor_name);
    KLLM_LOG_DEBUG << "PytorchFileTensorLoader read: " << tensor_name << " finished.";
    py::object value_obj = py::reinterpret_borrow<py::object>(item.second);
    pytorch_tensor_map_[tensor_name] = THPVariable_Unpack(value_obj.ptr());
  }
}

DataType PytorchFileTensorLoader::GetTensorDataType(const std::string& tensor_name) {
  DataType data_type = TYPE_INVALID;
  c10::ScalarType dtype = pytorch_tensor_map_[tensor_name].scalar_type();
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
    // (TODO)catheywang: if torch>=2.3
    // case c10::kFloat8_e4m3fn:
    //  data_type = TYPE_FP8_E4M3;
    //  break;
    default:
      break;
  }
  return data_type;
}

std::string PytorchFileTensorLoader::GetTensorFileName() { return file_name_; }

std::tuple<void*, size_t> PytorchFileTensorLoader::GetTensor(const std::string& tensor_name) {
  if (!pytorch_tensor_map_.count(tensor_name)) {
    return std::make_tuple(nullptr, 0);
  }
  auto& tensor = pytorch_tensor_map_[tensor_name];
  return std::make_tuple(tensor.data_ptr(), tensor.numel() * tensor.element_size());
}
std::vector<std::size_t> PytorchFileTensorLoader::GetTensorShape(const std::string& tensor_name) {
  if (!pytorch_tensor_map_.count(tensor_name)) {
    return {};
  }
  torch::Tensor& tensor = pytorch_tensor_map_[tensor_name];
  std::vector<size_t> tensor_shape(tensor.sizes().begin(), tensor.sizes().end());
  return tensor_shape;
}

}  // namespace ksana_llm
