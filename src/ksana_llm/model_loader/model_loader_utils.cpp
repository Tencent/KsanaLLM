/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/model_loader_utils.h"

#include <Python.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <regex>
#include <string>

#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

Status GetModelFileList(const std::string& model_dir, std::vector<std::string>& model_file_list) {
  if (!std::filesystem::is_directory(model_dir)) {
    return Status(RET_INVALID_ARGUMENT, FormatStr("The directory %s is not exists.", model_dir.c_str()));
  }

  constexpr const char* SKIP_FILE_PREFIX = ".etag.";
  const std::vector<std::string> SKIP_FILE_LIST = {"training_args.bin", "optimizer.bin"};

  for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
    if (entry.is_regular_file()) {
      std::string relative_path = std::filesystem::relative(entry.path(), model_dir);

      if (relative_path.length() >= 6 && relative_path.compare(0, strlen(SKIP_FILE_PREFIX), SKIP_FILE_PREFIX) == 0) {
        continue;
      }

      if (std::find(SKIP_FILE_LIST.begin(), SKIP_FILE_LIST.end(), relative_path) != SKIP_FILE_LIST.end()) {
        continue;
      }

      model_file_list.emplace_back(entry.path());
    }
  }

  // Sort files by increasing order.
  std::sort(model_file_list.begin(), model_file_list.end());

  return Status();
}

Status GetModelFormat(const std::string& model_dir, ModelFormat& model_format) {
  constexpr const char* GGUF_EXT = ".gguf";
  constexpr const char* PYTORCH_EXT = ".bin";
  constexpr const char* SAFETENSOR_EXT = ".safetensors";

  std::vector<std::string> model_file_list;
  Status status = GetModelFileList(model_dir, model_file_list);
  if (!status.OK()) {
    return status;
  }

  bool format_matched = false;
  for (const auto& filename : model_file_list) {
    if (filename.find(SAFETENSOR_EXT) != std ::string::npos) {
      model_format = ModelFormat::PYTORCH_SAFETENSOR;
      format_matched = true;
      break;
    }

    if (filename.find(GGUF_EXT) != std::string::npos) {
      model_format = ModelFormat::GGUF;
      format_matched = true;
      break;
    }

    if (filename.find(PYTORCH_EXT) != std::string::npos) {
      model_format = ModelFormat::PYTORCH_BIN;
      format_matched = true;
      break;
    }
  }

  if (!format_matched) {
    return Status(RET_INVALID_ARGUMENT, "Get model for error, no file extension matched.");
  }

  return Status();
}

Status GetPythonWeightMapPath(std::vector<std::string>& paths) {
  PyObject* sys_path = PySys_GetObject("path");
  if (!sys_path || !PyList_Check(sys_path)) {
    return Status(RET_INTERNAL_UNKNOWN_ERROR, "Get python sys path error.");
  }

  // Search within the ksana_llm packages
  PyObject* path_obj = PyList_GetItem(sys_path, 0);
  if (path_obj && PyUnicode_Check(path_obj) && PyUnicode_AsUTF8String(path_obj)) {
    std::string python_dir(PyBytes_AsString(PyUnicode_AsUTF8String(path_obj)));
    paths.push_back(python_dir);
  }

  // Search within the python site-packages
  Py_ssize_t size = PyList_Size(sys_path);
  std::string package_suffix = "site-packages";
  for (Py_ssize_t i = 1; i < size; ++i) {
    PyObject* path_obj = PyList_GetItem(sys_path, i);
    if (path_obj && PyUnicode_Check(path_obj) && PyUnicode_AsUTF8String(path_obj)) {
      std::string python_dir(PyBytes_AsString(PyUnicode_AsUTF8String(path_obj)));
      if (python_dir.length() > package_suffix.length() &&
          python_dir.substr(python_dir.length() - package_suffix.length()) == package_suffix) {
        paths.push_back(python_dir + "/ksana_llm");
      }
    }
  }

  return Status();
}

Status FilterPytorchBinModelFileList(std::vector<std::string>& model_file_list) {
  for (auto it = model_file_list.begin(); it != model_file_list.end();) {
    std::filesystem::path file_path = *it;
    if (file_path.extension() != ".bin") {
      it = model_file_list.erase(it);
      continue;
    }
    ++it;
  }

  if (model_file_list.empty()) {
    return Status(RET_INVALID_ARGUMENT, "FilterPytorchBinModelFileList error, empty file list.");
  }

  return Status();
}

Status FilterPytorchSafetensorModelFileList(std::vector<std::string>& model_file_list) {
  for (auto it = model_file_list.begin(); it != model_file_list.end();) {
    std::filesystem::path file_path = *it;
    if (file_path.extension() != ".safetensors") {
      it = model_file_list.erase(it);
      continue;
    }
    ++it;
  }

  if (model_file_list.empty()) {
    return Status(RET_INVALID_ARGUMENT, "FilterPytorchSafetensorModelFileList error, empty safetensors file list.");
  }

  return Status();
}

std::string GetFileShardFormatRegex() {
  const std::string regex_special_chars = ".^$|()[]{}*+?\\";
  const std::string format = "{:s}-{:05d}-of-{:05d}.gguf";

  std::string regex_str;
  for (size_t i = 0; i < format.size(); ++i) {
    if (format[i] == '{') {
      size_t end_pos = format.find('}', i);
      if (end_pos != std::string::npos) {
        std::string placeholder = format.substr(i, end_pos - i + 1);
        if (placeholder == "{:s}") {
          regex_str += ".*";
        } else if (placeholder == "{:05d}") {
          regex_str += "\\d{5}";
        } else {
          regex_str += ".*";
        }
        i = end_pos;
      } else {
        if (regex_special_chars.find(format[i]) != std::string::npos) {
          regex_str += '\\';
        }
        regex_str += format[i];
      }
    } else {
      if (regex_special_chars.find(format[i]) != std::string::npos) {
        regex_str += '\\';
      }
      regex_str += format[i];
    }
  }
  regex_str += '$';
  return regex_str;
}

Status FilterGGUFModelFileList(std::vector<std::string>& model_file_list) {
  std::vector<std::string> gguf_files;
  std::vector<std::string> shard_files;

  std::string shard_regex_str = GetFileShardFormatRegex();
  std::regex shard_regex(shard_regex_str, std::regex_constants::ECMAScript | std::regex_constants::optimize);

  for (auto it = model_file_list.begin(); it != model_file_list.end(); ++it) {
    std::filesystem::path file_path = *it;
    if (file_path.extension() == ".gguf") {
      gguf_files.push_back(file_path.string());
      const auto& filename = file_path.filename().string();
      if (std::regex_match(filename, shard_regex)) {
        shard_files.push_back(file_path.string());
      }
    }
  }

  if (gguf_files.empty()) {
    return Status(RET_INVALID_ARGUMENT, "FilterGGUFModelFileList error, empty gguf file list.");
  }

  if (!shard_files.empty()) {
    std::sort(shard_files.begin(), shard_files.end());
    model_file_list = shard_files;
    return Status();
  }

  model_file_list = gguf_files;
  return Status();
}

Status FilterModelFormatFiles(ModelFormat model_format, std::vector<std::string>& model_file_list) {
  switch (model_format) {
    case ModelFormat::PYTORCH_BIN: {
      return FilterPytorchBinModelFileList(model_file_list);
    }
    case ModelFormat::PYTORCH_SAFETENSOR: {
      return FilterPytorchSafetensorModelFileList(model_file_list);
    }
    case ModelFormat::GGUF: {
      return FilterGGUFModelFileList(model_file_list);
    }
    default: {
      return Status(RET_INVALID_ARGUMENT, "Not supported model format.");
    }
  }
}

int GetLayerIdxFromName(const std::string& weight_name) {
  const std::regex layer_regex("model\\.layers.(\\d+)\\..+");
  std::smatch layer_idx_match;

  if (std::regex_match(weight_name, layer_idx_match, layer_regex)) {
    if (layer_idx_match.size() == 2) {
      std::ssub_match base_sub_match = layer_idx_match[1];
      return std::stoi(base_sub_match.str());
    }
  }

  return -1;
}

}  // namespace ksana_llm
