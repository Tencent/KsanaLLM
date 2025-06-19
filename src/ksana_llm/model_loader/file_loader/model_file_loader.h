/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>

#include "ksana_llm/models/base/model_format.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

#include "ksana_llm/model_loader/file_loader/gguf_file_loader.h"
#include "ksana_llm/model_loader/file_loader/pytorch_bin_file_loader.h"
#include "ksana_llm/model_loader/file_loader/pytorch_safetensor_file_loader.h"

namespace ksana_llm {

// A file loader that support both pytorch_bin & pytorch_safetensor & gguf file format.
class FileLoader {
 public:
  FileLoader() {}
  explicit FileLoader(const std::string& filename);

  // Load weight names from file, but not load it.
  Status LoadWeightNames(ModelFormat model_format, std::vector<std::string>& weight_names);

  // Load weights in weight_names.
  Status LoadModelWeights(ModelFormat model_format, const std::vector<std::string>& weight_names,
                          std::unordered_map<std::string, Tensor>& result);

 private:
  Status InitFileLoader(ModelFormat model_format);

 private:
  std::string filename_;

  std::shared_ptr<BaseFileLoader> file_loader_impl_ = nullptr;

  std::shared_ptr<PytorchBinFileLoader> pytorch_bin_file_loader_ = nullptr;
  std::shared_ptr<PytorchSafetensorFileLoader> pytorch_safetensor_file_loader_ = nullptr;
  std::shared_ptr<GGUFFileLoader> gguf_file_loader_ = nullptr;
};

}  // namespace ksana_llm
