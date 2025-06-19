/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <memory>
#include "ksana_llm/model_loader/file_loader/model_file_loader.h"

namespace ksana_llm {

FileLoader::FileLoader(const std::string& filename) : filename_(filename) {}

Status FileLoader::InitFileLoader(ModelFormat model_format) {
  if (file_loader_impl_) {
    return Status();
  }

  switch (model_format) {
    case ModelFormat::PYTORCH_BIN: {
      file_loader_impl_ = std::make_shared<PytorchBinFileLoader>(filename_);
      break;
    }
    case ModelFormat::PYTORCH_SAFETENSOR: {
      file_loader_impl_ = std::make_shared<PytorchSafetensorFileLoader>(filename_);
      break;
    }
    case ModelFormat::GGUF: {
      file_loader_impl_ = std::make_shared<GGUFFileLoader>(filename_);
      break;
    }
    default: {
      return Status(RET_INVALID_ARGUMENT, "Not supported model format.");
    }
  }
  return Status();
}

Status FileLoader::LoadWeightNames(ModelFormat model_format, std::vector<std::string>& weight_names) {
  Status status = InitFileLoader(model_format);
  if (!status.OK()) {
    return status;
  }
  return file_loader_impl_->LoadWeightNames(weight_names);
}

Status FileLoader::LoadModelWeights(ModelFormat model_format, const std::vector<std::string>& weight_names,
                                    std::unordered_map<std::string, Tensor>& result) {
  Status status = InitFileLoader(model_format);
  if (!status.OK()) {
    return status;
  }
  return file_loader_impl_->LoadModelWeights(weight_names, result);
}

}  // namespace ksana_llm
