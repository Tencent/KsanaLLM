/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/gguf_file_utils.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

#include "ksana_llm/model_loader/file_loader/base_file_loader.h"

namespace ksana_llm {

// Used to load gguf file loader.
class GGUFFileLoader : public BaseFileLoader {
 public:
  explicit GGUFFileLoader(const std::string& filename);
  virtual ~GGUFFileLoader();

  // Load weight names from file, but not load it.
  virtual Status LoadWeightNames(std::vector<std::string>& weight_names) override;

  // Load weights in weight_names.
  virtual Status LoadModelWeights(const std::vector<std::string>& weight_names,
                                  std::unordered_map<std::string, Tensor>& result) override;

  // Get the meta info from gguf file, used to construct model config.
  Status GetMetaDict(std::unordered_map<std::string, NewGGUFMetaValue>& result);

 private:
  Status LoadGGUFModelMeta();

 private:
  std::string filename_;

  std::ifstream gguf_file_;

  int64_t file_size_ = 0;

  std::shared_ptr<NewGGUFContext> gguf_context_ = nullptr;

  // Whether the file have been loaded.
  bool loaded_ = false;
};

}  // namespace ksana_llm
