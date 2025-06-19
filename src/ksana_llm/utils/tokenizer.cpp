/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include "ksana_llm/utils/tokenizer.h"

#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace py = pybind11;

namespace ksana_llm {

Status Tokenizer::InitTokenizer(const std::string& tokenizer_path) {
  try {
    pybind11::gil_scoped_acquire acquire;
    py::module transformers = py::module::import("transformers");
    py::object auto_tokenizer = transformers.attr("AutoTokenizer");
    tokenizer_ = auto_tokenizer.attr("from_pretrained")(tokenizer_path, py::arg("trust_remote_code") = true);
  } catch (const py::error_already_set& e) {
    KLLM_LOG_ERROR << fmt::format("Failed to load tokenizer from tokenizer_path {}: {}", tokenizer_path, e.what());
    PyErr_Clear();
    return Status(RET_INVALID_ARGUMENT, fmt::format("Failed to init the tokenizer from {}.", tokenizer_path));
  }
  return Status();
}

void Tokenizer::DestroyTokenizer() {
  pybind11::gil_scoped_acquire acquire;
  tokenizer_ = py::none();
}

Status Tokenizer::Decode(std::vector<int>& output_tokens, std::string& output, bool skip_special_tokens) {
  pybind11::gil_scoped_acquire acquire;
  py::object tokens = tokenizer_.attr("decode")(output_tokens, py::arg("skip_special_tokens") = skip_special_tokens);
  output = tokens.cast<std::string>();
  return Status();
}

Status Tokenizer::Encode(const std::string& prompt, std::vector<int>& input_tokens, bool add_special_tokens) {
  pybind11::gil_scoped_acquire acquire;
  py::object tokens = tokenizer_.attr("encode")(prompt, py::arg("add_special_tokens") = add_special_tokens);
  input_tokens = tokens.cast<std::vector<int>>();
  return Status();
}
}  // namespace ksana_llm
