/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "ksana_llm/utils/status.h"

namespace py = pybind11;

namespace ksana_llm {

// Wraps the tokenizer for internal various usage
class Tokenizer {
 public:
  // Initialize the tokenizer from the given tokenizer_path.
  Status InitTokenizer(const std::string& tokenizer_path);

  // Destroy the tokenizer.
  void DestroyTokenizer();

  // Decode the given input token ids into string
  Status Decode(std::vector<int>& input_tokens, std::string& output, bool skip_special_tokens = true);

  // Encode the given prompt into token ids
  Status Encode(const std::string& prompt, std::vector<int>& input_tokens, bool add_special_tokens = true);

  // Extract vocabulary information for grammar initialization
  Status GetVocabInfo(std::vector<std::string>& vocab, int& vocab_size, std::vector<int>& stop_token_ids);

 public:
  py::object tokenizer_;
};

}  // namespace ksana_llm
