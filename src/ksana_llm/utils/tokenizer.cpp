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

Status Tokenizer::GetVocabInfo(std::vector<std::string>& vocab, int& vocab_size, std::vector<int>& stop_token_ids) {
  // Ref: https://github.com/mlc-ai/xgrammar/blob/v0.1.21/python/xgrammar/tokenizer_info.py#L146
  try {
    pybind11::gil_scoped_acquire acquire;
    py::dict vocab_dict = tokenizer_.attr("get_vocab")();

    // Some tokenizer don't have token id 0 or 1 or 2. So the max_id could be larger than the
    // number of tokens. This follows xgrammar's from_huggingface implementation.
    int max_id = -1;
    for (auto item : vocab_dict) {
      int token_id = item.second.cast<int>();
      max_id = std::max(max_id, token_id);
    }

    // Use the input vocab_size (from model_config) or tokenizer_vocab_size, whichever is larger
    // This follows xgrammar's logic where vocab_size can be larger than tokenizer's vocabulary size
    int tokenizer_vocab_size = std::max(static_cast<int>(vocab_dict.size()), max_id + 1);
    int final_vocab_size = std::max(vocab_size, tokenizer_vocab_size);

    vocab.resize(final_vocab_size);

    // maintain tokenizer's indexing
    for (auto item : vocab_dict) {
      std::string token = item.first.cast<std::string>();
      int token_id = item.second.cast<int>();
      if (token_id < final_vocab_size) {
        vocab[token_id] = token;
      }
    }

    // Get stop token ids
    stop_token_ids.clear();
    if (py::hasattr(tokenizer_, "eos_token_id")) {
      py::object eos_token_id = tokenizer_.attr("eos_token_id");
      if (!eos_token_id.is_none()) {
        stop_token_ids.push_back(eos_token_id.cast<int>());
      }
    }

    // Update vocab_size to the final calculated value
    vocab_size = final_vocab_size;

    KLLM_LOG_INFO << "Extracted tokenizer info: input_vocab_size=" << vocab_size
                  << ", tokenizer_vocab_size=" << tokenizer_vocab_size << ", final_vocab_size=" << final_vocab_size
                  << ", vocab_dict_size=" << vocab_dict.size() << ", max_token_id=" << max_id
                  << ", stop_tokens=" << stop_token_ids.size();

    return Status();
  } catch (const std::exception& e) {
    KLLM_LOG_WARNING << "Failed to extract tokenizer info: " << e.what();
    return Status(RET_INVALID_ARGUMENT, "Failed to extract tokenizer information");
  }
}

}  // namespace ksana_llm
