/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <pybind11/embed.h>
#include "ksana_llm/utils/nvidia/grammar_backend_nvidia.h"
#include "ksana_llm/utils/grammar_matcher.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tokenizer.h"

namespace ksana_llm {

// Grammar compiler configuration constants
constexpr int kDefaultMaxThreads = 8;
constexpr bool kDefaultCacheEnabled = true;
constexpr int kDefaultMaxMemoryBytes = -1;  // unlimited

GrammarBackendNvidia::GrammarBackendNvidia(const std::vector<std::string>& vocab, int vocab_size,
                                           const std::vector<int>& stop_token_ids) {
  // Detect tokenizer type
  int vocab_type = 0;  // Default: RAW
  bool add_prefix_space = false;
  DetectTokenizerType(vocab_type, add_prefix_space);

  // Convert stop_token_ids to int32_t
  std::vector<int32_t> stop_tokens_int32(stop_token_ids.begin(), stop_token_ids.end());

  xgrammar::VocabType xgrammar_vocab_type = static_cast<xgrammar::VocabType>(vocab_type);

  // Create TokenizerInfo with detected parameters
  tokenizer_info_ = std::make_unique<xgrammar::TokenizerInfo>(vocab,                  // encoded_vocab
                                                              xgrammar_vocab_type,    // vocab_type
                                                              vocab_size,             // vocab_size
                                                              stop_tokens_int32,      // stop_token_ids
                                                              add_prefix_space);      // add_prefix_space

  // Create GrammarCompiler
  compiler_ = std::make_unique<xgrammar::GrammarCompiler>(*tokenizer_info_,         // tokenizer_info
                                                          kDefaultMaxThreads,       // max_threads
                                                          kDefaultCacheEnabled,     // cache_enabled
                                                          kDefaultMaxMemoryBytes);  // max_memory_bytes (unlimited)

  initialized_ = true;
}

GrammarBackendNvidia::~GrammarBackendNvidia() {}

std::shared_ptr<CompiledGrammar> GrammarBackendNvidia::CompileJSONSchema(const std::string& schema) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto compiled_grammar = std::make_shared<CompiledGrammar>(compiler_->CompileJSONSchema(schema,  // schema
                                                                                         true,    // any_whitespace
                                                                                         std::nullopt,  // indent
                                                                                         std::nullopt,  // separators
                                                                                         true));        // strict_mode

  KLLM_LOG_DEBUG << "JSON schema compilation completed successfully";
  KLLM_LOG_DEBUG << "Compiled grammar memory usage: " << compiled_grammar->MemorySizeBytes()
                 << " bytes (strict_mode=true, any_whitespace=true)";
  return compiled_grammar;
}

std::shared_ptr<GrammarMatcherWrapper> GrammarBackendNvidia::CreateMatcher(std::shared_ptr<CompiledGrammar> grammar) {
  return GrammarMatcherWrapper::Create(grammar);
}

const xgrammar::TokenizerInfo& GrammarBackendNvidia::GetTokenizerInfo() const {
  return *tokenizer_info_;
}

void GrammarBackendNvidia::DetectTokenizerType(int& vocab_type, bool& add_prefix_space) {
  // Set default values
  vocab_type = 0;  // RAW
  add_prefix_space = false;

  // Check if Python interpreter is initialized
  if (!Py_IsInitialized()) {
    KLLM_LOG_WARNING << "Python interpreter not initialized, using default tokenizer type: RAW";
    return;
  }

  try {
    pybind11::gil_scoped_acquire acquire;

    // Get tokenizer object from Tokenizer singleton
    auto tokenizer = Singleton<Tokenizer>::GetInstance();
    if (!tokenizer) return;
    pybind11::object tokenizer_obj = tokenizer->tokenizer_;

    // Check if it's PreTrainedTokenizerFast
    pybind11::module transformers = pybind11::module::import("transformers");
    pybind11::object fast_tokenizer_class = transformers.attr("PreTrainedTokenizerFast");

    if (!pybind11::isinstance(tokenizer_obj, fast_tokenizer_class)) {
      KLLM_LOG_WARNING << "Not a PreTrainedTokenizerFast, using default tokenizer type: RAW";
      return;
    }

    // Get backend_tokenizer.to_str()
    pybind11::object backend_tokenizer = tokenizer_obj.attr("backend_tokenizer");
    std::string backend_str = backend_tokenizer.attr("to_str")().cast<std::string>();

    KLLM_LOG_DEBUG << "Backend tokenizer string length: " << backend_str.length();

    // Use XGrammar API to detect metadata
    std::string metadata = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_str);

    KLLM_LOG_INFO << "XGrammar detected metadata: " << metadata;

    // Parse metadata: {"vocab_type": 2, "add_prefix_space": false}
    size_t vocab_type_pos = metadata.find("\"vocab_type\":");
    size_t prefix_space_pos = metadata.find("\"add_prefix_space\":");

    if (vocab_type_pos != std::string::npos) {
      size_t start = metadata.find_first_of("0123456789", vocab_type_pos);
      if (start != std::string::npos) {
        vocab_type = std::stoi(metadata.substr(start, 1));
      }
    }

    if (prefix_space_pos != std::string::npos) {
      add_prefix_space = (metadata.find("true", prefix_space_pos) != std::string::npos);
    }

    KLLM_LOG_INFO << "Tokenizer type detected: vocab_type=" << vocab_type
                  << ", add_prefix_space=" << (add_prefix_space ? "true" : "false");
  } catch (const std::exception& e) {
    KLLM_LOG_WARNING << "Failed to detect tokenizer type: " << e.what()
                     << ", using default: RAW";
    vocab_type = 0;
    add_prefix_space = false;
  }
}

}  // namespace ksana_llm