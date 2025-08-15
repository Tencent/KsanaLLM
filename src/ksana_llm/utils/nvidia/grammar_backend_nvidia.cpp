/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/nvidia/grammar_backend_nvidia.h"
#include "ksana_llm/utils/grammar_matcher.h"

namespace ksana_llm {

// Grammar compiler configuration constants
constexpr int kDefaultMaxThreads = 8;
constexpr bool kDefaultCacheEnabled = true;
constexpr int kDefaultMaxMemoryBytes = -1;  // unlimited

GrammarBackendNvidia::GrammarBackendNvidia(const std::vector<std::string>& vocab, int vocab_size,
                                           const std::vector<int>& stop_token_ids) {
  // Convert stop_token_ids to int32_t
  std::vector<int32_t> stop_tokens_int32(stop_token_ids.begin(), stop_token_ids.end());

  // Create TokenizerInfo
  tokenizer_info_ = std::make_unique<xgrammar::TokenizerInfo>(vocab,                     // encoded_vocab
                                                              xgrammar::VocabType::RAW,  // vocab_type
                                                              vocab_size,                // vocab_size
                                                              stop_tokens_int32,         // stop_token_ids
                                                              false);                    // add_prefix_space

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

}  // namespace ksana_llm