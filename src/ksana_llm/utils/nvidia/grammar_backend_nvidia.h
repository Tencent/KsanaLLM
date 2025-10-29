/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ksana_llm/utils/grammar_backend.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"
#include "xgrammar/xgrammar.h"

namespace ksana_llm {

class GrammarBackendNvidia : public GrammarBackend {
 public:
  GrammarBackendNvidia(const std::vector<std::string>& vocab, int vocab_size,
                       const std::vector<int>& stop_token_ids);
  ~GrammarBackendNvidia() override;

  // Implement pure virtual methods from base class
  std::shared_ptr<CompiledGrammar> CompileJSONSchema(const std::string& schema) override;
  std::shared_ptr<GrammarMatcherWrapper> CreateMatcher(std::shared_ptr<CompiledGrammar> grammar) override;
  const xgrammar::TokenizerInfo& GetTokenizerInfo() const override;
  bool IsInitialized() const override { return initialized_; }

 private:
  // Detect tokenizer type using XGrammar API
  void DetectTokenizerType(int& vocab_type, bool& add_prefix_space);

  std::unique_ptr<xgrammar::TokenizerInfo> tokenizer_info_;
  std::unique_ptr<xgrammar::GrammarCompiler> compiler_;
  bool initialized_ = false;
};

}  // namespace ksana_llm