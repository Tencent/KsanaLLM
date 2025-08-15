/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/utils/grammar_matcher.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"
#include "xgrammar/xgrammar.h"

namespace ksana_llm {

class GrammarMatcherWrapperNvidia : public GrammarMatcherWrapper {
 public:
  explicit GrammarMatcherWrapperNvidia(std::shared_ptr<CompiledGrammar> compiled_grammar);
  ~GrammarMatcherWrapperNvidia() override;

  // Implement pure virtual methods from base class
  bool FillNextTokenBitmask(void* bitmask_data, int batch_index = 0) override;
  bool AcceptToken(int token_id) override;
  bool IsTerminated() const override;
  bool IsInitialized() const override { return matcher_ != nullptr; }

 private:
  std::unique_ptr<xgrammar::GrammarMatcher> matcher_;
  int bitmask_size_;
};

}  // namespace ksana_llm