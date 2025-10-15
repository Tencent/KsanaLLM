/* Copyright 2025 Tencent Inc.  All rights reserved.
Ref:
https://github.com/mlc-ai/xgrammar/blob/v0.1.21/include/xgrammar/matcher.h

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"
#include "xgrammar/xgrammar.h"

namespace ksana_llm {

using CompiledGrammar = xgrammar::CompiledGrammar;

// 语法匹配器抽象基类，用于在文本生成过程中约束token选择，确保输出符合语法规则
class GrammarMatcherWrapper {
 public:
  explicit GrammarMatcherWrapper(std::shared_ptr<CompiledGrammar> compiled_grammar)
      : compiled_grammar_(compiled_grammar) {}
  virtual ~GrammarMatcherWrapper() = default;

  GrammarMatcherWrapper(const GrammarMatcherWrapper&) = delete;
  GrammarMatcherWrapper& operator=(const GrammarMatcherWrapper&) = delete;

  GrammarMatcherWrapper(GrammarMatcherWrapper&&) = delete;
  GrammarMatcherWrapper& operator=(GrammarMatcherWrapper&&) = delete;

  // Pure virtual methods that must be implemented by derived classes
  virtual bool FillNextTokenBitmask(void* bitmask_data, int batch_index = 0) = 0;
  virtual bool AcceptToken(int token_id) = 0;
  virtual void Rollback(int token_num) = 0;
  virtual bool IsTerminated() const = 0;
  virtual bool IsInitialized() const = 0;

  // Factory method to create appropriate matcher implementation
  static std::shared_ptr<GrammarMatcherWrapper> Create(std::shared_ptr<CompiledGrammar> compiled_grammar);

 protected:
  std::shared_ptr<CompiledGrammar> compiled_grammar_;
};

}  // namespace ksana_llm
