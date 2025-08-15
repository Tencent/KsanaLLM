/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"
#include "xgrammar/xgrammar.h"

namespace ksana_llm {

class GrammarMatcherWrapper;
using CompiledGrammar = xgrammar::CompiledGrammar;

// 语法后端抽象基类，用于编译语法规则和创建语法匹配器，支持结构化文本生成
class GrammarBackend {
 public:
  GrammarBackend() = default;
  virtual ~GrammarBackend() = default;

  // Pure virtual methods that must be implemented by derived classes
  virtual std::shared_ptr<CompiledGrammar> CompileJSONSchema(const std::string& schema) = 0;
  virtual std::shared_ptr<GrammarMatcherWrapper> CreateMatcher(std::shared_ptr<CompiledGrammar> grammar) = 0;
  virtual const xgrammar::TokenizerInfo& GetTokenizerInfo() const = 0;
  virtual bool IsInitialized() const = 0;

  // Factory method to create appropriate backend implementation
  // Returns nullptr if creation fails.
  static std::unique_ptr<GrammarBackend> Create(const std::vector<std::string>& vocab,
                                                int vocab_size,
                                                const std::vector<int>& stop_token_ids);

 protected:
  mutable std::mutex mutex_;
};

}  // namespace ksana_llm
