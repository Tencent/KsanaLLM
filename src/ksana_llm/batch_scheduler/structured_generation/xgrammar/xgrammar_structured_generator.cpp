/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "xgrammar_structured_generator.h"

#include <algorithm>
#include <stdexcept>

namespace ksana_llm {

GrammarStructuredGenerator::GrammarStructuredGenerator(std::shared_ptr<CompiledGrammar> compiled_grammar) {
  if (!compiled_grammar) {
    throw std::invalid_argument("Compiled grammar cannot be null");
  }
  matcher_ = GrammarMatcherWrapper::Create(compiled_grammar);
}

bool GrammarStructuredGenerator::AcceptToken(int token_id) {
  if (!matcher_) {
    return false;
  }

  return matcher_->AcceptToken(token_id);
}

bool GrammarStructuredGenerator::FillNextTokenBitmask(void* next_token_bitmask) {
  if (!matcher_) {
    return false;
  }
  return matcher_->FillNextTokenBitmask(next_token_bitmask);
}

bool GrammarStructuredGenerator::FindJumpForwardTokens(int& rollback_token_num, std::vector<int>& jump_tokens) {
  // For grammar constraints, jump-forward tokens are not directly supported
  rollback_token_num = 0;
  jump_tokens.clear();
  return false;
}

bool GrammarStructuredGenerator::IsTerminated() const {
  if (!matcher_) {
    return true;
  }
  return matcher_->IsTerminated();
}

bool GrammarStructuredGenerator::IsValid() const {
  return matcher_ && !matcher_->IsTerminated() && matcher_->IsInitialized();
}

StructuredConstraintType GrammarStructuredGenerator::GetConstraintType() const {
  return StructuredConstraintType::JSON;
}

}  // namespace ksana_llm
