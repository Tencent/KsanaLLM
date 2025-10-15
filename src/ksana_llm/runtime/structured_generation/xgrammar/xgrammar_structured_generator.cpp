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

void GrammarStructuredGenerator::Rollback(int rollback_token_num) {
  if (rollback_token_num < 0) {
    throw std::invalid_argument("Rollback token num must be greater than or equal to 0");
  }
  if (!matcher_) {
    return;
  }
  matcher_->Rollback(rollback_token_num);
}

bool GrammarStructuredGenerator::FindJumpForwardTokens(std::vector<int>& jump_tokens) {
  // For grammar constraints, jump-forward tokens are not directly supported
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
