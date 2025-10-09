/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/batch_scheduler/structured_generation/structured_generator_interface.h"
#include "ksana_llm/utils/grammar_matcher.h"

namespace ksana_llm {

/*!
 * \\brief Grammar-based structured generator implementation.
 *
 * This class wraps a GrammarMatcherWrapper to provide the StructuredGeneratorInterface
 * interface for grammar constraints.
 */
class GrammarStructuredGenerator : public StructuredGeneratorInterface {
 public:
  explicit GrammarStructuredGenerator(std::shared_ptr<CompiledGrammar> compiled_grammar);

  virtual ~GrammarStructuredGenerator() = default;

  // StructuredGeneratorInterface implementation
  bool AcceptToken(int token_id) override;
  bool FillNextTokenBitmask(void* next_token_bitmask) override;
  bool FindJumpForwardTokens(int& rollback_token_num, std::vector<int>& jump_tokens) override;
  bool IsTerminated() const override;
  bool IsValid() const override;
  StructuredConstraintType GetConstraintType() const override;

 private:
  std::shared_ptr<GrammarMatcherWrapper> matcher_;
};

}  // namespace ksana_llm
