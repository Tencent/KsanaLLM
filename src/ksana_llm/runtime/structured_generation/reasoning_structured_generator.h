/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "ksana_llm/runtime/structured_generation/structured_generator_interface.h"

namespace ksana_llm {

/*!
 * \brief Wrapper for structured generators that handles reasoning content.
 *
 * This wrapper delays the activation of the underlying constraint generator
 * until the reasoning phase is complete (detected by think_end_token_id).
 * During the reasoning phase, all tokens are accepted without constraint.
 */
class ReasoningStructuredGenerator : public StructuredGeneratorInterface {
 public:
  /*!
   * \brief Constructor
   * \param inner_generator The underlying structured generator to wrap
   * \param think_end_token_id The token ID that marks the end of reasoning
   * \param start_in_reasoning Whether to start in reasoning mode
   *                          Set to true for models like DeepSeek-R1/QwQ that start with thinking
   *                          Set to false if reasoning needs to be explicitly triggered
   */
  ReasoningStructuredGenerator(std::shared_ptr<StructuredGeneratorInterface> inner_generator,
                               int think_end_token_id,
                               bool start_in_reasoning);

  virtual ~ReasoningStructuredGenerator() = default;

  // StructuredGeneratorInterface implementation
  bool AcceptToken(int token_id) override;
  bool FillNextTokenBitmask(void* next_token_bitmask) override;
  void Rollback(int rollback_token_num) override;
  bool FindJumpForwardTokens(std::vector<int>& jump_tokens) override;
  bool IsTerminated() const override;
  bool IsValid() const override;
  StructuredConstraintType GetConstraintType() const override;

  /*!
   * \brief Check if currently in reasoning phase
   * \return true if in reasoning phase, false otherwise
   */
  bool IsInReasoningPhase() const { return is_in_reasoning_; }

 private:
  std::shared_ptr<StructuredGeneratorInterface> inner_generator_;
  int think_end_token_id_;
  bool is_in_reasoning_;
};

}  // namespace ksana_llm