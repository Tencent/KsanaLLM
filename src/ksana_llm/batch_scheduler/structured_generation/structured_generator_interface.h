/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <vector>

namespace ksana_llm {

/*!
 * \\brief The type of structured generation constraint.
 */
enum class StructuredConstraintType {
  NONE,  // No constraint
  JSON,  // JSON schema constraint
  REGEX  // Regular expression constraint
};

/*!
 * \\brief Configuration for structured generator creation.
 */
struct StructuredGeneratorConfig {
  StructuredConstraintType constraint_type = StructuredConstraintType::NONE;
  std::string constraint_spec;  // JSON schema, regex etc.

  StructuredGeneratorConfig() = default;
  StructuredGeneratorConfig(StructuredConstraintType type, const std::string& spec)
      : constraint_type(type), constraint_spec(spec) {
    if (constraint_spec.empty()) {
      constraint_type = StructuredConstraintType::NONE;
    }
  }

  bool HaveConstraint() const {
    return (constraint_type != StructuredConstraintType::NONE) && !constraint_spec.empty();
  }
};

/*!
 * \\brief Abstract interface for structured generators.
 *
 * auto generator = CreateStructuredGenerator(config);
 *
 * generator->AcceptToken(67);
 *
 * std::vector<int32_t> next_token_bitmask;
 * generator->FillNextTokenBitmask(next_token_bitmask.data());
 *
 * int rollback_token_num = -1;
 * std::vector<int> jump_tokens;
 * generator->FindJumpForwardTokens(rollback_token_num, jump_tokens);
 *
 * \\endcode
 */
class StructuredGeneratorInterface {
 public:
  virtual ~StructuredGeneratorInterface() = default;

  /*!
   * \\brief Accept one token and update the generator state.
   * \\param token_id The id of the token to accept.
   * \\return Whether the token is accepted by the constraint.
   */
  virtual bool AcceptToken(int token_id) = 0;

  /*!
   * \\brief Get the set of tokens that are acceptable for the next step.
   * \\param next_token_bitmask The bitmask to store the result.
   * \\return Whether the constraint needs to be applied (not all tokens are valid).
   */
  virtual bool FillNextTokenBitmask(void* next_token_bitmask) = 0;

  /*!
   * \\brief Try to find the jump-forward tokens for jump-forward decoding.
   * \\param rollback_token_num The number of tokens to rollback.
   * \\param jump_tokens The vector to store the jump-forward tokens.
   * \\return Whether jump-forward tokens are found, if true, internal states are changed according to jump operations
   * \\note The invoker must handle rollback and jump operations to keep the states consistent.
   */
  virtual bool FindJumpForwardTokens(int& rollback_token_num, std::vector<int>& jump_tokens) = 0;

  /*!
   * \\brief Check if the generator has reached a termination state.
   * \\return Whether the generator is terminated.
   */
  virtual bool IsTerminated() const = 0;

  /*!
   * \\brief Check if the current generation is valid according to the constraint.
   * \\return Whether the current generation satisfies the constraint.
   */
  virtual bool IsValid() const = 0;

  /*!
   * \\brief Get the type of constraint being used.
   * \\return The constraint type.
   */
  virtual StructuredConstraintType GetConstraintType() const = 0;
};

}  // namespace ksana_llm
