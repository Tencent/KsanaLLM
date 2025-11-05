/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/structured_generation/reasoning_structured_generator.h"

#include <stdexcept>

namespace ksana_llm {

ReasoningStructuredGenerator::ReasoningStructuredGenerator(
    std::shared_ptr<StructuredGeneratorInterface> inner_generator,
    int think_end_token_id,
    bool start_in_reasoning)
    : inner_generator_(inner_generator),
      think_end_token_id_(think_end_token_id),
      is_in_reasoning_(start_in_reasoning) {
  if (!inner_generator_) {
    throw std::invalid_argument("Inner generator cannot be null");
  }
}

bool ReasoningStructuredGenerator::AcceptToken(int token_id) {
  // 检测到思考结束标记，退出推理阶段
  if (token_id == think_end_token_id_) {
    is_in_reasoning_ = false;
    // 思考结束标记本身不传递给内部生成器
    return true;
  }

  // 如果仍在推理阶段，接受所有token
  if (is_in_reasoning_) {
    return true;
  }

  // 推理阶段结束后，将token传递给内部约束生成器
  return inner_generator_->AcceptToken(token_id);
}

bool ReasoningStructuredGenerator::FillNextTokenBitmask(void* next_token_bitmask) {
  // 如果仍在推理阶段，不应用任何约束（返回false表示不需要约束）
  if (is_in_reasoning_) {
    return false;
  }

  // 推理阶段结束后，使用内部生成器的约束
  return inner_generator_->FillNextTokenBitmask(next_token_bitmask);
}

void ReasoningStructuredGenerator::Rollback(int rollback_token_num) {
  if (rollback_token_num < 0) {
    throw std::invalid_argument("Rollback token num must be greater than or equal to 0");
  }

  // 如果不在推理阶段，将回滚传递给内部生成器
  if (!is_in_reasoning_) {
    inner_generator_->Rollback(rollback_token_num);
  }
  // 注意：在推理阶段的回滚不需要特殊处理，因为没有状态需要维护
}

bool ReasoningStructuredGenerator::FindJumpForwardTokens(std::vector<int>& jump_tokens) {
  // 如果仍在推理阶段，不支持跳跃前进
  if (is_in_reasoning_) {
    jump_tokens.clear();
    return false;
  }

  // 推理阶段结束后，使用内部生成器的跳跃前进功能
  return inner_generator_->FindJumpForwardTokens(jump_tokens);
}

bool ReasoningStructuredGenerator::IsTerminated() const {
  // 如果仍在推理阶段，未终止
  if (is_in_reasoning_) {
    return false;
  }

  // 推理阶段结束后，检查内部生成器的终止状态
  return inner_generator_->IsTerminated();
}

bool ReasoningStructuredGenerator::IsValid() const {
  return inner_generator_ && inner_generator_->IsValid();
}

StructuredConstraintType ReasoningStructuredGenerator::GetConstraintType() const {
  return inner_generator_->GetConstraintType();
}

}  // namespace ksana_llm