/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/nvidia/grammar_matcher_nvidia.h"
#include <dlpack/dlpack.h>

namespace ksana_llm {

GrammarMatcherWrapperNvidia::GrammarMatcherWrapperNvidia(std::shared_ptr<CompiledGrammar> compiled_grammar)
    : GrammarMatcherWrapper(compiled_grammar) {
  matcher_ = std::make_unique<xgrammar::GrammarMatcher>(*compiled_grammar_,  // compiled_grammar
                                                        std::nullopt,        // override_stop_tokens
                                                        false,               // terminate_without_stop_token
                                                        0);                  // max_rollback_tokens

  int vocab_size_ = compiled_grammar_->GetTokenizerInfo().GetVocabSize();
  bitmask_size_ = xgrammar::GetBitmaskSize(vocab_size_);
  KLLM_LOG_DEBUG << "GrammarMatcher created, vocab_size: " << vocab_size_ << ", bitmask_size: " << bitmask_size_;
}

GrammarMatcherWrapperNvidia::~GrammarMatcherWrapperNvidia() {}

bool GrammarMatcherWrapperNvidia::FillNextTokenBitmask(void* bitmask_data, int batch_index) {
  if (matcher_ == nullptr) {
    KLLM_LOG_ERROR << "Grammar matcher is null, cannot fill token bitmask";
    return false;
  }

  // Create a DLTensor for the bitmask
  DLTensor bitmask_tensor;
  bitmask_tensor.data = bitmask_data;
  bitmask_tensor.device = {kDLCPU, 0};  // CPU
  bitmask_tensor.ndim = 1;
  bitmask_tensor.dtype = xgrammar::GetBitmaskDLType();  // int32

  // Shape: (bitmask_size,)
  int64_t shape[1] = {static_cast<int64_t>(bitmask_size_)};
  bitmask_tensor.shape = shape;
  bitmask_tensor.strides = nullptr;
  bitmask_tensor.byte_offset = 0;

  bool needs_mask = matcher_->FillNextTokenBitmask(&bitmask_tensor,  // next_token_bitmask
                                                   batch_index,      // index
                                                   false);           // debug_print
  return needs_mask;
}

bool GrammarMatcherWrapperNvidia::AcceptToken(int token_id) {
  if (matcher_ == nullptr) {
    KLLM_LOG_ERROR << "Grammar matcher is null, cannot accept token " << token_id;
    return false;
  }

  bool accepted = matcher_->AcceptToken(static_cast<int32_t>(token_id),  // token_id
                                        false);                          // debug_print
  KLLM_LOG_DEBUG << "AcceptToken(" << token_id << ") = " << accepted;
  return accepted;
}

bool GrammarMatcherWrapperNvidia::IsTerminated() const {
  if (matcher_ == nullptr) {
    KLLM_LOG_ERROR << "Grammar matcher is null, assuming terminated";
    return true;
  }
  return matcher_->IsTerminated();
}

}  // namespace ksana_llm