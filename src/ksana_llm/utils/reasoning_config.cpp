/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/reasoning_config.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

ReasoningConfigManager::ReasoningConfigManager(int think_end_token_id,
                                                std::shared_ptr<Environment> env) {
  // Skip if reasoning mode is disabled
  if (!IsValidThinkEndTokenId(think_end_token_id)) {
    KLLM_LOG_DEBUG << "Reasoning mode is disabled (think_end_token_id: " << think_end_token_id << ")";
    return;
  }

  if (!env) {
    env = Singleton<Environment>::GetInstance();
    if (!env) {
      KLLM_LOG_ERROR << "Failed to get Environment instance";
      throw std::runtime_error("Environment not initialized");
    }
  }

  ksana_llm::ReasoningConfig reasoning_config;
  reasoning_config.think_end_token_id = think_end_token_id;
  env->SetReasoningConfig(reasoning_config);

  KLLM_LOG_INFO << "Reasoning config applied successfully with think_end_token_id: " << think_end_token_id;
}

bool ReasoningConfigManager::IsValidThinkEndTokenId(int think_end_token_id) {
  // -1 is used to indicate reasoning mode is disabled
  return think_end_token_id >= 0;
}

}  // namespace ksana_llm