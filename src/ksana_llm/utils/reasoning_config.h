/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include "ksana_llm/utils/config/schedule_config_parser.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {
class ReasoningConfigManager {
 public:
  /**
   * @brief Constructor that automatically applies reasoning configuration
   *
   * @param think_end_token_id Token ID for the end of thinking block (e.g., "</think>")
   *                           -1 means reasoning mode is disabled
   * @param env The environment instance to configure.
   */
  explicit ReasoningConfigManager(int think_end_token_id,
                                   std::shared_ptr<Environment> env = nullptr);

 private:
  static bool IsValidThinkEndTokenId(int think_end_token_id);
};

}  // namespace ksana_llm