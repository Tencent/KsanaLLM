/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "logger.h"

namespace ksana_llm {

std::vector<std::string> g_categories;

// Function definition moved here
void category_log_handler(void* user_data, const loguru::Message& message) {
  const std::string output_file = GetLogFile();

  for (const auto& category : g_categories) {
    if (message.message && std::string_view(message.message).find(category) != std::string_view::npos) {
      std::ofstream out(output_file, std::ios::app);  // Open file in append mode
      if (out.is_open()) {
        out << message.preamble
            << message.message << std::endl;
        out.close();
      } else {
        std::cerr << "Error opening file for writing: " << output_file << std::endl;
      }
    }
  }
}

}  // namespace ksana_llm
