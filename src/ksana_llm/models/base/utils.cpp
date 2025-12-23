/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/utils.h"

#include <algorithm>
#include <regex>

#include "fmt/core.h"

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

bool CheckWeightNameMatched(const std::string& weight_name, const std::vector<std::string>& name_list,
                            bool full_match) {
  if (full_match) {
    return std::find(name_list.begin(), name_list.end(), weight_name) != name_list.end();
  }

  for (const auto& name_piece : name_list) {
    if (weight_name.find(name_piece) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool CheckWeightNameMatched(const std::string& weight_name, const std::vector<std::string>& name_list,
                            MatchMode match_mode) {
  if (match_mode == MatchMode::FullMatch) {
    return CheckWeightNameMatched(weight_name, name_list, true);
  } else if (match_mode == MatchMode::PartialMatch) {
    return CheckWeightNameMatched(weight_name, name_list, false);
  } else {
    KLLM_THROW("MatchMode only support FullMatch or PartialMatch");
  }
}

bool CheckAllWeightsExist(const std::unordered_map<std::string, Tensor>& host_model_weights,
                          const std::vector<std::string>& name_list) {
  for (const auto& name : name_list) {
    if (host_model_weights.find(name) == host_model_weights.end()) {
      return false;
    }
  }
  return true;
}

bool CheckWeightNameEndMatched(const std::string& weight_name, const std::vector<std::string>& suffix_list) {
  for (const auto& suffix : suffix_list) {
    if (suffix.size() > weight_name.size()) {
      continue;
    }
    if (weight_name.rfind(suffix) == (weight_name.size() - suffix.size())) {
      return true;
    }
  }
  return false;
}

std::string WeightNameReplace(const std::string& weight_name, const std::string& match_name,
                              const std::string& replace_name) {
  // Escape special regex characters in match_name to treat them as literals
  std::string escaped_match_name;
  escaped_match_name.reserve(match_name.size() * 2);

  for (char c : match_name) {
    // Escape regex special characters: . * + ? ^ $ ( ) [ ] { } |
    if (c == '.' || c == '*' || c == '+' || c == '?' || c == '^' || c == '$' || c == '(' || c == ')' || c == '[' ||
        c == ']' || c == '{' || c == '}' || c == '|' || c == '\\') {
      escaped_match_name += '\\';
    }
    escaped_match_name += c;
  }

  // Create regex pattern from escaped match_name
  std::regex pattern(escaped_match_name);

  // Replace all occurrences of match_name with replace_name
  return std::regex_replace(weight_name, pattern, replace_name);
}

std::string CutPrefix(const std::string& weight_name, const std::string& prefix) {
  // Check if weight_name starts with prefix
  if (weight_name.size() >= prefix.size() && weight_name.compare(0, prefix.size(), prefix) == 0) {
    // Return weight_name without the prefix
    return weight_name.substr(prefix.size());
  }
  // Return original weight_name if prefix doesn't match
  return weight_name;
}

}  // namespace ksana_llm