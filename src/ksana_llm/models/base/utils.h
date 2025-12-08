/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace ksana_llm {

// Forward declaration
class Tensor;

// Weight name matching mode
enum MatchMode {
  FullMatch,     // Exact match of the entire weight name
  PartialMatch,  // Partial match (substring) of the weight name
};

// Check if weight name matches any name in the list
bool CheckWeightNameMatched(const std::string& weight_name, const std::vector<std::string>& name_list, bool full_match);

// Check if weight name matches any name in the list with match mode
bool CheckWeightNameMatched(const std::string& weight_name, const std::vector<std::string>& name_list,
                            MatchMode match_mode);

// Check if all weights exist in the host model weights
bool CheckAllWeightsExist(const std::unordered_map<std::string, Tensor>& host_model_weights,
                          const std::vector<std::string>& name_list);

// Check if weight name ends with any suffix in the list
bool CheckWeightNameEndMatched(const std::string& weight_name, const std::vector<std::string>& suffix_list);

// Replace match_name in weight_name with replace_name using regex
std::string WeightNameReplace(const std::string& weight_name, const std::string& match_name,
                              const std::string& replace_name);

}  // namespace ksana_llm
