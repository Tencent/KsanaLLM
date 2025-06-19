// Copyright 2024 Tencent Inc.  All rights reserved.

#pragma once

#include <fmt/format.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

// Example:
//   class T {
//    public:
//     DELETE_COPY_AND_MOVE(T);
//
//     ...
//   };
#define DELETE_COPY(class_name)           \
  class_name(const class_name&) = delete; \
  class_name& operator=(const class_name&) = delete
#define DELETE_MOVE(class_name)      \
  class_name(class_name&&) = delete; \
  class_name& operator=(class_name&&) = delete
#define DELETE_COPY_AND_MOVE(class_name) \
  DELETE_COPY(class_name);               \
  DELETE_MOVE(class_name)

// like python "for i in range(begin, end)"
#define FOR_RANGE(type, i, begin, end) for (type i = begin; i < end; ++i)

inline int GetEnvAsPositiveInt(const std::string& env_name, int default_value) {
  const char* env_value = std::getenv(env_name.c_str());
  if (env_value == nullptr) {
    KLLM_LOG_DEBUG << fmt::format("ENV ${} is empty, set to default:{}", env_name, default_value);
    return default_value;
  }

  try {
    int var = std::stoi(env_value);
    if (var < 0) {
      KLLM_LOG_WARNING << fmt::format("ENV ${}:{} < 0, set to 0", env_name, var);
      return 0;
    } else {
      KLLM_LOG_WARNING << fmt::format("ENV ${}:{}, if too big, it will cost a lot of time", env_name, var);
      return var;
    }
  } catch (const std::invalid_argument& e) {
    KLLM_LOG_WARNING << fmt::format("ENV ${} is invalid argument, set to default:{}, error: {}", env_name,
                                    default_value, e.what());
    return default_value;
  } catch (const std::out_of_range& e) {
    KLLM_LOG_WARNING << fmt::format("ENV ${} is out of range, set to default:{}, error: {}", env_name, default_value,
                                    e.what());
    return default_value;
  }
}

/**
 * @brief Calculate the cosine distance between two float vectors
 */
inline float CalculateCosineDist(const std::vector<float>& vec1, const std::vector<float>& vec2) {
  KLLM_CHECK_WITH_INFO(vec1.size() == vec2.size(), "Vectors must have the same dimension");

  float dot_product = 0.0f;
  float norm1 = 0.0f;
  float norm2 = 0.0f;

  for (size_t i = 0; i < vec1.size(); ++i) {
    dot_product += vec1[i] * vec2[i];
    norm1 += vec1[i] * vec1[i];
    norm2 += vec2[i] * vec2[i];
  }

  // Avoid division by zero
  if (norm1 < 1e-6f || norm2 < 1e-6f) {
    return 1.0f;  // Default to orthogonal when one or both vectors are zero
  }

  float cos_sim = dot_product / (std::sqrt(norm1) * std::sqrt(norm2));

  // Clamp to [-1, 1] to handle floating point errors
  cos_sim = std::max(-1.0f, std::min(1.0f, cos_sim));

  // Return cosine distance (1 - cosine similarity)
  return 1.0f - cos_sim;
}

}  // namespace ksana_llm
