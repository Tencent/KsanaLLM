/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <chrono>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace ksana_llm {

template <typename... Args>
inline std::string FormatStr(const std::string& format, Args... args) {
  // This function came from a code snippet in stackoverflow under cc-by-1.0
  //   https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

  // Disable format-security warning in this function.
#if defined(__GNUC__) || defined(__clang__)  // for gcc or clang
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wformat-security"
#endif
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format.c_str(), args...);
#if defined(_MSC_VER)
#  pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif
  return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}

inline std::string GetReplacedName(const std::string& tensor_name, const std::string& pattern,
                                   const std::string& replacement) {
  return std::regex_replace(tensor_name, std::regex(pattern), replacement);
}

// Split string to a vector.
inline std::vector<std::string> Str2Vector(const std::string& str, const std::string& delimiters = " ") {
  std::vector<std::string> results;

  std::string::size_type last_pos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos = str.find_first_of(delimiters, last_pos);
  while (std::string::npos != pos || std::string::npos != last_pos) {
    results.push_back(str.substr(last_pos, pos - last_pos));
    last_pos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, last_pos);
  }

  return results;
}

template <typename T>
inline std::string Vector2Str(std::vector<T> vec) {
  std::stringstream ss;
  ss << "(";
  if (!vec.empty()) {
    for (size_t i = 0; i < vec.size() - 1; ++i) {
      ss << vec[i] << ", ";
    }
    ss << vec.back();
  }
  ss << ")";
  return ss.str();
}

template <typename T>
inline std::string Array2Str(T* arr, size_t size) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < size - 1; ++i) {
    ss << arr[i] << ", ";
  }
  if (size > 0) {
    ss << arr[size - 1];
  }
  ss << ")";
  return ss.str();
}

}  // namespace ksana_llm
