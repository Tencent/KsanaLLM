/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/runtime_dll_manager/runtime_dll_manager.h"

#include <cxxabi.h>
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

bool RuntimeDllManager::Load(const std::string& lib_path) {
  try {
    lib_ = boost::dll::shared_library(lib_path);
    return true;
  } catch (const boost::system::system_error& e) {
    KLLM_LOG_ERROR << "Failed to Load library " << lib_path << ": " << e.what();
    return false;
  }
}

bool RuntimeDllManager::IsLoaded() const {
  return lib_.is_loaded();
}

void RuntimeDllManager::Unload() {
  if (IsLoaded()) {
    lib_.unload();
  }
}

std::vector<std::string> RuntimeDllManager::GetAllExportedSymbols() const {
  std::vector<std::string> symbols;

  if (!IsLoaded()) {
    KLLM_LOG_ERROR << "Library not loaded";
    return symbols;
  }

  try {
    boost::dll::library_info lib_info(lib_.location());
    symbols = lib_info.symbols();
  } catch (const boost::system::system_error& e) {
    KLLM_LOG_ERROR << "Failed to get symbols: " << e.what();
  }

  return symbols;
}

// 在动态库的所有导出符号中查找包含指定子字符串的符号
// 输入: substring - 要搜索的子字符串，通常是未经修饰（mangle）的函数名
// 输出: 返回包含该子字符串的所有符号名称列表，如果没有找到则返回空vector
std::vector<std::string> RuntimeDllManager::FindSymbolsForFunction(const std::string& substring) const {
  std::vector<std::string> matched_symbols;
  auto all_symbols = GetAllExportedSymbols();

  for (const auto& symbol : all_symbols) {
    if (symbol.find(substring) != std::string::npos) {
      matched_symbols.push_back(symbol);
    }
  }

  return matched_symbols;
}

// 将C++编译器生成的mangled符号名转换为可读的函数名
// 输入: mangled_name - mangled形式的符号名称，可能来源于：
//       1) typeid(T).name() 获取的类型名称
//       2) 动态库中导出的mangled符号名称
// 输出: 返回demangle后的可读函数名，如果demangle失败则返回原始输入字符串
std::string RuntimeDllManager::DemangleSymbol(const std::string& mangled_name) const {
  int status = 0;
  char* demangled = abi::__cxa_demangle(mangled_name.c_str(), nullptr, nullptr, &status);

  if (status == 0 && demangled) {
    std::string result(demangled);
    free(demangled);
    return result;
  }

  // 如果 demangle 失败，返回原始名称
  return mangled_name;
}

// 从demangle后的函数指针类型字符串中提取参数列表部分
// 输入: demangled_func_ptr - demangle后的函数指针类型字符串，格式如"void (*)(int, float)"
// 输出: 返回参数列表部分字符串，如"(int, float)"，如果解析失败则返回空字符串
std::string RuntimeDllManager::ExtractParametersFromDemangledFuncString(const std::string& demangled_func_ptr) const {
  // 查找 (*) 模式，提取参数部分
  size_t func_ptr_start = demangled_func_ptr.find("(*)");
  if (func_ptr_start == std::string::npos) {
    KLLM_LOG_DEBUG << "No function pointer pattern found in: " << demangled_func_ptr;
    return "";
  }

  // 从 (*) 之后查找参数列表的开始位置
  size_t params_start = demangled_func_ptr.find('(', func_ptr_start + 3);
  if (params_start == std::string::npos) {
    KLLM_LOG_DEBUG << "No parameter list found in: " << demangled_func_ptr;
    return "";
  }

  // 从参数列表开始位置到字符串结尾就是完整的参数列表
  return demangled_func_ptr.substr(params_start);
}

}  // namespace ksana_llm
