/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cxxabi.h>
#include <string>
#include <functional>
#include <vector>

#include <boost/dll/shared_library.hpp>
#include <boost/dll/library_info.hpp>

#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

class RuntimeDllManager {
 public:
  RuntimeDllManager() = default;
  ~RuntimeDllManager() = default;

  bool Load(const std::string& lib_path);

  bool IsLoaded() const;

  template<typename FuncType>
  std::function<FuncType> GetFunction(const std::string& func_name) const;

  template<typename FuncPtrType>
  FuncPtrType GetRawFunctionPointer(const std::string& func_name) const;

  void Unload();

  std::vector<std::string> GetAllExportedSymbols() const;

  std::vector<std::string> FindSymbolsForFunction(const std::string& substring) const;

  std::string DemangleSymbol(const std::string& mangled_name) const;

  std::string ExtractParametersFromDemangledFuncString(const std::string& demangled_func_ptr) const;

  template<typename FuncPtrType>
  std::string FindMangledFunctionSymbol(const std::string& function_name, FuncPtrType func_ptr_type) const;

 private:
  boost::dll::shared_library lib_;
};

template<typename FuncType>
std::function<FuncType> RuntimeDllManager::GetFunction(const std::string& func_name) const {
  if (!IsLoaded()) {
    return nullptr;
  }

  try {
    return lib_.get<FuncType>(func_name);
  } catch (const boost::system::system_error& e) {
    // 函数不存在或其他错误
    KLLM_LOG_ERROR << "Failed to get function " << func_name << ": " << e.what();
    return nullptr;
  }
}

template<typename FuncPtrType>
FuncPtrType RuntimeDllManager::GetRawFunctionPointer(const std::string& func_name) const {
  if (!IsLoaded()) {
    return nullptr;
  }

  try {
    if constexpr (std::is_same_v<FuncPtrType, void*>) {
      // FuncPtrType 是 void* 类型, 返回空指针
      KLLM_LOG_ERROR << "FuncPtrType is void*, returning nullptr";
      return nullptr;
    } else {
      // FuncPtrType 是函数指针类型，如 mha_varlen_fwd_func_t*
      // 需要转换为函数签名类型，如 mha_varlen_fwd_func_t
      // 使用 std::remove_pointer 去掉指针，得到函数签名类型
      using FuncSignatureType = typename std::remove_pointer<FuncPtrType>::type;
      return lib_.get<FuncSignatureType>(func_name);
    }
  } catch (const boost::system::system_error& e) {
    // 函数不存在或其他错误
    KLLM_LOG_ERROR << "Failed to get raw function pointer " << func_name << ": " << e.what();
    return nullptr;
  }
}

// 根据函数名和函数指针类型查找动态库中匹配的mangled符号名
// 输入: function_name - 目标函数名称
//      func_ptr_type - 函数指针类型，用于验证函数参数匹配
// 输出: 返回匹配的mangled符号名，如果没有找到匹配的符号则返回空字符串
template<typename FuncPtrType>
std::string RuntimeDllManager::FindMangledFunctionSymbol(
  const std::string& function_name,
  FuncPtrType func_ptr_type) const {
  auto symbols = FindSymbolsForFunction(function_name);
  if (symbols.empty()) {
    KLLM_LOG_ERROR << "No symbols found containing: " << function_name;
    return "";
  }

  // 进行验证
  KLLM_LOG_INFO << "Found " << symbols.size()
                << " symbols containing '" << function_name
                << "', attempting to validate with function pointer type";

  // 获取函数指针类型的 demangle 结果
  std::string func_ptr_type_name = typeid(FuncPtrType).name();
  std::string func_ptr_demangled = DemangleSymbol(func_ptr_type_name);
  KLLM_LOG_DEBUG << "Function pointer type demangled: " << func_ptr_demangled;

  // 提取参数信息
  std::string func_ptr_demangled_args = ExtractParametersFromDemangledFuncString(func_ptr_demangled);
  if (func_ptr_demangled_args.empty()) {
    KLLM_LOG_WARNING << "Failed to extract parameters from function pointer type";
    return "";
  }

  // 组装目标符号的 demangle 结果
  std::string target_symbol_demangled = function_name + func_ptr_demangled_args;
  KLLM_LOG_DEBUG << "Target symbol demangled: " << target_symbol_demangled;

  // 遍历所有符号进行验证
  for (const auto& symbol : symbols) {
    std::string symbol_demangled = DemangleSymbol(symbol);
    KLLM_LOG_DEBUG << "Checking symbol: " << symbol << " -> " << symbol_demangled;

    if (symbol_demangled == target_symbol_demangled) {
      KLLM_LOG_INFO << "Found matching symbol for " << function_name << ": " << symbol;
      return symbol;
    }
  }

  // 如果没有找到匹配的符号，警告并返回空字符串
  KLLM_LOG_WARNING << "No exact match found for " << function_name
                   << " with expected signature: " << target_symbol_demangled;

  return "";
}

}  // namespace ksana_llm
