/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

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

  std::vector<std::string> FindSymbolsContaining(const std::string& substring) const;

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

}  // namespace ksana_llm
