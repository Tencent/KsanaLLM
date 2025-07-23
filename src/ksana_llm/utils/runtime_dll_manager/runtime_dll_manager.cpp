/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/runtime_dll_manager/runtime_dll_manager.h"

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

std::vector<std::string> RuntimeDllManager::FindSymbolsContaining(const std::string& substring) const {
  std::vector<std::string> matched_symbols;
  auto all_symbols = GetAllExportedSymbols();

  for (const auto& symbol : all_symbols) {
    if (symbol.find(substring) != std::string::npos) {
      matched_symbols.push_back(symbol);
    }
  }

  return matched_symbols;
}

}  // namespace ksana_llm
