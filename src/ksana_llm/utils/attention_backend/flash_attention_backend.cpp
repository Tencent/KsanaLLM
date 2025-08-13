/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <filesystem>
#include <functional>
#include <sstream>
#include <vector>

#include "ksana_llm/utils/attention_backend/flash_attention_backend.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/utils/nvidia/cuda_utils.h"
#endif

#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {
mha_varlen_fwd_vllm_flash_attn_v26_ptr FlashAttentionBackend::mha_varlen_fwd_vllm_flash_attn_v26_ = nullptr;
mha_fwd_kvcache_vllm_flash_attn_v26_ptr FlashAttentionBackend::mha_fwd_kvcache_vllm_flash_attn_v26_ = nullptr;

mha_varlen_fwd_flash_attn_v25_ptr FlashAttentionBackend::mha_varlen_fwd_flash_attn_v25_ = nullptr;
mha_fwd_kvcache_flash_attn_v25_ptr FlashAttentionBackend::mha_fwd_kvcache_flash_attn_v25_ = nullptr;

mha_varlen_fwd_flash_attn_v26_ptr FlashAttentionBackend::mha_varlen_fwd_flash_attn_v26_ = nullptr;
mha_fwd_kvcache_flash_attn_v26_ptr FlashAttentionBackend::mha_fwd_kvcache_flash_attn_v26_ = nullptr;

// FA3 function pointer
mha_fwd_fa3_ptr FlashAttentionBackend::mha_fwd_fa3_ = nullptr;

bool FlashAttentionBackend::Initialize() {
  // 1. 平台检测
  if (!IsCudaPlatform()) {
    KLLM_LOG_WARNING << "FlashAttentionBackend only support cuda platform, will not be initialized";
    return false;
  }

  // 2. 获取并检查 CUDA compute capability
  int compute_capability = GetCudaComputeCapability();
  if (compute_capability < 80) {  // SM 8.0 及以上支持 FlashAttention 2
    KLLM_LOG_WARNING << "Compute capability " << compute_capability << " not support FlashAttention 2";
    return false;
  }

  // 3. 确定所有可用的库信息
  std::vector<LibraryInfo> available_libraries = DetermineAllLibraries(compute_capability);
  if (available_libraries.empty()) {
    KLLM_LOG_WARNING << "No compatible FlashAttention library found";
    return false;
  }

  // 4. 尝试加载所有可用的库
  loaded_libraries_.clear();
  for (auto& lib_info : available_libraries) {
    // 为每个库创建独立的DLL管理器
    lib_info.dll_manager = std::make_shared<RuntimeDllManager>();

    if (!lib_info.dll_manager->Load(lib_info.path)) {
      KLLM_LOG_WARNING << "Failed to load FlashAttention library from " << lib_info.path << ", skipping";
      continue;
    }

    // 5. 为该库加载函数指针
    if (!LoadFunctions(lib_info)) {
      KLLM_LOG_WARNING << "Failed to load functions from FlashAttention library " << lib_info.name << ", skipping";
      continue;
    }

    // 成功加载的库添加到列表中
    loaded_libraries_.push_back(lib_info);
    KLLM_LOG_INFO << "Successfully loaded FlashAttention library: " << lib_info.name
                  << " (version: " << lib_info.version << ", path: " << lib_info.path << ")";
  }

  if (loaded_libraries_.empty()) {
    KLLM_LOG_ERROR << "Failed to load any FlashAttention library";
    return false;
  }

  // 6. 设置初始化状态
  initialized_ = true;
  KLLM_LOG_INFO << "FlashAttentionBackend initialized successfully with " << loaded_libraries_.size()
                << " libraries loaded";
  return true;
}

// 平台检测：是否是 CUDA 平台
bool FlashAttentionBackend::IsCudaPlatform() {
#ifdef ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

// 获取 CUDA compute capability（SM）（使用项目统一接口和错误处理）
int FlashAttentionBackend::GetCudaComputeCapability() {
#ifdef ENABLE_CUDA
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    KLLM_THROW("There is no cuda GPU available on this machine.");
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int sm = prop.major * 10 + prop.minor;
  return sm;
#else
  return -1;
#endif
}

// 辅助函数：检查版本是否大于等于最小版本要求
bool FlashAttentionBackend::IsVersionGreaterOrEqual(const std::string& version, const std::string& min_version) {
  if (version.empty() || min_version.empty()) {
    KLLM_LOG_DEBUG << "Invalid version strings: version=" << version << ", min_version=" << min_version;
    return false;
  }

  // 简单的版本比较，假设版本格式为 "x.y.z"
  std::vector<int> version_parts, min_version_parts;

  // 解析版本号
  auto parse_version = [](const std::string& v) -> std::vector<int> {
    std::vector<int> parts;
    std::stringstream ss(v);
    std::string part;

    while (std::getline(ss, part, '.')) {
      // 检查是否包含非数字字符
      auto non_digit = std::find_if(part.begin(), part.end(), [](char c) { return !std::isdigit(c); });

      if (non_digit != part.end()) {
        KLLM_LOG_DEBUG << "Stopping version parsing due to non-digit character '" << *non_digit << "' in part '" << part
                       << "' of version: " << v;
        break;
      }

      parts.push_back(std::stoi(part));  // 此时可以安全调用stoi
    }
    return parts;
  };

  version_parts = parse_version(version);
  min_version_parts = parse_version(min_version);

  // 比较版本号
  for (size_t i = 0; i < std::max(version_parts.size(), min_version_parts.size()); ++i) {
    int version_part = (i < version_parts.size()) ? version_parts[i] : 0;
    int min_version_part = (i < min_version_parts.size()) ? min_version_parts[i] : 0;

    if (version_part > min_version_part) return true;
    if (version_part < min_version_part) return false;
  }

  return true;  // 版本相等
}

// 确定所有可用的库信息
std::vector<FlashAttentionBackend::LibraryInfo> FlashAttentionBackend::DetermineAllLibraries(int compute_capability) {
  std::vector<LibraryInfo> available_libraries;

  // 1. 检查 FlashAttention 3 (适用于 Hopper 架构)
  if (compute_capability >= 90 && compute_capability < 100) {
    LibraryInfo fa3_info = GetFlashAttention3LibInfo();
    if (!fa3_info.path.empty()) {
      KLLM_LOG_DEBUG << "Found FlashAttention 3 library: " << fa3_info.path << ", version: " << fa3_info.version;
      available_libraries.push_back(fa3_info);
    }
  }

  if (compute_capability >= 80) {
    // 2. 检查 VLLM FlashAttention 2
    LibraryInfo vllm_info = GetVllmFlashAttentionLibInfo();
    if (!vllm_info.path.empty() && IsVersionGreaterOrEqual(vllm_info.version, "2.6.0")) {
      KLLM_LOG_DEBUG << "Found VLLM FlashAttention 2 library: " << vllm_info.path << ", version: " << vllm_info.version;
      available_libraries.push_back(vllm_info);
    } else if (!vllm_info.path.empty()) {
      KLLM_LOG_DEBUG << "VLLM FlashAttention version " << vllm_info.version << " doesn't meet requirement (>= 2.6.0)";
    }

    // 3. 检查标准 FlashAttention 2
    LibraryInfo fa2_info = GetFlashAttention2LibInfo();
    if (!fa2_info.path.empty() && IsVersionGreaterOrEqual(fa2_info.version, "2.5.0")) {
      KLLM_LOG_DEBUG << "Found FlashAttention 2 library: " << fa2_info.path << ", version: " << fa2_info.version;
      available_libraries.push_back(fa2_info);
    } else if (!fa2_info.path.empty()) {
      KLLM_LOG_DEBUG << "FlashAttention version " << fa2_info.version << " doesn't meet requirement (>= 2.5.0)";
    }
  }

  KLLM_LOG_INFO << "Found " << available_libraries.size() << " compatible FlashAttention libraries";
  return available_libraries;
}

// 获取 flash attention 3 库路径
std::string FlashAttentionBackend::GetFlashAttention3LibPath() { return GetPythonLibPath("flash_attn_3._C"); }

// 获取 vllm flash attention 库路径
std::string FlashAttentionBackend::GetVllmFlashAttentionLibPath() { return GetPythonLibPath("vllm_flash_attn_2_cuda"); }

// 获取 flash attention 2 库路径
std::string FlashAttentionBackend::GetFlashAttention2LibPath() { return GetPythonLibPath("flash_attn_2_cuda"); }

// 通过 Python 获取库路径
std::string FlashAttentionBackend::GetPythonLibPath(const std::string& module_name) {
  // 去除字符串两端的空白字符
  std::string module_name_processed = module_name;
  module_name_processed.erase(module_name_processed.find_last_not_of(" \n\r\t\f\v") + 1);
  module_name_processed.erase(0, module_name_processed.find_first_not_of(" \n\r\t\f\v"));

  if (module_name_processed.empty()) {
    KLLM_LOG_ERROR << "Module name cannot be empty";
    return "";
  }

  std::string command =
      "python -c \"import torch, " + module_name_processed + ";print(" + module_name_processed + ".__file__)\"";
  std::string result = ExecutePythonCommand(command);

  if (result.empty()) {
    KLLM_LOG_WARNING << "Python module " << module_name_processed << " not found or import failed.";
  }

  return result;
}

// 辅助函数：执行 Python 命令
std::string FlashAttentionBackend::ExecutePythonCommand(const std::string& command) {
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) {
    KLLM_LOG_ERROR << "Failed to run command: " << command;
    return "";
  }

  char buffer[256];
  std::string result;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }

  int exit_code = pclose(pipe);
  if (exit_code != 0) {
    KLLM_LOG_DEBUG << "Command failed with exit code: " << exit_code;
    return "";
  }

  // 去除末尾的换行符
  if (!result.empty() && result.back() == '\n') {
    result.pop_back();
  }

  return result;
}

// 获取 Python 模块的库信息
FlashAttentionBackend::LibraryInfo FlashAttentionBackend::GetPythonLibInfo(const std::string& lib_module_name,
                                                                           const std::string& version_module_name) {
  LibraryInfo info;

  // 去除字符串两端的空白字符
  std::string lib_module_processed = lib_module_name;
  lib_module_processed.erase(lib_module_processed.find_last_not_of(" \n\r\t\f\v") + 1);
  lib_module_processed.erase(0, lib_module_processed.find_first_not_of(" \n\r\t\f\v"));

  std::string version_module_processed = version_module_name;
  version_module_processed.erase(version_module_processed.find_last_not_of(" \n\r\t\f\v") + 1);
  version_module_processed.erase(0, version_module_processed.find_first_not_of(" \n\r\t\f\v"));

  if (lib_module_processed.empty() || version_module_processed.empty()) {
    KLLM_LOG_ERROR << "Module names cannot be empty";
    return info;  // 返回空的 LibraryInfo
  }

  // 设置库名称（使用版本模块名称作为标识）
  info.name = version_module_processed;

  // 1. 获取模块路径（使用库模块名称）
  info.path = GetPythonLibPath(lib_module_processed);
  if (info.path.empty()) {
    KLLM_LOG_DEBUG << "Failed to get path for module: " << lib_module_processed;
    return info;  // 如果路径获取失败，直接返回
  }

  // 2. 获取版本信息（使用版本模块名称）
  std::string version_command =
      "python -c \"from importlib import metadata; print(metadata.version('" + version_module_processed + "'))\"";
  info.version = ExecutePythonCommand(version_command);

  // 3. 获取次要版本号
  if (!info.version.empty()) {
    std::string minor_version_command = "python -c \"from importlib import metadata; print(metadata.version('" +
                                        version_module_processed + "').split('.')[1])\"";
    info.minor_version = ExecutePythonCommand(minor_version_command);
  }

  KLLM_LOG_DEBUG << "Python module info: name=" << info.name << ", lib_module=" << lib_module_processed
                 << ", version_module=" << version_module_processed << ", path=" << info.path
                 << ", version=" << info.version << ", minor_version=" << info.minor_version;

  return info;
}

// 获取 flash attention 3 库信息
FlashAttentionBackend::LibraryInfo FlashAttentionBackend::GetFlashAttention3LibInfo() {
  return GetPythonLibInfo("flash_attn_3._C", "flash_attn_3");
}

// 获取 vllm flash attention 库信息
FlashAttentionBackend::LibraryInfo FlashAttentionBackend::GetVllmFlashAttentionLibInfo() {
  return GetPythonLibInfo("vllm_flash_attn_2_cuda", "vllm_flash_attn");
}

// 获取 flash attention 2 库信息
FlashAttentionBackend::LibraryInfo FlashAttentionBackend::GetFlashAttention2LibInfo() {
  return GetPythonLibInfo("flash_attn_2_cuda", "flash_attn");
}

// 辅助函数：加载单个函数指针
template <typename FuncPtrType>
bool FlashAttentionBackend::LoadSingleFunction(const std::shared_ptr<RuntimeDllManager>& dll_manager,
                                               const std::string& function_name, FuncPtrType& func_ptr,
                                               const std::string& func_description) {
  std::string symbol = dll_manager->FindMangledFunctionSymbol(function_name, func_ptr);
  if (symbol.empty()) {
    return false;
  }

  func_ptr = dll_manager->GetRawFunctionPointer<FuncPtrType>(symbol);
  if (!func_ptr) {
    KLLM_LOG_ERROR << "Failed to load " << func_description << " with symbol: " << symbol;
    return false;
  } else {
    KLLM_LOG_DEBUG << func_description << " loaded successfully with symbol: " << symbol;
  }

  return true;
}

// 加载函数 - 为指定库加载函数
bool FlashAttentionBackend::LoadFunctions(LibraryInfo& lib_info) {
  if (!lib_info.dll_manager || !lib_info.dll_manager->IsLoaded()) {
    KLLM_LOG_ERROR << "Runtime DLL manager is not loaded for library: " << lib_info.name;
    return false;
  }

  // 根据库信息的版本确定加载哪个函数指针
  if (lib_info.name == "flash_attn_3") {
    // FlashAttention 3.0+ 版本
    KLLM_LOG_DEBUG << "Loading FlashAttention 3.0+ functions for library: " << lib_info.path;

    if (!LoadSingleFunction(lib_info.dll_manager, "mha_fwd", mha_fwd_fa3_, "FlashAttention 3 function mha_fwd_fa3")) {
      return false;
    }

    KLLM_LOG_DEBUG << "FlashAttention 3.0+ functions loaded successfully";

  } else if (lib_info.name == "vllm_flash_attn" && IsVersionGreaterOrEqual(lib_info.version, "2.6.0")) {
    // VLLM FlashAttention 2.6+ 版本
    KLLM_LOG_DEBUG << "Loading VLLM FlashAttention 2.6+ functions for library: " << lib_info.path;

    if (!LoadSingleFunction(lib_info.dll_manager, "mha_varlen_fwd", mha_varlen_fwd_vllm_flash_attn_v26_,
                            "VLLM function mha_varlen_fwd_vllm_flash_attn_v26") ||
        !LoadSingleFunction(lib_info.dll_manager, "mha_fwd_kvcache", mha_fwd_kvcache_vllm_flash_attn_v26_,
                            "VLLM function mha_fwd_kvcache_vllm_flash_attn_v26")) {
      return false;
    }

    KLLM_LOG_DEBUG << "VLLM FlashAttention 2.6+ functions loaded successfully";

  } else if (lib_info.name == "flash_attn" && IsVersionGreaterOrEqual(lib_info.version, "2.6.0")) {
    // FlashAttention 2.6+ 版本
    KLLM_LOG_DEBUG << "Loading FlashAttention 2.6+ functions for library: " << lib_info.path;

    if (!LoadSingleFunction(lib_info.dll_manager, "mha_varlen_fwd", mha_varlen_fwd_flash_attn_v26_,
                            "FlashAttention function mha_varlen_fwd_flash_attn_v26") ||
        !LoadSingleFunction(lib_info.dll_manager, "mha_fwd_kvcache", mha_fwd_kvcache_flash_attn_v26_,
                            "FlashAttention function mha_fwd_kvcache_flash_attn_v26")) {
      return false;
    }

    KLLM_LOG_DEBUG << "FlashAttention 2.6+ functions loaded successfully";

  } else if (lib_info.name == "flash_attn" && IsVersionGreaterOrEqual(lib_info.version, "2.5.0")) {
    // FlashAttention 2.5+ 版本
    KLLM_LOG_DEBUG << "Loading FlashAttention 2.5+ functions for library: " << lib_info.path;

    if (!LoadSingleFunction(lib_info.dll_manager, "mha_varlen_fwd", mha_varlen_fwd_flash_attn_v25_,
                            "FlashAttention function mha_varlen_fwd_flash_attn_v25") ||
        !LoadSingleFunction(lib_info.dll_manager, "mha_fwd_kvcache", mha_fwd_kvcache_flash_attn_v25_,
                            "FlashAttention function mha_fwd_kvcache_flash_attn_v25")) {
      return false;
    }

    KLLM_LOG_DEBUG << "FlashAttention 2.5+ functions loaded successfully";

  } else {
    // 未找到匹配的版本，返回错误
    KLLM_LOG_ERROR << "No matching version of FlashAttention library found for: " << lib_info.name << " v"
                   << lib_info.version;
    return false;
  }

  return true;
}

}  // namespace ksana_llm
