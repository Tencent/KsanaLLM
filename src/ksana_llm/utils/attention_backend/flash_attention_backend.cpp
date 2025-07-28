/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <filesystem>
#include <functional>
#include <sstream>
#include <vector>

#include "ksana_llm/utils/attention_backend/flash_attention_backend.h"

#ifdef ENABLE_CUDA
#include "ksana_llm/utils/nvidia/cuda_utils.h"
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

bool FlashAttentionBackend::Initialize() {
  // 1. 平台检测
  if (!IsCudaPlatform()) {
    KLLM_LOG_WARNING << "FlashAttentionBackend only support cuda platform, will not be initialized";
    return false;
  }

  // 2. 获取并检查 CUDA compute capability
  int compute_capability = GetCudaComputeCapability();
  if (compute_capability < 80) {  // SM 8.0 及以上支持 FlashAttention 2
    KLLM_LOG_WARNING << "Compute capability " << compute_capability
                  << " not support FlashAttention 2";
    return false;
  }

  // 3. 确定库信息
  current_library_info_ = DetermineLibrary(compute_capability);
  if (current_library_info_.path.empty()) {
    KLLM_LOG_WARNING << "No compatible FlashAttention library found";
    return false;
  }

  // 4. 按照库信息的path加载库
  runtime_dll_manager_ = std::make_shared<RuntimeDllManager>();
  if (!runtime_dll_manager_->Load(current_library_info_.path)) {
    KLLM_LOG_ERROR << "Failed to Load FlashAttention library from " << current_library_info_.path;
    return false;
  }

  // 5. 按照库信息的版本确定加载哪个函数指针
  if (!LoadFunctions()) {
    KLLM_LOG_ERROR << "Failed to load functions from FlashAttention library";
    return false;
  }

  // 6. 设置初始化状态
  initialized_ = true;
  KLLM_LOG_INFO << "FlashAttentionBackend initialized successfully with library: "
                << current_library_info_.name << " (version: " << current_library_info_.version
                << ", path: " << current_library_info_.path << ")";
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


// 通过编译宏确定库路径
std::string FlashAttentionBackend::DetermineLibraryPathByMacro() {
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  std::string lib_path = GetVllmFlashAttentionLibPath();
  if (!lib_path.empty()) {
    KLLM_LOG_INFO << "Using VLLM FlashAttention 2 library: " << lib_path;
    return lib_path;
  }
#endif
#ifdef ENABLE_FLASH_ATTN_2
  std::string lib_path = GetFlashAttention2LibPath();
  if (!lib_path.empty()) {
    KLLM_LOG_INFO << "Using FlashAttention 2 library: " << lib_path;
    return lib_path;
  }
#endif
  return "";
}

// 辅助函数：检查版本是否大于等于最小版本要求
bool FlashAttentionBackend::IsVersionGreaterOrEqual(const std::string& version, const std::string& min_version) {
  if (version.empty() || min_version.empty()) {
    KLLM_LOG_DEBUG << "Invalid version strings: version=" << version
                   << ", min_version=" << min_version;
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
      auto non_digit = std::find_if(part.begin(), part.end(),
          [](char c) { return !std::isdigit(c); });

      if (non_digit != part.end()) {
          KLLM_LOG_DEBUG << "Stopping version parsing due to non-digit character '"
                          << *non_digit << "' in part '" << part
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

// 确定库信息
FlashAttentionBackend::LibraryInfo FlashAttentionBackend::DetermineLibrary(int compute_capability) {
  LibraryInfo lib_info;

  if (compute_capability >= 90 && compute_capability < 100) {  // Hopper 架构
    lib_info = GetFlashAttention3LibInfo();
    if (!lib_info.path.empty()) {
      KLLM_LOG_INFO << "Using FlashAttention 3 library: " << lib_info.path
                    << ", version: " << lib_info.version;
      return lib_info;
    }
  }

  if (compute_capability >= 80) {
    #ifdef ENABLE_VLLM_FLASH_ATTN_2
      // FlashAttention 2 根据宏使用 vllm 版本，要求 2.6 版本
      lib_info = GetVllmFlashAttentionLibInfo();
      if (!lib_info.path.empty() && IsVersionGreaterOrEqual(lib_info.version, "2.6.0")) {
        KLLM_LOG_INFO << "Using VLLM FlashAttention 2 library: " << lib_info.path
                      << ", version: " << lib_info.version;
        return lib_info;
      } else if (!lib_info.path.empty()) {
        KLLM_LOG_ERROR << "VLLM FlashAttention version " << lib_info.version << " doesn't meet requirement (>= 2.6.0)";
      }
    #elif defined(ENABLE_FLASH_ATTN_2)
      // 尝试标准 flash-attn，要求 2.5 或更高版本
      lib_info = GetFlashAttention2LibInfo();
      if (!lib_info.path.empty() && IsVersionGreaterOrEqual(lib_info.version, "2.5.0")) {
        KLLM_LOG_INFO << "Using FlashAttention 2 library: " << lib_info.path
                      << ", version: " << lib_info.version;
        return lib_info;
      } else if (!lib_info.path.empty()) {
        KLLM_LOG_ERROR << "FlashAttention version " << lib_info.version << " doesn't meet requirement (>= 2.5.0)";
      }
    #else
      // 如果没有启用任何宏，提示报错
      KLLM_LOG_ERROR << "No FlashAttention library enabled. "
                        "Please define ENABLE_VLLM_FLASH_ATTN_2 or ENABLE_FLASH_ATTN_2.";
    #endif
  }

  KLLM_LOG_INFO << "No compatible FlashAttention library found";
  return LibraryInfo();  // 返回空的 LibraryInfo
}

// 获取 flash attention 3 库路径
std::string FlashAttentionBackend::GetFlashAttention3LibPath() {
  // TODO(raybxu): 未来支持 flash attention 3
  return "";
}


// 获取 vllm flash attention 库路径
std::string FlashAttentionBackend::GetVllmFlashAttentionLibPath() {
  return GetPythonLibPath("vllm_flash_attn_2_cuda");
}

// 获取 flash attention 2 库路径
std::string FlashAttentionBackend::GetFlashAttention2LibPath() {
  return GetPythonLibPath("flash_attn_2_cuda");
}

// 通过 Python 获取库路径
std::string FlashAttentionBackend::GetPythonLibPath(const std::string& module_name) {
  // 去除字符串两端的空白字符
  std::string module_name_processed = module_name;
  module_name_processed.erase(module_name_processed.find_last_not_of(" \n\r\t\f\v")+1);
  module_name_processed.erase(0, module_name_processed.find_first_not_of(" \n\r\t\f\v"));

  if (module_name_processed.empty()) {
    KLLM_LOG_ERROR << "Module name cannot be empty";
    return "";
  }

  std::string command = "python -c \"import torch, " + module_name_processed + ";print("
                        + module_name_processed + ".__file__)\"";
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
FlashAttentionBackend::LibraryInfo FlashAttentionBackend::GetPythonLibInfo(
    const std::string& lib_module_name,
    const std::string& version_module_name) {
  LibraryInfo info;

  // 去除字符串两端的空白字符
  std::string lib_module_processed = lib_module_name;
  lib_module_processed.erase(lib_module_processed.find_last_not_of(" \n\r\t\f\v")+1);
  lib_module_processed.erase(0, lib_module_processed.find_first_not_of(" \n\r\t\f\v"));

  std::string version_module_processed = version_module_name;
  version_module_processed.erase(version_module_processed.find_last_not_of(" \n\r\t\f\v")+1);
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
  std::string version_command = "python -c \"import " + version_module_processed
                                + ";print(" + version_module_processed + ".__version__)\"";
  info.version = ExecutePythonCommand(version_command);

  // 3. 获取次要版本号
  if (!info.version.empty()) {
    std::string minor_version_command = "python -c \"import " + version_module_processed
                                        + ";print(" + version_module_processed + ".__version__.split('.')[1])\"";
    info.minor_version = ExecutePythonCommand(minor_version_command);
  }

  KLLM_LOG_DEBUG << "Python module info: name=" << info.name
                 << ", lib_module=" << lib_module_processed
                 << ", version_module=" << version_module_processed
                 << ", path=" << info.path << ", version=" << info.version
                 << ", minor_version=" << info.minor_version;

  return info;
}

// 获取 flash attention 3 库信息
FlashAttentionBackend::LibraryInfo FlashAttentionBackend::GetFlashAttention3LibInfo() {
  // TODO(raybxu): 未来支持 flash attention 3
  LibraryInfo info;
  return info;
}

// 获取 vllm flash attention 库信息
FlashAttentionBackend::LibraryInfo FlashAttentionBackend::GetVllmFlashAttentionLibInfo() {
  return GetPythonLibInfo("vllm_flash_attn_2_cuda", "vllm_flash_attn");
}

// 获取 flash attention 2 库信息
FlashAttentionBackend::LibraryInfo FlashAttentionBackend::GetFlashAttention2LibInfo() {
  return GetPythonLibInfo("flash_attn_2_cuda", "flash_attn");
}

// 辅助函数：查找函数符号
std::string FlashAttentionBackend::FindFunctionSymbol(const std::string& function_name) {
  auto symbols = runtime_dll_manager_->FindSymbolsContaining(function_name);
  if (symbols.empty()) {
    KLLM_LOG_ERROR << "No symbols found containing: " << function_name;
    return "";
  }

  // 简单选择第一个找到的符号
  std::string symbol = symbols[0];
  KLLM_LOG_DEBUG << "Found symbol for " << function_name << ": " << symbol;
  return symbol;
}

// 辅助函数：加载单个函数指针
template<typename FuncPtrType>
bool FlashAttentionBackend::LoadSingleFunction(
    const std::string& function_name,
    FuncPtrType& func_ptr,
    const std::string& func_description) {
  std::string symbol = FindFunctionSymbol(function_name);
  if (symbol.empty()) {
    return false;
  }

  func_ptr = runtime_dll_manager_->GetRawFunctionPointer<FuncPtrType>(symbol);
  if (!func_ptr) {
    KLLM_LOG_ERROR << "Failed to load " << func_description << " with symbol: " << symbol;
    return false;
  } else {
    KLLM_LOG_DEBUG << func_description << " loaded successfully with symbol: " << symbol;
  }

  return true;
}

// 加载函数
bool FlashAttentionBackend::LoadFunctions() {
  if (!runtime_dll_manager_ || !runtime_dll_manager_->IsLoaded()) {
    KLLM_LOG_ERROR << "Runtime DLL manager is not loaded.";
    return false;
  }

  // 根据库信息的版本确定加载哪个函数指针
  if (current_library_info_.name == "vllm_flash_attn" &&
      IsVersionGreaterOrEqual(current_library_info_.version, "2.6.0")) {
    // VLLM FlashAttention 2.6+ 版本
    KLLM_LOG_INFO << "Loading VLLM FlashAttention 2.6+ functions";

    if (!LoadSingleFunction("mha_varlen_fwd", mha_varlen_fwd_vllm_flash_attn_v26_,
                           "VLLM function mha_varlen_fwd_vllm_flash_attn_v26") ||
        !LoadSingleFunction("mha_fwd_kvcache", mha_fwd_kvcache_vllm_flash_attn_v26_,
                           "VLLM function mha_fwd_kvcache_vllm_flash_attn_v26")) {
      return false;
    }

    KLLM_LOG_DEBUG << "VLLM FlashAttention 2.6+ functions loaded successfully";

  } else if (current_library_info_.name == "flash_attn" &&
             IsVersionGreaterOrEqual(current_library_info_.version, "2.6.0")) {
    // FlashAttention 2.6+ 版本
    KLLM_LOG_INFO << "Loading FlashAttention 2.6+ functions";

    if (!LoadSingleFunction("mha_varlen_fwd", mha_varlen_fwd_flash_attn_v26_,
                           "FlashAttention function mha_varlen_fwd_flash_attn_v26") ||
        !LoadSingleFunction("mha_fwd_kvcache", mha_fwd_kvcache_flash_attn_v26_,
                           "FlashAttention function mha_fwd_kvcache_flash_attn_v26")) {
      return false;
    }

    KLLM_LOG_DEBUG << "FlashAttention 2.6+ functions loaded successfully";

  } else if (current_library_info_.name == "flash_attn" &&
             IsVersionGreaterOrEqual(current_library_info_.version, "2.5.0")) {
    // FlashAttention 2.5+ 版本
    KLLM_LOG_INFO << "Loading FlashAttention 2.5+ functions";

    if (!LoadSingleFunction("mha_varlen_fwd", mha_varlen_fwd_flash_attn_v25_,
                           "FlashAttention function mha_varlen_fwd_flash_attn_v25") ||
        !LoadSingleFunction("mha_fwd_kvcache", mha_fwd_kvcache_flash_attn_v25_,
                           "FlashAttention function mha_fwd_kvcache_flash_attn_v25")) {
      return false;
    }

    KLLM_LOG_DEBUG << "FlashAttention 2.5+ functions loaded successfully";

  } else {
    // 未找到匹配的版本，返回错误
    KLLM_LOG_ERROR << "No matching version of FlashAttention library found";
    return false;
  }

  return true;
}

}  // namespace ksana_llm