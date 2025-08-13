/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>
#include <string>

#include "ksana_llm/utils/runtime_dll_manager/runtime_dll_manager.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#else
// 非CUDA环境下的类型定义
using mha_varlen_fwd_vllm_flash_attn_v26_ptr = void*;
using mha_fwd_kvcache_vllm_flash_attn_v26_ptr = void*;
using mha_varlen_fwd_flash_attn_v25_ptr = void*;
using mha_fwd_kvcache_flash_attn_v25_ptr = void*;
using mha_varlen_fwd_flash_attn_v26_ptr = void*;
using mha_fwd_kvcache_flash_attn_v26_ptr = void*;
using mha_fwd_fa3_ptr = void*;
#endif

namespace ksana_llm {

class FlashAttentionBackend {
 private:
  struct LibraryInfo {
    std::string name;  // 库的名称或标识
    std::string path;
    std::string version;
    std::string minor_version;
    std::shared_ptr<RuntimeDllManager> dll_manager;  // 每个库独立的DLL管理器
  };

 public:
  // Backend 初始化：加载具体版本库
  bool Initialize();

  bool IsInitialized() const { return initialized_; }

  const std::vector<LibraryInfo>& GetLoadedLibraries() const { return loaded_libraries_; }

  static mha_varlen_fwd_vllm_flash_attn_v26_ptr mha_varlen_fwd_vllm_flash_attn_v26_;
  static mha_fwd_kvcache_vllm_flash_attn_v26_ptr mha_fwd_kvcache_vllm_flash_attn_v26_;

  static mha_varlen_fwd_flash_attn_v25_ptr mha_varlen_fwd_flash_attn_v25_;
  static mha_fwd_kvcache_flash_attn_v25_ptr mha_fwd_kvcache_flash_attn_v25_;

  static mha_varlen_fwd_flash_attn_v26_ptr mha_varlen_fwd_flash_attn_v26_;
  static mha_fwd_kvcache_flash_attn_v26_ptr mha_fwd_kvcache_flash_attn_v26_;

  static mha_fwd_fa3_ptr mha_fwd_fa3_;

 private:
  // 私有成员变量
  bool initialized_ = false;
  std::vector<LibraryInfo> loaded_libraries_;  // 所有已加载的库信息

  // 平台检测
  bool IsCudaPlatform();

  // CUDA 计算能力检测
  int GetCudaComputeCapability();

  // 库路径确定
  std::string DetermineLibraryPathByMacro();

  // 库确定 - 返回所有可用的库
  std::vector<LibraryInfo> DetermineAllLibraries(int compute_capability);

  // 获取不同版本的 FlashAttention 库路径
  std::string GetFlashAttention3LibPath();
  std::string GetVllmFlashAttentionLibPath();
  std::string GetFlashAttention2LibPath();

  // 通过 Python 获取库路径
  std::string GetPythonLibPath(const std::string& module_name);

  // 辅助函数：执行 Python 命令
  std::string ExecutePythonCommand(const std::string& command);

  // 辅助函数：检查版本是否大于等于最小版本要求
  bool IsVersionGreaterOrEqual(const std::string& version, const std::string& min_version);

  LibraryInfo GetFlashAttention3LibInfo();
  LibraryInfo GetVllmFlashAttentionLibInfo();
  LibraryInfo GetFlashAttention2LibInfo();

  // 获取 Python 模块的库信息
  LibraryInfo GetPythonLibInfo(const std::string& lib_module_name, const std::string& version_module_name);

  // 加载函数 - 为指定库加载函数
  bool LoadFunctions(LibraryInfo& lib_info);

  // 辅助函数：加载单个函数指针
  template <typename FuncPtrType>
  bool LoadSingleFunction(const std::shared_ptr<RuntimeDllManager>& dll_manager, const std::string& function_name,
                          FuncPtrType& func_ptr, const std::string& func_description);
};

}  // namespace ksana_llm