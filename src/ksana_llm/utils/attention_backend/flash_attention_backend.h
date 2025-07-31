/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <functional>

#include "ksana_llm/utils/runtime_dll_manager/runtime_dll_manager.h"

#ifdef ENABLE_CUDA
#include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#else
// 非CUDA环境下的类型定义
using mha_varlen_fwd_vllm_flash_attn_v26_ptr = void*;
using mha_fwd_kvcache_vllm_flash_attn_v26_ptr = void*;
using mha_varlen_fwd_flash_attn_v25_ptr = void*;
using mha_fwd_kvcache_flash_attn_v25_ptr = void*;
using mha_varlen_fwd_flash_attn_v26_ptr = void*;
using mha_fwd_kvcache_flash_attn_v26_ptr = void*;
#endif

namespace ksana_llm {

class FlashAttentionBackend {
 private:
  struct LibraryInfo {
    std::string name;           // 库的名称或标识
    std::string path;
    std::string version;
    std::string minor_version;
  };

 public:
  // Backend 初始化：加载具体版本库
  bool Initialize();

  bool IsInitialized() const { return initialized_; }

  const LibraryInfo& GetCurrentLibraryInfo() const { return current_library_info_; }

  static mha_varlen_fwd_vllm_flash_attn_v26_ptr mha_varlen_fwd_vllm_flash_attn_v26_;
  static mha_fwd_kvcache_vllm_flash_attn_v26_ptr mha_fwd_kvcache_vllm_flash_attn_v26_;

  static mha_varlen_fwd_flash_attn_v25_ptr mha_varlen_fwd_flash_attn_v25_;
  static mha_fwd_kvcache_flash_attn_v25_ptr mha_fwd_kvcache_flash_attn_v25_;

  static mha_varlen_fwd_flash_attn_v26_ptr mha_varlen_fwd_flash_attn_v26_;
  static mha_fwd_kvcache_flash_attn_v26_ptr mha_fwd_kvcache_flash_attn_v26_;

 private:
  // 私有成员变量
  std::shared_ptr<RuntimeDllManager> runtime_dll_manager_;
  bool initialized_ = false;
  LibraryInfo current_library_info_;  // 当前使用的库信息

  // 平台检测
  bool IsCudaPlatform();

  // CUDA 计算能力检测
  int GetCudaComputeCapability();

  // 库路径确定
  std::string DetermineLibraryPathByMacro();

  // 库确定
  LibraryInfo DetermineLibrary(int compute_capability);

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

  // 加载函数
  bool LoadFunctions();

  // 辅助函数：加载单个函数指针
  template<typename FuncPtrType>
  bool LoadSingleFunction(const std::string& function_name, FuncPtrType& func_ptr, const std::string& func_description);
};

}  // namespace ksana_llm