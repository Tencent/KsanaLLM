/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/attention_backend/flash_attention_backend.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

enum class AttentionBackendType {
  FLASH_ATTENTION,
  CUSTOM_BACKEND  // TODO(raybxu): 预留扩展性：后续如有必要可以考虑抽象一个BaseBackend
};

class AttentionBackendManager {
 public:
  AttentionBackendManager() = default;
  ~AttentionBackendManager() = default;

  // Singleton模版封装单例获取方法
  static std::shared_ptr<AttentionBackendManager> GetInstance() {
    return Singleton<AttentionBackendManager>::GetInstance();
  }

  // 单例模式禁止拷贝, 禁止赋值
  AttentionBackendManager(const AttentionBackendManager&) = delete;
  AttentionBackendManager& operator=(const AttentionBackendManager&) = delete;

  // Backend类型选择
  bool Initialize(AttentionBackendType backend_type = AttentionBackendType::FLASH_ATTENTION);

  // 辅助函数：初始化具体Backend实例（具体Backend内部自行处理版本选择逻辑）
  // 上层：AttentionBackendManager 与 下层：具体Backend内部逻辑 解耦合
  bool InitializeBackend();

  bool IsReady() const { return is_ready_; }

  // 测试用的重置方法
  void Reset() {
    is_ready_ = false;
    current_backend_.reset();
    backend_type_ = AttentionBackendType::FLASH_ATTENTION;
  }

 private:
  std::unique_ptr<FlashAttentionBackend> current_backend_;
  AttentionBackendType backend_type_ = AttentionBackendType::FLASH_ATTENTION;
  bool is_ready_ = false;
};

}  // namespace ksana_llm