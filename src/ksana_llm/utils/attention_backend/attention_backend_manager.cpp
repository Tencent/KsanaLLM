/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/attention_backend/attention_backend_manager.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {


bool AttentionBackendManager::Initialize(AttentionBackendType backend_type) {
  // 重置状态
  is_ready_ = false;
  current_backend_.reset();

  // 设置后端类型
  backend_type_ = backend_type;

  // 根据后端类型创建相应的后端实例
  switch (backend_type_) {
    case AttentionBackendType::CUSTOM_BACKEND:
      // 预留扩展其他后端类型的创建逻辑
      KLLM_LOG_INFO << "Custom backend not supported yet.";
      return false;
    case AttentionBackendType::FLASH_ATTENTION:
      current_backend_ = std::make_unique<FlashAttentionBackend>();
      break;
    default:
      KLLM_LOG_ERROR << "Invalid backend type specified.";
      return false;
  }

  // 检查后端是否创建成功
  if (!current_backend_) {
    KLLM_LOG_ERROR << "Failed to create attention backend.";
    return false;
  }

  // 调用 InitializeBackend 进行具体的后端初始化
  return InitializeBackend();
}

bool AttentionBackendManager::InitializeBackend() {
  // 检查当前后端是否存在
  if (!current_backend_) {
    return false;
  }

  // 初始化当前后端
  bool init_success = current_backend_->Initialize();

  if (init_success) {
    // 验证后端是否正确初始化
    is_ready_ = current_backend_->IsInitialized();
  } else {
    is_ready_ = false;
  }

  return is_ready_;
}

}  // namespace ksana_llm