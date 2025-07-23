/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/attention_backend/attention_backend_manager.h"
#include "test.h"

namespace ksana_llm {

class AttentionBackendManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 每个测试开始前重置管理器状态
    resetManagerState();
  }

  void TearDown() override {
    // 每个测试结束后重置管理器状态
    resetManagerState();
  }

  // 重置管理器状态的辅助方法
  void resetManagerState() {
    // 使用单例模式重置状态
    auto manager = AttentionBackendManager::GetInstance();
    manager->Reset();
  }
};

// ============================================================================
// 初始化测试
// ============================================================================

TEST_F(AttentionBackendManagerTest, InitializeDifferentBackendsTest) {
  auto manager = AttentionBackendManager::GetInstance();

  // 先初始化 FLASH_ATTENTION
  bool result1 = manager->Initialize(AttentionBackendType::FLASH_ATTENTION);
  bool ready1 = manager->IsReady();

  // 再初始化 CUSTOM_BACKEND（应该失败）
  bool result2 = manager->Initialize(AttentionBackendType::CUSTOM_BACKEND);
  bool ready2 = manager->IsReady();

  // CUSTOM_BACKEND 应该失败
  EXPECT_FALSE(result2);
  EXPECT_FALSE(ready2);

  // 验证状态一致性
  EXPECT_EQ(result2, ready2);

  // 调用 InitializeBackend 应该失败，因为没有有效的后端
  bool load_result = manager->InitializeBackend();
  EXPECT_FALSE(load_result);

  // 测试无效的后端类型
  // 注意：这里使用强制类型转换来测试边界情况
  AttentionBackendType invalid_type = static_cast<AttentionBackendType>(999);

  bool result = manager->Initialize(invalid_type);

  // 无效类型应该导致初始化失败
  EXPECT_FALSE(result);
  EXPECT_FALSE(manager->IsReady());
}

}  // namespace ksana_llm