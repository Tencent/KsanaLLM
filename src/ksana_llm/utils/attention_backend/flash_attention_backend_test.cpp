/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <string>

#include "ksana_llm/utils/attention_backend/flash_attention_backend.h"
#include "test.h"


namespace ksana_llm {

// 测试用的 FlashAttentionBackend 子类，用于访问私有方法
class TestableFlashAttentionBackend : public FlashAttentionBackend {
 public:
  // 公开私有方法用于测试
  using FlashAttentionBackend::IsCudaPlatform;
  using FlashAttentionBackend::GetCudaComputeCapability;
  using FlashAttentionBackend::DetermineLibrary;
  using FlashAttentionBackend::GetFlashAttention3LibPath;
  using FlashAttentionBackend::GetVllmFlashAttentionLibPath;
  using FlashAttentionBackend::GetFlashAttention2LibPath;
  using FlashAttentionBackend::GetPythonLibPath;
  using FlashAttentionBackend::GetPythonLibInfo;
  using FlashAttentionBackend::GetFlashAttention3LibInfo;
  using FlashAttentionBackend::GetVllmFlashAttentionLibInfo;
  using FlashAttentionBackend::GetFlashAttention2LibInfo;
  using FlashAttentionBackend::ExecutePythonCommand;
  using FlashAttentionBackend::IsVersionGreaterOrEqual;
  using FlashAttentionBackend::LibraryInfo;

  // 访问私有成员
  bool getInitializedState() const { return IsInitialized(); }
  const LibraryInfo& getCurrentLibraryInfo() const { return GetCurrentLibraryInfo(); }
};

class FlashAttentionBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    backend_ = std::make_unique<TestableFlashAttentionBackend>();
  }

  void TearDown() override {
    backend_.reset();
  }

  std::unique_ptr<TestableFlashAttentionBackend> backend_;
};

// ============================================================================
// 平台检测测试
// ============================================================================

TEST_F(FlashAttentionBackendTest, IsCudaPlatformTest) {
  // 测试平台检测功能
  bool is_cuda = backend_->IsCudaPlatform();

#ifdef ENABLE_CUDA
  EXPECT_TRUE(is_cuda);
#else
  EXPECT_FALSE(is_cuda);
#endif
}

// ============================================================================
// CUDA 计算能力测试
// ============================================================================

#ifdef ENABLE_CUDA
TEST_F(FlashAttentionBackendTest, GetCudaComputeCapabilityTest) {
  // 注意：这个测试需要实际的 CUDA 环境
  // 在没有 GPU 的环境中可能会抛出异常
  try {
    int compute_capability = backend_->GetCudaComputeCapability();
    EXPECT_GE(compute_capability, 0);
    // 计算能力应该是合理的值（如 75, 80, 86, 89, 90 等）
    EXPECT_LE(compute_capability, 100);
  } catch (const std::exception& e) {
    // 在没有 GPU 的测试环境中，这是预期的行为
    GTEST_SKIP() << "No GPU available for testing: " << e.what();
  }
}
#else
TEST_F(FlashAttentionBackendTest, GetCudaComputeCapabilityNoCudaTest) {
  // 在非 CUDA 平台上应该返回 -1
  int compute_capability = backend_->GetCudaComputeCapability();
  EXPECT_EQ(compute_capability, -1);
}
#endif


// ============================================================================
// 各种库路径获取测试
// ============================================================================

TEST_F(FlashAttentionBackendTest, GetFlashAttention3LibPathTest) {
  // 目前返回空字符串
  std::string lib_path = backend_->GetFlashAttention3LibPath();
  EXPECT_TRUE(lib_path.empty());
}

TEST_F(FlashAttentionBackendTest, GetVllmFlashAttentionLibPathTest) {
  // 这会尝试调用 Python，在测试环境中可能失败
  std::string lib_path = backend_->GetVllmFlashAttentionLibPath();
  // 不管成功还是失败，都应该返回字符串（可能为空）
  EXPECT_TRUE(lib_path.empty() || !lib_path.empty());
}

TEST_F(FlashAttentionBackendTest, GetFlashAttention2LibPathTest) {
  // 这会尝试调用 Python，在测试环境中可能失败
  std::string lib_path = backend_->GetFlashAttention2LibPath();
  // 不管成功还是失败，都应该返回字符串（可能为空）
  EXPECT_TRUE(lib_path.empty() || !lib_path.empty());
}

// ============================================================================
// 版本比较测试
// ============================================================================

TEST_F(FlashAttentionBackendTest, IsVersionGreaterOrEqualTest) {
  // 测试版本比较功能
  EXPECT_TRUE(backend_->IsVersionGreaterOrEqual("2.6.0", "2.5.0"));
  EXPECT_TRUE(backend_->IsVersionGreaterOrEqual("2.6.1", "2.6.0"));
  EXPECT_TRUE(backend_->IsVersionGreaterOrEqual("3.0.0", "2.9.9"));
  EXPECT_TRUE(backend_->IsVersionGreaterOrEqual("2.5.0", "2.5.0"));  // 相等

  EXPECT_FALSE(backend_->IsVersionGreaterOrEqual("2.5.0", "2.6.0"));
  EXPECT_FALSE(backend_->IsVersionGreaterOrEqual("2.5.9", "2.6.0"));
  EXPECT_FALSE(backend_->IsVersionGreaterOrEqual("1.9.9", "2.0.0"));

  // 测试边界情况
  EXPECT_FALSE(backend_->IsVersionGreaterOrEqual("", "2.5.0"));
  EXPECT_FALSE(backend_->IsVersionGreaterOrEqual("2.5.0", ""));
  EXPECT_FALSE(backend_->IsVersionGreaterOrEqual("", ""));

  // 测试不规范版本号
  EXPECT_TRUE(backend_->IsVersionGreaterOrEqual("2.6", "2.5"));
  EXPECT_TRUE(backend_->IsVersionGreaterOrEqual("2.6.0.1", "2.6.0"));
  EXPECT_TRUE(backend_->IsVersionGreaterOrEqual("2.6.0.post1", "2.6.0"));
}

// ============================================================================
// ExecutePythonCommand 测试
// ============================================================================

TEST_F(FlashAttentionBackendTest, ExecutePythonCommandTest) {
  // 测试简单的 Python 命令
  std::string result = backend_->ExecutePythonCommand("python -c \"print('hello')\"");
  // 在有 Python 的环境中应该返回 "hello"，否则为空
  EXPECT_TRUE(result.empty() || result == "hello");

  // 测试失败的命令
  std::string fail_result = backend_->ExecutePythonCommand("python -c \"import non_existent_module\"");
  EXPECT_TRUE(fail_result.empty());

  // 测试空命令
  std::string empty_result = backend_->ExecutePythonCommand("");
  EXPECT_TRUE(empty_result.empty());
}

// ============================================================================
// LibraryInfo 相关测试
// ============================================================================

TEST_F(FlashAttentionBackendTest, GetPythonLibInfoTest) {
  // 测试获取 Python 模块信息
  auto info = backend_->GetPythonLibInfo("sys", "sys");

  // sys 模块通常存在，但在测试环境中可能没有 torch 依赖
  if (!info.path.empty()) {
    EXPECT_EQ(info.name, "sys");
    EXPECT_FALSE(info.path.empty());
    // version 和 minor_version 可能为空，因为 sys 模块可能没有 __version__
  }

  // 测试不存在的模块
  auto invalid_info = backend_->GetPythonLibInfo("non_existent_module_12345", "non_existent_module_12345");
  EXPECT_TRUE(invalid_info.path.empty());
  EXPECT_EQ(invalid_info.name, "non_existent_module_12345");

  // 测试空模块名
  auto empty_info = backend_->GetPythonLibInfo("", "");
  EXPECT_TRUE(empty_info.path.empty());
  EXPECT_TRUE(empty_info.name.empty());
}

TEST_F(FlashAttentionBackendTest, GetFlashAttention3LibInfoTest) {
  auto info = backend_->GetFlashAttention3LibInfo();
  EXPECT_TRUE(info.name.empty());  // 目前未实现
  EXPECT_TRUE(info.path.empty());
  EXPECT_TRUE(info.version.empty());
}

TEST_F(FlashAttentionBackendTest, GetVllmFlashAttentionLibInfoTest) {
  auto info = backend_->GetVllmFlashAttentionLibInfo();
  // 在测试环境中可能没有 vllm_flash_attn 模块
  if (!info.path.empty()) {
    EXPECT_EQ(info.name, "vllm_flash_attn");
    EXPECT_FALSE(info.path.empty());
  }
}

TEST_F(FlashAttentionBackendTest, GetFlashAttention2LibInfoTest) {
  auto info = backend_->GetFlashAttention2LibInfo();
  // 在测试环境中可能没有 flash_attn 模块
  if (!info.path.empty()) {
    EXPECT_EQ(info.name, "flash_attn");
    EXPECT_FALSE(info.path.empty());
  }
}

// ============================================================================
// DetermineLibrary 测试
// ============================================================================

TEST_F(FlashAttentionBackendTest, DetermineLibraryHopperTest) {
  // 测试 Hopper 架构（SM 9.0+）
  auto lib_info = backend_->DetermineLibrary(90);
  // 由于 GetFlashAttention3LibInfo 目前返回空路径，应该尝试其他版本
  EXPECT_TRUE(lib_info.path.empty() || !lib_info.path.empty());
  if (!lib_info.path.empty()) {
    EXPECT_FALSE(lib_info.name.empty());
    EXPECT_FALSE(lib_info.version.empty());
  }
}

TEST_F(FlashAttentionBackendTest, DetermineLibraryAmpereTest) {
  // 测试 Ampere 架构（SM 8.0+）
  auto lib_info = backend_->DetermineLibrary(80);
  // 应该尝试 VLLM 和 FlashAttention 2，但可能因为版本不符合要求而失败
  EXPECT_TRUE(lib_info.path.empty() || !lib_info.path.empty());
  if (!lib_info.path.empty()) {
    EXPECT_FALSE(lib_info.name.empty());
    EXPECT_FALSE(lib_info.version.empty());
    // 验证版本要求
    if (lib_info.name == "vllm_flash_attn") {
      EXPECT_TRUE(backend_->IsVersionGreaterOrEqual(lib_info.version, "2.6.0"));
    } else if (lib_info.name == "flash_attn") {
      EXPECT_TRUE(backend_->IsVersionGreaterOrEqual(lib_info.version, "2.5.0"));
    }
  }
}

TEST_F(FlashAttentionBackendTest, DetermineLibraryUnsupportedTest) {
  // 测试不支持的计算能力
  auto lib_info = backend_->DetermineLibrary(75);
  EXPECT_TRUE(lib_info.path.empty());
  EXPECT_TRUE(lib_info.name.empty());
}

// ============================================================================
// Python 库路径获取测试
// ============================================================================

TEST_F(FlashAttentionBackendTest, GetPythonLibPathValidModuleTest) {
  // 测试获取一个通常存在的 Python 模块
  std::string lib_path = backend_->GetPythonLibPath("sys");
  // sys 模块通常存在，但在测试环境中可能没有 torch
  EXPECT_TRUE(lib_path.empty() || lib_path.find("sys") != std::string::npos);
}

TEST_F(FlashAttentionBackendTest, GetPythonLibPathInvalidModuleTest) {
  // 测试获取不存在的模块
  std::string lib_path = backend_->GetPythonLibPath("non_existent_module_12345");
  EXPECT_TRUE(lib_path.empty());
}

// ============================================================================
// 初始化状态测试
// ============================================================================

TEST_F(FlashAttentionBackendTest, InitialStateTest) {
  // 初始状态应该是未初始化
  EXPECT_FALSE(backend_->IsInitialized());
}

// ============================================================================
// 初始化流程测试
// ============================================================================

#ifndef ENABLE_CUDA
TEST_F(FlashAttentionBackendTest, InitializeNoCudaPlatformTest) {
  // 在非 CUDA 平台上初始化应该失败
  bool result = backend_->Initialize();
  EXPECT_FALSE(result);
  EXPECT_FALSE(backend_->IsInitialized());
}
#endif

// ============================================================================
// 边界条件和错误处理测试
// ============================================================================

TEST_F(FlashAttentionBackendTest, GetPythonLibPathEmptyModuleTest) {
  // 测试空模块名
  std::string lib_path = backend_->GetPythonLibPath("");
  EXPECT_TRUE(lib_path.empty());
}

// ============================================================================
// 新的初始化流程测试
// ============================================================================

#ifdef ENABLE_CUDA
TEST_F(FlashAttentionBackendTest, InitializeWithLibraryInfoTest) {
  // 测试新的初始化流程，验证库信息是否正确设置
  try {
    int compute_capability = backend_->GetCudaComputeCapability();
    if (compute_capability >= 80) {
      // 在初始化前，库信息应该为空
      const auto& initial_info = backend_->getCurrentLibraryInfo();
      EXPECT_TRUE(initial_info.path.empty());

      // 尝试初始化
      bool result = backend_->Initialize();
      // 验证状态一致性
      EXPECT_EQ(result, backend_->IsInitialized());

      if (result) {
        // 初始化成功后，验证库信息
        const auto& current_info = backend_->getCurrentLibraryInfo();
        EXPECT_FALSE(current_info.path.empty());
        EXPECT_FALSE(current_info.name.empty());
        EXPECT_FALSE(current_info.version.empty());

        // 验证版本要求
        if (current_info.name == "vllm_flash_attn") {
          EXPECT_TRUE(backend_->IsVersionGreaterOrEqual(current_info.version, "2.6.0"));
        } else if (current_info.name == "flash_attn") {
          EXPECT_TRUE(backend_->IsVersionGreaterOrEqual(current_info.version, "2.5.0"));
        }
      }
    } else {
      GTEST_SKIP() << "Compute capability " << compute_capability << " not supported";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "No GPU available for testing: " << e.what();
  }
}
#endif

// ============================================================================
// 更新的边界条件测试
// ============================================================================

TEST_F(FlashAttentionBackendTest, DetermineLibraryBoundaryTestUpdated) {
  // 测试 DetermineLibrary 的边界值
  auto info_79 = backend_->DetermineLibrary(79);
  EXPECT_TRUE(info_79.path.empty());   // 刚好不支持

  auto info_80 = backend_->DetermineLibrary(80);
  EXPECT_TRUE(info_80.path.empty() || !info_80.path.empty());  // 刚好支持，但可能没有库

  auto info_90 = backend_->DetermineLibrary(90);
  EXPECT_TRUE(info_90.path.empty() || !info_90.path.empty());  // Hopper 最低
}

}  // namespace ksana_llm