/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <string>
#include <functional>

#include "ksana_llm/utils/runtime_dll_manager/runtime_dll_manager.h"
#include "test.h"

namespace ksana_llm {

// 定义一个简单的函数类型用于测试
using SimpleFunc = int(*)(int);

TEST(RuntimeDllManager, LoadNonExistentLibrary) {
  RuntimeDllManager manager;
  // 测试加载不存在的库
  bool result = manager.Load("non_existent_library.so");
  EXPECT_FALSE(result);
  EXPECT_FALSE(manager.IsLoaded());
}

#ifdef __linux__
TEST(RuntimeDllManager, LoadExistingLibrary) {
  RuntimeDllManager manager;
  // 在Linux上测试加载libm.so (数学库)
  bool result = manager.Load("/lib64/libm.so.6");
  EXPECT_TRUE(result);
  EXPECT_TRUE(manager.IsLoaded());
}
#endif

TEST(RuntimeDllManager, LoadTest) {
  RuntimeDllManager manager;
  // 初始状态应该是未加载
  EXPECT_FALSE(manager.IsLoaded());

#ifdef __linux__
  // 加载一个存在的库
  bool result = manager.Load("/lib64/libm.so.6");
#else
  bool result = false;
#endif

  // 如果加载成功，is_loaded应该返回true，卸载后is_loaded应该返回false
  if (result) {
    EXPECT_TRUE(manager.IsLoaded());
    manager.Unload();
    EXPECT_FALSE(manager.IsLoaded());
  }
}

#ifdef __linux__
TEST(RuntimeDllManager, GetFunctionTest) {
  RuntimeDllManager manager;
  // 加载数学库
  bool result = manager.Load("/lib64/libm.so.6");
  EXPECT_TRUE(result);

  // 尝试获取sin函数
  auto sin_func = manager.GetFunction<double(double)>("sin");
  EXPECT_TRUE(sin_func != nullptr);

  // 测试函数调用
  if (sin_func) {
    double sin_value = sin_func(0.0);
    EXPECT_DOUBLE_EQ(sin_value, 0.0);
  }

  // 尝试获取不存在的函数
  auto non_existent_func = manager.GetFunction<void()>("non_existent_function");
  EXPECT_TRUE(non_existent_func == nullptr);
}
#endif

#ifdef __linux__
TEST(RuntimeDllManager, GetRawFunctionPointerTest) {
  RuntimeDllManager manager;
  // 加载数学库
  bool result = manager.Load("/lib64/libm.so.6");
  EXPECT_TRUE(result);

  // 尝试获取sin函数的原始指针
  using SinFuncPtr = double(*)(double);
  SinFuncPtr sin_ptr = manager.GetRawFunctionPointer<SinFuncPtr>("sin");
  EXPECT_TRUE(sin_ptr != nullptr);

  // 测试函数调用
  if (sin_ptr) {
    double sin_value = sin_ptr(0.0);
    EXPECT_DOUBLE_EQ(sin_value, 0.0);

    // 测试更多值
    double sin_pi_2 = sin_ptr(1.5707963267948966);  // π/2
    EXPECT_NEAR(sin_pi_2, 1.0, 1e-10);
  }

  // 尝试获取不存在的函数
  using NonExistentFuncPtr = void(*)();
  NonExistentFuncPtr non_existent_ptr = manager.GetRawFunctionPointer<NonExistentFuncPtr>("non_existent_function");
  EXPECT_TRUE(non_existent_ptr == nullptr);
}
#endif

// 测试在未加载库的情况下获取函数
TEST(RuntimeDllManager, GetFunctionWithoutLoadingTest) {
  RuntimeDllManager manager;
  // 未加载库的情况下，get_function应该返回nullptr
  auto func = manager.GetFunction<void()>("any_function");
  EXPECT_TRUE(func == nullptr);
}

// 测试在未加载库的情况下获取原始函数指针
TEST(RuntimeDllManager, GetRawFunctionPointerWithoutLoadingTest) {
  RuntimeDllManager manager;
  // 未加载库的情况下，GetRawFunctionPointer应该返回nullptr
  using AnyFuncPtr = void(*)();
  AnyFuncPtr func_ptr = manager.GetRawFunctionPointer<AnyFuncPtr>("any_function");
  EXPECT_TRUE(func_ptr == nullptr);
}

#ifdef __linux__
// 比较GetFunction和GetRawFunctionPointer的性能和行为
TEST(RuntimeDllManager, CompareGetFunctionAndGetRawFunctionPointer) {
  RuntimeDllManager manager;
  bool result = manager.Load("/lib64/libm.so.6");
  EXPECT_TRUE(result);

  // 使用GetFunction获取sin函数
  auto sin_func = manager.GetFunction<double(double)>("sin");
  EXPECT_TRUE(sin_func != nullptr);

  // 使用GetRawFunctionPointer获取sin函数
  using SinFuncPtr = double(*)(double);
  SinFuncPtr sin_ptr = manager.GetRawFunctionPointer<SinFuncPtr>("sin");
  EXPECT_TRUE(sin_ptr != nullptr);

  // 测试两种方式的结果应该相同
  if (sin_func && sin_ptr) {
    double test_value = 0.5;
    double result_func = sin_func(test_value);
    double result_ptr = sin_ptr(test_value);
    EXPECT_DOUBLE_EQ(result_func, result_ptr);
  }
}
#endif

#ifdef __linux__
// 测试获取所有导出符号
TEST(RuntimeDllManager, GetAllExportedSymbolsTest) {
  RuntimeDllManager manager;

  // 测试未加载库的情况
  auto symbols_empty = manager.GetAllExportedSymbols();
  EXPECT_TRUE(symbols_empty.empty());

  // 加载数学库
  bool result = manager.Load("/lib64/libm.so.6");
  EXPECT_TRUE(result);

  // 获取所有符号
  auto symbols = manager.GetAllExportedSymbols();
  EXPECT_FALSE(symbols.empty());

  // 检查是否包含常见的数学函数
  bool found_sin = false;
  bool found_cos = false;
  for (const auto& symbol : symbols) {
    if (symbol == "sin") found_sin = true;
    if (symbol == "cos") found_cos = true;
  }
  EXPECT_TRUE(found_sin);
  EXPECT_TRUE(found_cos);
}
#endif

#ifdef __linux__
// 测试查找包含特定子字符串的符号
TEST(RuntimeDllManager, FindSymbolsContainingTest) {
  RuntimeDllManager manager;

  // 测试未加载库的情况
  auto symbols_empty = manager.FindSymbolsContaining("sin");
  EXPECT_TRUE(symbols_empty.empty());

  // 加载数学库
  bool result = manager.Load("/lib64/libm.so.6");
  EXPECT_TRUE(result);

  // 查找包含 "sin" 的符号
  auto sin_symbols = manager.FindSymbolsContaining("sin");
  EXPECT_FALSE(sin_symbols.empty());

  // 验证所有返回的符号都包含 "sin"
  for (const auto& symbol : sin_symbols) {
    EXPECT_TRUE(symbol.find("sin") != std::string::npos);
  }

  // 查找不存在的子字符串
  auto nonexistent_symbols = manager.FindSymbolsContaining("nonexistent_substring_12345");
  EXPECT_TRUE(nonexistent_symbols.empty());
}
#endif

}  // namespace ksana_llm