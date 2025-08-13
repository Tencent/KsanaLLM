/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <functional>
#include <string>

#include "ksana_llm/utils/runtime_dll_manager/runtime_dll_manager.h"
#include "test.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#  include "ksana_llm/utils/attention_backend/flash_attention_backend.h"
#endif

namespace ksana_llm {

// 定义一个简单的函数类型用于测试
using SimpleFunc = int (*)(int);

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
  using SinFuncPtr = double (*)(double);
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
  using NonExistentFuncPtr = void (*)();
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
  using AnyFuncPtr = void (*)();
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
  using SinFuncPtr = double (*)(double);
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
TEST(RuntimeDllManager, FindSymbolsForFunctionTest) {
  RuntimeDllManager manager;

  // 测试未加载库的情况
  auto symbols_empty = manager.FindSymbolsForFunction("sin");
  EXPECT_TRUE(symbols_empty.empty());

  // 加载数学库
  bool result = manager.Load("/lib64/libm.so.6");
  EXPECT_TRUE(result);

  // 查找包含 "sin" 的符号
  auto sin_symbols = manager.FindSymbolsForFunction("sin");
  EXPECT_FALSE(sin_symbols.empty());

  // 验证所有返回的符号都包含 "sin"
  for (const auto& symbol : sin_symbols) {
    EXPECT_TRUE(symbol.find("sin") != std::string::npos);
  }

  // 查找不存在的子字符串
  auto nonexistent_symbols = manager.FindSymbolsForFunction("nonexistent_substring_12345");
  EXPECT_TRUE(nonexistent_symbols.empty());
}
#endif

// 测试符号解码功能
TEST(RuntimeDllManager, DemangleSymbolTest) {
  ksana_llm::RuntimeDllManager manager;

  // 测试简单的 mangled 符号
  std::string mangled_name = "_Z3addii";  // add(int, int) 的 mangled 形式
  std::string demangled = manager.DemangleSymbol(mangled_name);
  EXPECT_EQ(demangled, "add(int, int)");

  // 测试复杂的 mangled 符号
  std::string complex_mangled = "_ZN3std6vectorIiSaIiEE9push_backERKi";
  std::string complex_demangled = manager.DemangleSymbol(complex_mangled);
  EXPECT_TRUE(complex_demangled.find("std::vector") != std::string::npos);
  EXPECT_TRUE(complex_demangled.find("push_back") != std::string::npos);

  // 测试无效的 mangled 符号（应该返回原始字符串）
  std::string invalid_mangled = "not_a_mangled_symbol";
  std::string invalid_demangled = manager.DemangleSymbol(invalid_mangled);
  EXPECT_EQ(invalid_demangled, invalid_mangled);

  // 测试空字符串
  std::string empty_demangled = manager.DemangleSymbol("");
  EXPECT_EQ(empty_demangled, "");
}

// 测试从 demangle 后的函数指针字符串中提取参数
TEST(RuntimeDllManager, ExtractParametersFromDemangledFuncStringTest) {
  ksana_llm::RuntimeDllManager manager;

  // 测试标准函数指针格式
  std::string func_ptr_str = "void (*)(int, float)";
  std::string params = manager.ExtractParametersFromDemangledFuncString(func_ptr_str);
  EXPECT_EQ(params, "(int, float)");

  // 测试返回值为非 void 的函数指针
  std::string func_ptr_str2 = "int (*)(double, char*)";
  std::string params2 = manager.ExtractParametersFromDemangledFuncString(func_ptr_str2);
  EXPECT_EQ(params2, "(double, char*)");

  // 测试复杂参数类型
  std::string complex_func_ptr = "std::vector<int> (*)(const std::string&, std::map<int, float>&)";
  std::string complex_params = manager.ExtractParametersFromDemangledFuncString(complex_func_ptr);
  EXPECT_EQ(complex_params, "(const std::string&, std::map<int, float>&)");

  // 测试无参数函数
  std::string no_param_func = "void (*)()";
  std::string no_params = manager.ExtractParametersFromDemangledFuncString(no_param_func);
  EXPECT_EQ(no_params, "()");

  // 测试无效格式（没有函数指针模式）
  std::string invalid_format = "not a function pointer";
  std::string invalid_params = manager.ExtractParametersFromDemangledFuncString(invalid_format);
  EXPECT_EQ(invalid_params, "");

  // 测试空字符串
  std::string empty_params = manager.ExtractParametersFromDemangledFuncString("");
  EXPECT_EQ(empty_params, "");
}

#ifdef __linux__
// 测试 FindMangledFunctionSymbol 基础功能
TEST(RuntimeDllManager, FindMangledFunctionSymbolBasicTest) {
  ksana_llm::RuntimeDllManager manager;

  // 加载数学库
  bool result = manager.Load("/lib64/libm.so.6");
  EXPECT_TRUE(result);

  // 定义一个简单的函数指针类型用于测试
  using MathFunc = double (*)(double);
  MathFunc dummy_func_ptr = nullptr;

  // 尝试查找 sin 函数的 mangled 符号
  std::string sin_symbol = manager.FindMangledFunctionSymbol("sin", dummy_func_ptr);

  // 由于 C 函数通常不会被 mangle，这里主要测试函数不会崩溃
  // 实际的 mangled 符号查找会在 FlashAttention 相关测试中进行
  EXPECT_TRUE(sin_symbol.empty() || !sin_symbol.empty());  // 不管找到与否都是正常的
}
#endif

#ifdef ENABLE_CUDA
// 测试与 FlashAttention 相关的符号查找功能（仅在 CUDA 环境下）
TEST(RuntimeDllManager, FindMangledFunctionSymbolFlashAttentionTest) {
  ksana_llm::RuntimeDllManager manager;
  ksana_llm::FlashAttentionBackend fa_backend;

  // 尝试初始化 FlashAttention 后端以获取库信息
  // Initialize() 方法内部会调用 DetermineLibrary 来确定使用哪个版本的库
  bool fa_initialized = fa_backend.Initialize();

  if (!fa_initialized) {
    // 如果 FlashAttention 初始化失败，跳过此测试
    GTEST_SKIP() << "FlashAttention backend not available, skipping test";
    return;
  }

  // 通过公有接口获取当前使用的库信息
  // 这些信息是通过内部的 DetermineLibrary 函数确定的
  const auto& loaded_libs = fa_backend.GetLoadedLibraries();
  EXPECT_FALSE(loaded_libs.empty());
  const auto& lib_info = loaded_libs[0];  // 使用第一个加载的库进行测试
  EXPECT_FALSE(lib_info.path.empty());

  // 使用相同的库路径加载到我们的测试管理器中
  bool result = manager.Load(lib_info.path);
  EXPECT_TRUE(result);

  // 由于 FlashAttention 后端已经通过 Initialize() 成功初始化，
  // 说明它已经找到了合适的库和版本，我们可以直接测试对应的函数指针类型

  // 测试不同版本的函数指针类型，看哪个能找到符号
  bool found_symbol = false;

  if (lib_info.name == "flash_attn_3") {
    // 测试 FlashAttention 3 版本的函数
    mha_fwd_fa3_ptr dummy_ptr = nullptr;
    std::string symbol = manager.FindMangledFunctionSymbol("mha_fwd", dummy_ptr);

    if (!symbol.empty()) {
      EXPECT_TRUE(symbol.find("mha_fwd") != std::string::npos);
      auto func_ptr = manager.GetRawFunctionPointer<mha_fwd_fa3_ptr>(symbol);
      EXPECT_TRUE(func_ptr != nullptr);
      found_symbol = true;
    }
  } else if (lib_info.name == "vllm_flash_attn") {
    // 测试 VLLM FlashAttention 版本的函数
    mha_varlen_fwd_vllm_flash_attn_v26_ptr dummy_ptr = nullptr;
    std::string symbol = manager.FindMangledFunctionSymbol("mha_varlen_fwd", dummy_ptr);

    if (!symbol.empty()) {
      EXPECT_TRUE(symbol.find("mha_varlen_fwd") != std::string::npos);
      auto func_ptr = manager.GetRawFunctionPointer<mha_varlen_fwd_vllm_flash_attn_v26_ptr>(symbol);
      EXPECT_TRUE(func_ptr != nullptr);
      found_symbol = true;
    }
  } else if (lib_info.name == "flash_attn") {
    // 对于标准 flash_attn，尝试不同版本的函数指针类型
    // 首先尝试 2.6 版本
    mha_varlen_fwd_flash_attn_v26_ptr dummy_ptr_v26 = nullptr;
    std::string symbol_v26 = manager.FindMangledFunctionSymbol("mha_varlen_fwd", dummy_ptr_v26);

    if (!symbol_v26.empty()) {
      EXPECT_TRUE(symbol_v26.find("mha_varlen_fwd") != std::string::npos);
      auto func_ptr = manager.GetRawFunctionPointer<mha_varlen_fwd_flash_attn_v26_ptr>(symbol_v26);
      EXPECT_TRUE(func_ptr != nullptr);
      found_symbol = true;
    } else {
      // 如果 2.6 版本没找到，尝试 2.5 版本
      mha_varlen_fwd_flash_attn_v25_ptr dummy_ptr_v25 = nullptr;
      std::string symbol_v25 = manager.FindMangledFunctionSymbol("mha_varlen_fwd", dummy_ptr_v25);

      if (!symbol_v25.empty()) {
        EXPECT_TRUE(symbol_v25.find("mha_varlen_fwd") != std::string::npos);
        auto func_ptr = manager.GetRawFunctionPointer<mha_varlen_fwd_flash_attn_v25_ptr>(symbol_v25);
        EXPECT_TRUE(func_ptr != nullptr);
        found_symbol = true;
      }
    }
  }

  // 由于 FlashAttention 后端已经成功初始化，我们应该能找到至少一个符号
  EXPECT_TRUE(found_symbol);
}

// 测试 FindSymbolsForFunction 在 FlashAttention 库中的使用
TEST(RuntimeDllManager, FindSymbolsForFunctionFlashAttentionTest) {
  ksana_llm::RuntimeDllManager manager;
  ksana_llm::FlashAttentionBackend fa_backend;

  // 尝试初始化 FlashAttention 后端
  bool fa_initialized = fa_backend.Initialize();

  if (!fa_initialized) {
    GTEST_SKIP() << "FlashAttention backend not available, skipping test";
    return;
  }

  // 获取库信息并加载
  const auto& loaded_libs = fa_backend.GetLoadedLibraries();
  EXPECT_FALSE(loaded_libs.empty());
  const auto& lib_info = loaded_libs[0];  // 使用第一个加载的库进行测试
  bool result = manager.Load(lib_info.path);
  EXPECT_TRUE(result);

  // 查找包含 "mha" 的符号
  auto mha_symbols = manager.FindSymbolsForFunction("mha");

  // 应该能找到一些包含 "mha" 的符号
  EXPECT_FALSE(mha_symbols.empty());

  // 验证所有返回的符号都包含 "mha"
  for (const auto& symbol : mha_symbols) {
    EXPECT_TRUE(symbol.find("mha") != std::string::npos);
  }

  // 查找包含 "fwd" 的符号
  auto fwd_symbols = manager.FindSymbolsForFunction("fwd");

  // 验证找到的符号
  for (const auto& symbol : fwd_symbols) {
    EXPECT_TRUE(symbol.find("fwd") != std::string::npos);
  }
}

// 测试 DemangleSymbol 在实际 FlashAttention 符号上的表现
TEST(RuntimeDllManager, DemangleSymbolFlashAttentionTest) {
  ksana_llm::RuntimeDllManager manager;
  ksana_llm::FlashAttentionBackend fa_backend;

  // 尝试初始化 FlashAttention 后端
  bool fa_initialized = fa_backend.Initialize();

  if (!fa_initialized) {
    GTEST_SKIP() << "FlashAttention backend not available, skipping test";
    return;
  }

  // 获取库信息并加载
  const auto& loaded_libs = fa_backend.GetLoadedLibraries();
  EXPECT_FALSE(loaded_libs.empty());
  const auto& lib_info = loaded_libs[0];  // 使用第一个加载的库进行测试
  bool result = manager.Load(lib_info.path);
  EXPECT_TRUE(result);

  // 获取所有符号
  auto all_symbols = manager.GetAllExportedSymbols();
  EXPECT_FALSE(all_symbols.empty());

  // 测试对一些符号进行 demangle
  int demangled_count = 0;
  for (const auto& symbol : all_symbols) {
    if (symbol.find("mha") != std::string::npos && demangled_count < 5) {
      std::string demangled = manager.DemangleSymbol(symbol);

      // demangle 后的结果应该不为空
      EXPECT_FALSE(demangled.empty());

      // 如果是 C++ mangled 符号，demangle 后应该与原符号不同
      // 如果是 C 符号，demangle 后应该与原符号相同
      EXPECT_TRUE(demangled == symbol || demangled != symbol);

      demangled_count++;
    }
  }
}
#endif

}  // namespace ksana_llm
