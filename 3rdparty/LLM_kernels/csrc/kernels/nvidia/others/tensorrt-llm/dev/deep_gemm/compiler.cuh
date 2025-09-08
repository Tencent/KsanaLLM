/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "csrc/kernels/nvidia/others/tensorrt-llm/dev/common/cudaUtils.h"
#include "jit_utils.cuh"
#include "nvrtc.h"
#include "runtime.cuh"
#include "scheduler.cuh"

#ifdef _WIN32
#  include <windows.h>
#endif

namespace deep_gemm::jit {

// Generate a unique ID for temporary directories to avoid collisions
inline std::string generateUniqueId() {
  // Use current time and random number to generate a unique ID
  static std::mt19937 gen(std::random_device{}());
  static std::uniform_int_distribution<> distrib(0, 999999);

  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto value = now_ms.time_since_epoch().count();

  // Use the static random generator
  int random_value = distrib(gen);

  return std::to_string(value) + "_" + std::to_string(random_value);
}

inline std::filesystem::path getDefaultUserDir() {
  static std::filesystem::path userDir;
  if (userDir.empty()) {
    char const* cacheDir = getenv("KLLM_DG_CACHE_DIR");
    if (cacheDir) {
      userDir = cacheDir;
      std::filesystem::create_directories(userDir);
    } else {
#ifdef _WIN32
      char const* appData = getenv("APPDATA");
      if (appData) {
        userDir = std::filesystem::path(appData) / "ksanallm";
      } else {
        userDir = std::filesystem::temp_directory_path() / "ksanallm";
      }
#else
      char const* homeDir = getenv("HOME");
      if (homeDir) {
        userDir = std::filesystem::path(homeDir) / ".ksanallm";
      } else {
        userDir = std::filesystem::temp_directory_path() / "ksanallm";
      }
#endif
    }
  }
  return userDir;
}

inline std::filesystem::path getTmpDir() { return getDefaultUserDir() / "tmp"; }

inline std::filesystem::path getCacheDir() { return getDefaultUserDir() / "cache"; }

inline std::string getNvccCompiler() {
  static std::string compiler;
  if (compiler.empty()) {
    // Check environment variable
    char const* envCompiler = getenv("KLLM_DG_NVCC_COMPILER");
    if (envCompiler) {
      compiler = envCompiler;
    } else {
      // Check CUDA_HOME
      char const* cudaHome = getenv("CUDA_HOME");
      if (cudaHome) {
        std::filesystem::path cudaPath(cudaHome);
#ifdef _WIN32
        compiler = (cudaPath / "bin" / "nvcc.exe").string();
#else
        compiler = (cudaPath / "bin" / "nvcc").string();
#endif
      } else {
// Default to system nvcc
#ifdef _WIN32
        compiler = "nvcc.exe";
#else
        compiler = "nvcc";
#endif
      }
    }
  }
  return compiler;
}

inline std::vector<std::filesystem::path> getJitIncludeDirs() {
  static std::vector<std::filesystem::path> includeDirs;
  if (includeDirs.empty()) {
    std::filesystem::path detail_path("3rdparty/LLM_kernels/csrc/kernels/nvidia/others/tensorrt-llm/dev");
    const char* env_path_str = std::getenv("DEEPGEMM_INCLUDE_DIR");
    if (env_path_str == nullptr || *env_path_str == '\0') {
      std::filesystem::path dir_path = std::filesystem::current_path() / ".." / detail_path;
      includeDirs.push_back(dir_path);
    } else {
      std::filesystem::path dir_path(env_path_str);
      dir_path = dir_path / detail_path;
      includeDirs.push_back(dir_path);
    }
  }
  return includeDirs;
}

inline std::string generateKernel(uint32_t const shape_n, uint32_t const shape_k, uint32_t const block_m,
                           uint32_t const block_n, uint32_t const block_k, uint32_t const num_groups,
                           uint32_t const num_stages, uint32_t const num_tma_multicast,
                           deep_gemm::GemmType const gemm_type, bool swapAB = false) {
  constexpr uint32_t kNumTMAThreads = 128;
  constexpr uint32_t kNumMathThreadsPerGroup = 128;

  std::string input_type;
  if (!swapAB) {
    switch (gemm_type) {
      case deep_gemm::GemmType::Normal:
        input_type = "NormalSchedulerInput";
        break;
      case deep_gemm::GemmType::GroupedContiguous:
        input_type = "GroupedContiguousSchedulerInput";
        break;
      case deep_gemm::GemmType::GroupedMasked:
        input_type = "GroupedMaskedSchedulerInput";
        break;
      case deep_gemm::GemmType::GroupedWithOffset:
        input_type = "GroupedWithOffsetSchedulerInput";
        break;
      case deep_gemm::GemmType::StridedBatched:
        input_type = "StridedBatchedSchedulerInput";
        break;
      default:
        throw std::runtime_error("Unsupported gemm type");
    }
  } else {
    switch (gemm_type) {
      case deep_gemm::GemmType::Normal:
        input_type = "NormalSchedulerInputSwapAB";
        break;
      case deep_gemm::GemmType::GroupedWithOffset:
        input_type = "GroupedWithOffsetSchedulerInputSwapAB";
        break;
      default:
        throw std::runtime_error("Unsupported gemm type");
    }
  }

  // Modify kernel name based on swapAB to determine which kernel function to use
  std::string kernel_name = swapAB ? "fp8_gemm_kernel_swapAB" : "fp8_gemm_kernel";
  std::string scheduler_name = swapAB ? "SchedulerSelectorSwapAB" : "SchedulerSelector";

  // Create the kernel source code using raw string literal
  std::string code = R"(
#ifdef __CUDACC_RTC__
#ifndef NVRTC_JIT_COMPILATION
#define NVRTC_JIT_COMPILATION
#endif

#include <deep_gemm/nvrtc_std.cuh>

#else

#include <string>
#include <cuda.h>

#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <deep_gemm/nvrtc_cutlass.cuh>
#include <deep_gemm/fp8_gemm_impl.cuh>

using namespace deep_gemm;

using SchedulerType =
typename )" + scheduler_name +
                     R"(<GemmType::)" + gemm_type_to_string(gemm_type) + R"(, )" + std::to_string(shape_n) + R"(, )" +
                     std::to_string(shape_k) + R"(, )" + std::to_string(block_m) + R"(, )" + std::to_string(block_n) +
                     R"(, )" + std::to_string(block_k) + R"(, )" + std::to_string(num_groups) + R"(, )" +
                     std::to_string(num_tma_multicast) + R"(>::type;

__global__ void dummy_kernel() {
  void *ptr = (void *)&)" +
                     kernel_name + R"(<)" + std::to_string(shape_n) + R"(, )" + std::to_string(shape_k) + R"(, )" +
                     std::to_string(block_m) + R"(, )" + std::to_string(block_n) + R"(, )" + std::to_string(block_k) +
                     R"(, )" + std::to_string(num_groups) + R"(, )" + std::to_string(num_stages) + R"(, )" +
                     std::to_string(kNumTMAThreads) + R"(, )" + std::to_string(kNumMathThreadsPerGroup) + R"(, )" +
                     std::to_string(num_tma_multicast) + R"(, SchedulerType, )" + input_type + R"(>;
}
)";

  return code;
}

/**
 * C++ implementation of the Compiler class
 * Compiles CUDA code into CUBINs
 * Modified to support thread-local storage
 */
class Compiler {
 public:
  // Get instance for specific thread
  static Compiler& getInstance(int thread_id = 0) {
    // 确保thread_id在有效范围内
    assert(thread_id >= 0 && thread_id < MAX_THREADS);
    static Compiler instances[MAX_THREADS];
    return instances[thread_id];
  }

  [[nodiscard]] bool isValid() const { return !includeDirs_.empty(); }

  // Build function with thread_id parameter
  Runtime* build(uint32_t const shape_n, uint32_t const shape_k, uint32_t const block_m, uint32_t const block_n,
                 uint32_t const block_k, uint32_t const num_groups, uint32_t const num_stages,
                 uint32_t const num_tma_multicast, deep_gemm::GemmType const gemm_type, bool swapAB = false,
                 int thread_id = 0) {
    int sm_version = llm_kernels::nvidia::tensorrt_llm::dev::common::getSMVersion();
    if (sm_version != 90) {
      KLLM_KERNEL_THROW(
          "DeepGEMM only supports Hopper (SM90) architectures, but current device compute "
          "capability is %d.",
          sm_version);
    }

    // Build signature - simplified, no MD5 calculation
    std::string name = std::string(swapAB ? "gemm_swapAB_" : "gemm_") + std::to_string(shape_n) + "_" +
                       std::to_string(shape_k) + "_" + std::to_string(block_m) + "_" + std::to_string(block_n) + "_" +
                       std::to_string(block_k) + "_" + std::to_string(num_groups) + "_" + std::to_string(num_stages) +
                       std::to_string(num_groups) + "_" + std::to_string(num_stages) + "_" +
                       std::to_string(num_tma_multicast) + "_" + gemm_type_to_string(gemm_type);
    std::filesystem::path path = getCacheDir() / name;

    // Check runtime cache or file system hit using thread-local cache
    auto& runtimeCache = getThreadRuntimeCache(thread_id);
    Runtime* cachedRuntime = runtimeCache[path.string()];
    if (cachedRuntime != nullptr) {
      if (kJitDebugging) {
        // TODO(jinxcwu): 这里原本是TLLM_LOG_INFO
        printf("Using cached JIT runtime %s during build (thread %d)\n", name.c_str(), thread_id);
      }
      return cachedRuntime;
    }

    // Compiler flags
    std::vector<std::string> flags = {"-std=c++17",
                                      "--gpu-architecture=sm_90a",
                                      "--ptxas-options=-allow-expensive-optimizations=true",
                                      "--ptxas-options=--register-usage-level=10",
                                      "--diag-suppress=161,174,177,940",
                                      "-D__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__=1",
                                      "-D__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__=1"};

    if (kJitUseNvcc) {
      flags.push_back("-O3");
      flags.push_back("-cubin");
      flags.push_back("--expt-relaxed-constexpr");
      flags.push_back("--expt-extended-lambda");

      std::vector<std::string> cxxFlags = {"-fPIC", "-O3", "-Wno-deprecated-declarations", "-Wno-abi"};
      std::string cxxFlagsStr = "--compiler-options=";
      for (size_t i = 0; i < cxxFlags.size(); ++i) {
        cxxFlagsStr += cxxFlags[i];
        if (i < cxxFlags.size() - 1) {
          cxxFlagsStr += ",";
        }
      }
      flags.push_back(cxxFlagsStr);
    } else {
      flags.push_back("-default-device");
    }

    std::filesystem::path tmpPath = getTmpDir() / (name + "_" + generateUniqueId());
    std::filesystem::path cubinPath = path / kKernelName;
    std::filesystem::path tmpCubinPath = tmpPath / kKernelName;

    // Create the target directory if it doesn't exist
    if (kJitUseNvcc || kJitDumpCubin) {
      std::filesystem::create_directories(tmpPath);
      std::filesystem::create_directories(path);
    }

    for (auto const& dir : includeDirs_) {
      flags.push_back("-I" + dir.string());
    }

    // Print options if debug enabled
    if (kJitDebugging) {
      printf("Compiling JIT runtime %s with options: ", name.c_str());
      for (auto const& flag : flags) {
        printf("%s ", flag.c_str());
      }
      printf("\n");
    }

    std::string code = generateKernel(shape_n, shape_k, block_m, block_n, block_k, num_groups, num_stages,
                                      num_tma_multicast, gemm_type, swapAB);

    if (kJitDebugging) {
      printf("Generated kernel code:\n%s\n", code.c_str());
    }

    if (kJitUseNvcc) {
      std::filesystem::path tmpSrcPath = tmpPath / "kernel.cu";

      // Write files
      std::ofstream srcFile(tmpSrcPath);
      srcFile << code;
      srcFile.close();

      // Build command
      std::vector<std::string> command = {getNvccCompiler(), tmpSrcPath.string(), "-o", tmpCubinPath.string()};
      command.insert(command.end(), flags.begin(), flags.end());

      // Execute command
      std::string cmd;
      for (auto const& arg : command) {
        cmd += arg + " ";
      }

      // Buffer to store the output
      std::array<char, 128> buffer;
      std::string result;

      // Time the compilation
      auto start = std::chrono::high_resolution_clock::now();

      // Open pipe to command
#ifdef _MSC_VER
      FILE* pipe = _popen(cmd.c_str(), "r");
#else
      FILE* pipe = popen(cmd.c_str(), "r");
#endif

      if (pipe) {
        // Read the output
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
          result += buffer.data();
        }

// Close the pipe
#ifdef _MSC_VER
        _pclose(pipe);
#else
        pclose(pipe);
#endif

        // Output result if debug enabled
        if (kJitDebugging) {
          auto end = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
          printf("NVCC compilation took %ld ms\n", duration.count());
          printf("Compilation log:\n%s\n", result.c_str());
        }
      }
    } else {
      nvrtcProgram prog;
      CHECK_NVRTC(nvrtcCreateProgram(&prog, code.c_str(), "kernel.cu", 0, nullptr, nullptr));

      std::vector<char const*> options;
      for (auto const& flag : flags) {
        options.push_back(flag.c_str());
      }

      // Time the compilation
      auto start = std::chrono::high_resolution_clock::now();
      nvrtcResult compileResult = nvrtcCompileProgram(prog, options.size(), options.data());

      if (kJitDebugging) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("NVRTC compilation took %ld ms\n", duration.count());

        size_t logSize;
        CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &logSize));
        std::vector<char> log(logSize);
        CHECK_NVRTC(nvrtcGetProgramLog(prog, log.data()));
        printf("Compilation log:\n%s\n", log.data());
      }

      // Check if compilation succeeded
      if (compileResult != NVRTC_SUCCESS) {
        // TODO(jinxcwu): 这里原本是TLLM_LOG_ERROR
        printf("NVRTC compilation failed\n");
        CHECK_NVRTC(nvrtcDestroyProgram(&prog));
        throw std::runtime_error("NVRTC compilation failed");
      }

      // Save CUBIN to a file
      size_t cubinSize;
      CHECK_NVRTC(nvrtcGetCUBINSize(prog, &cubinSize));
      std::vector<char> cubin(cubinSize);
      CHECK_NVRTC(nvrtcGetCUBIN(prog, cubin.data()));

      // Cache the runtime in memory by default
      if (!kJitDumpCubin) {
        auto runtime = std::make_unique<Runtime>(path.string(), cubin, gemm_type);
        Runtime* result = runtime.get();
        runtimeCache.set(path.string(), std::move(runtime));
        if (kJitDebugging) {
          printf("Successfully cached JIT runtime %s in memory (thread %d)\n", name.c_str(), thread_id);
        }
        return result;
      }

      std::ofstream cubinFile(tmpCubinPath.string(), std::ios::binary);
      cubinFile.write(cubin.data(), static_cast<std::streamsize>(cubinSize));
      cubinFile.close();
      CHECK_NVRTC(nvrtcDestroyProgram(&prog));
    }

    // Copy the source and compiled files to the cache directory
    try {
      // Rename (atomic operation) to final locations
      std::filesystem::rename(tmpCubinPath, cubinPath);
      if (kJitDebugging) {
        printf("Successfully copied kernel files to cache directory: %s\n", path.string().c_str());
      }
    } catch (std::exception const& e) {
      printf("Warning: Failed to copy kernel files to cache: %s\n", e.what());
    }

    // Clean up temporary directory after successful compilation
    try {
      std::filesystem::remove_all(tmpPath);
    } catch (std::exception const& e) {
      printf("Warning: Failed to clean up temporary directory: %s\n", e.what());
    }

    // Create runtime and cache it in thread-local storage
    auto runtime = std::make_unique<Runtime>(path.string(), std::vector<char>(), gemm_type);
    Runtime* result = runtime.get();
    runtimeCache.set(path.string(), std::move(runtime));
    return result;
  }

 private:
  std::vector<std::filesystem::path> includeDirs_;

  // Private constructor for singleton pattern
  Compiler() : includeDirs_(getJitIncludeDirs()) {
    // Create necessary directories
    if (kJitUseNvcc || kJitDumpCubin) {
      std::filesystem::create_directories(getTmpDir());
      std::filesystem::create_directories(getCacheDir());
    }
  }

  // Delete copy constructor and assignment operator
  Compiler(Compiler const&) = delete;
  Compiler& operator=(Compiler const&) = delete;
};

// 获取指定线程的Compiler实例
inline Compiler& getThreadCompiler(int thread_id = 0) { return Compiler::getInstance(thread_id); }

// 为了保持向后兼容性，保留原有的getGlobalCompiler函数
// TODO(jinxcwu) 后面要删掉这种，强制用getThreadCompiler
inline Compiler& getGlobalCompiler() { return Compiler::getInstance(0); }

}  // namespace deep_gemm::jit
