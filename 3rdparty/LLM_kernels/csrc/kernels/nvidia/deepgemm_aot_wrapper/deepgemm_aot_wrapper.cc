/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/deepgemm_aot_wrapper/deepgemm_aot_wrapper.h"

#include <dlfcn.h>
#include <fmt/format.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include "yaml-cpp/yaml.h"

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

DeepGEMMAOTWrapper::DeepGEMMAOTWrapper(size_t m, size_t n, size_t k, bool need_generate_kernel,
                                       size_t tuner_device_id) {
  op_config_.m = m;
  op_config_.n = n;
  op_config_.k = k;
  std::string dynamic_lib_path = fmt::format("{}/.deepgemm/cache/deep_gemm_kernel_{}_{}_{}", InitCacheDir(), m, n, k);
  int status_code = 0;
  if (need_generate_kernel && !IsKernelExist(dynamic_lib_path)) {
    std::filesystem::path current_path = __FILE__;
    std::string current_dir = std::filesystem::absolute(current_path.parent_path()).string();
    std::string cmd = fmt::format(
        "python "
        "{}/../../../../tools/search_best_gemm_algo/deep_gemm_kernel_generator.py "
        "--m {} --n {} --k {} --kernel_saved_path {} --tuner_device_id {}",
        current_dir, m, n, k, dynamic_lib_path, tuner_device_id);
    // Execute the command
    status_code = std::system(cmd.c_str());
  }
  if (status_code == 0) {
    op_config_.dynamic_lib_path = fmt::format("{}/kernel.so", dynamic_lib_path);
    LoadFromYaml(fmt::format("{}/config.yaml", dynamic_lib_path));
  } else {
    ThrowError(fmt::format("Failed to generate GEMM algorithm configuration. Error code: {}", status_code));
  }

  LoadFuncFromSO();
}

DeepGEMMAOTWrapper::DeepGEMMAOTWrapper(const DeepGEMMOpConfig& op_config) : op_config_(op_config) {
  if (op_config_.dynamic_lib_path.empty()) {
    ThrowError("DeepGEMMAOTWrapper: dynamic_lib_path is empty");
  }

  handle_ = dlopen(op_config_.dynamic_lib_path.c_str(), RTLD_NOW);
  if (!handle_) {
    ThrowError(fmt::format("DeepGEMMAOTWrapper: Failed to load dynamic library: {}, error: {}",
      op_config_.dynamic_lib_path, dlerror()));
  }

  LoadFuncFromSO();
}

void DeepGEMMAOTWrapper::LoadFuncFromSO() {
  if (op_config_.dynamic_lib_path.empty()) {
    ThrowError("DeepGEMMAOTWrapper: dynamic_lib_path is empty");
  }

  handle_ = dlopen(op_config_.dynamic_lib_path.c_str(), RTLD_NOW);
  if (!handle_) {
    ThrowError(fmt::format("DeepGEMMAOTWrapper: Failed to load dynamic library: {}, error: {}",
                           op_config_.dynamic_lib_path, dlerror()));
  }

  launch_func_ = reinterpret_cast<LaunchFunc>(dlsym(handle_, "launch"));
  if (!launch_func_) {
    ThrowError(fmt::format("DeepGEMMAOTWrapper::LoadFuncFromSO: Failed to get launch function: {}", dlerror()));
  }
}

DeepGEMMAOTWrapper::~DeepGEMMAOTWrapper() {
  if (handle_) {
    dlclose(handle_);
    handle_ = nullptr;
  }
}

std::string DeepGEMMAOTWrapper::InitCacheDir() {
  if (std::getenv("DEEPGEMM_KERNEL_CACHE_PATH") != nullptr) {
    return std::string(std::getenv("DEEPGEMM_KERNEL_CACHE_PATH"));
  } else if (std::getenv("HOME") != nullptr) {
    return std::string(std::getenv("HOME"));
  } else {
    std::cerr << "DeepGEMMAOTWrapper: HOME environment variable not set, using current directory." << std::endl;
    return "./";
  }
}

void DeepGEMMAOTWrapper::ThrowError(const std::string& error_msg){
  std::cerr << error_msg << std::endl<<std::flush;
  throw std::runtime_error(error_msg);
}

bool DeepGEMMAOTWrapper::IsKernelExist(const std::string& directory) {

  if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
    return false;
  }
  std::filesystem::path dir_path(directory);
  std::vector<std::string> required_files = {"kernel.args", "kernel.cu", "kernel.cubin", "kernel.so", "config.yaml"};
  for( const auto& file : required_files) {
    if (!std::filesystem::exists(dir_path / file)) {
      return false;
    }
  }
  return true;
}

void DeepGEMMAOTWrapper::LoadFromYaml(const std::string& config_file) {
  YAML::Node config;
  try {
    config = YAML::LoadFile(config_file);
  } catch (YAML::BadFile& e) {
    ThrowError(fmt::format("DeepGEMMAOTWrapper: YAML config file {} could not be loaded: {}", config_file, e.what()));
  }
  op_config_.smem_size = config["smem_size"].as<uint64_t>();
  op_config_.num_sms = config["num_sms"].as<uint32_t>();
  uint32_t n = config["n"].as<uint32_t>();
  uint32_t k = config["k"].as<uint32_t>();
  if (n != op_config_.n || k != op_config_.k) {
    ThrowError(fmt::format("DeepGEMMAOTWrapper: n or k mismatch, expected n: {}, k: {}, but got n: {}, k: {}",
                           op_config_.n, op_config_.k, n, k));
  }
}

cudaError_t DeepGEMMAOTWrapper::Forward(void* raw_lhs, void* raw_lhs_scales, void* raw_rhs, void* raw_rhs_scales,
                                        void* raw_out, int m, void* raw_stream) {
  // Call the launch function
  int return_code = 0;
  launch_func_(raw_lhs, raw_lhs_scales, raw_rhs, raw_rhs_scales, raw_out, m, raw_stream, op_config_.num_sms,
              op_config_.smem_size, return_code);

  if (return_code != 0) {
    std::cerr << "DeepGEMMAOTWrapper::Forward: launch function returned error code: " << return_code << std::endl;
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

}  // namespace nvidia
}  // namespace llm_kernels