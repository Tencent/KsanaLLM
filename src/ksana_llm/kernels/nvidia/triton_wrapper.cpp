/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/kernels/nvidia/triton_wrapper.h"

#include <filesystem>
#include <iostream>

#include <nlohmann/json.hpp>

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/string_utils.h"

namespace fs = std::filesystem;

namespace ksana_llm {

#define CUDA_CHECK_WITH_LOG(call, info)                                \
  do {                                                                 \
    CUresult err = call;                                               \
    if (err != CUDA_SUCCESS) {                                         \
      const char* errStr;                                              \
      cuGetErrorString(err, &errStr);                                  \
      KLLM_LOG_ERROR << "CUDA Error: " << errStr << ", info:" << info; \
      abort();                                                         \
    }                                                                  \
  } while (0)

inline size_t CeilDiv(size_t a, size_t b) { return (a + b - 1) / b; }

template <typename T>
inline std::string GetComputeType() {
  if (std::is_same<T, float>::value) {
    return "FP32";
  } else if (std::is_same<T, half>::value) {
    return "FP16";
  } else if (std::is_same<T, nv_bfloat16>::value) {
    return "BF16";
  } else {
    return "Unknown";
  }
}

template <typename T>
inline std::string ConvertToString(const T& value) {
  if (std::is_same<T, int>::value) {
    return std::to_string(value);
  } else if (std::is_same<T, bool>::value) {
    return value ? "True" : "False";
  } else if (std::is_same<T, size_t>::value) {
    return std::to_string(value);
  } else {
    return "Unknown";
  }
}

TritonWrapper::TritonWrapper() {
  InitKernelsDir();

  // 初始化,声明kernel关键key的组合顺序
  kernel_key_map_["fused_moe_kernel"] = {"group_n",      "group_k",      "BLOCK_SIZE_M",      "BLOCK_SIZE_N",
                                         "BLOCK_SIZE_K", "GROUP_SIZE_M", "MUL_ROUTED_WEIGHT", "top_k",
                                         "compute_type", "use_fp8_w8a8", "use_int8_w8a16", "even_Ks"};
  kernel_key_map_["fused_moe_gptq_awq_kernel"] = {"BLOCK_SIZE_M",      "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
                                                  "MUL_ROUTED_WEIGHT", "top_k",        "compute_type", "has_zp",
                                                  "weight_bits",       "group_size"};
  kernel_key_map_["fused_moe_gptq_int4_fp8_kernel"] = {"BLOCK_SIZE_M", "BLOCK_SIZE_N",      "BLOCK_SIZE_K",
                                                       "GROUP_SIZE_M", "MUL_ROUTED_WEIGHT", "top_k",
                                                       "compute_type", "group_size"};
}

void TritonWrapper::InitKernelsDir() {
  // If KSANA_TRITON_KERNEL_PATH is set, KsanaLLM will load the kernel from
  // ${KSANA_TRITON_KERNEL_PATH}/triton_kernel_files. Otherwise, it will load the kernel from the default path
  // ${HOME}/.cache/KsanaLLM/triton_kernel_files, or from the relative path ./triton_kernel_files.
  const char* env_path = std::getenv("KSANA_TRITON_KERNEL_PATH");
  if (env_path != nullptr) {
    kernels_dir_ = fmt::format("{}/triton_kernel_files/", env_path);
  } else {
    const char* home_dir = std::getenv("HOME");
    if (home_dir != nullptr) {
      kernels_dir_ = std::string(home_dir) + "/.cache/KsanaLLM/triton_kernel_files";
    } else {
      kernels_dir_ = "./triton_kernel_files";
      KLLM_LOG_WARNING << "Cannot get HOME environment variable, using relative path: " << kernels_dir_;
    }
  }
}

void TritonWrapper::SetKernelsDir(const std::string& kernels_dir) { kernels_dir_ = kernels_dir; }

std::optional<std::string> TritonWrapper::ConstructKernelName(const std::string& kernel_base_name,
                                                              const std::unordered_map<std::string, std::string>& map) {
  // 根据关键 key 的组合顺序,获取本地文件的文件名
  if (kernel_key_map_.find(kernel_base_name) == kernel_key_map_.end()) {
    KLLM_LOG_ERROR << fmt::format("Could not get triton kernel [{}]'s config from local file!", kernel_base_name);
    return std::nullopt;
  }
  std::string kernel_name = kernel_base_name;
  for (auto& key : kernel_key_map_[kernel_base_name]) {
    if (map.find(key) == map.end()) {
      KLLM_LOG_ERROR << fmt::format("The necessary key {} is missing from the user's input.", key);
      return std::nullopt;
    }
    kernel_name += fmt::format("_{}_{}", key, map.at(key));
  }
  return kernel_name;
}

std::optional<std::shared_ptr<TritonKernel>> TritonWrapper::LoadTritonKernelJsonFile(const std::string& file_path,
                                                                                     const std::string& kernel_name) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    KLLM_LOG_ERROR << fmt::format("File {} does not exist.", file_path);
    return std::nullopt;
  }
  nlohmann::json json_data;
  file >> json_data;
  std::shared_ptr<TritonKernel> kernel = std::make_shared<TritonKernel>();
  try {
    kernel->shm_size = json_data.at("shm_size");
  } catch (const nlohmann::json::out_of_range& e) {
    std::cout << "Error: Missing key in JSON file." << std::endl;
    return std::nullopt;
  }
  kernel->num_warps = json_data.value("num_warps", 4);
  kernel->num_stages = json_data.value("num_stages", 2);
  kernel->block_x = kernel->num_warps * 32;
  kernel->block_y = 1;
  kernel->block_z = 1;
  kernel->grid_y = 1;
  kernel->grid_z = 1;
  kernel->kernel_name = kernel_name;
  return kernel;
}

CUresult TritonWrapper::LoadTritonKernelFromFile(const std::string& file_path,
                                                 std::shared_ptr<TritonKernel>& triton_kernel) {
  fs::path file_full_path = fs::absolute(fs::path(file_path));
  KLLM_LOG_DEBUG << fmt::format("Try to load kernel {} from {}", triton_kernel->kernel_name, file_full_path.c_str());
  CUDA_CHECK_WITH_LOG(cuModuleLoad(&triton_kernel->module, file_full_path.c_str()),
                      fmt::format("cuModuleLoad failed. filename: {} ", file_full_path.c_str()));
  CUDA_CHECK_WITH_LOG(
      cuModuleGetFunction(&triton_kernel->kernel, triton_kernel->module, triton_kernel->kernel_name.c_str()),
      fmt::format("cuModuleGetFunction failed. kernel name: {} ", triton_kernel->kernel_name.c_str()));
  CUDA_CHECK_WITH_LOG(cuFuncSetAttribute(triton_kernel->kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                         triton_kernel->shm_size),
                      fmt::format("cuFuncSetAttribute failed. shm_size: {} ", triton_kernel->shm_size));
  KLLM_LOG_DEBUG << fmt::format("Load triton kernel from {} successd.", file_full_path.c_str());
  return CUDA_SUCCESS;
}

void TritonWrapper::InvokeTritonKernel(const std::string& kernel_base_name, void* args[],
                                       const std::unordered_map<std::string, std::string>& map, size_t grid_x,
                                       size_t grid_y, size_t grid_z, cudaStream_t stream) {
  const auto kernel_name = ConstructKernelName(kernel_base_name, map);
  if (!kernel_name) {
    KLLM_THROW(fmt::format("Failed to get kernel file name by kernel name {}.", kernel_base_name));
    return;
  }

  const std::string kernel_stream_name = fmt::format("{}_Stream_{}", *kernel_name, *reinterpret_cast<int64_t*>(stream));
  std::shared_ptr<TritonKernel> kernel;
  if (kernel_map_.find(kernel_stream_name) == kernel_map_.end()) {
    std::string config_path = fmt::format("{}/{}.json", kernels_dir_, *kernel_name);
    auto triton_kernel = LoadTritonKernelJsonFile(config_path, kernel_base_name);
    if (!triton_kernel) {
      KLLM_LOG_ERROR << fmt::format("Failed to load triton kernel json file. {}", config_path);
      return;
    }

    std::string cubin_path = fmt::format("{}/{}.cubin", kernels_dir_, *kernel_name);
    CUresult result = LoadTritonKernelFromFile(cubin_path, *triton_kernel);
    if (result != CUDA_SUCCESS) {
      KLLM_LOG_ERROR << fmt::format("Could not load kernel {}", cubin_path);
      return;
    }
    kernel_map_[kernel_stream_name] = *triton_kernel;
  }
  kernel = kernel_map_[kernel_stream_name];
  CUDA_CHECK_WITH_LOG(
      cuLaunchKernel(kernel->kernel, grid_x, grid_y, grid_z,             // Grid dimensions
                     kernel->block_x, kernel->block_y, kernel->block_z,  // Block dimensions
                     kernel->shm_size,                                   // Shared memory size
                     stream,                                             // Stream
                     args,                                               // Kernel parameters
                     nullptr),
      fmt::format("cuLaunchKernel failed. kernel_stream_name: {}, grid_x={} ,grid_y={}, grid_z={}, "
                  "kernel->block_x={}, kernel->block_y={}, kernel->block_z={}, kernel->shm_size={}",
                  kernel_stream_name, grid_x, grid_y, grid_z, kernel->block_x, kernel->block_y, kernel->block_z,
                  kernel->shm_size));  // Extra options
}

template <typename T>
void TritonWrapper::InvokeFusedMoeKernel(void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights,
                                         void* sorted_token_ids, void* expert_ids, void* num_tokens_post_padded, int n,
                                         int k, int max_num_tokens_padded, int numel, int a_stride0, int a_stride1,
                                         int b_stride0, int b_stride2, int b_stride1, int c_stride1, int c_stride2,
                                         int a_scale_stride0, int a_scale_stride1, int b_scale_stride0,
                                         int b_scale_stride2, int b_scale_stride1, int block_shape0, int block_shape1,
                                         bool mul_routed_weight, int top_k, bool use_fp8_w8a8, bool use_int8_w8a16,
                                         std::unordered_map<std::string, int> config, cudaStream_t stream) {
  std::string kernel_name = "fused_moe_kernel";

  void* args[] = {
      &a,
      &b,
      &c,
      &a_scale,
      &b_scale,
      &topk_weights,
      &sorted_token_ids,
      &expert_ids,
      &num_tokens_post_padded,
      &n,
      &k,
      &max_num_tokens_padded,
      &numel,
  };
  std::unordered_map<std::string, std::string> map = {{"group_n", ConvertToString(block_shape0)},
                                                      {"group_k", ConvertToString(block_shape1)},
                                                      {"BLOCK_SIZE_M", ConvertToString(config["block_size_m"])},
                                                      {"BLOCK_SIZE_N", ConvertToString(config["block_size_n"])},
                                                      {"BLOCK_SIZE_K", ConvertToString(config["block_size_k"])},
                                                      {"GROUP_SIZE_M", ConvertToString(config["group_size_m"])},
                                                      {"MUL_ROUTED_WEIGHT", ConvertToString(mul_routed_weight)},
                                                      {"top_k", ConvertToString(top_k)},
                                                      {"compute_type", GetComputeType<T>()},
                                                      {"use_fp8_w8a8", ConvertToString(use_fp8_w8a8)},
                                                      {"use_int8_w8a16", ConvertToString(use_int8_w8a16)}};
  map["even_Ks"] = (k % config["block_size_k"] == 0) ? "True" : "False";
  size_t grid_x = CeilDiv(max_num_tokens_padded, config.at("block_size_m")) * CeilDiv(n, config.at("block_size_n"));
  InvokeTritonKernel(kernel_name, args, map, grid_x, 1, 1, stream);
}
#define INVOKE_FUSED_MOE_KERNEL(T)                                                                                  \
  template void TritonWrapper::InvokeFusedMoeKernel<T>(                                                             \
      void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights, void* sorted_token_ids,          \
      void* expert_ids, void* num_tokens_post_padded, int n, int k, int max_num_tokens_padded, int numel,           \
      int a_stride0, int a_stride1, int b_stride0, int b_stride2, int b_stride1, int c_stride1, int c_stride2,      \
      int a_scale_stride0, int a_scale_stride1, int b_scale_stride0, int b_scale_stride2, int b_scale_stride1,      \
      int block_shape0, int block_shape1, bool mul_routed_weight, int topk, bool use_fp8_w8a8, bool use_int8_w8a16, \
      std::unordered_map<std::string, int> config, cudaStream_t stream)
INVOKE_FUSED_MOE_KERNEL(float);
INVOKE_FUSED_MOE_KERNEL(half);
INVOKE_FUSED_MOE_KERNEL(__nv_bfloat16);
#undef INVOKE_FUSED_MOE_KERNEL

template <typename T>
void TritonWrapper::InvokeFusedMoeGptqAwqKernel(
    void* a, void* b, void* c, void* b_scale, void* b_zp, void* topk_weights, void* sorted_token_ids, void* expert_ids,
    void* num_tokens_post_padded, int n, int k, int max_num_tokens_padded, int numel, int a_stride0, int a_stride1,
    int b_stride0, int b_stride2, int b_stride1, int c_stride1, int c_stride2, int b_scale_stride0, int b_scale_stride2,
    int b_scale_stride1, int b_zp_stride0, int b_zp_stride2, int b_zp_stride1, bool mul_routed_weight, int top_k,
    bool has_zp, int weight_bits, int group_size, std::unordered_map<std::string, int> config, cudaStream_t stream) {
  std::string kernel_name = "fused_moe_gptq_awq_kernel";

  void* args[] = {
      &a,
      &b,
      &c,
      &b_scale,
      &b_zp,
      &topk_weights,
      &sorted_token_ids,
      &expert_ids,
      &num_tokens_post_padded,
      &n,
      &k,
      &max_num_tokens_padded,
      &numel,
  };
  std::unordered_map<std::string, std::string> map = {{"BLOCK_SIZE_M", ConvertToString(config["block_size_m"])},
                                                      {"BLOCK_SIZE_N", ConvertToString(config["block_size_n"])},
                                                      {"BLOCK_SIZE_K", ConvertToString(config["block_size_k"])},
                                                      {"GROUP_SIZE_M", ConvertToString(config["group_size_m"])},
                                                      {"MUL_ROUTED_WEIGHT", ConvertToString(mul_routed_weight)},
                                                      {"top_k", ConvertToString(top_k)},
                                                      {"compute_type", GetComputeType<T>()},
                                                      {"has_zp", ConvertToString(has_zp)},
                                                      {"weight_bits", ConvertToString(weight_bits)},
                                                      {"group_size", ConvertToString(group_size)}};
  size_t grid_x = CeilDiv(max_num_tokens_padded, config.at("block_size_m")) * CeilDiv(n, config.at("block_size_n"));
  InvokeTritonKernel(kernel_name, args, map, grid_x, 1, 1, stream);
}
#define INVOKE_FUSED_MOE_GPTQ_AWQ_KERNEL(T)                                                                    \
  template void TritonWrapper::InvokeFusedMoeGptqAwqKernel<T>(                                                 \
      void* a, void* b, void* c, void* b_scale, void* b_zp, void* topk_weights, void* sorted_token_ids,        \
      void* expert_ids, void* num_tokens_post_padded, int n, int k, int max_num_tokens_padded, int numel,      \
      int a_stride0, int a_stride1, int b_stride0, int b_stride2, int b_stride1, int c_stride1, int c_stride2, \
      int b_scale_stride0, int b_scale_stride2, int b_scale_stride1, int b_zp_stride0, int b_zp_stride2,       \
      int b_zp_stride1, bool mul_routed_weight, int top_k, bool has_zp, int weight_bits, int group_size,       \
      std::unordered_map<std::string, int> config, cudaStream_t stream)
INVOKE_FUSED_MOE_GPTQ_AWQ_KERNEL(float);
INVOKE_FUSED_MOE_GPTQ_AWQ_KERNEL(half);
INVOKE_FUSED_MOE_GPTQ_AWQ_KERNEL(__nv_bfloat16);
#undef INVOKE_FUSED_MOE_GPTQ_AWQ_KERNEL

template <typename T>
void TritonWrapper::InvokeFusedMoeGptqInt4Fp8Kernel(void* a, void* b, void* c, void* a_scale, void* b_scale,
                                                    void* topk_weights, void* sorted_token_ids, void* expert_ids,
                                                    void* num_tokens_post_padded, int n, int k,
                                                    int max_num_tokens_padded, int numel, bool mul_routed_weight,
                                                    int top_k, int group_size,
                                                    std::unordered_map<std::string, int> config, cudaStream_t stream) {
  std::string kernel_name = "fused_moe_gptq_int4_fp8_kernel";

  void* args[] = {
      &a,
      &b,
      &c,
      &a_scale,
      &b_scale,
      &topk_weights,
      &sorted_token_ids,
      &expert_ids,
      &num_tokens_post_padded,
      &n,
      &k,
      &max_num_tokens_padded,
      &numel,
  };
  std::unordered_map<std::string, std::string> map = {{"BLOCK_SIZE_M", ConvertToString(config["block_size_m"])},
                                                      {"BLOCK_SIZE_N", ConvertToString(config["block_size_n"])},
                                                      {"BLOCK_SIZE_K", ConvertToString(config["block_size_k"])},
                                                      {"GROUP_SIZE_M", ConvertToString(config["group_size_m"])},
                                                      {"MUL_ROUTED_WEIGHT", ConvertToString(mul_routed_weight)},
                                                      {"top_k", ConvertToString(top_k)},
                                                      {"compute_type", GetComputeType<T>()},
                                                      {"group_size", ConvertToString(group_size)}};
  size_t grid_x = CeilDiv(max_num_tokens_padded, config.at("block_size_m")) * CeilDiv(n, config.at("block_size_n"));
  InvokeTritonKernel(kernel_name, args, map, grid_x, 1, 1, stream);
}
#define INVOKE_FUSED_MOE_GPTQ_INT4_FP8_KERNEL(T)                                                           \
  template void TritonWrapper::InvokeFusedMoeGptqInt4Fp8Kernel<T>(                                         \
      void* a, void* b, void* c, void* a_scale, void* b_scale, void* topk_weights, void* sorted_token_ids, \
      void* expert_ids, void* num_tokens_post_padded, int n, int k, int max_num_tokens_padded, int numel,  \
      bool mul_routed_weight, int top_k, int group_size, std::unordered_map<std::string, int> config,      \
      cudaStream_t stream)
INVOKE_FUSED_MOE_GPTQ_INT4_FP8_KERNEL(float);
INVOKE_FUSED_MOE_GPTQ_INT4_FP8_KERNEL(half);
INVOKE_FUSED_MOE_GPTQ_INT4_FP8_KERNEL(__nv_bfloat16);
#undef INVOKE_FUSED_MOE_GPTQ_INT4_FP8_KERNEL

}  // namespace ksana_llm
