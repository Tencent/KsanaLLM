/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <assert.h>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"
#include "atb/types.h"

namespace llm_kernels {
namespace utils {

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)
#define FATAL_LOG(fmt, args...)                  \
  fprintf(stderr, "[FATAL]  " fmt "\n", ##args); \
  assert(false)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

template <typename T>
void InnerCheckACLError(T result, const char* func, const char* file, const int line) {
  if (result != ACL_SUCCESS) {
    std::ostringstream ss;
    ss << "ACL runtime error code: " << result << std::endl << file << ":" << line << "@" << func;
    throw std::runtime_error(ss.str());
  }
}

#define ACL_CHECK_RET(val) llm_kernels::utils::InnerCheckACLError((val), #val, __FILE__, __LINE__)

static const std::unordered_map<aclDataType, size_t> SizeOfAclDataType = {
    {aclDataType::ACL_FLOAT16, sizeof(uint16_t)}, {aclDataType::ACL_INT64, sizeof(int64_t)},
    {aclDataType::ACL_FLOAT, sizeof(float)},      {aclDataType::ACL_BOOL, sizeof(bool)},
    {aclDataType::ACL_UINT8, sizeof(uint8_t)},    {aclDataType::ACL_BF16, sizeof(uint16_t)}};

int64_t GetShapeSize(const std::vector<int64_t>& shape);

void CalShapeStrides(const std::vector<int64_t>& shape, std::vector<int64_t>& strides);

void ReallocWorkspace(const uint64_t new_size, uint64_t& old_size, void** ws_dev,
                      aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_NORMAL_ONLY);

void CreateAclTensor(const int64_t hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                     aclFormat dataFormat, aclTensor** tensor);

void CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclFormat dataFormat,
                     aclTensor** tensor);

void CreateAclTensorWithData(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                             aclFormat dataFormat, aclTensor** tensor);

void SaveNpy(const aclTensor* tensor, const void* tensor_workspace_ptr, const std::string& filename,
             aclrtStream& stream, bool is_on_device = true);

template <typename T>
void SaveNpyFromPtr(const std::string& numpy_type, const std::vector<T>& tensor_shape, const size_t dtype_size,
                    void* data_ptr, const std::string& filename);

template <typename T>
void LoadNpyToPtr(const std::string& filename, T* data_ptr, std::vector<size_t>& tensor_shape, bool is_on_host = true);

std::string GetNumpyTypeDesc(aclDataType dtype);

// Define a function to create kernel workspace for test
typedef void (*WorkSpaceFunc)(size_t, void**);

// Define a function to create kernel workspace for test
void GetTestWorkSpaceFunc(size_t size, void** ws_addr);

struct AscendNPUDeviceAttribute {
  // NOTE(karlluo): all these config is needed for NPU performance optimization
  uint32_t ai_core_num{0};
  uint32_t cube_core_num{0};
  uint32_t vector_core_num{0};
  uint32_t ai_cpu_num{0};
  size_t l2_size{0};
  size_t l0_a_size{0};
  size_t l0_b_size{0};
  size_t l0_c_size{0};
  size_t l1_size{0};
};

// Loading hardware from local config
void LoadDeviceAttribute(const std::string& platform_config_path, AscendNPUDeviceAttribute& device_attr);

// ACLNNMatmulComputeType(int8_t, Calculation Input): Integer type on the Host side, determines which calculation logic
// the Cube unit uses for operations. The data type supports INT8, and the supported enumeration values are as follows:
// 0: KEEP_DTYPE - Keep the input data type for calculation. When the input is FLOAT, the Cube calculation unit of the
// Atlas training series products does not support it. An error will occur if 0 is selected. 1:
// ALLOW_FP32_DOWN_PRECISION - Allow the input data to be downcast for calculation. When the input is FLOAT, the Atlas
// training series products convert it to FLOAT16 for calculation, and the Atlas A2 training series products convert it
// to HFLOAT32 for calculation. 2: USE_FP16 - Allow conversion to the data type FLOAT16 for calculation. When the input
// data type is FLOAT, it is converted to FLOAT16 for calculation. 3: USE_HF32 - Allow conversion to the data type
// HFLOAT32 for calculation. When the input is FLOAT, the Cube calculation unit of the Atlas training series products
// does not support it. An error will occur if 3 is selected. The Atlas A2 training series products convert it to
// HFLOAT32 for calculation.
enum ACLNNMatmulComputeType { KEEP_DTYPE = 0, ALLOW_FP32_DOWN_PRECISION = 1, USE_FP16 = 2, USE_HF32 = 3 };

__attribute__((unused)) static const std::string GetATBErrorString(atb::ErrorType error) {
  static const std::unordered_map<atb::ErrorType, std::string> error_to_string_map{
      {atb::NO_ERROR, "NO_ERROR"},
      {atb::ERROR_INVALID_PARAM, "ERROR_INVALID_PARAM"},
      {atb::ERROR_INVALID_GRAPH, "ERROR_INVALID_GRAPH"},
      {atb::ERROR_INTERNAL_ERROR, "ERROR_INTERNAL_ERROR"},
      {atb::ERROR_RT_FAIL, "ERROR_RT_FAIL"},
      {atb::ERROR_INVALID_IN_TENSOR_NUM, "ERROR_INVALID_IN_TENSOR_NUM"},
      {atb::ERROR_INVALID_TENSOR_DTYPE, "ERROR_INVALID_TENSOR_DTYPE"},
      {atb::ERROR_INVALID_TENSOR_FORMAT, "ERROR_INVALID_TENSOR_FORMAT"},
      {atb::ERROR_INVALID_TENSOR_DIM, "ERROR_INVALID_TENSOR_DIM"},
      {atb::ERROR_INVALID_TENSOR_SIZE, "ERROR_INVALID_TENSOR_SIZE"},
      {atb::ERROR_OPERATION_NULL_RUNNER, "ERROR_OPERATION_NULL_RUNNER"},
      {atb::ERROR_GRAPH_INFERSHAPE_FUNC_FAIL, "ERROR_GRAPH_INFERSHAPE_FUNC_FAIL"},
      {atb::ERROR_CANN_ERROR, "ERROR_CANN_ERROR"},
      {atb::ERROR_INVALID_TENSOR_INI_MATCH, "ERROR_INVALID_TENSOR_INI_MATCH"}};
  if (error_to_string_map.count(error) != 0ul) {
    return error_to_string_map.at(error);
  } else {
    return "UNKOWN, refer: "
           "https://www.hiascend.com/document/detail/zh/mindie/1.0.RC1/mindiert/rtdev/ascendtb_01_0008.html";
  }
}

template <typename T>
void CheckATBError(T result, const char* func, const char* file, const int line) {
  if (result != atb::NO_ERROR) {
    std::ostringstream ss;
    ss << "ATB runtime error code: " << result << std::endl
       << GetATBErrorString(result) << std::endl
       << file << ":" << line << "@" << func;
    throw std::runtime_error(ss.str());
  }
}

#define ATB_CHECK_RET(val) \
  llm_kernels::utils::CheckATBError<atb::ErrorType>(static_cast<atb::ErrorType>(val), #val, __FILE__, __LINE__)
}  // namespace utils
}  // namespace llm_kernels
