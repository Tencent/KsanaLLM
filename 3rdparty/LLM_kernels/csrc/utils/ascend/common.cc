/**
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "common.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <numeric>

#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_copy.h"
#include "aclnnop/aclnn_random.h"

#include "3rdparty/ini_reader.h"

#include "csrc/utils/common.h"

namespace llm_kernels {
namespace utils {

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

void CalShapeStrides(const std::vector<int64_t>& shape, std::vector<int64_t>& strides) {
  strides.resize(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
}

void ReallocWorkspace(const uint64_t new_size, uint64_t& old_size, void** ws_dev, aclrtMemMallocPolicy policy) {
  if (new_size > old_size) {
    if (old_size > 0) {
      ACL_CHECK_RET(aclrtFree(*ws_dev));
      *ws_dev = nullptr;
    }
    old_size = new_size;
    ACL_CHECK_RET(aclrtMalloc(ws_dev, new_size, policy));
  }
}

void CreateAclTensor(const int64_t hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                     aclFormat dataFormat, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * SizeOfAclDataType.at(dataType);
  ACL_CHECK_RET(aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(*deviceAddr, size, &hostData, size, ACL_MEMCPY_HOST_TO_DEVICE));
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat, shape.data(),
                            shape.size(), *deviceAddr);
}

void CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclFormat dataFormat,
                     aclTensor** tensor) {
  auto size = GetShapeSize(shape) * SizeOfAclDataType.at(dataType);
  ACL_CHECK_RET(aclrtMalloc(deviceAddr, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat, shape.data(),
                            shape.size(), *deviceAddr);
}

void CreateAclTensorWithData(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                             aclFormat dataFormat, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * SizeOfAclDataType.at(dataType);
  std::vector<int64_t> strides;
  CalShapeStrides(shape, strides);
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat, shape.data(),
                            shape.size(), *deviceAddr);
}

std::string GetNumpyTypeDesc(aclDataType dtype) {
  switch (dtype) {
    case aclDataType::ACL_BF16:
      std::cerr << "GetNumpyTypeDesc(Bfloat16) returns an invalid type 'x' since Numpy doesn't "
                   "support bfloat16 as of now, it will be properly extended if numpy supports. "
                   "Please refer for the discussions https://github.com/numpy/numpy/issues/19808."
                << std::endl;
      return "x";
    case aclDataType::ACL_BOOL:
      return "?";
    case aclDataType::ACL_UINT8:
      return "u1";
    case aclDataType::ACL_UINT16:
      return "u2";
    case aclDataType::ACL_UINT32:
      return "u4";
    case aclDataType::ACL_UINT64:
      return "u8";
    case aclDataType::ACL_INT8:
      return "i1";
    case aclDataType::ACL_INT16:
      return "i2";
    case aclDataType::ACL_INT32:
      return "i4";
    case aclDataType::ACL_INT64:
      return "i8";
    case aclDataType::ACL_FLOAT16:
      return "f2";
    case aclDataType::ACL_FLOAT:
      return "f4";
    case aclDataType::ACL_DOUBLE:
      return "f8";
    default:
      return "x";
  }
}

template <typename T>
void SaveNpyFromPtr(const std::string& numpy_type, const std::vector<T>& tensor_shape, const size_t dtype_size,
                    void* data_ptr, const std::string& filename) {
  size_t elem_nums = 1ul;
  for (uint64_t i = 0; i < tensor_shape.size(); ++i) {
    elem_nums *= tensor_shape[i];
  }

  const char magic[] =
      "\x93"
      "NUMPY";
  const uint8_t npy_major = 1;
  const uint8_t npy_minor = 0;

  std::stringstream header_stream;
  header_stream << "{'descr': '" << numpy_type << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < tensor_shape.size(); ++i) {
    header_stream << tensor_shape[i];
    if (i + 1 < tensor_shape.size() || tensor_shape.size() == 1) {
      header_stream << ", ";
    }
  }
  header_stream << ")}";
  int32_t base_length = 6 + 4 + header_stream.str().size();
  int32_t pad_length = 16 * ((base_length + 1 + 15) / 16);  // Take ceiling of base_length + 1 (for '\n' ending)
  for (int32_t i = 0; i < pad_length - base_length; ++i) {
    header_stream << ((i == pad_length - base_length - 1) ? "\n" : "\x20");
  }
  std::string header = header_stream.str();
  const uint16_t header_len = header.size();

  FILE* f_ptr = fopen(filename.c_str(), "wb");
  if (f_ptr == nullptr) {
    std::cerr << "Unable to open " << filename << " for writing." << std::endl;
  }
  fwrite(magic, sizeof(char), sizeof(magic) - 1, f_ptr);
  fwrite(&npy_major, sizeof(uint8_t), 1, f_ptr);
  fwrite(&npy_minor, sizeof(uint8_t), 1, f_ptr);
  fwrite(&header_len, sizeof(uint16_t), 1, f_ptr);
  fwrite(header.c_str(), sizeof(char), header_len, f_ptr);
  fwrite(data_ptr, dtype_size, elem_nums, f_ptr);
  fclose(f_ptr);
}

template void SaveNpyFromPtr(const std::string& numpy_type, const std::vector<int64_t>& tensor_shape,
                             const size_t dtype_size, void* data_ptr, const std::string& filename);
template void SaveNpyFromPtr(const std::string& numpy_type, const std::vector<size_t>& tensor_shape,
                             const size_t dtype_size, void* data_ptr, const std::string& filename);

template <typename T>
void LoadNpyToPtr(const std::string& filename, T* data_ptr, std::vector<size_t>& tensor_shape, bool is_on_host) {
  std::vector<size_t> shape;
  FILE* f_ptr = fopen(filename.c_str(), "rb");
  if (f_ptr == nullptr) {
    throw std::runtime_error("Could not open file " + filename);
  }
  uint32_t header_len, start_data;
  ParseNpyIntro(f_ptr, header_len, start_data);
  ParseNpyHeader(f_ptr, header_len, shape);

  const size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  void* data_cpu = malloc(size * sizeof(T));
  void* data = data_cpu;

  size_t n_elems = fread(data_cpu, sizeof(T), size, f_ptr);
  if (n_elems != size) {
    throw std::runtime_error("reading tensor failed");
  }
  if (!is_on_host) {
    ACL_CHECK_RET(aclrtMemcpy(reinterpret_cast<void*>(data_ptr), size * sizeof(T), data_cpu, size * sizeof(T),
                              ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK_RET(aclrtSynchronizeDevice());
  } else {
    std::memcpy(reinterpret_cast<void*>(data_ptr), data_cpu, size * sizeof(T));
  }
  free(data_cpu);

  fclose(f_ptr);
  tensor_shape = shape;
}

template void LoadNpyToPtr(const std::string& filename, aclFloat16* data_ptr, std::vector<size_t>& tensor_shape,
                           bool is_on_host);
template void LoadNpyToPtr(const std::string& filename, float* data_ptr, std::vector<size_t>& tensor_shape,
                           bool is_on_host);
template void LoadNpyToPtr(const std::string& filename, int32_t* data_ptr, std::vector<size_t>& tensor_shape,
                           bool is_on_host);
template void LoadNpyToPtr(const std::string& filename, int64_t* data_ptr, std::vector<size_t>& tensor_shape,
                           bool is_on_host);
template void LoadNpyToPtr(const std::string& filename, uint64_t* data_ptr, std::vector<size_t>& tensor_shape,
                           bool is_on_host);

void SaveNpy(const aclTensor* tensor, const void* tensor_workspace_ptr, const std::string& filename,
             aclrtStream& stream, bool is_on_device) {
  aclDataType acl_dtype;
  ACL_CHECK_RET(aclGetDataType(tensor, &acl_dtype));
  int64_t* shape = nullptr;
  uint64_t dims = 0;
  ACL_CHECK_RET(aclGetViewShape(tensor, &shape, &dims));
  size_t elem_nums = dims == 0ul ? 0ul : 1ul;
  std::vector<int64_t> tensor_shape(dims);
  for (uint64_t i = 0; i < dims; ++i) {
    elem_nums *= shape[i];
    tensor_shape[i] = shape[i];
  }

  void* host_buffer_ptr = nullptr;
  if (is_on_device) {
    ACL_CHECK_RET(aclrtMallocHost(&host_buffer_ptr, elem_nums * aclDataTypeSize(acl_dtype)));
    ACL_CHECK_RET(aclrtMemcpy(host_buffer_ptr, elem_nums * aclDataTypeSize(acl_dtype), tensor_workspace_ptr,
                              elem_nums * aclDataTypeSize(acl_dtype), ACL_MEMCPY_DEVICE_TO_HOST));
  } else {
    host_buffer_ptr = (void*)tensor_workspace_ptr;
  }

  std::string numpy_type_str = GetNumpyTypeDesc(acl_dtype);
  SaveNpyFromPtr(numpy_type_str, tensor_shape, aclDataTypeSize(acl_dtype), host_buffer_ptr, filename);
  if (is_on_device) {
    ACL_CHECK_RET(aclrtFreeHost(host_buffer_ptr));
  }
}

void GetTestWorkSpaceFunc(size_t size, void** ws_addr) {
  static void* cached_addr = nullptr;
  static size_t cached_size = 0;

  if (size > 0) {
    if (cached_size < size) {
      if (cached_addr != nullptr) {
        aclrtFree(cached_addr);
      }
      ACL_CHECK_RET(aclrtMalloc(&cached_addr, size, ACL_MEM_MALLOC_HUGE_FIRST));
      cached_size = size;
    }

    *ws_addr = cached_addr;
  }
}

void LoadDeviceAttribute(const std::string& platform_config_path, AscendNPUDeviceAttribute& device_attr) {
  INIReader ini_reader = INIReader(platform_config_path);
  if (ini_reader.ParseError() < 0) {
    throw std::runtime_error(platform_config_path + " Device Attribute parse error");
  }

  device_attr.ai_core_num = ini_reader.GetInteger("SoCInfo", "ai_core_cnt");
  device_attr.cube_core_num = ini_reader.GetInteger("SoCInfo", "cube_core_cnt");
  device_attr.vector_core_num = ini_reader.GetInteger("SoCInfo", "vector_core_cnt");
  device_attr.ai_cpu_num = ini_reader.GetInteger("SoCInfo", "ai_cpu_cnt");
  device_attr.l2_size = ini_reader.GetInteger("SoCInfo", "l2_size");
  device_attr.l0_a_size = ini_reader.GetInteger("AICoreSpec", "l0_a_size");
  device_attr.l0_b_size = ini_reader.GetInteger("AICoreSpec", "l0_b_size");
  device_attr.l0_c_size = ini_reader.GetInteger("AICoreSpec", "l0_c_size");
  device_attr.l1_size = ini_reader.GetInteger("AICoreSpec", "l1_size");
}

}  // namespace utils
}  // namespace llm_kernels
