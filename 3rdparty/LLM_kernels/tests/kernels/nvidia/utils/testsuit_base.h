/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "3rdparty/half/include/half.hpp"
#include "csrc/utils/nvidia/cuda_utils.h"

#include "csrc/utils/common.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace test {

typedef enum memorytype_enum { MEMORY_CPU, MEMORY_CPU_PINNED, MEMORY_GPU } MemoryType;

struct BufferMeta {
  void* data_ptr{nullptr};
  std::vector<size_t> shape;
  size_t n_elmts{0};
  size_t buf_size{0};
  MemoryType memory_type;

  template <typename T>
  std::string GetNumpyTypeDesc() const {
    if (std::is_same<T, __nv_bfloat16>::value) {
      std::cerr << "GetNumpyTypeDesc(Bfloat16) returns an invalid type 'x' since Numpy doesn't "
                   "support bfloat16 as of now, it will be properly extended if numpy supports. "
                   "Please refer for the discussions https://github.com/numpy/numpy/issues/19808."
                << std::endl;
      return "x";
    } else if (std::is_same<T, bool>::value) {
      return "?";
    } else if (std::is_same<T, uint8_t>::value) {
      return "u1";
    } else if (std::is_same<T, uint16_t>::value) {
      return "u2";
    } else if (std::is_same<T, uint32_t>::value) {
      return "u4";
    } else if (std::is_same<T, uint64_t>::value) {
      return "u8";
    } else if (std::is_same<T, int8_t>::value) {
      return "i1";
    } else if (std::is_same<T, int16_t>::value) {
      return "i2";
    } else if (std::is_same<T, int32_t>::value) {
      return "i4";
    } else if (std::is_same<T, int64_t>::value) {
      return "i8";
    } else if (std::is_same<T, half>::value) {
      return "f2";
    } else if (std::is_same<T, float>::value) {
      return "f4";
    } else if (std::is_same<T, double>::value) {
      return "f8";
    } else {
      // others will return invalid
      return "x";
    }
  }

  size_t GetElementNum() const {
    if (data_ptr == nullptr || shape.size() == 0) {
      return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
  }

  template <typename T>
  void SaveNpy(const std::string& filename) const {
    // Save tensor to NPY 1.0 format (see https://numpy.org/neps/nep-0001-npy-format.html)
    void* cpu_data = reinterpret_cast<void*>(data_ptr);
    bool is_data_temp = false;
    size_t tensor_size = GetElementNum();
    if (memory_type == MemoryType::MEMORY_GPU) {
      cpu_data = malloc(tensor_size * sizeof(T));
      is_data_temp = true;
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(cpu_data, data_ptr, tensor_size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    const char magic[] =
        "\x93"
        "NUMPY";
    const uint8_t npy_major = 1;
    const uint8_t npy_minor = 0;

    std::stringstream header_stream;
    header_stream << "{'descr': '" << GetNumpyTypeDesc<T>() << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
      header_stream << shape[i];
      if (i + 1 < shape.size() || shape.size() == 1) {
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
    fwrite(cpu_data, sizeof(T), tensor_size, f_ptr);

    fclose(f_ptr);

    if (is_data_temp) {
      free(cpu_data);
    }
  }

  // Since NumPy lacks native bf16 support, we store bf16 data as float16 (same binary representation, different type
  // interpretation). Note: When loading such npy files, you must reinterpret the data to the correct type.
  template <typename T>
  void SaveToNpy(const std::string& npy_name) {
    if (std::is_same<T, __nv_bfloat16>::value) {
      SaveNpy<half>(npy_name);
    } else {
      SaveNpy<T>(npy_name);
    }
  }

  inline void ParseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data) {
    const char magic[] =
        "\x93"
        "NUMPY";
    char magic_test[sizeof(magic)] = "\0";

    size_t n_elems = fread((void*)magic_test, sizeof(char), sizeof(magic) - 1, f_ptr);
    if (n_elems != sizeof(magic) - 1 || std::string(magic) != std::string(magic_test)) {
      throw std::runtime_error("Could read magic token in NPY file");
    }

    uint8_t npy_major = 0;
    uint8_t npy_minor = 0;
    n_elems = fread((void*)&npy_major, sizeof(uint8_t), 1, f_ptr);
    n_elems += fread((void*)&npy_minor, sizeof(uint8_t), 1, f_ptr);

    if (npy_major == 1) {
      uint16_t header_len_u16 = 0;
      n_elems = fread((void*)&header_len_u16, sizeof(uint16_t), 1, f_ptr);
      header_len = header_len_u16;
    } else if (npy_major == 2) {
      uint32_t header_len_u32 = 0;
      n_elems = fread((void*)&header_len_u32, sizeof(uint32_t), 1, f_ptr);
      header_len = header_len_u32;
    } else {
      throw std::runtime_error("Unsupported npy version: " + std::to_string(npy_major));
    }

    start_data = 8 + 2 * npy_major + header_len;
  }

  inline int32_t ParseNpyHeader(FILE*& f_ptr, uint32_t header_len, std::vector<size_t>& shape) {
    char* header_c = (char*)malloc(header_len * sizeof(char));
    size_t n_elems = fread((void*)header_c, sizeof(char), header_len, f_ptr);
    if (n_elems != header_len) {
      free(header_c);
      return -1;
    }
    std::string header(header_c, header_len);
    free(header_c);

    size_t start, end;
    start = header.find("'descr'") + 7;
    start = header.find("'", start);
    end = header.find("'", start + 1);

    start = header.find("'fortran_order'") + 15;
    start = header.find(":", start);
    end = header.find(",", start + 1);
    if (header.substr(start + 1, end - start - 1).find("False") == std::string::npos) {
      throw std::runtime_error("Unsupported value for fortran_order while reading npy file");
    }

    start = header.find("'shape'") + 7;
    start = header.find("(", start);
    end = header.find(")", start + 1);

    std::istringstream shape_stream(header.substr(start + 1, end - start - 1));
    std::string token;

    shape.clear();
    while (std::getline(shape_stream, token, ',')) {
      if (token.find_first_not_of(' ') == std::string::npos) {
        break;
      }
      shape.push_back(std::stoul(token));
    }

    return 0;
  }

  template <typename T>
  inline void LoadNpy(const std::string& npy_file, const MemoryType dst_memory = MEMORY_CPU) {
    std::vector<size_t> loaded_data_shape;

    FILE* f_ptr = fopen(npy_file.c_str(), "rb");
    if (f_ptr == nullptr) {
      throw std::runtime_error("Could not open file " + npy_file);
    }
    uint32_t header_len, start_data;
    ParseNpyIntro(f_ptr, header_len, start_data);
    ParseNpyHeader(f_ptr, header_len, loaded_data_shape);

    const size_t size =
        std::accumulate(loaded_data_shape.begin(), loaded_data_shape.end(), 1, std::multiplies<size_t>());
    void* data_cpu = malloc(size * sizeof(T));
    void* data = data_cpu;

    size_t n_elems = fread(data_cpu, sizeof(T), size, f_ptr);
    if (n_elems != size) {
      throw std::runtime_error("reading tensor failed");
    }
    if (dst_memory == MEMORY_GPU) {
      CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&data, size * sizeof(T)));
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(data, data_cpu, size * sizeof(T), cudaMemcpyHostToDevice));
      free(data_cpu);
    }

    fclose(f_ptr);

    data_ptr = data;
    shape = loaded_data_shape;
    n_elmts = size;
    buf_size = size * sizeof(T);
    memory_type = dst_memory;
  }
};

class NvidiaTestSuitBase : public testing::Test {
 public:
  void SetUp() override {
    CHECK_NVIDIA_CUDA_ERROR(cudaGetDevice(&device));
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
  }

  void TearDown() override {
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    // free cpu
    for (auto& buffer : allocated_cpu_buffers) {
      if (freed_cpu_buffers.find(reinterpret_cast<uintptr_t>(buffer)) == freed_cpu_buffers.end()) {
        free(reinterpret_cast<void*>(buffer));
      }
    }
    // free gpu
    for (auto& buffer : allocated_gpu_buffers) {
      if (freed_gpu_buffers.find(reinterpret_cast<uintptr_t>(buffer)) == freed_gpu_buffers.end()) {
        CHECK_NVIDIA_CUDA_ERROR(cudaFree(reinterpret_cast<void*>(buffer)));
      }
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    allocated_gpu_buffers.clear();
    allocated_cpu_buffers.clear();
    freed_gpu_buffers.clear();
    freed_cpu_buffers.clear();

    CHECK_NVIDIA_CUDA_ERROR(cudaStreamDestroy(stream));
  }

 protected:
  int32_t device{-1};
  cudaStream_t stream;
  std::set<uintptr_t> allocated_gpu_buffers;
  std::set<uintptr_t> allocated_cpu_buffers;
  std::set<uintptr_t> freed_gpu_buffers;
  std::set<uintptr_t> freed_cpu_buffers;
  size_t total_cpu_mem_used{0};
  size_t total_gpu_mem_used{0};

  size_t max_cpu_mem_used{0};
  size_t max_gpu_mem_used{0};

  void RecordMaxMemoryUsed() {
    if (total_gpu_mem_used > max_gpu_mem_used) {
      max_gpu_mem_used = total_gpu_mem_used;
    }
    if (total_cpu_mem_used > max_cpu_mem_used) {
      max_cpu_mem_used = total_cpu_mem_used;
    }
  }

  template <typename T>
  void RandomCPUBuffer(T* data_ptr, size_t n_elems, const float max_val = 1.0f, const float min_val = -1.0f) {
    for (size_t i = 0; i < n_elems; ++i) {
      if (max_val == min_val) {
        data_ptr[i] = static_cast<T>(max_val);
      } else {
        data_ptr[i] =
            static_cast<T>(min_val + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_val - min_val))));
      }
    }
  }

  // Utilities to easily handle tensor instances in test cases.
  // is_random_init == false buffer will init with zero
  template <typename T>
  BufferMeta CreateBuffer(const MemoryType mtype, const std::vector<size_t> shape, const bool is_random_init = false,
                          const float min_val = -1, const float max_val = 1) {
    size_t n_elmts = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    size_t buf_size = sizeof(T) * n_elmts;

    void* data = nullptr;
    if (mtype == MEMORY_CPU || mtype == MEMORY_CPU_PINNED) {
      data = malloc(buf_size);
      allocated_cpu_buffers.insert(reinterpret_cast<uintptr_t>(data));
      total_cpu_mem_used += buf_size;
      if (is_random_init) {
        // half does not has reinterpret_cast implement
        RandomCPUBuffer<T>((T*)(data), n_elmts, max_val, min_val);
      } else {
        memset(data, 0x0, buf_size);
      }
    } else {
      CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&data, buf_size));
      if (is_random_init) {
        RandomGPUBuffer(reinterpret_cast<T*>(data), n_elmts, max_val, min_val);
      } else {
        CHECK_NVIDIA_CUDA_ERROR(cudaMemset(data, 0x0, buf_size));
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(data));
      total_gpu_mem_used += buf_size;
    }

    BufferMeta buffer_meta;
    buffer_meta.data_ptr = data;
    buffer_meta.shape = shape;
    buffer_meta.memory_type = mtype;
    buffer_meta.n_elmts = n_elmts;
    buffer_meta.buf_size = buf_size;

    RecordMaxMemoryUsed();

    return buffer_meta;
  };

  void DeleteBuffer(BufferMeta& target_buffer) {
    if (target_buffer.data_ptr == nullptr) {
      return;
    }

    if (target_buffer.memory_type == MEMORY_CPU || target_buffer.memory_type == MEMORY_CPU_PINNED) {
      freed_cpu_buffers.insert(reinterpret_cast<uintptr_t>(target_buffer.data_ptr));
      free(target_buffer.data_ptr);
      total_cpu_mem_used -= target_buffer.buf_size;
    } else {
      freed_gpu_buffers.insert(reinterpret_cast<uintptr_t>(target_buffer.data_ptr));
      CHECK_NVIDIA_CUDA_ERROR(cudaFree(target_buffer.data_ptr));
      total_gpu_mem_used -= target_buffer.buf_size;
    }

    target_buffer.data_ptr = nullptr;
    target_buffer.buf_size = 0ul;
    target_buffer.n_elmts = 0ul;
    target_buffer.shape.clear();

    RecordMaxMemoryUsed();
  }

  template <typename T>
  BufferMeta CopyToHost(const BufferMeta buffer_meta) {
    if (buffer_meta.data_ptr == nullptr || buffer_meta.memory_type == MemoryType::MEMORY_CPU ||
        buffer_meta.memory_type == MemoryType::MEMORY_CPU_PINNED) {
      return buffer_meta;
    }
    BufferMeta host_buffer_meta = CreateBuffer<T>(MemoryType::MEMORY_CPU, buffer_meta.shape);
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(host_buffer_meta.data_ptr, buffer_meta.data_ptr, buffer_meta.buf_size, cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
    return host_buffer_meta;
  };

  template <typename T>
  BufferMeta CopyToDevice(const BufferMeta buffer_meta) {
    if (buffer_meta.data_ptr == nullptr || buffer_meta.memory_type == MemoryType::MEMORY_GPU) {
      return buffer_meta;
    }
    BufferMeta device_buffer_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, buffer_meta.shape);
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(device_buffer_meta.data_ptr, buffer_meta.data_ptr, buffer_meta.buf_size, cudaMemcpyHostToDevice));
    return device_buffer_meta;
  };

  bool AlmostEqual(float a, float b, float atol = 1e-5, float rtol = 1e-8) {
    // Params: a = value to compare and b = reference
    // This function follows implementation of numpy.isclose(), which checks
    //   abs(a - b) <= (atol + rtol * abs(b)).
    // Note that the inequality above is asymmetric where b is considered as
    // a reference value. To account into both absolute/relative errors, it
    // uses absolute tolerance and relative tolerance at the same time. The
    // default values of atol and rtol borrowed from numpy.isclose(). For the
    // case of nan value, the result will be true.
    if (isnan(a) && isnan(b)) {
      return true;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
  }

  inline float CastBFloat16ToFloat32(uint16_t src_val) {
    unsigned int tmp_vale = src_val << 16;
    return *reinterpret_cast<float*>(&tmp_vale);
  }

  template <typename T>
  bool CheckResult(std::string name, BufferMeta& out, BufferMeta& ref, float atol, float rtol,
                   float miss_match_rate = 0.01f, bool is_print_case_verbose_result = false) {
    size_t out_size = out.n_elmts;
    size_t ref_size = ref.n_elmts;
    if (out_size != ref_size) {
      std::cerr << "Target data size: " << out_size << " and refer data size " << ref_size << " is different."
                << std::endl;
      return false;
    }

    T* h_out = nullptr;
    T* h_ref = nullptr;

    if (out.memory_type == MemoryType::MEMORY_GPU) {
      h_out = reinterpret_cast<T*>(malloc(sizeof(T) * out_size));
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_out, out.data_ptr, sizeof(T) * out_size, cudaMemcpyDeviceToHost));
    } else {
      h_out = reinterpret_cast<T*>(out.data_ptr);
    }

    if (ref.memory_type == MemoryType::MEMORY_GPU) {
      h_ref = reinterpret_cast<T*>(malloc(sizeof(T) * ref_size));
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_ref, ref.data_ptr, sizeof(T) * ref_size, cudaMemcpyDeviceToHost));
    } else {
      h_ref = reinterpret_cast<T*>(ref.data_ptr);
    }
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    std::stringstream ss;

    size_t failures = 0;
    for (size_t i = 0; i < out_size; ++i) {
      float a = 0.0f;
      float b = 0.0f;
      // The values for the output and the reference.
      if (std::is_same<T, __nv_bfloat16>::value) {
        a = CastBFloat16ToFloat32(static_cast<uint16_t>(h_out[i]));
        b = CastBFloat16ToFloat32(static_cast<uint16_t>(h_ref[i]));
      } else {
        a = static_cast<float>(h_out[i]);
        b = static_cast<float>(h_ref[i]);
      }

      bool is_almost_equal = AlmostEqual(a, b, atol, rtol);
      // Print the error.
      if (is_print_case_verbose_result && !is_almost_equal) {
        ss << "invalid result for idx=" << i << ", found: " << a << ", expected: " << b
           << ", abs error: " << fabsf(a - b) << std::endl;
      }

      // Update the number of failures.
      failures += is_almost_equal ? 0 : 1;
    }

    // Allow not matched up to percent rate of elements.
    size_t tol_failures = static_cast<size_t>(miss_match_rate * out_size);
    if (failures > tol_failures && is_print_case_verbose_result) {
      std::cerr << ss.str();
      std::cerr << "Test failed in " << name << " by two tensor diff rate: " << 100. * failures / out_size
                << " with atol: " << atol << ", rtol: " << rtol << std::endl;
    }
    return failures <= tol_failures;
  }
};

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
