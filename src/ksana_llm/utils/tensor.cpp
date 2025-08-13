/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/tensor.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "3rdparty/LLM_kernels/csrc/utils/common.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/dynamic_memory_pool.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

Tensor::Tensor(const std::string& name) : name(name) {}

Tensor::Tensor(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, int device_id, void* data_ptr,
               Stream* stream, bool lazy_allocate, const std::string& name)
    : location(location),
      device_id(device_id),
      dtype(dtype),
      shape(shape),
      data_ptr(data_ptr),
      stream_(stream),
      name(name) {
  if (dtype == DataType::TYPE_INVALID) {
    // For dummy tensor, with type TYPE_INVALID, checker is disabled.
    return;
  }

  if (this->shape.empty()) {
    KLLM_THROW("Tensor could not be created with empty shape");
  }

  if (data_ptr != nullptr) {
    is_shared_buffer_ = true;
  } else if (GetElementNumber() == 0) {
    this->data_ptr = nullptr;
  } else if (!lazy_allocate || location != MemoryLocation::LOCATION_DEVICE ||
             DeviceMemoryPool::IsDisabled()) {  // Do not allocate memory if lazy_allocate enabled.
    AcquireImpl();
  }

  reference_ = std::make_shared<int>(0);
}

Tensor::~Tensor() {
  // Free underlying memory until no more tensor instances referenced.
  if (reference_.use_count() == 1) {
    ReleaseImpl();

    location = MemoryLocation::LOCATION_UNKNOWN;
    device_id = -1;

    dtype = DataType::TYPE_INVALID;
    shape.clear();
    offset = 0;
  }
}

Tensor::Tensor(const Tensor& other) { AssignMembers(other); }

Tensor& Tensor::operator=(const Tensor& other) {
  // Free underlying memory until no more tensor instances referenced.
  if (this != &other) {
    if (reference_.use_count() == 1) {
      ReleaseImpl();
    }
    AssignMembers(other);
  }
  return *this;
}

void Tensor::AcquireImpl() {
  if (data_ptr == nullptr && !is_shared_buffer_) {
    size_t total_bytes = GetTotalBytes();
    if (location == MemoryLocation::LOCATION_DEVICE) {
      if (!DeviceMemoryPool::Empty()) {
        constexpr size_t head_meta_bytes = 4096;
        constexpr size_t tail_meta_bytes = 4096;
        if (MemoryChecker::Enabled()) {
          total_bytes += (head_meta_bytes + tail_meta_bytes);
        }

        data_ptr = DeviceMemoryPool::GetMemoryPool(device_id)->Allocate(total_bytes, false);

        if (MemoryChecker::Enabled()) {
          data_ptr += head_meta_bytes;
          total_bytes -= (head_meta_bytes + tail_meta_bytes);
          std::string tensor_name = name.empty() ? std::to_string(reinterpret_cast<uintptr_t>(this)) : name;
          KLLM_LOG_DEBUG << "Add tensor " << tensor_name << " to check list.";

          SetDevice(device_id);
          Memset(data_ptr - head_meta_bytes, 0, head_meta_bytes);
          Memset(data_ptr + total_bytes, 0, tail_meta_bytes);

          MemoryChecker::AddMemoryBlock(tensor_name, device_id, data_ptr - head_meta_bytes, head_meta_bytes,
                                        data_ptr + total_bytes, tail_meta_bytes, 0);
        }
      } else {
        SetDevice(device_id);
        Malloc(&data_ptr, total_bytes);
      }
    } else if (location == MemoryLocation::LOCATION_HOST) {
      HostAlloc(&data_ptr, total_bytes);
    }
  }
}

void Tensor::Acquire() {
  // Allocate memory only if memory pool is enabled.
  if (DeviceMemoryPool::IsEnabled()) {
    AcquireImpl();
  }
}

void Tensor::ReleaseImpl() {
  if (data_ptr != nullptr && !is_shared_buffer_) {
    if (location == MemoryLocation::LOCATION_DEVICE) {
      if (!DeviceMemoryPool::Empty()) {
        constexpr size_t head_meta_bytes = 4096;
        if (MemoryChecker::Enabled()) {
          std::string tensor_name = name.empty() ? std::to_string(reinterpret_cast<uintptr_t>(this)) : name;
          KLLM_LOG_DEBUG << "Remove tensor " << tensor_name << " from check list.";
          MemoryChecker::RemoveMemoryBlock(tensor_name, device_id);
          data_ptr -= head_meta_bytes;
        }
        DeviceMemoryPool::GetMemoryPool(device_id)->Free(data_ptr);
      } else {
        SetDevice(device_id);
        Free(data_ptr);
      }
    } else if (location == MemoryLocation::LOCATION_HOST) {
      HostFree(data_ptr);
    }
    data_ptr = nullptr;
  }
}

void Tensor::Release() {
  // Release memory only if memory pool is enabled.
  if (DeviceMemoryPool::IsEnabled()) {
    ReleaseImpl();
  }
}

void Tensor::AssignMembers(const Tensor& other) {
  name = other.name;
  location = other.location;
  device_id = other.device_id;
  stream_ = other.stream_;

  dtype = other.dtype;
  shape = other.shape;
  data_format = other.data_format;
  offset = other.offset;
  data_ptr = other.data_ptr;

  is_shared_buffer_ = other.is_shared_buffer_;
  reference_ = other.reference_;

  scales = other.scales;
  zeros = other.zeros;

  g_idx = other.g_idx;
  perm = other.perm;

  input_scales = other.input_scales;
  weight_scales = other.weight_scales;
}

uint8_t* Tensor::GetPtrImpl(bool check_empty) const {
  if (check_empty && data_ptr == nullptr) {
    // do nothing now.
  }
  return reinterpret_cast<uint8_t*>(data_ptr) + static_cast<size_t>(offset);
}

size_t Tensor::GetElementNumber() const {
  if (shape.empty()) {
    return 0;
  }

  return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

size_t Tensor::GetDTypeSize() const {
  DataType dtype_impl = dtype;
  return GetTypeSize(dtype_impl);
}

size_t Tensor::GetTotalBytes() const {
  DataType dtype_impl = dtype;
  return GetElementNumber() * GetTypeSize(dtype_impl);
}

std::string Tensor::GetLocationString() const {
  static const std::unordered_map<MemoryLocation, std::string> loc_to_string{
      {MemoryLocation::LOCATION_HOST, "host"}, {MemoryLocation::LOCATION_DEVICE, "device"}};
  return loc_to_string.at(location);
}

std::string Tensor::ToString() const {
  static const std::unordered_map<DataType, std::string> dtype_to_string{
      {TYPE_BOOL, "BOOL"},     {TYPE_UINT8, "UINT8"},     {TYPE_UINT16, "UINT16"},   {TYPE_UINT32, "UINT32"},
      {TYPE_UINT64, "UINT64"}, {TYPE_INT8, "INT8"},       {TYPE_INT16, "INT16"},     {TYPE_INT32, "INT32"},
      {TYPE_INT64, "INT64"},   {TYPE_BF16, "BF16"},       {TYPE_FP16, "FP16"},       {TYPE_FP32, "FP32"},
      {TYPE_FP64, "FP64"},     {TYPE_BYTES, "BYTES"},     {TYPE_INVALID, "INVALID"}, {TYPE_FP8_E4M3, "E4M3"},
      {TYPE_VOID, "VOID"},     {TYPE_POINTER, "POINTER"}, {TYPE_FP8_E5M2, "E5M2"},
  };

  DataType dtype_impl = dtype;
  std::string loc_str = GetLocationString();
  return FormatStr("Tensor[where=%s, dtype=%s, shape=%s]", loc_str.c_str(), dtype_to_string.at(dtype_impl).c_str(),
                   Vector2Str(shape).c_str());
}

std::string GetNumpyType(DataType dtype) {
  static const std::unordered_map<DataType, std::string> type_map{
      {TYPE_INVALID, "x"}, {TYPE_BOOL, "?"},      {TYPE_BYTES, "b"},    {TYPE_UINT8, "u1"}, {TYPE_UINT16, "u2"},
      {TYPE_UINT32, "u4"}, {TYPE_UINT64, "u8"},   {TYPE_POINTER, "u8"}, {TYPE_INT8, "i1"},  {TYPE_INT16, "i2"},
      {TYPE_INT32, "i4"},  {TYPE_INT64, "i8"},    {TYPE_FP16, "f2"},    {TYPE_BF16, "u2"},  {TYPE_FP32, "f4"},
      {TYPE_FP64, "f8"},   {TYPE_FP8_E4M3, "u1"}, {TYPE_FP8_E5M2, "u1"}};

  DataType dtype_impl = dtype;
  return type_map.count(dtype_impl) ? type_map.at(dtype_impl) : "x";
}

bool Tensor::Equal(const Tensor& other) const {
  if (dtype != other.dtype) {
    return false;
  }

  if (GetTotalBytes() != other.GetTotalBytes()) {
    return false;
  }

  if (location != other.location) {
    return false;
  }

  void* host_data_a = nullptr;
  void* host_data_b = nullptr;

  bool need_free = false;
  size_t total_bytes = GetTotalBytes();
  if (location == MemoryLocation::LOCATION_HOST) {
    host_data_a = GetPtr<void>();
    host_data_b = other.GetPtr<void>();
  } else if (location == MemoryLocation::LOCATION_DEVICE) {
    host_data_a = malloc(total_bytes);
    host_data_b = malloc(total_bytes);

    SetDevice(device_id);
    Memcpy(host_data_a, GetPtr<void>(), total_bytes, MEMCPY_DEVICE_TO_HOST);
    SetDevice(other.device_id);
    Memcpy(host_data_b, other.GetPtr<void>(), total_bytes, MEMCPY_DEVICE_TO_HOST);
    need_free = true;
  }

  bool is_equal = (memcmp(host_data_a, host_data_b, total_bytes) == 0);
  if (need_free) {
    free(host_data_a);
    free(host_data_b);
  }

  return is_equal;
}

void Tensor::Fill(float f) {
  DeviceSynchronize();
  void* tensor_data_ptr = GetPtr<void>();
  int val = reinterpret_cast<int&>(f);
  if (location == MemoryLocation::LOCATION_DEVICE) {
    Memset(tensor_data_ptr, val, GetTotalBytes());
  } else if (location == MemoryLocation::LOCATION_HOST) {
    std::memset(tensor_data_ptr, val, GetTotalBytes());
  } else {
    KLLM_LOG_WARNING << "Do nothing when LOCATION_UNKNOWN";
    return;
  }
  DeviceSynchronize();
}

void Tensor::SaveToNpyFile(const std::string& file_path) {
  std::string full_file_path = file_path;
  std::filesystem::path dir_path = std::filesystem::path(full_file_path).parent_path();
  if (dir_path.string().empty()) {
    // If the directory path is empty, use the current working directory.
    dir_path = std::filesystem::current_path();
  }

  if (!std::filesystem::exists(dir_path)) {
    KLLM_LOG_WARNING << fmt::format("Do not exists the saved path {}", dir_path.string());
    std::filesystem::create_directories(dir_path);
  }

  KLLM_LOG_DEBUG << fmt::format("Save {} To file {}", ToString(), full_file_path);

  size_t total_size = GetTotalBytes();
  void* cpu_data = malloc(total_size);
  void* tensor_data_ptr = GetPtr<void>();
  auto memcpy_type = (location == MemoryLocation::LOCATION_DEVICE) ? MEMCPY_DEVICE_TO_HOST : MEMCPY_HOST_TO_HOST;

  if (location == MemoryLocation::LOCATION_DEVICE) {
    SetDevice(device_id);
  }
  DeviceSynchronize();
  Memcpy(cpu_data, tensor_data_ptr, total_size, memcpy_type);

  DataType dtype_impl = dtype;
  std::ofstream file(full_file_path, std::ios::binary);
  if (!file.is_open()) {
    KLLM_LOG_ERROR << fmt::format("Could not open file {}", full_file_path);
    return;
  }
  // Header of numpy file
  file << "\x93NUMPY";
  uint8_t major_version = 1;
  uint8_t minor_version = 0;
  file.write(reinterpret_cast<const char*>(&major_version), sizeof(uint8_t));
  file.write(reinterpret_cast<const char*>(&minor_version), sizeof(uint8_t));
  std::stringstream header_stream;
  header_stream << "{'descr': '" << GetNumpyType(dtype_impl) << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    header_stream << shape[i];
    if (shape.size() == 1 || i < shape.size() - 1) {
      header_stream << ",";
    }
  }
  header_stream << ")}";
  int base_length = 6 + 4 + header_stream.str().size();
  int pad_length = 16 * ((base_length + 1 + 15) / 16);
  for (int i = 0; i < pad_length - base_length; ++i) {
    header_stream << ((i == pad_length - base_length - 1) ? "\n" : "\x20");
  }
  std::string header = header_stream.str();
  const uint16_t header_len = header.size();
  file.write(reinterpret_cast<const char*>(&header_len), sizeof(uint16_t));
  file << header;

  // Tensor Data
  file.write(reinterpret_cast<const char*>(cpu_data), total_size);
  file.close();
  free(cpu_data);
}

void Tensor::LoadFromNpyFile(const std::string& file_path) {
  KLLM_LOG_DEBUG << fmt::format("Load {} To Tensor {}", file_path, ToString());

  std::vector<size_t> file_data_shape;
  FILE* f_ptr = fopen(file_path.c_str(), "rb");
  if (f_ptr == nullptr) {
    throw std::runtime_error("Could not open file " + file_path);
  }
  uint32_t header_len, start_data;
  llm_kernels::utils::ParseNpyIntro(f_ptr, header_len, start_data);
  llm_kernels::utils::ParseNpyHeader(f_ptr, header_len, file_data_shape);

  const size_t file_elems_num =
      std::accumulate(file_data_shape.begin(), file_data_shape.end(), 1, std::multiplies<size_t>());

  DataType dtype_impl = dtype;
  size_t data_size = file_elems_num * GetTypeSize(dtype_impl);

  if (data_size > GetTotalBytes()) {
    KLLM_THROW(fmt::format("LoadFromFile {} {} Bytes is more than tensor's total {} Bytes.", file_path, data_size,
                           GetTotalBytes()));
  }

  void* file_host_data_ptr = malloc(data_size);
  size_t n_elems = fread(file_host_data_ptr, GetTypeSize(dtype_impl), file_elems_num, f_ptr);
  if (n_elems != file_elems_num) {
    KLLM_THROW(fmt::format("LoadFromFile {} to tensor failed.", file_path));
  }
  auto memcpy_type = (location == MemoryLocation::LOCATION_DEVICE) ? MEMCPY_HOST_TO_DEVICE : MEMCPY_HOST_TO_HOST;

  SetDevice(device_id);
  DeviceSynchronize();
  Memcpy(GetPtr<void>(), file_host_data_ptr, data_size, memcpy_type);
  DeviceSynchronize();

  free(file_host_data_ptr);
  fclose(f_ptr);
}

MemoryChecker::MemoryChecker() {}

bool MemoryChecker::Enabled() {
  static bool enabled = (std::getenv("ENABLE_MEMORY_CHECK") != nullptr);
  if (enabled && check_memory_map_.empty()) {
    int dev_count;
    GetDeviceCount(&dev_count);
    check_memory_map_.resize(dev_count);
  }
  return enabled;
}

void MemoryChecker::AddMemoryBlock(const std::string& name, int rank, void* head_ptr, size_t head_bytes, void* tail_ptr,
                                   size_t tail_bytes, uint8_t expect_value) {
  if (!Enabled()) {
    return;
  }

  if (rank < 0 || rank >= check_memory_map_.size()) {
    throw std::runtime_error(
        fmt::format("AddMemoryBlock error, invalid device rank {}, device count {}.", rank, check_memory_map_.size()));
  }

  MemoryChecker::MemoryMeta memory_meta{head_ptr, head_bytes, tail_ptr, tail_bytes, expect_value};
  std::unordered_map<std::string, MemoryMeta>& check_map = check_memory_map_[rank];
  check_map[name] = memory_meta;
}

void MemoryChecker::RemoveMemoryBlock(const std::string& name, int rank) {
  if (!Enabled()) {
    return;
  }

  if (rank < 0 || rank >= check_memory_map_.size()) {
    throw std::runtime_error(fmt::format("RemoveMemoryBlock error, invalid device rank {}, device count {}.", rank,
                                         check_memory_map_.size()));
  }

  std::unordered_map<std::string, MemoryMeta>& check_map = check_memory_map_[rank];
  if (check_map.find(name) != check_map.end()) {
    check_map.erase(name);
  }
}

void MemoryChecker::CheckMemory(int rank, bool synchronize_device) {
  if (!Enabled()) {
    return;
  }

  if (rank < 0 || rank >= check_memory_map_.size()) {
    throw std::runtime_error(
        fmt::format("CheckMemory error, invalid device rank {}, device count {}.", rank, check_memory_map_.size()));
  }

  SetDevice(rank);
  if (synchronize_device) {
    DeviceSynchronize();
  }

  bool check_result = true;
  std::unordered_map<std::string, MemoryMeta>& check_map = check_memory_map_[rank];
  for (const auto& [name, memory_meta] : check_map) {
    if (memory_meta.head_ptr == nullptr || memory_meta.tail_ptr == nullptr) {
      KLLM_LOG_ERROR << "Check memory " << name << " on device " << rank << " error, null pointer found.";
      continue;
    }

    bool check_head = CheckMemoryValueImpl(memory_meta.head_ptr, memory_meta.head_bytes, memory_meta.expect_value);
    bool check_tail = CheckMemoryValueImpl(memory_meta.tail_ptr, memory_meta.tail_bytes, memory_meta.expect_value);
    if (!check_head) {
      check_result = false;
      KLLM_LOG_ERROR << "Check memory " << name << " on device " << rank << " error, overwrite by other memory.";
    } else if (!check_tail) {
      check_result = false;
      KLLM_LOG_ERROR << "Check memory " << name << " on device " << rank << " error, out of memory bound.";
    }
  }

  if (!check_result) {
    throw std::runtime_error("CheckMemory error, out of memory bound is found.");
  }
}

bool MemoryChecker::CheckMemoryValueImpl(void* check_ptr, size_t check_bytes, uint8_t expect_value) {
  std::vector<uint8_t> vec(check_bytes);
  Memcpy(vec.data(), check_ptr, check_bytes, MEMCPY_DEVICE_TO_HOST);

  bool check_succ = true;
  for (const auto& v : vec) {
    if (v != expect_value) {
      check_succ = false;
      break;
    }
  }
  return check_succ;
}

}  // namespace ksana_llm
