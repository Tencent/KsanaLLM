/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/tensor.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>

#include "3rdparty/LLM_kernels/csrc/utils/common.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

template <typename T>
NotAssignable<T>::NotAssignable(T val) {
  value_ = val;
}

template <typename T>
NotAssignable<T>& NotAssignable<T>::operator=(T val) {
  throw std::runtime_error(error_message_);
}

template <typename T>
NotAssignable<T>::NotAssignable(const NotAssignable<T>& other) {
  throw std::runtime_error(error_message_);
}

template <typename T>
NotAssignable<T>::operator T() const {
  return value_;
}

template <typename T>
void NotAssignable<T>::Set(const T& val) {
  value_ = val;
}

template <typename T>
const T& NotAssignable<T>::Get() const {
  return value_;
}

template <typename T>
T& NotAssignable<T>::Get() {
  return value_;
}

template <typename T>
void NotAssignable<T>::SetErrorMessage(const std::string& error_message) {
  error_message_ = error_message;
}

template class NotAssignable<MemoryLocation>;
template class NotAssignable<int>;
template class NotAssignable<void*>;

template <typename T>
TypeWithCheck<T>::TypeWithCheck() {}

template <typename T>
TypeWithCheck<T>::TypeWithCheck(T val) {
  value_ = val;
}

template <typename T>
void TypeWithCheck<T>::SetChecker(std::function<void()> checker) {
  checker_ = checker;
}

template <typename T>
TypeWithCheck<T>& TypeWithCheck<T>::operator=(T val) {
  value_ = val;
  if (checker_) {
    checker_();
  }
  return *this;
}

template <typename T>
TypeWithCheck<T>& TypeWithCheck<T>::operator+=(T val) {
  if (std::is_integral<T>::value) {
    value_ += val;
    if (checker_) {
      checker_();
    }
  }
  return *this;
}

template <>
TypeWithCheck<DataType>& TypeWithCheck<DataType>::operator+=(DataType val) {
  KLLM_THROW("The operator+= is not supported by DataType.");
}

template <typename T>
TypeWithCheck<T>& TypeWithCheck<T>::operator-=(T val) {
  if (std::is_integral<T>::value) {
    value_ -= val;
    if (checker_) {
      checker_();
    }
  }
  return *this;
}

template <>
TypeWithCheck<DataType>& TypeWithCheck<DataType>::operator-=(DataType val) {
  KLLM_THROW("The operator-= is not supported by DataType.");
}

template <typename T>
TypeWithCheck<T>::TypeWithCheck(const TypeWithCheck<T>& other) {
  value_ = other.value_;
  if (checker_) {
    checker_();
  }
}

template <typename T>
TypeWithCheck<T>::operator T() const {
  return value_;
}

template class TypeWithCheck<size_t>;
template class TypeWithCheck<DataType>;

ShapeTypeWithCheck::ShapeTypeWithCheck() {}

ShapeTypeWithCheck::ShapeTypeWithCheck(const std::vector<size_t> val) {
  val_.resize(val.size());
  std::transform(val.begin(), val.end(), val_.begin(), [](size_t v) -> TypeWithCheck<size_t> { return size_t(v); });
}

ShapeTypeWithCheck::~ShapeTypeWithCheck() {}

void ShapeTypeWithCheck::SetChecker(std::function<void()> checker) { checker_ = checker; }

ShapeTypeWithCheck& ShapeTypeWithCheck::operator=(std::vector<size_t> val) {
  val_.resize(val.size());
  std::transform(val.begin(), val.end(), val_.begin(),
                 [](size_t v) -> TypeWithCheck<size_t> { return TypeWithCheck<size_t>(v); });
  if (checker_) {
    checker_();
  }
  return *this;
}

TypeWithCheck<size_t>& ShapeTypeWithCheck::operator[](std::vector<size_t>::size_type index) {
  val_[index].SetChecker(checker_);
  return val_[index];
}

const TypeWithCheck<size_t>& ShapeTypeWithCheck::operator[](std::vector<size_t>::size_type index) const {
  return val_[index];
}

ShapeTypeWithCheck::operator std::vector<size_t>() const {
  std::vector<size_t> result(val_.size());
  std::transform(val_.begin(), val_.end(), result.begin(), [](TypeWithCheck<size_t> v) -> size_t { return v; });
  return result;
}

std::vector<TypeWithCheck<size_t>>::iterator ShapeTypeWithCheck::begin() { return val_.begin(); }

std::vector<TypeWithCheck<size_t>>::const_iterator ShapeTypeWithCheck::begin() const { return val_.begin(); }

std::vector<TypeWithCheck<size_t>>::iterator ShapeTypeWithCheck::end() { return val_.end(); }

std::vector<TypeWithCheck<size_t>>::const_iterator ShapeTypeWithCheck::end() const { return val_.end(); }

std::vector<size_t>::size_type ShapeTypeWithCheck::size() const { return val_.size(); }

bool ShapeTypeWithCheck::empty() const { return val_.empty(); }

void ShapeTypeWithCheck::resize(std::vector<size_t>::size_type count) {
  val_.resize(count);
  if (checker_) {
    checker_();
  }
}

void ShapeTypeWithCheck::resize(std::vector<size_t>::size_type count, const std::vector<size_t>::value_type& value) {
  val_.resize(count, value);
  if (checker_) {
    checker_();
  }
}

TypeWithCheck<size_t>& ShapeTypeWithCheck::back() { return val_.back(); }

const TypeWithCheck<size_t>& ShapeTypeWithCheck::back() const { return val_.back(); }

TypeWithCheck<size_t>& ShapeTypeWithCheck::front() { return val_.front(); }

const TypeWithCheck<size_t>& ShapeTypeWithCheck::front() const { return val_.front(); }

std::vector<TypeWithCheck<size_t>>::iterator ShapeTypeWithCheck::erase(
    std::vector<TypeWithCheck<size_t>>::iterator pos) {
  auto result = val_.erase(pos);
  if (checker_) {
    checker_();
  }
  return result;
}

std::vector<TypeWithCheck<size_t>>::iterator ShapeTypeWithCheck::insert(
    std::vector<TypeWithCheck<size_t>>::const_iterator pos, const TypeWithCheck<size_t>& value) {
  auto result = val_.insert(pos, value);
  if (checker_) {
    checker_();
  }
  return result;
}

bool ShapeTypeWithCheck::operator==(const ShapeTypeWithCheck& other) { return (val_ == other.val_); }

Tensor::Tensor() { InitializeChecker(); }

Tensor::Tensor(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, int device_id, void* data_ptr,
               Stream* stream)
    : location(location), device_id(device_id), dtype(dtype), shape(shape), data_ptr(data_ptr), stream_(stream) {
  if (dtype == DataType::TYPE_INVALID) {
    // For dummy tensor, with type TYPE_INVALID, checker is disabled.
    return;
  }

  if (this->shape.empty()) {
    KLLM_THROW("Tensor could not be created with empty shape");
  }

  if (data_ptr != nullptr) {
    is_shared_buffer_ = true;
  } else {
    AllocateMemory();
  }

  reference_ = std::make_shared<int>(0);
  max_buffer_size_ = GetTotalBytes();

  InitializeChecker();
}

Tensor::~Tensor() {
  // Free underlying memory until no more tensor instances referenced.
  if (reference_.use_count() == 1) {
    checker_ = nullptr;
    dtype.SetChecker(nullptr);
    shape.SetChecker(nullptr);
    offset.SetChecker(nullptr);

    FreeMemory();

    location.Set(MemoryLocation::LOCATION_UNKNOWN);
    device_id.Set(-1);

    dtype = DataType::TYPE_INVALID;
    shape.val_.clear();
    offset = 0;
  }
}

void Tensor::FreeMemory() {
  if (data_ptr != nullptr && !is_shared_buffer_) {
    if (location == MemoryLocation::LOCATION_HOST) {
      HostFree(data_ptr);
    } else if (location == MemoryLocation::LOCATION_DEVICE) {
      SetDevice(device_id);
      if (stream_ == nullptr) {
        Free(data_ptr);
      } else {
        FreeAsync(data_ptr, *stream_);
      }
    }
    data_ptr.Set(nullptr);
  }
}

void Tensor::AllocateMemory() {
  if (data_ptr == nullptr && !is_shared_buffer_) {
    size_t total_bytes = GetTotalBytes();
    if (location == MemoryLocation::LOCATION_HOST) {
      HostAlloc(&data_ptr.Get(), total_bytes);
    } else if (location == MemoryLocation::LOCATION_DEVICE) {
      SetDevice(device_id);
      if (stream_ == nullptr) {
        Malloc(&data_ptr.Get(), total_bytes);
      } else {
        // NOTE(karlluo): for NVIDIA GPU ref
        // https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
        // for Huawei NPU ref
        // https://support.enflame-tech.com/onlinedoc_dev_3.3/_static/topsplatform_html/3-guide
        // /programing_guide/content/source/memory_model.html
        // create device memory space with stream and memory pool as extra memory management.
        MallocAsync(&data_ptr.Get(), total_bytes, *stream_);
      }
    }
  }
}

void Tensor::AssignMembers(const Tensor& other) {
  location.Set(other.location.Get());
  device_id.Set(other.device_id.Get());
  data_ptr.Set(other.data_ptr.Get());

  dtype = other.dtype;
  shape = other.shape;
  data_format = other.data_format;
  offset = other.offset;

  is_shared_buffer_ = other.is_shared_buffer_;
  reference_ = other.reference_;
  max_buffer_size_ = other.max_buffer_size_;

  // NOTE: Must bind current checker to new dtype & shape.
  dtype.SetChecker(this->checker_);
  shape.SetChecker(this->checker_);
  offset.SetChecker(this->checker_);

  scales = other.scales;
  zeros = other.zeros;

  g_idx = other.g_idx;
  perm = other.perm;

  input_scales = other.input_scales;
  weight_scales = other.weight_scales;
}

void Tensor::InitializeChecker() {
  location.SetErrorMessage("The location could not be modified.");

  checker_ = [this]() -> void {
    DataType dtype_impl = DataType(this->dtype);
    if (dtype_impl == DataType::TYPE_INVALID) {
      return;
    }

    size_t bytes = GetTypeSize(dtype_impl);
    std::vector<size_t> dims = this->shape;
    for (auto dim : dims) {
      bytes *= dim;
    }

    if (this->offset > this->max_buffer_size_ || bytes > (this->max_buffer_size_ - this->offset)) {
      KLLM_THROW(fmt::format("The tensor dtype {} and shape {} with offset {} exceed max memory size {}.", dtype_impl,
                             Vector2Str(dims), this->offset, this->max_buffer_size_));
    }
  };
  dtype.SetChecker(checker_);
  shape.SetChecker(checker_);
  offset.SetChecker(checker_);
}

size_t Tensor::GetElementNumber() const {
  if (shape.val_.empty()) {
    return 0;
  }

  return std::accumulate(shape.val_.begin(), shape.val_.end(), static_cast<size_t>(1), std::multiplies<size_t>());
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
                   Vector2Str(shape.val_).c_str());
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
  for (size_t i = 0; i < shape.val_.size(); ++i) {
    header_stream << shape.val_[i];
    if (shape.val_.size() == 1 || i < shape.val_.size() - 1) {
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

}  // namespace ksana_llm
