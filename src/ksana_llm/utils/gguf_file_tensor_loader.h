/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <any>
#include <string>
#include <unordered_map>
#include <vector>
#include "base_file_tensor_loader.h"

// GGUF file magic number
#define GGUF_MAGIC 0x46554747
#define GGUF_VERSION 3
#define GGUF_ALIGNMENT 32
#define MAX_STRING_LENGTH (1024 * 1024)
#define MAX_DIMS 4

namespace ksana_llm {

enum GGUFMetaValueType : uint32_t {
  // The value is a 8-bit unsigned integer.
  GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
  // The value is a 8-bit signed integer.
  GGUF_METADATA_VALUE_TYPE_INT8 = 1,
  // The value is a 16-bit unsigned little-endian integer.
  GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
  // The value is a 16-bit signed little-endian integer.
  GGUF_METADATA_VALUE_TYPE_INT16 = 3,
  // The value is a 32-bit unsigned little-endian integer.
  GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
  // The value is a 32-bit signed little-endian integer.
  GGUF_METADATA_VALUE_TYPE_INT32 = 5,
  // The value is a 32-bit IEEE754 floating point number.
  GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
  // The value is a boolean.
  // 1-byte value where 0 is false and 1 is true.
  // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
  GGUF_METADATA_VALUE_TYPE_BOOL = 7,
  // The value is a UTF-8 non-null-terminated string, with length prepended.
  GGUF_METADATA_VALUE_TYPE_STRING = 8,
  // The value is an array of other values, with the length and type prepended.
  ///
  // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
  GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
  // The value is a 64-bit unsigned little-endian integer.
  GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
  // The value is a 64-bit signed little-endian integer.
  GGUF_METADATA_VALUE_TYPE_INT64 = 11,
  // The value is a 64-bit IEEE754 floating point number.
  GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

enum GGMLType : uint32_t {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  // GGML_TYPE_Q4_2 = 4, support has been removed
  // GGML_TYPE_Q4_3 = 5, support has been removed
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_IQ2_XXS = 16,
  GGML_TYPE_IQ2_XS = 17,
  GGML_TYPE_IQ3_XXS = 18,
  GGML_TYPE_IQ1_S = 19,
  GGML_TYPE_IQ4_NL = 20,
  GGML_TYPE_IQ3_S = 21,
  GGML_TYPE_IQ2_S = 22,
  GGML_TYPE_IQ4_XS = 23,
  GGML_TYPE_I8 = 24,
  GGML_TYPE_I16 = 25,
  GGML_TYPE_I32 = 26,
  GGML_TYPE_I64 = 27,
  GGML_TYPE_F64 = 28,
  GGML_TYPE_IQ1_M = 29,
  GGML_TYPE_BF16 = 30,
  GGML_TYPE_Q4_0_4_4 = 31,
  GGML_TYPE_Q4_0_4_8 = 32,
  GGML_TYPE_Q4_0_8_8 = 33,
  GGML_TYPE_TQ1_0 = 34,
  GGML_TYPE_TQ2_0 = 35,
  GGML_TYPE_COUNT,
};

// model file types
enum GGUFModelFileType : uint32_t {
  LLAMA_FTYPE_ALL_F32 = 0,
  LLAMA_FTYPE_MOSTLY_F16 = 1,   // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
  // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
  // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
  // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
  LLAMA_FTYPE_MOSTLY_Q8_0 = 7,       // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q5_0 = 8,       // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q5_1 = 9,       // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q2_K = 10,      // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q3_K_S = 11,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q3_K_M = 12,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q3_K_L = 13,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q4_K_S = 14,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q4_K_M = 15,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q5_K_S = 16,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q5_K_M = 17,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q6_K = 18,      // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19,   // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ2_XS = 20,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q2_K_S = 21,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ3_XS = 22,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23,   // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ1_S = 24,     // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ4_NL = 25,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ3_S = 26,     // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ3_M = 27,     // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ2_S = 28,     // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ2_M = 29,     // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ4_XS = 30,    // except 1d tensors
  LLAMA_FTYPE_MOSTLY_IQ1_M = 31,     // except 1d tensors
  LLAMA_FTYPE_MOSTLY_BF16 = 32,      // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q4_0_4_4 = 33,  // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q4_0_4_8 = 34,  // except 1d tensors
  LLAMA_FTYPE_MOSTLY_Q4_0_8_8 = 35,  // except 1d tensors
  LLAMA_FTYPE_MOSTLY_TQ1_0 = 36,     // except 1d tensors
  LLAMA_FTYPE_MOSTLY_TQ2_0 = 37,     // except 1d tensors

  LLAMA_FTYPE_GUESSED = 1024,  // not specified in the model file
};

struct GGUFHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

struct GGUFMetaValue {
  GGUFMetaValueType type;
  std::any value;
};

struct GGUFTensorInfo {
  std::string name;
  uint32_t n_dims;
  std::vector<uint64_t> dims;
  DataType data_type;
  uint64_t offset;  // offset from start of `data`, must be a multiple of `ALIGNMENT`
  size_t size;
};

struct GGUFContext {
  GGUFHeader header;
  std::unordered_map<std::string, GGUFMetaValue> metadata_map;
  std::unordered_map<std::string, GGUFTensorInfo> tensor_info_map;
  size_t alignment;
  size_t offset;  // size of `data` in bytes
};

class GGUFFileTensorLoader : public BaseFileTensorLoader {
 public:
  explicit GGUFFileTensorLoader(const std::string& file_name, const bool load_bias);

  ~GGUFFileTensorLoader();

  GGUFFileTensorLoader(const GGUFFileTensorLoader&) = delete;
  GGUFFileTensorLoader& operator=(const GGUFFileTensorLoader&) = delete;

  const std::vector<std::string>& GetTensorNameList();

  std::tuple<void*, size_t> GetTensor(const std::string& tensor_name);
  void SetTensor(const std::string& tensor_name, torch::Tensor tensor);
  DataType GetTensorDataType(const std::string& tensor_name);

  std::string GetTensorFileName();

  std::vector<size_t> GetTensorShape(const std::string& tensor_name);

  const std::shared_ptr<GGUFContext> GetMetadata() const { return context_; }
  static std::vector<std::string> FindModelFiles(const std::string& model_dir_path);
  static DataType ConverGGUFModelFileTypeToDataType(uint32_t gguf_model_file_type);

  void InitTokenizer(const std::string& model_dir_path);

 private:
  template <typename T>
  T ReadDataFromFile();
  std::string ReadStringFromFile();
  void LoadGGUFContext();
  void LoadGGUFData();
  std::any ReadGGUFMetadataValue(GGUFMetaValueType type);
  static std::string ConvertFormatToRegex();
  DataType ConvertGGMLTypeToDataType(uint32_t ggml_type);

 private:
  std::unordered_map<std::string, void*> tensor_ptr_map_;
  std::unordered_map<std::string, torch::Tensor> tensor_map_;

  std::ifstream gguf_file_;
  int64_t file_size_ = 0;
  char* weights_buffer_ = nullptr;
  std::shared_ptr<GGUFContext> context_ = nullptr;
};

}  // namespace ksana_llm