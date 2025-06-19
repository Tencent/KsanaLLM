/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <any>
#include <cstdint>
#include <string>
#include <vector>

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/status.h"

// GGUF file magic number
#define GGUF_MAGIC 0x46554747
#define GGUF_VERSION 3
#define GGUF_ALIGNMENT 32
#define MAX_STRING_LENGTH (1024 * 1024)
#define MAX_DIMS 4

namespace ksana_llm {

enum NewGGUFMetaValueType : uint32_t {
  // The value is a 8-bit unsigned integer.
  NEW_GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
  // The value is a 8-bit signed integer.
  NEW_GGUF_METADATA_VALUE_TYPE_INT8 = 1,
  // The value is a 16-bit unsigned little-endian integer.
  NEW_GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
  // The value is a 16-bit signed little-endian integer.
  NEW_GGUF_METADATA_VALUE_TYPE_INT16 = 3,
  // The value is a 32-bit unsigned little-endian integer.
  NEW_GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
  // The value is a 32-bit signed little-endian integer.
  NEW_GGUF_METADATA_VALUE_TYPE_INT32 = 5,
  // The value is a 32-bit IEEE754 floating point number.
  NEW_GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
  // The value is a boolean.
  // 1-byte value where 0 is false and 1 is true.
  // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
  NEW_GGUF_METADATA_VALUE_TYPE_BOOL = 7,
  // The value is a UTF-8 non-null-terminated string, with length prepended.
  NEW_GGUF_METADATA_VALUE_TYPE_STRING = 8,
  // The value is an array of other values, with the length and type prepended.
  // /
  // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
  NEW_GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
  // The value is a 64-bit unsigned little-endian integer.
  NEW_GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
  // The value is a 64-bit signed little-endian integer.
  NEW_GGUF_METADATA_VALUE_TYPE_INT64 = 11,
  // The value is a 64-bit IEEE754 floating point number.
  NEW_GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

enum NewGGMLType : uint32_t {
  NEW_GGML_TYPE_F32 = 0,
  NEW_GGML_TYPE_F16 = 1,
  NEW_GGML_TYPE_Q4_0 = 2,
  NEW_GGML_TYPE_Q4_1 = 3,
  // NEW_GGML_TYPE_Q4_2 = 4, support has been removed
  // NEW_GGML_TYPE_Q4_3 = 5, support has been removed
  NEW_GGML_TYPE_Q5_0 = 6,
  NEW_GGML_TYPE_Q5_1 = 7,
  NEW_GGML_TYPE_Q8_0 = 8,
  NEW_GGML_TYPE_Q8_1 = 9,
  NEW_GGML_TYPE_Q2_K = 10,
  NEW_GGML_TYPE_Q3_K = 11,
  NEW_GGML_TYPE_Q4_K = 12,
  NEW_GGML_TYPE_Q5_K = 13,
  NEW_GGML_TYPE_Q6_K = 14,
  NEW_GGML_TYPE_Q8_K = 15,
  NEW_GGML_TYPE_IQ2_XXS = 16,
  NEW_GGML_TYPE_IQ2_XS = 17,
  NEW_GGML_TYPE_IQ3_XXS = 18,
  NEW_GGML_TYPE_IQ1_S = 19,
  NEW_GGML_TYPE_IQ4_NL = 20,
  NEW_GGML_TYPE_IQ3_S = 21,
  NEW_GGML_TYPE_IQ2_S = 22,
  NEW_GGML_TYPE_IQ4_XS = 23,
  NEW_GGML_TYPE_I8 = 24,
  NEW_GGML_TYPE_I16 = 25,
  NEW_GGML_TYPE_I32 = 26,
  NEW_GGML_TYPE_I64 = 27,
  NEW_GGML_TYPE_F64 = 28,
  NEW_GGML_TYPE_IQ1_M = 29,
  NEW_GGML_TYPE_BF16 = 30,
  NEW_GGML_TYPE_Q4_0_4_4 = 31,
  NEW_GGML_TYPE_Q4_0_4_8 = 32,
  NEW_GGML_TYPE_Q4_0_8_8 = 33,
  NEW_GGML_TYPE_TQ1_0 = 34,
  NEW_GGML_TYPE_TQ2_0 = 35,
  NEW_GGML_TYPE_COUNT,
};

// model file types
enum NewGGUFModelFileType : uint32_t {
  NEW_LLAMA_FTYPE_ALL_F32 = 0,
  NEW_LLAMA_FTYPE_MOSTLY_F16 = 1,   // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
  // NEW_LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
  // NEW_LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
  // NEW_LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
  NEW_LLAMA_FTYPE_MOSTLY_Q8_0 = 7,       // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q5_0 = 8,       // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q5_1 = 9,       // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q2_K = 10,      // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q3_K_S = 11,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q3_K_M = 12,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q3_K_L = 13,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q4_K_S = 14,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q4_K_M = 15,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q5_K_S = 16,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q5_K_M = 17,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q6_K = 18,      // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19,   // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ2_XS = 20,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q2_K_S = 21,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ3_XS = 22,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23,   // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ1_S = 24,     // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ4_NL = 25,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ3_S = 26,     // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ3_M = 27,     // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ2_S = 28,     // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ2_M = 29,     // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ4_XS = 30,    // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_IQ1_M = 31,     // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_BF16 = 32,      // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q4_0_4_4 = 33,  // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q4_0_4_8 = 34,  // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_Q4_0_8_8 = 35,  // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_TQ1_0 = 36,     // except 1d tensors
  NEW_LLAMA_FTYPE_MOSTLY_TQ2_0 = 37,     // except 1d tensors

  NEW_LLAMA_FTYPE_GUESSED = 1024,  // not specified in the model file
};

struct NewGGUFHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

struct NewGGUFMetaValue {
  NewGGUFMetaValueType type;
  std::any value;
};

struct NewGGUFTensorInfo {
  std::string name;
  uint32_t n_dims;
  std::vector<uint64_t> dims;
  DataType data_type;
  uint64_t offset;  // offset from start of `data`, must be a multiple of `ALIGNMENT`
  size_t size;
};

struct NewGGUFContext {
  NewGGUFHeader header;
  std::unordered_map<std::string, NewGGUFMetaValue> metadata_map;
  std::unordered_map<std::string, NewGGUFTensorInfo> tensor_info_map;
  size_t alignment;
  size_t offset;  // size of `data` in bytes
};

// Get value from meta, error if not found.
std::any GetValueFromGGUFMeta(const std::unordered_map<std::string, NewGGUFMetaValue>& gguf_meta,
                              const std::string& key);

// Get value from meta, return default value if not found.
std::any GetValueFromGGUFMeta(const std::unordered_map<std::string, NewGGUFMetaValue>& gguf_meta,
                              const std::string& key, const std::any& default_value);

}  // namespace ksana_llm
