/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/utils/common.h"

#include <sstream>
#include <string>
#include <vector>

namespace llm_kernels {
namespace utils {

void ParseNpyIntro(FILE*& f_ptr, uint32_t& header_len, uint32_t& start_data) {
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

int32_t ParseNpyHeader(FILE*& f_ptr, uint32_t header_len, std::vector<size_t>& shape) {
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

}  // namespace utils
}  // namespace llm_kernels