/*
 * Copyright (c) 2025 DeepSeek
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Adapted from
 * [FlashMLA Project] https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/kernels/utils.h
 */

#pragma once

namespace llm_kernels {
namespace nvidia {

#define CHECK_CUDA(call)                                                                            \
  do {                                                                                              \
    cudaError_t status_ = call;                                                                     \
    if (status_ != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

#define FLASH_ASSERT(cond)                                                          \
  do {                                                                              \
    if (not(cond)) {                                                                \
      fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

#define FLASH_DEVICE_ASSERT(cond)                                          \
  do {                                                                     \
    if (not(cond)) {                                                       \
      printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond); \
      asm("trap;");                                                        \
    }                                                                      \
  } while (0)

#define println(fmt, ...)      \
  {                            \
    print(fmt, ##__VA_ARGS__); \
    print("\n");               \
  }

// FP8 FlashMLA
#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                                             \
    [&]                                                                                                                \
    {                                                                                                                  \
        if (COND)                                                                                                      \
        {                                                                                                              \
            constexpr static bool CONST_NAME = true;                                                                   \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            constexpr static bool CONST_NAME = false;                                                                  \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)                                                                   \
    [&]                                                                                                                \
    {                                                                                                                  \
        if (NUM_SPLITS <= 32)                                                                                          \
        {                                                                                                              \
            constexpr static int NAME = 32;                                                                            \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (NUM_SPLITS <= 64)                                                                                     \
        {                                                                                                              \
            constexpr static int NAME = 64;                                                                            \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (NUM_SPLITS <= 96)                                                                                     \
        {                                                                                                              \
            constexpr static int NAME = 96;                                                                            \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (NUM_SPLITS <= 128)                                                                                    \
        {                                                                                                              \
            constexpr static int NAME = 128;                                                                           \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (NUM_SPLITS <= 160)                                                                                    \
        {                                                                                                              \
            constexpr static int NAME = 160;                                                                           \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            FLASH_ASSERT(false);                                                                                       \
        }                                                                                                              \
    }()

}  // namespace nvidia
}  // namespace llm_kernels
