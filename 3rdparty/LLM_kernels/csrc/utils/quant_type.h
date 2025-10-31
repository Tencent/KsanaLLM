#pragma once

namespace llm_kernels {
namespace utils {

enum KVCacheType {
  kAuto = 0,
  kFp8E4M3 = 1,
  kFp8E5M2 = 2,
  /* DeepSeek Sparse MLA adopts the following custom FP8 format (FP8 with scale).
   * Each token's KV cache is 656 Bytes, structured as:
   * - **First 512 bytes:** The "quantized NoPE" part, containing 512
   *   `float8_e4m3` values.
   * - **Next 16 bytes:** Scale factors, containing 4 `float32` values.
   *   The first `float32` is the scale for the first 128 `float8_e4m3` values,
   *   the second for the next 128, and so on.
   * - **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
   *   part is not quantized for accuracy.
   */
  kFp8DsMla = 3,
};

}  // namespace utils
}  // namespace llm_kernels
