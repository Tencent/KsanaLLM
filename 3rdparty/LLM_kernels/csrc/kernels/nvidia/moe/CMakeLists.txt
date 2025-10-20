# Copyright 2025 Tencent Inc.  All rights reserved.

add_subdirectory(cutlass_moe)
add_subdirectory(expert_map)
add_subdirectory(moe_wna16)
add_subdirectory(fused_moe_gptq_int4_fp8_kernel)
if(SM VERSION_GREATER_EQUAL "90")
  add_subdirectory(fused_moe)
  add_subdirectory(fused_moe_kernel_gptq_awq)
endif()