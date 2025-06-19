# Copyright 2024 Tencent Inc.  All rights reserved.

set(kernel_lib_name llm_kernels_ascend_utils_common)
add_library(llm_kernels_ascend_utils_common STATIC common.cc atb_executor.cc ../common.cc)
set_property(TARGET llm_kernels_ascend_utils_common PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(llm_kernels_ascend_utils_common PUBLIC ${ACL_SHARED_LIBS})
