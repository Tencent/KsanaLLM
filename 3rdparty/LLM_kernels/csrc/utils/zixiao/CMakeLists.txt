# Copyright 2024 Tencent Inc.  All rights reserved.

set(kernel_lib_name llm_kernels_zixiao_utils_common)
add_library(llm_kernels_zixiao_utils_common STATIC ../common.cc)
set_property(TARGET llm_kernels_zixiao_utils_common PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(llm_kernels_zixiao_utils_common PUBLIC ${TOPS_SHARED_LIBS})
