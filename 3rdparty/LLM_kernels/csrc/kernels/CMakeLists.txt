# Copyright 2024 Tencent Inc.  All rights reserved.

if(WITH_CUDA)
  if(SM VERSION_GREATER_EQUAL "90")
    add_definitions(-DENABLE_FLASHINFER=ON)
  endif()
  add_subdirectory(nvidia)
endif()

if(WITH_ACL)
  add_subdirectory(ascend)
  set(llm_kernels_ascend_libs ${llm_kernels_ascend_libs} PARENT_SCOPE)
endif()
