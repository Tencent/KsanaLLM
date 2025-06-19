# Copyright 2024 Tencent Inc.  All rights reserved.

if(WITH_CUDA)
  add_subdirectory(nvidia)
endif()

if(WITH_ACL)
  add_subdirectory(ascend)
endif()

if(WITH_TOPS)
  add_subdirectory(zixiao)
endif()
