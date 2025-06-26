# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

# Declare TBB
FetchContent_Declare(
  tbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG v2022.2.0-rc1
)

# Set TBB options BEFORE making it available
set(TBB_STRICT OFF CACHE BOOL "Disable strict compilation for TBB")
set(TBB_TEST OFF CACHE BOOL "Disable TBB tests")
set(TBB_EXAMPLES OFF CACHE BOOL "Disable TBB examples")

# Set compiler flags BEFORE making TBB available
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow -Wno-error")

# Now make TBB available
FetchContent_MakeAvailable(tbb)

message(STATUS "Simplified TBB configuration without CET checks.")

if(NOT TARGET TBB::tbb)
    message(FATAL_ERROR "TBB::tbb target is NOT defined. Please check TBB FetchContent.")
endif()