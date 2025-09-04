# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE distributed_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/*.cpp)

set(data_channel_nvidia_SRCS "")
set(data_channel_nvidia_LIBS "")

if(WITH_CUDA)
  file(GLOB_RECURSE data_channel_nvidia_SRCS ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/nvidia/*.cpp)
  list(FILTER data_channel_nvidia_SRCS EXCLUDE REGEX ".*test.cpp")
  list(APPEND data_channel_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)

  # nccl_data_channel_test
  file(GLOB_RECURSE nccl_data_channel_test_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/nvidia/nccl_data_channel_test.cpp)

  message(STATUS "nccl_data_channel_test_SRCS: ${nccl_data_channel_test_SRCS}")
  cpp_test(distributed_nccl_data_channel_test SRCS ${nccl_data_channel_test_SRCS} DEPS utils runtime data_hub LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)

  # expert_parallel_nccl_data_channel_test
  file(GLOB_RECURSE expert_parallel_nccl_data_channel_test_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/nvidia/expert_parallel_nccl_data_channel_test.cpp)

  message(STATUS "expert_parallel_nccl_data_channel_test_SRCS: ${expert_parallel_nccl_data_channel_test_SRCS}")
  cpp_test(expert_parallel_nccl_data_channel_test SRCS ${expert_parallel_nccl_data_channel_test_SRCS} DEPS utils runtime data_hub LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)
endif()

if(WITH_ACL)
  # NOTE(karlluo): should not load NCCL in Ascend mode
  list(FILTER distributed_SRCS EXCLUDE REGEX ".*nccl_data_channel.cpp")
endif()

list(FILTER distributed_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "distributed_SRCS: ${distributed_SRCS}")

add_library(distributed STATIC ${distributed_SRCS} ${data_channel_nvidia_SRCS})
target_link_libraries(distributed PUBLIC utils data_hub ${data_channel_nvidia_LIBS})

# raw_socket_test
file(GLOB_RECURSE raw_socket_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/raw_socket_test.cpp)

message(STATUS "raw_socket_test_SRCS: ${raw_socket_test_SRCS}")
cpp_test(distributed_raw_socket_test SRCS ${raw_socket_test_SRCS} DEPS utils runtime data_hub LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# control_channel_test
file(GLOB_RECURSE control_channel_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/control_channel_test.cpp)

message(STATUS "control_channel_test_SRCS: ${control_channel_test_SRCS}")
cpp_test(distributed_control_channel_test SRCS ${control_channel_test_SRCS} DEPS utils runtime data_hub LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# expert_parallel_control_channel_test
file(GLOB_RECURSE expert_parallel_control_channel_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/expert_parallel_control_channel_test.cpp)

message(STATUS "expert_parallel_control_channel_test_SRCS: ${expert_parallel_control_channel_test_SRCS}")
cpp_test(expert_parallel_control_channel_test SRCS ${expert_parallel_control_channel_test_SRCS} DEPS utils runtime data_hub LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})



# data_channel_test
file(GLOB_RECURSE data_channel_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/data_channel_test.cpp)

message(STATUS "data_channel_test_SRCS: ${data_channel_test_SRCS}")
cpp_test(distributed_data_channel_test SRCS ${data_channel_test_SRCS} DEPS utils runtime data_hub LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# expert_parallel_data_channel_test
file(GLOB_RECURSE expert_parallel_data_channel_test_SRCS
  ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/expert_parallel_data_channel_test.cpp)

message(STATUS "expert_parallel_data_channel_test_SRCS: ${expert_parallel_data_channel_test_SRCS}")
cpp_test(expert_parallel_data_channel_test SRCS ${expert_parallel_data_channel_test_SRCS} DEPS utils runtime data_hub LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# distributed_coordinator_test
if(WITH_CUDA)
  file(GLOB_RECURSE distributed_coordinator_test_SRCS
    ${PROJECT_SOURCE_DIR}/src/ksana_llm/distributed/distributed_coordinator_test.cpp)

  message(STATUS "distributed_coordinator_test_SRCS: ${distributed_coordinator_test_SRCS}")
  cpp_test(distributed_coordinator_test SRCS ${distributed_coordinator_test_SRCS} DEPS utils runtime data_hub LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
endif()
