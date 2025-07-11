# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/tests
)

add_subdirectory(batch_manager)
add_subdirectory(batch_scheduler)
add_subdirectory(cache_manager)
add_subdirectory(distributed)
add_subdirectory(endpoints)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(model_performance)
add_subdirectory(modules)
add_subdirectory(model_loader)
add_subdirectory(models)
add_subdirectory(models/base)
add_subdirectory(multi_batch_controller)
add_subdirectory(plugins)
add_subdirectory(data_hub)
add_subdirectory(runtime)
add_subdirectory(samplers)
add_subdirectory(service)
add_subdirectory(connector)
add_subdirectory(transfer)
add_subdirectory(profiler)
add_subdirectory(utils)
add_subdirectory(periphery)
add_subdirectory(cpp)

add_library(ksana_llm_libs INTERFACE)
target_link_libraries(ksana_llm_libs INTERFACE
  model_loader
  models
  model_performance
  modules
  model_base
  layers
  plugins
  service
  utils
  connector
  data_hub
  batch_manager
  multi_batch_controller
  batch_scheduler
  cache_manager
  distributed
  runtime
  endpoints
  samplers
  kernels
  profiler
  loguru
  gflags
  periphery
  ${NCCL_LIBRARIES}
)

add_subdirectory(torch_op)
