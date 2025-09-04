/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <random>

#include "wrapper.h"

namespace deepep_wrapper {

int Process(IPCData* shared_data, int num_ranks, int rank, int world_size, int node_rank) {
  // 进程创建并记录 PID
  // std::cout << "Process " << rank << " starting with PID: " << getpid() << std::endl;
  if (!shared_data) {
    std::cerr << "Process " << rank << ": Invalid shared data pointer" << std::endl;
    return 1;
  }
  shared_data->child_pids[rank] = getpid();

  // 初始化CUDA
  cudaSetDevice(rank);
  if (!InitCuda(rank)) {
    std::cerr << "Process " << rank << ": Failed to initialize CUDA" << std::endl;
    shared_data->error_code = 1;
    strncpy(shared_data->error_message, "Failed to initialize CUDA", sizeof(shared_data->error_message) - 1);
    return 1;
  }

  std::shared_ptr<Wrapper> runtime = std::make_shared<Wrapper>(shared_data, rank, num_ranks, world_size, node_rank);
  runtime->Run();

  return 0;
}

}  // namespace deepep_wrapper
