/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/model_performance/communication_performance_runner.h"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include "tests/test.h"

namespace ksana_llm {
namespace {
TEST(CommunicationPerformanceRunnerTest, TestCommunication) {
  std::filesystem::path current_path = __FILE__;
  std::filesystem::path parent_path = current_path.parent_path();
  std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
  std::string config_path = std::filesystem::absolute(config_path_relate).string();

  std::string master_host = "127.0.0.1";
  std::string master_port = "11888";
  std::string world_size = "2";

  // master process
  pid_t master_pid = fork();
  if (master_pid == 0) {
    setenv("MASTER_HOST", master_host.c_str(), 1);
    setenv("MASTER_PORT", master_port.c_str(), 1);
    setenv("WORLD_SIZE", world_size.c_str(), 1);
    setenv("NODE_RANK", "0", 1);
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);

    try {
      CommunicationPerformanceRunner runner(config_path);
      runner.Run();
      exit(0);
    } catch (const std::exception& e) {
      std::cerr << "Master node error: " << e.what() << std::endl;
      exit(1);
    }
  } else if (master_pid < 0) {
    // failed to fork
    FAIL() << "Failed to fork master process";
  }

  std::cout << "Waiting for master process to start..." << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // worker process
  pid_t worker_pid = fork();
  if (worker_pid == 0) {
    setenv("MASTER_HOST", master_host.c_str(), 1);
    setenv("MASTER_PORT", master_port.c_str(), 1);
    setenv("WORLD_SIZE", world_size.c_str(), 1);
    setenv("NODE_RANK", "1", 1);
    setenv("CUDA_VISIBLE_DEVICES", "1", 1);

    try {
      CommunicationPerformanceRunner runner(config_path);
      runner.Run();
      exit(0);
    } catch (const std::exception& e) {
      std::cerr << "Worker node error: " << e.what() << std::endl;
      exit(1);
    }
  } else if (worker_pid < 0) {
    // failed to fork
    FAIL() << "Failed to fork worker process";
    kill(master_pid, SIGTERM);
  }

  // main process
  int master_status, worker_status;
  waitpid(master_pid, &master_status, 0);
  waitpid(worker_pid, &worker_status, 0);

  bool master_success = WIFEXITED(master_status) && WEXITSTATUS(master_status) == 0;
  bool worker_success = WIFEXITED(worker_status) && WEXITSTATUS(worker_status) == 0;

  EXPECT_TRUE(master_success) << "Master process failed with status: " << WEXITSTATUS(master_status);
  EXPECT_TRUE(worker_success) << "Worker process failed with status: " << WEXITSTATUS(worker_status);
}

}  // namespace
}  // namespace ksana_llm
