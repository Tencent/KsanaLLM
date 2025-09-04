/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include "common.h"

namespace deepep_wrapper {

class ProcessManager {
 private:
  std::vector<pid_t> process_pids_;
  IPCData* shared_data_ = nullptr;
  bool cleanup_called_ = false;
  int num_ranks_ = 8;
  int local_num_ranks_ = 8;
  int world_size_ = 1;
  int node_rank_ = 0;

 public:
  ~ProcessManager() {
    if (!cleanup_called_) {
      CleanUp();
    }
  }

  void CleanUp() {
    if (cleanup_called_) return;
    cleanup_called_ = true;

    std::cout << "Cleaning up processes..." << std::endl;

    // 发送SIGTERM信号给子进程
    for (auto& pid : process_pids_) {
      if (pid > 0) {
        kill(pid, SIGTERM);
      }
    }

    // 等待一段时间
    sleep(2);

    // 如果还没结束，发送SIGKILL
    for (auto& pid : process_pids_) {
      if (pid > 0) {
        kill(pid, SIGKILL);
      }
    }

    // 清理共享内存
    if (shared_data_) {
      int ret = munmap(shared_data_, sizeof(IPCData));
      if (ret != 0) {
        std::cerr << "Failed to clear shared_data, ret = " << ret << std::endl;
      }
    }
    CleanupSharedMemory(node_rank_);
  }

  int Run(int num_ranks, int world_size, int node_rank) {
    num_ranks_ = num_ranks;
    local_num_ranks_ = num_ranks_ / world_size;
    world_size_ = world_size;
    node_rank_ = node_rank;
    std::cout << "===== deepep_wrapper =====" << std::endl;

    // 等待主进程创建共享内存
    CleanupSharedMemory(node_rank);
    std::cout << "waiting for open shared memory " << std::endl;
    while (true) {
      shared_data_ = OpenSharedMemory(node_rank);
      if (shared_data_ != nullptr) {
        break;
      }
      sleep(10);
    }
    while (!shared_data_->ready) {
      sleep(10);
    }
    std::cout << "SharedData Ready" << std::endl;

    // 创建子进程
    process_pids_.resize(local_num_ranks_);
    for (int rank = 0; rank < local_num_ranks_; rank++) {
      process_pids_[rank] = fork();
      if (process_pids_[rank] == -1) {
        std::cerr << "Failed to fork Process " << rank << std::endl;
        CleanUp();
        return 1;
      } else if (process_pids_[rank] == 0) {
        // 进程创建成功,且为子进程(子进程查询 pid = 0, 父进程查询 pid > 0)
        try {
          int result = Process(shared_data_, num_ranks_, rank, world_size_, node_rank_);
          std::cout << "[Process " << rank << "] Exiting with code: " << result << std::endl;
          munmap(shared_data_, sizeof(IPCData));
          return 0;
        } catch (const std::exception& e) {
          std::cerr << "[Process " << rank << "] Exception: " << e.what() << std::endl;
          munmap(shared_data_, sizeof(IPCData));
          exit(1);
        } catch (...) {
          std::cerr << "[Process " << rank << "] Unknown exception" << std::endl;
          munmap(shared_data_, sizeof(IPCData));
          exit(1);
        }
      }

      // 父进程：等待一段时间确保进程先启动
      shared_data_->child_pids[rank] = process_pids_[rank];
      sleep(2);
    }

    bool process_done = false;
    while (!process_done) {
      int status;
      pid_t finished_pid = waitpid(-1, &status, 0);
      if (finished_pid == -1) {
        std::cerr << "Error waiting for child processes" << std::endl;
        break;
      }
      process_done = true;
      for (int rank = 0; rank < local_num_ranks_; rank++) {
        if (finished_pid == process_pids_[rank]) {
          shared_data_->process_finished[rank] = true;
          if (WIFEXITED(status)) {
            int process_status = WEXITSTATUS(status);
            std::cout << "Process " << rank << " completed with status: " << process_status << std::endl;
          } else {
            std::cout << "Process " << rank << " terminated abnormally" << std::endl;
          }
        }
        process_done &= shared_data_->process_finished[rank];
      }
    }
    return 0;
  }
};

}  // namespace deepep_wrapper

// 信号处理函数
deepep_wrapper::ProcessManager* g_process_manager_ = nullptr;

void SignalHandle(int signal) {
  std::cout << "\nReceived signal " << signal << ", cleaning up..." << std::endl;
  if (g_process_manager_) {
    g_process_manager_->CleanUp();
  }
  exit(signal);
}

int main(int argc, char* argv[]) {
  // 注册信号处理函数
  signal(SIGINT, SignalHandle);
  signal(SIGTERM, SignalHandle);

  deepep_wrapper::ProcessManager manager;
  g_process_manager_ = &manager;
  int num_ranks = 8;
  if (argc > 1) {
    num_ranks = std::atoi(argv[1]);
  }
  int world_size = 1;
  int node_rank = 0;
  if (argc > 3) {
    world_size = std::atoi(argv[2]);
    node_rank = std::atoi(argv[3]);
  }
  if (world_size > 1) {
    // 多机场景需要配置 nvshmem 环境变量
    std::vector<std::pair<std::string, std::string>> env_vars = {{"NVSHMEM_DISABLE_P2P", "0"},
                                                                 {"NVSHMEM_IB_ENABLE_IBGDA", "1"},
                                                                 {"NVSHMEM_IBGDA_NUM_RC_PER_PE", "24"},
                                                                 {"NVSHMEM_QP_DEPTH", "1024"},
                                                                 {"NVSHMEM_MAX_TEAMS", "7"},
                                                                 {"NVSHMEM_DISABLE_NVLS", "1"},
                                                                 {"NVSHMEM_CUMEM_GRANULARITY", "536870912"},
                                                                 {"NVSHMEM_DISABLE_MNNVL", "1"}};
    for (const auto& env_var : env_vars) {
      if (setenv(env_var.first.c_str(), env_var.second.c_str(), 1) != 0) {
        std::cerr << "Failed to set " << env_var.first << std::endl;
      } else {
        std::cout << "Set " << env_var.first << "=" << env_var.second << std::endl;
      }
    }
  }
  int result = manager.Run(num_ranks, world_size, node_rank);

  g_process_manager_ = nullptr;
  return result;
}
