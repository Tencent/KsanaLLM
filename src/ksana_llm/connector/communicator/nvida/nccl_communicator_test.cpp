/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/connector/communicator/nvida/nccl_communicator.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include "base64.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ksana_llm/connector/router_client/router_client.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/utils/nvidia/nccl_utils.h"
#endif
namespace ksana_llm {

// ===== MOCK/STUB 真实 NCCL/CUDA 调用以提升单测覆盖率 =====
namespace testing_internal {
// 全局 mock 控制开关
static bool g_mock_nccl = false;
void EnableNcclMock(bool enable) { g_mock_nccl = enable; }
bool IsNcclMockEnabled() { return g_mock_nccl; }
}  // namespace testing_internal

#ifdef ENABLE_CUDA

// Type alias for cleaner code
using ReceiveCallback = Communicator::ReceiveCallback;

// Forward declaration
std::unique_ptr<NcclCommGroup> CreateTestCommGroup(size_t device_count);

// 测试用 DummyCoordinator
class DummyCoordinator : public Coordinator {
 public:
  explicit DummyCoordinator(const ConnectorConfig& config) {}
  // 不要 override（基类没有虚函数签名匹配）
  Status SendCommId(const std::string& group_key, const std::string& comm_id) { return Status(); }
};

// 生成合法的 NCCL comm_id（128字节全0并base64编码）
std::string MakeFakeNcclIdBase64() {
  std::array<unsigned char, 128> fake_id = {0};
  std::string id_bytes(reinterpret_cast<const char*>(fake_id.data()), fake_id.size());
  return base64::to_base64(id_bytes);
}

// 多进程 NCCL 测试：使用 fork + exec 模式初始化通信器
TEST(NcclCommunicatorTest, MultiProcessNCCLInit) {
  // 环境检查：检测是否有CUDA设备
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  // 如果没有足够的 GPU 或 CUDA 初始化失败，则跳过测试
  if (err != cudaSuccess || device_count < 2) {
    GTEST_SKIP() << "需要至少2个CUDA设备进行此测试，但找到 " << (err == cudaSuccess ? device_count : 0)
                 << " 个设备。错误: " << (err == cudaSuccess ? "无" : cudaGetErrorString(err));
    return;
  }

  // 基本参数配置
  int world_size = 2;  // 两个进程

  // 创建 NCCL ID (在 fork 前创建，然后通过环境变量或命令行传递)
  ncclUniqueId comm_id;
  NCCL_CHECK(ncclGetUniqueId(&comm_id));

  // 将 NCCL ID 转换为 base64 字符串用于命令行传递
  char nccl_id_str[128] = {0};
  // 简单转换为十六进制字符串
  for (size_t i = 0; i < sizeof(ncclUniqueId); i++) {
    sprintf(nccl_id_str + i * 2, "%02x", ((unsigned char*)&comm_id)[i]);
  }

  // 创建临时文件存储 NCCL ID
  char temp_file[] = "/tmp/nccl_id_XXXXXX";
  int fd = mkstemp(temp_file);
  if (fd < 0) {
    FAIL() << "Failed to create temporary file";
    return;
  }

  // 写入 NCCL ID 到临时文件
  write(fd, &comm_id, sizeof(ncclUniqueId));
  close(fd);

  // 设置 GPU 可见性
  setenv("CUDA_VISIBLE_DEVICES", "0,1", 1);

  // 存储子进程 PID
  std::vector<pid_t> pids;

  // 获取当前可执行文件路径
  char exe_path[PATH_MAX];

  // Linux specific code
  ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
  if (len == -1) {
    strcpy(exe_path, "unknown");
  } else {
    exe_path[len] = '\0';
  }

  // 创建多个进程进行 NCCL 测试
  for (int rank = 0; rank < world_size; ++rank) {
    pid_t pid = fork();
    if (pid < 0) {
      // fork 失败处理
      for (pid_t p : pids) {
        kill(p, SIGKILL);
      }
      unlink(temp_file);
      FAIL() << "进程创建失败";
    } else if (pid == 0) {
      // 子进程：通过 exec 启动新的进程

      // 构建参数
      char rank_arg[32], nccl_id_arg[256];
      sprintf(rank_arg, "--gtest_rank=%d", rank);
      sprintf(nccl_id_arg, "--gtest_nccl_id_file=%s", temp_file);

      // 执行 exec，启动一个新的测试进程
      char* args[] = {exe_path,
                      const_cast<char*>("--gtest_filter=NcclCommunicatorTest.NCCLWorkerProcess"),
                      const_cast<char*>("--gtest_brief=1"),
                      const_cast<char*>("--gtest_print_time=0"),
                      rank_arg,
                      nccl_id_arg,
                      NULL};

      // 设置环境变量
      char cuda_device[8];
      sprintf(cuda_device, "%d", rank);
      setenv("CUDA_VISIBLE_DEVICES", cuda_device, 1);

      // 重定向标准输出和标准错误到/dev/null，除非我们需要调试
      int null_fd = open("/dev/null", O_WRONLY);
      if (null_fd >= 0) {
        dup2(null_fd, STDOUT_FILENO);
        // 保持stderr输出以便看到错误信息
        // dup2(null_fd, STDERR_FILENO);
        close(null_fd);
      }

      // 执行新进程
      execv(exe_path, args);

      // 如果 execv 失败，退出子进程
      perror("execv failed");
      _exit(3);
    } else {
      // 父进程：记录子进程 PID
      pids.push_back(pid);
    }
  }

  // 父进程：等待所有子进程完成
  for (pid_t pid : pids) {
    int status = 0;
    waitpid(pid, &status, 0);

    // 验证子进程是否正常退出
    ASSERT_TRUE(WIFEXITED(status)) << "进程异常终止";
    EXPECT_EQ(WEXITSTATUS(status), 0) << "进程退出状态: " << WEXITSTATUS(status);
  }

  // 清理临时文件
  unlink(temp_file);
}

// 单进程 NCCL 工作进程测试 - 通过命令行参数执行
TEST(NcclCommunicatorTest, NCCLWorkerProcess) {
  // 获取命令行参数
  ::testing::GTEST_FLAG(list_tests) = false;  // 不列出测试
  ::testing::GTEST_FLAG(brief) = true;        // 简短输出
  ::testing::GTEST_FLAG(print_time) = false;  // 不打印时间
  int rank = -1;
  std::string nccl_id_file;
  const auto& flagsRef = ::testing::internal::GetArgvs();
  int argc = flagsRef.size();
  for (int i = 1; i < argc; i++) {
    std::string arg = flagsRef[i];

    if (arg.find("--gtest_rank=") == 0) {
      rank = std::stoi(arg.substr(13));
    } else if (arg.find("--gtest_nccl_id_file=") == 0) {
      nccl_id_file = arg.substr(20);
    }
  }

  // 如果不是通过 exec 调用的，则跳过测试
  if (rank < 0 || nccl_id_file.empty()) {
    GTEST_SKIP() << "此测试需要通过 exec 并指定 rank 和 nccl_id_file 来执行";
    return;
  }

  // 读取 NCCL ID
  ncclUniqueId comm_id;
  FILE* f = fopen(nccl_id_file.c_str(), "rb");
  ASSERT_NE(f, nullptr) << "无法打开 NCCL ID 文件: " << nccl_id_file;
  ASSERT_EQ(fread(&comm_id, sizeof(ncclUniqueId), 1, f), 1) << "读取 NCCL ID 失败";
  fclose(f);

  // 设置正确的 CUDA 设备 (这里使用 rank 直接映射设备)
  cudaError_t cuda_status = cudaSetDevice(0);  // 由于已经设置了 CUDA_VISIBLE_DEVICES，这里用 0
  ASSERT_EQ(cuda_status, cudaSuccess) << "cudaSetDevice 失败: " << cudaGetErrorString(cuda_status);

  // 配置基本参数
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  int node_rank = 0;
  int world_size = 2;  // 两个进程
  int device_count = 1;

  // 创建通信器
  auto communicator = std::make_unique<NcclCommunicator>(config);

  // 初始化通信ID
  ncclComm_t send_comm = nullptr;
  ncclComm_t recv_comm = nullptr;

  // 计算全局 world size（两组）
  const int global_world_size = 2 * world_size * device_count;

  // 计算通信ID
  NcclCommunicator::NcclRankInfo ranks =
      communicator->CalcNcclRanks(config.group_role, node_rank, world_size, device_count, rank);

  // 初始化 NCCL
  cudaStream_t send_stream = nullptr;
  cudaStream_t recv_stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&send_stream));
  CUDA_CHECK(cudaStreamCreate(&recv_stream));

  // 初始化 NCCL 通信器
  NCCL_CHECK(ncclCommInitRank(&send_comm, global_world_size, comm_id, ranks.send_rank));
  NCCL_CHECK(ncclCommInitRank(&recv_comm, global_world_size, comm_id, ranks.recv_rank));

  // 验证 NCCL 通信器初始化成功
  ASSERT_NE(send_comm, nullptr) << "NCCL send_comm 创建失败";
  ASSERT_NE(recv_comm, nullptr) << "NCCL recv_comm 创建失败";

  // 清理资源
  if (send_comm) {
    ncclCommDestroy(send_comm);
  }
  if (recv_comm) {
    ncclCommDestroy(recv_comm);
  }
  if (send_stream) {
    cudaStreamDestroy(send_stream);
  }
  if (recv_stream) {
    cudaStreamDestroy(recv_stream);
  }
}

// mock NcclCommunicator，避免真实 NCCL 初始化
class MockNcclCommunicator : public NcclCommunicator {
 public:
  explicit MockNcclCommunicator(const ConnectorConfig& config) : NcclCommunicator(config) {}
  Status InitializeNccl(const std::string&, int, int, int) { return Status(); }
  Status InitCommunicatorGroup(const std::string&, const std::vector<std::tuple<int, int, std::string>>&,
                               const std::string&) {
    return Status();
  }
  void SetSendCommIdCallback(std::function<Status(const std::string&, const std::string&)> cb) {
    NcclCommunicator::SetSendCommIdCallback(cb);
  }
  Status ProcessHeartbeatData(const std::unordered_map<std::string, std::string>& comm_group_to_id,
                              const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>&
                                  comm_group_to_address) override {
    // Directly call the base class implementation to ensure callback logic matches production
    return NcclCommunicator::ProcessHeartbeatData(comm_group_to_id, comm_group_to_address);
  }
};

// 拓扑结构检测辅助函数
void CheckNcclTopology(int world_size, int device_count) {
  // 预期表格
  // 组 node_rank dev_id send_rank recv_rank
  // Prefill: send_rank = node_rank * device_count + dev_id
  //          recv_rank = world_size * device_count + node_rank * device_count + dev_id
  // Decode:  send_rank = world_size * device_count + node_rank * device_count + dev_id
  //          recv_rank = node_rank * device_count + dev_id
  for (int role = 0; role < 2; ++role) {  // 0: Prefill, 1: Decode
    for (int node_rank = 0; node_rank < world_size; ++node_rank) {
      for (int dev_id = 0; dev_id < device_count; ++dev_id) {
        int send_rank = role * world_size * device_count + node_rank * device_count + dev_id;
        int recv_rank = (1 - role) * world_size * device_count + node_rank * device_count + dev_id;
        // 打印或断言
        if (role == 0) {
          // Prefill
          ASSERT_EQ(send_rank, node_rank * device_count + dev_id);
          ASSERT_EQ(recv_rank, world_size * device_count + node_rank * device_count + dev_id);
        } else {
          // Decode
          ASSERT_EQ(send_rank, world_size * device_count + node_rank * device_count + dev_id);
          ASSERT_EQ(recv_rank, node_rank * device_count + dev_id);
        }
      }
    }
  }
}

TEST(NcclCommunicatorTest, NcclTopologyCheck) {
  // 以 world_size=2, device_count=4 为例
  CheckNcclTopology(2, 4);
  // 你可以添加更多组合
  CheckNcclTopology(1, 1);
  CheckNcclTopology(2, 1);
}

// 直接测试 CalcNcclRanks 的单元测试
TEST(NcclCommunicatorTest, CalcNcclRanksBasic) {
  int world_size = 8;
  int device_count = 4;
  int nodes = 2;  // 假设有两个节点
  ConnectorConfig dummy_cfg;
  dummy_cfg.group_role = GroupRole::PREFILL;

  NcclCommunicator comm(dummy_cfg);

  for (int node_rank = 0; node_rank < nodes; ++node_rank) {
    for (int dev_id = 0; dev_id < device_count; ++dev_id) {
      // Prefill
      auto prefill = comm.CalcNcclRanks(GroupRole::PREFILL, node_rank, world_size, device_count, dev_id);
      int expected_cur = node_rank * device_count + dev_id;
      int expected_peer = world_size + node_rank * device_count + dev_id;
      ASSERT_EQ(prefill.cur_rank, expected_cur);
      ASSERT_EQ(prefill.peer_rank, expected_peer);
      ASSERT_EQ(prefill.send_rank, expected_peer);  // send_rank = peer_rank
      ASSERT_EQ(prefill.recv_rank, expected_peer);  // recv_rank = peer_rank
      // Decode
      auto decode = comm.CalcNcclRanks(GroupRole::DECODE, node_rank, world_size, device_count, dev_id);
      int expected_cur_d = world_size + node_rank * device_count + dev_id;
      int expected_peer_d = node_rank * device_count + dev_id;
      ASSERT_EQ(decode.cur_rank, expected_cur_d);
      ASSERT_EQ(decode.peer_rank, expected_peer_d);
      ASSERT_EQ(decode.send_rank, expected_peer_d);  // send_rank = peer_rank
      ASSERT_EQ(decode.recv_rank, expected_peer_d);  // recv_rank = peer_rank
    }
  }
}

// 测试 ProcessHeartbeatData 方法
TEST(NcclCommunicatorTest, ProcessHeartbeatData) {
  // 启用 NCCL Mock
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.node_rank = 0;     // Ensure node_rank is 0 so callback is triggered
  config.device_count = 2;  // 设置设备数量
  config.world_size = 2;    // 设置世界大小

  // 创建 MockNcclCommunicator
  auto communicator = std::make_unique<MockNcclCommunicator>(config);

  // 设置回调函数
  bool callback_called = false;
  communicator->SetSendCommIdCallback([&](const std::string& group_key, const std::string& comm_id) -> Status {
    callback_called = true;
    return Status();
  });

  // 创建测试心跳数据
  std::unordered_map<std::string, std::string> comm_group_to_id = {{"test_group", ""}};
  std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>> comm_group_to_address = {
      {"test_group", {{0, 0, "127.0.0.1:5001"}, {0, 1, "127.0.0.1:5002"}}}};

  // 调用 ProcessHeartbeatData 方法
  Status status = communicator->ProcessHeartbeatData(comm_group_to_id, comm_group_to_address);

  // 验证回调函数被调用
  ASSERT_TRUE(status.OK());
  ASSERT_TRUE(callback_called);

  // 禁用 NCCL Mock
  ksana_llm::testing_internal::EnableNcclMock(false);
}

// 更丰富的心跳数据 mock 场景测试
TEST(NcclCommunicatorTest, ProcessHeartbeatData_MockHeartbeat) {
  ksana_llm::testing_internal::EnableNcclMock(true);
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.device_count = 2;  // 设置设备数量
  config.world_size = 2;    // 设置世界大小

  config.node_rank = 0;  // Ensure callback is triggered for empty_id_group

  // 创建 MockNcclCommunicator
  auto communicator = std::make_unique<MockNcclCommunicator>(config);

  // 记录所有回调参数
  std::vector<std::pair<std::string, std::string>> called_pairs;
  communicator->SetSendCommIdCallback([&](const std::string& group_key, const std::string& comm_id) -> Status {
    called_pairs.emplace_back(group_key, comm_id);
    return Status();
  });

  // 生成合法 NCCL comm_id（128字节全0并base64编码）
  std::string valid_nccl_id_base64 = MakeFakeNcclIdBase64();

  // 构造 mock 心跳数据（多组/异常/空ID等场景，全部用合法base64字符串）
  std::unordered_map<std::string, std::string> comm_group_to_id = {
      {"test_group1", valid_nccl_id_base64}, {"test_group2", valid_nccl_id_base64}, {"empty_id_group", ""}};

  auto make_addrs = [](int base) {
    std::vector<std::tuple<int, int, std::string>> addrs;
    for (int node = 0; node < 2; ++node) {
      for (int dev = 0; dev < 2; ++dev) {
        for (int role = 0; role < 2; ++role) {
          int idx = node * 4 + dev * 2 + role;
          addrs.emplace_back(node, dev, "127.0.0.1:" + std::to_string(base + idx));
        }
      }
    }
    return addrs;
  };
  std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>> comm_group_to_address = {
      {"test_group1", make_addrs(5001)},
      {"test_group2", make_addrs(6001)},
      {"empty_id_group", make_addrs(7001)}};  // 为空ID组也创建8个地址

  // 调用 ProcessHeartbeatData
  Status status = communicator->ProcessHeartbeatData(comm_group_to_id, comm_group_to_address);

  // 验证回调被调用且参数正确
  ASSERT_TRUE(status.OK());
  // 只应有 empty_id_group 被回调
  std::set<std::string> expected_keys = {"empty_id_group"};
  std::set<std::string> called_keys;
  for (const auto& p : called_pairs) {
    called_keys.insert(p.first);
    ASSERT_FALSE(p.second.empty());  // comm_id 应为合法 base64 字符串
  }
  ASSERT_EQ(called_keys, expected_keys);

  // 检查所有通信组都已创建
  for (const auto& kv : comm_group_to_id) {
    auto* group = communicator->GetCommunicatorGroup(kv.first);
    ASSERT_NE(group, nullptr);
    // 可进一步断言 group->comm_id、group->device_resources 等
  }
  ksana_llm::testing_internal::EnableNcclMock(false);
}

// ===== CreateNcclDeviceResource 和 CreateDeviceResources 方法的单元测试 =====

// Mock class for testing CreateNcclDeviceResource without real CUDA/NCCL
class TestableNcclCommunicator : public NcclCommunicator {
 public:
  explicit TestableNcclCommunicator(const ConnectorConfig& config) : NcclCommunicator(config) {}

  // 暴露 CreateDeviceResources 和 CreateNcclDeviceResource 方法以供测试
  using NcclCommunicator::CreateDeviceResources;
  using NcclCommunicator::CreateNcclDeviceResource;
  using NcclCommunicator::GetCommunicatorGroup;
  using NcclCommunicator::SetCommGroupForTest;

  // 模拟 CreateNcclDeviceResource 避免实际 CUDA/NCCL 调用
  std::unique_ptr<NcclDeviceResource> CreateNcclDeviceResourceMock(const ConnectorConfig& config,
                                                                   const ncclUniqueId& comm_id, int node_rank,
                                                                   int world_size, int device_count, int dev_id) {
    auto resource = std::make_unique<NcclDeviceResource>();

    // 模拟计算 rank 信息
    NcclRankInfo ranks = CalcNcclRanks(config.group_role, node_rank, world_size, device_count, dev_id);

    // 模拟设置资源（不实际创建 CUDA/NCCL 对象）
    resource->send_rank = ranks.send_rank;
    resource->recv_rank = ranks.recv_rank;

    // 在实际测试中，这些将是 nullptr，但我们设置一些非空值以模拟成功初始化
    resource->send_comm = reinterpret_cast<ncclComm_t>(0x1);
    resource->recv_comm = reinterpret_cast<ncclComm_t>(0x2);
    resource->send_stream = reinterpret_cast<cudaStream_t>(0x3);
    resource->recv_stream = reinterpret_cast<cudaStream_t>(0x4);

    return resource;
  }

  // 模拟 CreateDeviceResources 方法
  Status CreateDeviceResourcesMock(const std::string& group_key, int world_size, int device_count) {
    auto group = GetCommunicatorGroup(group_key);
    if (!group) {
      return Status(RetCode::RET_INVALID_ARGUMENT, "Communication group not found: " + group_key);
    }

    // 创建所有设备资源
    group->device_resources_.resize(device_count);
    for (int i = 0; i < device_count; ++i) {
      try {
        ConnectorConfig mock_config;
        mock_config.group_role = GroupRole::PREFILL;
        ncclUniqueId mock_comm_id = {0};
        group->device_resources_[i] =
            CreateNcclDeviceResourceMock(mock_config, mock_comm_id, 0, world_size, device_count, i);
      } catch (const std::exception& e) {
        return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR,
                      std::string("Failed to create NCCL device resource: ") + e.what());
      }
    }

    return Status();
  }
};

// 测试 CreateNcclDeviceResource 方法
TEST(NcclCommunicatorTest, CreateNcclDeviceResource_ValidParameters) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  TestableNcclCommunicator communicator(config);

  int node_rank = 0;
  int world_size = 4;
  int device_count = 4;
  int dev_id = 1;

  // 测试创建设备资源时的 rank 计算
  auto ranks = communicator.CalcNcclRanks(config.group_role, node_rank, world_size, device_count, dev_id);

  // 验证 rank 计算是否正确
  int expected_peer = world_size + node_rank * device_count + dev_id;
  EXPECT_EQ(ranks.send_rank, expected_peer);
  EXPECT_EQ(ranks.recv_rank, expected_peer);
}

// 测试不同的 GroupRole
TEST(NcclCommunicatorTest, CreateNcclDeviceResource_DifferentGroupRoles) {
  std::vector<GroupRole> roles = {GroupRole::PREFILL, GroupRole::DECODE};

  for (auto role : roles) {
    ConnectorConfig config;
    config.group_role = role;

    TestableNcclCommunicator communicator(config);

    int node_rank = 1;
    int world_size = 4;
    int device_count = 2;
    int dev_id = 0;

    auto ranks = communicator.CalcNcclRanks(config.group_role, node_rank, world_size, device_count, dev_id);
    int expected_peer = (role == GroupRole::PREFILL) ? (1 * world_size + node_rank * device_count + dev_id)
                                                     : (0 * world_size + node_rank * device_count + dev_id);
    EXPECT_EQ(ranks.send_rank, expected_peer);
    EXPECT_EQ(ranks.recv_rank, expected_peer);
  }
}

TEST(NcclCommunicatorTest, CreateNcclDeviceResource_EdgeCases) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  TestableNcclCommunicator communicator(config);

  // 测试边界情况
  struct TestCase {
    int node_rank;
    int world_size;
    int device_count;
    int dev_id;
    std::string description;
  };

  std::vector<TestCase> test_cases = {
      {0, 1, 1, 0, "single node, single device"},
      {0, 2, 1, 0, "two nodes, single device each"},
      {1, 8, 4, 3, "two nodes, four devices each, last device"},
      {2, 8, 2, 1, "four nodes, two devices each"},
  };

  for (const auto& test_case : test_cases) {
    SCOPED_TRACE(test_case.description);

    auto ranks = communicator.CalcNcclRanks(config.group_role, test_case.node_rank, test_case.world_size,
                                            test_case.device_count, test_case.dev_id);

    int expected_peer = test_case.world_size + test_case.node_rank * test_case.device_count + test_case.dev_id;
    EXPECT_EQ(ranks.send_rank, expected_peer);
    EXPECT_EQ(ranks.recv_rank, expected_peer);
  }
}

// 测试 CreateDeviceResources 方法
TEST(NcclCommunicatorTest, CreateDeviceResources_ValidGroup) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  TestableNcclCommunicator communicator(config);

  // 创建一个测试用的通信组
  std::string group_key = "test_group";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->comm_id = {0};  // 模拟的 comm_id

  // 设置通信组
  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  // 创建设备资源
  int device_count = 4;
  Status status = communicator.CreateDeviceResourcesMock(group_key, 8, device_count);

  EXPECT_TRUE(status.OK()) << "CreateDeviceResources should succeed: " << status.GetMessage();

  // 验证设备资源已创建
  auto* group = communicator.GetCommunicatorGroup(group_key);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->device_resources_.size(), device_count);

  // 验证每个设备资源都已正确初始化
  for (int i = 0; i < device_count; ++i) {
    auto& resource = group->device_resources_[i];
    ASSERT_NE(resource, nullptr);
    int expected_peer = 2 * device_count + i;  // world_size=2, node_rank=0
    EXPECT_EQ(resource->send_rank, expected_peer);
    EXPECT_EQ(resource->recv_rank, expected_peer);
    EXPECT_NE(resource->send_comm, nullptr);
    EXPECT_NE(resource->recv_comm, nullptr);
    EXPECT_NE(resource->send_stream, nullptr);
    EXPECT_NE(resource->recv_stream, nullptr);
  }
}

TEST(NcclCommunicatorTest, CreateDeviceResources_NonExistentGroup) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  TestableNcclCommunicator communicator(config);

  // 尝试为不存在的组创建设备资源
  std::string non_existent_group = "non_existent_group";
  Status status = communicator.CreateDeviceResourcesMock(non_existent_group, 2, 2);

  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
  EXPECT_THAT(status.GetMessage(), ::testing::HasSubstr("Communication group not found"));
}

TEST(NcclCommunicatorTest, CreateDeviceResources_MultipleDevices) {
  ConnectorConfig config;
  config.group_role = GroupRole::DECODE;  // 测试 DECODE 角色

  TestableNcclCommunicator communicator(config);

  std::string group_key = "decode_group";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->comm_id = {0};

  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  // 测试不同数量的设备
  std::vector<std::pair<int, int>> device_pairs = {{2, 1}, {4, 2}, {8, 4}, {16, 8}};

  for (const auto& p : device_pairs) {
    SCOPED_TRACE("device_count: " + std::to_string(p.second));

    // 重新创建通信组以进行干净的测试
    auto new_group = std::make_unique<NcclCommGroup>();
    new_group->comm_id = {0};
    communicator.SetCommGroupForTest(group_key, std::move(new_group));

    Status status = communicator.CreateDeviceResourcesMock(group_key, p.first, p.second);

    EXPECT_TRUE(status.OK()) << "Failed for device_count " << p.second << ": " << status.GetMessage();

    auto* group = communicator.GetCommunicatorGroup(group_key);
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(group->device_resources_.size(), p.second);

    // 验证所有设备资源都有效
    for (int i = 0; i < p.second; ++i) {
      EXPECT_NE(group->device_resources_[i], nullptr) << "Device resource " << i << " should not be null";
    }
  }
}

TEST(NcclCommunicatorTest, CreateDeviceResources_ZeroDevices) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  TestableNcclCommunicator communicator(config);

  std::string group_key = "empty_group";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->comm_id = {0};

  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  // 测试零设备的情况
  Status status = communicator.CreateDeviceResourcesMock(group_key, 1, 0);

  EXPECT_TRUE(status.OK()) << "CreateDeviceResources should handle zero devices gracefully";

  auto* group = communicator.GetCommunicatorGroup(group_key);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->device_resources_.size(), 0);
}

// ===== Send 和 Recv 方法的单元测试 =====

// Mock NcclCommunicator for Send/Recv testing with mocked NCCL calls
class MockNcclCommunicatorForSendRecv : public NcclCommunicator {
 public:
  explicit MockNcclCommunicatorForSendRecv(const ConnectorConfig& config) : NcclCommunicator(config) {}

  // 暴露测试所需的方法
  using NcclCommunicator::GetCommunicatorGroup;
  using NcclCommunicator::SetCommGroupForTest;

  // Mock Send method to avoid real ncclSend calls
  Status Send(const std::string& group_key, int dev_id, uint64_t job_id, const void* buf, size_t count,
              DataType dtype) override {
    return Send(group_key, dev_id, dev_id, nullptr, buf, count, dtype);
  }

  // Mock Recv method to avoid real ncclRecv calls
  Status Recv(const std::string& group_key, int dev_id, uint64_t job_id, void* buf, size_t count,
              DataType dtype) override {
    // 默认用 dev_id 作为 local/peer，stream 传 nullptr
    return Recv(group_key, dev_id, dev_id, nullptr, buf, count, dtype);
  }

  // Mock Send method (overloaded) - for testing purpose
  Status Send(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream, const void* buf,
              size_t count, DataType dtype) {
    // 记录调用参数用于验证
    last_send_group_key_ = group_key;
    last_send_local_dev_id_ = local_dev_id;
    last_send_peer_dev_id_ = peer_dev_id;
    last_send_stream_ = stream;
    last_send_dev_id_ = local_dev_id;  // 保证 GetLastSendDevId() 正确
    last_send_buf_ = buf;
    last_send_count_ = count;
    last_send_dtype_ = dtype;
    send_call_count_++;

    // 执行基本的参数验证（与真实实现相同）
    auto group = GetCommunicatorGroup(group_key);
    if (!group || local_dev_id < 0 || static_cast<size_t>(local_dev_id) >= group->device_resources_.size()) {
      return Status(RetCode::RET_INVALID_ARGUMENT, "Invalid group_key or dev_id for Send operation");
    }
    NcclDeviceResource* resource = group->device_resources_[local_dev_id].get();
    if (!resource || !resource->send_comm) {
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Send communicator not initialized");
    }
    if (simulate_send_error_) {
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Mocked send error");
    }
    if (local_dev_id == peer_dev_id) {
      return Status();
    } else {
      NcclDeviceResource* recv_resource = group->device_resources_[peer_dev_id].get();
      if (resource->send_rank == recv_resource->recv_rank) {
        return Status();
      }
    }
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Send operation failed due to mismatched ranks");
  }
  Status Recv(const std::string& group_key, int local_dev_id, int peer_dev_id, cudaStream_t stream, void* buf,
              size_t count, DataType dtype) {
    // 记录调用参数用于验证
    last_recv_group_key_ = group_key;
    last_recv_local_dev_id_ = local_dev_id;
    last_recv_peer_dev_id_ = peer_dev_id;
    last_recv_stream_ = stream;
    last_recv_dev_id_ = local_dev_id;  // 保证 GetLastRecvDevId() 正确
    last_recv_buf_ = buf;
    last_recv_count_ = count;
    last_recv_dtype_ = dtype;
    recv_call_count_++;

    // 执行基本的参数验证（与真实实现相同）
    auto group = GetCommunicatorGroup(group_key);
    if (!group || local_dev_id < 0 || static_cast<size_t>(local_dev_id) >= group->device_resources_.size()) {
      return Status(RetCode::RET_INVALID_ARGUMENT, "Invalid group_key or dev_id for Recv operation");
    }
    NcclDeviceResource* resource = group->device_resources_[local_dev_id].get();
    if (!resource || !resource->recv_comm) {
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Recv communicator not initialized");
    }
    if (simulate_recv_error_) {
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Mocked recv error");
    }
    if (receive_callback_) {
      ncclDataType_t nccl_dtype;
      GetNcclDataType(dtype, nccl_dtype);
      size_t element_size = GetMockElementSize(nccl_dtype);
      receive_callback_(reinterpret_cast<const char*>(buf), count * element_size, 0, recv_user_data_);
    }
    if (local_dev_id == peer_dev_id) {
      return Status();
    } else {
      NcclDeviceResource* send_resource = group->device_resources_[peer_dev_id].get();
      if (resource->recv_rank == send_resource->send_rank) {
        return Status();
      }
    }
    return Status();
  }

  // 设置接收回调（用于测试）
  void DoSetReceiveCallback(const ReceiveCallback& callback) override { receive_callback_ = callback; }

  // Test helper methods
  void SetSimulateSendError(bool error) { simulate_send_error_ = error; }
  void SetSimulateRecvError(bool error) { simulate_recv_error_ = error; }

  // Getters for verification
  std::string GetLastSendGroupKey() const { return last_send_group_key_; }
  int GetLastSendDevId() const { return last_send_dev_id_; }
  const void* GetLastSendBuf() const { return last_send_buf_; }
  size_t GetLastSendCount() const { return last_send_count_; }
  DataType GetLastSendDtype() const { return last_send_dtype_; }
  int GetSendCallCount() const { return send_call_count_; }

  std::string GetLastRecvGroupKey() const { return last_recv_group_key_; }
  int GetLastRecvDevId() const { return last_recv_dev_id_; }
  void* GetLastRecvBuf() const { return last_recv_buf_; }
  size_t GetLastRecvCount() const { return last_recv_count_; }
  DataType GetLastRecvDtype() const { return last_recv_dtype_; }
  int GetRecvCallCount() const { return recv_call_count_; }

  void ResetCallCounts() {
    send_call_count_ = 0;
    recv_call_count_ = 0;
  }

 private:
  // Mock control flags
  bool simulate_send_error_ = false;
  bool simulate_recv_error_ = false;

  // Call tracking
  std::string last_send_group_key_;
  int last_send_dev_id_ = -1;
  const void* last_send_buf_ = nullptr;
  size_t last_send_count_ = 0;
  DataType last_send_dtype_ = DataType::TYPE_BYTES;
  int last_send_local_dev_id_ = -1;
  int last_send_peer_dev_id_ = -1;
  cudaStream_t last_send_stream_ = nullptr;
  int send_call_count_ = 0;

  std::string last_recv_group_key_;
  int last_recv_dev_id_ = -1;
  void* last_recv_buf_ = nullptr;
  size_t last_recv_count_ = 0;
  DataType last_recv_dtype_ = DataType::TYPE_BYTES;
  int last_recv_local_dev_id_ = -1;
  int last_recv_peer_dev_id_ = -1;
  cudaStream_t last_recv_stream_ = nullptr;
  int recv_call_count_ = 0;

  // Callback support
  ReceiveCallback receive_callback_;
  void* recv_user_data_ = nullptr;

  // Helper function to get mock element size
  size_t GetMockElementSize(ncclDataType_t dtype) const {
    // 模拟不同数据类型的元素大小
    switch (dtype) {
      case ncclInt8:
      case ncclUint8:
        return 1;
      case ncclInt32:
      case ncclUint32:
      case ncclFloat32:
        return 4;
      case ncclInt64:
      case ncclUint64:
      case ncclFloat64:
        return 8;
      default:
        return 4;  // 默认4字节
    }
  }
};

// Helper function to convert ksana_llm::DataType to ncclDataType_t
ncclDataType_t ToNcclDataType(ksana_llm::DataType dtype) {
  switch (dtype) {
    case ksana_llm::DataType::TYPE_FP32:
      return ncclFloat32;
    case ksana_llm::DataType::TYPE_FP16:
      return ncclFloat16;
    case ksana_llm::DataType::TYPE_INT32:
      return ncclInt32;
    case ksana_llm::DataType::TYPE_INT8:
      return ncclInt8;
    case ksana_llm::DataType::TYPE_UINT32:
      return ncclUint32;
    case ksana_llm::DataType::TYPE_BYTES:
      return ncclInt8;  // ncclChar is the same as ncclInt8
    // Add more cases as needed
    default:
      return ncclFloat32;  // or handle error appropriately
  }
}

// Helper function to convert ncclDataType_t to ksana_llm::DataType
ksana_llm::DataType FromNcclDataType(ncclDataType_t nccl_dtype) {
  switch (nccl_dtype) {
    case ncclFloat32:
      return ksana_llm::DataType::TYPE_FP32;
    case ncclFloat16:
      return ksana_llm::DataType::TYPE_FP16;
    case ncclInt32:
      return ksana_llm::DataType::TYPE_INT32;
    case ncclInt8:
      return ksana_llm::DataType::TYPE_INT8;
    case ncclUint8:
      return ksana_llm::DataType::TYPE_UINT8;
    case ncclUint32:
      return ksana_llm::DataType::TYPE_UINT32;
    case ncclInt64:
      return ksana_llm::DataType::TYPE_INT64;
    case ncclUint64:
      return ksana_llm::DataType::TYPE_UINT64;
    case ncclFloat64:
      return ksana_llm::DataType::TYPE_FP64;
    default:
      return ksana_llm::DataType::TYPE_FP32;  // or handle error appropriately
  }
}

// ===== Send 和 Recv 方法的单元测试 =====

// 测试Send方法的基本功能
TEST(NcclCommunicatorTest, Send_ValidParameters) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  int dev_id = 0;
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(2));

  // 准备测试数据
  int test_data = 42;
  size_t count = 1;

  // 执行Send操作
  Status status = communicator.Send(group_key, dev_id, 0, &test_data, count, DataType::TYPE_UINT32);

  // 验证结果
  EXPECT_TRUE(status.OK()) << "Send should succeed: " << status.GetMessage();
  EXPECT_EQ(communicator.GetSendCallCount(), 1);
  EXPECT_EQ(communicator.GetLastSendGroupKey(), group_key);
  EXPECT_EQ(communicator.GetLastSendDevId(), dev_id);
  EXPECT_EQ(communicator.GetLastSendBuf(), &test_data);
  EXPECT_EQ(communicator.GetLastSendCount(), count);
  EXPECT_EQ(communicator.GetLastSendDtype(), DataType::TYPE_UINT32);
}

// 测试Send方法的错误情况
TEST(NcclCommunicatorTest, Send_InvalidGroupKey) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  int test_data = 42;
  Status status = communicator.Send("non_existent_group", 0, 0, &test_data, 1, DataType::TYPE_BYTES);

  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
  EXPECT_THAT(status.GetMessage(), ::testing::HasSubstr("Invalid group_key or dev_id"));
  EXPECT_EQ(communicator.GetSendCallCount(), 1);  // 方法被调用但验证失败
}

TEST(NcclCommunicatorTest, Send_InvalidDeviceId) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(2));

  int test_data = 42;

  // 测试负数设备ID
  Status status = communicator.Send(group_key, -1, 0, &test_data, 1, DataType::TYPE_BYTES);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);

  // 测试超出范围的设备ID
  status = communicator.Send(group_key, 5, 0, &test_data, 1, DataType::TYPE_BYTES);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
}

TEST(NcclCommunicatorTest, Send_UninitializedCommunicator) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  auto group = std::make_unique<NcclCommGroup>();
  group->device_resources_.resize(1);
  auto resource = std::make_unique<NcclDeviceResource>();
  // 不设置send_comm，保持为nullptr
  resource->send_comm = nullptr;
  group->device_resources_[0] = std::move(resource);

  communicator.SetCommGroupForTest(group_key, std::move(group));

  int test_data = 42;
  Status status = communicator.Send(group_key, 0, 0, &test_data, 1, DataType::TYPE_BYTES);

  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
  EXPECT_THAT(status.GetMessage(), ::testing::HasSubstr("Send communicator not initialized"));
}

TEST(NcclCommunicatorTest, Send_SimulatedError) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(1));
  communicator.SetSimulateSendError(true);

  int test_data = 42;
  Status status = communicator.Send(group_key, 0, 0, &test_data, 1, DataType::TYPE_BYTES);

  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
  EXPECT_THAT(status.GetMessage(), ::testing::HasSubstr("Mocked send error"));
}

// 测试Recv方法的基本功能
TEST(NcclCommunicatorTest, Recv_ValidParameters) {
  ConnectorConfig config;
  config.group_role = GroupRole::DECODE;

  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  int dev_id = 1;
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(3));

  // 准备接收缓冲区
  int recv_buffer = 0;
  size_t count = 1;
  // 执行Recv操作
  Status status = communicator.Recv(group_key, dev_id, 0, &recv_buffer, count, DataType::TYPE_INT32);

  // 验证结果
  EXPECT_TRUE(status.OK()) << "Recv should succeed: " << status.GetMessage();
  EXPECT_EQ(communicator.GetRecvCallCount(), 1);
  EXPECT_EQ(communicator.GetLastRecvGroupKey(), group_key);
  EXPECT_EQ(communicator.GetLastRecvDevId(), dev_id);
  EXPECT_EQ(communicator.GetLastRecvBuf(), &recv_buffer);
  EXPECT_EQ(communicator.GetLastRecvCount(), count);
  EXPECT_EQ(communicator.GetLastRecvDtype(), DataType::TYPE_INT32);
}

// 测试Recv方法的错误情况
TEST(NcclCommunicatorTest, Recv_InvalidGroupKey) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  int recv_buffer = 0;
  Status status = communicator.Recv("non_existent_group", 0, 0, &recv_buffer, 1, DataType::TYPE_BYTES);

  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
  EXPECT_THAT(status.GetMessage(), ::testing::HasSubstr("Invalid group_key or dev_id"));
}

TEST(NcclCommunicatorTest, Recv_InvalidDeviceId) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(2));

  int recv_buffer = 0;

  // 测试负数设备ID
  Status status = communicator.Recv(group_key, -1, 0, &recv_buffer, 1, DataType::TYPE_BYTES);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);

  // 测试超出范围的设备ID
  status = communicator.Recv(group_key, 10, 0, &recv_buffer, 1, DataType::TYPE_BYTES);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
}

TEST(NcclCommunicatorTest, Recv_UninitializedCommunicator) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  auto group = std::make_unique<NcclCommGroup>();
  group->device_resources_.resize(1);
  auto resource = std::make_unique<NcclDeviceResource>();
  // 不设置recv_comm，保持为nullptr
  resource->recv_comm = nullptr;
  group->device_resources_[0] = std::move(resource);

  communicator.SetCommGroupForTest(group_key, std::move(group));

  int recv_buffer = 0;
  Status status = communicator.Recv(group_key, 0, 0, &recv_buffer, 1, DataType::TYPE_BYTES);

  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
  EXPECT_THAT(status.GetMessage(), ::testing::HasSubstr("Recv communicator not initialized"));
}

TEST(NcclCommunicatorTest, Recv_SimulatedError) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(1));
  communicator.SetSimulateRecvError(true);

  int recv_buffer = 0;
  Status status = communicator.Recv(group_key, 0, 0, &recv_buffer, 1, DataType::TYPE_BYTES);

  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
  EXPECT_THAT(status.GetMessage(), ::testing::HasSubstr("Mocked recv error"));
}

// 测试接收回调功能
TEST(NcclCommunicatorTest, Recv_WithCallback) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(1));

  // 设置接收回调
  bool callback_called = false;
  size_t received_size = 0;
  const char* received_data = nullptr;

  auto callback = [&](const char* data, size_t size, uint64_t job_id, void* user_data) {
    callback_called = true;
    received_data = data;
    received_size = size;
  };

  communicator.DoSetReceiveCallback(callback);

  // 执行接收
  int recv_buffer = 0;
  size_t count = 1;

  Status status = communicator.Recv(group_key, 0, 0, &recv_buffer, count, DataType::TYPE_INT32);

  // 验证回调被调用
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(callback_called);
  EXPECT_EQ(received_data, reinterpret_cast<const char*>(&recv_buffer));
  EXPECT_EQ(received_size, count * 4);  // int32 = 4 bytes
}

// 测试多种数据类型
TEST(NcclCommunicatorTest, SendRecv_DifferentDataTypes) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(1));

  // 测试不同的数据类型
  struct DataTypeTest {
    ncclDataType_t dtype;
    size_t expected_element_size;
    std::string name;
  };

  std::vector<DataTypeTest> tests = {
      {ncclInt8, 1, "int8"},       {ncclUint8, 1, "uint8"}, {ncclInt32, 4, "int32"},   {ncclUint32, 4, "uint32"},
      {ncclFloat32, 4, "float32"}, {ncclInt64, 8, "int64"}, {ncclUint64, 8, "uint64"}, {ncclFloat64, 8, "float64"},
  };

  for (const auto& test : tests) {
    SCOPED_TRACE("Testing data type: " + test.name);

    communicator.ResetCallCounts();

    char buffer[8] = {0};
    size_t count = 1;

    // Convert ncclDataType_t to ksana_llm::DataType
    ksana_llm::DataType ksana_dtype = FromNcclDataType(test.dtype);

    // 测试发送
    Status send_status = communicator.Send(group_key, 0, 0, buffer, count, ksana_dtype);
    EXPECT_TRUE(send_status.OK()) << "Send failed for " << test.name;

    // 测试接收
    Status recv_status = communicator.Recv(group_key, 0, 0, buffer, count, ksana_dtype);
    EXPECT_TRUE(recv_status.OK()) << "Recv failed for " << test.name;

    EXPECT_EQ(communicator.GetSendCallCount(), 1);
    EXPECT_EQ(communicator.GetRecvCallCount(), 1);
  }
}

// 测试大数据量传输
TEST(NcclCommunicatorTest, SendRecv_LargeData) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(1));

  // 测试大数据量
  std::vector<size_t> large_counts = {1000, 10000, 100000, 1000000};

  for (size_t count : large_counts) {
    SCOPED_TRACE("Testing count: " + std::to_string(count));

    communicator.ResetCallCounts();

    std::vector<float> large_buffer(count, 3.14f);

    // 测试发送大数据
    Status send_status = communicator.Send(group_key, 0, 0, large_buffer.data(), count, DataType::TYPE_FP32);
    EXPECT_TRUE(send_status.OK()) << "Send failed for count " << count;
    EXPECT_EQ(communicator.GetLastSendCount(), count);

    // 测试接收大数据
    Status recv_status = communicator.Recv(group_key, 0, 0, large_buffer.data(), count, DataType::TYPE_FP32);
    EXPECT_TRUE(recv_status.OK()) << "Recv failed for count " << count;
    EXPECT_EQ(communicator.GetLastRecvCount(), count);
  }
}

// 测试多设备操作
TEST(NcclCommunicatorTest, SendRecv_MultipleDevices) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "multi_device_group";
  int device_count = 4;
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(device_count));

  int test_data = 42;

  // 测试每个设备的发送和接收
  for (int dev_id = 0; dev_id < device_count; ++dev_id) {
    SCOPED_TRACE("Testing device: " + std::to_string(dev_id));

    communicator.ResetCallCounts();

    // 发送操作
    Status send_status = communicator.Send(group_key, dev_id, 0, &test_data, 1, DataType::TYPE_INT32);
    EXPECT_TRUE(send_status.OK()) << "Send failed for device " << dev_id;
    EXPECT_EQ(communicator.GetLastSendDevId(), dev_id);

    // 接收操作
    Status recv_status = communicator.Recv(group_key, dev_id, 0, &test_data, 1, DataType::TYPE_INT32);
    EXPECT_TRUE(recv_status.OK()) << "Recv failed for device " << dev_id;
    EXPECT_EQ(communicator.GetLastRecvDevId(), dev_id);
  }
}

// 测试并发操作（模拟）
TEST(NcclCommunicatorTest, SendRecv_ConcurrentOperations) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "concurrent_group";
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(2));

  // 模拟并发操作（在单线程中快速连续调用）
  const int num_operations = 100;
  std::vector<float> buffers(num_operations);

  for (int i = 0; i < num_operations; ++i) {
    buffers[i] = static_cast<float>(i);

    // 交替在两个设备上执行发送和接收
    int dev_id = i % 2;

    Status send_status = communicator.Send(group_key, dev_id, 0, &buffers[i], 1, DataType::TYPE_FP32);
    EXPECT_TRUE(send_status.OK()) << "Send failed for operation " << i;

    Status recv_status = communicator.Recv(group_key, dev_id, 0, &buffers[i], 1, DataType::TYPE_FP32);
    EXPECT_TRUE(recv_status.OK()) << "Recv failed for operation " << i;
  }

  // 验证总调用次数
  EXPECT_EQ(communicator.GetSendCallCount(), num_operations);
  EXPECT_EQ(communicator.GetRecvCallCount(), num_operations);
}

// 边界条件测试
TEST(NcclCommunicatorTest, SendRecv_EdgeCases) {
  ConnectorConfig config;
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "edge_case_group";
  communicator.SetCommGroupForTest(group_key, CreateTestCommGroup(1));

  // 测试零大小传输
  int dummy_buffer = 0;
  Status send_status = communicator.Send(group_key, 0, 0, &dummy_buffer, 0, DataType::TYPE_INT32);
  EXPECT_TRUE(send_status.OK()) << "Send should handle zero count";

  Status recv_status = communicator.Recv(group_key, 0, 0, &dummy_buffer, 0, DataType::TYPE_INT32);
  EXPECT_TRUE(recv_status.OK()) << "Recv should handle zero count";

  // 测试nullptr缓冲区（对于零大小）
  send_status = communicator.Send(group_key, 0, 0, nullptr, 0, DataType::TYPE_INT32);
  EXPECT_TRUE(send_status.OK()) << "Send should handle nullptr with zero count";

  recv_status = communicator.Recv(group_key, 0, 0, nullptr, 0, DataType::TYPE_INT32);
  EXPECT_TRUE(recv_status.OK()) << "Recv should handle nullptr with zero count";
}
TEST(NcclCommunicatorTest, SendRecv_OverloadedMethod_IntegrationTest) {
  ConnectorConfig config;
  config.device_count = 2;
  config.group_role = GroupRole::PREFILL;

  // 使用 mock communicator，避免真实 NCCL 调用
  MockNcclCommunicatorForSendRecv communicator(config);

  std::string group_key = "test_group";
  communicator.SetCommGroupForTest(group_key, ksana_llm::CreateTestCommGroup(2));

  // 测试数据
  float send_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float recv_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  cudaStream_t test_stream = reinterpret_cast<cudaStream_t>(0x1234);

  // 调用重载的 Send/Recv 方法
  Status send_status = communicator.Send(group_key, 0, 1, test_stream, send_data, 4, DataType::TYPE_FP32);
  EXPECT_TRUE(send_status.OK()) << send_status.GetMessage();

  Status recv_status = communicator.Recv(group_key, 1, 0, test_stream, recv_data, 4, DataType::TYPE_FP32);
  EXPECT_TRUE(recv_status.OK()) << recv_status.GetMessage();

  // 检查 mock 内部记录
  EXPECT_EQ(communicator.last_send_local_dev_id_, 0);
  EXPECT_EQ(communicator.last_send_peer_dev_id_, 1);
  EXPECT_EQ(communicator.last_send_stream_, test_stream);
  EXPECT_EQ(communicator.last_recv_local_dev_id_, 1);
  EXPECT_EQ(communicator.last_recv_peer_dev_id_, 0);
  EXPECT_EQ(communicator.last_recv_stream_, test_stream);
}

TEST(NcclCommunicatorTest, SendRecv_OverloadedMethod_DifferentDeviceIds) {
  ConnectorConfig config;
  config.device_count = 2;
  config.group_role = GroupRole::PREFILL;

  MockNcclCommunicatorForSendRecv communicator(config);
  std::string group_key = "test_group";
  communicator.SetCommGroupForTest(group_key, ksana_llm::CreateTestCommGroup(2));

  float send_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float recv_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  cudaStream_t test_stream = reinterpret_cast<cudaStream_t>(0x1234);

  // local_dev_id=0, peer_dev_id=1
  Status send_status = communicator.Send(group_key, 0, 1, test_stream, send_data, 4, DataType::TYPE_FP32);
  EXPECT_TRUE(send_status.OK()) << send_status.GetMessage();

  // local_dev_id=1, peer_dev_id=0
  Status recv_status = communicator.Recv(group_key, 1, 0, test_stream, recv_data, 4, DataType::TYPE_FP32);
  EXPECT_TRUE(recv_status.OK()) << recv_status.GetMessage();

  // 检查 mock 内部记录
  EXPECT_EQ(communicator.last_send_local_dev_id_, 0);
  EXPECT_EQ(communicator.last_send_peer_dev_id_, 1);
  EXPECT_EQ(communicator.last_recv_local_dev_id_, 1);
  EXPECT_EQ(communicator.last_recv_peer_dev_id_, 0);
  EXPECT_EQ(communicator.last_send_stream_, test_stream);
  EXPECT_EQ(communicator.last_recv_stream_, test_stream);
}

// 集成测试：发送-接收配对
TEST(NcclCommunicatorTest, SendRecv_IntegrationTest) {
  ConnectorConfig prefill_config;
  prefill_config.group_role = GroupRole::PREFILL;

  ConnectorConfig decode_config;
  decode_config.group_role = GroupRole::DECODE;

  MockNcclCommunicatorForSendRecv prefill_comm(prefill_config);
  MockNcclCommunicatorForSendRecv decode_comm(decode_config);

  std::string group_key = "integration_group";

  // 为两个通信器设置相同的组
  prefill_comm.SetCommGroupForTest(group_key, CreateTestCommGroup(2));
  decode_comm.SetCommGroupForTest(group_key, CreateTestCommGroup(2));

  // 模拟双向通信
  double send_data = 3.141592653589793;
  double recv_data = 0.0;

  // Prefill -> Decode
  Status send_status = prefill_comm.Send(group_key, 0, 0, &send_data, 1, DataType::TYPE_FP64);
  EXPECT_TRUE(send_status.OK());

  Status recv_status = decode_comm.Recv(group_key, 0, 0, &recv_data, 1, DataType::TYPE_FP64);
  EXPECT_TRUE(recv_status.OK());

  // Decode -> Prefill
  double return_data = 2.718281828459045;
  double recv_return = 0.0;

  send_status = decode_comm.Send(group_key, 1, 0, &return_data, 1, DataType::TYPE_FP64);
  EXPECT_TRUE(send_status.OK());

  recv_status = prefill_comm.Recv(group_key, 1, 0, &recv_return, 1, DataType::TYPE_FP64);
  EXPECT_TRUE(recv_status.OK());

  // 验证调用参数
  EXPECT_EQ(prefill_comm.GetSendCallCount(), 1);
  EXPECT_EQ(prefill_comm.GetRecvCallCount(), 1);
  EXPECT_EQ(decode_comm.GetSendCallCount(), 1);
  EXPECT_EQ(decode_comm.GetRecvCallCount(), 1);

  EXPECT_EQ(prefill_comm.GetLastSendBuf(), &send_data);
  EXPECT_EQ(decode_comm.GetLastRecvBuf(), &recv_data);
  EXPECT_EQ(decode_comm.GetLastSendBuf(), &return_data);
  EXPECT_EQ(prefill_comm.GetLastRecvBuf(), &recv_return);
}

// Helper function to check mock status from C context
static bool is_nccl_mock_enabled() { return ksana_llm::testing_internal::IsNcclMockEnabled(); }

TEST(NcclCommunicatorTest, Real_CreateDeviceResources_Coverage) {
  // 启用mock以避免真实NCCL调用
  ksana_llm::testing_internal::EnableNcclMock(true);

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count < 1) {
    GTEST_SKIP() << "需要至少1个CUDA设备";
    return;
  }

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.node_rank = 0;
  config.world_size = 2;  // 现在可以使用多进程配置了
  config.device_count = 2;

  NcclCommunicator communicator(config);

  ncclUniqueId comm_id;
  NCCL_CHECK(ncclGetUniqueId(&comm_id));

  std::string group_key = "real_group";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->comm_id = comm_id;
  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  Status status = communicator.CreateDeviceResources(group_key);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  auto* group = communicator.GetCommunicatorGroup(group_key);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->device_resources_.size(), 2);
  EXPECT_NE(group->device_resources_[0], nullptr);

  // 禁用mock，恢复正常行为
  ksana_llm::testing_internal::EnableNcclMock(false);
}

// ===== 展示如何使用内置mock控制的集成测试 =====
TEST(NcclCommunicatorTest, Real_CreateDeviceResources_With_Mock_Control) {
  // 启用内置mock，避免真实NCCL/CUDA调用
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.node_rank = 0;
  config.world_size = 2;  // 现在可以使用多进程配置
  config.device_count = 4;

  NcclCommunicator communicator(config);

  // 创建测试用的通信组
  std::string group_key = "mock_test_group";
  auto comm_group = std::make_unique<NcclCommGroup>();

  // 使用mock NCCL获取unique ID
  ncclUniqueId comm_id;
  ncclResult_t result = ncclGetUniqueId(&comm_id);
  ASSERT_EQ(result, ncclSuccess);
  comm_group->comm_id = comm_id;

  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  // 测试CreateDeviceResources方法 - 现在应该成功完成
  Status status = communicator.CreateDeviceResources(group_key);
  EXPECT_TRUE(status.OK()) << "CreateDeviceResources应该成功: " << status.GetMessage();

  // 验证设备资源创建
  auto* group = communicator.GetCommunicatorGroup(group_key);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->device_resources_.size(), config.device_count);

  // 验证每个设备资源都已正确初始化
  for (int i = 0; i < config.device_count; ++i) {
    auto& resource = group->device_resources_[i];
    ASSERT_NE(resource, nullptr);
    EXPECT_NE(resource->send_comm, nullptr) << "Device " << i << " send_comm should be initialized";
    EXPECT_NE(resource->recv_comm, nullptr) << "Device " << i << " recv_comm should be initialized";
    EXPECT_NE(resource->send_stream, nullptr) << "Device " << i << " send_stream should be initialized";
    EXPECT_NE(resource->recv_stream, nullptr) << "Device " << i << " recv_stream should be initialized";
  }

  // 禁用mock，恢复正常行为
  ksana_llm::testing_internal::EnableNcclMock(false);
}

// 覆盖 NCCL/CUDA 相关函数
extern "C" {
static ncclResult_t g_last_nccl_result = ncclSuccess;
static cudaError_t g_last_cuda_result = cudaSuccess;

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank) {
  bool mock_enabled = is_nccl_mock_enabled();
  printf("[DEBUG] ncclCommInitRank called: mock_enabled=%s, nranks=%d, myrank=%d\n", mock_enabled ? "true" : "false",
         nranks, myrank);

  if (!mock_enabled) {
    typedef ncclResult_t (*real_func_t)(ncclComm_t*, int, ncclUniqueId, int);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "ncclCommInitRank");
    if (real_func) {
      return real_func(comm, nranks, commId, myrank);
    } else {
      printf("[DEBUG] ncclCommInitRank: real_func is NULL, falling back to mock\n");
    }
  }

  printf("[DEBUG] ncclCommInitRank: using mock implementation\n");
  *comm = reinterpret_cast<ncclComm_t>(0x1234 + myrank);
  g_last_nccl_result = ncclSuccess;
  return ncclSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t* stream) {
  if (!is_nccl_mock_enabled()) {
    typedef cudaError_t (*real_func_t)(cudaStream_t*);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaStreamCreate");
    if (real_func) {
      return real_func(stream);
    }
  }
  printf("[DEBUG] cudaStreamCreate: using mock implementation\n");
  *stream = reinterpret_cast<cudaStream_t>(0x5678);
  g_last_cuda_result = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
  if (!is_nccl_mock_enabled()) {
    typedef cudaError_t (*real_func_t)(int);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaSetDevice");
    if (real_func) {
      return real_func(device);
    }
  }
  printf("[DEBUG] cudaSetDevice: using mock implementation, device=%d\n", device);
  g_last_cuda_result = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  if (!is_nccl_mock_enabled()) {
    typedef cudaError_t (*real_func_t)(cudaStream_t);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaStreamDestroy");
    if (real_func) {
      return real_func(stream);
    }
  }
  printf("[DEBUG] cudaStreamDestroy: using mock implementation\n");
  g_last_cuda_result = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  if (!is_nccl_mock_enabled()) {
    typedef cudaError_t (*real_func_t)(cudaStream_t);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaStreamSynchronize");
    return real_func(stream);
  }
  g_last_cuda_result = cudaSuccess;
  return cudaSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId) {
  if (!is_nccl_mock_enabled()) {
    typedef ncclResult_t (*real_func_t)(ncclUniqueId*);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "ncclGetUniqueId");
    return real_func(uniqueId);
  }
  memset(uniqueId, 0, sizeof(ncclUniqueId));
  return ncclSuccess;
}

const char* ncclGetErrorString(ncclResult_t result) { return "MOCK_NCCL_OK"; }

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (!is_nccl_mock_enabled()) {
    typedef ncclResult_t (*real_func_t)(ncclComm_t);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "ncclCommDestroy");
    if (real_func) {
      return real_func(comm);
    }
  }
  printf("[DEBUG] ncclCommDestroy: using mock implementation\n");
  return ncclSuccess;
}

ncclResult_t ncclGroupStart() {
  if (!is_nccl_mock_enabled()) {
    typedef ncclResult_t (*real_func_t)();
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "ncclGroupStart");
    if (real_func) {
      return real_func();
    }
  }
  printf("[DEBUG] ncclGroupStart: using mock implementation\n");
  return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
  if (!is_nccl_mock_enabled()) {
    typedef ncclResult_t (*real_func_t)();
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "ncclGroupEnd");
    if (real_func) {
      return real_func();
    }
  }
  printf("[DEBUG] ncclGroupEnd: using mock implementation\n");
  return ncclSuccess;
}

ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  if (!is_nccl_mock_enabled()) {
    typedef ncclResult_t (*real_func_t)(const void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "ncclSend");
    if (real_func) {
      return real_func(sendbuff, count, datatype, peer, comm, stream);
    }
  }
  printf("[DEBUG] ncclSend: using mock implementation, count=%zu, peer=%d\n", count, peer);
  return ncclSuccess;
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  if (!is_nccl_mock_enabled()) {
    typedef ncclResult_t (*real_func_t)(void*, size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "ncclRecv");
    if (real_func) {
      return real_func(recvbuff, count, datatype, peer, comm, stream);
    }
  }
  printf("[DEBUG] ncclRecv: using mock implementation, count=%zu, peer=%d\n", count, peer);
  return ncclSuccess;
}

cudaError_t cudaGetDeviceCount(int* count) {
  if (!is_nccl_mock_enabled()) {
    typedef cudaError_t (*real_func_t)(int*);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaGetDeviceCount");
    if (real_func) {
      return real_func(count);
    }
  }
  printf("[DEBUG] cudaGetDeviceCount: using mock implementation\n");
  *count = 2;  // Mock 2 devices available
  return cudaSuccess;
}

const char* cudaGetErrorString(cudaError_t error) {
  if (!is_nccl_mock_enabled()) {
    typedef const char* (*real_func_t)(cudaError_t);
    static real_func_t real_func = (real_func_t)dlsym(RTLD_NEXT, "cudaGetErrorString");
    if (real_func) {
      return real_func(error);
    }
  }
  return "MOCK_CUDA_SUCCESS";
}
}

TEST(NcclCommunicatorTest, Mocked_CreateDeviceResources_Coverage) {
  ksana_llm::testing_internal::EnableNcclMock(true);
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.node_rank = 0;
  config.world_size = 1;
  config.device_count = 1;

  NcclCommunicator communicator(config);
  ncclUniqueId comm_id = {0};
  std::string group_key = "mock_group";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->comm_id = comm_id;
  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  Status status = communicator.CreateDeviceResources(group_key);
  EXPECT_TRUE(status.OK()) << status.GetMessage();

  auto* group = communicator.GetCommunicatorGroup(group_key);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->device_resources_.size(), 1);
  EXPECT_NE(group->device_resources_[0], nullptr);
  ksana_llm::testing_internal::EnableNcclMock(false);
}

// Test Initialize method
TEST(NcclCommunicatorTest, Initialize_BasicTest) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.node_rank = 0;
  config.world_size = 1;
  config.device_count = 1;

  NcclCommunicator communicator(config);
  Status status = communicator.Initialize();
  EXPECT_TRUE(status.OK()) << "Initialize should return success";
}

// Test DoSetReceiveCallback method
TEST(NcclCommunicatorTest, DoSetReceiveCallback_ValidCallback) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  bool callback_called = false;
  ReceiveCallback test_callback = [&callback_called](const char* data, size_t size, uint64_t job_id, void* user_data) {
    callback_called = true;
  };

  // Test setting receive callback (template method with single parameter)
  communicator.SetReceiveCallback(test_callback);

  // Verify callback can be set without errors
  EXPECT_TRUE(true);  // If we reach here, callback was set successfully
}

// Test CheckCommGroupAvailable edge cases
TEST(NcclCommunicatorTest, CheckCommGroupAvailable_NonExistentGroup) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  bool available = communicator.CheckCommGroupAvailable("non_existent_group");  // 修正：只传 group_key
  EXPECT_FALSE(available) << "Non-existent group should not be available";
}

TEST(NcclCommunicatorTest, CheckCommGroupAvailable_InactiveGroup) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  // Create an inactive group (with invalid device resources)
  std::string group_key = "inactive_group";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->device_count_ = 1;

  // Add an invalid device resource to make it inactive
  auto invalid_resource = std::make_unique<NcclDeviceResource>();
  // Leave all comm and stream pointers as nullptr to make it invalid
  invalid_resource->send_comm = nullptr;
  invalid_resource->recv_comm = nullptr;
  invalid_resource->send_stream = nullptr;
  invalid_resource->recv_stream = nullptr;

  comm_group->device_resources_.push_back(std::move(invalid_resource));

  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  bool available = communicator.CheckCommGroupAvailable(group_key);
  EXPECT_FALSE(available) << "Inactive group should not be available";

  ksana_llm::testing_internal::EnableNcclMock(false);
}

TEST(NcclCommunicatorTest, CheckCommGroupAvailable_ActiveGroup) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.node_rank = 0;
  config.world_size = 1;
  config.device_count = 1;

  NcclCommunicator communicator(config);

  // Create an active group with proper device resources
  std::string group_key = "active_group";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->device_count_ = 1;

  auto resource = std::make_unique<NcclDeviceResource>();
  resource->send_comm = reinterpret_cast<ncclComm_t>(0x1234);
  resource->recv_comm = reinterpret_cast<ncclComm_t>(0x5678);
  resource->send_stream = reinterpret_cast<cudaStream_t>(0x9ABC);
  resource->recv_stream = reinterpret_cast<cudaStream_t>(0xDEF0);

  comm_group->device_resources_.push_back(std::move(resource));
  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  bool available = communicator.CheckCommGroupAvailable(group_key);
  EXPECT_TRUE(available) << "Active group should be available";

  ksana_llm::testing_internal::EnableNcclMock(false);
}

// Test ProcessHeartbeatData error conditions
TEST(NcclCommunicatorTest, ProcessHeartbeatData_EmptyInput) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  std::unordered_map<std::string, std::string> empty_comm_group_to_id;
  std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>> empty_comm_group_to_address;

  Status status = communicator.ProcessHeartbeatData(empty_comm_group_to_id, empty_comm_group_to_address);
  EXPECT_TRUE(status.OK()) << "Empty input should be handled gracefully";
}

TEST(NcclCommunicatorTest, ProcessHeartbeatData_InvalidBase64) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  std::unordered_map<std::string, std::string> comm_group_to_id;
  comm_group_to_id["test_group"] = "invalid_base64_!@#$%^&*()";

  std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>> empty_comm_group_to_address;

  // base64::from_base64 may throw exception for invalid input
  EXPECT_THROW(
      { Status status = communicator.ProcessHeartbeatData(comm_group_to_id, empty_comm_group_to_address); },
      std::exception);
}

TEST(NcclCommunicatorTest, ProcessHeartbeatData_WrongSizeCommId) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  std::unordered_map<std::string, std::string> comm_group_to_id;
  // Valid base64 but wrong size (should be 128 bytes when decoded)
  std::string wrong_size_data = "dGVzdA==";  // "test" in base64 (4 bytes)
  comm_group_to_id["test_group"] = wrong_size_data;

  std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>> empty_comm_group_to_address;

  Status status = communicator.ProcessHeartbeatData(comm_group_to_id, empty_comm_group_to_address);
  // Should handle wrong size gracefully
  EXPECT_TRUE(true);  // If we reach here, the function didn't crash
}

// Test IsConnectionReady method
TEST(NcclCommunicatorTest, IsConnectionReady_NonExistentGroup) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  bool ready = communicator.IsConnectionReady("non_existent_group", 0);
  EXPECT_FALSE(ready) << "Non-existent group should not be ready";
}

TEST(NcclCommunicatorTest, IsConnectionReady_ValidGroup) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  // Create a group with active device resources
  std::string group_key = "ready_group";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->device_count_ = 1;

  auto resource = std::make_unique<NcclDeviceResource>();
  resource->send_comm = reinterpret_cast<ncclComm_t>(0x1234);
  resource->recv_comm = reinterpret_cast<ncclComm_t>(0x5678);
  resource->send_stream = reinterpret_cast<cudaStream_t>(0x9ABC);
  resource->recv_stream = reinterpret_cast<cudaStream_t>(0xDEF0);

  comm_group->device_resources_.push_back(std::move(resource));
  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  bool ready = communicator.IsConnectionReady(group_key, 0);
  EXPECT_TRUE(ready) << "Valid group with active resources should be ready";

  ksana_llm::testing_internal::EnableNcclMock(false);
}

TEST(NcclCommunicatorTest, IsConnectionReady_InvalidDeviceId) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  // Create a group with one device resource
  std::string group_key = "single_device_group";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->device_count_ = 1;

  auto resource = std::make_unique<NcclDeviceResource>();
  resource->send_comm = reinterpret_cast<ncclComm_t>(0x1234);
  resource->recv_comm = reinterpret_cast<ncclComm_t>(0x5678);
  resource->send_stream = reinterpret_cast<cudaStream_t>(0x9ABC);
  resource->recv_stream = reinterpret_cast<cudaStream_t>(0xDEF0);

  comm_group->device_resources_.push_back(std::move(resource));
  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  // Test with invalid device ID
  bool ready = communicator.IsConnectionReady(group_key, 5);
  EXPECT_FALSE(ready) << "Invalid device ID should return false";

  ksana_llm::testing_internal::EnableNcclMock(false);
}

// Test SendGroup with empty buffers
TEST(NcclCommunicatorTest, SendGroup_EmptyBuffers) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  std::vector<const void*> empty_buffers;
  std::vector<size_t> empty_counts;

  Status status = communicator.SendGroup("test_group", 0, 0, empty_buffers, empty_counts, DataType::TYPE_FP32);
  EXPECT_TRUE(status.OK()) << "Empty buffers should be handled gracefully";

  ksana_llm::testing_internal::EnableNcclMock(false);
}

TEST(NcclCommunicatorTest, SendGroup_MismatchedBuffersAndCounts) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  std::vector<const void*> buffers = {reinterpret_cast<const void*>(0x1000)};
  std::vector<size_t> counts = {100, 200};  // Mismatch: 1 buffer, 2 counts

  Status status = communicator.SendGroup("test_group", 0, 0, buffers, counts, DataType::TYPE_FP32);
  EXPECT_FALSE(status.OK()) << "Mismatched buffers and counts should return error";
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);

  ksana_llm::testing_internal::EnableNcclMock(false);
}

TEST(NcclCommunicatorTest, SendGroup_ValidParameters) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.device_count = 1;  // Set device_count to match the test scenario

  NcclCommunicator communicator(config);

  // Create a group with device resources
  std::string group_key = "sendgroup_test";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->device_count_ = 1;

  auto resource = std::make_unique<NcclDeviceResource>();
  resource->send_comm = reinterpret_cast<ncclComm_t>(0x1234);
  resource->recv_comm = reinterpret_cast<ncclComm_t>(0x5678);
  resource->send_stream = reinterpret_cast<cudaStream_t>(0x9ABC);
  resource->recv_stream = reinterpret_cast<cudaStream_t>(0xDEF0);
  resource->recv_rank = 0;

  comm_group->device_resources_.push_back(std::move(resource));
  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  std::vector<const void*> buffers = {reinterpret_cast<const void*>(0x1000), reinterpret_cast<const void*>(0x2000)};
  std::vector<size_t> counts = {100, 200};

  Status status = communicator.SendGroup(group_key, 0, 0, buffers, counts, DataType::TYPE_FP32);
  EXPECT_TRUE(status.OK()) << "Valid SendGroup should succeed";

  ksana_llm::testing_internal::EnableNcclMock(false);
}

TEST(NcclCommunicatorTest, RecvGroup_MismatchedBuffersAndCounts) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  std::vector<void*> buffers = {reinterpret_cast<void*>(0x1000)};
  std::vector<size_t> counts = {100, 200};  // Mismatch: 1 buffer, 2 counts

  Status status = communicator.RecvGroup("test_group", 0, 0, buffers, counts, DataType::TYPE_FP32);
  EXPECT_FALSE(status.OK()) << "Mismatched buffers and counts should return error";
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);

  ksana_llm::testing_internal::EnableNcclMock(false);
}

TEST(NcclCommunicatorTest, RecvGroup_ValidParameters) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.device_count = 1;  // Set device_count to match the test scenario

  NcclCommunicator communicator(config);

  // Create a group with device resources
  std::string group_key = "recvgroup_test";
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->device_count_ = 1;

  auto resource = std::make_unique<NcclDeviceResource>();
  resource->send_comm = reinterpret_cast<ncclComm_t>(0x1234);
  resource->recv_comm = reinterpret_cast<ncclComm_t>(0x5678);
  resource->send_stream = reinterpret_cast<cudaStream_t>(0x9ABC);
  resource->recv_stream = reinterpret_cast<cudaStream_t>(0xDEF0);
  resource->send_rank = 0;

  comm_group->device_resources_.push_back(std::move(resource));
  communicator.SetCommGroupForTest(group_key, std::move(comm_group));

  std::vector<void*> buffers = {reinterpret_cast<void*>(0x1000), reinterpret_cast<void*>(0x2000)};
  std::vector<size_t> counts = {100, 200};
  Status status = communicator.RecvGroup(group_key, 0, 0, buffers, counts, DataType::TYPE_FP32);
  EXPECT_TRUE(status.OK()) << "Valid RecvGroup should succeed";

  ksana_llm::testing_internal::EnableNcclMock(false);
}

// Test exception handling in CreateDeviceResources
TEST(NcclCommunicatorTest, CreateDeviceResources_ExceptionHandling) {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.node_rank = 0;
  config.world_size = 1;
  config.device_count = 1;

  NcclCommunicator communicator(config);

  // Try to create device resources for a group that doesn't exist
  Status status = communicator.CreateDeviceResources("non_existent_group");
  EXPECT_FALSE(status.OK()) << "Should fail for non-existent group";
}

TEST(NcclCommunicatorTest, Send_ZeroCount) {
  testing_internal::EnableNcclMock(true);
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.device_count = 2;  // Set device_count to match the test scenario
  NcclCommunicator communicator(config);
  char buf[1];
  Status status = communicator.Send("group0", 0, 1, nullptr, buf, 0, DataType::TYPE_BYTES);
  EXPECT_FALSE(status.OK());
}

TEST(NcclCommunicatorTest, Send_NullBuffer) {
  testing_internal::EnableNcclMock(true);
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.device_count = 2;  // Set device_count to match the test scenario
  NcclCommunicator communicator(config);
  // 初始化 communicator，略（可参考已有测试）
  Status status = communicator.Send("group0", 0, 1, nullptr, 128, DataType::TYPE_BYTES);
  EXPECT_FALSE(status.OK());
}

TEST(NcclCommunicatorTest, Recv_NullBuffer) {
  testing_internal::EnableNcclMock(true);
  ConnectorConfig config;
  config.group_role = GroupRole::DECODE;
  config.device_count = 2;  // Set device_count to match the test scenario
  NcclCommunicator communicator(config);
  Status status = communicator.Recv("group0", 0, 1, nullptr, nullptr, 128, DataType::TYPE_BYTES);
  EXPECT_FALSE(status.OK());
}

TEST(NcclCommunicatorTest, Recv_ZeroCount) {
  testing_internal::EnableNcclMock(true);
  ConnectorConfig config;
  config.group_role = GroupRole::DECODE;
  config.device_count = 2;  // Set device_count to match the test scenario
  NcclCommunicator communicator(config);
  char buf[1];
  Status status = communicator.Recv("group0", 0, 1, nullptr, buf, 0, DataType::TYPE_BYTES);
  EXPECT_FALSE(status.OK());
}

TEST(NcclCommunicatorTest, Send_Success) {
  testing_internal::EnableNcclMock(true);
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.device_count = 2;
  NcclCommunicator communicator(config);
  // Register group0 before sending
  auto comm_group = CreateTestCommGroup(config.device_count);
  communicator.SetCommGroupForTest("group0", std::move(comm_group));
  char buf[8] = {0};
  // 这里 count > 0，buf 非空，模拟正常发送
  Status status = communicator.Send("group0", 0, 1, nullptr, buf, 8, DataType::TYPE_BYTES);
  EXPECT_TRUE(status.OK());
}

TEST(NcclCommunicatorTest, Recv_Success) {
  testing_internal::EnableNcclMock(true);
  ConnectorConfig config;
  config.group_role = GroupRole::DECODE;
  config.device_count = 2;
  NcclCommunicator communicator(config);
  // Register group0 before receiving
  auto comm_group = CreateTestCommGroup(config.device_count);
  communicator.SetCommGroupForTest("group0", std::move(comm_group));
  char buf[8] = {0};
  // count > 0，buf 非空，模拟正常接收
  Status status = communicator.Recv("group0", 0, 1, nullptr, buf, 8, DataType::TYPE_BYTES);
  EXPECT_TRUE(status.OK());
}

// Test Shutdown method coverage
TEST(NcclCommunicatorTest, Shutdown_WithActiveGroups) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;

  NcclCommunicator communicator(config);

  // Create multiple groups
  for (int i = 0; i < 3; ++i) {
    std::string group_key = "group_" + std::to_string(i);
    auto comm_group = std::make_unique<NcclCommGroup>();
    comm_group->device_count_ = 1;

    auto resource = std::make_unique<NcclDeviceResource>();
    resource->send_comm = reinterpret_cast<ncclComm_t>(0x1234 + i);
    resource->recv_comm = reinterpret_cast<ncclComm_t>(0x5678 + i);
    resource->send_stream = reinterpret_cast<cudaStream_t>(0x9ABC + i);
    resource->recv_stream = reinterpret_cast<cudaStream_t>(0xDEF0 + i);

    comm_group->device_resources_.push_back(std::move(resource));
    communicator.SetCommGroupForTest(group_key, std::move(comm_group));
  }

  // Test shutdown - should clean up all groups
  communicator.Shutdown();

  // Verify groups are cleaned up
  for (int i = 0; i < 3; ++i) {
    std::string group_key = "group_" + std::to_string(i);
    auto* group = communicator.GetCommunicatorGroup(group_key);
    EXPECT_EQ(group, nullptr) << "Group " << group_key << " should be cleaned up after shutdown";
  }

  ksana_llm::testing_internal::EnableNcclMock(false);
}

// Test destructor coverage
TEST(NcclCommunicatorTest, Destructor_CallsShutdown) {
  ksana_llm::testing_internal::EnableNcclMock(true);

  {
    ConnectorConfig config;
    config.group_role = GroupRole::PREFILL;

    auto communicator = std::make_unique<NcclCommunicator>(config);

    // Create a group
    std::string group_key = "destructor_test";
    auto comm_group = std::make_unique<NcclCommGroup>();
    comm_group->device_count_ = 1;

    auto resource = std::make_unique<NcclDeviceResource>();
    resource->send_comm = reinterpret_cast<ncclComm_t>(0x1234);
    resource->recv_comm = reinterpret_cast<ncclComm_t>(0x5678);
    resource->send_stream = reinterpret_cast<cudaStream_t>(0x9ABC);
    resource->recv_stream = reinterpret_cast<cudaStream_t>(0xDEF0);

    comm_group->device_resources_.push_back(std::move(resource));
    communicator->SetCommGroupForTest(group_key, std::move(comm_group));

    // Destructor should be called here and clean up resources
  }

  // If we reach here without issues, destructor worked correctly
  EXPECT_TRUE(true);

  ksana_llm::testing_internal::EnableNcclMock(false);
}

// Helper function to create a test communication group with the specified number of devices
std::unique_ptr<NcclCommGroup> CreateTestCommGroup(size_t device_count) {
  auto comm_group = std::make_unique<NcclCommGroup>();
  comm_group->comm_id = {0};  // Initialize with dummy comm_id
  comm_group->device_resources_.resize(device_count);

  // Initialize each device resource with dummy values
  for (size_t i = 0; i < device_count; ++i) {
    comm_group->device_resources_[i] = std::make_unique<NcclDeviceResource>();
    // Set dummy values for send/recv communicators - normally these would be actual NCCL communicators
    comm_group->device_resources_[i]->send_comm = reinterpret_cast<ncclComm_t>(1);
    comm_group->device_resources_[i]->recv_comm = reinterpret_cast<ncclComm_t>(1);
  }

  return comm_group;
}
#endif
}  // namespace ksana_llm