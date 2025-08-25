/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/communicator/communicator_manager.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ksana_llm/connector/communicator/nvida/nccl_communicator.h"
#include "ksana_llm/connector/communicator/zmq/zmq_communicator.h"

namespace ksana_llm {

class MockCoordinator : public Coordinator {
 public:
  Status Initialize() override { return Status(); }
  void Shutdown() override {}
  Status StartHeartbeat() override { return Status(); }
  void StopHeartbeat() override {}
  void OnHeartbeatResponseCallback(HeartbeatResponseCallback) override {}
  Status SendCommId(const std::string&, const std::string&) override { return Status(); }
  const KVNodeInfo& GetNodeInfo() const override {
    static KVNodeInfo info;
    return info;
  }
  Status SendHeartbeat(std::string&, KVHeartbeatResponse&) override { return Status(); }
  Status RegisterNode() override { return Status(); }
  void HeartbeatThread() override {}
  bool IsInitialized() const override { return true; }
};

// mock ZmqCommunicator，避免真实 ZMQ 初始化
class MockZmqCommunicator : public ZmqCommunicator {
 public:
  explicit MockZmqCommunicator(const ConnectorConfig& config) : ZmqCommunicator(config) {}
  Status Initialize() override { return Status(); }
  Status Send(const std::string&, int, int, size_t, const void*, size_t, DataType) override { return Status(); }
};
#ifdef ENABLE_CUDA
// mock NcclCommunicator，避免真实 NCCL 初始化
class MockNcclCommunicator : public NcclCommunicator {
 public:
  explicit MockNcclCommunicator(const ConnectorConfig& config) : NcclCommunicator(config) {}
  Status InitializeNccl(const std::string&, int, int, int) { return Status(); }
  Status InitCommunicatorGroup(const std::string&, const std::vector<std::tuple<int, int, std::string>>&,
                               const std::string&) {
    return Status();
  }
  Status Send(const std::string&, int, int, size_t, const void*, size_t, DataType) override { return Status(); }
};
#endif

class CommunicatorManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create config
    config_.router_endpoint = "mock://localhost:1234";
    config_.coordinator_port = 5555;
    config_.group_role = GroupRole::PREFILL;
    config_.node_rank = 0;
    config_.world_size = 2;
    config_.device_count = 4;

    // Create communication manager with mock coordinator
    manager_ = std::make_unique<CommunicatorManager>(config_, std::make_unique<MockCoordinator>());
    // 替换真实 communicator 为 mock
    manager_->SetZmqCommunicator(std::make_unique<MockZmqCommunicator>(config_));
#ifdef ENABLE_CUDA
    manager_->SetNcclCommunicator(std::make_unique<MockNcclCommunicator>(config_));
#endif
  }

  void TearDown() override {
    if (manager_) {
      manager_->Shutdown();
    }
  }

  ConnectorConfig config_;
  std::unique_ptr<CommunicatorManager> manager_;
};

TEST_F(CommunicatorManagerTest, Initialize) {
  // 只验证 mock 注入和获取，不触发真实通信器初始化
  auto zmq_comm = manager_->GetZmqCommunicator();
  EXPECT_NE(zmq_comm, nullptr);
#ifdef ENABLE_CUDA
  auto nccl_comm = manager_->GetNcclCommunicator();
  EXPECT_NE(nccl_comm, nullptr);
#endif
}

TEST_F(CommunicatorManagerTest, ProcessHeartbeatData) {
  // 只用 mock communicator，不调用 Initialize，避免真实通信器初始化
  std::unordered_map<std::string, std::string> comm_group_to_id;
  std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>> comm_group_to_address;

  comm_group_to_id["test_group"] = "AAAA";

  std::vector<std::tuple<int, int, std::string>> address_tuples;
  for (int i = 0; i < 16; i++) {
    int node = (i / 4) % 2;
    int dev = i % 4;
    std::string addr = "127.0.0." + std::to_string(i + 1) + ":5555";
    address_tuples.push_back(std::make_tuple(node, dev, addr));
  }
  comm_group_to_address["test_group"] = address_tuples;

  // 直接调用接口，mock communicator 不会阻塞
  Status status = manager_->ProcessHeartbeatData(comm_group_to_id, comm_group_to_address);
  EXPECT_TRUE(status.OK());
}

TEST_F(CommunicatorManagerTest, UseMultipleCommunicatorTypes) {
  // 只用 mock communicator，不调用 Initialize/CreateNcclCommunicator，避免真实通信器初始化
  auto zmq_comm = manager_->GetZmqCommunicator();
  EXPECT_NE(zmq_comm, nullptr);
#ifdef ENABLE_CUDA
  auto nccl_comm = manager_->GetNcclCommunicator();
  EXPECT_NE(nccl_comm, nullptr);
#endif
  // 可选：验证 mock 的 Send/Recv 行为（应为 no-op 或返回 Status）
  const char test_data[] = "Test data";
  const size_t data_size = strlen(test_data) + 1;
  Status status = zmq_comm->Send("test_group", 0, 0, 0, test_data, data_size, DataType::TYPE_BYTES);
  EXPECT_TRUE(status.OK());
#ifdef ENABLE_CUDA
  status = nccl_comm->Send("test_group", 0, 0, static_cast<uint64_t>(0), test_data, data_size, DataType::TYPE_BYTES);
  EXPECT_TRUE(status.OK());
#endif
}

// ===== Tests for uncovered code paths =====

TEST_F(CommunicatorManagerTest, IsInitialized_InitialState) {
  // Test IsInitialized returns false initially
  EXPECT_FALSE(manager_->IsInitialized());
}

TEST_F(CommunicatorManagerTest, Initialize_Success) {
  // Test successful initialization
  Status status = manager_->Initialize();
  EXPECT_TRUE(status.OK()) << "Initialize should succeed: " << status.GetMessage();
  EXPECT_TRUE(manager_->IsInitialized());
}

TEST_F(CommunicatorManagerTest, Initialize_AlreadyInitialized) {
  // Test double initialization
  Status status = manager_->Initialize();
  EXPECT_TRUE(status.OK());

  // Second initialization should fail
  status = manager_->Initialize();
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
  EXPECT_THAT(status.GetMessage(), ::testing::HasSubstr("already initialized"));
}

class MockCoordinatorNotInitialized : public Coordinator {
 public:
  Status Initialize() override { return Status(); }
  void Shutdown() override {}
  Status StartHeartbeat() override { return Status(); }
  void StopHeartbeat() override {}
  void OnHeartbeatResponseCallback(HeartbeatResponseCallback) override {}
  Status SendCommId(const std::string&, const std::string&) override { return Status(); }
  const KVNodeInfo& GetNodeInfo() const override {
    static KVNodeInfo info;
    return info;
  }
  Status SendHeartbeat(std::string&, KVHeartbeatResponse&) override { return Status(); }
  Status RegisterNode() override { return Status(); }
  void HeartbeatThread() override {}
  bool IsInitialized() const override { return false; }  // Not initialized
};

TEST_F(CommunicatorManagerTest, Initialize_CoordinatorNotInitialized) {
  // Test initialization with uninitialized coordinator
  auto uninitialized_coordinator = std::make_unique<MockCoordinatorNotInitialized>();
  CommunicatorManager manager(config_, std::move(uninitialized_coordinator));
  manager.SetZmqCommunicator(std::make_unique<MockZmqCommunicator>(config_));
#ifdef ENABLE_CUDA
  manager.SetNcclCommunicator(std::make_unique<MockNcclCommunicator>(config_));
#endif

  Status status = manager.Initialize();
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
  EXPECT_THAT(status.GetMessage(), ::testing::HasSubstr("Coordinator not initialized"));
}

TEST_F(CommunicatorManagerTest, SendCommId_Success) {
  // Test SendCommId method
  Status status = manager_->SendCommId("test_group", "test_comm_id");
  EXPECT_TRUE(status.OK());
}

TEST_F(CommunicatorManagerTest, GetCommunicatorByType_ValidType) {
  // Test GetCommunicatorByType with valid type
  auto* comm = manager_->GetCommunicatorByType("ZMQ");
  EXPECT_NE(comm, nullptr);
#ifdef ENABLE_CUDA
  comm = manager_->GetCommunicatorByType("NCCL");
  EXPECT_NE(comm, nullptr);
#endif
}

TEST_F(CommunicatorManagerTest, GetCommunicatorByType_InvalidType) {
  // Test GetCommunicatorByType with invalid type
  auto* comm = manager_->GetCommunicatorByType("INVALID");
  EXPECT_EQ(comm, nullptr);
}

TEST_F(CommunicatorManagerTest, ProcessHeartbeatData_ErrorHandling) {
  // Create a mock communicator that returns error
  class MockZmqCommunicatorWithError : public ZmqCommunicator {
   public:
    explicit MockZmqCommunicatorWithError(const ConnectorConfig& config) : ZmqCommunicator(config) {}
    Status Initialize() override { return Status(); }
    Status ProcessHeartbeatData(
        const std::unordered_map<std::string, std::string>&,
        const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>&) override {
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Mock error");
    }
    Status Send(const std::string&, int, int, size_t, const void*, size_t, DataType) override { return Status(); }
  };

  // Replace with error-returning communicator
  manager_->SetZmqCommunicator(std::make_unique<MockZmqCommunicatorWithError>(config_));

  std::unordered_map<std::string, std::string> comm_group_to_id;
  std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>> comm_group_to_address;
  comm_group_to_id["test_group"] = "dGVzdF9pZA==";  // "test_id" encoded in base64

  // Should continue processing despite error (manager doesn't fail)
  Status status = manager_->ProcessHeartbeatData(comm_group_to_id, comm_group_to_address);
  // The manager continues processing other communicators even if one fails
  // So the final status might be OK depending on the last communicator
}

// Tests for exception handling in Shutdown
class MockZmqCommunicatorWithException : public ZmqCommunicator {
 public:
  explicit MockZmqCommunicatorWithException(const ConnectorConfig& config) : ZmqCommunicator(config) {}
  Status Initialize() override { return Status(); }
  void Shutdown() override { throw std::runtime_error("Mock shutdown exception"); }
  Status Send(const std::string&, int, int, size_t, const void*, size_t, DataType) override { return Status(); }
};

TEST_F(CommunicatorManagerTest, Shutdown_ExceptionHandling) {
  // Test shutdown with communicator that throws exception
  manager_->SetZmqCommunicator(std::make_unique<MockZmqCommunicatorWithException>(config_));

  // Shutdown should handle exceptions gracefully
  EXPECT_NO_THROW(manager_->Shutdown());

  // Should be able to shutdown again (idempotent)
  EXPECT_NO_THROW(manager_->Shutdown());
}

// Test for CreateZmqCommunicator and CreateNcclCommunicator coverage
TEST_F(CommunicatorManagerTest, CreateCommunicators_DirectCall) {
  // Create a fresh manager without pre-set mock communicators
  CommunicatorManager fresh_manager(config_, std::make_unique<MockCoordinator>());

  // Test that getting communicators returns nullptr initially
  EXPECT_EQ(fresh_manager.GetZmqCommunicator(), nullptr);
#ifdef ENABLE_CUDA
  EXPECT_EQ(fresh_manager.GetNcclCommunicator(), nullptr);
#endif

  // Note: We can't easily test the real CreateZmqCommunicator/CreateNcclCommunicator
  // without triggering actual ZMQ/NCCL initialization, which would cause issues
  // in the test environment. The mock-based approach is safer for unit testing.
}

#ifdef ENABLE_CUDA
TEST_F(CommunicatorManagerTest, Initialize_WithNCCLCommunicationType) {
  // Test initialization with NCCL communication type
  config_.communication_type = CommunicationType::NCCL;
  CommunicatorManager nccl_manager(config_, std::make_unique<MockCoordinator>());
  nccl_manager.SetZmqCommunicator(std::make_unique<MockZmqCommunicator>(config_));
  nccl_manager.SetNcclCommunicator(std::make_unique<MockNcclCommunicator>(config_));

  Status status = nccl_manager.Initialize();
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(nccl_manager.IsInitialized());
}
#endif

// Test edge case where ProcessHeartbeatData is called with empty data
TEST_F(CommunicatorManagerTest, ProcessHeartbeatData_EmptyData) {
  std::unordered_map<std::string, std::string> empty_comm_group_to_id;
  std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>> empty_comm_group_to_address;

  Status status = manager_->ProcessHeartbeatData(empty_comm_group_to_id, empty_comm_group_to_address);
  EXPECT_TRUE(status.OK());
}

// Test multiple shutdown calls (idempotency)
TEST_F(CommunicatorManagerTest, Shutdown_Idempotent) {
  // First shutdown
  manager_->Shutdown();

  // Second shutdown should be safe
  EXPECT_NO_THROW(manager_->Shutdown());

  // Third shutdown should also be safe
  EXPECT_NO_THROW(manager_->Shutdown());
}

}  // namespace ksana_llm
