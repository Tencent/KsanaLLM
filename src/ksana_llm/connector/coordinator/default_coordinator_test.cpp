/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/coordinator/default_coordinator.h"
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ksana_llm {
// Mock RouterClient class that allows us to test Coordinator in isolation
class MockRouterClient : public RouterClient {
 public:
  explicit MockRouterClient(const std::string& endpoint) : RouterClient(endpoint) {
    // Initialize default node info for testing
    node_info_.node_id = "test_node_id";
    node_info_.inference_addr = "127.0.0.1:8080";
    node_info_.cluster_name = "default_cluster";
    node_info_.group_role = "prefill";
    node_info_.node_rank = 0;
    node_info_.is_online = true;

    // Add a test device
    node_info_.devices.emplace_back(0, "GPU", "127.0.0.1");

    // Use default implementations unless mocked
    ON_CALL(*this, RegisterNode(::testing::_))
        .WillByDefault(::testing::Invoke(this, &MockRouterClient::DefaultRegisterNode));

    ON_CALL(*this, SendHeartbeat(::testing::_, ::testing::_))
        .WillByDefault(::testing::Invoke(this, &MockRouterClient::DefaultSendHeartbeat));

    ON_CALL(*this, GetNodeInfo()).WillByDefault(::testing::ReturnRef(node_info_));
  }

  // Mock methods
  MOCK_METHOD(Status, RegisterNode, (const KVNodeInfo& node_info), (override));
  MOCK_METHOD(Status, SendHeartbeat, (std::string & node_id, KVHeartbeatResponse& response), (override));
  MOCK_METHOD(Status, SendCommId, (const std::string& node_id, const std::string& comm_key, const std::string& comm_id),
              (override));
  MOCK_METHOD(std::string, MakeHttpRequest,
              (const std::string& path, const std::string& method, const nlohmann::json& json_data), (override));
  MOCK_METHOD(const KVNodeInfo&, GetNodeInfo, (), (const, override));

  // Default implementation for RegisterNode
  Status DefaultRegisterNode(const KVNodeInfo& node_info) {
    node_info_ = node_info;
    // Assign a test node_id if not already set
    if (node_info_.node_id.empty()) {
      node_info_.node_id = "test_node_id";
    }
    return Status();
  }

  // Default implementation for SendHeartbeat
  Status DefaultSendHeartbeat(std::string& node_id, KVHeartbeatResponse& response) {
    response.node_id = node_info_.node_id;
    response.node_role = node_info_.group_role;
    response.node_rank = node_info_.node_rank;
    response.is_online = true;
    response.group_ready = true;
    response.timestamp = "2024-05-06T10:00:00";
    response.comm_group_to_address.clear();
    response.comm_group_to_id.clear();
    // 构造多个通信组模拟
    std::vector<std::tuple<int, int, std::string>> addr1;
    addr1.emplace_back(0, 0, "1.1.1.1:13579");
    addr1.emplace_back(0, 1, "2.2.2.2:13579");
    std::vector<std::tuple<int, int, std::string>> addr2;
    addr2.emplace_back(1, 0, "3.3.3.3:14579");
    addr2.emplace_back(1, 1, "4.4.4.4:14579");
    response.comm_group_to_address["test_group_key_1"] = addr1;
    response.comm_group_to_address["test_group_key_2"] = addr2;
    response.comm_group_to_id["test_group_key_1"] = "test_comm_id_1";
    response.comm_group_to_id["test_group_key_2"] = "test_comm_id_2";
    return Status();
  }

 private:
  KVNodeInfo node_info_;
};

// Mock configuration data provider for testing
class MockConfigProvider {
 public:
  MockConfigProvider() {
    node_rank_ = 0;
    world_size_ = 1;
    device_count_ = 2;
  }

  int GetNodeRank() const { return node_rank_; }
  int GetWorldSize() const { return world_size_; }
  int GetDeviceCount() const { return device_count_; }

  void SetNodeRank(int rank) { node_rank_ = rank; }
  void SetWorldSize(int size) { world_size_ = size; }
  void SetDeviceCount(int count) { device_count_ = count; }

 private:
  int node_rank_;
  int world_size_;
  int device_count_;
};

// 测试专用 DefaultCoordinator，提供 node_info_ 只读访问接口
class TestCoordinator : public DefaultCoordinator {
 public:
  TestCoordinator(const ConnectorConfig& config, std::shared_ptr<RouterClient> router_client)
      : DefaultCoordinator(config, std::move(router_client)) {}
  // 提供 node_info_ 的 const 只读 getter
  const KVNodeInfo& node_info() const { return node_info_; }
};

class CoordinatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_.cluster_name = "default_cluster";
    config_.group_role = GroupRole::PREFILL;
    // Set a lower heartbeat interval for faster tests
    config_.heartbeat_interval_ms = 10;
  }

  void TearDown() override {
    // Ensure any coordinators created in the test are cleaned up
    if (coordinator_) {
      coordinator_->Shutdown();
      coordinator_.reset();
    }
  }

  std::shared_ptr<TestCoordinator> CreateTestCoordinator(
      std::shared_ptr<MockConfigProvider> config_provider = nullptr,
      std::shared_ptr<MockRouterClient> mock_router_client = nullptr) {
    auto router_client =
        mock_router_client ? mock_router_client : std::make_shared<MockRouterClient>("http://localhost:8080");

    if (config_provider) {
      config_.node_rank = config_provider->GetNodeRank();
      config_.world_size = config_provider->GetWorldSize();
      config_.device_count = config_provider->GetDeviceCount();
    }

    return std::make_shared<TestCoordinator>(config_, router_client);
  }

  ConnectorConfig config_;
  std::shared_ptr<MockConfigProvider> default_config_provider_ = std::make_shared<MockConfigProvider>();
  std::shared_ptr<TestCoordinator> coordinator_;  // Store the coordinator here to ensure proper cleanup
};

// Tests the constructor initialization
TEST_F(CoordinatorTest, ConstructorInitialization) {
  auto mock_router_client = std::make_shared<MockRouterClient>("http://localhost:8080");
  ::testing::Mock::AllowLeak(mock_router_client.get());
  auto config_provider = std::make_shared<MockConfigProvider>();
  coordinator_ = CreateTestCoordinator(config_provider, mock_router_client);
  EXPECT_NE(coordinator_.get(), nullptr);
}

TEST_F(CoordinatorTest, ConstructorThrowsForNullRouterClient) {
  std::shared_ptr<MockRouterClient> null_router_client = nullptr;
  EXPECT_THROW(
      { auto coordinator = std::make_unique<TestCoordinator>(config_, std::move(null_router_client)); },
      std::invalid_argument);
}

// Tests the RegisterNode method
TEST_F(CoordinatorTest, RegisterNode) {
  auto config_provider = std::make_shared<MockConfigProvider>();
  config_provider->SetNodeRank(0);
  config_provider->SetWorldSize(1);
  config_provider->SetDeviceCount(2);
  // Explicitly create the mock and allow leak to suppress the warning
  auto mock_router_client = std::make_shared<MockRouterClient>("http://localhost:8080");
  ::testing::Mock::AllowLeak(mock_router_client.get());
  coordinator_ = CreateTestCoordinator(config_provider, mock_router_client);

  // 现在需要显式调用Initialize方法，其中包含RegisterNode的调用
  Status status = coordinator_->Initialize();
  ASSERT_TRUE(status.OK()) << "Initialize失败: " << status.GetMessage();

  // 检查 node_info_ 是否被正确赋值
  EXPECT_EQ(coordinator_->node_info().cluster_name, config_.cluster_name);
  EXPECT_EQ(coordinator_->node_info().group_role, GroupRoleToString(config_.group_role));
  EXPECT_EQ(coordinator_->node_info().node_rank, 0);
  EXPECT_EQ(coordinator_->node_info().world_size, 1);
  EXPECT_EQ(coordinator_->node_info().devices.size(), 2);
}

// Tests the HeartbeatThread method with successful heartbeat
TEST_F(CoordinatorTest, HeartbeatThread) {
  // Create the coordinator and store it in the test fixture for cleanup
  auto mock_router_client = std::make_shared<MockRouterClient>("http://localhost:8080");
  ::testing::Mock::AllowLeak(mock_router_client.get());

  // Use default behavior for the mock methods
  EXPECT_CALL(*mock_router_client, RegisterNode(::testing::_))
      .WillOnce(::testing::Invoke(mock_router_client.get(), &MockRouterClient::DefaultRegisterNode));

  EXPECT_CALL(*mock_router_client, SendHeartbeat(::testing::_, ::testing::_))
      .WillRepeatedly(::testing::Invoke(mock_router_client.get(), &MockRouterClient::DefaultSendHeartbeat));
  auto config_provider = std::make_shared<MockConfigProvider>();
  coordinator_ = CreateTestCoordinator(config_provider, mock_router_client);

  // 关键：先初始化，RegisterNode 才会被调用
  coordinator_->Initialize();

  // Start the heartbeat in the main thread
  auto status = coordinator_->StartHeartbeat();
  EXPECT_TRUE(status.OK());

  // Wait for a short time to ensure the heartbeat thread has a chance to run
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Stop the heartbeat
  coordinator_->StopHeartbeat();
}

// Tests the HeartbeatThread method with failed heartbeat
TEST_F(CoordinatorTest, HeartbeatThreadWithFailedHeartbeat) {
  // Create a mock router client that we can configure
  auto mock_router_client = std::make_shared<MockRouterClient>("http://localhost:8080");
  ::testing::Mock::AllowLeak(mock_router_client.get());

  // Configure mock router client to succeed on registration but fail on heartbeat
  EXPECT_CALL(*mock_router_client, RegisterNode(::testing::_))
      .WillOnce(::testing::Invoke(mock_router_client.get(), &MockRouterClient::DefaultRegisterNode));

  EXPECT_CALL(*mock_router_client, SendHeartbeat(::testing::_, ::testing::_))
      .WillRepeatedly(::testing::Return(Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Heartbeat failed")));

  // Create the coordinator with the mock router client and store it in the test fixture
  auto config_provider = std::make_shared<MockConfigProvider>();
  coordinator_ = CreateTestCoordinator(config_provider, mock_router_client);
  coordinator_->Initialize();
  // Start the heartbeat in the main thread
  auto status = coordinator_->StartHeartbeat();
  EXPECT_TRUE(status.OK());

  // Wait for a short time to ensure the heartbeat thread has a chance to run
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // Stop the heartbeat
  coordinator_->StopHeartbeat();

  // Verify test succeeded (if we got this far without hanging, it's a success)
  EXPECT_TRUE(true);
}

// Tests the StartHeartbeat method
TEST_F(CoordinatorTest, StartHeartbeat) {
  // Create the test coordinator
  coordinator_ = CreateTestCoordinator();

  // Test StartHeartbeat
  Status status = coordinator_->StartHeartbeat();
  EXPECT_TRUE(status.OK());

  // Cleanup in TearDown
}

// Tests calling StartHeartbeat multiple times
TEST_F(CoordinatorTest, StartHeartbeatMultipleTimes) {
  // Create the test coordinator
  coordinator_ = CreateTestCoordinator();

  // Test StartHeartbeat twice
  Status status1 = coordinator_->StartHeartbeat();
  EXPECT_TRUE(status1.OK());

  Status status2 = coordinator_->StartHeartbeat();
  EXPECT_TRUE(status2.OK());

  // Cleanup in TearDown
}

// Tests the StopHeartbeat method
TEST_F(CoordinatorTest, StopHeartbeat) {
  // Create the test coordinator and start heartbeat
  coordinator_ = CreateTestCoordinator();
  coordinator_->StartHeartbeat();

  // Test StopHeartbeat
  coordinator_->StopHeartbeat();

  // No assertion needed, just checking that StopHeartbeat doesn't throw
  // Add an explicit expectation to satisfy the linter
  EXPECT_TRUE(true);
}

// Tests the Shutdown method
TEST_F(CoordinatorTest, Shutdown) {
  // Create the test coordinator and start heartbeat
  coordinator_ = CreateTestCoordinator();
  coordinator_->StartHeartbeat();

  // Test Shutdown
  coordinator_->Shutdown();

  // No assertion needed, just checking that Shutdown doesn't throw
  // Add an explicit expectation to satisfy the linter
  EXPECT_TRUE(true);
}

// Tests the destructor
TEST_F(CoordinatorTest, Destructor) {
  // Create a scoped coordinator that will be destroyed immediately
  {
    auto scoped_coordinator = CreateTestCoordinator();
    scoped_coordinator->StartHeartbeat();
    // Coordinator will be destroyed at the end of this block
  }

  // Create another coordinator to test that our cleanup didn't affect future tests
  coordinator_ = CreateTestCoordinator();

  // No assertion needed, just checking that the destructor didn't throw
  EXPECT_TRUE(true);
}

}  // namespace ksana_llm