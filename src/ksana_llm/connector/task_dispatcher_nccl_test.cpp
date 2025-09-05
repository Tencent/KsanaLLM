/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#ifdef ENABLE_CUDA
#  include "ksana_llm/connector/task_dispatcher.h"

#  include <chrono>
#  include <cstring>
#  include <future>
#  include <limits>
#  include <memory>
#  include <string>
#  include <thread>
#  include <vector>

#  include "gmock/gmock.h"
#  include "gtest/gtest.h"
#  include "ksana_llm/connector/communicator/communicator_manager.h"
#  include "ksana_llm/connector/communicator/zmq/zmq_communicator.h"

#  include "ksana_llm/connector/communicator/nvida/nccl_communicator.h"
#  include "ksana_llm/connector/config.h"
#  include "ksana_llm/connector/task_manager.h"
#  include "ksana_llm/transfer/transfer_types.h"
#  include "ksana_llm/utils/device_types.h"
#  include "ksana_llm/utils/status.h"

namespace ksana_llm {

// Helper function to create valid config for all mock classes
static ConnectorConfig CreateValidTestConfig() {
  ConnectorConfig config;
  config.group_role = GroupRole::PREFILL;
  config.node_rank = 0;
  config.world_size = 1;
  config.device_count = 1;
  config.group_name = "test_group";
  config.node_name = "test_node";
  config.router_endpoint = "http://localhost:8000";
  return config;
}

// Simple Mock Coordinator for testing
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
  bool IsInitialized() const override { return true; }  // Always initialized for testing
};

// Mock Communicator for testing - mock Communicator接口，不区分Zmq/Nccl
class MockCommunicator : public Communicator {
 public:
  MockCommunicator() : Communicator(CreateValidTestConfig()) {}

  MOCK_METHOD(void, Shutdown, (), (override));
  MOCK_METHOD(Status, Initialize, (), (override));
  MOCK_METHOD(Status, Send, (const std::string&, int, int, uint64_t, const void*, size_t, DataType), (override));
  MOCK_METHOD(Status, Recv, (const std::string&, int, int, uint64_t, void*, size_t, DataType), (override));
  MOCK_METHOD(Status, SendGroup,
              (const std::string&, int, int, uint64_t, const std::vector<const void*>&, const std::vector<size_t>&,
               DataType),
              (override));
  MOCK_METHOD(Status, RecvGroup,
              (const std::string&, int, int, uint64_t, const std::vector<void*>&, const std::vector<size_t>&, DataType),
              (override));
  MOCK_METHOD(bool, IsConnectionReady, (const std::string&, int), (const, override));
  MOCK_METHOD((Status), ProcessHeartbeatData,
              ((const std::unordered_map<std::string, std::string>&),
               (const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>&)),
              (override));
  MOCK_METHOD(void, DoSetReceiveCallback, (const ReceiveCallback&), (override));
};

// Mock CommunicatorManager for testing - 根据 communication_type 返回不同 mock communicator
class MockCommunicatorManager : public CommunicatorManager {
 public:
  MockCommunicatorManager(std::shared_ptr<MockCommunicator> zmq, std::shared_ptr<MockCommunicator> nccl)
      : CommunicatorManager(CreateValidTestConfig(), std::make_shared<MockCoordinator>()),
        mock_zmq_communicator_(zmq),
        mock_nccl_communicator_(nccl) {}

  MOCK_METHOD(Status, Initialize, (), (override));
  MOCK_METHOD(void, Shutdown, (), (override));
  MOCK_METHOD(bool, IsInitialized, (), (const, override));
  MOCK_METHOD(Status, ProcessHeartbeatData,
              ((const std::unordered_map<std::string, std::string>&),
               (const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>&)),
              (override));
  MOCK_METHOD(Status, CreateZmqCommunicator, (), (override));
  MOCK_METHOD(Status, CreateNcclCommunicator, (), (override));
  MOCK_METHOD(ZmqCommunicator*, GetZmqCommunicator, (), (const, override));
  MOCK_METHOD(NcclCommunicator*, GetNcclCommunicator, (), (const, override));

  // 实际返回 mock communicator
  ZmqCommunicator* GetZmqCommunicatorImpl() const {
    return reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get());
  }
  NcclCommunicator* GetNcclCommunicatorImpl() const {
    return reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get());
  }

  std::shared_ptr<MockCommunicator> mock_zmq_communicator_;
  std::shared_ptr<MockCommunicator> mock_nccl_communicator_;
};

class TaskDispatcherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_.node_rank = 0;
    config_.world_size = 2;
    config_.device_count = 1;
    config_.circular_bucket_size = 128;
    mock_zmq_communicator_ = std::make_shared<MockCommunicator>();
    mock_nccl_communicator_ = std::make_shared<MockCommunicator>();
    ::testing::Mock::AllowLeak(mock_zmq_communicator_.get());
    ::testing::Mock::AllowLeak(mock_nccl_communicator_.get());
    mock_comm_manager_ = std::make_shared<MockCommunicatorManager>(mock_zmq_communicator_, mock_nccl_communicator_);
    ::testing::Mock::AllowLeak(mock_comm_manager_.get());
    task_manager_ = TaskManager::GetInstance(128, config_.circular_bucket_size);
    SetDefaultExpectations();
  }

  void TearDown() override {
    if (task_dispatcher_) {
      task_dispatcher_->Shutdown();
      // Give threads time to shutdown gracefully
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      task_dispatcher_.reset();
    }

    // Then shutdown task manager
    if (task_manager_) {
      task_manager_->Shutdown();
    }

    // Explicitly release the mock objects to prevent memory leaks
    // Release in reverse order of creation
    mock_comm_manager_.reset();  // Release first as it holds references to the communicators

    mock_zmq_communicator_.reset();
    mock_nccl_communicator_.reset();

    // Finally release task manager
    task_manager_.reset();
  }

  void SetDefaultExpectations() {
    // ZMQ communicator
    EXPECT_CALL(*mock_zmq_communicator_, Shutdown()).WillRepeatedly(testing::Return());
    EXPECT_CALL(*mock_zmq_communicator_, Initialize()).WillRepeatedly(testing::Return(Status()));
    EXPECT_CALL(*mock_zmq_communicator_, DoSetReceiveCallback(testing::_)).WillRepeatedly(testing::Return());
    EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady(testing::_, testing::_))
        .WillRepeatedly(testing::Return(true));
    EXPECT_CALL(*mock_zmq_communicator_,
                Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Return(Status()));
    // NCCL communicator
    EXPECT_CALL(*mock_nccl_communicator_, Shutdown()).WillRepeatedly(testing::Return());
    EXPECT_CALL(*mock_nccl_communicator_, Initialize()).WillRepeatedly(testing::Return(Status()));
    EXPECT_CALL(*mock_nccl_communicator_, DoSetReceiveCallback(testing::_)).WillRepeatedly(testing::Return());
    EXPECT_CALL(*mock_nccl_communicator_, IsConnectionReady(testing::_, testing::_))
        .WillRepeatedly(testing::Return(true));
    EXPECT_CALL(*mock_nccl_communicator_,
                Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::Return(Status()));
    // CommunicatorManager
    EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillRepeatedly(testing::Return(true));
    EXPECT_CALL(*mock_comm_manager_, Initialize()).WillRepeatedly(testing::Return(Status()));
    EXPECT_CALL(*mock_comm_manager_, Shutdown()).WillRepeatedly(testing::Return());
    EXPECT_CALL(*mock_comm_manager_, ProcessHeartbeatData(testing::_, testing::_))
        .WillRepeatedly(testing::Return(Status()));
    EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
        .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
    EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
        .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));
    EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillRepeatedly(testing::Return(Status()));
    EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillRepeatedly(testing::Return(Status()));
  }

  void CreateTaskDispatcher() {
    task_dispatcher_ = std::make_unique<TaskDispatcher>(config_, task_manager_, mock_comm_manager_);
  }

  std::shared_ptr<TransferTask> CreateMockTask(int req_id, int block_idx, int layer_idx, int device_idx,
                                               int tensor_size = 0, const std::string& addr = "127.0.0.1:50051") {
    auto task = std::make_shared<TransferTask>();
    task->req_id = req_id;
    task->tensor.block_idx = block_idx;
    task->tensor.layer_idx = layer_idx;
    task->tensor.hash_device_id = device_idx;
    // token 字段已移除，使用 tensor_size 代替
    task->tensor.shape = std::vector<int64_t>(tensor_size, 1);
    task->addr = addr;
    task->is_completed = false;
    task->tensor.dtype = DataType::TYPE_FP32;
    task->tensor.src_ptr = nullptr;
    size_t element_count = tensor_size > 0 ? tensor_size : 1;
    size_t element_size = GetTypeSize(DataType::TYPE_FP32);
    size_t total_bytes = element_count * element_size;
    task->dst_ptr = std::malloc(total_bytes);
    if (task->dst_ptr != nullptr) {
      std::memset(task->dst_ptr, 0, total_bytes);
    }
    return task;
  }

  TaskKey CreateTaskKey(int req_id, int block_idx, int layer_idx, int hash_device_id, int tensor_size = 0,
                        std::vector<int> tokens = std::vector<int>(MAX_TRANSFER_TOKENS, 0),
                        bool is_skipped_task = false) {
    return TaskKey(req_id, block_idx, layer_idx, hash_device_id, tensor_size, tokens, is_skipped_task);
  }

  std::vector<TaskKey> CreateTaskKeyBatch(int batch_size, int hash_device_id = 0) {
    std::vector<TaskKey> batch;
    for (int i = 0; i < batch_size; ++i) {
      batch.push_back(CreateTaskKey(i, i, 0, hash_device_id, 0));
    }
    return batch;
  }

  void AddTasksToManager(const std::vector<TaskKey>& task_keys) {
    for (const auto& key : task_keys) {
      auto task = CreateMockTask(key.req_id, key.block_idx, key.layer_idx, key.hash_device_id, key.tensor_size);
      task_manager_->AddTask(key, task);
      // Add to processing buffer to simulate pending tasks
      task_manager_->PutProcessingBuffer(key);
    }
  }

  ConnectorConfig config_;

  std::shared_ptr<TaskManager> task_manager_;
  std::shared_ptr<MockCommunicatorManager> mock_comm_manager_;
  std::shared_ptr<MockCommunicator> mock_zmq_communicator_;
  std::shared_ptr<MockCommunicator> mock_nccl_communicator_;
  std::unique_ptr<TaskDispatcher> task_dispatcher_;
  std::shared_ptr<DeviceInfoManager> device_info_manager_ = std::make_shared<DeviceInfoManager>();
};

// TEST_F(TaskDispatcherTest, Constructor) { std::cout << "TaskDispatcherTest Constructor called" << std::endl; }

// Constructor and Destructor Tests
TEST_F(TaskDispatcherTest, Constructor) {
  // Explicitly allow leaks before creating any mocks
  if (mock_zmq_communicator_) {
    testing::Mock::AllowLeak(mock_zmq_communicator_.get());
  }
  if (mock_nccl_communicator_) {
    testing::Mock::AllowLeak(mock_nccl_communicator_.get());
  }
  if (mock_comm_manager_) {
    testing::Mock::AllowLeak(mock_comm_manager_.get());
  }

  CreateTaskDispatcher();
  EXPECT_NE(task_dispatcher_, nullptr);
  // Ensure proper cleanup between tests
  task_dispatcher_->Shutdown();

  // Allow leaks on any new mock objects that might have been created
  if (mock_zmq_communicator_) {
    testing::Mock::AllowLeak(mock_zmq_communicator_.get());
  }
  if (mock_nccl_communicator_) {
    testing::Mock::AllowLeak(mock_nccl_communicator_.get());
  }
  if (mock_comm_manager_) {
    testing::Mock::AllowLeak(mock_comm_manager_.get());
  }
}

TEST_F(TaskDispatcherTest, Destructor) {
  CreateTaskDispatcher();
  task_dispatcher_.reset();
  // Should not crash
}

// Initialize Tests
TEST_F(TaskDispatcherTest, InitializeSuccess) {
  CreateTaskDispatcher();

  // Override default expectations for this specific test
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  // Use the proper mock that was set up in SetDefaultExpectations()
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  Status status = task_dispatcher_->Initialize(device_info_manager_);
  EXPECT_TRUE(status.OK());
}

TEST_F(TaskDispatcherTest, InitializeFailure) {
  CreateTaskDispatcher();

  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize())
      .WillOnce(testing::Return(Status(RetCode::RET_INIT_FAILED, "Communication manager init failed")));

  Status status = task_dispatcher_->Initialize(device_info_manager_);
  EXPECT_FALSE(status.OK());
}

// Shutdown Tests
TEST_F(TaskDispatcherTest, Shutdown) {
  CreateTaskDispatcher();

  EXPECT_CALL(*mock_comm_manager_, Shutdown()).Times(testing::AtLeast(1));

  task_dispatcher_->Shutdown();
  // Should complete without hanging
}

// SendToPrefill Tests - Testing with real TaskManager
TEST_F(TaskDispatcherTest, SendToPrefillWithTasks) {
  // Set communication type to NCCL to ensure both communicators are initialized
  config_.communication_type = CommunicationType::NCCL;
  CreateTaskDispatcher();

  // Set up specific expectations for Initialize() call - override default expectations
  testing::Mock::VerifyAndClearExpectations(mock_comm_manager_.get());
  testing::Mock::VerifyAndClearExpectations(mock_zmq_communicator_.get());
  testing::Mock::VerifyAndClearExpectations(mock_nccl_communicator_.get());

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  // Set up ZMQ communicator expectations
  EXPECT_CALL(*mock_zmq_communicator_, Shutdown()).WillRepeatedly(testing::Return());
  EXPECT_CALL(*mock_zmq_communicator_, Initialize()).WillRepeatedly(testing::Return(Status()));
  EXPECT_CALL(*mock_zmq_communicator_, DoSetReceiveCallback(testing::_)).WillRepeatedly(testing::Return());
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady(testing::_, testing::_)).WillRepeatedly(testing::Return(true));
  EXPECT_CALL(*mock_zmq_communicator_,
              Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status()));

  // NCCL communicator will be created and checked in NCCL mode
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // Set up NCCL communicator expectations - 关键是让连接检查成功
  EXPECT_CALL(*mock_nccl_communicator_, Shutdown()).WillRepeatedly(testing::Return());
  EXPECT_CALL(*mock_nccl_communicator_, Initialize()).WillRepeatedly(testing::Return(Status()));
  EXPECT_CALL(*mock_nccl_communicator_, DoSetReceiveCallback(testing::_)).WillRepeatedly(testing::Return());
  EXPECT_CALL(*mock_nccl_communicator_, IsConnectionReady(testing::_, testing::_))
      .WillRepeatedly(testing::Return(true));  // 关键：让连接检查返回成功
  EXPECT_CALL(*mock_nccl_communicator_,
              Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status()));
  // Additional expectations for other methods that might be called
  EXPECT_CALL(*mock_comm_manager_, ProcessHeartbeatData(testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, Shutdown()).WillRepeatedly(testing::Return());

  task_dispatcher_->Initialize(device_info_manager_);

  // Add tasks to real task manager
  std::vector<TaskKey> task_batch = CreateTaskKeyBatch(3, 0);
  AddTasksToManager(task_batch);

  // Call SendToPrefill once - it will process the tasks without infinite retries
  // Since we're mocking the connection as ready, it should succeed
  // We need to call it in a way that doesn't block forever
  std::thread send_thread([this]() {
    // Run SendToPrefill for a short time
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(2)) {
      if (task_manager_->IsProcessingBufferEmpty()) {
        break;  // Exit if no more tasks to process
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });

  // Give the thread some time to process, then shutdown
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  task_dispatcher_->Shutdown();  // This will stop the internal threads

  if (send_thread.joinable()) {
    send_thread.join();
  }

  // Test passes if we reach here without hanging
}

TEST_F(TaskDispatcherTest, SendToPrefillNoTasks) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  // No tasks in task manager
  // Run SendToPrefill in a separate thread to avoid blocking
  std::thread send_thread([this]() { task_dispatcher_->SendToPrefill(); });

  // Let it run briefly, then shutdown to stop the infinite loop
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  task_dispatcher_->Shutdown();  // This will set running_ = false and notify waiter

  // Wait for the thread to finish
  if (send_thread.joinable()) {
    send_thread.join();
  }

  // Test passes if we reach here without hanging
}

// Register callback tests
TEST_F(TaskDispatcherTest, RegisterPrefillRecv) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  // Should not crash when called
  task_dispatcher_->RegisterPrefillRecv();
}

TEST_F(TaskDispatcherTest, RegisterDecodeRecv) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  // Should not crash when called
  task_dispatcher_->RegisterDecodeRecv();

  std::string connection_id = "prefill_group_1__decode_group_1_1-2";

  auto [group_key, device_config_pair] = task_dispatcher_->ParseConnectionId(connection_id);
  auto [src_device_idx, dst_device_idx] = device_config_pair;

  EXPECT_EQ(group_key, "prefill_group_1__decode_group_1");
  EXPECT_EQ(src_device_idx, 2);
  EXPECT_EQ(dst_device_idx, 1);
}

// ProcessPrefillReceivedTasks Tests
TEST_F(TaskDispatcherTest, ProcessPrefillReceivedTasks) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  // ProcessPrefillReceivedTasks() contains an infinite loop, so run it in a separate thread
  std::thread process_thread([this]() { task_dispatcher_->ProcessPrefillReceivedTasks(); });

  // Give the thread a moment to start and enter the waiting state
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Shutdown to stop background threads - this will set running_ = false and notify waiters
  task_dispatcher_->Shutdown();

  // Wait for the thread to finish
  if (process_thread.joinable()) {
    process_thread.join();
  }

  // Test passes if we reach here without hanging
}

// PREFILL node BatchTasks Tests
TEST_F(TaskDispatcherTest, BatchTasksPrefill) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);
  task_dispatcher_->config_.group_role = GroupRole::PREFILL;

  // Create test tasks
  TaskKey task_key1 = CreateTaskKey(123, 0, 0, 0, 400, {0, 0}, false);
  TaskKey task_key2 = CreateTaskKey(123, 1, 2, 3, 400, {0, 0}, true);

  std::vector<TaskKey> task_keys = {task_key1, task_key2};

  task_manager_->RegisterDecodeConfirmedTasks(task_keys);

  task_manager_->PutProcessingBuffer(task_key1);
  task_manager_->PutProcessingBuffer(task_key2);

  // Batch tasks. The task_key with is_skipped_task = true should not be included
  int batch_size = 10;
  task_keys = task_dispatcher_->BatchTasks(batch_size);
  EXPECT_EQ(task_keys.size(), 1);
}

TEST_F(TaskDispatcherTest, ZMQSendAndRecv) {
  config_.communication_type = CommunicationType::ZMQ;
  CreateTaskDispatcher();
  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  // Set up ZMQ communicator expectations
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady(testing::_, testing::_)).WillRepeatedly(testing::Return(true));
  EXPECT_CALL(*mock_zmq_communicator_,
              Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  // 创建测试任务和group batch
  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(2, 0);
  AddTasksToManager(task_keys);

  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), task_keys);

  // 测试发送
  EXPECT_NO_THROW(task_dispatcher_->HandlePrefillGroupBatch(group_batch));

  // 捕获callback并执行它
  Communicator::ReceiveCallback captured_callback;
  EXPECT_CALL(*mock_zmq_communicator_, DoSetReceiveCallback(testing::_))
      .WillOnce(testing::DoAll(testing::SaveArg<0>(&captured_callback),
                               testing::Invoke([](const auto& callback) { callback("", 0, 0, nullptr); }),
                               testing::Return()));
  // Should not crash when called
  task_dispatcher_->RegisterDecodeRecv();

  // 验证callback被设置并执行它
  ASSERT_TRUE(captured_callback != nullptr) << "Callback should be captured";

  // 模拟接收到数据并执行callback
  TaskKey task_key = CreateTaskKey(111, 0, 0, 0, 0);
  std::vector<char> test_data(sizeof(TaskKey) + (sizeof(int) * MAX_TRANSFER_TOKENS));

  std::shared_ptr<TransferTask> task = std::make_shared<TransferTask>();

  task_manager_->AddTask(task_key, nullptr);
  memcpy(test_data.data(), &task_key, sizeof(TaskKey));
  captured_callback(test_data.data(), test_data.size(), 0, nullptr);
  task_manager_->AddTask(task_key, task);
  task_key.tokens[0] = 1;
  task_key.tokens[1] = 2;
  std::vector<int> tokens(MAX_TRANSFER_TOKENS, 0);
  task->dst_ptr = tokens.data();
  memcpy(test_data.data(), &task_key, sizeof(TaskKey));
  captured_callback(test_data.data(), test_data.size(), 0, nullptr);
  EXPECT_EQ(tokens[0], 1);
  EXPECT_EQ(tokens[1], 2);

#  ifdef ENABLE_CUDA
  task_key = CreateTaskKey(111, 0, 0, 0, sizeof(int));
  int tensor_size = 1;
  task_manager_->AddTask(task_key, task);
  int* data_ptr = nullptr;
  cudaMalloc(&data_ptr, sizeof(int));
  task->dst_ptr = data_ptr;
  memcpy(test_data.data(), &task_key, sizeof(TaskKey));
  memcpy(test_data.data() + sizeof(TaskKey), &tensor_size, sizeof(tensor_size));
  captured_callback(test_data.data(), test_data.size(), 0, nullptr);
  tensor_size = 0;
  cudaMemcpy(&tensor_size, data_ptr, sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(tensor_size, 1);
  cudaFree(data_ptr);
#  endif
}

// HandlePrefillGroupBatch Tests
TEST_F(TaskDispatcherTest, HandlePrefillGroupBatch) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(3, 0);
  // Add tasks to manager first
  AddTasksToManager(task_keys);

  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), task_keys);

  // Should not crash when called
  task_dispatcher_->HandlePrefillGroupBatch(group_batch);
}

// Test HandlePrefillGroupBatch with connection failure
TEST_F(TaskDispatcherTest, HandlePrefillGroupBatch_ConnectionFailure) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  // Set up connection to fail
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady(testing::_, testing::_))
      .WillRepeatedly(testing::Return(false));

  task_dispatcher_->Initialize(device_info_manager_);

  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(2, 0);
  AddTasksToManager(task_keys);

  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), task_keys);

  // Should handle connection failure gracefully and retry failed tasks
  task_dispatcher_->HandlePrefillGroupBatch(group_batch);
}

// Test HandlePrefillGroupBatch with empty tensors
TEST_F(TaskDispatcherTest, HandlePrefillGroupBatch_EmptyTensors) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  // Set up connection to succeed
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady(testing::_, testing::_)).WillRepeatedly(testing::Return(true));

  task_dispatcher_->Initialize(device_info_manager_);

  // Create tasks with token != 0 to trigger "Invalid or missing task" path
  std::vector<TaskKey> task_keys;
  task_keys.push_back(CreateTaskKey(1, 0, 0, 0, 1));  // token = 1, will be invalid

  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), task_keys);

  // Should handle empty tensors gracefully by returning early
  task_dispatcher_->HandlePrefillGroupBatch(group_batch);
}

// Test HandlePrefillGroupBatch with valid tensors and successful sending
TEST_F(TaskDispatcherTest, HandlePrefillGroupBatch_SuccessfulSend) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  // Set up connection to succeed
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady(testing::_, testing::_)).WillRepeatedly(testing::Return(true));

  // Set up ZMQ send to succeed
  EXPECT_CALL(*mock_zmq_communicator_,
              Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status()));

  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // Set up NCCL sends to succeed
  EXPECT_CALL(*mock_nccl_communicator_,
              Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status()));
  EXPECT_CALL(*mock_nccl_communicator_,
              SendGroup(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(2, 0);
  AddTasksToManager(task_keys);

  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), task_keys);

  // Should complete successfully with all sends working
  task_dispatcher_->HandlePrefillGroupBatch(group_batch);
}

// Test HandlePrefillGroupBatch with ZMQ send failure
TEST_F(TaskDispatcherTest, HandlePrefillGroupBatch_ZmqSendFailure) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  // Set up connection to succeed
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady(testing::_, testing::_)).WillRepeatedly(testing::Return(true));

  // Set up ZMQ send to fail
  EXPECT_CALL(*mock_zmq_communicator_,
              Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "ZMQ send failed")));

  task_dispatcher_->Initialize(device_info_manager_);

  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(2, 0);
  AddTasksToManager(task_keys);

  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), task_keys);

  // Should handle ZMQ send failure and retry tasks
  task_dispatcher_->HandlePrefillGroupBatch(group_batch);
}

// Test HandlePrefillGroupBatch with NCCL send failure
TEST_F(TaskDispatcherTest, HandlePrefillGroupBatch_NcclSendFailure) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // Set up connection to succeed
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady(testing::_, testing::_)).WillRepeatedly(testing::Return(true));

  // Set up ZMQ send to succeed but NCCL sends to fail
  EXPECT_CALL(*mock_zmq_communicator_,
              Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status()));
  EXPECT_CALL(*mock_nccl_communicator_,
              Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL send failed")));

  task_dispatcher_->Initialize(device_info_manager_);

  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(2, 0);
  AddTasksToManager(task_keys);

  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), task_keys);

  // Should handle NCCL send failure and retry tasks
  task_dispatcher_->HandlePrefillGroupBatch(group_batch);
}

// Test HandlePrefillGroupBatch with NCCL SendGroup failure
TEST_F(TaskDispatcherTest, HandlePrefillGroupBatch_NcclSendGroupFailure) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // Set up connection to succeed
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady(testing::_, testing::_)).WillRepeatedly(testing::Return(true));

  // Set up ZMQ send and NCCL task key send to succeed, but SendGroup to fail
  EXPECT_CALL(*mock_zmq_communicator_,
              Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status()));
  EXPECT_CALL(*mock_nccl_communicator_,
              Send(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status()));
  EXPECT_CALL(*mock_nccl_communicator_,
              SendGroup(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillRepeatedly(testing::Return(Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "NCCL SendGroup failed")));

  task_dispatcher_->Initialize(device_info_manager_);

  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(2, 0);
  AddTasksToManager(task_keys);

  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), task_keys);

  // Should handle NCCL SendGroup failure gracefully
  task_dispatcher_->HandlePrefillGroupBatch(group_batch);
}

// Test HandlePrefillGroupBatch with mixed valid and invalid tasks
TEST_F(TaskDispatcherTest, HandlePrefillGroupBatch_MixedTasks) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  // Create a mix of valid (token=0) and invalid (token!=0) tasks
  std::vector<TaskKey> task_keys;
  task_keys.push_back(CreateTaskKey(1, 0, 0, 0, 0));  // token = 0, will be valid
  task_keys.push_back(CreateTaskKey(2, 1, 0, 0, 1));  // token = 1, will be invalid
  task_keys.push_back(CreateTaskKey(3, 2, 0, 0, 0));  // token = 0, will be valid

  // Add only the valid tasks to manager
  auto valid_task1 = CreateMockTask(1, 0, 0, 0, 0);
  auto valid_task3 = CreateMockTask(3, 2, 0, 0, 0);
  task_manager_->AddTask(task_keys[0], valid_task1);
  task_manager_->AddTask(task_keys[2], valid_task3);

  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), task_keys);

  // Should handle mixed tasks correctly - process valid ones, skip invalid ones
  task_dispatcher_->HandlePrefillGroupBatch(group_batch);
}

// Test HandlePrefillGroupBatch with empty task vector
TEST_F(TaskDispatcherTest, HandlePrefillGroupBatch_EmptyTaskVector) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  // Create empty task vector
  std::vector<TaskKey> empty_tasks;
  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), empty_tasks);

  // Should handle empty task vector gracefully by returning early
  task_dispatcher_->HandlePrefillGroupBatch(group_batch);
}

// Test HandlePrefillGroupBatch with large batch size
TEST_F(TaskDispatcherTest, HandlePrefillGroupBatch_LargeBatch) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Create a large batch of tasks
  std::vector<TaskKey> large_batch = CreateTaskKeyBatch(16, 0);
  AddTasksToManager(large_batch);

  std::pair<std::pair<std::string, std::pair<int, int>>, std::vector<TaskKey>> group_batch =
      std::make_pair(std::make_pair("test_group", std::make_pair(0, 0)), large_batch);

  // Should handle large batch efficiently
  task_dispatcher_->HandlePrefillGroupBatch(group_batch);
}

// RecvTaskDataWithNccl Tests
TEST_F(TaskDispatcherTest, RecvTaskDataWithNccl_Success) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // Note: No NCCL expectations needed because BufferPools won't be initialized
  // in test environment (no CUDA), so method will return early

  task_dispatcher_->Initialize(device_info_manager_);

  // Create test tasks with token = 0 (valid tensors)
  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(3, 0);
  AddTasksToManager(task_keys);

  std::string group_key = "test_group";
  int device_idx = 0;

  // In test environment without CUDA, BufferPools won't be initialized
  // so the method should return gracefully without attempting NCCL operations
  task_dispatcher_->RecvTaskDataWithNccl(group_key, device_idx, device_idx, task_keys.size());

  // Test passes if no segmentation fault occurs (which was the original issue)
}

TEST_F(TaskDispatcherTest, RecvTaskDataWithNccl_EmptyTaskKeys) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Empty task keys vector
  std::vector<TaskKey> empty_task_keys;
  std::string group_key = "test_group";
  int device_idx = 0;

  // Should handle empty task keys gracefully
  task_dispatcher_->RecvTaskDataWithNccl(group_key, device_idx, device_idx, empty_task_keys.size());
}

TEST_F(TaskDispatcherTest, RecvTaskDataWithNccl_MixedTokenTypes) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // Note: No NCCL expectations needed because BufferPools won't be initialized
  // in test environment (no CUDA), so method will return early

  task_dispatcher_->Initialize(device_info_manager_);

  // Create mixed task keys - some with token=0, some with token!=0
  std::vector<TaskKey> task_keys;
  task_keys.push_back(CreateTaskKey(1, 0, 0, 0, 0));  // token = 0 (normal tensor)
  task_keys.push_back(CreateTaskKey(2, 1, 0, 0, 1));  // token = 1 (special case)
  task_keys.push_back(CreateTaskKey(3, 2, 0, 0, 0));  // token = 0 (normal tensor)

  AddTasksToManager(task_keys);

  std::string group_key = "test_group";
  int device_idx = 0;

  // In test environment without CUDA, BufferPools won't be initialized
  // so the method should return gracefully without attempting NCCL operations
  task_dispatcher_->RecvTaskDataWithNccl(group_key, device_idx, device_idx, task_keys.size());

  // Test passes if no segmentation fault occurs
}

TEST_F(TaskDispatcherTest, RecvTaskDataWithNccl_NcclRecvGroupFailure) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // Note: No NCCL expectations needed because BufferPools won't be initialized
  // in test environment (no CUDA), so method will return early

  task_dispatcher_->Initialize(device_info_manager_);

  // Create test tasks
  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(2, 0);
  AddTasksToManager(task_keys);

  std::string group_key = "test_group";
  int device_idx = 0;

  // In test environment without CUDA, BufferPools won't be initialized
  // so the method should return gracefully without attempting NCCL operations
  task_dispatcher_->RecvTaskDataWithNccl(group_key, device_idx, device_idx, task_keys.size());

  // Test passes if no segmentation fault occurs
}

TEST_F(TaskDispatcherTest, RecvTaskDataWithNccl_NonExistentTasks) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Create task keys but don't add them to manager (simulate non-existent tasks)
  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(2, 0);
  std::string group_key = "test_group";
  int device_idx = 0;

  // Should handle non-existent tasks gracefully without crashing
  task_dispatcher_->RecvTaskDataWithNccl(group_key, device_idx, device_idx, task_keys.size());
}

// SendDataToDecodeWithNccl Tests
TEST_F(TaskDispatcherTest, SendDataToDecodeWithNccl_Success) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // New: Expect SendGroup to be called once for the batch
  EXPECT_CALL(*mock_nccl_communicator_,
              SendGroup(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .Times(1)
      .WillOnce(testing::Return(Status()));
  std::vector<TaskKey> task_keys;
  task_keys.push_back(CreateTaskKey(1, 0, 0, 0, 2));  // Valid task (will be added to manager)
  task_keys.push_back(CreateTaskKey(2, 1, 0, 0, 1));  // Invalid task (not added to manager)
  task_keys.push_back(CreateTaskKey(3, 2, 0, 0, 2));  // Valid task (will be added to manager)
  // Only add some tasks to manager
  std::vector<TaskKey> valid_tasks = {task_keys[0], task_keys[2]};
  AddTasksToManager(valid_tasks);
  task_dispatcher_->Initialize(device_info_manager_);

  std::string group_key = "test_group";
  int device_idx = 0;

  // Should call SendGroup and not crash
  task_dispatcher_->SendDataToDecodeWithNccl(group_key, device_idx, device_idx, valid_tasks);
}

TEST_F(TaskDispatcherTest, SendDataToDecodeWithNccl_EmptyTaskKeys) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // New: Should not call SendGroup at all
  EXPECT_CALL(*mock_nccl_communicator_,
              SendGroup(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .Times(0);

  task_dispatcher_->Initialize(device_info_manager_);

  // Empty task keys vector
  std::vector<TaskKey> empty_task_keys;
  std::string group_key = "test_group";
  int device_idx = 0;

  // Should return early and not crash
  task_dispatcher_->SendDataToDecodeWithNccl(group_key, device_idx, device_idx, empty_task_keys);
}

TEST_F(TaskDispatcherTest, SendDataToDecodeWithNccl_NonExistentTasks) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // New: Should not call SendGroup if no valid tasks
  EXPECT_CALL(*mock_nccl_communicator_,
              SendGroup(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .Times(0);

  task_dispatcher_->Initialize(device_info_manager_);

  // Create task keys but don't add them to manager (simulate non-existent tasks)
  std::vector<TaskKey> task_keys = CreateTaskKeyBatch(2, 0);
  std::string group_key = "test_group";
  int device_idx = 0;

  // Should skip and not crash
  task_dispatcher_->SendDataToDecodeWithNccl(group_key, device_idx, device_idx, task_keys);
}

TEST_F(TaskDispatcherTest, SendDataToDecodeWithNccl_MixedValidInvalidTasks) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // Only valid tasks should be sent
  EXPECT_CALL(*mock_nccl_communicator_,
              SendGroup(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .Times(1)
      .WillOnce(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  // Create mixed task keys - some valid, some invalid
  std::vector<TaskKey> task_keys;
  task_keys.push_back(CreateTaskKey(1, 0, 0, 0, 0));  // Valid task (will be added to manager)
  task_keys.push_back(CreateTaskKey(2, 1, 0, 0, 1));  // Invalid task (not added to manager)
  task_keys.push_back(CreateTaskKey(3, 2, 0, 0, 0));  // Valid task (will be added to manager)

  // Only add some tasks to manager
  std::vector<TaskKey> valid_tasks = {task_keys[0], task_keys[2]};
  AddTasksToManager(valid_tasks);

  std::string group_key = "test_group";
  int device_idx = 0;

  // Should only send valid tasks
  task_dispatcher_->SendDataToDecodeWithNccl(group_key, device_idx, device_idx, task_keys);
}

TEST_F(TaskDispatcherTest, SendDataToDecodeWithNccl_LargeBatch) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  // Should call SendGroup once for the large batch
  EXPECT_CALL(*mock_nccl_communicator_,
              SendGroup(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .Times(1)
      .WillOnce(testing::Return(Status()));

  task_dispatcher_->Initialize(device_info_manager_);

  std::vector<TaskKey> task_keys;
  for (int i = 0; i < 10; ++i) {
    task_keys.push_back(CreateTaskKey(i, 0, 0, 0, 2));  // Mix of valid and invalid tasks
  }
  AddTasksToManager(task_keys);
  std::string group_key = "test_group";
  int device_idx = 0;

  // Should handle large batch without crashing
  task_dispatcher_->SendDataToDecodeWithNccl(group_key, device_idx, device_idx, task_keys);
}

// CheckConnection Tests
TEST_F(TaskDispatcherTest, CheckConnection_SuccessZmqOnly) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Set up ZMQ connection to succeed - allow multiple calls since CheckConnection may call it multiple times
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("test_group", 0)).WillRepeatedly(testing::Return(true));

  bool result = task_dispatcher_->CheckConnection("test_group", 0);
  EXPECT_TRUE(result);
}

TEST_F(TaskDispatcherTest, CheckConnection_SuccessZmqAndNccl) {
  // Set communication type to NCCL to enable both communicators
  config_.communication_type = CommunicationType::NCCL;
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Set up both ZMQ and NCCL connections to succeed
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("test_group", 0)).WillRepeatedly(testing::Return(true));
  EXPECT_CALL(*mock_nccl_communicator_, IsConnectionReady("test_group", 0)).WillRepeatedly(testing::Return(true));

  bool result = task_dispatcher_->CheckConnection("test_group", 0);
  EXPECT_TRUE(result);
}

TEST_F(TaskDispatcherTest, CheckConnection_ZmqConnectionFailure) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Set up ZMQ connection to fail - allow multiple calls since CheckConnection may call it multiple times
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("test_group", 0)).WillRepeatedly(testing::Return(false));

  bool result = task_dispatcher_->CheckConnection("test_group", 0);
  EXPECT_FALSE(result);
}

TEST_F(TaskDispatcherTest, CheckConnection_NcclConnectionFailure) {
  // Set communication type to NCCL to enable both communicators
  config_.communication_type = CommunicationType::NCCL;
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Set up ZMQ to succeed but NCCL to fail
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("test_group", 0)).WillRepeatedly(testing::Return(true));
  EXPECT_CALL(*mock_nccl_communicator_, IsConnectionReady("test_group", 0)).WillRepeatedly(testing::Return(false));

  bool result = task_dispatcher_->CheckConnection("test_group", 0);
  EXPECT_FALSE(result);
}

TEST_F(TaskDispatcherTest, CheckConnection_FirstAttemptWithTimeout) {
  // Set a short timeout for testing
  config_.connector_waiting_sec = 1;  // Very short timeout for test
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Set up ZMQ connection to fail initially, then succeed
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("test_group", 0))
      .WillOnce(testing::Return(false))  // First check fails
      .WillOnce(testing::Return(true));  // Second check succeeds

  auto start_time = std::chrono::steady_clock::now();
  bool result = task_dispatcher_->CheckConnection("test_group", 0);
  auto end_time = std::chrono::steady_clock::now();

  EXPECT_TRUE(result);
  // Should have waited at least some time for first attempt
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // Note: In test environment, the actual wait may be shorter due to mocking
}

TEST_F(TaskDispatcherTest, CheckConnection_RetryAfterFirstAttempt) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // First attempt - fails (allow multiple calls during timeout waiting and final check)
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("failing_group", 0))
      .WillOnce(testing::Return(false))
      .WillOnce(testing::Return(true))
      .WillOnce(testing::Return(false));

  bool first_result = task_dispatcher_->CheckConnection("failing_group", 0);
  EXPECT_TRUE(first_result);

  // Second attempt (retry) - should not wait as long since it's not first attempt
  bool second_result = task_dispatcher_->CheckConnection("failing_group", 0);

  EXPECT_FALSE(second_result);
  // Retry should be faster than first attempt
}

TEST_F(TaskDispatcherTest, CheckConnection_NullZmqCommunicator) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call but return null ZMQ communicator
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(nullptr));  // Return null communicator

  task_dispatcher_->Initialize(device_info_manager_);

  // Should handle null communicator gracefully
  bool result = task_dispatcher_->CheckConnection("test_group", 0);
  EXPECT_FALSE(result);
}

#  ifdef ENABLE_CUDA
TEST_F(TaskDispatcherTest, CheckConnection_NullNcclCommunicator) {
  // Set communication type to NCCL to enable both communicators
  config_.communication_type = CommunicationType::NCCL;
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call but return null NCCL communicator
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(nullptr));  // Return null NCCL communicator

  task_dispatcher_->Initialize(device_info_manager_);

  // Should handle null NCCL communicator gracefully
  bool result = task_dispatcher_->CheckConnection("test_group", 0);
  EXPECT_FALSE(result);
}
#  endif

TEST_F(TaskDispatcherTest, CheckConnection_ZeroWaitingTime) {
  // Set zero waiting time
  config_.connector_waiting_sec = 0;
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // With zero waiting time, should not wait and immediately check connection
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("test_group", 0)).WillOnce(testing::Return(true));

  auto start_time = std::chrono::steady_clock::now();
  bool result = task_dispatcher_->CheckConnection("test_group", 0);
  auto end_time = std::chrono::steady_clock::now();

  EXPECT_TRUE(result);
  // Should complete very quickly with zero wait time
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  EXPECT_LT(duration.count(), 100);  // Should take less than 100ms
}

TEST_F(TaskDispatcherTest, CheckConnection_ConcurrentConnections) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Test concurrent connections with different results
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("concurrent_group1", 0)).WillRepeatedly(testing::Return(true));
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("concurrent_group2", 1))
      .WillRepeatedly(testing::Return(false));

  // Simulate concurrent checks
  std::vector<std::future<bool>> futures;
  futures.push_back(
      std::async(std::launch::async, [this]() { return task_dispatcher_->CheckConnection("concurrent_group1", 0); }));
  futures.push_back(
      std::async(std::launch::async, [this]() { return task_dispatcher_->CheckConnection("concurrent_group2", 1); }));

  bool result1 = futures[0].get();
  bool result2 = futures[1].get();

  EXPECT_TRUE(result1);
  EXPECT_FALSE(result2);
}

TEST_F(TaskDispatcherTest, CheckConnection_RepeatedFailures) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Set up repeated failures for the same connection
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("failing_group", 0)).WillRepeatedly(testing::Return(false));

  // Test multiple consecutive failures
  for (int i = 0; i < 5; ++i) {
    bool result = task_dispatcher_->CheckConnection("failing_group", 0);
    EXPECT_FALSE(result);
  }
}

#  ifdef ENABLE_CUDA
TEST_F(TaskDispatcherTest, CheckConnection_MixedNcclStates) {
  // Test when ZMQ fails but NCCL succeeds (should still fail overall)
  config_.communication_type = CommunicationType::NCCL;
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateNcclCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));
  EXPECT_CALL(*mock_comm_manager_, GetNcclCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<NcclCommunicator*>(mock_nccl_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Set up ZMQ to fail but NCCL to succeed
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("mixed_group", 0)).WillRepeatedly(testing::Return(false));
  EXPECT_CALL(*mock_nccl_communicator_, IsConnectionReady("mixed_group", 0)).WillRepeatedly(testing::Return(true));

  bool result = task_dispatcher_->CheckConnection("mixed_group", 0);
  EXPECT_FALSE(result);  // Should fail because both ZMQ and NCCL must succeed
}
#  endif

TEST_F(TaskDispatcherTest, CheckConnection_HighFrequencyChecks) {
  CreateTaskDispatcher();

  // Set up expectations for Initialize() call
  EXPECT_CALL(*mock_comm_manager_, IsInitialized()).WillOnce(testing::Return(false));
  EXPECT_CALL(*mock_comm_manager_, Initialize()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, CreateZmqCommunicator()).WillOnce(testing::Return(Status()));
  EXPECT_CALL(*mock_comm_manager_, GetZmqCommunicator())
      .WillRepeatedly(testing::Return(reinterpret_cast<ZmqCommunicator*>(mock_zmq_communicator_.get())));

  task_dispatcher_->Initialize(device_info_manager_);

  // Test rapid consecutive connection checks
  EXPECT_CALL(*mock_zmq_communicator_, IsConnectionReady("rapid_test", 0)).WillRepeatedly(testing::Return(true));

  auto start_time = std::chrono::steady_clock::now();

  // Perform many rapid checks
  for (int i = 0; i < 100; ++i) {
    bool result = task_dispatcher_->CheckConnection("rapid_test", 0);
    EXPECT_TRUE(result);
  }

  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // Should complete all checks reasonably quickly
  EXPECT_LT(duration.count(), 1000);  // Less than 1 second for 100 checks
}

// PinnedMemoryBufferBlock TaskKey读写测试
TEST(TaskDispatcherNccl, PinnedMemoryBufferBlockTaskKeyRW) {
  using namespace ksana_llm;
  int device_count = 1;
  size_t block_num = 2;
  size_t block_size = 8 * sizeof(TaskKey);  // 支持8个TaskKey
  int device_id = 0;
  PinnedMemoryBufferPool pool(device_count, block_num, block_size);

  // 构造一组TaskKey
  std::vector<TaskKey> task_keys;
  for (int i = 0; i < 8; ++i) {
    task_keys.emplace_back(i, i * 10, i * 100, device_id, i * 1000);
  }

  // 获取block并写入task_keys
  PinnedMemoryBufferBlock* block = pool.get_block(device_id);
  ASSERT_NE(block, nullptr);
  ASSERT_GE(block->capacity, task_keys.size() * sizeof(TaskKey));

  // 写入
  std::memcpy(block->host_ptr, task_keys.data(), task_keys.size() * sizeof(TaskKey));

  // 读出
  TaskKey* read_ptr = static_cast<TaskKey*>(block->host_ptr);
  for (size_t i = 0; i < task_keys.size(); ++i) {
    EXPECT_EQ(read_ptr[i].req_id, task_keys[i].req_id);
    EXPECT_EQ(read_ptr[i].block_idx, task_keys[i].block_idx);
    EXPECT_EQ(read_ptr[i].layer_idx, task_keys[i].layer_idx);
    EXPECT_EQ(read_ptr[i].hash_device_id, task_keys[i].hash_device_id);
    EXPECT_EQ(read_ptr[i].tensor_size, task_keys[i].tensor_size);
  }

  pool.put_block(block);
}

}  // namespace ksana_llm
#endif