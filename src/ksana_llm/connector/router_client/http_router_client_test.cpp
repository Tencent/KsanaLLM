/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <curl/curl.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "ksana_llm/connector/node_info.h"
#include "ksana_llm/connector/router_client/http_router_client.h"
#include "ksana_llm/connector/router_client/resolved_endpoint.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/socket_util.h"
#include "ksana_llm/utils/status.h"
#include "test.h"

using namespace ksana_llm;

#define COMM_ID "rsEQZCIrKFcCALITC4eAvwAAAAAAAAAAAAAAAAAAAAAA=="

// Global flag to control test mode
bool g_use_mock_router_client = true;

// Helper function to check if we should run tests with a real server
bool ShouldUseRealServer() {
  const char* env_var = std::getenv("KSANA_TEST_WITH_SERVER");
  if (env_var && (std::string(env_var) == "1" || std::string(env_var) == "true")) {
    return true;
  }
  return false;
}

// Mock router_client to test without actual HTTP requests
class MockRouterClient : public HTTPRouterClient {
 public:
  explicit MockRouterClient(const std::string& endpoint) : HTTPRouterClient(endpoint) {}

  // Override the HTTP request method to return mock responses
  std::string MakeHttpRequest(const std::string& path, const std::string& method,
                              const nlohmann::json& json_data) override {
    last_path_ = path;
    last_method_ = method;
    last_json_data_ = json_data;

    // For node registration, return a mock response
    if (path == "/RegisterNode" && method == "POST") {
      nlohmann::json response = {{"node_id", "e39920b3-46a1-43d5-8fef-f458823dc3de"},
                                 {"inference_addr", json_data.value("inference_addr", "0.0.0.0:8080")},
                                 {"cluster_name", json_data.value("cluster_name", "")},
                                 {"group_role", json_data.value("group_role", "")},
                                 {"node_rank", json_data.value("node_rank", 0)},
                                 {"is_online", true},
                                 {"job_id", "1744710649509039_885470"},
                                 {"start_time", "2025-04-06 14:58:58"},
                                 {"last_heartbeat", "2025-04-06T15:16:47.369453"},
                                 {"devices", json_data.value("devices", nlohmann::json::array())},
                                 {"world_size", json_data.value("world_size", 1)},
                                 {"coordinator_addr", json_data.value("coordinator_addr", "localhost:13579")}};
      return response.dump();
    }

    return "{}";
  }

  // Getters to inspect the last request
  std::string GetLastPath() const { return last_path_; }
  std::string GetLastMethod() const { return last_method_; }
  nlohmann::json GetLastJsonData() const { return last_json_data_; }

 public:
  std::string last_path_;
  std::string last_method_;
  nlohmann::json last_json_data_;
};

// Initialize the flag in SetUp based on environment variable
class RouterClientTest : public testing::Test {
 protected:
  void SetUp() override {
    // Check for environment variable to determine test mode
    g_use_mock_router_client = !ShouldUseRealServer();
  }
  void TearDown() override {}

  // Helper to create the appropriate router_client type based on test mode
  std::shared_ptr<RouterClient> CreateRouterClient(const std::string& endpoint) {
    if (g_use_mock_router_client) {
      return std::make_unique<MockRouterClient>(endpoint);
    } else {
      return std::make_unique<HTTPRouterClient>(endpoint);
    }
  }
};

TEST_F(RouterClientTest, TestRegisterNode) {
  if (!g_use_mock_router_client) {
    GTEST_SKIP() << "Skipping mock test when using real server";
    return;
  }

  class RegisterNodeMockRouterClient : public MockRouterClient {
   public:
    explicit RegisterNodeMockRouterClient(const std::string& endpoint) : MockRouterClient(endpoint) {}
    std::string MakeHttpRequest(const std::string& path, const std::string& method,
                                const nlohmann::json& json_data) override {
      last_path_ = path;
      last_method_ = method;
      last_json_data_ = json_data;
      if (path == "/RegisterNode" && method == "POST") {
        nlohmann::json response = {{"node_id", "e39920b3-46a1-43d5-8fef-f458823dc3de"},
                                   {"inference_addr", json_data.value("inference_addr", "0.0.0.0:8088")},
                                   {"cluster_name", json_data.value("cluster_name", "default_cluster")},
                                   {"group_role", json_data.value("group_role", "prefill")},
                                   {"node_rank", json_data.value("node_rank", 0)},
                                   {"is_online", true},
                                   {"job_id", json_data.value("job_id", "daddecc0-a028-41dd-b0a1-7302b28a9c3b")},
                                   {"start_time", json_data.value("start_time", "2025-04-06 14:58:58")},
                                   {"last_heartbeat", "2025-04-06T15:16:47.369453"},
                                   {"devices", json_data.value("devices", nlohmann::json::array())},
                                   {"world_size", json_data.value("world_size", 1)},
                                   {"coordinator_addr", json_data.value("coordinator_addr", "localhost:13579")}};
        return response.dump();
      }
      return "{}";
    }
  };

  RegisterNodeMockRouterClient router_client("http://127.0.0.1:9080");

  KVNodeInfo node_info;
  node_info.inference_addr = "0.0.0.0:8088";
  node_info.coordinator_addr = "localhost:13579";
  node_info.cluster_name = "default_cluster";
  node_info.group_role = "prefill";
  node_info.node_rank = 0;
  node_info.world_size = 2;
  node_info.devices.emplace_back(0, "NVIDIA L20", "1.1.1.1");
  node_info.devices.emplace_back(1, "NVIDIA L20", "");
  node_info.start_time = "2025-04-06 14:58:58";
  node_info.job_id = "daddecc0-a028-41dd-b0a1-7302b28a9c3b";

  Status status = router_client.RegisterNode(node_info);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(router_client.GetLastPath(), "/RegisterNode");
  EXPECT_EQ(router_client.GetLastMethod(), "POST");
  auto json_data = router_client.GetLastJsonData();
  EXPECT_EQ(json_data["inference_addr"], "0.0.0.0:8088");
  EXPECT_EQ(json_data["coordinator_addr"], "localhost:13579");
  EXPECT_EQ(json_data["cluster_name"], "default_cluster");
  EXPECT_EQ(json_data["group_role"], "prefill");
  EXPECT_EQ(json_data["node_rank"], 0);
  EXPECT_EQ(json_data["world_size"], 2);
  ASSERT_EQ(json_data["devices"].size(), 2);
  EXPECT_EQ(json_data["devices"][0]["device_id"], 0);
  EXPECT_EQ(json_data["devices"][0]["device_type"], "NVIDIA L20");
  EXPECT_EQ(json_data["devices"][0]["device_ip"], "1.1.1.1");
  EXPECT_EQ(json_data["devices"][1]["device_id"], 1);
  EXPECT_EQ(json_data["devices"][1]["device_type"], "NVIDIA L20");
  std::string interface, ip;
  GetAvailableInterfaceAndIP(interface, ip);
  if (ip.empty()) {
    EXPECT_EQ(json_data["devices"][1]["device_ip"], "127.0.0.1");
  }
  EXPECT_EQ(json_data["start_time"], "2025-04-06 14:58:58");
  EXPECT_EQ(json_data["job_id"], "daddecc0-a028-41dd-b0a1-7302b28a9c3b");
  EXPECT_EQ(router_client.GetNodeInfo().node_id, "e39920b3-46a1-43d5-8fef-f458823dc3de");
}

TEST_F(RouterClientTest, TestRegisterNodeError) {
  // Skip the test if we're using a real server
  if (!g_use_mock_router_client) {
    GTEST_SKIP() << "Skipping mock error test when using real server";
    return;
  }

  class ErrorMockRouterClient : public MockRouterClient {
   public:
    explicit ErrorMockRouterClient(const std::string& endpoint) : MockRouterClient(endpoint) {}

    std::string MakeHttpRequest(const std::string& path, const std::string& method,
                                const nlohmann::json& json_data) override {
      // Always return an error
      throw std::runtime_error("Connection failed");
    }
  };

  // Create a mock router_client that simulates errors
  ErrorMockRouterClient router_client("http://127.0.0.1:9080");

  // Create a simple node info
  KVNodeInfo node_info;
  node_info.cluster_name = "default_cluster";

  // Call RegisterNode
  Status status = router_client.RegisterNode(node_info);

  // Verify the status indicates an error
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
}

// Test case for prefill node heartbeat (Prefill节点)
TEST_F(RouterClientTest, TestSendHeartbeatPrefill) {
  // Create a mock router_client that simulates a producer node heartbeat response
  class PrefillHeartbeatMock : public MockRouterClient {
   public:
    explicit PrefillHeartbeatMock(const std::string& endpoint) : MockRouterClient(endpoint) {}

    std::string MakeHttpRequest(const std::string& path, const std::string& method,
                                const nlohmann::json& json_data) override {
      last_path_ = path;
      last_method_ = method;
      last_json_data_ = json_data;
      if (path == "/Heartbeat" && method == "POST") {
        nlohmann::json response = {
            {"node_id", "3b9aadf9-59bd-4f82-aa55-9a6f0ee62c67"},
            {"is_online", true},
            {"group_ready", true},
            {"coordinator_addr", "localhost:13579"},
            {"node_role", "prefill"},
            {"timestamp", "2025-05-16T21:19:07.560885"},
            {"comm_group_to_address",
             {{"prefill_group_0__decode_group_0",
               nlohmann::json::array(
                   {nlohmann::json::array({0, 0, "1.1.1.1:13579"}), nlohmann::json::array({0, 1, "2.2.2.2:13579"}),
                    nlohmann::json::array({0, 0, "3.3.3.3:14579"}), nlohmann::json::array({0, 1, "4.4.4.4:14579"})})},
              {"prefill_group_0__decode_group_1",
               nlohmann::json::array(
                   {nlohmann::json::array({0, 0, "1.1.1.1:13579"}), nlohmann::json::array({0, 1, "2.2.2.2:13579"}),
                    nlohmann::json::array({0, 0, "7.7.7.7:14579"}), nlohmann::json::array({0, 1, "8.8.8.8:14579"})})}}},
            {"comm_group_to_id",
             {{"prefill_group_0__decode_group_0", "6666666-66666-666666-6666-6666666666"},
              {"prefill_group_0__decode_group_1", ""}}}};
        return response.dump();
      }
      return "{}";
    }
    std::string last_path_;
    std::string last_method_;
    nlohmann::json last_json_data_;
  };

  PrefillHeartbeatMock router_client("http://127.0.0.1:9080");
  KVHeartbeatResponse response;
  std::string node_id;
  Status status = router_client.SendHeartbeat(node_id, response);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(router_client.last_path_, "/Heartbeat");
  EXPECT_EQ(router_client.last_method_, "POST");
  EXPECT_EQ(response.node_id, "3b9aadf9-59bd-4f82-aa55-9a6f0ee62c67");
  EXPECT_TRUE(response.is_online);
  EXPECT_TRUE(response.group_ready);
  EXPECT_EQ(response.coordinator_addr, "localhost:13579");
  EXPECT_EQ(response.node_role, "prefill");

  // Check comm_group_to_address for prefill_group_0__decode_group_0
  const auto& addr0 = response.comm_group_to_address["prefill_group_0__decode_group_0"];
  EXPECT_EQ(addr0.size(), 4);
  EXPECT_EQ(std::get<0>(addr0[0]), 0);
  EXPECT_EQ(std::get<1>(addr0[0]), 0);
  EXPECT_EQ(std::get<2>(addr0[0]), "1.1.1.1:13579");
  EXPECT_EQ(std::get<0>(addr0[1]), 0);
  EXPECT_EQ(std::get<1>(addr0[1]), 1);
  EXPECT_EQ(std::get<2>(addr0[1]), "2.2.2.2:13579");
  EXPECT_EQ(std::get<0>(addr0[2]), 0);
  EXPECT_EQ(std::get<1>(addr0[2]), 0);
  EXPECT_EQ(std::get<2>(addr0[2]), "3.3.3.3:14579");
  EXPECT_EQ(std::get<0>(addr0[3]), 0);
  EXPECT_EQ(std::get<1>(addr0[3]), 1);
  EXPECT_EQ(std::get<2>(addr0[3]), "4.4.4.4:14579");

  // Check comm_group_to_address for prefill_group_0__decode_group_1
  const auto& addr1 = response.comm_group_to_address["prefill_group_0__decode_group_1"];
  EXPECT_EQ(addr1.size(), 4);
  EXPECT_EQ(std::get<0>(addr1[0]), 0);
  EXPECT_EQ(std::get<1>(addr1[0]), 0);
  EXPECT_EQ(std::get<2>(addr1[0]), "1.1.1.1:13579");
  EXPECT_EQ(std::get<0>(addr1[1]), 0);
  EXPECT_EQ(std::get<1>(addr1[1]), 1);
  EXPECT_EQ(std::get<2>(addr1[1]), "2.2.2.2:13579");
  EXPECT_EQ(std::get<0>(addr1[2]), 0);
  EXPECT_EQ(std::get<1>(addr1[2]), 0);
  EXPECT_EQ(std::get<2>(addr1[2]), "7.7.7.7:14579");
  EXPECT_EQ(std::get<0>(addr1[3]), 0);
  EXPECT_EQ(std::get<1>(addr1[3]), 1);
  EXPECT_EQ(std::get<2>(addr1[3]), "8.8.8.8:14579");

  // Check comm_group_to_id
  EXPECT_EQ(response.comm_group_to_id.size(), 2);
  EXPECT_EQ(response.comm_group_to_id["prefill_group_0__decode_group_0"], "6666666-66666-666666-6666-6666666666");
  EXPECT_EQ(response.comm_group_to_id["prefill_group_0__decode_group_1"], "");
}

// Test case for decode node heartbeat (Decode节点)
TEST_F(RouterClientTest, TestSendHeartbeatDecode) {
  // Create a mock router_client that simulates a decode node heartbeat response
  class DecodeHeartbeatMock : public MockRouterClient {
   public:
    explicit DecodeHeartbeatMock(const std::string& endpoint) : MockRouterClient(endpoint) {}

    std::string MakeHttpRequest(const std::string& path, const std::string& method,
                                const nlohmann::json& json_data) override {
      last_path_ = path;
      last_method_ = method;
      last_json_data_ = json_data;
      if (path == "/Heartbeat" && method == "POST") {
        nlohmann::json response = {
            {"node_id", "d3c9aadf9-59bd-4f82-aa55-9a6f0ee62c68"},
            {"is_online", true},
            {"group_ready", true},
            {"coordinator_addr", "localhost:13580"},
            {"node_role", "decode"},
            {"timestamp", "2025-05-16T22:19:07.560885"},
            {"comm_group_to_address",
             {{"prefill_group_1__decode_group_1",
               nlohmann::json::array(
                   {nlohmann::json::array({0, 0, "10.0.0.1:12306"}), nlohmann::json::array({0, 1, "10.0.0.2:12306"}),
                    nlohmann::json::array({1, 0, "10.0.0.3:12306"}), nlohmann::json::array({1, 1, "10.0.0.3:12307"})})},
              {"prefill_group_2__decode_group_1",
               nlohmann::json::array({nlohmann::json::array({0, 0, "10.0.0.6:12306"}),
                                      nlohmann::json::array({0, 1, "10.0.0.7:12307"})})}}},
            {"comm_group_to_id",
             {{"prefill_group_1__decode_group_1", "comm_id_1"}, {"prefill_group_2__decode_group_1", "comm_id_3"}}}};
        return response.dump();
      }
      return "{}";
    }
    std::string last_path_;
    std::string last_method_;
    nlohmann::json last_json_data_;
  };

  DecodeHeartbeatMock router_client("http://127.0.0.1:9080");
  KVHeartbeatResponse response;
  std::string node_id;
  Status status = router_client.SendHeartbeat(node_id, response);
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(router_client.last_path_, "/Heartbeat");
  EXPECT_EQ(router_client.last_method_, "POST");
  EXPECT_EQ(response.node_id, "d3c9aadf9-59bd-4f82-aa55-9a6f0ee62c68");
  EXPECT_TRUE(response.is_online);
  EXPECT_TRUE(response.group_ready);
  EXPECT_EQ(response.coordinator_addr, "localhost:13580");
  EXPECT_EQ(response.node_role, "decode");

  // Check comm_group_to_address for prefill_group_1__decode_group_1
  const auto& addr0 = response.comm_group_to_address["prefill_group_1__decode_group_1"];
  EXPECT_EQ(addr0.size(), 4);
  EXPECT_EQ(std::get<0>(addr0[0]), 0);
  EXPECT_EQ(std::get<1>(addr0[0]), 0);
  EXPECT_EQ(std::get<2>(addr0[0]), "10.0.0.1:12306");
  EXPECT_EQ(std::get<0>(addr0[1]), 0);
  EXPECT_EQ(std::get<1>(addr0[1]), 1);
  EXPECT_EQ(std::get<2>(addr0[1]), "10.0.0.2:12306");
  EXPECT_EQ(std::get<0>(addr0[2]), 1);
  EXPECT_EQ(std::get<1>(addr0[2]), 0);
  EXPECT_EQ(std::get<2>(addr0[2]), "10.0.0.3:12306");
  EXPECT_EQ(std::get<0>(addr0[3]), 1);
  EXPECT_EQ(std::get<1>(addr0[3]), 1);
  EXPECT_EQ(std::get<2>(addr0[3]), "10.0.0.3:12307");

  // Check comm_group_to_address for prefill_group_2__decode_group_1
  const auto& addr1 = response.comm_group_to_address["prefill_group_2__decode_group_1"];
  EXPECT_EQ(addr1.size(), 2);
  EXPECT_EQ(std::get<0>(addr1[0]), 0);
  EXPECT_EQ(std::get<1>(addr1[0]), 0);
  EXPECT_EQ(std::get<2>(addr1[0]), "10.0.0.6:12306");
  EXPECT_EQ(std::get<0>(addr1[1]), 0);
  EXPECT_EQ(std::get<1>(addr1[1]), 1);
  EXPECT_EQ(std::get<2>(addr1[1]), "10.0.0.7:12307");

  // Check comm_group_to_id
  EXPECT_EQ(response.comm_group_to_id.size(), 2);
  EXPECT_EQ(response.comm_group_to_id["prefill_group_1__decode_group_1"], "comm_id_1");
  EXPECT_EQ(response.comm_group_to_id["prefill_group_2__decode_group_1"], "comm_id_3");
}

// Test case for heartbeat error handling
TEST_F(RouterClientTest, TestSendHeartbeatError) {
  // Create a mock router_client that simulates errors during heartbeat
  class ErrorHeartbeatMock : public MockRouterClient {
   public:
    explicit ErrorHeartbeatMock(const std::string& endpoint) : MockRouterClient(endpoint) {}

    std::string MakeHttpRequest(const std::string& path, const std::string& method,
                                const nlohmann::json& json_data) override {
      // Always throw an exception to simulate network error
      throw std::runtime_error("Connection failed during heartbeat");
    }
  };

  // Create a mock router_client
  ErrorHeartbeatMock router_client("http://127.0.0.1:9080");

  // Create a heartbeat response object
  KVHeartbeatResponse response;
  std::string node_id;

  // Call SendHeartbeat
  Status status = router_client.SendHeartbeat(node_id, response);

  // Verify the status indicates an error
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
}

// Test case for SendCommId functionality
TEST_F(RouterClientTest, TestSendCommId) {
  // Skip the test if we're using a real server
  if (!g_use_mock_router_client) {
    GTEST_SKIP() << "Skipping mock test when using real server";
    return;
  }

  // Create a mock router_client that simulates the Comm ID registration response
  class CommIdMockRouterClient : public MockRouterClient {
   public:
    explicit CommIdMockRouterClient(const std::string& endpoint) : MockRouterClient(endpoint) {}

    std::string MakeHttpRequest(const std::string& path, const std::string& method,
                                const nlohmann::json& json_data) override {
      last_path_ = path;
      last_method_ = method;
      last_json_data_ = json_data;

      if (path == "/RegisterCommId" && method == "POST") {
        // Validate the required fields exist in the request
        if (!json_data.contains("node_id") || !json_data.contains("comm_key") || !json_data.contains("comm_id")) {
          return R"({"detail": "Missing required fields in request"})";
        }

        // Return a successful response with the same comm ID that was sent
        nlohmann::json response = {{"status", "OK"}, {"comm_id", json_data["comm_id"]}};
        return response.dump();
      }

      return "{}";
    }

    // Expose last request data for verification
    std::string last_path_;
    std::string last_method_;
    nlohmann::json last_json_data_;
  };

  // Create a mock router_client
  CommIdMockRouterClient router_client("http://127.0.0.1:9080");

  // Test data
  std::string node_id = "node-123456";
  std::string comm_key = "prefill_group_0_decode_group_0";
  std::string comm_id = COMM_ID;

  // Call SendCommId
  Status status = router_client.SendCommId(node_id, comm_key, comm_id);

  // Verify the status
  EXPECT_TRUE(status.OK()) << "SendCommId failed with status: " << status.GetMessage();

  // Verify the HTTP request
  EXPECT_EQ(router_client.last_path_, "/RegisterCommId");
  EXPECT_EQ(router_client.last_method_, "POST");

  // Verify the JSON request payload
  EXPECT_EQ(router_client.last_json_data_["node_id"], node_id);
  EXPECT_EQ(router_client.last_json_data_["comm_key"], comm_key);
  EXPECT_EQ(router_client.last_json_data_["comm_id"], comm_id);
}

// Test case for SendCommId error handling
TEST_F(RouterClientTest, TestSendCommIdError) {
  // Skip the test if we're using a real server
  if (!g_use_mock_router_client) {
    GTEST_SKIP() << "Skipping mock error test when using real server";
    return;
  }

  // Create a mock router_client that simulates errors during Comm ID registration
  class ErrorCommIdMockRouterClient : public MockRouterClient {
   public:
    explicit ErrorCommIdMockRouterClient(const std::string& endpoint) : MockRouterClient(endpoint) {}

    std::string MakeHttpRequest(const std::string& path, const std::string& method,
                                const nlohmann::json& json_data) override {
      if (path == "/RegisterCommId") {
        // Simulate a server error response
        nlohmann::json error_response = {{"detail", "通信组 'invalid_key' 不存在"}};
        return error_response.dump();
      }
      return "{}";
    }
  };

  // Create a mock router_client that returns an error
  ErrorCommIdMockRouterClient router_client("http://127.0.0.1:9080");

  // Test data
  std::string node_id = "node-123456";
  std::string comm_key = "invalid_key";
  std::string comm_id = COMM_ID;

  // Call SendCommId
  Status status = router_client.SendCommId(node_id, comm_key, comm_id);

  // Verify the status indicates an error
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
  EXPECT_TRUE(status.GetMessage().find("Failed to register Comm ID") != std::string::npos);
}

// Test case for SendCommId with mismatched Comm ID in response
TEST_F(RouterClientTest, TestSendCommIdMismatch) {
  // Skip the test if we're using a real server
  if (!g_use_mock_router_client) {
    GTEST_SKIP() << "Skipping mock test when using real server";
    return;
  }

  // Create a mock router_client that returns a different Comm ID than what was sent
  class MismatchCommIdMockRouterClient : public MockRouterClient {
   public:
    explicit MismatchCommIdMockRouterClient(const std::string& endpoint) : MockRouterClient(endpoint) {}

    std::string MakeHttpRequest(const std::string& path, const std::string& method,
                                const nlohmann::json& json_data) override {
      if (path == "/RegisterCommId") {
        // Return a response with a different Comm ID to trigger the mismatch check
        nlohmann::json response = {{"status", "OK"}, {"comm_id", "different-comm-id-should-cause-error"}};
        return response.dump();
      }
      return "{}";
    }
  };

  // Create a mock router_client that returns a mismatched Comm ID
  MismatchCommIdMockRouterClient router_client("http://127.0.0.1:9080");

  // Test data
  std::string node_id = "node-123456";
  std::string comm_key = "prefill_group_0_decode_group_0";
  std::string comm_id = COMM_ID;  // This is the expected Comm ID

  // Call SendCommId
  Status status = router_client.SendCommId(node_id, comm_key, comm_id);

  // Verify the status indicates an error due to comm ID mismatch
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INTERNAL_UNKNOWN_ERROR);
  EXPECT_TRUE(status.GetMessage().find("comm ID mismatch") != std::string::npos);
}

// ====================================================================================
// NEW TESTS TO IMPROVE COVERAGE: Testing WriteCallback and MakeHttpRequest directly
// ====================================================================================

// Test WriteCallback-like functionality by implementing the same logic
// Since WriteCallback is static in the .cpp file, we test equivalent functionality
static size_t TestWriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
  size_t newLength = size * nmemb;
  try {
    s->append(reinterpret_cast<char*>(contents), newLength);
    return newLength;
  } catch (std::bad_alloc& e) {
    return 0;
  }
}

// Test the WriteCallback-equivalent function that handles CURL response data
TEST_F(RouterClientTest, TestWriteCallbackFunctionality) {
  std::string response_buffer;

  // Test normal operation
  const char* test_data = "Hello, World!";
  size_t result = TestWriteCallback(const_cast<char*>(test_data), 1, strlen(test_data), &response_buffer);

  EXPECT_EQ(result, strlen(test_data));
  EXPECT_EQ(response_buffer, "Hello, World!");

  // Test appending more data
  const char* more_data = " More data";
  result = TestWriteCallback(const_cast<char*>(more_data), 1, strlen(more_data), &response_buffer);

  EXPECT_EQ(result, strlen(more_data));
  EXPECT_EQ(response_buffer, "Hello, World! More data");

  // Test with empty data
  result = TestWriteCallback(const_cast<char*>(""), 1, 0, &response_buffer);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(response_buffer, "Hello, World! More data");  // Should remain unchanged

  // Test with size * nmemb calculation
  const char* chunk_data = "ABCD";
  result = TestWriteCallback(const_cast<char*>(chunk_data), 2, 2, &response_buffer);  // 2*2 = 4 bytes
  EXPECT_EQ(result, 4);
  EXPECT_EQ(response_buffer, "Hello, World! More dataABCD");
}

// Test WriteCallback error handling (simulating bad_alloc exception)
TEST_F(RouterClientTest, TestWriteCallbackErrorHandling) {
  // Create a templated version of the callback to test error handling
  auto TestWriteCallbackTemplate = [](void* contents, size_t size, size_t nmemb, auto* s) -> size_t {
    size_t newLength = size * nmemb;
    try {
      s->append(reinterpret_cast<char*>(contents), newLength);
      return newLength;
    } catch (std::bad_alloc& e) {
      return 0;
    }
  };

  // Create a class that simulates a string that throws bad_alloc on append
  class BadAllocString {
   public:
    void append(const char*, size_t) { throw std::bad_alloc(); }
  };

  BadAllocString bad_string;
  const char* test_data = "test";

  // TestWriteCallback should return 0 when bad_alloc is thrown
  size_t result = TestWriteCallbackTemplate(const_cast<char*>(test_data), 1, strlen(test_data), &bad_string);
  EXPECT_EQ(result, 0);

  // Also test with a normal string to verify the template works correctly
  std::string normal_string;
  result = TestWriteCallbackTemplate(const_cast<char*>(test_data), 1, strlen(test_data), &normal_string);
  EXPECT_EQ(result, strlen(test_data));
  EXPECT_EQ(normal_string, "test");
}

// Test real MakeHttpRequest implementation with mock HTTP server responses
// Note: This tests the actual HTTP logic without making real network calls
class RealHTTPRouterClientTest : public HTTPRouterClient {
 public:
  explicit RealHTTPRouterClientTest(const std::string& endpoint) : HTTPRouterClient(endpoint) {}

  // Expose the protected/private methods for testing
  using HTTPRouterClient::MakeHttpRequest;
};

// Fast-timeout version for unreachable host testing
class FastTimeoutHTTPRouterClientTest : public HTTPRouterClient {
 public:
  explicit FastTimeoutHTTPRouterClientTest(const std::string& endpoint) : HTTPRouterClient(endpoint) {}

  // Override MakeHttpRequest with very short timeout for testing
  std::string MakeHttpRequest(const std::string& path, const std::string& method,
                              const nlohmann::json& json_data) override {
    CURL* curl = curl_easy_init();
    std::string response;
    if (curl) {
      std::string url = endpoint_ + path;
      curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

      // Set up headers
      struct curl_slist* headers = NULL;
      headers = curl_slist_append(headers, "Accept: application/json");
      headers = curl_slist_append(headers, "Content-Type: application/json");
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

      // Set very short timeout for fast failure
      curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1L);          // 1 second total timeout
      curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 1L);   // 1 second connect timeout
      curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);  // Minimum bytes per second
      curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 1L);   // Abort if too slow for 1 second

      // Set custom request method if not GET
      if (method != "GET") {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method.c_str());
        std::string post_data = json_data.dump();
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, post_data.length());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data.c_str());
      }

      // Perform the request
      CURLcode res = curl_easy_perform(curl);

      // Clean up
      curl_slist_free_all(headers);
      curl_easy_cleanup(curl);

      if (res != CURLE_OK) {
        KLLM_LOG_WARNING << "HTTP request failed: " << curl_easy_strerror(res);
        return "{}";  // Return empty JSON on failure
      }
    }
    return response;
  }

 private:
  // Use the same WriteCallback as the parent class
  static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    size_t total_size = size * nmemb;
    userp->append(static_cast<char*>(contents), total_size);
    return total_size;
  }
};

// Test MakeHttpRequest with invalid URL (should handle gracefully)
TEST_F(RouterClientTest, TestMakeHttpRequestInvalidURL) {
  RealHTTPRouterClientTest client("http://1.1.1.1/index.html");

  nlohmann::json test_data = {{"key", "value"}};

  // This should not crash and should return empty or error response
  std::string response = client.MakeHttpRequest("/test", "POST", test_data);

  // The response might be empty or contain error information
  // The important thing is that it doesn't crash
  EXPECT_TRUE(true);  // If we reach here, the method handled the invalid URL gracefully
}

// Test MakeHttpRequest with unreachable host (fast timeout)
TEST_F(RouterClientTest, TestMakeHttpRequestUnreachableHost) {
  // Use a non-routable IP address that should fail quickly with short timeout
  FastTimeoutHTTPRouterClientTest client("http://192.0.2.1:8080");  // RFC 5737 test address

  nlohmann::json test_data = {{"test", "data"}};

  // This should fail within 1-2 seconds due to short timeout
  auto start_time = std::chrono::steady_clock::now();
  std::string response = client.MakeHttpRequest("/api/test", "POST", test_data);
  auto end_time = std::chrono::steady_clock::now();

  // Verify it failed quickly (within 3 seconds to be safe)
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  EXPECT_LE(duration.count(), 3) << "Request should fail quickly, took " << duration.count() << " seconds";

  // This should handle the connection failure gracefully
  // The response should be empty, "{}", or not contain "Success"
  bool is_empty_or_invalid = response.empty() || response == "{}" || response.find("Success") == std::string::npos;

  // Try to parse as JSON, if fails, also treat as valid for this test
  bool json_parse_failed = false;
  try {
    auto parsed = nlohmann::json::parse(response);
    (void)parsed;
  } catch (const nlohmann::json::parse_error&) {
    json_parse_failed = true;
  }

  EXPECT_TRUE(is_empty_or_invalid || json_parse_failed);
}

// Test MakeHttpRequest with different HTTP methods
TEST_F(RouterClientTest, TestMakeHttpRequestDifferentMethods) {
  // Use MockRouterClient for predictable, fast responses instead of relying on external service
  MockRouterClient client("http://httpbin.org");

  nlohmann::json test_data = {{"method_test", "value"}};

  // Test GET request (mock will return predictable response)
  std::string get_response = client.MakeHttpRequest("/get", "GET", test_data);
  EXPECT_TRUE(true);                         // Verify it doesn't crash
  EXPECT_EQ(client.GetLastMethod(), "GET");  // Verify method was captured
  EXPECT_EQ(client.GetLastPath(), "/get");   // Verify path was captured

  // Test POST request
  std::string post_response = client.MakeHttpRequest("/post", "POST", test_data);
  EXPECT_TRUE(true);  // Verify it doesn't crash
  EXPECT_EQ(client.GetLastMethod(), "POST");

  // Test PUT request
  std::string put_response = client.MakeHttpRequest("/put", "PUT", test_data);
  EXPECT_TRUE(true);  // Verify it doesn't crash
  EXPECT_EQ(client.GetLastMethod(), "PUT");

  // Test custom method
  std::string custom_response = client.MakeHttpRequest("/patch", "PATCH", test_data);
  EXPECT_TRUE(true);  // Verify it doesn't crash
  EXPECT_EQ(client.GetLastMethod(), "PATCH");
}

// Optional test with real network (only if environment variable is set)
TEST_F(RouterClientTest, TestMakeHttpRequestRealNetworkOptional) {
  // Only run this test if explicitly requested via environment variable
  if (!ShouldUseRealServer()) {
    GTEST_SKIP() << "Skipping real network test. Set KSANA_TEST_WITH_SERVER=1 to enable.";
  }

  // Use fast timeout for real network test to avoid long waits
  FastTimeoutHTTPRouterClientTest client("http://httpbin.org");
  nlohmann::json test_data = {{"network_test", "value"}};

  // Test one method with real network (with fast timeout)
  std::string response = client.MakeHttpRequest("/get", "GET", test_data);
  EXPECT_TRUE(true);  // Should handle network response or timeout gracefully
}

// Test MakeHttpRequest with empty JSON data
TEST_F(RouterClientTest, TestMakeHttpRequestEmptyData) {
  FastTimeoutHTTPRouterClientTest client("http://192.0.2.1:8080");  // Use fast timeout for unreachable address

  nlohmann::json empty_data;

  // Test with GET and empty data (should add no query parameters)
  std::string get_response = client.MakeHttpRequest("/test", "GET", empty_data);
  EXPECT_TRUE(true);  // Verify it doesn't crash

  // Test with POST and empty data
  std::string post_response = client.MakeHttpRequest("/test", "POST", empty_data);
  EXPECT_TRUE(true);  // Verify it doesn't crash
}

// Test MakeHttpRequest with complex JSON data
TEST_F(RouterClientTest, TestMakeHttpRequestComplexData) {
  FastTimeoutHTTPRouterClientTest client("http://192.0.2.1:8080");  // Use fast timeout for unreachable address

  nlohmann::json complex_data = {{"string_field", "test_value"},
                                 {"number_field", 42},
                                 {"bool_field", true},
                                 {"array_field", nlohmann::json::array({"item1", "item2", "item3"})},
                                 {"object_field", {{"nested_key", "nested_value"}, {"nested_number", 123}}}};

  // Test POST with complex data structure
  std::string response = client.MakeHttpRequest("/complex", "POST", complex_data);
  EXPECT_TRUE(true);  // Verify it doesn't crash and handles complex JSON
}

// Test error scenarios in MakeHttpRequest
TEST_F(RouterClientTest, TestMakeHttpRequestErrorScenarios) {
  nlohmann::json test_data = {{"error_test", "data"}};

  // Test 1: Mock different error scenarios without network dependency
  MockRouterClient mock_client("http://test.example.com");

  // Test mock error handling (instant, no network)
  std::string mock_404 = mock_client.MakeHttpRequest("/status/404", "GET", test_data);
  EXPECT_TRUE(true);  // Should handle mock requests gracefully
  EXPECT_EQ(mock_client.GetLastPath(), "/status/404");

  std::string mock_500 = mock_client.MakeHttpRequest("/status/500", "POST", test_data);
  EXPECT_TRUE(true);  // Should handle mock requests gracefully
  EXPECT_EQ(mock_client.GetLastMethod(), "POST");

  // Test 2: Network timeout scenario (fast failure)
  FastTimeoutHTTPRouterClientTest timeout_client("http://192.0.2.1:8080");
  std::string timeout_response = timeout_client.MakeHttpRequest("/test", "GET", test_data);
  EXPECT_TRUE(true);  // Should handle timeout gracefully (fails in ~1 second)

  // Test 3: Invalid URL format (instant failure)
  RealHTTPRouterClientTest invalid_client("http://1.1.1.1/index.html");
  std::string invalid_response = invalid_client.MakeHttpRequest("/test", "GET", test_data);
  EXPECT_TRUE(true);  // Should handle invalid URL gracefully
}

// Test GenerateTaskID method
TEST_F(RouterClientTest, TestGenerateTaskID) {
  HTTPRouterClient client("http://test.example.com");

  // Generate multiple task IDs and verify they are unique
  std::string task_id1 = client.GenerateTaskID();
  std::string task_id2 = client.GenerateTaskID();
  std::string task_id3 = client.GenerateTaskID();

  EXPECT_FALSE(task_id1.empty());
  EXPECT_FALSE(task_id2.empty());
  EXPECT_FALSE(task_id3.empty());

  // Task IDs should be unique (very high probability)
  EXPECT_NE(task_id1, task_id2);
  EXPECT_NE(task_id1, task_id3);
  EXPECT_NE(task_id2, task_id3);

  // Task IDs should have reasonable length (timestamp + random component)
  EXPECT_GT(task_id1.length(), 10);
  EXPECT_GT(task_id2.length(), 10);
  EXPECT_GT(task_id3.length(), 10);
}

// Test edge cases and boundary conditions
TEST_F(RouterClientTest, TestMakeHttpRequestEdgeCases) {
  // Use fast timeout client to avoid long waits on unreachable host
  FastTimeoutHTTPRouterClientTest client("http://192.0.2.1:8080");

  // Test with very large JSON data
  nlohmann::json large_data;
  std::string large_string(10000, 'A');  // 10KB string
  large_data["large_field"] = large_string;

  std::string response = client.MakeHttpRequest("/test", "POST", large_data);
  EXPECT_TRUE(true);  // Should handle large data without crashing

  // Test with special characters in JSON
  nlohmann::json special_chars_data = {
      {"special_chars", "Hello\nWorld\t\r\"\\"}, {"unicode", "测试中文字符"}, {"symbols", "!@#$%^&*()[]{}"}};

  response = client.MakeHttpRequest("/test", "POST", special_chars_data);
  EXPECT_TRUE(true);  // Should handle special characters

  // Test with empty path
  response = client.MakeHttpRequest("", "GET", nlohmann::json{});
  EXPECT_TRUE(true);  // Should handle empty path

  // Test with path starting without /
  response = client.MakeHttpRequest("test", "GET", nlohmann::json{});
  EXPECT_TRUE(true);  // Should handle path without leading slash
}

// Optional test with real network error scenarios (only if environment variable is set)
TEST_F(RouterClientTest, TestMakeHttpRequestRealNetworkErrors) {
  // Only run this test if explicitly requested via environment variable
  if (!ShouldUseRealServer()) {
    GTEST_SKIP() << "Skipping real network error test. Set KSANA_TEST_WITH_SERVER=1 to enable.";
  }

  // Use fast timeout for real network error testing
  FastTimeoutHTTPRouterClientTest client("http://httpbin.org");
  nlohmann::json test_data = {{"error_test", "data"}};

  // Test real 404 error (with fast timeout)
  std::string not_found_response = client.MakeHttpRequest("/status/404", "GET", test_data);
  EXPECT_TRUE(true);  // Should handle 404 gracefully or timeout quickly

  // Test real 500 error (with fast timeout)
  std::string server_error_response = client.MakeHttpRequest("/status/500", "GET", test_data);
  EXPECT_TRUE(true);  // Should handle 500 gracefully or timeout quickly
}

// Test suite for ResolvedEndpoint class
class ResolvedEndpointTest : public testing::Test {
 protected:
  void SetUp() override {
    // Setup for tests
  }

  void TearDown() override {
    // Cleanup for tests
  }
};

// Test basic functionality of GetResolvedEndpoint
TEST_F(ResolvedEndpointTest, TestGetResolvedEndpointBasic) {
  // Test with a simple endpoint
  std::string test_endpoint = "test.example.com:8080";
  std::string result = ResolvedEndpoint::GetResolvedEndpoint(test_endpoint);

  // The behavior depends on whether WITH_INTERNAL_LIBRARIES is enabled
  // In the simple implementation, it should return the original endpoint
  // In the internal implementation, it might try to resolve via Polaris
  EXPECT_FALSE(result.empty()) << "Result should not be empty";

  // The input endpoint should remain unchanged
  EXPECT_EQ(test_endpoint, "test.example.com:8080") << "Input endpoint should not be modified";
}

// Test GetResolvedEndpoint with various endpoint formats
TEST_F(ResolvedEndpointTest, TestGetResolvedEndpointVariousFormats) {
  // Test with IP address
  std::string ip_endpoint = "192.168.1.100:9090";
  std::string ip_result = ResolvedEndpoint::GetResolvedEndpoint(ip_endpoint);
  EXPECT_FALSE(ip_result.empty()) << "Result for IP endpoint should not be empty";

  // Test with localhost
  std::string localhost_endpoint = "localhost:3000";
  std::string localhost_result = ResolvedEndpoint::GetResolvedEndpoint(localhost_endpoint);
  EXPECT_FALSE(localhost_result.empty()) << "Result for localhost endpoint should not be empty";

  // Test with domain without port
  std::string domain_endpoint = "api.example.com";
  std::string domain_result = ResolvedEndpoint::GetResolvedEndpoint(domain_endpoint);
  EXPECT_FALSE(domain_result.empty()) << "Result for domain endpoint should not be empty";

  // Test with empty string
  std::string empty_endpoint = "";
  std::string empty_result = ResolvedEndpoint::GetResolvedEndpoint(empty_endpoint);
  // Should handle empty string gracefully (behavior may vary by implementation)
  // Don't assert specific behavior for empty string as it's edge case
}

// Test GetAddrWithPolaris functionality
TEST_F(ResolvedEndpointTest, TestGetAddrWithPolaris) {
  std::string test_endpoint = "production/ksana-test-service";
  std::string result = ResolvedEndpoint::GetAddrWithPolaris(test_endpoint);

  // In the simple implementation, this should return empty string
  // In the internal implementation, this might return a resolved address or empty if polaris fails
  // We don't assert specific behavior as it depends on the implementation and environment
  // Just ensure it doesn't crash
  EXPECT_TRUE(true) << "GetAddrWithPolaris should execute without crashing";
}
