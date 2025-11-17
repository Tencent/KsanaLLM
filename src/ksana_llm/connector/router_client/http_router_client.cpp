/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/router_client/http_router_client.h"
#include <arpa/inet.h>
#include <curl/curl.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include "ksana_llm/utils/socket_util.h"
#include "ksana_llm/connector/router_client/resolved_endpoint.h"


namespace ksana_llm {

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
  size_t newLength = size * nmemb;
  try {
    s->append(reinterpret_cast<char*>(contents), newLength);
    return newLength;
  } catch (std::bad_alloc& e) {
    return 0;
  }
}

std::string HTTPRouterClient::GenerateTaskID() {
  // 获取当前时间戳（微秒级）
  auto now = std::chrono::system_clock::now();
  auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

  // 生成随机数
  static std::mt19937 rng(std::random_device {} ());
  std::uniform_int_distribution<int> dist(0, 999999);

  // 拼接成字符串
  std::ostringstream oss;
  oss << now_us << "_" << std::setw(6) << std::setfill('0') << dist(rng);

  return oss.str();
}

std::string HTTPRouterClient::MakeHttpRequest(const std::string& path, const std::string& method,
                                              const nlohmann::json& json_data) {
  CURL* curl = curl_easy_init();
  std::string response;
  if (curl) {
    std::string url = ResolvedEndpoint::GetResolvedEndpoint(endpoint_) + path;
    KLLM_LOG_INFO << "Making " << method << " request to: " << url;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // Set up headers
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Accept: application/json");
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Set timeout to prevent hanging forever
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 1L);

    // Set custom request method if not GET
    if (method != "GET") {
      // For POST, PUT, etc.
      curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, method.c_str());

      // Convert JSON to string and send as POST data
      std::string post_data = json_data.dump();
      KLLM_LOG_INFO << "Request payload: " << post_data;

      // Set the POST data
      curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, post_data.length());
      curl_easy_setopt(curl, CURLOPT_COPYPOSTFIELDS, post_data.c_str());
    } else {
      // For GET requests, convert parameters to URL query
      if (!json_data.empty()) {
        std::string query = "?";
        bool first = true;
        for (auto it = json_data.begin(); it != json_data.end(); ++it) {
          if (!first) {
            query += "&";
          }
          query += it.key() + "=" + it.value().get<std::string>();
          first = false;
        }
        std::string full_url = url + query;
        KLLM_LOG_INFO << "Full URL with query: " << full_url;
        curl_easy_setopt(curl, CURLOPT_URL, full_url.c_str());
      }
    }

    // Verbose mode for debugging (enabled during troubleshooting)
    // curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    // Perform the request
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      KLLM_LOG_ERROR << "curl_easy_perform() failed: " << curl_easy_strerror(res);
    } else {
      int64_t http_code = 0;
      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
      KLLM_LOG_INFO << "HTTP response code: " << http_code;
      KLLM_LOG_INFO << "Response body: " << response;

      // Check for common error status codes
      if (http_code >= 400) {
        KLLM_LOG_ERROR << "HTTP error " << http_code << " received from server";
      }
    }

    // Clean up
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
  }

  return response;
}

Status HTTPRouterClient::CheckConnectivity() {
  try {
    // Make a simple GET request to check if the service is responding
    nlohmann::json empty_data;
    std::string response = MakeHttpRequest("/health", "GET", empty_data);

    // If we got any response (even empty), consider it connected
    // The actual HTTP status code is checked inside MakeHttpRequest
    if (!response.empty()) {
      return Status();  // Success
    } else {
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to check connectivity: empty response");
    }
  } catch (const std::exception& e) {
    std::string error_msg = "Failed to check connectivity: " + std::string(e.what());
    KLLM_LOG_ERROR << error_msg;
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, error_msg);
  }
}

Status HTTPRouterClient::RegisterNode(const KVNodeInfo& node_info) {
  // Add debug output
  KLLM_LOG_INFO << "RegisterNode for node with inference addr: " << node_info.inference_addr;
  // Save a local copy of the node info
  node_info_ = node_info;

  nlohmann::json data;

  // Populate data according to the updated protocol
  data["inference_addr"] = node_info.inference_addr;
  data["coordinator_addr"] = node_info.coordinator_addr;
  data["cluster_name"] = node_info.cluster_name;
  data["group_role"] = node_info.group_role;
  data["node_rank"] = node_info.node_rank;
  data["world_size"] = node_info.world_size;

  // Create devices array with detailed information
  nlohmann::json devices_array = nlohmann::json::array();
  for (const auto& device : node_info.devices) {
    nlohmann::json device_json;
    device_json["device_id"] = device.device_id;
    device_json["device_type"] = device.device_type;
    device_json["device_ip"] = device.device_ip;
    devices_array.push_back(device_json);
  }
  data["devices"] = devices_array;

  // Use the provided start_time and job_id if available, otherwise generate them
  if (!node_info.start_time.empty()) {
    data["start_time"] = node_info.start_time;
  } else {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    data["start_time"] = oss.str();
  }

  data["job_id"] = !node_info.job_id.empty() ? node_info.job_id : GenerateTaskID();

  // Try to send the request and parse the response
  try {
    // Send the registration request
    std::string response_str = MakeHttpRequest("/RegisterNode", "POST", data);

    // Check if the response is empty or has unexpected format (like HTML)
    if (response_str.empty()) {
      KLLM_LOG_ERROR << "Empty response received from server";
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Empty response received from server");
    }

    // Check if response starts with anything other than { (likely HTML or error page)
    if (response_str[0] != '{') {
      KLLM_LOG_ERROR << "Invalid response format - not valid JSON. Response starts with: "
                     << response_str.substr(0, std::min(size_t(50), response_str.size()));
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR,
                    "Invalid response format. Expected JSON, received: " +
                        response_str.substr(0, std::min(size_t(100), response_str.size())));
    }

    // Try to parse the JSON response
    nlohmann::json response_json = nlohmann::json::parse(response_str);

    // Check if the response contains a node ID
    if (response_json.contains("node_id")) {
      // Set the node ID in the stored node info
      node_info_.node_id = response_json["node_id"].get<std::string>();
      KLLM_LOG_INFO << "Registered node with ID: " << node_info_.node_id;
    }

    // Check online status if available
    if (response_json.contains("is_online")) {
      bool is_online = response_json["is_online"].get<bool>();
      node_info_.is_online = is_online;
      KLLM_LOG_INFO << "Node online status: " << (is_online ? "online" : "offline");
    }

    // Check last_heartbeat if available
    if (response_json.contains("last_heartbeat")) {
      node_info_.last_heartbeat = response_json["last_heartbeat"].get<std::string>();
      KLLM_LOG_INFO << "Last heartbeat from registration: " << node_info_.last_heartbeat;
    }

    return Status();
  } catch (const nlohmann::json::parse_error& e) {
    KLLM_LOG_ERROR << "Failed to parse JSON response: " << e.what();
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to parse JSON response: " + std::string(e.what()));
  } catch (const std::exception& e) {
    KLLM_LOG_ERROR << "Exception during node registration: " << e.what();
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Exception during node registration: " + std::string(e.what()));
  }
}

Status HTTPRouterClient::SendHeartbeat(std::string& node_id, KVHeartbeatResponse& response) {
  // Add debug output
  KLLM_LOG_INFO << "Sending heartbeat for node ID: " << node_id;

  // According to the API, the heartbeat request only needs the node ID
  nlohmann::json data;
  data["node_id"] = node_id;

  try {
    // Send heartbeat request
    std::string response_str = MakeHttpRequest("/Heartbeat", "POST", data);

    nlohmann::json response_json;
    try {
      response_json = nlohmann::json::parse(response_str);
    } catch (const std::exception& e) {
      KLLM_LOG_ERROR << "Failed to parse heartbeat JSON response: " << e.what() << ", response: " << response_str;
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR,
                    "Failed to parse heartbeat JSON response: " + std::string(e.what()));
    }

    // Check if the response indicates an error
    if (response_str.find("不存在") != std::string::npos) {
      KLLM_LOG_ERROR << "Node not found in heartbeat response: " << response_str;
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Node not found in heartbeat response");
    }

    // Fill the response struct with basic data from the JSON
    response.node_id = response_json.value("node_id", "");
    response.is_online = response_json.value("is_online", false);
    response.group_ready = response_json.value("group_ready", false);
    response.node_role = response_json.value("node_role", "");
    response.timestamp = response_json.value("timestamp", "");

    // Only populate if present
    if (response_json.contains("node_rank")) {
      response.node_rank = response_json["node_rank"].get<int>();
    }

    if (response_json.contains("coordinator_addr")) {
      std::string coordinator_addr = response_json["coordinator_addr"].get<std::string>();
      node_info_.coordinator_addr = coordinator_addr;
      response.coordinator_addr = coordinator_addr;
    }

    if (response_json.contains("node_role")) {
      KLLM_LOG_INFO << "Node role from heartbeat: " << response_json["node_role"].get<std::string>();
    }

    // Parse comm_group_to_address (new tuple list format)
    if (response_json.contains("comm_group_to_address") && response_json["comm_group_to_address"].is_object()) {
      response.comm_group_to_address.clear();
      for (auto& [group_key, tuple_list] : response_json["comm_group_to_address"].items()) {
        std::vector<std::tuple<int, int, std::string>> addr;
        if (tuple_list.is_array()) {
          for (const auto& item : tuple_list) {
            addr.emplace_back(item[0], item[1], item[2].get<std::string>());
          }
        }
        response.comm_group_to_address[group_key] = std::move(addr);
      }
    }

    // Parse comm_group_to_id (new format)
    if (response_json.contains("comm_group_to_id") && response_json["comm_group_to_id"].is_object()) {
      response.comm_group_to_id.clear();
      for (auto& [group_key, comm_id] : response_json["comm_group_to_id"].items()) {
        if (comm_id.is_string()) {
          response.comm_group_to_id[group_key] = comm_id.get<std::string>();
        }
      }
    }

    // Check if the response contains a last heartbeat timestamp
    if (response_json.contains("timestamp")) {
      std::string timestamp = response_json["timestamp"].get<std::string>();
      node_info_.last_heartbeat = timestamp;
      KLLM_LOG_INFO << "Last heartbeat timestamp: " << timestamp;
    }

    // Log what we found
    KLLM_LOG_INFO << "Heartbeat response: online=" << response.is_online << ", ready=" << response.group_ready
                  << ", comm groups=" << response.comm_group_to_id.size();

    return Status();
  } catch (const std::exception& e) {
    KLLM_LOG_ERROR << "Failed to handle heartbeat response: " << e.what();
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to handle heartbeat response: " + std::string(e.what()));
  }
}

Status HTTPRouterClient::SendCommId(const std::string& node_id, const std::string& comm_key,
                                    const std::string& comm_id) {
  // Log debug information
  KLLM_LOG_INFO << "Registering Communication ID for node ID: " << node_id << ", comm_key: " << comm_key
                << ", Communication_id: " << comm_id;

  // Prepare the request data
  nlohmann::json data;
  data["node_id"] = node_id;
  data["comm_key"] = comm_key;
  data["comm_id"] = comm_id;

  try {
    // Send request to register the Communication ID
    std::string response_str = MakeHttpRequest("/RegisterCommId", "POST", data);

    // Parse the response
    nlohmann::json response_json = nlohmann::json::parse(response_str);

    // Check if the response indicates an error
    if (response_json.contains("detail")) {
      std::string error_message = response_json["detail"].get<std::string>();
      KLLM_LOG_ERROR << "Failed to register Communication ID: " << error_message;
      return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to register Comm ID: " + error_message);
    }

    // Check if the response contains the message indicating success
    if (response_json.contains("message")) {
      KLLM_LOG_INFO << "Communication ID registration response: " << response_json["message"].get<std::string>();
    }

    // Verify that the returned Communication ID matches the one we sent
    std::string returned_comm_id = "";
    if (response_json.contains("comm_id")) {
      returned_comm_id = response_json["comm_id"].get<std::string>();

      if (returned_comm_id != comm_id) {
        KLLM_LOG_ERROR << "Returned Communication ID does not match the sent Communication ID. Sent: " << comm_id
                       << ", Received: " << returned_comm_id;
        return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR,
                      "comm ID mismatch: returned=" + returned_comm_id + ", expected=" + comm_id);
      }

      KLLM_LOG_INFO << "Successfully registered Communication ID " << comm_id << " for communication group "
                    << comm_key;
    }

    return Status();
  } catch (const std::exception& e) {
    KLLM_LOG_ERROR << "Failed to register Communication ID: " << e.what();
    return Status(RetCode::RET_INTERNAL_UNKNOWN_ERROR, "Failed to register Communication ID: " + std::string(e.what()));
  }
}
}  // namespace ksana_llm
