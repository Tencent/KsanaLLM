
/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/periphery/version_reporter.h"
#include <arpa/inet.h>
#include <curl/curl.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>
#include "ksana_llm/periphery/version_info.h"
#include "ksana_llm/utils/logger.h"

using json = nlohmann::json;

namespace ksana_llm {

// Network utility constants

// Function to check if an IPv4 address is private
// This function determines if the given IPv4 address is a private address.
// Private IPv4 addresses include the following ranges:
// - 0.0.0.0 (unspecified address)
// - 10.0.0.0 to 10.255.255.255 (10.0.0.0/8)
// - 172.16.0.0 to 172.31.255.255 (172.16.0.0/12)
// - 192.168.0.0 to 192.168.255.255 (192.168.0.0/16)
// - 127.0.0.0 to 127.255.255.255 (127.0.0.0/8) (loopback address)
bool IsPrivateIPv4(uint32_t addr) {
  return addr == 0 ||              // Check if the address is 0.0.0.0 (unspecified address)
         (addr >> 20) == 0xac1 ||  // Check if the address is in the range 172.16.0.0 to 172.31.255.255 (172.16.0.0/12)
         (addr >> 16) ==
             0xc0a8 ||          // Check if the address is in the range 192.168.0.0 to 192.168.255.255 (192.168.0.0/16)
         (addr >> 24) == 0x7f;  // Check if the address is in the range 127.0.0.0 to 127.255.255.255 (127.0.0.0/8)
                                // (loopback address)
}

#ifndef IN6_IS_ADDR_UNIQUELOCAL
#  define IN6_IS_ADDR_UNIQUELOCAL(a) ((((__const uint32_t *)(a))[0] & htonl(0xfe000000)) == htonl(0xfc000000))
#endif

// Function to check if an IPv6 address is private
// This function determines if the given IPv6 address is a private address.
// Private IPv6 addresses include unspecified addresses, loopback addresses,
// site-local addresses, unique local addresses, and link-local addresses.
bool IsPrivateIPv6(const in6_addr *addr) {
  return IN6_IS_ADDR_UNSPECIFIED(addr) ||  // Check if the address is unspecified (::)
         IN6_IS_ADDR_LOOPBACK(addr) ||     // Check if the address is loopback (::1)
         IN6_IS_ADDR_SITELOCAL(addr) ||    // Check if the address is site-local (deprecated)
         IN6_IS_ADDR_UNIQUELOCAL(addr) ||  // Check if the address is unique local (fc00::/7)
         IN6_IS_ADDR_LINKLOCAL(addr);      // Check if the address is link-local (fe80::/10)
}

// Function to get the default IP address of the machine
// This function retrieves the default IP address of the machine by iterating over
// the network interfaces and selecting the first non-private IPv4 or IPv6 address found.
// It uses the getifaddrs function to get the list of network interfaces and their addresses.
// The function returns the first non-private IPv4 address if available, otherwise it returns
// the first non-private IPv6 address. If no suitable address is found, it returns an empty string.
std::string GetDefaultIP() {
#if !defined(__arm__) && !defined(__aarch64__)
  struct ifaddrs *ifaddr;
  if (getifaddrs(&ifaddr) == -1) {
    KLLM_LOG_DEBUG << "getifaddrs failed when getting default ip slowly";
    return "";
  }

  union {
    char v4[INET_ADDRSTRLEN];
    char v6[INET6_ADDRSTRLEN];
  } ip_buf;
  std::string ipv4, ipv4_nic, ipv6, ipv6_nic;
  for (struct ifaddrs *ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == NULL) {
      continue;
    }
    int family = ifa->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) {
      continue;
    }

    if (family == AF_INET && ipv4.empty()) {
      auto sin_addr = &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
      uint32_t addr = ntohl(sin_addr->s_addr);
      if (IsPrivateIPv4(addr)) {
        continue;
      }
      ipv4_nic.assign(ifa->ifa_name);
      ipv4.assign(inet_ntop(family, sin_addr, ip_buf.v4, INET_ADDRSTRLEN));
    } else if (family == AF_INET6 && ipv6.empty()) {
      auto sin_addr = &((struct sockaddr_in6 *)ifa->ifa_addr)->sin6_addr;
      if (IsPrivateIPv6(sin_addr)) {
        continue;
      }
      ipv6_nic.assign(ifa->ifa_name);
      ipv6.assign(inet_ntop(family, sin_addr, ip_buf.v6, INET6_ADDRSTRLEN));
    }
  }

  freeifaddrs(ifaddr);

  if (!ipv4.empty()) {
    KLLM_LOG_DEBUG << "get default ipv4 slowly from " << ipv4_nic << ", " << ipv4;
    return ipv4;
  }
  if (!ipv6.empty()) {
    KLLM_LOG_DEBUG << "get default ipv6 slowly from " << ipv6_nic << ", " << ipv6;
    return ipv6;
  }
#endif
  return "";
}

// Function to check if the code is running inside a Docker container
std::string IsInDocker() {
  std::ifstream cgroup_file("/proc/self/cgroup");
  if (!cgroup_file.is_open()) {
    return "";  // Return an empty string if no matching line is found
  }

  std::string line;
  while (std::getline(cgroup_file, line)) {
    if (line.find("cpuset") != std::string::npos &&
        (line.find("docker") != std::string::npos || line.find("kubepods") != std::string::npos)) {
      // Extract the substring after the second colon
      size_t secondColon = line.find(':', line.find(':') + 1);
      if (secondColon != std::string::npos) {
        std::string result = line.substr(secondColon + 1);

        /*
         * in /proc/self/cgroup data schema is:
         * 9:cpuset:/docker/a31cc35bd4752b553c2519a60e7d63e2e8fb2bb9846ce116805bccbee71ae96e
         *
         * get column 3 (:) and Replace '/' with '-'
         *  -docker-a31cc35bd4752b553c2519a60e7d63e2e8fb2bb9846ce116805bccbee71ae96e
         * */
        std::replace(result.begin(), result.end(), '/', '-');
        return result;
      } else {
        return "";  // Return an empty string if no matching line is found
      }
    }
  }
  return "";  // Return an empty string if no matching line is found
}

std::string GetEnvVar(const std::string &var) {
  const char *val = std::getenv(var.c_str());
  return val == nullptr ? std::string("unknow") : std::string(val);
}

std::string GetDefaultContainerName() {
  std::string cgroup_name = IsInDocker();
  if (cgroup_name == "") {
    return "";
  }

  // get k8s pod name like 'cls-q0jtdas2-308c51d66bababfda150da72eaf49474-1'
  std::string pod_name = GetEnvVar("POD_NAME");
  std::string sumeru_pod_name = GetEnvVar("SUMERU_POD_NAME");

  if (pod_name != "unknow") {
    return pod_name;
  } else if (sumeru_pod_name != "unknow") {
    return sumeru_pod_name;
  } else if (cgroup_name != "") {
    return cgroup_name;
  } else {
    return GetDefaultIP();
  }
}

uint64_t GetCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();

  auto duration = now.time_since_epoch();

  auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  return static_cast<uint64_t>(milliseconds);
}

// Implementation of VersionReporter class

// Function to initialize version reporting with given options
bool VersionReporter::Init(const ReportOption &option) {
  std::lock_guard<std::mutex> lock(version_report_mutex);
  if (version_report_init_flag_) {
    return true;
  }

  VersionInfo info;

  info.app = GetEnvVar("SUMERU_APP");
  info.server = GetEnvVar("SUMERU_SERVER");
  info.ip = GetDefaultIP();
  info.container_name = GetDefaultContainerName();
  info.version = GIT_VERSION;
  info.commit_hash = GIT_COMMIT_HASH;
  info.branch = GIT_BRANCH;

  version_info_ = std::make_shared<VersionInfo>(info);
  option_ = std::make_shared<ReportOption>(option);

  if (!Start()) return false;
  version_report_init_flag_ = true;
  initialized_ = true;

  return true;
}

bool VersionReporter::Start() {
  if (!running_) {
    running_ = true;
    report_thread_ = std::thread([this]() {
      std::unique_lock<std::mutex> lock(stop_cv_mutex_);
      while (running_) {
        lock.unlock();
        ReportFunction();
        std::uint64_t interval = ExecutionInterval();
        lock.lock();
        stop_cv_.wait_for(lock, std::chrono::milliseconds(interval), [this] { return !running_; });
      }
    });
    return true;
  }
  return false;
}

void VersionReporter::StopReporting() { Stop(); }

void VersionReporter::Destroy() {
  std::lock_guard<std::mutex> lock(version_report_mutex);
  if (!version_report_init_flag_) {
    return;
  }

  Stop();

  version_report_init_flag_ = false;
  initialized_ = false;
}

void VersionReporter::Stop() {
  if (running_) {
    running_ = false;
    stop_cv_.notify_one();
    if (report_thread_.joinable()) {
      report_thread_.join();
    }
  }
}

std::uint64_t VersionReporter::ExecutionInterval() {
  if (report_result_.last_report_time == 0) return 1000;
  return report_result_.is_succ ? option_->report_interval : option_->fail_report_interval;
}

size_t DummyWriteCallback(void *contents, size_t size, size_t nmemb, void *userp) { return size * nmemb; }

void VersionReporter::ReportFunction() {
  // Serialize VersionInfo to JSON using nlohmann::json
  json j;
  j["app"] = version_info_->app;
  j["server"] = version_info_->server;
  j["ip"] = version_info_->ip;
  j["container_name"] = version_info_->container_name;
  j["version"] = version_info_->version;
  j["commit_hash"] = version_info_->commit_hash;
  j["branch"] = version_info_->branch;

  std::string result = j.dump();

  // Initialize CURL
  CURL *curl = curl_easy_init();
  if (!curl) {
    KLLM_LOG_DEBUG << "Failed to initialize CURL.";
    report_result_.is_succ = false;
    return;
  }

  // Set CURL options
  std::string url = "http://" + option_->report_host + option_->report_api;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, result.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, result.size());

  // Set HTTP headers
  struct curl_slist *headers = nullptr;
  std::string content_length = "Content-Length: " + std::to_string(result.size());
  std::string host = "Host: " + option_->report_host;

  headers = curl_slist_append(headers, content_length.c_str());
  headers = curl_slist_append(headers, host.c_str());
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "Accept: */*");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, DummyWriteCallback);

  // Perform the request
  CURLcode res = curl_easy_perform(curl);

  // curl uses the long type, but due to cpp lint, we default
  // to a 64-bit machine and write it as int64_t type
  int64_t http_code = 0;
  if (res != CURLE_OK) {
    KLLM_LOG_DEBUG << "curl_easy_perform() failed: " << curl_easy_strerror(res);
    KLLM_LOG_DEBUG << "HTTP request data is: " << result;
    report_result_.is_succ = false;
  } else {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code != 200) {
      KLLM_LOG_DEBUG << "HTTP request failed with code: " << http_code;
      KLLM_LOG_DEBUG << "HTTP request data is: " << result;
      KLLM_LOG_DEBUG << "curl_easy_perform() failed: " << curl_easy_strerror(res);
      report_result_.is_succ = false;
    } else {
      report_result_.is_succ = true;
    }
  }

  report_result_.last_report_time = GetCurrentTimestamp();
  // Clean up
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
}
}  // namespace ksana_llm
