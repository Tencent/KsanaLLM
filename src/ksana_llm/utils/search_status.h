// Copyright 2025 Tencent Inc.  All rights reserved.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace ksana_llm {

class CutlassSearchStatus {
 public:
  bool IsCutlassScheduleContain(size_t n, size_t k) {
    auto it1 = cutlass_schedule_cache.find(n);
    if (it1 != cutlass_schedule_cache.end()) {
      auto it2 = it1->second.find(k);
      if (it2 != it1->second.end()) {
        return true;
      }
    }
    return false;
  }

  bool IsCutlassWorkspaceContain(size_t m, size_t n, size_t k) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    auto it = cutlass_workspace_cache.find(index_str);
    if (it != cutlass_workspace_cache.end()) {
      return true;
    }
    return false;
  }

  void AddCutlassSchedule(size_t n, size_t k, std::vector<size_t> best) { cutlass_schedule_cache[n][k] = best; }

  void AddCutlassWorkspace(size_t m, size_t n, size_t k, size_t ws) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    cutlass_workspace_cache[index_str] = ws;
  }

  std::vector<size_t>& GetCutlassSchedule(size_t n, size_t k) { return cutlass_schedule_cache[n][k]; }

  size_t GetCutlassWorkspace(size_t m, size_t n, size_t k) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    return cutlass_workspace_cache[index_str];
  }

  void ClearCutlassSchedule() { cutlass_schedule_cache.clear(); }

  void ClearCutlassWorkspace() { cutlass_workspace_cache.clear(); }

 private:
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t>>> cutlass_schedule_cache;
  std::unordered_map<std::string, size_t> cutlass_workspace_cache;
};

class MacheteSearchStatus {
 public:
  bool IsMacheteScheduleContain(size_t n, size_t k) {
    auto it1 = machete_schedule_cache.find(n);
    if (it1 != machete_schedule_cache.end()) {
      auto it2 = it1->second.find(k);
      if (it2 != it1->second.end()) {
        return true;
      }
    }
    return false;
  }

  bool IsMacheteWorkspaceContain(size_t m, size_t n, size_t k) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    auto it = machete_workspace_cache.find(index_str);
    if (it != machete_workspace_cache.end()) {
      return true;
    }
    return false;
  }

  void AddMacheteSchedule(size_t n, size_t k, std::vector<std::string> best) { machete_schedule_cache[n][k] = best; }

  void AddMacheteWorkspace(size_t m, size_t n, size_t k, size_t ws) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    machete_workspace_cache[index_str] = ws;
  }

  std::vector<std::string>& GetMacheteSchedule(size_t n, size_t k) { return machete_schedule_cache[n][k]; }

  size_t GetMacheteWorkspace(size_t m, size_t n, size_t k) {
    std::string index_str = fmt::format("({},{},{})", m, n, k);
    return machete_workspace_cache[index_str];
  }

  void ClearMacheteSchedule() { machete_schedule_cache.clear(); }

  void ClearMacheteWorkspace() { machete_workspace_cache.clear(); }

 private:
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<std::string>>> machete_schedule_cache;
  std::unordered_map<std::string, size_t> machete_workspace_cache;
};

}  // namespace ksana_llm
