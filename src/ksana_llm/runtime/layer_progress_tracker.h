/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

// 用于跟踪每张卡执行到的层位置
class LayerProgressTracker {
 public:
  using LayerProgressCallback = std::function<void(int device_id, int layer_index)>;

  struct EventInfo {
    Event event;
    int device_id;
    int layer_index;
    bool processed;

    // 用于优先队列的比较函数，按 layer_index 从小到大排序
    bool operator<(const EventInfo& other) const {
      // 注意：优先队列默认是最大堆，所以这里需要反向比较
      return layer_index > other.layer_index;
    }
  };

  // 初始化 LayerProgressTracker
  void Initialize(int max_devices, int max_layers);

  // 清理资源
  void Cleanup();

  // 注册回调函数
  void RegisterCallback(LayerProgressCallback callback);

  // 记录层执行进度（由 attention 层调用）
  void RecordLayerProgress(int device_id, int layer_index, Stream& stream);

  // 获取指定设备的当前层进度
  int GetLayerProgress(int device_id);

  // 内部辅助函数，不获取锁，用于在已持有锁的情况下获取层进度
  int GetLayerProgressNoLock(int device_id) const;

  // 重置当前状态，清除所有进度记录，但保留 CUDA event 以便复用
  void ResetState();

  // 使用 Singleton 类实现单例模式，需要将构造函数和析构函数设为公有
  LayerProgressTracker();
  ~LayerProgressTracker();

  // 声明 Singleton 为友元类，使其能够访问私有构造函数
  friend class Singleton<LayerProgressTracker>;

  // 监控线程函数，用于等待 event 完成并通知
  void MonitorEvents();

 private:
  std::mutex mutex_;
  std::unordered_map<int, int> device_layer_progress_;  // device_id -> layer_index
  std::vector<LayerProgressCallback> callbacks_;

  // 自定义 pair 的哈希函数
  struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
      return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
  };

  // 持久的 CUDA event 管理 - 使用 unordered_map 快速查找
  std::unordered_map<std::pair<int, int>, EventInfo, PairHash> events_map_;

  // 优先队列，按 layer_index 从小到大排序
  std::priority_queue<EventInfo> pending_events_;

  // 监控线程
  std::thread monitor_thread_;
  bool monitor_thread_running_;
};

}  // namespace ksana_llm