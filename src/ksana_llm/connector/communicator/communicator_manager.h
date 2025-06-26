/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "ksana_llm/connector/config.h"
#include "ksana_llm/connector/coordinator/default_coordinator.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

class Communicator;     // forward declare
class ZmqCommunicator;  // forward declare
#ifdef ENABLE_CUDA
class NcclCommunicator;  // forward declare
#endif
/**
 * @class CommunicatorManager
 * @brief Manages communication across different protocols (ZMQ, NCCL)
 *
 * This class provides a unified interface for managing multiple communication protocols.
 * It holds a single Coordinator instance and distributes heartbeat data to registered communicators.
 * It manages communicators by type string rather than using templates, reducing complexity.
 */
class CommunicatorManager {
 public:
  /**
   * @brief Constructor
   * @param config 全局配置（包含node_rank、world_size、device_count）
   * @param coordinator 协调器，所有权转移
   */
  CommunicatorManager(const ConnectorConfig& config, std::shared_ptr<Coordinator> coordinator);

  /**
   * @brief Destructor
   */
  virtual ~CommunicatorManager();

  /**
   * @brief Initialize all communicators
   *
   * @return Status indicating success or failure
   */
  virtual Status Initialize();

  /**
   * @brief Shutdown all communicators
   */
  virtual void Shutdown();

  /**
   * @brief Get a communicator by type name
   *
   * @param type_name Type name of the communicator (e.g., "ZMQ", "NCCL")
   * @return Pointer to communicator (nullptr if not found)
   */
  virtual Communicator* GetCommunicatorByType(const std::string& type_name) const;

  /**
   * @brief Get ZMQ communicator (convenience method)
   *
   * @return Pointer to ZMQ communicator (nullptr if not found)
   */
  virtual ZmqCommunicator* GetZmqCommunicator() const;
#ifdef ENABLE_CUDA
  /**
   * @brief Get NCCL communicator (convenience method)
   *
   * @return Pointer to NCCL communicator (nullptr if not found)
   */
  virtual NcclCommunicator* GetNcclCommunicator() const;
#endif
  /**
   * @brief Process heartbeat data and distribute to communicators
   *
   * @param comm_group_to_id Map of group keys to communication IDs
   * @param comm_group_to_address Map of group keys to address tuples
   * @return Status indicating success or failure
   */
  virtual Status ProcessHeartbeatData(
      const std::unordered_map<std::string, std::string>& comm_group_to_id,
      const std::unordered_map<std::string, std::vector<std::tuple<int, int, std::string>>>& comm_group_to_address);

  /**
   * @brief Create ZMQ communicator
   *
   * @return Status indicating success or failure
   */
  virtual Status CreateZmqCommunicator();
#ifdef ENABLE_CUDA
  /**
   * @brief Create NCCL communicator
   *
   * @return Status indicating success or failure
   */
  virtual Status CreateNcclCommunicator();
#endif
  /**
   * @brief Send communication ID through coordinator
   *
   * @param group_key Key identifying the communication group
   * @param comm_id Communication ID to send
   * @return Status indicating success or failure
   */
  virtual Status SendCommId(const std::string& group_key, const std::string& comm_id);

  /**
   * @brief Set ZMQ communicator (for testing purposes)
   *
   * @param communicator Unique pointer to a ZMQ communicator instance
   */
  virtual void SetZmqCommunicator(std::unique_ptr<ZmqCommunicator> communicator);
#ifdef ENABLE_CUDA
  /**
   * @brief Set NCCL communicator (for testing purposes)
   *
   * @param communicator Unique pointer to a NCCL communicator instance
   */
  virtual void SetNcclCommunicator(std::unique_ptr<NcclCommunicator> communicator);
#endif
  virtual bool IsInitialized() const;

 private:
  /** @brief Configuration for communication */
  ConnectorConfig config_;
  /** @brief Single coordinator instance for all communicators */
  std::shared_ptr<Coordinator> coordinator_;
  bool initialized_ = false;  // Flag to indicate if initialized
  bool shutdown_ = false;     // Flag to indicate if shutdown

  /** @brief Map of communicator type names to communicator instances */
  std::unordered_map<std::string, std::unique_ptr<Communicator>> communicators_;

  /** @brief Mutex for thread safety */
  mutable std::mutex mutex_;

  /** @brief Set of enabled communicator types */
  std::unordered_set<std::string> enabled_communicator_types_;
};

}  // namespace ksana_llm
