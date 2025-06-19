/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>
#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/distributed/packet_type.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// Initialize and destroy the data hub.
void InitializeScheduleOutputPool();
void InitializeHiddenUnitBufferPool();

void DestroyScheduleOutputPool();
void DestroyHiddenUnitBufferPool();

// Get the object pool of schedule output.
ScheduleOutputPool* GetScheduleOutputPool();

// Get the object pool of hidden unit buffer.
HiddenUnitBufferPool* GetHiddenUnitBufferPool();

// Set and get current device buffer for compute thread.
void SetCurrentHiddenUnitBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer);

HiddenUnitDeviceBuffer* GetCurrentHiddenUnitBuffer(size_t schedule_id);

Status CopyFromHiddenUnitBuffer(Tensor& tensor, HiddenUnitDeviceBuffer* device_buffer, int rank, bool is_prefill);
Status CopyToHiddenUnitBuffer(HiddenUnitDeviceBuffer* device_buffer, Tensor& tensor, int rank, bool is_prefill);
Status CopyToHiddenUnitBuffer(HiddenUnitDeviceBuffer* device_buffer, void* device_ptr, std::vector<size_t> shape,
                              DataType dtype, int rank, bool is_prefill);
Status CopyHostMemToHiddenUnitBuffer(HiddenUnitDeviceBuffer* device_buffer, void* host_ptr, std::vector<size_t> shape,
                                  DataType dtype, int rank, bool is_prefill);
Status CopyHiddenUnitBufferToHostMem(void* host_ptr, HiddenUnitDeviceBuffer* device_buffer, std::vector<size_t> shape,
                                     DataType dtype, int rank, bool is_prefill);

// Broadcast to all workers.
Status BroadcastScheduleOutput(ScheduleOutput* schedule_output);

// Set default current hidden units.
Status InitHiddenUnits(size_t schedule_id);

// Send hidden_units to downstream.
Status SendHiddenUnits(size_t schedule_id);

// Reset waiter for recv operation.
Status ResetReceiveWaiter();

// Recv from remote and set as current device buffer.
Status RecvHiddenUnits(bool do_recv, size_t schedule_id);

// Free current hidden_unit.
Status FreeHiddenUnits(size_t schedule_id);

// Set and get hidden unit.
Status GetHiddenUnitMeta(std::vector<size_t>& shape, DataType& data_type);
Status SetHiddenUnitMeta(const std::vector<size_t>& shape, DataType data_type);

// Get and set model instance.
Status SetModelInstance(const std::string model_name, std::shared_ptr<ModelInstance> model_instance);
std::shared_ptr<ModelInstance> GetModelInstance(const std::string& model_name);
void DestroyModelInstance();

// Get and set cache_managers, for worker only.
Status SetCacheManagers(const std::vector<std::shared_ptr<CacheManagerInterface>>& cache_managers);
std::shared_ptr<CacheManagerInterface> GetCacheManager(int group_id);

}  // namespace ksana_llm
