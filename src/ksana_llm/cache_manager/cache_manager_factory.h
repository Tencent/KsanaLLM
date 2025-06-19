/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/cache_manager/cache_manager_interface.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

// Create CacheManager instance.
class CacheManagerFactory {
 public:
  // Create a prefix cache manager instance.
  static std::shared_ptr<CacheManagerInterface> CreateCacheManager(
      const CacheManagerConfig& cache_manager_config,
      std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group);
};

}  // namespace ksana_llm
