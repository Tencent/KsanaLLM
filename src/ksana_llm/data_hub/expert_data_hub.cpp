/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/data_hub/expert_data_hub.h"
#include <cstddef>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/distributed/control_message.h"
#include "ksana_llm/distributed/packet_type.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// Used to connect with DeepEP
std::shared_ptr<ExpertParallelDeepepWrapper> g_deepep_wrapper = nullptr;

const std::shared_ptr<ExpertParallelDeepepWrapper>& GetExpertParallelDeepepWrapper() { return g_deepep_wrapper; }

void SetExpertParallelDeepepWrapper(const std::shared_ptr<ExpertParallelDeepepWrapper>& deepep_wrapper) {
  g_deepep_wrapper = deepep_wrapper;
}

Status InitializeExpertParallelDeepepWrapper(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                             const std::shared_ptr<Context>& context) {
  size_t num_ranks = runtime_config.parallel_basic_config.expert_world_size *
                     runtime_config.parallel_basic_config.expert_parallel_size;
  size_t num_ranks_per_node = runtime_config.parallel_basic_config.expert_parallel_size;
  size_t max_token_num = runtime_config.max_step_token_num;
  size_t hidden_size = static_cast<size_t>(model_config.hidden_units);
  size_t expert_topk = model_config.moe_config.experts_topk;
  size_t num_experts = model_config.moe_config.num_experts;
  size_t node_rank = context->GetExpertParallelExpertNodeRank();
  // TODO(zezhao): 当前仅有开启 enable_full_shared_expert 时才会使用 DeepEP，后续会将该变量更名并调整在 EP 中的耦合关系.
  if (num_ranks > 1 && runtime_config.enable_full_shared_expert) {
    // TODO(zezhao): 临时限制，Expert-Parallel 开启时 Data-Parallel 需相等，后续将支持 ATP > 1
    if (runtime_config.parallel_basic_config.expert_parallel_size !=
        runtime_config.parallel_basic_config.attn_data_parallel_size) {
      KLLM_LOG_ERROR << fmt::format("Expert-Parallel only supports DP=EP");
      KLLM_THROW(fmt::format("Expert-Parallel only supports DP=EP"));
    }
    // Initialize deepep_wrapper when using Expert-Parallel
    g_deepep_wrapper = std::make_shared<ExpertParallelDeepepWrapper>(
        num_ranks, num_ranks_per_node, node_rank, max_token_num, hidden_size, expert_topk, num_experts, context);
    g_deepep_wrapper->Init();
  }
  return Status();
}

}  // namespace ksana_llm
