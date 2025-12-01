/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/llm_runtime.h"

#include <algorithm>
#include <atomic>
#include <execution>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/generation_controller.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/model_instance.h"
#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

LlmRuntime::LlmRuntime(const BatchSchedulerConfig& batch_scheduler_config, const RuntimeConfig& runtime_config,
                       std::shared_ptr<Context> context)
    : mtp_step_num_(batch_scheduler_config.mtp_step_num), context_(context), threadpool_(4) {
  worker_group_ = std::make_shared<WorkerGroup>(context_->GetTensorParallelSize(),
                                                batch_scheduler_config.max_pp_batch_num, context_);

  mtp_prepared_data_.resize(runtime_config.max_pp_batch_num);

  samplers_.resize(context_->GetTensorParallelSize());
  for (size_t worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    samplers_[worker_id] = std::make_shared<Sampler>(batch_scheduler_config, worker_id, context_);
  }

  threadpool_.Start();
}

void LlmRuntime::BuildForwardRequests(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
                                      std::map<ModelInstance*, std::vector<ForwardRequest*>>& grouped_reqs) {
  PROFILE_EVENT_SCOPE(BuildForwardRequests, fmt::format("BuildForwardRequests_{}", multi_batch_id));
  grouped_reqs.clear();
  for (auto& req : reqs) {
    ++req->step;
    auto& model_reqs = grouped_reqs[req->model_instance.get()];
    model_reqs.reserve(reqs.size());
    model_reqs.emplace_back(req->GetForwardRequest());
  }
}

void LlmRuntime::BuildForwardRequests(std::vector<std::shared_ptr<WorkerInferRequest>>& reqs,
                                      std::map<ModelInstance*, std::vector<ForwardRequest*>>& grouped_reqs) {
  PROFILE_EVENT_SCOPE(BuildForwardRequests, "BuildForwardRequests");
  for (auto& req : reqs) {
    ++req->step;
    auto& model_reqs = grouped_reqs[req->model_instance.get()];
    model_reqs.reserve(reqs.size());
    model_reqs.emplace_back(req->GetForwardRequest());
  }
}

void LlmRuntime::BuildSamplingRequest(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
                                      std::vector<SamplingRequest*>& sampling_reqs, bool enable_main_layers_sampler) {
  PROFILE_EVENT_SCOPE(BuildSamplingRequest_, fmt::format("BuildSamplingRequest_{}", multi_batch_id));
  sampling_reqs.resize(reqs.size());
  for (size_t i = 0; i < reqs.size(); ++i) {
    sampling_reqs[i] = reqs[i]->GetSamplingRequest(multi_batch_id);
  }
}

void LlmRuntime::Forward(size_t multi_batch_id, std::map<ModelInstance*, std::vector<ForwardRequest*>>& grouped_reqs,
                         bool epilogue, RunMode run_mode) {
  PROFILE_EVENT_SCOPE(Forward_, fmt::format("Forward_{}_{}", multi_batch_id, epilogue));

  // this wg only indicates completion the worker threads have finished, the device may not synchronized.
  auto wg = std::make_shared<WaitGroup>(0, true);
  // context decode and decode run serially in single thread
  const bool run_serially = context_->IsRunContextDecodeAndDecodeSerially();
  for (auto& [model_inst, reqs] : grouped_reqs) {
#ifdef ENABLE_CUDA
    model_inst->ForwardAsync(multi_batch_id, worker_group_, reqs, epilogue, wg, run_mode);
    if (run_serially) {
      wg->Wait();
    }
#else
    for (const auto& stage : {InferStage::kContext, InferStage::kDecode}) {
      std::vector<ForwardRequest*> stage_reqs;
      stage_reqs.reserve(reqs.size());
      for (const auto& req : reqs) {
        if (req->infer_stage == stage) {
          stage_reqs.emplace_back(req);
        }
      }
      if (stage_reqs.empty()) {
        continue;
      }
      model_inst->ForwardAsync(multi_batch_id, worker_group_, stage_reqs, epilogue, wg, run_mode);
      if (run_serially) {
        wg->Wait();
      }
    }
#endif
  }
  wg->Wait();
}

void LlmRuntime::Sampling(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
                          std::vector<SamplingRequest*>& sampling_reqs, bool enable_main_layers_sampler) {
  PROFILE_EVENT_SCOPE(Sampling, fmt::format("Sampling_{}", multi_batch_id));

  // Take the shortcut when all sampling are greedy
  if (int* output_tokens_ptr = reqs.front()->model_instance->GetOutputTokensPtr(multi_batch_id).front();
      output_tokens_ptr != nullptr) {
    for (auto& sampling_req : sampling_reqs) {
      sampling_req->sampling_result_tokens->insert(
          sampling_req->sampling_result_tokens->end(), output_tokens_ptr + sampling_req->logits_offset,
          output_tokens_ptr + sampling_req->logits_offset + sampling_req->sampling_token_num);
    }
  } else {
    auto wg = std::make_shared<WaitGroup>(context_->GetTensorParallelSize(), true);
    for (size_t worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
      worker_group_->GetWorker(worker_id)->SamplingAsync(multi_batch_id, samplers_[worker_id], sampling_reqs, wg);
    }
    wg->Wait();
  }

  threadpool_.Submit([reqs]() mutable {
    const auto current_time = ProfileTimer::GetCurrentTimeInUs();
    static std::atomic<time_t> s_last_call_finish_time_us{0};
    const time_t prev = s_last_call_finish_time_us.exchange(current_time);
    if (prev != 0 && current_time > prev) {
      REPORT_METRIC("inter_token_interval_latency_us", current_time - prev);
    }
    std::for_each(std::execution::par_unseq, reqs.begin(), reqs.end(), [current_time](const auto& req) {
      if (req->step == 1) {
        REPORT_METRIC("time_to_first_token_us", current_time - req->timestamp_in_us);
      }
    });
  });
}

std::shared_ptr<WaitGroup> LlmRuntime::PrepareMtpInfoAsync(const size_t multi_batch_id,
                                                           const std::vector<std::shared_ptr<InferRequest>>& reqs) {
  if (mtp_step_num_ == 0 || !context_->IsChief()) {
    return std::make_shared<WaitGroup>();
  }

  auto wg = std::make_shared<WaitGroup>(1);
  threadpool_.Submit([&, wg, multi_batch_id]() {
    auto& batch_prepared_data = mtp_prepared_data_[multi_batch_id];
    batch_prepared_data.clear();
    batch_prepared_data.reserve(reqs.size());
    for (size_t i = 0; i < reqs.size(); ++i) {
      const auto& req = reqs[i];
      auto& prepare_data = batch_prepared_data[req->req_id];
      prepare_data.tokens = std::make_shared<std::vector<int>>();
      prepare_data.tokens->reserve(req->forwarding_tokens.size() + mtp_step_num_);
      prepare_data.tokens->assign(req->forwarding_tokens.begin() + 1, req->forwarding_tokens.end());
      prepare_data.infer_req = req.get();
      prepare_data.order_pos = i;
    }
    wg->Done();
  });
  return wg;
}

Status LlmRuntime::MtpForward(const size_t multi_batch_id,
                              std::map<ModelInstance*, std::vector<ForwardRequest*>>& grouped_reqs,
                              std::vector<SamplingRequest*>& sampling_reqs,
                              std::vector<std::shared_ptr<InferRequest>>& reqs, const bool epilogue) {
  if (mtp_step_num_ == 0 || !context_->IsChief()) {
    return Status();
  }
  KLLM_CHECK_WITH_INFO(grouped_reqs.size() == 1, "only support 1 model");
  auto& forward_reqs = grouped_reqs.begin()->second;

  // update forward request
  auto& batch_prepared_data = mtp_prepared_data_[multi_batch_id];
  for (auto& forward_req : forward_reqs) {
    auto& prepared = batch_prepared_data[forward_req->req_id];
    prepared.infer_req->draft_tokens.mtp.clear();
    forward_req->forwarding_tokens = prepared.tokens;
    forward_req->forwarding_tokens->resize(forward_req->forwarding_tokens->size() -
                                           prepared.infer_req->forwarding_tokens_draft_num +
                                           prepared.infer_req->accepted_tokens.size());
    forward_req->forwarding_tokens->insert(forward_req->forwarding_tokens->end(),
                                           prepared.infer_req->generated_tokens.begin(),
                                           prepared.infer_req->generated_tokens.end());
    forward_req->sampling_token_num = kStepGenerateTokenNum;
  }
  ReorderInferRequests(forward_reqs);

  // update sampling request async
  WaitGroup sampling_wg(1);
  threadpool_.Submit([&]() {
    std::vector<SamplingRequest*> updated_reqs(sampling_reqs.size());
    for (size_t i = 0; i < forward_reqs.size(); ++i) {
      updated_reqs[i] = sampling_reqs[batch_prepared_data[forward_reqs[i]->req_id].order_pos];
    }
    for (size_t i = 0; i < updated_reqs.size(); ++i) {
      auto& sampling_req = updated_reqs[i];
      auto& forward_req = *forward_reqs[i];
      sampling_req->forwarding_tokens = forward_req.forwarding_tokens.get();
      sampling_req->sampling_token_num = forward_req.sampling_token_num;
      sampling_req->logits_offset = forward_req.logits_offset;
      sampling_req->apply_structured_constraint = false;
      sampling_req->enable_mtp_sampler = true;
      sampling_req->sampling_result_tokens->clear();
    }
    sampling_reqs = std::move(updated_reqs);
    sampling_wg.Done();
  });

  for (size_t mtp_step = 0; mtp_step < mtp_step_num_; ++mtp_step) {
    KLLM_LOG_DEBUG << "MTP forward step: " << mtp_step;
    context_->SetIsLastLayer(mtp_step + 1 == mtp_step_num_);

    Forward(multi_batch_id, grouped_reqs, epilogue, RunMode::kNextN);
    sampling_wg.Wait();
    Sampling(multi_batch_id, reqs, sampling_reqs, false);

    for (auto& req : forward_reqs) {
      auto& infer_req = batch_prepared_data[req->req_id].infer_req;
      infer_req->draft_tokens.mtp.emplace_back(infer_req->sampling_result_tokens.back());
      if (mtp_step != mtp_step_num_ - 1) {
        req->kv_cached_token_num = req->forwarding_tokens->size();
        req->prefix_cache_len = req->kv_cached_token_num;
        req->forwarding_tokens->emplace_back(infer_req->draft_tokens.mtp.back());
        infer_req->sampling_result_tokens.clear();
      }
    }
  }

  return Status();
}

// Speculative decoding, generate draft token with trie
void LlmRuntime::GenerateDraftToken(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  if (draft_generator_controller_ == nullptr) {
    return;
  }
  PROFILE_EVENT_SCOPE(GenerateDraftToken_, fmt::format("GenerateDraftToken"));
  for (auto& req : reqs) {
    draft_generator_controller_->GenerateDraft(req);
  }
}

void LlmRuntime::TransferGeneratedToken(std::vector<std::shared_ptr<InferRequest>>& reqs,
                                        std::shared_ptr<TransferEngine> transfer_engine) {
  if (transfer_engine->GetGroupRole() != GroupRole::PREFILL) {
    return;
  }

  std::vector<std::tuple<std::string, int, std::vector<int>>> reqs_transfer_tokens;
  for (auto& req : reqs) {
    std::vector<int> draft_tokens = req->draft_tokens.GetDraftTokens();
    // {generated_tokens.size(), generated_tokens, draft_tokens.size(), draft_tokens}
    std::vector<int> send_tokens(MAX_TRANSFER_TOKENS, -1);
    send_tokens[0] = req->generated_tokens.size();
    std::copy(req->generated_tokens.begin(), req->generated_tokens.end(), send_tokens.begin() + 1);
    if (draft_tokens.size() > (MAX_TRANSFER_TOKENS - req->generated_tokens.size() - 2)) {
      KLLM_LOG_ERROR << "Out of token transfer memory: draft_tokens size: " << draft_tokens.size() << " > "
                     << MAX_TRANSFER_TOKENS - req->generated_tokens.size() - 2;
      KLLM_THROW("Out of token transfer memory");
    }
    send_tokens[req->generated_tokens.size() + 1] = draft_tokens.size();
    if (draft_tokens.size() > 0) {
      std::copy(draft_tokens.begin(), draft_tokens.end(), send_tokens.begin() + 2 + req->generated_tokens.size());
    }
    KLLM_LOG_DEBUG << "TransferGeneratedToken req " << req->req_id << " send tokens: " << Vector2Str(send_tokens);
    reqs_transfer_tokens.emplace_back(req->kv_comm_group_key, req->kv_comm_request_id, send_tokens);
  }
  transfer_engine->Send(reqs_transfer_tokens);
}

Status LlmRuntime::Step(ScheduleOutput* schedule_output,
                        std::map<ModelInstance*, std::vector<ForwardRequest*>>& grouped_reqs,
                        std::vector<SamplingRequest*>& sampling_reqs, bool epilogue) {
  if (context_->IsChief()) {
    return StepOnChief(schedule_output, grouped_reqs, sampling_reqs, epilogue);
  }
  return StepOnWorker(schedule_output, grouped_reqs, sampling_reqs, epilogue);
}

Status LlmRuntime::StepOnChief(ScheduleOutput* schedule_output,
                               std::map<ModelInstance*, std::vector<ForwardRequest*>>& grouped_reqs,
                               std::vector<SamplingRequest*>& sampling_reqs, bool epilogue) {
  KLLM_LOG_MAIN << "Enter llm runtime StepOnChief. multi_batch_id=" << schedule_output->multi_batch_id
                << ", epilogue=" << epilogue;
  PROFILE_EVENT_SCOPE(StepOnChief_, fmt::format("StepOnChief_{}_{}", schedule_output->multi_batch_id, epilogue));

  std::shared_ptr<ModelInstance> model_instance = schedule_output->running_reqs[0]->model_instance;
  context_->SetIsLastLayer(mtp_step_num_ == 0);
  if (!epilogue) {
    // Alloc resources before forwarding
    model_instance->AllocResources(schedule_output->multi_batch_id);
  }
  // Inference forward.
  if (epilogue && !context_->IsStandalone()) {
    multi_batch_controller_->NotifyAnotherBatchCanRun(schedule_output->multi_batch_id);
    KLLM_LOG_MAIN << "wait to recv cur_multi_batch_id=" << schedule_output->multi_batch_id;
    multi_batch_controller_->WaitUtilCanRecvCurrentHiddenUnits(schedule_output->multi_batch_id);
    SetHiddenUnitMeta(schedule_output->multi_batch_id, schedule_output, model_instance);

    RecvHiddenUnits(schedule_output->multi_batch_id);
    KLLM_LOG_MAIN << "try to run multi_batch_id=" << schedule_output->multi_batch_id << " again, epilogue=true";
    multi_batch_controller_->WaitUntilCurrentBatchCanRun(schedule_output->multi_batch_id);
  }

  auto mtp_prepare_wg = PrepareMtpInfoAsync(schedule_output->multi_batch_id, schedule_output->running_reqs);
  Forward(schedule_output->multi_batch_id, grouped_reqs, epilogue, RunMode::kMain);

  // Sampling only in standalone mode or epilogue=true in distributed mode
  if (context_->IsStandalone() || epilogue) {
    PROFILE_EVENT_SCOPE(SamplingAndMTP_, fmt::format("SamplingAndMTP_{}", schedule_output->multi_batch_id));
    Sampling(schedule_output->multi_batch_id, schedule_output->running_reqs, sampling_reqs);
    generation_controller_->UpdateGenerationState(schedule_output->running_reqs);
    mtp_prepare_wg->Wait();
    MtpForward(schedule_output->multi_batch_id, grouped_reqs, sampling_reqs, schedule_output->running_reqs, epilogue);
    GenerateDraftToken(schedule_output->running_reqs);
    TransferGeneratedToken(schedule_output->running_reqs);

    // Forwarding finished, free resources.
    model_instance->FreeResources(schedule_output->multi_batch_id);
    // Note(TJ): donot need NotifyAnotherBatchCanRun, because maybe this batch will enter again.
    multi_batch_controller_->NotifyCurrentBatchIsFinish(schedule_output->multi_batch_id);
    KLLM_LOG_MAIN << "finish multi_batch_id=" << schedule_output->multi_batch_id << ", epilogue=" << epilogue;
  }

  KLLM_LOG_DEBUG << "Leave llm runtime StepOnChief. multi_batch_id=" << schedule_output->multi_batch_id
                 << ", epilogue=" << epilogue;
  return Status();
}

Status LlmRuntime::StepOnWorker(ScheduleOutput* schedule_output,
                                std::map<ModelInstance*, std::vector<ForwardRequest*>>& grouped_reqs,
                                std::vector<SamplingRequest*>& sampling_reqs, bool epilogue) {
  KLLM_LOG_DEBUG << "llm runtime StepOnWorker invoked multi_batch_id=" << schedule_output->multi_batch_id;
  PROFILE_EVENT_SCOPE(StepOnWorker_, fmt::format("StepOnWorker_{}_{}", schedule_output->multi_batch_id, epilogue));
  // Worker always pass result to next step
  KLLM_CHECK(epilogue == false);

  ReorderInferRequests(schedule_output->worker_running_reqs);

  for (size_t dp_swapout_req_block_ids_idx = 0;
       dp_swapout_req_block_ids_idx < schedule_output->swapout_req_block_ids.size(); ++dp_swapout_req_block_ids_idx) {
    std::stringstream so_ss;
    if (!schedule_output->swapout_req_block_ids[dp_swapout_req_block_ids_idx].empty()) {
      for (auto it = schedule_output->swapout_req_block_ids[dp_swapout_req_block_ids_idx].begin();
           it != schedule_output->swapout_req_block_ids[dp_swapout_req_block_ids_idx].end(); ++it) {
        so_ss << it->first << ", ";
        Status status =
            cache_managers_[dp_swapout_req_block_ids_idx]->SwapoutRequestMemoryBlockAsync(it->first, it->second);
        if (!status.OK()) {
          return status;
        }
      }
      KLLM_LOG_DEBUG << "multi_batch_id=" << schedule_output->multi_batch_id
                     << ", dp_idx=" << dp_swapout_req_block_ids_idx << ", SwapoutRequestMemoryBlockAsync req_ids=("
                     << so_ss.str() << ")";
    }
  }

  for (size_t dp_merged_swapout_req_ids_idx = 0;
       dp_merged_swapout_req_ids_idx < schedule_output->merged_swapout_req_ids.size();
       ++dp_merged_swapout_req_ids_idx) {
    auto& dp_merged_swapout_req_ids = schedule_output->merged_swapout_req_ids[dp_merged_swapout_req_ids_idx];
    if (dp_merged_swapout_req_ids.empty()) {
      continue;
    }
    KLLM_LOG_DEBUG << "multi_batch_id=" << schedule_output->multi_batch_id
                   << ", WaitSwapoutRequestMemoryBlock dp_idx=" << dp_merged_swapout_req_ids_idx
                   << ", req num=" << dp_merged_swapout_req_ids.size()
                   << ", ids=" << Vector2Str(dp_merged_swapout_req_ids);
    cache_managers_[dp_merged_swapout_req_ids_idx]->WaitSwapoutRequestMemoryBlock(dp_merged_swapout_req_ids);
  }

  for (size_t dp_swapin_req_block_ids_idx = 0;
       dp_swapin_req_block_ids_idx < schedule_output->swapin_req_block_ids.size(); ++dp_swapin_req_block_ids_idx) {
    std::stringstream si_ss;
    if (!schedule_output->swapin_req_block_ids[dp_swapin_req_block_ids_idx].empty()) {
      for (auto it = schedule_output->swapin_req_block_ids[dp_swapin_req_block_ids_idx].begin();
           it != schedule_output->swapin_req_block_ids[dp_swapin_req_block_ids_idx].end(); ++it) {
        si_ss << it->first << ", ";

        Status status =
            cache_managers_[dp_swapin_req_block_ids_idx]->SwapinRequestMemoryBlockAsync(it->first, it->second);
        if (!status.OK()) {
          return status;
        }
      }
      KLLM_LOG_DEBUG << "multi_batch_id=" << schedule_output->multi_batch_id
                     << ", dp_idx=" << dp_swapin_req_block_ids_idx << ", SwapinRequestMemoryBlockAsync req_ids=("
                     << si_ss.str() << ")";
    }
  }

  for (size_t dp_merged_swapin_req_ids_idx = 0;
       dp_merged_swapin_req_ids_idx < schedule_output->merged_swapin_req_ids.size(); ++dp_merged_swapin_req_ids_idx) {
    auto& dp_merged_swapin_req_ids = schedule_output->merged_swapin_req_ids[dp_merged_swapin_req_ids_idx];
    if (dp_merged_swapin_req_ids.empty()) {
      continue;
    }
    KLLM_LOG_DEBUG << "multi_batch_id=" << schedule_output->multi_batch_id
                   << ", WaitSwapinRequestMemoryBlock dp_idx=" << dp_merged_swapin_req_ids_idx
                   << ", req num=" << dp_merged_swapin_req_ids.size()
                   << ", ids=" << Vector2Str(dp_merged_swapin_req_ids);
    cache_managers_[dp_merged_swapin_req_ids_idx]->WaitSwapinRequestMemoryBlock(dp_merged_swapin_req_ids);
  }

  std::shared_ptr<ModelInstance> model_instance = schedule_output->worker_running_reqs[0]->model_instance;

  // Inference forward.
  model_instance->AllocResources(schedule_output->multi_batch_id);

  SetHiddenUnitMeta(schedule_output->multi_batch_id, schedule_output->worker_running_reqs, model_instance);
  RecvHiddenUnits(schedule_output->multi_batch_id);
  Forward(schedule_output->multi_batch_id, grouped_reqs, epilogue, RunMode::kMain);
  model_instance->FreeResources(schedule_output->multi_batch_id);
  return Status();
}

}  // namespace ksana_llm
