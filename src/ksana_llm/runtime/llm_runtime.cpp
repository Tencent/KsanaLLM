/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/llm_runtime.h"

#include <algorithm>
#include <execution>
#include <atomic>
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
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {
LlmRuntime::LlmRuntime(const BatchSchedulerConfig& batch_scheduler_config, const RuntimeConfig& runtime_config,
                       std::shared_ptr<Context> context)
    : context_(context) {
  worker_group_ = std::make_shared<WorkerGroup>(context_->GetTensorParallelSize(),
                                                batch_scheduler_config.max_pp_batch_num, context_);

  for (size_t worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    samplers_.push_back(std::make_shared<Sampler>(batch_scheduler_config, worker_id, context_));
  }
  threadpool_ = std::make_shared<ThreadPool>(2);
  threadpool_->Start();
}

void LlmRuntime::SetCacheManagers(std::vector<std::shared_ptr<CacheManagerInterface>> cache_managers) {
  cache_managers_ = cache_managers;
}

void LlmRuntime::SetMultiBatchController(std::shared_ptr<MultiBatchController> controller) {
  multi_batch_controller_ = controller;
}

void LlmRuntime::SetDraftGenerator(std::shared_ptr<DraftGeneratorInterface> draft_generator) {
  if (draft_generator_ != nullptr) {
    KLLM_LOG_WARNING << "draft_generator already exists, currently only supports one, will replace the previous one";
  }
  draft_generator_ = draft_generator;
}

#ifdef ENABLE_CUDA
// In a CUDA environment, it's necessary to compute perfill and decode together.
// Therefore, kGroupStageMap is required to map to a single stage for synchronized processing.
static std::unordered_map<InferStage, InferStage> kGroupStageMap = {
    {InferStage::STAGE_CONTEXT, InferStage::STATE_DECODE}, {InferStage::STATE_DECODE, InferStage::STATE_DECODE}};
#else
static std::unordered_map<InferStage, InferStage> kGroupStageMap = {
    {InferStage::STAGE_CONTEXT, InferStage::STAGE_CONTEXT}, {InferStage::STATE_DECODE, InferStage::STATE_DECODE}};
#endif

void LlmRuntime::BuildForwardRequests(
    size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs) {
  PROFILE_EVENT_SCOPE(BuildForwardRequests, fmt::format("BuildForwardRequests_{}", multi_batch_id));

  grouped_reqs.clear();
  for (auto& req : reqs) {
    ++req->step;
    ModelInstance* const key = req->model_instance.get();
    auto& group_reqs = grouped_reqs[key][GetGroupStage(req->infer_stage)];
    group_reqs.reserve(reqs.size());
    group_reqs.emplace_back(req->GetForwardRequest(key->GetLogitsPtr(multi_batch_id)));
  }
}

void LlmRuntime::BuildForwardRequests(
    std::vector<std::shared_ptr<WorkerInferRequest>>& reqs,
    std::map<ModelInstance*, std::map<InferStage, std::vector<ForwardRequest*>>>& grouped_reqs) {
  PROFILE_EVENT_SCOPE(BuildForwardRequests, "BuildForwardRequests");
  for (auto& req : reqs) {
    ++req->step;
    auto& group_reqs = grouped_reqs[req->model_instance.get()][GetGroupStage(req->infer_stage)];
    group_reqs.reserve(reqs.size());
    group_reqs.emplace_back(req->GetForwardRequest());
  }
}

void LlmRuntime::DeepCopyAndSyncSamplingRequests(const std::vector<std::shared_ptr<InferRequest>>& running_reqs,
                                                 std::vector<SamplingRequest>& sampling_reqs) {
  // Create a map of InferRequests for quick lookup
  std::unordered_map<size_t, std::shared_ptr<InferRequest>> req_map;
  for (auto& req : running_reqs) {
    req_map[req->req_id] = req;
  }

  size_t logits_offset = 0;
  // Deep copy and synchronize data for each SamplingRequest
  for (auto& sampling_req : sampling_reqs) {
    // Find the corresponding InferRequest
    auto it = req_map.find(sampling_req.req_id);
    if (it == req_map.end()) {
      continue;
    }
    auto& infer_req = it->second;

    // Synchronize data from InferRequest to SamplingRequest
    sampling_req.step = infer_req->step;
    sampling_req.logits_custom_length = infer_req->logits_custom_length;
    sampling_req.sampling_token_num = infer_req->sampling_token_num;
    sampling_req.last_step_token_num = infer_req->last_step_token_num;
    sampling_req.logits_offset = logits_offset;
    logits_offset += infer_req->sampling_token_num;

    // Deep copy forwarding_tokens - this is the most critical part
    if (sampling_req.forwarding_tokens && sampling_req.origin_tokens) {
      // Create a new copy of forwarding_tokens from origin_tokens (the real token data)
      sampling_req.forwarding_tokens = std::make_shared<std::vector<int>>(*sampling_req.origin_tokens);
    }

    // Synchronize other related fields
    sampling_req.sampling_config = &(infer_req->sampling_config);
  }
}

Status LlmRuntime::RunSerially(
    size_t multi_batch_id,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs,
    bool epilogue, RunMode run_mode) {
  PROFILE_EVENT_SCOPE(RunSerially_, fmt::format("RunSerially_{}_{}", multi_batch_id, epilogue));
  Status result_status = Status();
  for (auto& [model_inst, stage_vec_reqs] : grouped_reqs) {
    for (auto& [stage, vec_req] : stage_vec_reqs) {
      std::vector<std::future<Status>> inst_results =
          model_inst->ForwardAsync(multi_batch_id, worker_group_, stage, vec_req, epilogue, run_mode);
      for (auto& worker_result : inst_results) {
        Status status = worker_result.get();
        if (!status.OK()) {
          result_status = status;
        }
      }
    }
  }
  return result_status;
}

Status LlmRuntime::Forward(
    size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs, bool epilogue,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs,
    RunMode run_mode) {
  return AuxForward(multi_batch_id, grouped_reqs, epilogue, run_mode);
}

Status LlmRuntime::Forward(
    size_t multi_batch_id, std::vector<std::shared_ptr<WorkerInferRequest>>& reqs, bool epilogue,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs) {
  return AuxForward(multi_batch_id, grouped_reqs, epilogue);
}

Status LlmRuntime::AuxForward(
    size_t multi_batch_id,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs,
    bool epilogue, RunMode run_mode) {
  PROFILE_EVENT_SCOPE(Forward_, fmt::format("Forward_{}_{}", multi_batch_id, epilogue));
  // context decode and decode run serially in single thread
  if (context_->IsRunContextDecodeAndDecodeSerially()) {
    // Wait all instances done and check status.
    auto ret = RunSerially(multi_batch_id, grouped_reqs, epilogue, run_mode);
    return ret;
  }

  std::vector<std::vector<std::future<Status>>> results;
  for (auto& [model_inst, stage_vec_reqs] : grouped_reqs) {
    for (auto& [stage, vec_req] : stage_vec_reqs) {
      results.push_back(model_inst->ForwardAsync(multi_batch_id, worker_group_, stage, vec_req, epilogue, run_mode));
    }
  }

  // Wait all instances done and check status.
  Status result_status = Status();
  for (auto& inst_results : results) {
    for (auto& worker_result : inst_results) {
      Status status = worker_result.get();
      if (!status.OK()) {
        result_status = status;
      }
    }
  }
  return result_status;
}

void LlmRuntime::BuildSamplingRequest(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
                                      std::vector<SamplingRequest>& sampling_reqs, bool enable_main_layers_sampler) {
  PROFILE_EVENT_SCOPE(BuildSamplingRequest_, fmt::format("BuildSamplingRequest_{}", multi_batch_id));
  sampling_reqs.resize(reqs.size());
  for (size_t i = 0; i < reqs.size(); ++i) {
    auto& req = reqs[i];
    SamplingRequest& sampling_req = sampling_reqs[i];
    sampling_req.req_id = req->req_id;
    sampling_req.step = req->step;
    sampling_req.logits_custom_length = req->logits_custom_length;
    sampling_req.input_tokens = std::shared_ptr<std::vector<int>>(req, &req->input_tokens);
    sampling_req.forwarding_tokens = std::shared_ptr<std::vector<int>>(req, &req->forwarding_tokens);
    sampling_req.origin_tokens = &(req->forwarding_tokens);
    sampling_req.sampling_token_num = req->sampling_token_num;
    sampling_req.last_step_token_num = req->last_step_token_num;
    sampling_req.sampling_result_tokens = &(req->sampling_result_tokens);
    sampling_req.sampling_result_tokens->clear();
    sampling_req.response = &(req_ptr->response);
    sampling_req.request_target =
        std::make_shared<const std::map<std::string, TargetDescribe>>(req_ptr->request_target);
    sampling_req.logprobs =
        std::shared_ptr<std::vector<std::vector<std::pair<int, float>>>>(req_ptr, &req_ptr->logprobs);
    sampling_req.logits_offset = req_ptr->logits_offset;
    sampling_req.logits_buf = req_ptr->model_instance->GetLogitsPtr(multi_batch_id);
    sampling_req.sampling_config = &(req_ptr->sampling_config);
    sampling_req.req_group = &(req_ptr->req_group);
    sampling_req.req_ctx = req_ptr->req_ctx;
    if (sampling_req.sampling_config->num_beams > 1) {
      sampling_req.sampling_config->logprobs_num =
          std::max(sampling_req.sampling_config->logprobs_num, sampling_req.sampling_config->num_beams);
      sampling_req.sampling_config->topk =
          std::max(sampling_req.sampling_config->topk, sampling_req.sampling_config->num_beams);
    }
    sampling_req.ngram_dict = &(req->ngram_dict);
    sampling_req.structured_generator = req->structured_generator;
    sampling_req.apply_structured_constraint = enable_main_layers_sampler;
    sampling_req.apply_no_repeat_ngram_constraint = enable_main_layers_sampler;
  }
}

Status LlmRuntime::Sampling(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
                            std::vector<SamplingRequest>& sampling_reqs, bool enable_main_layers_sampler) {
  PROFILE_EVENT_SCOPE(Sampling, fmt::format("Sampling_{}", multi_batch_id));

  std::vector<std::future<Status>> results;
  for (size_t worker_id = 0; worker_id < context_->GetTensorParallelSize(); ++worker_id) {
    results.push_back(
        worker_group_->GetWorker(worker_id)->SamplingAsync(multi_batch_id, samplers_[worker_id], sampling_reqs));
  }

  // Wait all instances done and check status.
  Status result_status = Status();
  for (auto& result : results) {
    try {
      Status status = result.get();
      if (!status.OK()) {
        result_status = status;
      }
    } catch (const std::exception& e) {
      KLLM_LOG_FATAL << "Exception in sampling, info: " << e.what();
      result_status = Status(RET_RUNTIME_FAILED, "Failed to sampling.");
    }
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

  return result_status;
}

Status LlmRuntime::MTPForward(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& reqs,
                              const bool epilogue,
                              std::map<ModelInstance*, std::map<InferStage, std::vector<ForwardRequest*>>> grouped_reqs,
                              std::vector<SamplingRequest>& sampling_reqs) {
  if (mtp_step_num_ == 0 || !context_->IsChief()) {
    return Status();
  }

  // select mtp req
  constexpr size_t kCompressKvMaxMtpLen = 1024;

  // TODO(lijiajieli): remove copy and do not modify InferRequest
  std::vector<std::shared_ptr<InferRequest>> mtp_reqs = reqs;

  for (const auto& req : mtp_reqs) {
    req->draft_tokens.mtp.clear();
    req->forwarding_tokens = req->GetVerifiedTokens();
    req->forwarding_tokens_draft_num = req->accepted_tokens.size();  // already removed wrong token
    req->sampling_token_num = kStepGenerateTokenNum;
    req->last_step_token_num = kStepGenerateTokenNum;
  }

  ReorderInferRequests(mtp_reqs);

  std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>> grouped_req;
  std::vector<SamplingRequest> sampling_req;
  BuildForwardRequests(multi_batch_id, mtp_reqs, grouped_req);
  BuildSamplingRequest(multi_batch_id, mtp_reqs, sampling_req);

    context_->SetIsLastLayer(mtp_step + 1 == mtp_step_num_);
    Forward(multi_batch_id, mtp_reqs, epilogue, sync_grouped_req, RunMode::kNextN);

    build_sampling_future.get();
    Sampling(multi_batch_id, mtp_reqs, sync_sampling_req, false);

    for (const auto& req : mtp_reqs) {
      req->draft_tokens.mtp.insert(req->draft_tokens.mtp.end(), req->sampling_result_tokens.begin(),
                                   req->sampling_result_tokens.end());
    }
  }

  // reset requests
  for (const auto& req : mtp_reqs) {
    req->forwarding_tokens.resize(req->forwarding_tokens.size() - mtp_step_num_);
    req->forwarding_tokens.insert(req->forwarding_tokens.begin(), first_tokens[req->req_id].begin(),
                                  first_tokens[req->req_id].end());
  }
  return Status();
}

void LlmRuntime::GenerateDraftToken(std::vector<std::shared_ptr<InferRequest>>& reqs) {
  if (draft_generator_ == nullptr) {
    return;
  }
  PROFILE_EVENT_SCOPE(GenerateDraftToken_, fmt::format("GenerateDraftToken"));
  for (auto& req : reqs) {
    std::vector<int> tokens;
    tokens.reserve(req->forwarding_tokens.size() - req->forwarding_tokens_draft_num + req->accepted_tokens.size() +
                   kStepGenerateTokenNum + req->draft_tokens.mtp.size());
    tokens.insert(tokens.end(), req->forwarding_tokens.begin(),
                  req->forwarding_tokens.end() - req->forwarding_tokens_draft_num);
    tokens.insert(tokens.end(), req->accepted_tokens.begin(), req->accepted_tokens.end());
    tokens.emplace_back(req->generated_token);
    tokens.insert(tokens.end(), req->draft_tokens.mtp.begin(), req->draft_tokens.mtp.end());
    draft_generator_->GenerateDraft(tokens, req->step, req->suggested_draft_num, req->draft_tokens.trie,
                                    req->draft_tokens.mtp.size(), req->accepted_tokens.size(), req->req_id);
  }
}

void LlmRuntime::TransferGeneratedToken(std::vector<std::shared_ptr<InferRequest>>& reqs,
                                        std::shared_ptr<TransferEngine> transfer_engine) {
  std::vector<std::tuple<std::string, int, std::vector<int>>> reqs_transfer_tokens;
  for (auto& req : reqs) {
    std::vector<int> draft_tokens = req->draft_tokens.GetDraftTokens();
    // TODO(winminkong): In the future, MTP will be improved to generate multiple draft tokens or gen tokens, and the
    // transmission method will be modified subsequently.
    std::vector<int> send_tokens(MAX_TRANSFER_TOKENS, -1);
    send_tokens[0] = req->generated_token;
    if (draft_tokens.size() > (MAX_TRANSFER_TOKENS - 1)) {
      KLLM_LOG_ERROR << "Out of token transfer memory: draft_tokens size: " << draft_tokens.size() << " > "
                     << MAX_TRANSFER_TOKENS - 1;
      KLLM_THROW("Out of token transfer memory");
    }
    std::copy(draft_tokens.begin(), draft_tokens.end(), send_tokens.begin() + 1);
    KLLM_LOG_DEBUG << "TranferGeneratedToken req " << req->req_id << " send tokens: " << Vector2Str(send_tokens);
    reqs_transfer_tokens.emplace_back(req->kv_comm_group_key, req->kv_comm_request_id, send_tokens);
  }
  transfer_engine->Send(reqs_transfer_tokens);
}

Status LlmRuntime::Step(
    ScheduleOutput* schedule_output,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs,
    std::vector<SamplingRequest>& sampling_reqs, bool epilogue) {
  if (context_->IsChief()) {
    return StepOnChief(schedule_output, grouped_reqs, sampling_reqs, epilogue);
  }
  return StepOnWorker(schedule_output, grouped_reqs, sampling_reqs, epilogue);
}

Status LlmRuntime::StepOnChief(
    ScheduleOutput* schedule_output,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs,
    std::vector<SamplingRequest>& sampling_reqs, bool epilogue) {
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
  time_t start_time_ms = ProfileTimer::GetCurrentTimeInMs();
  if (epilogue && !context_->IsStandalone()) {
    multi_batch_controller_->NotifyAnotherBatchCanRun(schedule_output->multi_batch_id);
    KLLM_LOG_MAIN << "wait to recv cur_multi_batch_id=" << schedule_output->multi_batch_id;
    multi_batch_controller_->WaitUtilCanRecvCurrentHiddenUnits(schedule_output->multi_batch_id);
    SetHiddenUnitMeta(schedule_output->multi_batch_id, schedule_output, model_instance);

    RecvHiddenUnits(schedule_output->multi_batch_id);
    KLLM_LOG_MAIN << "try to run multi_batch_id=" << schedule_output->multi_batch_id << " again, epilogue=true";
    multi_batch_controller_->WaitUntilCurrentBatchCanRun(schedule_output->multi_batch_id);
  }
  Forward(schedule_output->multi_batch_id, schedule_output->running_reqs, epilogue, grouped_reqs);
  time_t end_time_ms = ProfileTimer::GetCurrentTimeInMs();
  KLLM_LOG_MAIN << "LlmRuntime Forward multi_batch_id=" << schedule_output->multi_batch_id << ", epilogue=" << epilogue
                << ", time cost=" << end_time_ms - start_time_ms << "ms";

  // Sampling only in standalone mode or epilogue=true in distributed mode
  if (context_->IsStandalone() || epilogue) {
    PROFILE_EVENT_SCOPE(SamplingAndMTP_, fmt::format("SamplingAndMTP_{}", schedule_output->multi_batch_id));
    Sampling(schedule_output->multi_batch_id, schedule_output->running_reqs, sampling_reqs);
    generation_controller_->UpdateGenerationState(schedule_output->running_reqs);
    MTPForward(schedule_output->multi_batch_id, schedule_output->running_reqs, epilogue, grouped_reqs, sampling_reqs);
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

Status LlmRuntime::StepOnWorker(
    ScheduleOutput* schedule_output,
    std::unordered_map<ModelInstance*, std::unordered_map<InferStage, std::vector<ForwardRequest>>>& grouped_reqs,
    std::vector<SamplingRequest>& sampling_reqs, bool epilogue) {
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
  Forward(schedule_output->multi_batch_id, schedule_output->worker_running_reqs, epilogue, grouped_reqs);
  model_instance->FreeResources(schedule_output->multi_batch_id);
  return Status();
}

template void LlmRuntime::ReorderInferRequests<InferRequest>(std::vector<std::shared_ptr<InferRequest>>& reqs);
template void LlmRuntime::ReorderInferRequests<WorkerInferRequest>(
    std::vector<std::shared_ptr<WorkerInferRequest>>& reqs);

}  // namespace ksana_llm
