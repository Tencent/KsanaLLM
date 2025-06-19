/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/request.h"
#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/utils/finite_state_machine.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

Status SamplingConfig::VerifyArgs() {
  if (topp == 0.f) {
    topp = 1.f;
  }
  if (temperature == 0.f) {
    temperature = 1.f;
  }
  // The TopkSampling kernel of TensorRT requires that `1 <= topk <= 1024`.
  // Refer to
  // https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/samplingTopKKernels.cu#L377
  if (topk < 1 || topk > 1024) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("topk should be between 1 and 1024, but {} was provided", topk));
  }
  if ((encoder_no_repeat_ngram_size > 0) + (decoder_no_repeat_ngram_size > 0) + (no_repeat_ngram_size > 0) > 1) {
    return Status(RET_INVALID_ARGUMENT,
                  "no_repeat_ngram_size/encoder_no_repeat_ngram_size/decoder_no_repeat_ngram_size"
                  " can not be used at the same time");
  }
  return Status();
}

Request::Request(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                 const std::shared_ptr<std::unordered_map<std::string, std::string>>& req_ctx)
    : req_id(0),
      req_ids(),
      model_name(ksana_python_input->model_name),
      input_tokens(ksana_python_input->input_tokens),
      logits_custom_length(0),
      input_refit_embedding(ksana_python_input->input_refit_embedding),
      output_group(std::max(std::max(ksana_python_input->sampling_config.num_beams,
                                     ksana_python_input->sampling_config.num_return_sequences),
                            1)),
      beam_search_group(),
      output_tokens(std::get<0>(output_group[0])),
      logprobs(std::get<1>(output_group[0])),
      sampling_config(ksana_python_input->sampling_config),
      waiter(nullptr),
      step_waiter(nullptr),
      finisheds(output_group.size(), false),
      finished(finisheds[0]),
      finish_status(),
      output_mutex(),
      request_target(ksana_python_input->request_target),
      response(),
      timestamp_in_ms(ProfileTimer::GetCurrentTimeInMs()),
      req_ctx(req_ctx) {
  for (auto& [output, req_logprobs, total_score] : output_group) {
    output = ksana_python_input->input_tokens;
    req_ids.push_back(id_generator_.Gen());
  }
  req_id = req_ids[0];
  auto it = request_target.find("logits");
  if (it != request_target.end()) {
    for (auto [l, r] : it->second.slice_pos) {
      logits_custom_length += (r - l + 1);
    }
  }
  if (!ksana_python_input->structured_output_regex.empty()) {
    std::string& output_structure = ksana_python_input->structured_output_regex;
    std::shared_ptr<FiniteStateMachineController> fsm_controller =
        Singleton<FiniteStateMachineController>::GetInstance();
    req_fsm = fsm_controller->CreateOrGetFSM(output_structure);
  }

  kv_comm_request_id = 0;
  kv_comm_group_key = "";
  if (req_ctx) {
    if (req_ctx->count("kv-comm-request-id")) {
      kv_comm_request_id = std::stoll((*req_ctx)["kv-comm-request-id"]);
    }
    if (req_ctx->count("kv-comm-group-key")) {
      kv_comm_group_key = (*req_ctx)["kv-comm-group-key"];
    }
  }
}

KsanaPythonOutput::KsanaPythonOutput(std::shared_ptr<Request> req) {
  finish_status = req->finish_status;
  input_tokens = req->input_tokens;
  for (const auto& [output, req_logprobs, total_score] : req->output_group) {
    std::vector<int> req_output = {output.begin() + input_tokens.size(), output.end()};
    output_tokens.emplace_back(req_output);
    if (req->sampling_config.logprobs_num > 0) {
      logprobs.emplace_back(req_logprobs);
    }
  }
  response = std::move(req->response);
}

Status KsanaPythonInput::VerifyRequestTarget() {
  /**
   * Verify each target specified by 'request_target' in this KsanaPythonInput object.
   *
   * This function iterates through each target description of a request, checks its validity, and throws a
   * std::runtime_error if:
   *   1. 'target_name' is missing.
   *   2. 'slice_pos' does not represent valid ordered intervals.
   *   3. both 'token_id' and 'slice_pos' are specified for the same target.
   *   4. 'token_reduce_mode" is invalid.
   *   5. GATHER_TOKEN_ID is specified for a transformer or layernorm target.
   *   6. 'token_ids' or the last logits is specified in a logits target.
   */

  // Iterate through each target
  for (auto& [target_name, target_desc] : request_target) {
    // Ensure 'target_name' is specified
    if (target_name.empty()) {
      KLLM_THROW("Missing 'target_name' in target description.");
    }
    // 'target_name' should be 'transformer', 'layernorm' or 'logits'
    if (!std::unordered_set<std::string>{"transformer", "layernorm", "logits"}.count(target_name)) {
      KLLM_THROW(fmt::format("Invalid target name {}.", target_name));
    }

    const int input_tokens_num = static_cast<int>(input_tokens.size());

    // Validate 'slice_pos' is a valid ordered intervals if specified
    if (!target_desc.slice_pos.empty()) {
      int min_required_begin = 0;
      for (auto& [slice_begin, slice_end] : target_desc.slice_pos) {
        // We allow negative indices to count from the end
        slice_begin = slice_begin < 0 ? slice_begin + input_tokens_num : slice_begin;
        slice_end = slice_end < 0 ? slice_end + input_tokens_num : slice_end;
        // Check if the end position is greater than or equal to the begin position
        if (slice_end < slice_begin) {
          KLLM_THROW(fmt::format("Error: The end position of interval [{}, {}] is less than its start position.",
                                 slice_begin, slice_end));
        }
        // Validate that the end position does not exceed the number of input tokens
        if (slice_end >= input_tokens_num) {
          KLLM_THROW(
              fmt::format("Error: The end position of interval [{}, {}] exceeds the total number of input tokens ({}).",
                          slice_begin, slice_end, input_tokens_num));
        }
        // Check for overlap with the previous interval
        if (slice_begin < min_required_begin) {
          KLLM_THROW(fmt::format("Error: Interval [{}, {}] overlaps with the previous interval ending at position {}.",
                                 slice_begin, slice_end, min_required_begin - 1));
        }
        min_required_begin = slice_end + 1;
      }
    }

    // Ensure that 'token_id' and 'slice_pos' are not both set for the same target
    if (!target_desc.token_id.empty() && !target_desc.slice_pos.empty()) {
      KLLM_THROW("Unable to set both token_id and slice_pos at the same time.");
    }

    // Validate the token reduce mode
    if (target_desc.token_reduce_mode == TokenReduceMode::INVALID_TYPE) {
      KLLM_THROW(fmt::format("The specified token reduce mode in {} is invalid.", target_name));
    }
    if (target_desc.token_reduce_mode == TokenReduceMode::GATHER_TOKEN_ID) {
      // Ensure GATHER_TOKEN_ID is not used with transformer, layernorm targets
      if (target_name == "transformer" || target_name == "layernorm") {
        KLLM_THROW(fmt::format("The output of the {} does not support 'GATHER_TOKEN_ID'.", target_name));
      }
    }

    // TODO(zakwang): Enhance support for additional request parameters
    if (target_name == "logits") {
      // Verify that no token IDs are specified, as they are not supported for logits output.
      if (!target_desc.token_id.empty()) {
        KLLM_THROW(fmt::format("Specifying token_id for {} output is not supported.", target_name));
      }
      // Ensure the GATHER_TOKEN_ID reduce mode does not use the last logits, as it might be unsupported or
      // not intended.
      if (target_desc.token_reduce_mode == TokenReduceMode::GATHER_TOKEN_ID && !target_desc.slice_pos.empty() &&
          target_desc.slice_pos.back().second + 1 == input_tokens_num) {
        KLLM_THROW(
            fmt::format("Get the last position is not supported for {} in the 'GATHER_TOKEN_ID' token reduction mode.",
                        target_name));
      }
    }
  }

  return Status();
}

}  // namespace ksana_llm
