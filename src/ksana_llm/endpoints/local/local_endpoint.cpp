/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/local/local_endpoint.h"

#include "ksana_llm/profiler/reporter.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

LocalEndpoint::LocalEndpoint(const EndpointConfig &endpoint_config,
                             Channel<std::pair<Status, std::shared_ptr<Request>>> &request_queue)
    : endpoint_config_(endpoint_config), request_queue_(request_queue) {}

Status LocalEndpoint::Handle(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                             const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                             ksana_llm::KsanaPythonOutput &ksana_python_output) {
  auto req = std::make_shared<Request>(ksana_python_input, req_ctx);
  req->waiter = std::make_shared<Waiter>(1);
  if (!ksana_python_input->sampling_config.stop_strings.empty()) {
    req->has_stop_strings = true;
  }
  Status status = Status();
  std::shared_ptr<Waiter> waiter = req->waiter;
  request_queue_.Write(std::pair<Status, std::shared_ptr<Request>>(status, req));

  // Get inference result
  KLLM_LOG_DEBUG << "LocalEndpoint::Handle start Wait.";
  waiter->Wait();
  KLLM_LOG_DEBUG << "LocalEndpoint::Handle Wait finished.";

  ksana_python_output = KsanaPythonOutput(req);
  STATUS_CHECK_AND_REPORT(req->finish_status);
}

Status LocalEndpoint::HandleStreaming(const std::shared_ptr<KsanaPythonInput> &ksana_python_input,
                                      const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                      std::shared_ptr<StreamingIterator> &streaming_iterator) {
  auto req = std::make_shared<Request>(ksana_python_input, req_ctx);
  req->step_waiter = std::make_shared<Waiter>(1);
  req->abort_waiter = std::make_shared<Waiter>(1);
  if (!ksana_python_input->sampling_config.stop_strings.empty()) {
    req->has_stop_strings = true;
  }
  streaming_iterator = std::make_shared<StreamingIterator>(req, ksana_python_input);

  Status status = Status();
  request_queue_.Write(std::pair<Status, std::shared_ptr<Request>>(status, req));
  STATUS_CHECK_AND_REPORT(status);
}

Status LocalEndpoint::HandleForward(const std::string &request_bytes,
                                    const std::shared_ptr<std::unordered_map<std::string, std::string>> &req_ctx,
                                    std::string &response_bytes) {
  auto it = req_ctx->find("Content-Type");
  std::optional<std::string> content_type = (it != req_ctx->end()) ? std::make_optional(it->second) : std::nullopt;

  // Unpack requests into ksana_python_input objects.
  std::vector<std::shared_ptr<KsanaPythonInput>> ksana_python_inputs;
  Status status = request_packer_.Unpack(request_bytes, ksana_python_inputs, content_type);
  if (!status.OK()) {
    // Ignore all the inputs since unpack failed.
    ksana_python_inputs.clear();
  }

  // Construct the batch of forward requests.
  const size_t batch_size = ksana_python_inputs.size();
  auto waiter = std::make_shared<Waiter>(batch_size);
  std::vector<std::pair<Status, std::shared_ptr<Request>>> reqs;
  reqs.reserve(batch_size);

  for (size_t i = 0; i < batch_size; i++) {
    auto req = std::make_shared<Request>(ksana_python_inputs[i], req_ctx);
    req->waiter = waiter;
    req->last_in_batch = (i == batch_size - 1);
    reqs.emplace_back(Status(), std::move(req));
  }
  // Write the batch of forward requests once
  request_queue_.Write(reqs.data(), batch_size);

  // Get inference result
  waiter->Wait();

  std::vector<KsanaPythonOutput> ksana_python_outputs;
  ksana_python_outputs.reserve(batch_size);
  for (const auto &[finish_status, req] : reqs) {
    // If any status fails, `status` is set to fail.
    if (!finish_status.OK()) {
      status = finish_status;
    }
    if (!req->finish_status.OK()) {
      status = req->finish_status;
    }
    ksana_python_outputs.emplace_back(req);
  }

  // Pack ksana_python_output objects into response.
  request_packer_.Pack(ksana_python_inputs, ksana_python_outputs, status, response_bytes, content_type);

  return status;
}

}  // namespace ksana_llm
