/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/request_serial.h"
#include "ksana_llm/utils/status.h"

namespace py = pybind11;

namespace ksana_llm {

// RequestPacker is responsible for packing and unpacking requests and responses serialized in msgpack format
// bytes into corresponding KsanaPythonInput and KsanaPythonOutput objects.
class RequestPacker {
 public:
  // Unpack a serialized request into KsanaPythonInput objects.
  Status Unpack(const std::string& request_bytes, std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                const std::optional<std::string>& content_type);

  // Pack KsanaPythonOutput objects into a serialized response.
  Status Pack(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
              const std::vector<KsanaPythonOutput>& ksana_python_outputs, const Status& response_status,
              std::string& response_bytes, const std::optional<std::string>& content_type);

 private:
  Status UnpackMsgpack(const std::string& request_bytes,
                        std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs);
  Status UnpackJson(const std::string& request_bytes,
                     std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs);
  Status PackMsgpack(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                      const std::vector<KsanaPythonOutput>& ksana_python_outputs, const Status& response_status,
                      std::string& response_bytes);
  Status PackJson(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                   const std::vector<KsanaPythonOutput>& ksana_python_outputs, const Status& response_status,
                   std::string& response_bytes);

  // Helper function to construct a KsanaPythonInput object from a RequestSerial object
  KsanaPythonInput GetKsanaPythonInput(const RequestSerial& req);

  // Helper function to construct a ResponseSerial object from KsanaPythonInput and KsanaPythonOutput objects
  ResponseSerial GetResponseSerial(const std::shared_ptr<KsanaPythonInput>& ksana_python_input,
                                   const KsanaPythonOutput& ksana_python_output);

  // Helper function to construct BatchResponseSerial from batch inputs and outputs
  void GetResponseSerials(const std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs,
                          const std::vector<KsanaPythonOutput>& ksana_python_outputs, const Status& response_status,
                          BatchResponseSerial& batch_rsp);

  // Helper function to convert batch request to KsanaPythonInputs
  void GetKsanaPythonInputs(const BatchRequestSerial& batch_req,
                            std::vector<std::shared_ptr<KsanaPythonInput>>& ksana_python_inputs);
};

}  // namespace ksana_llm
