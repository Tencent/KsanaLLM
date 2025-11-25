/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include "ksana_llm/utils/request_packer.h"

#include "base64.hpp"
#include "logger.h"
#include "msgpack.hpp"
#include "request_serial.h"
#include "tests/test.h"

#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tokenizer.h"

namespace ksana_llm {

class RequestPackerTest : public testing::Test {
 protected:
  void SetUp() override {
    // Any tokenizer works here.
    Singleton<Tokenizer>::GetInstance()->InitTokenizer("/model/llama-hf/7B");
  }
  void TearDown() override {
    // Destroy the tokenizer.
    Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
  }

  RequestPacker request_packer_;

  // test input
  msgpack::sbuffer sbuf;
  std::vector<std::shared_ptr<KsanaPythonInput>> ksana_python_inputs;
  std::vector<KsanaPythonOutput> ksana_python_outputs;
  BatchRequestSerial batch_request;
  RequestSerial request;
  TargetRequestSerial target;
};

// Test for empty unpacking.
TEST_F(RequestPackerTest, EmptyUnpack) {
  std::string request_bytes;
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find("Request content is empty."),
            std::string::npos);
}

// Test for simple unpacking.
TEST_F(RequestPackerTest, SimpleUnpack) {
  request.prompt = "hello world";
  target.target_name = "logits";
  target.slice_pos.emplace_back(0, 0);  // [0, 0]
  target.token_reduce_mode = "GATHER_TOKEN_ID";
  target.input_top_logprobs_num = 0;
  request.request_target.push_back(target);
  batch_request.requests.push_back(request);

  msgpack::pack(sbuf, batch_request);

  // Parse request
  std::string request_bytes(sbuf.data(), sbuf.size());
  ASSERT_TRUE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack").OK());
  ASSERT_EQ(ksana_python_inputs.size(), 1ul);
  ASSERT_EQ(ksana_python_inputs[0]->sampling_config.max_new_tokens, 1);  // forward interface
  ASSERT_EQ(ksana_python_inputs[0]->request_target.size(), 1);
  ASSERT_TRUE(ksana_python_inputs[0]->request_target.count(target.target_name));
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target.target_name].slice_pos, target.slice_pos);
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target.target_name].token_reduce_mode,
            TokenReduceMode::GATHER_TOKEN_ID);
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target.target_name].input_top_logprobs_num, 0);
}

// Test for complex unpacking: multiple requests and multiple targets.
TEST_F(RequestPackerTest, ComplexUnpack) {
  request.prompt = "Once upon a time";
  target.target_name = "logits";
  target.slice_pos.emplace_back(0, 1);    // [0, 1]
  target.slice_pos.emplace_back(-2, -2);  // [-2, -2]
  target.token_reduce_mode = "GATHER_TOKEN_ID";
  target.input_top_logprobs_num = 0;
  TargetRequestSerial target2;
  target2.target_name = "layernorm";
  target2.slice_pos.emplace_back(2, 3);  // [2, 3]
  target2.token_reduce_mode = "GATHER_ALL";
  target2.input_top_logprobs_num = 0;
  request.request_target.push_back(target);
  request.request_target.push_back(target2);
  batch_request.requests.push_back(request);
  RequestSerial request2 = request;
  request2.prompt.clear();
  request2.input_tokens = std::vector<int>{10, 11, 12, 13, 14};
  request2.request_target[1].target_name = "layernorm";
  request2.request_target[1].slice_pos.clear();
  request2.request_target[1].token_id.push_back(12);
  batch_request.requests.push_back(request2);

  msgpack::pack(sbuf, batch_request);

  // Parse request
  std::string request_bytes(sbuf.data(), sbuf.size());
  ASSERT_TRUE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack").OK());
  ASSERT_EQ(ksana_python_inputs.size(), 2ul);
  for (const auto& ksana_python_input : ksana_python_inputs) {
    ASSERT_EQ(ksana_python_input->sampling_config.max_new_tokens, 1);
    ASSERT_EQ(ksana_python_input->request_target.size(), 2ul);
  }
  // Some random checks
  ASSERT_TRUE(ksana_python_inputs[0]->request_target.count(target.target_name));
  ASSERT_TRUE(ksana_python_inputs[0]->request_target.count(target2.target_name));
  std::vector<std::pair<int, int>> real_slice_pos{{0, 1}, {3, 3}};  // 3 = -2 + 5
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target.target_name].slice_pos, real_slice_pos);
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target2.target_name].token_reduce_mode, TokenReduceMode::GATHER_ALL);
  ASSERT_TRUE(ksana_python_inputs[1]->request_target.count(target.target_name));
  ASSERT_TRUE(ksana_python_inputs[1]->request_target.count(target2.target_name));
  ASSERT_EQ(ksana_python_inputs[1]->request_target[target.target_name].token_id, request2.request_target[0].token_id);
  ASSERT_EQ(ksana_python_inputs[1]->request_target[target.target_name].token_reduce_mode,
            TokenReduceMode::GATHER_TOKEN_ID);
}

struct BatchRequestNewSerial {
  std::string id;
  std::vector<RequestSerial> requests;
  int error_code;

  MSGPACK_DEFINE_MAP(id, requests, error_code);
};

// Test redundant fields for api compatibility, expect to be ignored.
TEST_F(RequestPackerTest, RedundantFileds) {
  target.target_name = "transformer";
  target.token_reduce_mode = "GATHER_ALL";
  request.request_target.push_back(target);
  request.input_tokens = std::vector<int>{7, 6, 1};
  BatchRequestNewSerial batch_request;
  batch_request.requests.push_back(request);
  batch_request.id = "1";
  batch_request.error_code = 200;

  msgpack::pack(sbuf, batch_request);

  // Parse request
  std::string request_bytes(sbuf.data(), sbuf.size());
  ASSERT_TRUE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack").OK());
  ASSERT_EQ(ksana_python_inputs.size(), 1ul);
  ASSERT_EQ(ksana_python_inputs[0]->input_tokens, request.input_tokens);
  ASSERT_EQ(ksana_python_inputs[0]->request_target.size(), 1);
  ASSERT_TRUE(ksana_python_inputs[0]->request_target.count(target.target_name));
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target.target_name].token_reduce_mode, TokenReduceMode::GATHER_ALL);
}

// Test for various input errors.
TEST_F(RequestPackerTest, WrongInput) {
  BatchRequestSerial batch_request;
  RequestSerial& request = batch_request.requests.emplace_back();
  TargetRequestSerial& target = request.request_target.emplace_back();
  std::string request_bytes;

  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find("Missing 'target_name' in target description."),
            std::string::npos);

  target.target_name = "logits";
  target.token_reduce_mode = "GATHER_TOKEN_ID";

  request.input_tokens = std::vector<int>{1};
  target.slice_pos.emplace_back(0, 1);  // [0, 1]
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find("Error: The end position of interval [0, 1] exceeds the total number of input tokens (1)."),
            std::string::npos);

  request.input_tokens = std::vector<int>{1, 2, 3, 4, 5};
  target.slice_pos.emplace_back(1, 2);  // [1, 2]
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find("Error: Interval [1, 2] overlaps with the previous interval ending at position 1."),
            std::string::npos);

  target.slice_pos.back().first = 2;  // [2, 2]
  target.slice_pos.emplace_back(-1, -2);
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find("Error: The end position of interval [4, 3] is less than its start position."),
            std::string::npos);

  target.slice_pos.back().second = -1;
  target.input_top_logprobs_num = -1;
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find("Specifying input_top_logprobs_num for logits output is not supported. it should be >= 0."),
            std::string::npos);
  target.slice_pos.pop_back();

  target.input_top_logprobs_num = 2;
  target.token_reduce_mode = "GATHER_ALL";
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find("Specifying return input_top_logprobs_num > 0 for logits output with GATHER_ALL "
                      "token_reduce_mode is not supported"),
            std::string::npos);

  target.token_reduce_mode = "GATHER_TOKEN_ID";
  target.token_id = std::vector<int>{2};
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find("Unable to set both token_id and slice_pos at the same time."),
            std::string::npos);

  target.slice_pos.clear();
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find("Specifying token_id for logits output is not supported."),
            std::string::npos);

  target.token_reduce_mode = "asdfg";  // an invalid mode
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find(fmt::format("The specified token reduce mode in {} is invalid.", target.target_name)),
            std::string::npos);
  target.token_reduce_mode = "GATHER_TOKEN_ID";

  target.target_name = "layernorm";
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find(fmt::format("The output of the {} does not support 'GATHER_TOKEN_ID'.", target.target_name)),
            std::string::npos);

  target.target_name = "abcd";  // an invalid target
  sbuf.clear();
  msgpack::pack(sbuf, batch_request);
  request_bytes.assign(sbuf.data(), sbuf.size());
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find(fmt::format("Invalid target name {}.", target.target_name)),
            std::string::npos);

  request_bytes = "bad request";
  ASSERT_NE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack")
                .GetMessage()
                .find("Failed to parse the request."),
            std::string::npos);
}

// Test for packing.
TEST_F(RequestPackerTest, NormalPack) {
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  ksana_python_input->input_tokens = {1, 2, 3, 4, 5};
  ksana_python_inputs.push_back(ksana_python_input);
  KsanaPythonOutput ksana_python_output;
  PythonTensor tensor;
  std::vector<float> probs_output = {-1.f, -2.f, .5f, 3.f, .7f};
  tensor.shape = {probs_output.size()};
  tensor.dtype = GetTypeString(TYPE_FP32);
  tensor.data.resize(probs_output.size() * sizeof(float));
  memcpy(tensor.data.data(), probs_output.data(), tensor.data.size());
  ksana_python_output.response["logits"] = tensor;
  ksana_python_outputs.push_back(ksana_python_output);
  Status response_status;

  std::string response_bytes;
  ASSERT_TRUE(
      request_packer_
          .Pack(ksana_python_inputs, ksana_python_outputs, response_status, response_bytes, "application/x-msgpack")
          .OK());

  auto handle = msgpack::unpack(response_bytes.data(), response_bytes.size());
  auto object = handle.get();
  BatchResponseSerial batch_response = object.as<BatchResponseSerial>();
  ASSERT_EQ(batch_response.responses.size(), 1ul);
  ASSERT_EQ(batch_response.responses[0].response.size(), 1ul);
  ASSERT_EQ(batch_response.responses[0].response[0].target_name, "logits");
  ASSERT_EQ(batch_response.responses[0].response[0].tensor.shape[0], probs_output.size());
  auto probs_response_bytes =
      base64::decode_into<std::vector<uint8_t>>(batch_response.responses[0].response[0].tensor.data.begin(),
                                                batch_response.responses[0].response[0].tensor.data.end());
  std::vector<float> probs_response(probs_output.size());
  memcpy(probs_response.data(), probs_response_bytes.data(), probs_response_bytes.size());
  ASSERT_EQ(probs_response, probs_output);
  ASSERT_EQ(batch_response.message, response_status.GetMessage());
  ASSERT_EQ(batch_response.code, response_status.GetCode());
}

// Test for logits with GATHER_ALL mode
TEST_F(RequestPackerTest, LogitsGatherAllTest) {
  // This test verifies that using target.target_name = "logits" with target.token_reduce_mode = "GATHER_ALL"
  // returns a 2D tensor (unlike GATHER_TOKEN_ID which returns a 1D tensor)

  // Step 1: Set up a request with logits target and GATHER_ALL mode
  request.prompt = "test prompt";
  target.target_name = "logits";
  target.slice_pos.emplace_back(0, 0);      // [0, 0]
  target.token_reduce_mode = "GATHER_ALL";  // Using GATHER_ALL mode for logits
  target.input_top_logprobs_num = 0;
  request.request_target.push_back(target);
  batch_request.requests.push_back(request);

  msgpack::pack(sbuf, batch_request);

  // Parse request
  std::string request_bytes(sbuf.data(), sbuf.size());
  ASSERT_TRUE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/x-msgpack").OK());
  ASSERT_EQ(ksana_python_inputs.size(), 1ul);
  ASSERT_EQ(ksana_python_inputs[0]->request_target.size(), 1);
  ASSERT_TRUE(ksana_python_inputs[0]->request_target.count(target.target_name));

  // Verify that the token_reduce_mode is correctly set to GATHER_ALL
  ASSERT_EQ(ksana_python_inputs[0]->request_target[target.target_name].token_reduce_mode, TokenReduceMode::GATHER_ALL);

  // Step 2: Create a KsanaPythonOutput with logits data
  // For GATHER_ALL mode, we need to simulate a 2D tensor with shape [token_count, vocab_size]
  KsanaPythonOutput ksana_python_output;
  PythonTensor tensor;

  // Simulate a vocabulary size of 6
  const size_t vocab_size = 6;
  std::vector<float> logits_data = {0.0, 1.0, 2.0, 1.5, 0.7, 1.8};

  // Shape for GATHER_ALL mode should be [token_count, vocab_size]
  // Here token_count is 1 (we're gathering for one position)
  tensor.shape = {1, vocab_size};
  tensor.dtype = GetTypeString(TYPE_FP32);
  tensor.data.resize(logits_data.size() * sizeof(float));
  memcpy(tensor.data.data(), logits_data.data(), tensor.data.size());
  ksana_python_output.response["logits"] = tensor;
  ksana_python_outputs.push_back(ksana_python_output);
  Status response_status;

  // Step 3: Pack the response
  std::string response_bytes;
  ASSERT_TRUE(
      request_packer_
          .Pack(ksana_python_inputs, ksana_python_outputs, response_status, response_bytes, "application/x-msgpack")
          .OK());

  // Step 4: Unpack and verify the response
  auto handle = msgpack::unpack(response_bytes.data(), response_bytes.size());
  auto object = handle.get();
  BatchResponseSerial batch_response = object.as<BatchResponseSerial>();
  ASSERT_EQ(batch_response.responses.size(), 1ul);
  ASSERT_EQ(batch_response.responses[0].response.size(), 1ul);
  ASSERT_EQ(batch_response.responses[0].response[0].target_name, "logits");

  // Verify the shape of the returned tensor - it should be 2D for GATHER_ALL mode
  ASSERT_EQ(batch_response.responses[0].response[0].tensor.shape.size(), 2ul);
  ASSERT_EQ(batch_response.responses[0].response[0].tensor.shape[0], 1);           // token_count
  ASSERT_EQ(batch_response.responses[0].response[0].tensor.shape[1], vocab_size);  // vocab_size

  // Decode and verify the logits data
  auto logits_response_bytes =
      base64::decode_into<std::vector<uint8_t>>(batch_response.responses[0].response[0].tensor.data.begin(),
                                                batch_response.responses[0].response[0].tensor.data.end());
  std::vector<float> logits_response(logits_data.size());
  memcpy(logits_response.data(), logits_response_bytes.data(), logits_response_bytes.size());

  // Compare the original logits with the returned logits
  for (size_t i = 0; i < logits_data.size(); ++i) {
    ASSERT_NEAR(logits_response[i], logits_data[i], 1e-6);
  }
}

// Test for JSON format forward interface
TEST_F(RequestPackerTest, JsonForward) {
  // JSON unpack test - cover main JSON parsing functionality
  nlohmann::json json_request = {
      {"requests",
       {{{"input_tokens", {1, 2, 3, 4, 5}},
         {"request_target",
          {{{"target_name", "logits"}, {"slice_pos", {{0, 0}}}, {"token_reduce_mode", "GATHER_TOKEN_ID"}}}}}}}};

  std::string request_bytes = json_request.dump();
  ASSERT_TRUE(request_packer_.Unpack(request_bytes, ksana_python_inputs, "application/json").OK());
  ASSERT_EQ(ksana_python_inputs.size(), 1ul);
  ASSERT_EQ(ksana_python_inputs[0]->input_tokens, std::vector<int>({1, 2, 3, 4, 5}));
  ASSERT_EQ(ksana_python_inputs[0]->request_target.size(), 1ul);
  ASSERT_TRUE(ksana_python_inputs[0]->request_target.count("logits"));

  // JSON pack test - cover main JSON generation functionality
  KsanaPythonOutput ksana_python_output;
  PythonTensor tensor;
  std::vector<float> data = {0.1f, 0.2f};
  tensor.shape = {2};
  tensor.dtype = GetTypeString(TYPE_FP32);
  tensor.data.resize(data.size() * sizeof(float));
  memcpy(tensor.data.data(), data.data(), tensor.data.size());
  ksana_python_output.response["logits"] = tensor;
  ksana_python_outputs.push_back(ksana_python_output);

  std::string response_bytes;
  Status response_status;
  ASSERT_TRUE(request_packer_
                  .Pack(ksana_python_inputs, ksana_python_outputs, response_status, response_bytes, "application/json")
                  .OK());

  // Validate JSON structure
  nlohmann::json json_response = nlohmann::json::parse(response_bytes);
  ASSERT_TRUE(json_response.contains("responses"));
  ASSERT_EQ(json_response["responses"].size(), 1ul);
}

}  // namespace ksana_llm
