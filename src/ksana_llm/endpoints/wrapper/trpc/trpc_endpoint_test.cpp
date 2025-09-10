/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <pybind11/embed.h>

#include <filesystem>
#include <memory>
#include <string>

#include "ksana_llm/endpoints/endpoint_factory.h"
#include "ksana_llm/endpoints/wrapper/trpc/trpc_endpoint.h"
#include "ksana_llm/service/inference_engine.h"
#include "ksana_llm/service/inference_server.h"
#include "ksana_llm/utils/request_serial.h"
#include "msgpack.hpp"
#include "tests/test.h"

namespace py = pybind11;

namespace ksana_llm {

class TrpcEndpointTest : public testing::Test {
 protected:
  void SetUp() override {
    // Initialize the python interpreter.
    py::initialize_interpreter();

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    // Initialize and start the inference server.
    inference_server_ = std::make_shared<InferenceServer>(config_path, endpoint_config_);
    inference_server_->Start();
  }

  void TearDown() override {
    // Stop the inference server.
    inference_server_->Stop();

    // Delete the python interpreter.
    py::finalize_interpreter();
  }

 private:
  // The endpoint configuration.
  EndpointConfig endpoint_config_;

  // The inference server.
  std::shared_ptr<InferenceServer> inference_server_;
};

TEST_F(TrpcEndpointTest, ForwardHandlerTest) {
  py::gil_scoped_release release;
  auto forward_handler = std::make_shared<ForwardHandler>(endpoint_config_, inference_server_->local_endpoint_);

  // Create the server context.
  trpc::STransportReqMsg req_msg;
  req_msg.basic_info = trpc::object_pool::MakeLwShared<trpc::BasicInfo>();
  req_msg.basic_info->addr.ip = "127.0.0.1";
  req_msg.basic_info->addr.port = 12345;
  trpc::ServerContextPtr ctx = trpc::MakeRefCounted<trpc::ServerContext>(req_msg);

  // Test case 1. Construct a normal forward request.
  trpc::http::HttpRequestPtr req = std::make_shared<trpc::http::HttpRequest>();
  req->SetMethodType(trpc::http::POST);
  req->SetUrl("/forward");
  req->SetVersion("1.1");
  req->SetHeader("Content-Type", "application/x-msgpack");
  msgpack::sbuffer sbuf;
  BatchRequestSerial batch_req;
  RequestSerial request;
  request.input_tokens = {1, 1, 1};
  request.request_target.push_back(
      TargetRequestSerial{"logits", std::vector<int>{},
      std::vector<int>{0, 1}, std::vector<std::pair<int, int>>{}, "GATHER_TOKEN_ID"});
  batch_req.requests.push_back(std::move(request));
  msgpack::pack(sbuf, batch_req);
  std::string request_bytes(sbuf.data(), sbuf.size());
  req->SetContent(request_bytes);
  req->SetHeader("Content-Length", std::to_string(request_bytes.size()));

  // Post the request and get the response.
  trpc::http::HttpResponse rsp;
  forward_handler->Post(ctx, req, &rsp);

  // Verify the forward response.
  EXPECT_EQ(rsp.GetHeaderValues("Content-Type")[0], "application/x-msgpack");
  EXPECT_EQ(rsp.GetStatus(), trpc::http::HttpResponse::StatusCode::kOk);
  std::string response_bytes = rsp.GetContent();
  auto handle = msgpack::unpack(response_bytes.data(), response_bytes.size());
  auto object = handle.get();
  auto batch_rsp = object.as<BatchResponseSerial>();
  EXPECT_EQ(batch_rsp.responses.size(), 1ul);
  EXPECT_TRUE(batch_rsp.message.empty());
  EXPECT_EQ(batch_rsp.code, 0);

  // Test case 2. Construct a bad forward request.
  batch_req.requests[0].request_target[0].slice_pos.push_back(std::make_pair(0, 2));
  sbuf.clear();
  msgpack::pack(sbuf, batch_req);
  request_bytes.assign(sbuf.data(), sbuf.size());
  req->SetContent(request_bytes);
  req->SetHeader("Content-Length", std::to_string(request_bytes.size()));

  // Post the request and get the response.
  forward_handler->Post(ctx, req, &rsp);

  // Verify the forward response.
  EXPECT_EQ(rsp.GetHeaderValues("Content-Type")[0], "application/x-msgpack");
  EXPECT_EQ(rsp.GetStatus(), trpc::http::HttpResponse::StatusCode::kInternalServerError);
  response_bytes = rsp.GetContent();
  handle = msgpack::unpack(response_bytes.data(), response_bytes.size());
  object = handle.get();
  batch_rsp = object.as<BatchResponseSerial>();
  EXPECT_EQ(batch_rsp.responses.size(), 0ul);
  EXPECT_NE(batch_rsp.message.find(
                "Get the last position is not supported for logits in the 'GATHER_TOKEN_ID' token reduction mode."),
            std::string::npos);
  EXPECT_EQ(batch_rsp.code, /* RET_INVALID_ARGUMENT */ 1);
}

}  // namespace ksana_llm
