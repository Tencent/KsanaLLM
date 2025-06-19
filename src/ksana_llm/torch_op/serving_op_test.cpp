/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/torch_op/serving_op.h"
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "ksana_llm/utils/status.h"

namespace py = pybind11;

class ServingOpTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!Py_IsInitialized()) {
      interpreter_guard = new py::scoped_interpreter();
    }
  }

  static void TearDownTestSuite() {
    if (interpreter_guard) {
      delete interpreter_guard;
      interpreter_guard = nullptr;
    }
  }

  void SetUp() override {
    // 创建一个新的 KsanaPythonOutput 实例用于测试
    output = std::make_shared<ksana_llm::KsanaPythonOutput>();
  }

  void TearDown() override { output.reset(); }

  static py::scoped_interpreter* interpreter_guard;
  std::shared_ptr<ksana_llm::KsanaPythonOutput> output;
};

py::scoped_interpreter* ServingOpTest::interpreter_guard = nullptr;

// 基本的 finish_status 测试
TEST_F(ServingOpTest, BasicFinishStatus) {
  ASSERT_TRUE(output != nullptr);

  // 创建一个新的 Status 对象并验证其初始状态
  ksana_llm::Status status(ksana_llm::RET_SUCCESS, "");
  EXPECT_TRUE(status.OK());
  EXPECT_EQ(status.GetCode(), ksana_llm::RET_SUCCESS);
  EXPECT_EQ(status.GetMessage(), "");

  // 设置到 output 并验证
  output->finish_status = status;
  EXPECT_TRUE(output->finish_status.OK());
  EXPECT_EQ(output->finish_status.GetCode(), ksana_llm::RET_SUCCESS);
  EXPECT_EQ(output->finish_status.GetMessage(), "");
}

// 测试不同的错误码
TEST_F(ServingOpTest, DifferentErrorCodes) {
  ASSERT_TRUE(output != nullptr);

  struct TestCase {
    ksana_llm::RetCode code;
    std::string message;
    bool should_be_ok;
  };

  std::vector<TestCase> test_cases = {{ksana_llm::RET_INVALID_ARGUMENT, "Invalid argument case", false},
                                      {ksana_llm::RET_STOP_ITERATION, "Stop iteration case", false}};

  for (const auto& test_case : test_cases) {
    // 直接创建 Status 对象
    ksana_llm::Status status(test_case.code, test_case.message);

    // 验证 Status 对象
    EXPECT_EQ(status.GetCode(), test_case.code);
    EXPECT_EQ(status.GetMessage(), test_case.message);
    EXPECT_EQ(status.OK(), test_case.should_be_ok);

    // 设置到 output 并再次验证
    output->finish_status = status;
    EXPECT_EQ(output->finish_status.GetCode(), test_case.code);
    EXPECT_EQ(output->finish_status.GetMessage(), test_case.message);
    EXPECT_EQ(output->finish_status.OK(), test_case.should_be_ok);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}