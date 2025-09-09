/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include "tests/test.h"

namespace ksana_llm {

// 期望的CSV数据结构
struct ExpectedCsvData {
  int profile_round;
  int layer_forward_round;
  int total_requests;
  int prefill_requests;
  int decode_requests;
  int total_sequence_len;
  double prefill_avg_seq_len;
  double decode_avg_seq_len;
  double prefill_avg_forwarding_tokens;
  double decode_avg_forwarding_tokens;
};

class RunModelPerformanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 获取当前文件路径，用于构建相对路径
    std::filesystem::path current_path = __FILE__;
    parent_path_ = current_path.parent_path();

    // 初始化共同的路径
    perf_config_path_ = parent_path_ / "test_config.json";
    runtime_config_path_ = parent_path_ / "../../../examples/llama7b/ksana_llm_tp.yaml";
    cmd_path_ = parent_path_ / "../../../build/bin/run_model_performance";

    // 验证配置文件存在
    ASSERT_TRUE(std::filesystem::exists(perf_config_path_))
        << "Performance config file not found: " << perf_config_path_;
    ASSERT_TRUE(std::filesystem::exists(runtime_config_path_))
        << "Runtime config file not found: " << runtime_config_path_;
    ASSERT_TRUE(std::filesystem::exists(cmd_path_)) << "Executable not found: " << cmd_path_;
  }

  // 执行命令并返回退出码
  int ExecuteCommand(const std::string& command) { return std::system(command.c_str()); }

  // 验证CSV文件内容，支持多行输出
  void ValidateCsvContent(const std::string& csv_file_path, const std::vector<ExpectedCsvData>& expected_data) {
    std::ifstream file(csv_file_path);
    ASSERT_TRUE(file.is_open()) << "Failed to open CSV file: " << csv_file_path;

    std::string line;
    // Read header
    ASSERT_TRUE(std::getline(file, line)) << "Failed to read CSV header";
    // Check if header contains all expected fields (with space before decode fields for grouping)
    std::string expected_fields[] = {"profile_round",
                                     "layer_forward_round",
                                     "total_requests",
                                     "total_sequence_len",
                                     "prefill_requests",
                                     "prefill_avg_seq_len",
                                     "prefill_avg_forwarding_tokens",
                                     "decode_requests",
                                     "decode_avg_seq_len",
                                     "decode_avg_forwarding_tokens",
                                     "avg_time_ms_per_profile_round"};

    for (const auto& field : expected_fields) {
      EXPECT_TRUE(line.find(field) != std::string::npos)
          << "Header should contain field: " << field << "\nActual header: " << line;
    }

    // Read and validate each data line
    for (size_t i = 0; i < expected_data.size(); ++i) {
      ASSERT_TRUE(std::getline(file, line)) << "Failed to read CSV data line " << (i + 1);

      // Parse the CSV line
      std::vector<std::string> fields;
      std::stringstream ss(line);
      std::string field;
      while (std::getline(ss, field, ',')) {
        fields.push_back(field);
      }

      ASSERT_EQ(fields.size(), 11) << "CSV line " << (i + 1) << " should have 11 fields";

      const auto& expected = expected_data[i];
      // Simple CSV format with space before decode fields for visual grouping
      // Need to handle potential leading space on decode_requests field
      auto trim = [](const std::string& str) {
        size_t start = str.find_first_not_of(" \t");
        if (start == std::string::npos) return std::string("");
        size_t end = str.find_last_not_of(" \t");
        return str.substr(start, end - start + 1);
      };

      EXPECT_EQ(std::stoi(fields[0]), expected.profile_round) << "Line " << (i + 1) << " profile_round mismatch";
      EXPECT_EQ(std::stoi(fields[1]), expected.layer_forward_round)
          << "Line " << (i + 1) << " layer_forward_round mismatch";
      EXPECT_EQ(std::stoi(fields[2]), expected.total_requests) << "Line " << (i + 1) << " total_requests mismatch";
      EXPECT_EQ(std::stoi(fields[3]), expected.total_sequence_len)
          << "Line " << (i + 1) << " total_sequence_len mismatch";
      EXPECT_EQ(std::stoi(fields[4]), expected.prefill_requests) << "Line " << (i + 1) << " prefill_requests mismatch";
      EXPECT_NEAR(std::stod(fields[5]), expected.prefill_avg_seq_len, 1e-3)
          << "Line " << (i + 1) << " prefill_avg_seq_len mismatch";
      EXPECT_NEAR(std::stod(fields[6]), expected.prefill_avg_forwarding_tokens, 1e-3)
          << "Line " << (i + 1) << " prefill_avg_forwarding_tokens mismatch";
      // decode_requests field may have leading space for visual grouping
      EXPECT_EQ(std::stoi(trim(fields[7])), expected.decode_requests)
          << "Line " << (i + 1) << " decode_requests mismatch";
      EXPECT_NEAR(std::stod(fields[8]), expected.decode_avg_seq_len, 1e-3)
          << "Line " << (i + 1) << " decode_avg_seq_len mismatch";
      EXPECT_NEAR(std::stod(fields[9]), expected.decode_avg_forwarding_tokens, 1e-3)
          << "Line " << (i + 1) << " decode_avg_forwarding_tokens mismatch";
      // avg_time_ms_per_profile_round should be a positive number
      EXPECT_GT(std::stof(fields[10]), 0.0f) << "Line " << (i + 1) << " average time should be positive";
    }

    // 确保没有更多的数据行
    EXPECT_FALSE(std::getline(file, line)) << "CSV should not have more data lines than expected";

    file.close();
  }

 protected:
  std::filesystem::path parent_path_;
  std::filesystem::path perf_config_path_;
  std::filesystem::path runtime_config_path_;
  std::filesystem::path cmd_path_;
};

// 测试不开DP的情况 (使用TP配置)
TEST_F(RunModelPerformanceTest, TestWithoutDataParallel) {
  // 构建命令行
  std::stringstream cmd;
  cmd << cmd_path_.string() << " --perf-config " << perf_config_path_.string() << " --runtime-config "
      << runtime_config_path_.string() << " --warmup-round 1"
      << " --profile-round 1"
      << " --output test_out.csv";

  // 执行命令并验证退出码
  int exit_code = ExecuteCommand(cmd.str());
  EXPECT_EQ(exit_code, 0) << "run_model_performance should exit successfully with TP configuration";

  // 验证输出CSV文件内容
  if (exit_code == 0) {
    // 基于新的 test_config.json，现在只有一个DP配置（dp_0），会生成一行CSV数据
    std::vector<ExpectedCsvData> expected_data;

    // dp_0配置期望值：
    // profile_round=1, layer_forward_round=6, total_requests=3 (2+1 from request_num)
    // prefill_requests=3 (forwarding_token_num=128 > decode_threshold=1), decode_requests=0
    // total_sequence_len=576 (192*2 + 192*1)
    // prefill_avg_seq_len=192.0 (576/3), decode_avg_seq_len=0.0 (no decode requests)
    // prefill_avg_forwarding_tokens=128.0 (384/3), decode_avg_forwarding_tokens=0.0 (no decode requests)
    expected_data.push_back({1, 6, 3, 3, 0, 576, 192.0, 0.0, 128.0, 0.0});

    ValidateCsvContent("test_out.csv", expected_data);
    // 清理输出文件
    std::filesystem::remove("test_out.csv");
  }
}

// 测试 --same-batch-in-dps 选项
TEST_F(RunModelPerformanceTest, TestWithSameBatchInDps) {
  // 构建命令行，包含 --same-batch-in-dps 选项
  std::stringstream cmd;
  cmd << cmd_path_.string() << " --perf-config " << perf_config_path_.string() << " --runtime-config "
      << runtime_config_path_.string() << " --warmup-round 1"
      << " --profile-round 1"
      << " --same-batch-in-dps"
      << " --output test_out_same_batch.csv";

  // 执行命令并验证退出码
  int exit_code = ExecuteCommand(cmd.str());
  EXPECT_EQ(exit_code, 0) << "run_model_performance should exit successfully with --same-batch-in-dps option";

  // 验证输出CSV文件内容
  if (exit_code == 0) {
    // --same-batch-in-dps 选项会让所有DP使用相同配置（dp_0的配置）
    // 但由于当前只有1个DP配置，所以只会生成1行CSV数据
    std::vector<ExpectedCsvData> expected_data;
    // 只有一行数据使用dp_0的配置
    expected_data.push_back({1, 6, 3, 3, 0, 576, 192.0, 0.0, 128.0, 0.0});

    ValidateCsvContent("test_out_same_batch.csv", expected_data);
    // 清理输出文件
    std::filesystem::remove("test_out_same_batch.csv");
  }
}

}  // namespace ksana_llm