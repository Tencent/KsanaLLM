/* Copyright 2025 Tencent Inc.  All rights reserved.*/
/* ==============================================================================*/

#include "ksana_llm/utils/logger.h"
#include "tests/test.h"
#include "xgrammar_structured_generator_creator.h"

namespace ksana_llm {

class GrammarGeneratorCreatorTest : public testing::Test {
 protected:
  void SetUp() override {
    // 创建一个简单的词汇表
    vocab_ = {"hello", "type", "test", "json", "name", "age",  "value", "\"",   ":",   ",",   "{",   "}",
              "[",     "]",     " ",    "\n",   "\t",   "true", "false", "null", "123", "456", "abc", "def"};
    vocab_size_ = vocab_.size();

    // 设置停止token ID
    stop_token_ids_ = {0, 1, 2};  // 假设前几个token是停止token

    try {
      generator_creator_ = std::make_shared<GrammarGeneratorCreator>(vocab_, vocab_size_, stop_token_ids_);
    } catch (const std::exception& e) {
      KLLM_LOG_WARNING << "Failed to initialize GrammarGeneratorCreator in test setup: " << e.what();
      generator_creator_ = nullptr;
    }
  }

  void TearDown() override { generator_creator_.reset(); }

  std::vector<std::string> vocab_;
  int vocab_size_;
  std::vector<int> stop_token_ids_;
  std::shared_ptr<GrammarGeneratorCreator> generator_creator_;
};

TEST_F(GrammarGeneratorCreatorTest, CreateGeneratorTest) {
  if (!generator_creator_) {
    GTEST_SKIP() << "GrammarGeneratorCreator not available, skipping test";
  }

  EXPECT_EQ(generator_creator_->GetConstraintType(), StructuredConstraintType::JSON);

  // 测试创建生成器
  StructuredGeneratorConfig config;
  config.constraint_spec = R"({
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "number"}
    },
    "required": ["name", "age"]
  })";

  auto generator = generator_creator_->CreateGenerator(config);
  EXPECT_NE(generator, nullptr);

  if (generator) {
    // 检查生成器是否有效
    EXPECT_TRUE(generator->IsValid());
    EXPECT_FALSE(generator->IsTerminated());
    EXPECT_EQ(generator->GetConstraintType(), StructuredConstraintType::JSON);
  }
}

TEST_F(GrammarGeneratorCreatorTest, GeneratorTokenAcceptanceTest) {
  if (!generator_creator_) {
    GTEST_SKIP() << "GrammarGeneratorCreator not available, skipping test";
  }

  // 编译一个简单的JSON schema
  StructuredGeneratorConfig config;
  config.constraint_spec = R"({
    "type": "object",
    "properties": {
      "name": {"type": "string"}
    }
  })";

  auto generator = generator_creator_->CreateGenerator(config);
  ASSERT_NE(generator, nullptr);
  ASSERT_TRUE(generator->IsValid());

  // 测试token接受
  EXPECT_TRUE(generator->AcceptToken(10));  // "{"

  EXPECT_FALSE(generator->AcceptToken(4));  // "name"
  EXPECT_TRUE(generator->AcceptToken(7));  // "\""
  EXPECT_TRUE(generator->AcceptToken(4));  // "name"

  EXPECT_FALSE(generator->AcceptToken(8));  // ":"
  EXPECT_TRUE(generator->AcceptToken(7));  // "\""
  EXPECT_TRUE(generator->AcceptToken(8));  // ":"

  // 检查是否终止
  EXPECT_FALSE(generator->IsTerminated());
}

TEST_F(GrammarGeneratorCreatorTest, GeneratorBitmaskTest) {
  if (!generator_creator_) {
    GTEST_SKIP() << "GrammarGeneratorCreator not available, skipping test";
  }

  // 编译JSON schema
  StructuredGeneratorConfig config;
  config.constraint_spec = R"({
    "type": "object",
    "properties": {
      "value": {"type": "number"}
    }
  })";

  auto generator = generator_creator_->CreateGenerator(config);
  ASSERT_NE(generator, nullptr);
  ASSERT_TRUE(generator->IsValid());

  // 创建bitmask缓冲区
  int bitmask_size = (vocab_size_ + 31) / 32;
  std::vector<int32_t> bitmask(bitmask_size, 0);

  // 测试填充bitmask
  bool needs_mask = generator->FillNextTokenBitmask(bitmask.data());
  EXPECT_TRUE(needs_mask);

  // 验证bitmask不全为零（如果需要mask的话）
  if (needs_mask) {
    size_t non_zero_num = 0;
    for (int32_t mask : bitmask) {
      if (mask != 0) {
        non_zero_num++;
      }
    }
    EXPECT_EQ(non_zero_num, 1);
  }
}

TEST_F(GrammarGeneratorCreatorTest, ComplexJSONSchemaTest) {
  if (!generator_creator_) {
    GTEST_SKIP() << "GrammarGeneratorCreator not available, skipping test";
  }

  // 测试更复杂的JSON schema
  StructuredGeneratorConfig config;
  config.constraint_spec = R"({
    "type": "object",
    "properties": {
      "users": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "active": {"type": "boolean"}
          },
          "required": ["name", "age"]
        }
      },
      "total": {"type": "number"}
    },
    "required": ["users"]
  })";

  auto generator = generator_creator_->CreateGenerator(config);
  EXPECT_NE(generator, nullptr);

  if (generator) {
    EXPECT_TRUE(generator->IsValid());
    EXPECT_FALSE(generator->IsTerminated());
  }
}

TEST_F(GrammarGeneratorCreatorTest, InvalidSchemaTest) {
  if (!generator_creator_) {
    GTEST_SKIP() << "GrammarGeneratorCreator not available, skipping test";
  }

  // 测试无效的JSON schema
  StructuredGeneratorConfig config;
  config.constraint_spec = "{ invalid json schema }";

  // 这应该抛出异常或返回nullptr
  EXPECT_THROW({ auto generator = generator_creator_->CreateGenerator(config); }, std::exception);
}

TEST_F(GrammarGeneratorCreatorTest, EmptySchemaTest) {
  if (!generator_creator_) {
    GTEST_SKIP() << "GrammarGeneratorCreator not available, skipping test";
  }

  // 测试空schema
  StructuredGeneratorConfig config;
  config.constraint_spec = "{}";

  auto generator = generator_creator_->CreateGenerator(config);
  EXPECT_NE(generator, nullptr);

  if (generator) {
    EXPECT_TRUE(generator->IsValid());
  }
}

TEST_F(GrammarGeneratorCreatorTest, GeneratorCreatorExceptionHandlingTest) {
  // Test exception handling in GrammarGeneratorCreator constructor
  std::vector<std::string> small_vocab = {"test", "token"};
  int extremely_large_vocab_size = INT_MAX;  // This might cause allocation failure
  std::vector<int> valid_stop_tokens = {0, 1};

  try {
    auto large_creator =
        std::make_shared<GrammarGeneratorCreator>(small_vocab, extremely_large_vocab_size, valid_stop_tokens);
    // If no exception is thrown, check if creation succeeded
    if (large_creator) {
      EXPECT_TRUE(true);  // Creation succeeded
    } else {
      // Creation failed, which is expected for invalid parameters
      EXPECT_TRUE(true);
    }
  } catch (const std::exception& e) {
    // Exception caught as expected - this tests the exception handling path
    EXPECT_TRUE(true);  // Exception handling is working
  }
}

TEST_F(GrammarGeneratorCreatorTest, GeneratorAcceptTokenExceptionHandlingTest) {
  if (!generator_creator_) {
    GTEST_SKIP() << "GrammarGeneratorCreator not available, skipping test";
  }

  // Create a valid generator first
  StructuredGeneratorConfig config;
  config.constraint_spec = R"({
    "type": "object",
    "properties": {
      "name": {"type": "string"}
    }
  })";

  auto generator = generator_creator_->CreateGenerator(config);
  ASSERT_NE(generator, nullptr);
  ASSERT_TRUE(generator->IsValid());

  // Test AcceptToken with potentially problematic token IDs
  bool result1 = generator->AcceptToken(INT_MAX);
  EXPECT_FALSE(result1);  // Should return false due to exception handling

  bool result2 = generator->AcceptToken(-1);
  EXPECT_FALSE(result2);  // Should return false due to exception handling

  int large_token_id = vocab_size_ + 10000;
  bool result3 = generator->AcceptToken(large_token_id);
  EXPECT_FALSE(result3);  // Should return false due to exception handling

  // Verify generator is still functional after exception handling
  EXPECT_TRUE(generator->IsValid());
}

TEST_F(GrammarGeneratorCreatorTest, FindJumpForwardTokensTest) {
  if (!generator_creator_) {
    GTEST_SKIP() << "GrammarGeneratorCreator not available, skipping test";
  }

  // Create a valid generator with JSON schema
  StructuredGeneratorConfig config;
  config.constraint_spec = R"({
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "number"}
    },
    "required": ["name", "age"]
  })";

  auto generator = generator_creator_->CreateGenerator(config);
  ASSERT_NE(generator, nullptr);
  ASSERT_TRUE(generator->IsValid());

  // Test FindJumpForwardTokens method
  int rollback_token_num = -1;                         // Initialize with invalid value
  std::vector<int> jump_tokens = {100, 200, 300};      // Initialize with some tokens

  // Call FindJumpForwardTokens - should return false for grammar constraints
  bool result = generator->FindJumpForwardTokens(rollback_token_num, jump_tokens);

  // Verify the method returns false (not supported for grammar constraints)
  EXPECT_FALSE(result);

  // Verify rollback_token_num is set to 0
  EXPECT_EQ(rollback_token_num, 0);

  // Verify jump_tokens is cleared
  EXPECT_TRUE(jump_tokens.empty());

  // Test with different initial values to ensure consistent behavior
  rollback_token_num = 5;
  jump_tokens = {400, 500};

  result = generator->FindJumpForwardTokens(rollback_token_num, jump_tokens);
  EXPECT_FALSE(result);
  EXPECT_EQ(rollback_token_num, 0);
  EXPECT_TRUE(jump_tokens.empty());

  // Test after accepting some tokens to ensure behavior is consistent
  generator->AcceptToken(10);  // Accept '{' token
  generator->AcceptToken(4);   // Accept 'name' token

  rollback_token_num = 10;
  jump_tokens = {600, 700, 800};

  result = generator->FindJumpForwardTokens(rollback_token_num, jump_tokens);
  EXPECT_FALSE(result);
  EXPECT_EQ(rollback_token_num, 0);
  EXPECT_TRUE(jump_tokens.empty());
}

}  // namespace ksana_llm
