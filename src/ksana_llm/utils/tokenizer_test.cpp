/* Copyright 2025 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include "ksana_llm/utils/tokenizer.h"

#include "gflags/gflags.h"
#include "test.h"

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

TEST(TokenizerTest, WrongTokenizerPath) {
  Status status = Singleton<Tokenizer>::GetInstance()->InitTokenizer("wrong_path");
  EXPECT_EQ(status.GetCode(), RET_INVALID_ARGUMENT);
}

TEST(TokenizerTest, TokenizeTest) {
  Singleton<Tokenizer>::GetInstance()->InitTokenizer("/model/llama-hf/7B");
  std::string prompt = "Hello. What's your name?";
  std::vector<int> token_list;
  std::vector<int> target_token_list = {1, 15043, 29889, 1724, 29915, 29879, 596, 1024, 29973};
  Singleton<Tokenizer>::GetInstance()->Encode(prompt, token_list, true);
  EXPECT_EQ(token_list.size(), target_token_list.size());
  for (size_t i = 0; i < token_list.size(); ++i) {
    EXPECT_EQ(token_list[i], target_token_list[i]);
  }

  std::string output_prompt = "";
  std::string target_prompt = "Hello. What's your name? My name is David.";
  token_list.emplace_back(1619);
  token_list.emplace_back(1024);
  token_list.emplace_back(338);
  token_list.emplace_back(4699);
  token_list.emplace_back(29889);
  Singleton<Tokenizer>::GetInstance()->Decode(token_list, output_prompt, true);
  EXPECT_EQ(target_prompt, output_prompt);

  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
}

TEST(TokenizerTest, GetVocabInfoTest) {
  // Initialize tokenizer first
  Singleton<Tokenizer>::GetInstance()->InitTokenizer("/model/llama-hf/7B");

  std::vector<std::string> vocab;
  int vocab_size = 32000;
  std::vector<int> stop_token_ids;

  Status status = Singleton<Tokenizer>::GetInstance()->GetVocabInfo(vocab, vocab_size, stop_token_ids);
  EXPECT_TRUE(status.OK());

  // Verify basic functionality
  EXPECT_GE(vocab_size, 32000);  // Should be at least the input size
  EXPECT_EQ(vocab.size(), static_cast<size_t>(vocab_size));  // vocab vector should match vocab_size
  EXPECT_GT(stop_token_ids.size(), 0);  // Should have at least one stop token (EOS)

  // Verify some tokens exist
  EXPECT_FALSE(vocab[1].empty());  // Token ID 1 should exist (BOS token for LLaMA)

  // Clean up
  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();
}

TEST(TokenizerTest, GetVocabInfoErrorHandlingTest) {
  // Test error handling when tokenizer is not properly initialized
  // This should trigger the catch block in GetVocabInfo

  // Ensure tokenizer is destroyed/uninitialized
  Singleton<Tokenizer>::GetInstance()->DestroyTokenizer();

  std::vector<std::string> vocab;
  int vocab_size = 32000;
  std::vector<int> stop_token_ids;

  // Call GetVocabInfo without proper initialization
  // This should trigger the exception handling in GetVocabInfo
  Status status = Singleton<Tokenizer>::GetInstance()->GetVocabInfo(vocab, vocab_size, stop_token_ids);

  // Verify that the error is properly handled
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RET_INVALID_ARGUMENT);
  EXPECT_EQ(status.GetMessage(), "Failed to extract tokenizer information");
}

}  // namespace ksana_llm
