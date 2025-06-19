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

}  // namespace ksana_llm
