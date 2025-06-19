/* Copyright 2025 Tencent Inc.  All rights reserved.
 * ==============================================================================*/

#include "ksana_llm/utils/utils.h"

#include "tests/test.h"

namespace ksana_llm {

TEST(TestEnvSetting, BasicTest) {
  const char* envName = "KLLM_TEST_ENV_SETTING";

  unsetenv(envName);
  EXPECT_TRUE(GetEnvAsPositiveInt(envName, 10) == 10);

  setenv(envName, "-10", 1);
  EXPECT_TRUE(GetEnvAsPositiveInt(envName, 10) == 0);

  setenv(envName, "10", 1);
  EXPECT_TRUE(GetEnvAsPositiveInt(envName, 0) == 10);

  setenv(envName, "abc", 1);
  EXPECT_TRUE(GetEnvAsPositiveInt(envName, 10) == 10);

  setenv(envName, "1000000000000000000000000", 1);
  EXPECT_TRUE(GetEnvAsPositiveInt(envName, 10) == 10);
}

}  // namespace ksana_llm
