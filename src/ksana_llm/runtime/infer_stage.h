/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

namespace ksana_llm {

enum class InferStage {
  // The context decode stage.
  kContext,

  // The decode stage.
  kDecode,
};

}  // namespace ksana_llm