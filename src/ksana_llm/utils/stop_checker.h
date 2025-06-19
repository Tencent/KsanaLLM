/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include "ksana_llm/runtime/infer_request.h"

namespace ksana_llm {

// Stop Checker to check stop strings
class StopChecker {
 public:
  // Do increment stop strings check during generation phase to do early stop
  bool CheckIncrementalStopStrings(const std::shared_ptr<InferRequest> req);

  void CheckCompleteStopStrings(const std::shared_ptr<InferRequest> req);
};

}  // namespace ksana_llm
