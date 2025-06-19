/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

namespace llm_kernels {
namespace nvidia {

enum class MOEExpertScaleNormalizationMode : int {
  NONE = 0,     //!< Run the softmax on all scales and select the topk
  RENORMALIZE,  //!< Renormalize the selected scales so they sum to one. This is equivalent to only running softmax on
                //!< the topk selected experts
};

enum class RoutingFunctionType : int {
  GREEDY_TOPK_SOFTMAX_SCORE = 1,  // fused topk with softmax
  FAST_TOPK_SIGMOID_SCORE         // fast topk with sigmoid
};

}  // namespace nvidia
}  // namespace llm_kernels
