/* Copyright 2024 Tencent Inc.  All rights reserved.
Partialy modify from
https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/custom_op/llama_infer/atb_ops

==============================================================================*/
#include "csrc/kernels/ascend/attention/attention.h"
#include "csrc/utils/ascend/common.h"
#include "csrc/utils/ascend/tiling_data_types.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <type_traits>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"
#include "atb/atb_infer.h"

namespace llm_kernels {
namespace ascend {

#define DISPATCH_BY_3_ACLTYPE(dtype, func, ...)                                                     \
  switch (dtype) {                                                                                  \
    case aclDataType::ACL_FLOAT16:                                                                  \
      return func<aclFloat16>(__VA_ARGS__);                                                         \
    case aclDataType::ACL_BF16:                                                                     \
      return func<int16_t>(__VA_ARGS__);                                                            \
    case aclDataType::ACL_FLOAT:                                                                    \
      return func<float>(__VA_ARGS__);                                                              \
    default:                                                                                        \
   throw std::invalid_argument("Invalid type, only support float16 or float32."); \
  }

// The max seq_len
static constexpr size_t kMaxSeqLen = 4096;

ATBAttention::~ATBAttention() {}

void ATBAttention::Forward(void* output, void* qkv_tensor, void* pos_ids, void* slot_mapping, void* k_cache,
                           void* v_cache, void* block_tables, const uint32_t max_num_blocks_per_query,
                           const uint32_t batch_size, const uint32_t total_token_num, const uint32_t total_block_num,
                           const uint32_t block_token_num, const uint32_t layer_index, void* seq_len,
                           const bool is_multi_token_forward, atb::Context* atb_context,
                           void (*ws_func)(size_t, void**)) {
  aclDataType acl_dtype = inter_data_type_;

  atb_op_executor_.ResetVariantPack();
  // qkv_input_tensor_id
  atb_op_executor_.SetInputTensor(qkv_tensor, {total_token_num, (head_size_ + 2 * kv_head_size_) * head_dim_},
                                  acl_dtype);
  // pos_input_tensor_id
  atb_op_executor_.SetInputTensor(pos_ids, {total_token_num}, aclDataType::ACL_INT64);
  // rope_cos_input_tensor_id
  atb_op_executor_.SetInputTensor(rope_cos_workspace_ptr_, {max_position_embeddings_, head_dim_}, acl_dtype);
  // rope_sin_input_tensor_id
  atb_op_executor_.SetInputTensor(rope_sin_workspace_ptr_, {max_position_embeddings_, head_dim_}, acl_dtype);
  if (is_multi_token_forward) {
    // mask_input_tensor_id
    atb_op_executor_.SetInputTensor(attn_mask_ptr_, {kMaxSeqLen, kMaxSeqLen}, acl_dtype);
  }
  // k_cache_input_tensor_id
  atb_op_executor_.SetInputTensor(k_cache, {total_block_num, block_token_num, kv_head_size_, head_dim_}, acl_dtype);
  // v_cache_input_tensor_id
  atb_op_executor_.SetInputTensor(v_cache, {total_block_num, block_token_num, kv_head_size_, head_dim_}, acl_dtype);
  // slots_input_tensor_id
  atb_op_executor_.SetInputTensor(slot_mapping, {total_token_num}, aclDataType::ACL_INT32);
  if (!is_multi_token_forward) {
    // block_tables_input_tensor_id
    atb_op_executor_.SetInputTensor(block_tables, {total_token_num, max_num_blocks_per_query}, aclDataType::ACL_INT32);
  }
  // seqlen_input_tensor_id
  atb_op_executor_.SetInputTensor(seq_len, {batch_size}, aclDataType::ACL_INT32);
  atb_op_executor_.SetOutputTensor(output, {total_token_num, head_size_ * head_dim_}, acl_dtype);

  atb_op_executor_.Run(atb_context, ws_func);
}

void ATBAttention::Initialize(aclDataType inter_data_type, uint32_t max_batch_size, uint32_t head_size,
                              uint32_t kv_head_size, uint32_t head_dim, uint32_t layer_num, uint32_t layer_idx,
                              uint32_t block_token_num, aclrtStream& stream, const int rank,
                              const bool is_multi_token_forward, const size_t max_position_embeddings,
                              const float rope_base, const RotaryEmbeddingType scaling_type,
                              const float scaling_factor) {
  inter_data_type_ = inter_data_type;
  max_batch_size_ = max_batch_size;
  head_size_ = head_size;
  kv_head_size_ = kv_head_size;
  head_dim_ = head_dim;
  layer_num_ = layer_num;
  block_token_num_ = block_token_num;
  rank_ = rank;
  is_multi_token_forward_ = is_multi_token_forward;
  max_position_embeddings_ = max_position_embeddings;

  // TODO(karlluo): tag which query should be handle
  batch_status_.resize(max_batch_size_, 1);

  // init rope cos and sin
  InitRopeCosSinWorkspace(max_position_embeddings, rope_base, head_dim, scaling_factor, scaling_type, stream);

  if (is_multi_token_forward) {
    InitAttnMask();
  }

  uint32_t tensor_id = 0;
  // input
  // shape: (ntokens, (head_size + 2 * kv_head_size) * head_dim)
  uint32_t qkv_input_tensor_id = tensor_id++;
  // shape: (ntokes,)
  uint32_t pos_input_tensor_id = tensor_id++;
  // shape: (ntokens, head_dim)
  uint32_t rope_cos_input_tensor_id = tensor_id++;
  // shape: (ntokens, head_dim)
  uint32_t rope_sin_input_tensor_id = tensor_id++;
  // is_multi_token_forward == true:
  //   shape: (kMaxSeqLen, kMaxSeqLen)
  uint32_t mask_input_tensor_id = is_multi_token_forward ? tensor_id++ : 0;
  // shape: (num_blocks, block_size, k_head_num, head_size)
  uint32_t k_cache_input_tensor_id = tensor_id++;
  // shape: (num_blocks, block_size, k_head_num, head_size)
  uint32_t v_cache_input_tensor_id = tensor_id++;
  // shape: (ntokens)
  uint32_t slots_input_tensor_id = tensor_id++;
  // is_multi_token_forward == false:
  //   shape: (num_tokens, max_num_blocks_per_query)
  uint32_t block_tables_input_tensor_id = !is_multi_token_forward ? tensor_id++ : 0;
  // shape: (batch)
  uint32_t seqlen_input_tensor_id = tensor_id++;
  // output
  // shape: (ntokens, head_size * head_dim)
  uint32_t attn_output_tensor_id = tensor_id++;

  // intermedia
  uint32_t q_inner_tensor_id = tensor_id++;
  uint32_t k_inner_tensor_id = tensor_id++;
  uint32_t v_inner_tensor_id = tensor_id++;
  uint32_t emb_q_inner_tensor_id = tensor_id++;
  uint32_t emb_k_inner_tensor_id = tensor_id++;
  uint32_t cos_inner_tensor_id = tensor_id++;
  uint32_t sin_inner_tensor_id = tensor_id++;

  atb::GraphParam op_graph;
  op_graph.name = is_multi_token_forward ? "ATBSelfAttentionOp" : "ATBPagedAttentionOp";
  op_graph.inTensorNum = seqlen_input_tensor_id - qkv_input_tensor_id + 1;
  op_graph.outTensorNum = 1;
  op_graph.internalTensorNum = sin_inner_tensor_id - q_inner_tensor_id + 1;
  op_graph.nodes.resize(4 + 2);  // gather rope's cos/sin + split qkv + rope + write kv + flash attn/page attn
  uint32_t node_idx = 0;

  // split q,k,v
  {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    CreateSplitQKVOperation(head_size, kv_head_size, head_dim, &op_node.operation);
    op_node.inTensorIds = {qkv_input_tensor_id};
    op_node.outTensorIds = {q_inner_tensor_id, k_inner_tensor_id, v_inner_tensor_id};
  }

  // gather cos
  {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::GatherParam gather_param;
    atb::CreateOperation(gather_param, &op_node.operation);
    op_node.inTensorIds = {rope_cos_input_tensor_id, pos_input_tensor_id};
    op_node.outTensorIds = {cos_inner_tensor_id};
  }

  // gather sin
  {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::GatherParam gather_param;
    atb::CreateOperation(gather_param, &op_node.operation);
    op_node.inTensorIds = {rope_sin_input_tensor_id, pos_input_tensor_id};
    op_node.outTensorIds = {sin_inner_tensor_id};
  }

  // rope
  {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::RopeParam op_param;
    op_param.rotaryCoeff = 2;
    atb::CreateOperation(op_param, &op_node.operation);
    op_node.inTensorIds = {q_inner_tensor_id, k_inner_tensor_id, cos_inner_tensor_id, sin_inner_tensor_id,
                           seqlen_input_tensor_id};
    op_node.outTensorIds = {emb_q_inner_tensor_id, emb_k_inner_tensor_id};
  }

  // write kv
  {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::ReshapeAndCacheParam op_param;
    atb::CreateOperation(op_param, &op_node.operation);
    op_node.inTensorIds = {emb_k_inner_tensor_id, v_inner_tensor_id, k_cache_input_tensor_id, v_cache_input_tensor_id,
                           slots_input_tensor_id};
    op_node.outTensorIds = {k_cache_input_tensor_id, v_cache_input_tensor_id};  // write in place
    op_node.inTensorReshapeFuncs.resize(op_node.inTensorIds.size());
    op_node.inTensorReshapeFuncs[0] = [=](const atb::Dims& old_shape, atb::Dims& new_shape) {
      new_shape.dimNum = 3;
      new_shape.dims[0] = old_shape.dims[0];
      new_shape.dims[1] = kv_head_size;
      new_shape.dims[2] = head_dim;
    };
    op_node.inTensorReshapeFuncs[1] = [=](const atb::Dims& old_shape, atb::Dims& new_shape) {
      new_shape.dimNum = 3;
      new_shape.dims[0] = old_shape.dims[0];
      new_shape.dims[1] = kv_head_size;
      new_shape.dims[2] = head_dim;
    };
  }

  // flash attn or paged attn
  if (is_multi_token_forward) {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::SelfAttentionParam op_param;
    op_param.headNum = head_size;
    op_param.kvHeadNum = kv_head_size;
    op_param.qkScale = 1.0f / sqrt(head_dim);
    op_param.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    op_param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
    op_param.kernelType = atb::infer::SelfAttentionParam::KernelType::KERNELTYPE_HIGH_PRECISION;
    op_param.isTriuMask = 1;
    atb::CreateOperation(op_param, &op_node.operation);
    op_node.inTensorIds = {emb_q_inner_tensor_id, emb_k_inner_tensor_id, v_inner_tensor_id, mask_input_tensor_id,
                           seqlen_input_tensor_id};
    op_node.outTensorIds = {attn_output_tensor_id};
    op_node.inTensorReshapeFuncs.resize(op_node.inTensorIds.size());
  } else {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::PagedAttentionParam op_param;
    op_param.headNum = head_size;
    op_param.qkScale = 1.0f / sqrt(head_dim);
    op_param.kvHeadNum = kv_head_size;
    atb::CreateOperation(op_param, &op_node.operation);

    op_node.inTensorIds = {emb_q_inner_tensor_id, k_cache_input_tensor_id, v_cache_input_tensor_id,
                           block_tables_input_tensor_id, seqlen_input_tensor_id};
    op_node.outTensorIds = {attn_output_tensor_id};
    op_node.inTensorReshapeFuncs.resize(op_node.inTensorIds.size());
    op_node.inTensorReshapeFuncs[0] = [=](const atb::Dims& old_shape, atb::Dims& new_shape) {
      new_shape.dimNum = 3;
      new_shape.dims[0] = old_shape.dims[0];
      new_shape.dims[1] = head_size;
      new_shape.dims[2] = head_dim;
    };
  }

  op_graph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& in_tensor_descs,
                                atb::SVector<atb::TensorDesc>& out_tensor_descs) {
    out_tensor_descs.resize(1);
    out_tensor_descs.at(0) = in_tensor_descs.at(0);
    out_tensor_descs.at(0).shape.dims[0] = in_tensor_descs.at(0).shape.dims[0];
    out_tensor_descs.at(0).shape.dims[1] = head_size * head_dim;
    return atb::NO_ERROR;
  };
  atb_op_executor_.Init(rank, op_graph);
  atb_op_executor_.ResetVariantPack();
}

bool ATBAttention::InitRopeCosSinWorkspace(const size_t max_position_embeddings, const float rope_base,
                                           const uint32_t head_dim, const float scaling_factor,
                                           const RotaryEmbeddingType scaling_type, aclrtStream& stream) {
  DISPATCH_BY_3_ACLTYPE(inter_data_type_, InitRopeCosSinWorkspaceT, max_position_embeddings, rope_base, head_dim,
                        scaling_factor, scaling_type, stream);
}

template <typename DTYPE>
bool ATBAttention::InitRopeCosSinWorkspaceT(const size_t max_position_embeddings, const float rope_base,
                                            const uint32_t head_dim, const float scaling_factor,
                                            const RotaryEmbeddingType scaling_type, aclrtStream& stream) {
  std::vector<DTYPE> cos_workspace_host(max_position_embeddings * head_dim, 0u);
  std::vector<DTYPE> sin_workspace_host(max_position_embeddings * head_dim, 0u);
  ACL_CHECK_RET(aclrtMalloc(&rope_cos_workspace_ptr_, max_position_embeddings * head_dim * sizeof(DTYPE),
                            ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMalloc(&rope_sin_workspace_ptr_, max_position_embeddings * head_dim * sizeof(DTYPE),
                            ACL_MEM_MALLOC_HUGE_FIRST));
  size_t extend_max_len = max_position_embeddings;
  float new_base = rope_base;
  float scaling = 1.0f;
  // https://github.com/vllm-project/vllm/blob/523e30ea0c5abcb447763dcd9a77b54d5c5f3239/vllm/model_executor/layers/rotary_embedding.py#L219
  if (scaling_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING) {
    extend_max_len = max_position_embeddings * scaling_factor;
    new_base =
        std::pow(rope_base * ((scaling_factor * extend_max_len / max_position_embeddings) - (scaling_factor - 1)),
                 (head_dim / (head_dim - 2)));
  }
  if (scaling_type == RotaryEmbeddingType::LINEAR_SCALING) {
    extend_max_len = max_position_embeddings * scaling_factor;
    scaling = scaling_factor;
  }
  for (size_t token_idx = 0; token_idx < extend_max_len; ++token_idx) {
    int pos = token_idx;
    for (size_t rid = 0; rid < head_dim / 2; ++rid) {
      float inv_freq = 1.0 / std::pow(new_base, rid * 2 / float(head_dim));
      float freq = pos * inv_freq / scaling;
      if (std::is_same<DTYPE, aclFloat16>::value) {
        cos_workspace_host[pos * head_dim + rid] = aclFloatToFloat16(std::cos(freq));
        cos_workspace_host[pos * head_dim + head_dim / 2 + rid] = cos_workspace_host[pos * head_dim + rid];
        sin_workspace_host[pos * head_dim + rid] = aclFloatToFloat16(std::sin(freq));
        sin_workspace_host[pos * head_dim + head_dim / 2 + rid] = sin_workspace_host[pos * head_dim + rid];
      } else if (std::is_same<DTYPE, float>::value) {
        cos_workspace_host[pos * head_dim + rid] = DTYPE(std::cos(freq));
        cos_workspace_host[pos * head_dim + head_dim / 2 + rid] = cos_workspace_host[pos * head_dim + rid];
        sin_workspace_host[pos * head_dim + rid] = DTYPE(std::sin(freq));
        sin_workspace_host[pos * head_dim + head_dim / 2 + rid] = sin_workspace_host[pos * head_dim + rid];
      } else if (std::is_same<DTYPE, int16_t>::value) {
        float cos_val = std::cos(freq);
        float sin_val = std::cos(freq);
        // NOTE(karlluo): there is not bfloat16 type in cann, so we take int16_t as bfloat16 as dtype indicator
        cos_workspace_host[pos * head_dim + rid] = (*reinterpret_cast<int*>(&(cos_val))) >> 16;
        cos_workspace_host[pos * head_dim + head_dim / 2 + rid] = cos_workspace_host[pos * head_dim + rid];
        sin_workspace_host[pos * head_dim + rid] = (*reinterpret_cast<int*>(&(sin_val))) >> 16;
        sin_workspace_host[pos * head_dim + head_dim / 2 + rid] = sin_workspace_host[pos * head_dim + rid];
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16, bfloat16 or float32.");
      }
    }
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(rope_cos_workspace_ptr_, cos_workspace_host.size() * sizeof(DTYPE),
                                 cos_workspace_host.data(), cos_workspace_host.size() * sizeof(DTYPE),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtMemcpyAsync(rope_sin_workspace_ptr_, sin_workspace_host.size() * sizeof(DTYPE),
                                 sin_workspace_host.data(), sin_workspace_host.size() * sizeof(DTYPE),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  return true;
}

bool ATBAttention::InitAttnMask() {
  DISPATCH_BY_3_ACLTYPE(inter_data_type_, InitAttnMaskT);
}

template <typename DTYPE>
bool ATBAttention::InitAttnMaskT() {
  constexpr uint16_t kMaskValue = 1;  // KERNELTYPE_HIGH_PRECISION: 1; KERNELTYPE_DEFAULT: -inf(0xFBFF)
  std::vector<uint16_t> mask(kMaxSeqLen * kMaxSeqLen, 0);
  for (size_t i = 0; i < kMaxSeqLen; ++i) {
    std::fill_n(mask.begin() + i * kMaxSeqLen + i + 1, kMaxSeqLen - i - 1, kMaskValue);
  }

  ACL_CHECK_RET(aclrtMalloc(&attn_mask_ptr_, kMaxSeqLen * kMaxSeqLen * sizeof(DTYPE), ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(attn_mask_ptr_, kMaxSeqLen * kMaxSeqLen * sizeof(DTYPE), mask.data(),
                            kMaxSeqLen * kMaxSeqLen * sizeof(DTYPE), aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE));
  return true;
}

void ATBAttention::CreateSplitQKVOperation(uint32_t head_size, uint32_t kv_head_size, uint32_t head_dim,
                                           atb::Operation** operation) {
  uint32_t tensor_idx = 0;
  uint32_t input_qkv_tensor = tensor_idx++;  // [ntokens, 3 * head_num * head_dim] or [ntokens,
                                             // (head_num + 2 * kv_head_num) * head_dim]
  uint32_t output_q_tensor = tensor_idx++;   // [ntokens, head_num * head_dim]
  uint32_t output_k_tensor = tensor_idx++;
  uint32_t output_v_tensor = tensor_idx++;

  auto kv_head_num = (kv_head_size > 0 && kv_head_size != head_size) ? kv_head_size : 0;

  uint32_t node_idx = 0;
  atb::GraphParam op_graph;
  op_graph.name = "SplitQKV";
  op_graph.inTensorNum = 1;
  op_graph.outTensorNum = 3;
  op_graph.internalTensorNum = 0;
  op_graph.nodes.resize(kv_head_num > 0 ? 3 : 1);

  if (kv_head_num > 0) {
    {
      atb::Node& op_node = op_graph.nodes.at(node_idx++);
      atb::infer::SliceParam op_param;
      op_param.offsets.resize(2);
      op_param.size.resize(2);
      op_param.offsets[0] = 0;
      op_param.offsets[1] = 0;
      op_param.size[0] = -1;
      op_param.size[1] = head_size * head_dim;
      atb::CreateOperation(op_param, &op_node.operation);
      op_node.inTensorIds = {input_qkv_tensor};
      op_node.outTensorIds = {output_q_tensor};
    }
    {
      atb::Node& op_node = op_graph.nodes.at(node_idx++);
      atb::infer::SliceParam op_param;
      op_param.offsets.resize(2);
      op_param.size.resize(2);
      op_param.offsets[0] = 0;
      op_param.offsets[1] = head_size * head_dim;
      op_param.size[0] = -1;
      op_param.size[1] = kv_head_size * head_dim;
      atb::CreateOperation(op_param, &op_node.operation);
      op_node.inTensorIds = {input_qkv_tensor};
      op_node.outTensorIds = {output_k_tensor};
    }
    {
      atb::Node& op_node = op_graph.nodes.at(node_idx++);
      atb::infer::SliceParam op_param;
      op_param.offsets.resize(2);
      op_param.size.resize(2);
      op_param.offsets[0] = 0;
      op_param.offsets[1] = (head_size + kv_head_size) * head_dim;
      op_param.size[0] = -1;
      op_param.size[1] = kv_head_size * head_dim;
      atb::CreateOperation(op_param, &op_node.operation);
      op_node.inTensorIds = {input_qkv_tensor};
      op_node.outTensorIds = {output_v_tensor};
    }
    op_graph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& input_tensor_descs,
                                  atb::SVector<atb::TensorDesc>& output_tensor_descs) {
      output_tensor_descs.resize(3);
      output_tensor_descs.at(0) = input_tensor_descs.at(0);
      output_tensor_descs.at(0).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(0).shape.dims[1] = head_size * head_dim;
      output_tensor_descs.at(1) = input_tensor_descs.at(0);
      output_tensor_descs.at(1).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(1).shape.dims[1] = kv_head_size * head_dim;
      output_tensor_descs.at(2) = input_tensor_descs.at(0);
      output_tensor_descs.at(2).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(2).shape.dims[1] = kv_head_size * head_dim;
      return atb::NO_ERROR;
    };
  } else {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::SplitParam op_param;
    op_param.splitDim = 1;
    op_param.splitNum = 3;  // only fp16
    atb::CreateOperation(op_param, &op_node.operation);
    op_node.inTensorIds = {input_qkv_tensor};
    op_node.outTensorIds = {output_q_tensor, output_k_tensor, output_v_tensor};
    op_graph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& input_tensor_descs,
                                  atb::SVector<atb::TensorDesc>& output_tensor_descs) {
      output_tensor_descs.resize(3);
      output_tensor_descs.at(0) = input_tensor_descs.at(0);
      output_tensor_descs.at(0).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(0).shape.dims[1] = head_size * head_dim;
      output_tensor_descs.at(1) = input_tensor_descs.at(0);
      output_tensor_descs.at(1).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(1).shape.dims[1] = head_size * head_dim;
      output_tensor_descs.at(2) = input_tensor_descs.at(0);
      output_tensor_descs.at(2).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(2).shape.dims[1] = head_size * head_dim;
      return atb::NO_ERROR;
    };
  }
  atb::CreateOperation(op_graph, operation);
}

}  // namespace ascend
}  // namespace llm_kernels
