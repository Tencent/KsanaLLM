/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

namespace ksana_llm {

enum PacketType {
  // Barrier
  CONTROL_REQ_BARRIER = 0,
  CONTROL_RSP_BARRIER = 1,

  // Layers
  CONTROL_REQ_LAYER = 2,
  CONTROL_RSP_LAYER = 3,

  // add node
  CONTROL_REQ_ADD_NODE = 4,
  CONTROL_RSP_ADD_NODE = 5,

  // cache block num
  CONTROL_REQ_BLOCKNUM = 6,
  CONTROL_RSP_BLOCKNUM = 7,

  // heartbeat.
  CONTROL_REQ_HEARTBEAT = 8,
  CONTROL_RSP_HEARTBEAT = 9,

  // schedule output.
  CONTROL_REQ_SCHEDULE = 10,
  CONTROL_RSP_SCHEDULE = 11,

  // hidden_units
  DATA_REQ_HIDDEN_UNIT = 12,
  DATA_RSP_HIDDEN_UNIT = 13,

  // shutdown
  CONTROL_REQ_SHUTDOWN = 14,
  CONTROL_RSP_SHUTDOWN = 15,

  // expert parallel.
  CONTROL_REQ_EXPERT_PARALLEL = 16,
  CONTROL_RSP_EXPERT_PARALLEL = 17,

  // Expert-Parallel DeepEP Meta data.
  CONTROL_REQ_DEEPEP_META = 18,
  CONTROL_RSP_DEEPEP_META = 19,

  // unknown type.
  PACKET_TYPE_UNKNOWN = 255
};

}  // namespace ksana_llm
