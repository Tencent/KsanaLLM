/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

enum RequestState {
  REQUEST_STATE_WAITING,
  REQUEST_STATE_RUNNING,
  REQUEST_STATE_SWAPPED,
  REQUEST_STATE_FINISHED,
  REQUEST_STATE_TRANSFER,
};
