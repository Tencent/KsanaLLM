

/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

namespace ksana_llm {

enum RetCode {
  /// Operation completed successfully.
  RET_SUCCESS = 0,

  /*  ====== download and init model status code 101X ====== */
  // The specified model was not found.
  // This error occurs when the system is unable load model.
  RET_MODEL_NOT_FOUND = 1011,

  // Model download failed.
  // This error is returned when there are issues with downloading the model, possibly due to network issues.
  RET_MODEL_DOWNLOAD_FAILED = 1012,

  // Invalid model configuration.
  // This error indicates that the model configuration does not meet the required specifications.
  RET_MODEL_INVALID = 1013,

  // Tokenizer loading failed.
  RET_TOKENIZER_MODEL_LOAD_FAILED = 1014,

  // Model loading failed.
  RET_MODEL_LOAD_FAILED = 1015,

  // Preprocessing load failed.
  RET_PREPROCESS_MODEL_LOAD_FAILED = 1016,

  // Postprocessing load failed.
  RET_POSTPROCESS_MODEL_LOAD_FAILED = 1017,

  // Model quantization failed.
  RET_MODEL_QUANT_FAILED = 1018,

  /*  ====== init server status code 102X ====== */
  // Insufficient video memory.
  // This error is returned when the device does not have enough video memory to load the model.
  RET_DEVICE_MEM_INSUFFICIENT = 1021,

  // General segment fault.
  RET_SEGMENT_FAULT = 1022,

  // The service has been terminated.
  RET_SERVICE_TERMINATED = 1023,

  // Service initialization failed.
  RET_INIT_FAILED = 1024,

  // Device memory allocation failed.
  RET_DEVICE_MEM_ALLOCATE_FAILED = 1025,

  // memory allocation failed.
  RET_HOST_MEM_ALLOCATE_FAILED = 1026,

  // Bind available port failed.
  RET_BIND_PORT_FAILED = 1027,

  // Failed initializing the connector.
  RET_CONFIG_NOT_FOUND = 1028,

  /* ====== runtime error code 20xx====== */
  // Invalid input argument.
  // Check if the inputs are within the expected range and types.
  RET_INVALID_ARGUMENT = 2000,

  // No prediction target found.
  RET_NO_PREDICT_TARGET = 2001,

  // Input length exceeds the model's maximum inference length.
  RET_INPUT_LENGTH_EXCEEDED = 2002,

  // Sequence length exceeds capacity.
  RET_EXCEED_CAPACITY = 2003,

  // request terminated.
  RET_REQUEST_TERMINATED = 2004,

  /*  ====== internal error code ====== 30XX */
  // Tokenizer execution failed.
  RET_TOKENIZER_FAILED = 3000,

  // Inference execution failed.
  RET_INFER_FAILED = 3001,

  // Preprocessing failed.
  RET_PREPROCESS_FAILED = 3002,

  // Postprocessing failed.
  RET_POSTPROCESS_FAILED = 3003,

  // Operation timed out.
  RET_REQUEST_TIMEOUT = 3004,

  // Service is overloaded, discarding some requests.
  RET_PREDICTOR_DISCARD = 3005,

  // While high bandwidth memory block are not enough.
  RET_OUT_OF_DEVICE_EMORY = 3007,

  // While memory block are not enough.
  RET_OUT_OF_HOST_MEMORY = 3008,

  // Memory double free block failed.
  RET_DEVICE_MEM_FREE_FAILED = 3009,

  // Runtime error occurred.
  RET_RUNTIME_FAILED = 3010,

  // Iteration stopped unexpectedly.
  RET_STOP_ITERATION = 3011,

  // Undefined reference error.
  RET_UNDEFINED_REFERENCE = 3012,

  // ====== Pluging status code 40XX======

  // ====== Other status code 50XX======

  // Failed to access ServingManage.
  RET_MANAGER_FAILED = 5001,

  RET_NOT_IMPLEMENTED = 5002,

  // ====== An unknow error status code 60XX======
  RET_INTERNAL_UNKNOWN_ERROR = 6000,
};

}  // namespace ksana_llm
