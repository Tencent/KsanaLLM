# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import asyncio
import os
import threading
import time
import traceback
import uuid

import ksana_llm
import numpy as np
import orjson
import triton_python_backend_utils as pb_utils
from ksana_llm.arg_utils import EngineArgs

_KSANA_CONFIG_FILENAME = "ksana_llm.yaml"

TRITON_TO_KSANA = {
    "request_output_len": "max_new_tokens",
    "presence_penalty": "presence_penalty",
    "random_seed": "random_seed",
    "min_length": "min_tokens",
    "repetition_penalty": "repetition_penalty",
    "temperature": "temperature",
    "runtime_top_k": "topk",
    "runtime_top_p": "topp",
    "len_penalty": "length_penalty",
    "beam_width": "num_beams",
    "stop_token_ids": "stop_token_ids",
    "stop_words_list": "stop_strings",
    "return_log_probs": "logprobs",
    "input_ids": "input_tokens",
    "text_input": "prompt",
    "sampling_parameters": "sampling_config",
    "messages": "messages",
    "input_refit_embedding": "input_refit_embedding",
    "pos": "pos",
    "embeddings": "embeddings",
    "request_type": "request_type",
    "request_bytes": "request_bytes"
}

GENERATION_CONFIG_KEYS = {
    "temperature",
    "topk",
    "topp",
    "logprobs",
    "max_new_tokens",
    "repetition_penalty",
    "stop_token_ids",
    "num_beams",
    "do_sample",
    "no_repeat_ngram_size",
    "encoder_no_repeat_ngram_size",
    "num_return_sequences",
    "length_penalty",
    "stop_strings",
    "ignore_eos",
}


def parse_input(log, request, input_dtypes):
    request_dict = {}
    for input_tensor in request.inputs():
        input_name = input_tensor.name()
        ksana_key = input_name

        # map triton input name to ksana input name
        if input_name in TRITON_TO_KSANA:
            ksana_key = TRITON_TO_KSANA[input_name]

        dtype, dims = input_dtypes.get(input_name)
        if not dtype:
            continue
        if input_name == "sampling_parameters" or input_name == "messages" or input_name == "input_refit_embedding":
            # Extract the data from the input tensor
            params_array = input_tensor.as_numpy()
            # Assuming the array has at least one element
            params_bytes = params_array[0][0]
            # Parse the bytes to a dictionary
            params_dict = orjson.loads(params_bytes)
            request_dict[ksana_key] = params_dict
            continue

        if input_name == "pos":
            # Handle pos as uint32 array
            value = input_tensor.as_numpy()[0].astype(np.uint32).tolist()
            request_dict[ksana_key] = value
            continue

        if input_name == "embeddings":
            # Handle embeddings as float array
            value = input_tensor.as_numpy()[0].astype(np.float32).tolist()
            request_dict[ksana_key] = value
            continue

        if input_name == "text_input":
            request_dict[ksana_key] = input_tensor.as_numpy()[0][0].decode("utf-8")
            continue

        if input_name == "return_log_probs":
            if input_tensor.as_numpy():
                request_dict["logprobs_num"] = 1

        value = input_tensor.as_numpy()
        if len(dims) == 1 and dims[0] == 1:  # dims:[1]
            if dtype is np.bool_:
                value = bool(value[0])
            if dtype == np.object_:
                value = value[0][0].decode("utf-8")
            if np.issubdtype(dtype, np.integer):
                value = int(value[0])
            if np.issubdtype(dtype, np.floating):
                value = float(value[0])
        else:
            # for compatible trt-llm here input_ids and inpput text is 2D array
            value = value[0].astype(dtype).tolist()

        request_dict[ksana_key] = value

    sampling_config = request_dict.get("sampling_config", {})
    sampling_config.update({
        key: value for key in GENERATION_CONFIG_KEYS
        if (value := request_dict.get(key)) is not None
    })

    request_dict["sampling_config"] = sampling_config
    return request_dict


def compute_cumulative_logprob(log_probs):
    return [
        [
            len(beam),
            sum(
                logprobs
                for logprobs_pair_list in beam
                for token_id, logprobs in logprobs_pair_list
            ),
        ]
        for beam in log_probs
    ]


class GenerateReq:
    def __init__(self, engine, request_id, request_dict):
        self.llm_engine = engine
        self.request_id = request_id
        self.request_dict = request_dict

    async def generate(self):
        status, generator = await self.llm_engine.generate(self.request_dict, streamer=True)
        pb_utils.Logger.log_info(f"status: {status}, generator: {generator}")
        return generator


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = orjson.loads(args["model_config"])

        self.using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config
        )
        assert (
            self.using_decoupled
        ), "Ksana_LLM Triton backend must be configured to use decoupled model transaction policy"

        # todo(shawnding): 远程获取配置文件
        ksana_yaml_path = os.path.join(pb_utils.get_model_dir(), _KSANA_CONFIG_FILENAME)
        assert os.path.isfile(
            ksana_yaml_path
        ), f"'{_KSANA_CONFIG_FILENAME}' must be provided in '{pb_utils.get_model_dir()}'"

        self.logger.log_info(f"Ksana_LLM engine config: {ksana_yaml_path}")

        # Initialize the engine
        engine_args = EngineArgs.from_config_file(ksana_yaml_path)
        self.llm_engine = ksana_llm.KsanaLLMEngine.from_engine_args(engine_args)
        self.llm_engine.initialize()
        self.tokenizer = self.llm_engine.tokenizer

        self.input_dtypes = {
            input_cfg["name"]: (
                pb_utils.triton_string_to_numpy(input_cfg["data_type"]),
                input_cfg["dims"],
            )
            for input_cfg in self.model_config["input"]
        }
        self.output_dtypes = {
            output_cfg["name"]: pb_utils.triton_string_to_numpy(output_cfg["data_type"])
            for output_cfg in self.model_config["output"]
        }

        self.ongoing_request_count = 0
        self._loop = asyncio.get_event_loop()
        self._loop_thread = threading.Thread(
            target=self.engine_loop, args=(self._loop,)
        )
        self._shutdown_event = asyncio.Event()
        self._loop_thread.start()

    def create_task(self, coroutine_task):
        assert (
            not self._shutdown_event.is_set()
        ), "Cannot create tasks after shutdown has been requested"
        return asyncio.run_coroutine_threadsafe(coroutine_task, self._loop)

    def engine_loop(self, loop):
        asyncio.set_event_loop(loop)
        self._loop.run_until_complete(self.await_shutdown())

    async def await_shutdown(self):
        while not self._shutdown_event.is_set():
            await asyncio.sleep(5)
        while self.ongoing_request_count > 0:
            self.logger.log_info(
                f"Awaiting remaining {self.ongoing_request_count} requests"
            )
            await asyncio.sleep(5)
        for task in asyncio.all_tasks(loop=self._loop):
            if task is not asyncio.current_task():
                task.cancel()
        self.logger.log_info("Shutdown complete")

    def create_response(self, ksana_python_output, length_penalty: float = 1.0):
        cumulative_logprob = []
        sequence_length = [len(lst) for lst in ksana_python_output.output_tokens]
        max_len = max(len(lst) for lst in ksana_python_output.output_tokens)
        padded_lists = [
            lst + [self.tokenizer.eos_token_id] * (max_len - len(lst))
            for lst in ksana_python_output.output_tokens
        ]

        output_text = [
            [
                self.tokenizer.decode(token_list, skip_special_tokens=True).rstrip(
                    "\ufffd"
                )
            ]
            for token_list in ksana_python_output.output_tokens
        ]

        """Calculate the beam search score with length penalty.
        Adapted from
        https://github.com/huggingface/transformers/blob/main/src/transformers/generation/beam_search.py
        """
        cumulative_logprob += [
            logprob / (seq_length**length_penalty)
            for seq_length, logprob in compute_cumulative_logprob(
                ksana_python_output.logprobs
            )
        ]
        if cumulative_logprob == []:
            cumulative_logprob = [None]

        finish_reason = [ksana_python_output.finish_status.GetCode()]

        triton_output_tensors = [
            pb_utils.Tensor(
                "text_output",
                np.asarray([output_text], dtype=self.output_dtypes["text_output"]),
            ),
            pb_utils.Tensor(
                "output_ids",
                np.asarray([padded_lists], dtype=self.output_dtypes["output_ids"]),
            ),
            pb_utils.Tensor(
                "completion_tokens",
                np.asarray(
                    [sequence_length], dtype=self.output_dtypes["completion_tokens"]
                ),
            ),
            pb_utils.Tensor(
                "sequence_length",
                np.asarray(
                    [sequence_length], dtype=self.output_dtypes["sequence_length"]
                ),
            ),
            pb_utils.Tensor(
                "cum_log_probs",
                np.asarray(
                    [cumulative_logprob], dtype=self.output_dtypes["cum_log_probs"]
                ),
            ),
            pb_utils.Tensor(
                "finish_reason",
                np.asarray(finish_reason, dtype=self.output_dtypes["finish_reason"]),
            ),
        ]
        response = pb_utils.InferenceResponse(output_tensors=triton_output_tensors)
        return output_text, padded_lists, sequence_length, finish_reason, response

    async def generate_per_req(self, response_sender, request_dict, req_id):
        self.ongoing_request_count += 1
        try:
            start_time = time.time()
            first_token_time = None
            request_id = uuid.uuid4()
            stream = request_dict.get("streaming", True)
            sampling_parameters = request_dict.get("sampling_config", {})

            generator = await GenerateReq(
                self.llm_engine, request_id, request_dict
            ).generate()

            last_output = None
            length_penalty = request_dict.get("length_penalty", 1.0)
            async for output in generator:
                if response_sender.is_cancelled():
                    self.logger.log_warn(f"Request {request_id} cancelled")
                    break
                if stream:
                    if first_token_time is None:
                        first_token_time = time.time()
                    (
                        output_text,
                        padded_lists,
                        sequence_length,
                        finish_reason,
                        response,
                    ) = self.create_response(output, length_penalty)
                    response_sender.send(response)
                else:
                    last_output = output

            if not stream and last_output is not None:
                output_text, padded_lists, sequence_length, finish_reason, response = (
                    self.create_response(last_output, length_penalty)
                )
                response_sender.send(response)

            finished_time = time.time()
            exec_t = (finished_time - start_time) * 1000
            ctx_t = (
                0
                if first_token_time is None
                else (first_token_time - start_time) * 1000
            )
            gen_speed = int(
                max(1, max(sequence_length) - 1) / (finished_time - start_time)
            )

            self.logger.log_info(
                "req_id {}, \
                input_tokens: {}, \
                sampling_parameters: {}, \
                input_len={}, \
                output_text: {}, \
                output_tokens: {}, \
                output.logprobs: {}, \
                output_len={}, \
                finish_reason: {}, \
                exec_t={:.3f}ms, \
                ctx_t={:.3f}ms, \
                tokens/s={}".format(
                    request_id if req_id is None else req_id,
                    output.input_tokens,
                    sampling_parameters,
                    len(output.input_tokens),
                    output_text,
                    padded_lists,
                    output.logprobs,
                    sequence_length,
                    finish_reason,
                    exec_t,
                    ctx_t,
                    gen_speed,
                )
            )

        except (ValueError, TypeError, RuntimeError, KeyError) as e:
            self.logger.log_error(
                f"Error in generate_per_req: {e}\n{traceback.format_exc()}"
            )
            error = pb_utils.TritonError(f"Error generating stream: {e}")
            response = pb_utils.InferenceResponse(error=error)
            response_sender.send(response)
            raise
        finally:
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            self.ongoing_request_count -= 1

    async def forward_per_req(self, response_sender, request_bytes, req_id):
        self.ongoing_request_count += 1
        try:
            request_id = uuid.uuid4()
            start_time = time.time()

            status, response_bytes = await self.llm_engine.forward(request_bytes, {})

            if response_sender.is_cancelled():
                self.logger.log_warn(f"Request {request_id} cancelled")
                return

            if not status.OK() or response_bytes is None:
                error = pb_utils.TritonError(
                    f"Error in forward request: {status.GetMessage()}")
                response = pb_utils.InferenceResponse(error=error)
                response_sender.send(response)
                return

            # Create a tensor with the msgpack response bytes
            output_tensor = pb_utils.Tensor(
                "forward_response",
                np.array([response_bytes], dtype=np.object_)
            )

            response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor])
            response_sender.send(response)

            finished_time = time.time()
            exec_t = (finished_time - start_time) * 1000

            self.logger.log_info(
                "req_id {}, forward request processed, exec_t={:.3f}ms".format(
                    request_id if req_id is None else req_id,
                    exec_t
                )
            )

        except Exception as e:
            self.logger.log_error(
                f"Error in forward_per_req: {e}\n{traceback.format_exc()}"
            )
            error = pb_utils.TritonError(
                f"Error processing forward request: {e}")
            response = pb_utils.InferenceResponse(error=error)
            response_sender.send(response)
        finally:
            response_sender.send(
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            self.ongoing_request_count -= 1

    async def process_request(self, request):
        response_sender = request.get_response_sender()
        try:
            # Get request type
            request_type = None
            for input_tensor in request.inputs():
                if input_tensor.name() == "request_type":
                    request_type = input_tensor.as_numpy()[
                        0][0].decode("utf-8")
                    break

            if request_type == "forward":
                # Process as forward request
                self.logger.log_info(f"Processing forward request")
                request_bytes = None
                for input_tensor in request.inputs():
                    if input_tensor.name() == "request_bytes":
                        request_bytes = input_tensor.as_numpy()[0][0]
                        break

                if request_bytes is None:
                    self.logger.log_error(
                        "No request_bytes found for forward request")
                    error = pb_utils.TritonError(
                        "No request_bytes found for forward request")
                    response = pb_utils.InferenceResponse(error=error)
                    response_sender.send(response)
                    return

                self.create_task(
                    self.forward_per_req(
                        response_sender, request_bytes, request.request_id()
                    )
                )
            else:
                self.logger.log_info(f"Processing generate request")
                # Default to generate request
                request_dic = parse_input(
                    self.logger, request, self.input_dtypes)
                self.create_task(
                    self.generate_per_req(
                        response_sender, request_dic, request.request_id()
                    )
                )
        except Exception as e:
            self.logger.log_error(
                f"Error in process_request: {e}\n{traceback.format_exc()}")
            error = pb_utils.TritonError(f"Error processing request: {e}")
            response = pb_utils.InferenceResponse(error=error)
            response_sender.send(response)
            response_sender.send(
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def execute(self, requests):
        for request in requests:
            self.create_task(self.process_request(request))
        return None

    def finalize(self):
        self.logger.log_info("Issuing finalize to Ksana_LLM backend")
        self._shutdown_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join()
            self._loop_thread = None
        del self.llm_engine
