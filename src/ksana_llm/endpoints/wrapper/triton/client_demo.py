#!/usr/bin/env python3

import argparse
import os
from functools import partial
import time
import json
import numpy as np
import chardet
import google.protobuf.json_format
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.grpc.service_pb2 import ModelInferResponse
import orjson

from LLMTextHandler.tritonft.utils import prepare_tensor
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlamaTokenizer,
    VideoLlavaProcessor,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config

TOKENIZER_DIR = "/model/qwen-hf/Qwen2-0.5B-Instruct"


def get_handler(tokenizer_type):
    tokenizer_config = get_tokenizer_config(TOKENIZER_DIR)
    if tokenizer_config.get("processor_class", "") == "VideoLlavaProcessor":
        return VideoLlavaProcessor.from_pretrained(TOKENIZER_DIR)
    if tokenizer_config.get("tokenizer_class", "") == "LlamaTokenizer":
        return LlamaTokenizer.from_pretrained(TOKENIZER_DIR)

    if os.path.exists(os.path.join(TOKENIZER_DIR, "preprocessor_config.json")):
        return AutoProcessor.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)


class ModelConfigurator:
    def __init__(self, client, model_name):
        if isinstance(client, httpclient.InferenceServerClient):
            self.model_config = client.get_model_config(model_name)
        elif isinstance(client, grpcclient.InferenceServerClient):
            self.model_config = client.get_model_config(model_name, as_json=True)[
                "config"
            ]
        else:
            assert (
                False
            ), "Unkown triton client type, only support httpclient/grpcclient now"

        self.inputs = set()
        for input_dict in self.model_config["input"]:
            self.inputs.add(input_dict["name"])

    def is_decoupled(self) -> bool:
        policy_dict = self.model_config["model_transaction_policy"]
        if "decoupled" not in policy_dict:
            return False
        return policy_dict["decoupled"]

    def has_input(self, input_name) -> bool:
        return input_name in self.inputs


def has_output_diamond_question_mark(s):
    """Checked if 's' is an undefined variable in the code."""
    s_bytes = s.encode("utf-8")
    encoding = chardet.detect(s_bytes)["encoding"]
    if encoding is None:
        return False
    try:
        s_decoded = s_bytes.decode(encoding)
        if "�" in s_decoded:
            return True
        else:
            return False
    except UnicodeDecodeError:
        return True


def prepare_inputs(
    client,
    model_name,
    batch_size,
    eos_token_id,
    output_seq_len=512,
    beam_size=2,
    top_k=40,
    top_p=1.0,
    random_seed=1234,
    diversity_rate=0.0,
    temperature=1.0,
    len_penalty=0.0,
    repetition_penalty=1.0,
    min_length=0,
    bad_words_list=None,
    stop_words_list=None,
    return_log_probs=False,
    vocab_size=0,
    logit_bias=None,
    is_streaming=None,
    draft_logits=None,
    frequency_penalty=None,
    return_context_logits=False,
    return_generation_logits=False,
    input_refit_embedding=None,
):
    model_config = ModelConfigurator(client, model_name)

    if isinstance(client, httpclient.InferenceServerClient):
        client_type = httpclient
        is_streaming = False
    elif isinstance(client, grpcclient.InferenceServerClient):
        client_type = grpcclient
        if is_streaming is None:
            is_streaming = model_config.is_decoupled()
    else:
        assert (
            False
        ), "Unkown triton client type, only support httpclient/grpcclient now"

    if return_context_logits or return_generation_logits:
        assert (
            not is_streaming
        ), "Return context/generation logits is not supported with streaming mode"

    runtime_request_seq_len = (output_seq_len * np.ones([batch_size, 1])).astype(np.uint32)
    runtime_end_ids = eos_token_id * np.ones([batch_size, 1]).astype(np.uint32)
    runtime_beam_width = (beam_size * np.ones([batch_size, 1])).astype(np.uint32)
    runtime_num_return_sequences = (beam_size * np.ones([batch_size, 1])).astype(np.uint32)
    runtime_temperature = temperature * np.ones([batch_size, 1]).astype(np.float32)
    runtime_top_k = (top_k * np.ones([batch_size, 1])).astype(np.uint32)
    runtime_top_p = top_p * np.ones([batch_size, 1]).astype(np.float32)
    runtime_len_penalty = len_penalty * np.ones([batch_size, 1]).astype(np.float32)
    runtime_repetition_penalty = repetition_penalty * np.ones([batch_size, 1]).astype(
        np.float32
    )
    runtime_request_min_length = (min_length * np.ones([batch_size, 1])).astype(np.uint32)
    runtime_random_seed = random_seed * np.ones([batch_size, 1]).astype(np.uint64)
    runtime_streaming = is_streaming * np.ones([batch_size, 1]).astype(bool)
    runtime_return_log_probs = return_log_probs * np.ones([batch_size, 1]).astype(bool)
    runtime_stop_token_ids = np.array([eos_token_id]).astype(np.uint32) * np.ones(
        [batch_size, 1]
    ).astype(np.uint32)

    inputs = [
        prepare_tensor(client_type, "request_output_len", runtime_request_seq_len),
        prepare_tensor(client_type, "end_id", runtime_end_ids),
        prepare_tensor(client_type, "beam_width", runtime_beam_width),
        prepare_tensor(client_type, "num_return_sequences", runtime_num_return_sequences),
        prepare_tensor(client_type, "temperature", runtime_temperature),
        prepare_tensor(client_type, "runtime_top_k", runtime_top_k),
        prepare_tensor(client_type, "runtime_top_p", runtime_top_p),
        prepare_tensor(client_type, "len_penalty", runtime_len_penalty),
        prepare_tensor(client_type, "repetition_penalty", runtime_repetition_penalty),
        prepare_tensor(client_type, "min_length", runtime_request_min_length),
        prepare_tensor(client_type, "random_seed", runtime_random_seed),
        prepare_tensor(client_type, "streaming", runtime_streaming),
        prepare_tensor(client_type, "return_log_probs", runtime_return_log_probs),
        prepare_tensor(client_type, "stop_token_ids", runtime_stop_token_ids),
    ]

    if model_config.has_input("bad_words_list"):
        if bad_words_list is not None and len(bad_words_list) > 0:
            runtime_bad_words_ids = np.array(
                [bad_words_list] * batch_size, dtype=np.int32
            )
            inputs.append(
                prepare_tensor(client_type, "bad_words_list", runtime_bad_words_ids)
            )
    if stop_words_list is not None and len(stop_words_list) > 0:
        runtime_stop_words_ids = np.array(
            [stop_words_list] * batch_size, dtype=np.int32
        )
        inputs.append(
            prepare_tensor(client_type, "stop_words_list", runtime_stop_words_ids)
        )

    if logit_bias is not None and len(logit_bias) > 0 and vocab_size > 0:
        embedding_bias = np.zeros([batch_size, vocab_size], dtype=np.float32)
        for bs in range(batch_size):
            for k, v in logit_bias.items():
                embedding_bias[bs][k] = v
            inputs.append(prepare_tensor(client_type, "embedding_bias", embedding_bias))

    if model_config.has_input("input_refit_embedding") and input_refit_embedding is not None:
        # Convert Python dict to JSON string before creating tensor
        input_refit_embedding_json = orjson.dumps(input_refit_embedding).decode('utf-8')
        # wrap the JSON string in a numpy array to create a string tensor
        runtime_input_refit_embedding = np.array([[input_refit_embedding_json]], dtype=object)
        inputs.append(
            prepare_tensor(client_type, "input_refit_embedding", runtime_input_refit_embedding)
        )
    return inputs


def grpc_infer(
    url,
    model_name,
    src_text,
    tokenizer_type="bloom_ptm_for_business_multiturn_c_v2",
    max_input_seq_len=2048,
    output_seq_len=512,
    beam_size=1,
    top_k=40,
    top_p=0.8,
    random_seed=1234,
    diversity_rate=0.0,
    temperature=1.0,
    len_penalty=0.0,
    repetition_penalty=1.0,
    min_length=0,
    request_id="",
    is_stop=False,
    bad_words_list=None,
    stop_words_list=None,
    return_log_probs=False,
    vocab_size=0,
    logit_bias=None,
    stream_mode=None,
    draft_logits=None,
    frequency_penalty=None,
    return_context_logits=False,
    return_generation_logits=False,
    input_refit_embedding=None,
):
    # Early stop
    if is_stop:
        assert (
            len(str(request_id).strip()) > 0
        ), "The stop request id should not be empty"
        print(f"[WARNING]: The inputs will be ignored in early-stop mode")
        with grpcclient.InferenceServerClient(url, verbose=False) as cl:
            inputs = [
                prepare_tensor(
                    grpcclient, "input_ids", np.empty([1, 1]).astype(np.uint32)
                ),
                prepare_tensor(
                    grpcclient, "input_lengths", np.zeros([1, 1]).astype(np.uint32)
                ),
                prepare_tensor(
                    grpcclient, "request_output_len", np.zeros([1, 1]).astype(np.uint32)
                ),
                prepare_tensor(
                    grpcclient, "stop", is_stop * np.ones([1, 1]).astype(bool)
                ),
            ]

            cl.start_stream(
                callback=(
                    lambda result, error: (
                        print(error)
                        if error
                        else print(f"Request {request_id} stop success")
                    )
                )
            )
            cl.async_stream_infer(model_name, inputs, request_id=str(request_id))
        # nothing need to return in early stop mode
        return ""

    tokenizer = get_handler(tokenizer_type)
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    token_ids = tokenizer.encode(src_text, add_special_tokens=True)
    if len(token_ids) > max_input_seq_len:
        token_ids = token_ids[:max_input_seq_len]
    input_len = len(token_ids)
    input_ids = np.array([token_ids]).astype(np.uint32)
    input_seq_len = np.array([input_len]).reshape(-1, 1).astype(np.uint32)
    output_text_list = []
    tmp_ids_list = []

    def stream_callback(start_time, output_text_list, tmp_ids_list, result, error):
        if error:
            print(error, flush=True)
        else:
            is_first_token = False
            if len(output_text_list) == 0:
                first_token_cost = time.perf_counter() - start_time
                is_first_token = True

            # Prepare result format
            result_json = result.get_response(as_json=True)
            message = ModelInferResponse()
            google.protobuf.json_format.Parse(json.dumps(result_json), message)

            # Get result
            result = grpcclient.InferResult(message)
            output_ids = result.as_numpy("output_ids")
            output_len = result.as_numpy("sequence_length")
            cum_log_probs = result.as_numpy("cum_log_probs")
            max_index = (
                cum_log_probs[0].argmax() if cum_log_probs[0] is not None else 0
            )
            start_index = 0
            end_index = output_len[0, max_index] if output_len is not None else None
            left_ids = output_ids[0, max_index, start_index:end_index].tolist()

            if eos_token_id in left_ids:
                left_ids = left_ids[: left_ids.index(eos_token_id)]
            if pad_token_id in left_ids:
                left_ids = left_ids[: left_ids.index(pad_token_id)]

            # Add pre ids to decode
            if len(tmp_ids_list) > 0:
                for ids in reversed(tmp_ids_list):
                    left_ids = ids + left_ids
                tmp_ids_list.clear()

            # Decode ids to text
            output_text = tokenizer.decode(left_ids, skip_special_tokens=True).rstrip(
                "\ufffd"
            )

            # Check output diamond
            if has_output_diamond_question_mark(output_text):
                tmp_ids_list.append(left_ids)
                return

            # output_text_list  used to record the last result
            prev_text = ""
            if len(output_text_list) > 0:
                prev_text = output_text_list[0]
                output_text_list.pop()
            output_text_list.append(prev_text + output_text)

            print(output_text, end="", flush=True)

            context_logits = result.as_numpy("context_logits")
            if return_context_logits:
                print(f"context_logits {context_logits}", flush=True)

            # save the infomation by specify `-l debug.log`
            if is_first_token:
                print(f"context_cost {first_token_cost:.2f}s")

            # save the infomation by specify `-l debug.log`
            if end_index is not None:
                output_len = end_index - start_index

    print("Output<:", flush=True)
    with grpcclient.InferenceServerClient(url, verbose=False) as cl:
        text_input = np.array([[src_text]]).astype(object)
        input_ids = input_ids.astype(np.uint32)
        input_seq_len = input_seq_len.astype(np.uint32)

        inputs = [
            prepare_tensor(grpcclient, "text_input", text_input),
            prepare_tensor(grpcclient, "input_ids", input_ids),
            prepare_tensor(grpcclient, "input_lengths", input_seq_len),
        ]

        inputs += prepare_inputs(
            cl,
            model_name,
            input_ids.shape[0],
            eos_token_id,
            output_seq_len,
            beam_size,
            top_k,
            top_p,
            random_seed,
            diversity_rate,
            temperature,
            len_penalty,
            repetition_penalty,
            min_length,
            bad_words_list=bad_words_list,
            stop_words_list=stop_words_list,
            return_log_probs=return_log_probs,
            vocab_size=vocab_size,
            logit_bias=logit_bias,
            is_streaming=stream_mode,
            draft_logits=draft_logits,
            frequency_penalty=frequency_penalty,
            return_context_logits=return_context_logits,
            return_generation_logits=return_generation_logits,
            input_refit_embedding=input_refit_embedding,
        )

        cl.start_stream(
            callback=partial(
                stream_callback, time.perf_counter(), output_text_list, tmp_ids_list
            )
        )
        cl.async_stream_infer(model_name, inputs, request_id=str(request_id))

    print("\n", flush=True)
    return output_text_list[0] if len(output_text_list) > 0 else None



text_list = [
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n海水为什么是咸的?<|im_end|>\n"
    "<|im_start|>assistant",
]


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--prompt", type=str, default=text_list[0], required=False
    )
    parser.add_argument("-m", "--model", type=str, default="ksana_llm", required=False)
    parser.add_argument("-H", "--host", type=str, default="localhost", required=False)
    parser.add_argument("--http-port", type=int, default=8020, required=False)
    parser.add_argument("--grpc-port", type=int, default=8021, required=False)
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default=os.devnull,
        required=False,
        help="file to save stderr log info",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="bloom_ptm_for_business_multiturn_c_v2",
        required=False,
    )
    parser.add_argument(
        "--is_stop",
        action="store_true",
        default=False,
        required=False,
        help="send a early stop request, must specify the request_id to stop",
    )
    parser.add_argument("--request_id", type=str, default="", required=False)
    parser.add_argument("--input_len", type=int, default=128, required=False)
    parser.add_argument("--output_len", type=int, default=128, required=False)
    parser.add_argument("--min_length", type=int, default=0, required=False)
    parser.add_argument("--req_protoc", type=str, default="grpc", required=False)
    parser.add_argument("--beam_size", type=int, default=1, required=False)
    parser.add_argument("--top_k", type=int, default=10, required=False)
    parser.add_argument("--top_p", type=float, default=1.0, required=False)
    parser.add_argument("--temperature", type=float, default=1.0, required=False)
    parser.add_argument("--stream", action="store_true", default=False, required=False)
    parser.add_argument("--len_penalty", type=float, default=0.0, required=False)
    parser.add_argument("--repetition_penalty", type=float, default=1.0, required=False)
    parser.add_argument("--end_id", type=int, default=1, required=False)
    parser.add_argument("--input_refit_embedding", type=str, default=None, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()

    args.request_id = args.request_id.strip()
    if not args.is_stop and len(args.request_id) == 0:
        args.request_id = str(np.random.randint(1, np.iinfo(np.int32).max))

    print(f"Input>:\n{args.prompt}", flush=True)
    grpc_infer(
        f"{args.host}:{args.grpc_port}",
        args.model,
        args.prompt,
        tokenizer_type=args.tokenizer,
        request_id=args.request_id,
        is_stop=args.is_stop,
        max_input_seq_len=args.input_len,
        output_seq_len=args.output_len,
        min_length=args.min_length,
        beam_size=args.beam_size,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        len_penalty=args.len_penalty,
        repetition_penalty=args.repetition_penalty,
        stream_mode=args.stream,
        input_refit_embedding=args.input_refit_embedding,
    )
