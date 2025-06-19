# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import os
import asyncio
from typing import List, Optional, Dict, Tuple, Union, Any, AsyncGenerator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlamaTokenizer,
    VideoLlavaProcessor,
    PreTrainedTokenizerFast,
    GenerationConfig,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config
import libtorch_serving

from .ksana_plugin import KsanaPlugin, PluginConfig
from .arg_utils import EngineArgs
from .processor_op_base import TokenizerProcessorOpBase


# Adjust the max_workers to a reasonable number based on the number of CPUs
model_executor = ThreadPoolExecutor(max_workers=256)


@dataclass
class EndpointConfig:
    endpoint: str = "python"  # endpoint type
    host: str = "0.0.0.0"  # endpoint host address
    port: int = 8080  # endpoint port
    access_log: bool = True  # whether to enable the endpoint access log


class PyAsyncStreamingIterator:
    """The streaming iterator."""

    def __init__(
        self,
        serving_iterator: libtorch_serving.StreamingIterator,
        ksana_plugin: KsanaPlugin,
        ksana_python_input: libtorch_serving.KsanaPythonInput,
    ):
        self._serving_iterator = serving_iterator
        self._ksana_plugin = ksana_plugin
        self._ksana_python_input = ksana_python_input

    def __aiter__(self):
        return self

    # Define an asynchronous iterator method
    async def __anext__(self):
        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Run the GetNext method of the serving iterator in an executor
        status, ksana_python_output = await loop.run_in_executor(
            model_executor, self._serving_iterator.GetNext
        )

        # Check the status of the iteration
        if status.OK():
            # If the iteration is successful, return the output
            self._ksana_plugin.postprocess(
                self._ksana_python_input, ksana_python_output
            )
            return ksana_python_output
        elif status.GetCode() == libtorch_serving.RetCode.RET_STOP_ITERATION:
            # If the iteration has finished, raise a StopAsyncIteration exception
            raise StopAsyncIteration(
                f"Iterator finished, ret code {status.GetCode()}, message {status.GetMessage()}."
            )
        else:
            # If an error occurred during iteration, raise a RuntimeError exception
            raise RuntimeError(
                f"Iterator error, ret code {status.GetCode()}, message {status.GetMessage()}."
            )


class KsanaLLMEngine:
    """The LLM serving model instance."""

    def __init__(
        self,
        config_file: str,
        tokenizer_path: str,
        plugin_config: PluginConfig,
        endpoint_config: EndpointConfig,
        pre_post_processor: TokenizerProcessorOpBase,
    ):
        """Initialize a KsanaLLMEngine instance.

        Args:
            config_file: The serving config file.
            tokenizer_path: Directory containing tokenizer files.
            plugin_config: Configuration for the Ksana plugin.
            endpoint_config: Configuration for the endpoint.
        """
        self._config_file = config_file

        # The serving instance.
        self._serving = libtorch_serving.Serving()

        self._ksana_plugin = KsanaPlugin(plugin_config)

        if endpoint_config.endpoint != "python":
            self._serving.endpoint_config.type = libtorch_serving.EndpointType.RPC
            self._serving.endpoint_config.rpc_plugin_name = endpoint_config.endpoint

        self._serving.endpoint_config.host = endpoint_config.host
        self._serving.endpoint_config.port = endpoint_config.port
        self._serving.endpoint_config.access_log = endpoint_config.access_log

        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.pre_post_processor = pre_post_processor

        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            print(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead."
            )

        self.eos_token_id = None
        if hasattr(self.tokenizer, "eos_token_id"):
            self.eos_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _load_tokenizer(tokenizer_path: str):
        tokenizer_config = get_tokenizer_config(tokenizer_path)
        if tokenizer_config.get("processor_class", "") == "VideoLlavaProcessor":
            return VideoLlavaProcessor.from_pretrained(tokenizer_path)
        if tokenizer_config.get("tokenizer_class", "") == "LlamaTokenizer":
            return LlamaTokenizer.from_pretrained(tokenizer_path)
        if tokenizer_config.get("processor_class", "") == "Llama4Processor" \
            and tokenizer_config.get("tokenizer_class", "") == "PreTrainedTokenizer":
            return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        if os.path.exists(os.path.join(tokenizer_path, "preprocessor_config.json")):
            return AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def initialize(self) -> None:
        """Initialize the model."""
        self._serving.init_serving(self._config_file)

    @classmethod
    def from_engine_args(cls, args: EngineArgs) -> "KsanaLLMEngine":
        """Create an instance from EngineArgs."""
        if args.config_file is None:
            raise ValueError("config_file is required.")
        plugin_config = PluginConfig(
            model_dir=args.model_dir,
            config_file=args.config_file,
            plugin_name=args.model_type,
            enable_trt=args.plugin_model_enable_trt,
            thread_pool_size=args.plugin_thread_pool_size,
            vit_model_type=args.multi_modal_vit_model_type
        )

        endpoint_config = EndpointConfig(
            endpoint=args.endpoint,
            host=args.host,
            port=args.port,
            access_log=args.access_log,
        )

        tokenizer_path = args.tokenizer_path or args.model_dir

        pre_post_processor = TokenizerProcessorOpBase(
            args.model_dir,
            args.tokenizer_path,
            args.tokenization
        )

        return cls(
            config_file=args.config_file,
            tokenizer_path=tokenizer_path,
            plugin_config=plugin_config,
            endpoint_config=endpoint_config,
            pre_post_processor=pre_post_processor,
        )


    def _build_generator_config(self, sampling_config: dict) -> GenerationConfig:
        """
        Build the transformers GenerationConfig object Validate the 
        values of the attributes.
        """
        def _get_sampling_value(sampling_config: dict, key: str, default_val=None):
            """Get value from sampling_config dict, return default if key not exists.
            """
            return sampling_config[key] if key in sampling_config else default_val

        stop_token_ids = _get_sampling_value(sampling_config, "stop_token_ids", None)
        if not stop_token_ids:
            stop_token_ids = self.pre_post_processor.get_stop_token_ids()
        ignore_eos = _get_sampling_value(sampling_config, "ignore_eos", False)

        if (
            self.pre_post_processor.get_tokenizer_stop_token_ids() is not None
            and not ignore_eos
        ):
            eos_token_id = self.pre_post_processor.get_tokenizer_stop_token_ids()
            if isinstance(eos_token_id, list):
                for token_id in eos_token_id:
                    if token_id not in stop_token_ids:
                        stop_token_ids.append(token_id)
            else:
                if eos_token_id not in stop_token_ids:
                    stop_token_ids.append(eos_token_id)

        generation_config = GenerationConfig(
            top_k=_get_sampling_value(sampling_config, "topk", 1),
            do_sample=_get_sampling_value(sampling_config, "do_sample", None),
            top_p=_get_sampling_value(sampling_config, "topp", 1.0),
            temperature=_get_sampling_value(sampling_config, "temperature", 1.0),
            logprobs_num=_get_sampling_value(sampling_config, "logprobs", 0),
            repetition_penalty=_get_sampling_value(sampling_config,
                                                "repetition_penalty", 1.0),
            no_repeat_ngram_size=_get_sampling_value(sampling_config,
                                                "no_repeat_ngram_size", 0),
            encoder_no_repeat_ngram_size=_get_sampling_value(sampling_config,
                                                "encoder_no_repeat_ngram_size", 0),
            decoder_no_repeat_ngram_size=_get_sampling_value(sampling_config,
                                                "decoder_no_repeat_ngram_size", 0),
            num_beams=_get_sampling_value(sampling_config,
                                                "num_beams", 1),
            num_return_sequences=_get_sampling_value(sampling_config,
                                                "num_return_sequences", 1),
            length_penalty=_get_sampling_value(sampling_config,
                                                "length_penalty", 1.0),
            stop_strings=_get_sampling_value(sampling_config,
                                                "stop_strings", []),
            stop_token_ids=stop_token_ids,
            ignore_eos=ignore_eos
        )
        if "max_new_tokens" in sampling_config:
            generation_config.max_new_tokens = sampling_config["max_new_tokens"]

        return generation_config

    def _build_python_input(
        self,
        request_dict: Dict[str, Any] = None,
        generation_config: GenerationConfig = None,
    ) -> libtorch_serving.KsanaPythonInput:
        """Build the KsanaPythonInput object."""


        prompt_text: Optional[str] = request_dict.pop("prompt", None)
        input_tokens = request_dict.pop("input_tokens", None)

        # `messages` is the OpenAI Chat Completion API that can contain visual input
        # `additonal_params` are model specific params packed in a dict, e.g.,
        # `max_pixels`, `fps` for imgs and videos
        messages: Optional[List[Dict]] = request_dict.pop("messages", None)
        use_chat_template = request_dict.pop("use_chat_template", False)

        if input_tokens is None and prompt_text:
            model_type = request_dict.get("model_type", "empty")
            prompt_text = self.pre_post_processor.build_prompt(prompt_text, model_type, use_chat_template)
            input_tokens = self.pre_post_processor.encode(prompt_text)

        # Create a KsanaPythonInput object
        ksana_python_input = libtorch_serving.KsanaPythonInput()
        ksana_python_input.model_name = request_dict.pop("model_name", "")

        # Set input tokens to the ksana_python_input
        if input_tokens is not None:
            ksana_python_input.input_tokens = input_tokens

        # Set the generation parameters to the ksana_python_input sampling_config
        sampling_config = ksana_python_input.sampling_config

        def _get_generation_value(
            generation_config: GenerationConfig, key: str, default_val
        ):
            """Get value from generation_config, return default if key not exists.
            """
            value = getattr(generation_config, key, default_val)
            return default_val if value is None else value

        sampling_config.num_beams = _get_generation_value(
            generation_config, "num_beams", 1
        )
        sampling_config.topk = _get_generation_value(generation_config, "top_k", 1)
        sampling_config.topp = _get_generation_value(generation_config, "top_p", 1.0)
        sampling_config.temperature = _get_generation_value(
            generation_config, "temperature", 1.0
        )
        sampling_config.max_new_tokens = _get_generation_value(
            generation_config, "max_new_tokens", -1
        )
        sampling_config.logprobs_num = _get_generation_value(
            generation_config, "logprobs_num", 0
        )
        sampling_config.num_return_sequences = _get_generation_value(
            generation_config, "num_return_sequences", 1
        )
        sampling_config.repetition_penalty = _get_generation_value(
            generation_config, "repetition_penalty", 1.0
        )
        sampling_config.no_repeat_ngram_size = _get_generation_value(
            generation_config, "no_repeat_ngram_size", 0
        )
        sampling_config.encoder_no_repeat_ngram_size = _get_generation_value(
            generation_config, "encoder_no_repeat_ngram_size", 0
        )
        sampling_config.decoder_no_repeat_ngram_size = _get_generation_value(
            generation_config, "decoder_no_repeat_ngram_size", 0
        )
        sampling_config.length_penalty = _get_generation_value(
            generation_config, "length_penalty", 1.0
        )
        sampling_config.stop_token_ids = _get_generation_value(
            generation_config, "stop_token_ids", []
        )
        sampling_config.ignore_eos = _get_generation_value(
            generation_config, "ignore_eos", False
        )
        sampling_config.stop_strings = _get_generation_value(
            generation_config, "stop_strings", []
        )

        def _check_do_sample_params():
            do_sample = True
            if (
                _get_generation_value(generation_config, "do_sample", None) is False
                or sampling_config.topk == 1
            ):
                do_sample = False

            if (
                sampling_config.topk == 1
                and _get_generation_value(generation_config, "do_sample", None) is True
            ):
                print(
                    "Generation parameter topk cannot be 1 when do_sample is explicitly set to True!"
                )

            if not do_sample:
                sampling_config.topk = 1
                sampling_config.topp = 1.0
                sampling_config.temperature = 1.0

        _check_do_sample_params()

        # Set additional parameters with the request_dict
        if "input_refit_embedding" in request_dict:
            input_refit_embedding = request_dict["input_refit_embedding"]
            if "pos" in input_refit_embedding:
                ksana_python_input.input_refit_embedding.pos = input_refit_embedding[
                    "pos"
                ]
            if "embeddings" in input_refit_embedding:
                ksana_python_input.input_refit_embedding.embeddings = (
                    input_refit_embedding["embeddings"]
                )
        if "structured_output_regex" in request_dict:
            ksana_python_input.structured_output_regex = request_dict[
                "structured_output_regex"
            ]

        plugin_kwargs = {
            "messages": messages,
            "additional_params": request_dict.get("additional_params", {}),
            "prompt": prompt_text,
        }

        # First exec preprocessing of ksana plugin before inference
        self._ksana_plugin.preprocess(ksana_python_input, **plugin_kwargs)

        return ksana_python_input

    def _call_serving_generate(self, ksana_python_input, req_ctx):
        return self._serving.generate(ksana_python_input, req_ctx)

    def _call_serving_generate_streaming(self, ksana_python_input, req_ctx):
        return self._serving.generate_streaming(ksana_python_input, req_ctx)

    def _call_serving_forward(self, request_bytes, req_ctx):
        return self._serving.forward(request_bytes, req_ctx)

    async def generate(
        self,
        request_dict: Dict[str, Any],
        model_name: Optional[str] = None,
        streamer: Optional[bool] = None,
        req_ctx: Optional[Dict[str, str]] = None,
    ) -> Tuple[Any, Union[Dict[str, Any], AsyncGenerator]]:
        """The model generate interface."""
        if req_ctx is None:
            req_ctx = {}

        # Build a  transformers generation_config from the request_dict 
        # used to validate the values
        generation_config = self._build_generator_config(request_dict.get("sampling_config", {}))

        # Build the ksana_python_input object
        ksana_python_input = self._build_python_input(request_dict, generation_config)
        loop = asyncio.get_running_loop()
        if streamer:
            status, streaming_iterator = await loop.run_in_executor(
                model_executor,
                self._call_serving_generate_streaming,
                ksana_python_input,
                req_ctx
            )

            if not status.OK():
                return status, None

            return status, PyAsyncStreamingIterator(
                streaming_iterator, self._ksana_plugin, ksana_python_input
            )

        status, ksana_python_output = await loop.run_in_executor(
            model_executor,
            self._call_serving_generate,
            ksana_python_input,
            req_ctx
        )
        if not status.OK():
            return status, None
        self._ksana_plugin.postprocess(ksana_python_input, ksana_python_output)

        return status, ksana_python_output

    async def forward(
        self, request_bytes: bytes, req_ctx: Optional[Dict[str, str]] = None
    ) -> Tuple[Any, Optional[bytes]]:
        """The model forward interface."""
        if req_ctx is None:
            req_ctx = {}
        loop = asyncio.get_running_loop()
        status, response_bytes = await loop.run_in_executor(
            model_executor,
            self._call_serving_forward,
            request_bytes,
            req_ctx
        )
        if status.OK():
            return status, response_bytes
        else:
            return status, None
