# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ref:
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/serving_engine.py
# ==============================================================================
"""
OpenAI API Adapter for KsanaLLM
Actually, this python module is a mixture of Configuration, Request Handling, and OpenAI API Compatibility.
"""

import time
import uuid
import asyncio
from collections.abc import Sequence
from concurrent.futures.thread import ThreadPoolExecutor
from http import HTTPStatus
from typing import Callable, Dict, Optional, Union, Any, TypedDict, Annotated
from enum import Enum
from dataclasses import dataclass
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field
from starlette.datastructures import Headers

from openaiapi.request_converter import RequestConverter

from openaiapi.openai_protocol import (
    ErrorResponse, UsageInfo, 
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    CompletionRequest,
    CompletionResponse, 
    DetokenizeRequest
)


from openaiapi.transformers_utils.chat_utils import (AnyTokenizer, MistralTokenizer,
                                                     ChatCompletionMessageParam,
                                                     ChatTemplateContentFormatOption,
                                                     ConversationMessage,
                                                     load_chat_template,
                                                     resolve_hf_chat_template,
                                                     resolve_mistral_chat_template,
                                                     apply_hf_chat_template,
                                                     apply_mistral_chat_template,
                                                     parse_chat_messages_futures,
                                                     resolve_chat_template_content_format)
from utilize.logger import get_logger
logger = get_logger(__name__)
DEFAULT_MAX_MODEL_LEN = 32768


def make_async(func, executor=None):
    import functools
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        if executor:
            return await loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))
        else:
            return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    
    return async_wrapper


@dataclass
class OpenAIConfig:
    default_model_name: str = "ksana-llm"
    default_temperature: float = 1.0
    default_do_sample: bool = False
    default_top_k: int = 0
    default_top_p: float = 1.0
    default_num_beams: int = 1
    default_num_return_sequences: int = 1
    default_length_penalty: float = 1.0
    default_repetition_penalty: float = 1.0
    default_no_repeat_ngram_size: int = 0
    default_encoder_no_repeat_ngram_size: int = 0
    default_decoder_no_repeat_ngram_size: int = 0
    default_logprobs: int = 0
    default_ignore_eos: bool = False
    token_estimation_factor: float = 4.0

    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    enable_auto_tool_choice: bool = False
    tool_parser_plugin: Optional[str] = None


class TextTokensPrompt(TypedDict):
    prompt: str
    prompt_token_ids: list[int]


class ErrorType(str, Enum):
    BAD_REQUEST_ERROR = "BadRequestError"
    NOT_FOUND_ERROR = "NotFoundError"
    INTERNAL_SERVER_ERROR = "InternalServerError"
    VALIDATION_ERROR = "ValidationError"


CompletionLikeRequest = Union[CompletionRequest, DetokenizeRequest]

ChatLikeRequest = Union[ChatCompletionRequest]

AnyRequest = Union[CompletionLikeRequest, ChatLikeRequest]

RequestPrompt = Union[list[int], str, TextTokensPrompt]

# 保持向后兼容的别名
ChatCompletionChoice = ChatCompletionResponseChoice
ChatCompletionUsage = UsageInfo
ChatCompletionStreamChoice = ChatCompletionResponseStreamChoice


class KsanaOpenAIServing:
    """
    Base class for OpenAI API serving in KsanaLLM.
    """
    
    def __init__(
        self,
        llm_server,
        config: Optional[OpenAIConfig] = None,
        *,
        request_logger: Optional[Any] = None,
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__()
        
        self.llm_server = llm_server
        self.config = config or OpenAIConfig()
        
        self._models = None
        self._models_initialized = False
        
        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids
        
        self.tokenizer = self._initialize_tokenizer()
        self._tokenizer_executor = ThreadPoolExecutor(max_workers=1)
        
        # 异步 tokenization 方法
        self._tokenize_prompt_input_async = make_async(
            self._tokenize_prompt_input, executor=self._tokenizer_executor)
        self._tokenize_prompt_input_or_inputs_async = make_async(
            self._tokenize_prompt_input_or_inputs,
            executor=self._tokenizer_executor)
        
        self._model_info = self._extract_model_info()
    
    def _initialize_tokenizer(self) -> Any:
        """Initialize tokenizer, avoid repeat getting tokenizer"""
        tokenizer = getattr(self.llm_server, 'tokenizer', None)
        if tokenizer is None:
            raise ValueError(
                "Tokenizer not found in llm_server. "
                "The tokenizer is required for OpenAI API compatibility. "
                "Please ensure llm_server has a valid tokenizer attribute."
            )
            
        return tokenizer
    
    @property
    def models(self):
        """延迟初始化模型服务"""
        if not self._models_initialized:
            from openaiapi.serving_models import KsanaOpenAIServingModels
            self._models = KsanaOpenAIServingModels(self.llm_server, self.config)
            self._models_initialized = True
        return self._models
    
    
    def _get_max_model_len(self) -> int:
        engine_args = getattr(self.llm_server, 'engine_args', None)
        if engine_args is None:
            raise ValueError("llm_server.engine_args is not available")
        
        if not hasattr(engine_args, 'max_token_len'):
            raise ValueError("engine_args.max_token_len attribute is not available")
        
        # If max_token_len is None or invalid, try to get from tokenizer
        if engine_args.max_token_len is None or engine_args.max_token_len <= 0:
            if self.tokenizer is not None:
                model_max_length = getattr(self.tokenizer, 'model_max_length', None)
                if model_max_length is not None and model_max_length > 0:
                    # If max_token_len in config_file is invalid, Use model_max_length from tokenizer_config.json
                    logger.info(f"Using model_max_length from tokenizer: {model_max_length}")
                    return model_max_length

            # If both engine_args.max_token_len and tokenizer are unavailable, use default value
            logger.warning(f"Both engine_args.max_token_len and tokenizer.model_max_length are unavailable, "
                          f"using default max_token_len: {DEFAULT_MAX_MODEL_LEN}")
            return DEFAULT_MAX_MODEL_LEN

        return engine_args.max_token_len
    
    def _extract_model_info(self) -> Dict[str, Any]:
    # extract model information from the llm_server
        try:
            engine_args = getattr(self.llm_server, 'engine_args', None)
            model_dir = getattr(engine_args, 'model_dir', None) if engine_args else None
            
            if model_dir:
                model_name = model_dir.rstrip('/').split('/')[-1]
                if model_name.startswith('./'):
                    model_name = model_name[2:]
            else:
                model_name = self.config.default_model_name
            
            return {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ksana-llm",
                "permission": [
                    {
                        "id": f"modelperm-{uuid.uuid4().hex[:8]}",
                        "object": "model_permission",
                        "created": int(time.time()),
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
                "root": model_name,
                "parent": None,
            }
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Failed to extract model info: {e}")
            return self._get_default_model_info()
    
    def _get_default_model_info(self) -> Dict[str, Any]:
        return {
            "id": self.config.default_model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ksana-llm",
            "permission": [ 
                {
                    "id": f"modelperm-{uuid.uuid4().hex[:8]}",
                    "object": "model_permission", 
                    "created": int(time.time()),
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False,
                }
            ],
            "root": self.config.default_model_name,
            "parent": None,
        }

    
    def create_error_response(
        self,
        message: str,
        err_type: str = ErrorType.BAD_REQUEST_ERROR,
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
    ) -> JSONResponse:
        converter = RequestConverter()
        error_response = converter.format_error_response(
            error_message=message,
            error_type=err_type,
            error_code=status_code.value
        )
        return JSONResponse(content=error_response, status_code=status_code.value)
    
    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = ErrorType.BAD_REQUEST_ERROR,
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
    ) -> str:
        converter = RequestConverter()
        return converter.format_streaming_error(
            error_message=message,
            error_type=err_type,
            error_code=status_code.value
        )
    
    async def _check_model(self, request: AnyRequest) -> Optional[Union[JSONResponse, ErrorResponse]]:
        if not request.model:
            return None
        
        if self._is_model_supported(request.model):
            return None
        
        # if model is not exist, return error response
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND
        )

    # check if model is supported.    
    def _is_model_supported(self, model_name: Optional[str]) -> bool:
        if not model_name:
            return True
        return self.models.is_base_model(model_name)
    
    def _get_model_name(self, model_name: Optional[str] = None) -> str:
        if not model_name or model_name == self.config.default_model_name:
            return self._model_info["id"]
        return model_name
    
    def _get_trace_context(self, request: Request) -> Optional[Dict[str, str]]:
        try:
            return self.llm_server.get_trace_context(request)
        except (AttributeError, TypeError) as e:
            logger.debug(f"Failed to get trace context: {e}")
            return None
    
    def _validate_input(
        self,
        request: AnyRequest,
        input_ids: list[int],
        input_text: str
    ) -> TextTokensPrompt:
        self.max_model_len = self._get_max_model_len()
        token_num = len(input_ids)
        
        if hasattr(request, '__class__') and 'Embedding' in request.__class__.__name__:
            if token_num > self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for embedding generation. "
                    f"Please reduce the length of the input."
                )
            return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)
        
        
        # 对于 chat completion 请求
        if isinstance(request, ChatCompletionRequest):
            max_tokens = getattr(request, 'max_new_tokens', None)
        else:
            max_tokens = getattr(request, 'max_tokens', None)
        
        if max_tokens is None:
            if token_num > self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the messages. "
                    f"Please reduce the length of the messages."
                )
        elif token_num + max_tokens > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, you requested "
                f"{max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion."
            )
        
        return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)
    
    def _normalize_prompt_text_to_input(
        self,
        request: AnyRequest,
        tokenizer: Any,
        prompt: str,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
        add_special_tokens: bool,
    ) -> TextTokensPrompt:
        
        """将文本 prompt 标准化为输入格式"""

        if truncate_prompt_tokens is None:
            try:
                if hasattr(tokenizer, 'encode'):
                    encoded = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
                elif hasattr(tokenizer, '__call__'):
                    encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
                    if hasattr(encoded, 'input_ids'):
                        encoded = encoded.input_ids
                else:
                    encoded = list(prompt.encode('utf-8'))
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.warning(f"Tokenization failed: {e}, using character-level fallback")
                encoded = list(prompt.encode('utf-8'))
        else:
            try:
                if hasattr(tokenizer, 'encode'):
                    encoded = tokenizer.encode(
                        prompt,
                        add_special_tokens=add_special_tokens,
                        max_length=truncate_prompt_tokens,
                        truncation=True
                    )
                elif hasattr(tokenizer, '__call__'):
                    encoded = tokenizer(
                        prompt,
                        add_special_tokens=add_special_tokens,
                        truncation=True,
                        max_length=truncate_prompt_tokens
                    )
                    if hasattr(encoded, 'input_ids'):
                        encoded = encoded.input_ids
                else:
                    encoded = list(prompt.encode('utf-8'))[:truncate_prompt_tokens]
            except (AttributeError, TypeError, RuntimeError) as e:
                logger.warning(f"Tokenization with truncation failed: {e}")
                encoded = list(prompt.encode('utf-8'))[:truncate_prompt_tokens or 1024]
        
        input_ids = encoded if isinstance(encoded, list) else encoded.tolist()
        input_text = prompt
        
        return self._validate_input(request, input_ids, input_text)
    
    # normalize prompt tokens to input format
    def _normalize_prompt_tokens_to_input(
        self,
        request: AnyRequest,
        tokenizer: Any,
        prompt_ids: list[int],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
    ) -> TextTokensPrompt:
        if truncate_prompt_tokens is None:
            input_ids = prompt_ids
        else:
            input_ids = prompt_ids[-truncate_prompt_tokens:]
        
        try:
            if hasattr(tokenizer, 'decode'):
                input_text = tokenizer.decode(input_ids)
            else:
                # 简单的字符级别解码作为后备
                input_text = bytes(input_ids).decode('utf-8', errors='ignore')
        except (AttributeError, TypeError, UnicodeDecodeError) as e:
            logger.warning(f"Detokenization failed: {e}")
            input_text = str(input_ids)
        
        return self._validate_input(request, input_ids, input_text)
    
    # Tokenize single prompt input
    def _tokenize_prompt_input(
        self,
        request: AnyRequest,
        tokenizer: Any,
        prompt_input: Union[str, list[int]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> TextTokensPrompt:
        if isinstance(prompt_input, str):
            return self._normalize_prompt_text_to_input(
                request,
                tokenizer,
                prompt=prompt_input,
                truncate_prompt_tokens=truncate_prompt_tokens,
                add_special_tokens=add_special_tokens,
            )
        else:
            return self._normalize_prompt_tokens_to_input(
                request,
                tokenizer,
                prompt_ids=prompt_input,
                truncate_prompt_tokens=truncate_prompt_tokens,
            )
    
    def _tokenize_prompt_input_or_inputs(
        self,
        request: AnyRequest,
        tokenizer: Any,
        input_or_inputs: Union[str, list[str], list[int], list[list[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> list[TextTokensPrompt]:
        if isinstance(input_or_inputs, str):
            return [self._tokenize_prompt_input(
                request, tokenizer, input_or_inputs,
                truncate_prompt_tokens, add_special_tokens
            )]
        
        # Handle Single Token List
        if isinstance(input_or_inputs, list) and len(input_or_inputs) > 0:
            if isinstance(input_or_inputs[0], int):
                return [self._tokenize_prompt_input(
                    request, tokenizer, input_or_inputs,
                    truncate_prompt_tokens, add_special_tokens
                )]
        
        # Handle Multi Token List
        results = []
        for input_item in input_or_inputs:
            result = self._tokenize_prompt_input(
                request, tokenizer, input_item,
                truncate_prompt_tokens, add_special_tokens
            )
            results.append(result)
        
        return results
    
    async def _get_trace_headers(self, headers: Headers) -> Optional[Dict[str, str]]:
        trace_headers = {}
        for key, value in headers.items():
            if key.lower().startswith('x-trace-') or key.lower() in ['x-request-id', 'traceparent']:
                trace_headers[key] = value
        
        return trace_headers if trace_headers else None
    
    @staticmethod
    def _base_request_id(raw_request: Optional[Request], default: Optional[str] = None) -> Optional[str]:
        # Get Request id
        if default is None:
            default = f"req-{uuid.uuid4().hex}"
        
        if raw_request is None:
            return default
        
        return raw_request.headers.get("X-Request-Id", default)
    
    def _log_inputs(
        self,
        request_id: str,
        inputs: RequestPrompt,
        params: Optional[Any],
    ) -> None:
        if self.request_logger is None:
            return
        
        try:
            if isinstance(inputs, str):
                prompt = inputs
                prompt_token_ids = None
            elif isinstance(inputs, list):
                prompt = None
                prompt_token_ids = inputs
            else:
                prompt = inputs.get("prompt")
                prompt_token_ids = inputs.get("prompt_token_ids")
            
            logger.debug(f"Request {request_id}: prompt_len={len(prompt) if prompt else 0}, "
                        f"token_len={len(prompt_token_ids) if prompt_token_ids else 0}")
        except (AttributeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to log inputs: {e}")
    
    @staticmethod
    def _get_decoded_token(logprob: Any,
                           token_id: int,
                           tokenizer: AnyTokenizer,
                           return_as_token_id: bool = False) -> str:
        if return_as_token_id:
            return f"token_id:{token_id}"

        if hasattr(logprob, 'decoded_token') and logprob.decoded_token is not None:
            return logprob.decoded_token
        
        try:
            return tokenizer.decode(token_id)
        except (AttributeError, TypeError, UnicodeDecodeError):
            return f"<token_id:{token_id}>"
    
    async def _preprocess_completion(
        self,
        request: CompletionLikeRequest,
        tokenizer: AnyTokenizer,
        input_or_inputs: Union[str, list[str], list[int], list[list[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> tuple[list[TextTokensPrompt], list[Any]]:
        # preprocess completion request
        request_prompts = await self._tokenize_prompt_input_or_inputs_async(
            request,
            tokenizer,
            input_or_inputs,
            truncate_prompt_tokens=truncate_prompt_tokens,
            add_special_tokens=add_special_tokens,
        )

        engine_prompts = []
        for request_prompt in request_prompts:
            engine_prompt = {
                "prompt_token_ids": request_prompt["prompt_token_ids"]
            }
            engine_prompts.append(engine_prompt)

        return request_prompts, engine_prompts
    
    # preprocess_chat_Completions_request
    async def _preprocess_chat(
        self,
        request: ChatLikeRequest,
        tokenizer: AnyTokenizer,
        messages: list[ChatCompletionMessageParam],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tool_dicts: Optional[list[dict[str, Any]]] = None,
        documents: Optional[list[dict[str, str]]] = None,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
        tool_parser: Optional[Callable[[AnyTokenizer], Any]] = None,
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = False,
    ) -> tuple[list[ConversationMessage], Sequence[RequestPrompt], list[Any]]:
        # Check if messages is empty
        if not messages:
            raise ValueError("Messages cannot be empty. At least one message is required.")
        
        model_config = self.llm_server.engine_args

        resolved_content_format = resolve_chat_template_content_format(
            chat_template,
            tool_dicts,
            chat_template_content_format,
            tokenizer,
            trust_remote_code=getattr(model_config, 'trust_remote_code', True),
        )

        conversation, mm_data_future = parse_chat_messages_futures(
            messages,
            model_config,
            tokenizer,
            content_format=resolved_content_format,
        )

        mm_data_future = None

        _chat_template_kwargs: dict[str, Any] = dict(
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tools=tool_dicts,
            documents=documents,
        )
        _chat_template_kwargs.update(chat_template_kwargs or {})

        request_prompt: Union[str, list[int]]
        if isinstance(tokenizer, MistralTokenizer):
            request_prompt = apply_mistral_chat_template(
                tokenizer,
                messages=messages,
                **_chat_template_kwargs,
            )
        else:
            request_prompt = apply_hf_chat_template(
                tokenizer,
                trust_remote_code=getattr(model_config, 'trust_remote_code', True),
                conversation=conversation,
                **_chat_template_kwargs,
            )

        # TODO(ethanyczeng): Multi Modal Data Support
        # now, it is temporarily set to None
        mm_data = [] if mm_data_future is None else await mm_data_future

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        should_parse_tools = tool_parser is not None and (hasattr(
            request, "tool_choice") and request.tool_choice != "none")

        if should_parse_tools:
            if not isinstance(request, ChatCompletionRequest):
                msg = "Tool usage is only supported for Chat Completions API"
                raise NotImplementedError(msg)
            
            if callable(tool_parser):
                tool_parser_instance = tool_parser(tokenizer)
            else:
                tool_parser_instance = tool_parser
            
            request = tool_parser_instance.adjust_request(request=request)


        if isinstance(request_prompt, str):
            prompt_inputs = await self._tokenize_prompt_input_async(
                request,
                tokenizer,
                request_prompt,
                truncate_prompt_tokens=truncate_prompt_tokens,
                add_special_tokens=add_special_tokens,
            )
        else:
            # For MistralTokenizer
            from utilize.utils import is_list_of
            assert is_list_of(request_prompt, int), (
                "Prompt has to be either a string or a list of token ids")
            prompt_inputs = TextTokensPrompt(
                prompt=tokenizer.decode(request_prompt),
                prompt_token_ids=request_prompt)

        # create engine prompts
        engine_prompt = {
            "prompt_token_ids": prompt_inputs["prompt_token_ids"]
        }
        if mm_data is not None:
            engine_prompt["multi_modal_data"] = mm_data
        if hasattr(request, 'mm_processor_kwargs') and request.mm_processor_kwargs is not None:
            engine_prompt["mm_processor_kwargs"] = request.mm_processor_kwargs

        return conversation, [request_prompt], [engine_prompt]


class OpenAIAdapter:
#OpenAI Adapter - contain all OpenAI compatible APIs.
    
    def __init__(self,
                 llm_server,
                 config: Optional[OpenAIConfig] = None,
                 *,
                 chat_template: Optional[str] = None,
                 chat_template_content_format: Optional[ChatTemplateContentFormatOption] = None):
        self.llm_server = llm_server
        self.config = config or OpenAIConfig()
        
        # Delay Import
        from openaiapi.serving_chat import KsanaOpenAIServingChat
        from openaiapi.serving_completion import KsanaOpenAIServingCompletion
        from openaiapi.serving_embedding import KsanaOpenAIServingEmbedding
        from openaiapi.serving_models import KsanaOpenAIServingModels
        
        self.chat_serving = KsanaOpenAIServingChat(
            llm_server,
            self.config,
            chat_template=chat_template,
            chat_template_content_format=chat_template_content_format
        )
        self.completion_serving = KsanaOpenAIServingCompletion(llm_server, self.config)
        self.embedding_serving = KsanaOpenAIServingEmbedding(llm_server, self.config)
        self.models_serving = KsanaOpenAIServingModels(llm_server, self.config)
    
    # ===== Basic APIs =====
    
    async def chat_completions(
        self,
        request: ChatCompletionRequest,
        http_request: Request
    ) -> Union[JSONResponse, StreamingResponse]:
        """处理Chat Completions请求"""
        result = await self.chat_serving.create_chat_completion(request, http_request)
        
        if isinstance(result, ChatCompletionResponse):
            return JSONResponse(content=result.model_dump(exclude_none=True))
        elif hasattr(result, '__aiter__'):  # AsyncGenerator
            return StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        else:
            return result 
    
    async def completions(self, request: Request) -> Union[JSONResponse, StreamingResponse]:
        """处理Legacy Completions API请求"""
        result = await self.completion_serving.create_completion(request)
        
        if isinstance(result, CompletionResponse):
            return JSONResponse(content=result.model_dump(exclude_none=True))
        elif hasattr(result, '__aiter__'):  # AsyncGenerator
            return StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        else:
            return result  # JSONResponse (error)
    
    async def embeddings(self, request: Request) -> JSONResponse:
    #TODO(ethanyczeng): Embedding API support
        return await self.embedding_serving.create_embedding(request)
    
    def list_models(self) -> JSONResponse:
        return self.models_serving.list_models()
    
    def get_model(self, model_id: str) -> JSONResponse:
        return self.models_serving.get_model(model_id)
    
    def delete_model(self, model_id: str) -> JSONResponse:
        return self.models_serving.delete_model(model_id)
    


def register_all_routes(app, adapter: OpenAIAdapter):
    
    @app.post("/v1/chat/completions")
    async def _chat_completions(request: ChatCompletionRequest, http_request: Request):
        return await adapter.chat_completions(request, http_request)
    
    @app.post("/v1/completions")
    async def _completions(request: Request):
        """处理Legacy Completions API请求"""
        return await adapter.completions(request)
    
    @app.get("/models")
    async def _list_models():
        return adapter.list_models()
    
    @app.get("/models/{model_id}")
    async def _get_model(model_id: str):
        return adapter.get_model(model_id)
    
    # ===== Extended APIs =====
    
    # Embeddings API
    # Temporarily using the same endpoint as completions
    @app.post("/v1/embeddings")
    async def _create_embeddings(request: Request):
        """处理Embeddings API请求"""
        return await adapter.embeddings(request)


def add_openai_routes(llm_server,
                     config: Optional[OpenAIConfig] = None,
                     *,
                     chat_template: Optional[str] = None,
                     chat_template_content_format: Optional[ChatTemplateContentFormatOption] = None):

    resolved_chat_template = load_chat_template(chat_template)
    if resolved_chat_template is not None:
        # Get the tokenizer to check official template
        tokenizer = getattr(llm_server, 'tokenizer', None)

        if isinstance(tokenizer, MistralTokenizer):
            # The warning is logged in resolve_mistral_chat_template.
            resolved_chat_template = resolve_mistral_chat_template(
                chat_template=resolved_chat_template)
        else:
            hf_chat_template = resolve_hf_chat_template(
                tokenizer=tokenizer,
                chat_template=None,
                tools=None,
                trust_remote_code=True,
            )

            if hf_chat_template != resolved_chat_template:
                logger.warning(
                    "Using supplied chat template: %s\n"
                    "It is different from official chat template. "
                    "This discrepancy may lead to performance degradation.",
                    resolved_chat_template)
    
    adapter = OpenAIAdapter(
        llm_server,
        config,
        chat_template=resolved_chat_template,
        chat_template_content_format=chat_template_content_format
    )
    
    register_all_routes(llm_server.app, adapter)
    
    return adapter
