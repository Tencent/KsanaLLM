# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
# ==============================================================================
# Adapted from
# [vLLM Project]
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/protocol.py
# ==============================================================================
"""
OpenAI API Protocol
Ref: https://platform.openai.com/docs/api-reference
"""

import time
from typing import Annotated, Any, ClassVar, Literal, Optional, Union, List, Dict
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from utilize.utils import random_uuid
from openaiapi.transformers_utils.chat_utils import (ChatCompletionMessageParam,
                                                     make_tool_call_id)

_LONG_INFO = torch.iinfo(torch.long)


class OpenAIBaseModel(BaseModel):
    """OpenAI API基础模型 - 允许额外字段"""
    model_config = ConfigDict(
        extra="allow",
        protected_namespaces=(),
        exclude_none=True
    )

    # 缓存类字段名
    field_names: ClassVar[Optional[set[str]]] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
        """记录额外字段的验证器"""
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            # 获取所有类字段名及其别名
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # 比较字段名和别名
        if any(k not in field_names for k in data):
            from utilize.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(
                "The following fields were present in the request "
                "but ignored: %s",
                data.keys() - field_names,
            )
        return result


class OpenAIRequestModel(OpenAIBaseModel):
    # KsanaLLM 通用扩展参数
    input_tokens: Optional[List[int]] = Field(None, description="预处理的 token 列表")
    num_beams: Optional[int] = Field(None, ge=1, description="束搜索的束宽度")
    num_return_sequences: Optional[int] = Field(None, ge=1, description="返回序列数量")
    no_repeat_ngram_size: Optional[int] = None
    encoder_no_repeat_ngram_size: Optional[int] = None
    decoder_no_repeat_ngram_size: Optional[int] = None
    stop_strings: Optional[List[str]] = None
    structured_output_regex: Optional[str] = None
    model_type: Optional[str] = None
    use_chat_template: Optional[bool] = None
    do_sample: Optional[bool] = None


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True 
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ksana-llm"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class PromptTokenUsageInfo(BaseModel):
    cached_tokens: Optional[int] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None


class RequestResponseMetadata(BaseModel):
    request_id: str
    final_usage_info: Optional[UsageInfo] = None


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # use alias to workaround pydantic conflict
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)
    strict: Optional[bool] = False


class StructuralTag(BaseModel):
    begin: str
    # schema is the field, but that causes conflicts with pydantic so
    # instead use structural_tag_schema with an alias
    structural_tag_schema: Optional[dict[str, Any]] = Field(default=None,
                                                            alias="schema")
    end: str


class StructuralTagResponseFormat(BaseModel):
    type: Literal["structural_tag"]
    structures: list[StructuralTag]
    triggers: list[str]


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


AnyResponseFormat = Union[ResponseFormat, StructuralTagResponseFormat]


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = False


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(default_factory=make_tool_call_id)
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: bool = False


class ChatCompletionToolsParam(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(BaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(BaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class LogitsProcessorConstructor(BaseModel):
    qualname: str
    args: Optional[list[Any]] = None
    kwargs: Optional[dict[str, Any]] = None


LogitsProcessors = list[Union[str, LogitsProcessorConstructor]]


class ChatMessage(BaseModel):
    role: str
    reasoning_content: Optional[str] = None
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = Field(default_factory=list)
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(OpenAIRequestModel):
    # 按照官方OpenAI API文档排序
    messages: List[ChatCompletionMessageParam]
    model: Optional[str] = None
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
        description="The maximum number of tokens that can be generated in the chat completion. ",
    )
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    # This is a KsanaLLM specific extension, used to control structured output option
    enable_structured_output: bool = False 
    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: Optional[Union[str, List[str]]] = []
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[
        Literal["none"],
        Literal["auto"],
        Literal["required"],
        ChatCompletionNamedToolChoiceParam,
    ]] = "none"
  
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    min_tokens: int = 0

    repetition_penalty: Optional[float] = None
    stop_token_ids: Optional[List[int]] = []
    no_stop_trim: bool = False
    length_penalty: float = 1.0
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    continue_final_message: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    prompt_logprobs: Optional[int] = Field(None, ge=0)

    # doc: begin-chat-completion-extra-params
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[list[dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        """验证流式选项"""
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError("Stream options can only be defined when `stream=True`.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        """检查logprobs参数"""
        if (prompt_logprobs := data.get("prompt_logprobs")) is not None:
            if data.get("stream") and prompt_logprobs > 0:
                raise ValueError("`prompt_logprobs` are not available when `stream=True`.")
            if prompt_logprobs < 0:
                raise ValueError("`prompt_logprobs` must be a positive value.")

        if (top_logprobs := data.get("top_logprobs")) is not None:
            if top_logprobs < 0:
                raise ValueError("`top_logprobs` must be a positive value.")
            if top_logprobs > 0 and not data.get("logprobs"):
                raise ValueError("when using `top_logprobs`, `logprobs` must be set to true.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_tool_usage(cls, data):
        # if "tool_choice" is not specified but tools are provided,
        # default to "auto" tool_choice
        if "tool_choice" not in data and data.get("tools"):
            data["tool_choice"] = "auto"

        # if "tool_choice" is "none" -- no validation is needed for tools
        if "tool_choice" in data and data["tool_choice"] == "none":
            return data

        # if "tool_choice" is specified -- validation
        if "tool_choice" in data and data["tool_choice"] is not None:

            # ensure that if "tool choice" is specified, tools are present
            if "tools" not in data or data["tools"] is None:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")

            # make sure that tool choice is either a named tool
            # OR that it's set to "auto" or "required"
            if data["tool_choice"] not in [
                    "auto", "required"
            ] and not isinstance(data["tool_choice"], dict):
                raise ValueError(
                    f'Invalid value for `tool_choice`: {data["tool_choice"]}! '\
                    'Only named tools, "none", "auto" or "required" '\
                    'are supported.'
                )

            # if tool_choice is "required" but the "tools" list is empty,
            # override the data to behave like "none" to align with
            # OpenAI’s behavior.
            if data["tool_choice"] == "required" and isinstance(
                    data["tools"], list) and len(data["tools"]) == 0:
                data["tool_choice"] = "none"
                del data["tools"]
                return data

            # ensure that if "tool_choice" is specified as an object,
            # it matches a valid tool
            correct_usage_message = 'Correct usage: `{"type": "function",' \
                ' "function": {"name": "my_function"}}`'
            if isinstance(data["tool_choice"], dict):
                valid_tool = False
                function = data["tool_choice"].get("function")
                if not isinstance(function, dict):
                    raise ValueError(
                        f"Invalid value for `function`: `{function}` in "
                        f"`tool_choice`! {correct_usage_message}")
                if "name" not in function:
                    raise ValueError(f"Expected field `name` in `function` in "
                                     f"`tool_choice`! {correct_usage_message}")
                function_name = function["name"]
                if not isinstance(function_name,
                                  str) or len(function_name) == 0:
                    raise ValueError(
                        f"Invalid `name` in `function`: `{function_name}`"
                        f" in `tool_choice`! {correct_usage_message}")
                for tool in data["tools"]:
                    if (not isinstance(tool, dict) or 
                            "function" not in tool or
                            not isinstance(tool["function"], dict) or
                            "name" not in tool["function"]):
                            continue
                    if tool["function"]["name"] == function_name:
                        valid_tool = True
                        break
                if not valid_tool:
                    raise ValueError(
                        "The tool specified in `tool_choice` does not match any"
                        " of the specified `tools`")
        return data


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class DeltaToolCall(BaseModel):
    id: str = Field(default_factory=make_tool_call_id)
    type: Literal["function"] = "function"
    index: int
    function: Optional[DeltaFunctionCall] = None


class ExtractedToolCallInformation(BaseModel):
    # indicate if tools were called
    tools_called: bool

    # extracted tool calls
    tool_calls: list[ToolCall]

    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: Optional[str] = None


class TopLogprob(BaseModel):
    token: str
    bytes: List[int]
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    bytes: List[int]
    logprob: float
    top_logprobs: List[TopLogprob]


class ChoiceLogprobs(BaseModel):
    content: List[ChatCompletionTokenLogprob]


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChoiceLogprobs] = None
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChoiceLogprobs] = None
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class CompletionRequest(OpenAIRequestModel):
    model: Optional[str] = None
    prompt: Optional[Union[List[int], List[List[int]], str, List[str]]] = None
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: Optional[Union[str, List[str]]] = []
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    # KsanaLLM
    use_beam_search: bool = False
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    length_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = []
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    prompt_logprobs: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        """验证流式选项"""
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError("Stream options can only be defined when `stream=True`.")
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_prompt(cls, data):
        """验证prompt参数"""
        if data.get("prompt") is None:
            raise ValueError("At least `prompt` must be set.")
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_sampling_params(cls, data):
        """验证采样参数的合理性"""
        # 验证互斥参数
        temperature = data.get("temperature", 1.0)
        top_p = data.get("top_p")
        if top_p is None:
            top_p = 1.0
        if temperature == 0 and top_p < 1.0:
            raise ValueError("When temperature is 0 (greedy sampling), top_p must be 1.0")
        
        # 验证 best_of 和 n 的关系
        best_of = data.get("best_of")
        n = data.get("n", 1)
        if best_of is not None:
            if best_of < n:
                raise ValueError(f"best_of must be greater than or equal to n, got best_of={best_of}, n={n}")
            if best_of > 1 and data.get("stream", False):
                raise ValueError("Cannot use best_of > 1 with streaming")
        
        # 验证 use_beam_search 相关参数
        if data.get("use_beam_search", False):
            if data.get("temperature", 1.0) != 0:
                raise ValueError("Beam search requires temperature=0")
            if not data.get("num_beams", 1) > 1:
                from utilize.logger import get_logger
                logger = get_logger(__name__)
                logger.warning("use_beam_search is True but num_beams is not set or <= 1")
        
        return data


class CompletionLogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class EmbeddingRequest(OpenAIRequestModel):
    model: Optional[str] = None
    input: Union[List[int], List[List[int]], str, List[str]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=-1)]] = None


class EmbeddingResponseData(BaseModel):
    index: int
    object: str = "embedding"
    embedding: Union[List[float], str]


class EmbeddingResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"embd-{random_uuid()}")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    data: List[EmbeddingResponseData]
    usage: UsageInfo


# ===== Tokenize API =====
# This API is Temporary not supported yet. 


class TokenizeRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    add_special_tokens: bool = Field(
        default=True,
        description="如果为true（默认），将在提示中添加特殊token（如BOS）"
    )


class TokenizeResponse(BaseModel):
    count: int
    max_model_len: int
    tokens: List[int]


class DetokenizeRequest(BaseModel):
    model: Optional[str] = None
    tokens: List[int]


class DetokenizeResponse(BaseModel):
    prompt: str


AnyResponseFormat = ResponseFormat
