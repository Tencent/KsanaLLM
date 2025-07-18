# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted From vLLM
# Ref:
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/chat_utils.py 
import json
from collections import  deque
from collections.abc import Awaitable, Iterable
from functools import lru_cache, partial
from pathlib import Path
from typing import (Any, Callable, Literal, Optional, TypeVar, Union,
                    cast)

import jinja2.nodes
from transformers import (PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
import transformers.utils.chat_template_utils as hf_chat_utils
# yapf conflicts with isort for this block
# yapf: disable
from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartInputAudioParam)
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam)
from openai.types.chat import (ChatCompletionContentPartRefusalParam,
                               ChatCompletionContentPartTextParam)
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)
from openai.types.chat import (ChatCompletionMessageToolCallParam,
                               ChatCompletionToolMessageParam)

from typing_extensions import Required, TypeAlias, TypedDict

from openaiapi.transformers_utils.tokenizer_base import TokenizerBase
from openaiapi.transformers_utils.tokenizers import MistralTokenizer

from transformers import (PreTrainedTokenizer, PreTrainedTokenizerFast,
                          ProcessorMixin)

from openaiapi.transformers_utils.processor import cached_get_processor
from pydantic import TypeAdapter
from utilize.utils import random_uuid
from utilize.logger import get_logger

logger = get_logger(__name__)

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast,
                     TokenizerBase]


class AudioURL(TypedDict, total=False):
    url: Required[str]
    """
    Either a URL of the audio or a data URL with base64 encoded audio data.
    """


class ChatCompletionContentPartAudioParam(TypedDict, total=False):
    audio_url: Required[AudioURL]

    type: Required[Literal["audio_url"]]
    """The type of the content part."""


class ChatCompletionContentPartImageEmbedsParam(TypedDict, total=False):
    image_embeds: Required[Union[str, dict[str, str]]]
    """
    The image embeddings. It can be either:
    - A single base64 string.
    - A dictionary where each value is a base64 string.
    """
    type: Required[Literal["image_embeds"]]
    """The type of the content part."""


class VideoURL(TypedDict, total=False):
    url: Required[str]
    """
    Either a URL of the video or a data URL with base64 encoded video data.
    """


class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    video_url: Required[VideoURL]

    type: Required[Literal["video_url"]]
    """The type of the content part."""


class CustomChatCompletionContentSimpleImageParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain image_url.
    This is supported by OpenAI API, although it is not documented.

    Example:
    {
        "image_url": "https://example.com/image.jpg"
    }
    """
    image_url: Required[str]


class CustomChatCompletionContentSimpleAudioParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain audio_url.

    Example:
    {
        "audio_url": "https://example.com/audio.mp3"
    }
    """
    audio_url: Required[str]


class CustomChatCompletionContentSimpleVideoParam(TypedDict, total=False):
    """A simpler version of the param that only accepts a plain audio_url.

    Example:
    {
        "video_url": "https://example.com/video.mp4"
    }
    """
    video_url: Required[str]


ChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam, ChatCompletionContentPartAudioParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartVideoParam, ChatCompletionContentPartRefusalParam,
    CustomChatCompletionContentSimpleImageParam,
    ChatCompletionContentPartImageEmbedsParam,
    CustomChatCompletionContentSimpleAudioParam,
    CustomChatCompletionContentSimpleVideoParam, str]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""

    content: Union[str, list[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """

    tool_call_id: Optional[str]
    """Tool call that this message is responding to."""

    tool_calls: Optional[Iterable[ChatCompletionMessageToolCallParam]]
    """The tool calls generated by the model, such as function calls."""


ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam,
                                   CustomChatCompletionMessageParam]

# TODO: Make fields ReadOnly once mypy supports it


class ConversationMessage(TypedDict, total=False):
    role: Required[str]
    """The role of the message's author."""

    content: Union[Optional[str], list[dict[str, str]]]
    """The contents of the message"""

    tool_call_id: Optional[str]
    """Tool call that this message is responding to."""

    name: Optional[str]
    """The name of the function to call"""

    tool_calls: Optional[Iterable[ChatCompletionMessageToolCallParam]]
    """The tool calls generated by the model, such as function calls."""


ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]

_ChatTemplateContentFormat = Literal["string", "openai"]


def _is_var_access(node: jinja2.nodes.Node, varname: str) -> bool:
    if isinstance(node, jinja2.nodes.Name):
        return node.ctx == "load" and node.name == varname

    return False


def _is_attr_access(node: jinja2.nodes.Node, varname: str, key: str) -> bool:
    if isinstance(node, jinja2.nodes.Getitem):
        return (_is_var_access(node.node, varname)
                and isinstance(node.arg, jinja2.nodes.Const)
                and node.arg.value == key)

    if isinstance(node, jinja2.nodes.Getattr):
        return _is_var_access(node.node, varname) and node.attr == key

    return False


def _is_var_or_elems_access(
    node: jinja2.nodes.Node,
    varname: str,
    key: Optional[str] = None,
) -> bool:
    if isinstance(node, jinja2.nodes.Filter):
        return (node.node is not None
                and _is_var_or_elems_access(node.node, varname, key))
    if isinstance(node, jinja2.nodes.Test):
        return _is_var_or_elems_access(node.node, varname, key)

    if (isinstance(node, jinja2.nodes.Getitem)
            and isinstance(node.arg, jinja2.nodes.Slice)):
        return _is_var_or_elems_access(node.node, varname, key)

    # yapf: disable
    return (
        _is_attr_access(node, varname, key) if key
        else _is_var_access(node, varname)
    )    # yapf: enable


def _try_extract_ast(chat_template: str) -> Optional[jinja2.nodes.Template]:
    try:
        jinja_compiled = hf_chat_utils._compile_jinja_template(chat_template)
        return jinja_compiled.environment.parse(chat_template)
    except (jinja2.exceptions.TemplateSyntaxError, jinja2.exceptions.TemplateError, ValueError):
        logger.exception("Error when compiling Jinja template")
        return None


def _iter_nodes_assign_var_or_elems(root: jinja2.nodes.Node, varname: str):
    # Global variable that is implicitly defined at the root
    yield root, varname

    # Iterative BFS
    related_varnames = deque([varname])
    while related_varnames:
        related_varname = related_varnames.popleft()

        for assign_ast in root.find_all(jinja2.nodes.Assign):
            lhs = assign_ast.target
            rhs = assign_ast.node

            if _is_var_or_elems_access(rhs, related_varname):
                assert isinstance(lhs, jinja2.nodes.Name)
                yield assign_ast, lhs.name

                # Avoid infinite looping for self-assignment
                if lhs.name != related_varname:
                    related_varnames.append(lhs.name)


def _iter_nodes_assign_messages_item(root: jinja2.nodes.Node):
    messages_varnames = [
        varname
        for _, varname in _iter_nodes_assign_var_or_elems(root, "messages")
    ]

    # Search for {%- for message in messages -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in messages_varnames:
            if _is_var_or_elems_access(loop_iter, varname):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break


def _iter_nodes_assign_content_item(root: jinja2.nodes.Node):
    message_varnames = [
        varname for _, varname in _iter_nodes_assign_messages_item(root)
    ]

    # Search for {%- for content in message['content'] -%} loops
    for loop_ast in root.find_all(jinja2.nodes.For):
        loop_iter = loop_ast.iter
        loop_target = loop_ast.target

        for varname in message_varnames:
            if _is_var_or_elems_access(loop_iter, varname, "content"):
                assert isinstance(loop_target, jinja2.nodes.Name)
                yield loop_ast, loop_target.name
                break


def _detect_content_format(
    chat_template: str,
    *,
    default: _ChatTemplateContentFormat,
) -> _ChatTemplateContentFormat:
    jinja_ast = _try_extract_ast(chat_template)
    if jinja_ast is None:
        return default

    try:
        next(_iter_nodes_assign_content_item(jinja_ast))
    except StopIteration:
        return "string"
    except (AttributeError, TypeError, RuntimeError):
        logger.exception("Error when parsing AST of Jinja template")
        return default
    else:
        return "openai"


def _load_chat_template(
    chat_template: Optional[Union[Path, str]],
    *,
    is_literal: bool = False,
) -> Optional[str]:
    if chat_template is None:
        return None

    if is_literal:
        if isinstance(chat_template, Path):
            raise TypeError("chat_template is expected to be read directly "
                            "from its value")

        return chat_template

    try:
        with open(chat_template) as f:
            return f.read()
    except OSError as e:
        if isinstance(chat_template, Path):
            raise

        jinjia_chars = "{}\n"
        if not any(c in chat_template for c in jinjia_chars):
            msg = (f"The supplied chat template ({chat_template}) "
                   f"looks like a file path, but it failed to be "
                   f"opened. Reason: {e}")
            raise ValueError(msg) from e

        # If opening a file fails, set chat template to be args to
        # ensure we decode so our escape are interpreted correctly
        return _load_chat_template(chat_template, is_literal=True)

_cached_load_chat_template = lru_cache(_load_chat_template)


def load_chat_template(
    chat_template: Optional[Union[Path, str]],
    *,
    is_literal: bool = False,
) -> Optional[str]:
    return _cached_load_chat_template(chat_template, is_literal=is_literal)


def resolve_mistral_chat_template(
    chat_template: Optional[str],
    **kwargs: Any,
) -> Optional[str]:
    if chat_template is not None:
        logger.warning_once(
            "'chat_template' cannot be overridden for mistral tokenizer.")
    if "add_generation_prompt" in kwargs:
        logger.warning_once(
            "'add_generation_prompt' is not supported for mistral tokenizer, "
            "so it will be ignored.")
    if "continue_final_message" in kwargs:
        logger.warning_once(
            "'continue_final_message' is not supported for mistral tokenizer, "
            "so it will be ignored.")
    return None


def resolve_hf_chat_template(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
    *,
    trust_remote_code: bool,
) -> Optional[str]:
    # 1st priority: The given chat template
    if chat_template is not None:
        return chat_template

    # 2nd priority: AutoProcessor chat template, unless tool calling is enabled
    if tools is None:
        try:
            processor = cached_get_processor(
                tokenizer.name_or_path,
                processor_cls=(PreTrainedTokenizer, PreTrainedTokenizerFast,
                               ProcessorMixin),
                trust_remote_code=trust_remote_code,
            )
            if isinstance(processor, ProcessorMixin) and \
                hasattr(processor, 'chat_template') and \
                processor.chat_template is not None:
                return processor.chat_template
        except (ImportError, AttributeError, ValueError):
            logger.debug("Failed to load AutoProcessor chat template for %s",
                        tokenizer.name_or_path, exc_info=True)

    # 3rd priority: AutoTokenizer chat template
    try:
        return tokenizer.get_chat_template(chat_template, tools=tools)
    except (AttributeError, ValueError, TypeError):
        logger.debug("Failed to load AutoTokenizer chat template for %s",
                     tokenizer.name_or_path, exc_info=True)

    return None


def _resolve_chat_template_content_format(
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
    given_format: ChatTemplateContentFormatOption,
    tokenizer: AnyTokenizer,
    *,
    trust_remote_code: bool,
) -> _ChatTemplateContentFormat:
    if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        hf_chat_template = resolve_hf_chat_template(
            tokenizer,
            chat_template=chat_template,
            trust_remote_code=trust_remote_code,
            tools=tools,
        )
    else:
        hf_chat_template = None

    jinja_text = (hf_chat_template if isinstance(hf_chat_template, str)
                  else load_chat_template(chat_template, is_literal=True))

    detected_format = ("string" if jinja_text is None else
                       _detect_content_format(jinja_text, default="string"))

    return detected_format if given_format == "auto" else given_format


def resolve_chat_template_content_format(
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
    given_format: ChatTemplateContentFormatOption,
    tokenizer: AnyTokenizer,
    *,
    trust_remote_code: bool = False,
) -> _ChatTemplateContentFormat:
    detected_format = _resolve_chat_template_content_format(
        chat_template,
        tools,
        given_format,
        tokenizer,
        trust_remote_code=trust_remote_code,
    )

    return detected_format


ModalityStr = Literal["image", "audio", "video", "image_embeds"]
_T = TypeVar("_T")



# No need to validate using Pydantic again
_TextParser = partial(cast, ChatCompletionContentPartTextParam)
_ImageEmbedsParser = partial(cast, ChatCompletionContentPartImageEmbedsParam)
_InputAudioParser = partial(cast, ChatCompletionContentPartInputAudioParam)
_RefusalParser = partial(cast, ChatCompletionContentPartRefusalParam)
# Need to validate url objects
_ImageParser = TypeAdapter(ChatCompletionContentPartImageParam).validate_python
_AudioParser = TypeAdapter(ChatCompletionContentPartAudioParam).validate_python
_VideoParser = TypeAdapter(ChatCompletionContentPartVideoParam).validate_python

_ContentPart: TypeAlias = Union[str, dict[str, str]]


# Define a mapping from part types to their corresponding parsing functions.
MM_PARSER_MAP: dict[
    str,
    Callable[[ChatCompletionContentPartParam], _ContentPart],
] = {
    "text":
    lambda part: _TextParser(part).get("text", None),
    "image_url":
    lambda part: _ImageParser(part).get("image_url", {}).get("url", None),
    "image_embeds":
    lambda part: _ImageEmbedsParser(part).get("image_embeds", None),
    "audio_url":
    lambda part: _AudioParser(part).get("audio_url", {}).get("url", None),
    "input_audio":
    lambda part: _InputAudioParser(part).get("input_audio", None),
    "refusal":
    lambda part: _RefusalParser(part).get("refusal", None),
    "video_url":
    lambda part: _VideoParser(part).get("video_url", {}).get("url", None),
}


def _parse_chat_message_content_mm_part(
        part: ChatCompletionContentPartParam) -> tuple[str, _ContentPart]:
    """
    Parses a given multi-modal content part based on its type.

    Args:
        part: A dict containing the content part, with a potential 'type' field.

    Returns:
        A tuple (part_type, content) where:
        - part_type: Type of the part (e.g., 'text', 'image_url').
        - content: Parsed content (e.g., text, image URL).

    Raises:
        ValueError: If the 'type' field is missing and no direct URL is found.
    """
    assert isinstance(
        part, dict)  # This is needed to avoid mypy errors: part.get() from str
    part_type = part.get("type", None)

    if isinstance(part_type, str) and part_type in MM_PARSER_MAP:
        content = MM_PARSER_MAP[part_type](part)

        # Special case for 'image_url.detail'
        # We only support 'auto', which is the default
        if part_type == "image_url" and part.get("detail", "auto") != "auto":
            logger.warning("'image_url.detail' is currently not supported "
                           "and will be ignored.")

        return part_type, content

    # Handle missing 'type' but provided direct URL fields.
    # 'type' is required field by pydantic
    if part_type is None:
        if part.get("image_url") is not None:
            image_params = cast(CustomChatCompletionContentSimpleImageParam,
                                part)
            return "image_url", image_params.get("image_url", "")
        if part.get("audio_url") is not None:
            audio_params = cast(CustomChatCompletionContentSimpleAudioParam,
                                part)
            return "audio_url", audio_params.get("audio_url", "")
        if part.get("input_audio") is not None:
            input_audio_params = cast(dict[str, str], part)
            return "input_audio", input_audio_params
        if part.get("video_url") is not None:
            video_params = cast(CustomChatCompletionContentSimpleVideoParam,
                                part)
            return "video_url", video_params.get("video_url", "")
        # Raise an error if no 'type' or direct URL is found.
        raise ValueError("Missing 'type' field in multimodal part.")

    if not isinstance(part_type, str):
        raise ValueError("Invalid 'type' field in multimodal part.")
    return part_type, "unknown part_type content"


VALID_MESSAGE_CONTENT_MM_PART_TYPES = ("text", "refusal", "image_url",
                                       "image_embeds",
                                       "audio_url", "input_audio", "video_url")


def _parse_chat_message_content_part(
    part: ChatCompletionContentPartParam,
    *,
    wrap_dicts: bool,
) -> Optional[_ContentPart]:
    """Parses a single part of a conversation. If wrap_dicts is True,
    structured dictionary pieces for texts and images will be
    wrapped in dictionaries, i.e., {"type": "text", "text", ...} and
    {"type": "image"}, respectively. Otherwise multimodal data will be
    handled by mm_parser, and texts will be returned as strings to be joined
    with multimodal placeholders.
    """
    if isinstance(part, str):  # Handle plain text parts
        return part

    # Handle structured dictionary parts
    part_type, content = _parse_chat_message_content_mm_part(part)


    if part_type in ("text", "refusal"):
        str_content = cast(str, content)
        if wrap_dicts:
            return {'type': 'text', 'text': str_content}
        else:
            return str_content

    raise NotImplementedError(f"Unknown part type: {part_type}")


def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    *,
    wrap_dicts: bool,
) -> list[ConversationMessage]:
    content = list[_ContentPart]()


    for part in parts:
        parse_res = _parse_chat_message_content_part(
            part,
            # mm_parser - 暂时设置为 None , 多模态的推理暂时不完全支持
            wrap_dicts=wrap_dicts,
        )
        if parse_res:
            content.append(parse_res)

    if wrap_dicts:
        # Parsing wraps images and texts as interleaved dictionaries
        return [ConversationMessage(role=role,
                                    content=content)]  # type: ignore
    texts = cast(list[str], content)
    text_prompt = "\n".join(texts)
    return [ConversationMessage(role=role, content=text_prompt)]


# No need to validate using Pydantic again
_AssistantParser = partial(cast, ChatCompletionAssistantMessageParam)
_ToolParser = partial(cast, ChatCompletionToolMessageParam)


def _parse_chat_message_content(
    message: ChatCompletionMessageParam,
    content_format: _ChatTemplateContentFormat,
) -> list[ConversationMessage]:
    # Handle both dict and ChatMessage object
    if hasattr(message, 'model_dump'):
        # Convert Pydantic model to dict
        message_dict = message.model_dump(exclude_none=True)
    else:
        message_dict = message
    
    role = message_dict["role"]
    content = message_dict.get("content")

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [
            ChatCompletionContentPartTextParam(type="text", text=content)
        ]
    result = _parse_chat_message_content_parts(
        role,
        content,  # type: ignore
        # mm_tracker,
        wrap_dicts=(content_format == "openai"),
    )

    for result_msg in result:
        if role == 'assistant':
            parsed_msg = _AssistantParser(message_dict)

            # The 'tool_calls' is not None check ensures compatibility.
            # It's needed only if downstream code doesn't strictly
            # follow the OpenAI spec.
            if ("tool_calls" in parsed_msg \
                    and parsed_msg["tool_calls"] is not None):
                result_msg["tool_calls"] = list(parsed_msg["tool_calls"])
        elif role == "tool":
            parsed_msg = _ToolParser(message_dict)
            if "tool_call_id" in parsed_msg:
                result_msg["tool_call_id"] = parsed_msg["tool_call_id"]

        if "name" in message_dict and isinstance(message_dict["name"], str):
            result_msg["name"] = message_dict["name"]

    return result


def _postprocess_messages(messages: list[ConversationMessage]) -> None:
    # per the Transformers docs & maintainers, tool call arguments in
    # assistant-role messages with tool_calls need to be dicts not JSON str -
    # this is how tool-use chat templates will expect them moving forwards
    # so, for messages that have tool_calls, parse the string (which we get
    # from openAI format) to dict
    for message in messages:
        if (message["role"] == "assistant" and "tool_calls" in message
                and isinstance(message["tool_calls"], list)):

            for item in message["tool_calls"]:
                item["function"]["arguments"] = json.loads(
                    item["function"]["arguments"])


def parse_chat_messages_futures(
    messages: list[ChatCompletionMessageParam],
    model_config: None,
    tokenizer: AnyTokenizer,
    content_format: _ChatTemplateContentFormat,
) -> tuple[list[ConversationMessage], Awaitable[Optional[dict]]]:
    conversation: list[ConversationMessage] = []

    for msg in messages:
        sub_messages = _parse_chat_message_content(
            msg,
            # mm_tracker,
            content_format,
        )

        conversation.extend(sub_messages)

    _postprocess_messages(conversation)
    return conversation, None
    # return conversation, mm_tracker.all_mm_data()


def apply_hf_chat_template(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    conversation: list[ConversationMessage],
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
    *,
    trust_remote_code: bool = False,
    tokenize: bool = False,  # Different from HF's default
    **kwargs: Any,
) -> str:
    # Check if conversation is empty
    if not conversation:
        raise ValueError("Conversation cannot be empty. At least one message is required.")
    
    hf_chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=chat_template,
        tools=tools,
        trust_remote_code=trust_remote_code,
    )

    if hf_chat_template is None:
        raise ValueError(
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one.")

    try:
        
        return tokenizer.apply_chat_template(
            conversation=conversation,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            chat_template=hf_chat_template,
            tokenize=tokenize,
            **kwargs,
        )

    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except (RuntimeError, TypeError, AttributeError) as e:

        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `transformers` while applying chat template")
        raise ValueError from e


def apply_mistral_chat_template(
    tokenizer: MistralTokenizer,
    messages: list[ChatCompletionMessageParam],
    chat_template: Optional[str],
    tools: Optional[list[dict[str, Any]]],
    **kwargs: Any,
) -> list[int]:
    from mistral_common.exceptions import MistralCommonException

    # The return value of resolve_mistral_chat_template is always None,
    # and we won't use it.
    resolve_mistral_chat_template(
        chat_template=chat_template,
        **kwargs,
    )

    try:
        return tokenizer.apply_chat_template(
            messages=messages,
            tools=tools,
            **kwargs,
        )
    # mistral-common uses assert statements to stop processing of input
    # if input does not comply with the expected format.
    # We convert those assertion errors to ValueErrors so they can be
    # are properly caught in the preprocessing_input step
    except (AssertionError, MistralCommonException) as e:
        raise ValueError from e

    # External library exceptions can sometimes occur despite the framework's
    # internal exception management capabilities.
    except (RuntimeError, TypeError, AttributeError) as e:

        # Log and report any library-related exceptions for further
        # investigation.
        logger.exception(
            "An error occurred in `mistral_common` while applying chat "
            "template")
        raise ValueError from e


def random_tool_call_id() -> str:
    return f"chatcmpl-tool-{random_uuid()}"
