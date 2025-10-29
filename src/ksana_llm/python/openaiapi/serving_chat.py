# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ==============================================================================
# Adapted from vLLM project
# [vLLM Project]
# Ref:
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/serving_chat.py
# ==============================================================================
"""
OpenAI Chat Completions API服务实现
"""

import uuid
import asyncio
import time
import json
import re
from http import HTTPStatus
from typing import Callable, Dict, List, Optional, Union, Any, AsyncGenerator, Tuple, Final
from functools import lru_cache

import partial_json_parser
from pydantic import TypeAdapter
from fastapi import Request
from fastapi.responses import JSONResponse

from openaiapi.openai_adapter import (
    OpenAIConfig, ErrorType, KsanaOpenAIServing,

)
from openaiapi.openai_protocol import (
    ChatCompletionResponseChoice, UsageInfo,
    ChatMessage, DeltaMessage, DeltaToolCall,
    ChatCompletionNamedToolChoiceParam,
    DeltaFunctionCall,
    ToolCall, FunctionCall, FunctionDefinition,
    ChatCompletionRequest, 
    ChatCompletionResponseStreamChoice, 
    ChatCompletionResponse, ChatCompletionStreamResponse,
    ChoiceLogprobs
)
from openaiapi.request_converter import RequestConverter
from openaiapi.transformers_utils.chat_utils import ChatTemplateContentFormatOption
from openaiapi.tool_parsers import ToolParser, ToolParserManager
from openaiapi.reasoning import ReasoningParser, ReasoningParserManager
from openaiapi.transformers_utils.chat_utils import (
    AnyTokenizer, make_tool_call_id, get_history_tool_calls_cnt)
from utilize.logger import get_logger

logger = get_logger(__name__)


class KsanaOpenAIServingChat(KsanaOpenAIServing):
    """
    KsanaLLM Chat Completions API实现
    """
    
    def __init__(self,
                 llm_server,
                 config: Optional[OpenAIConfig] = None,
                 *,
                 reasoning_parser: str = "",
                 tool_parser: Optional[str] = None,
                 chat_template: Optional[str] = None,
                 chat_template_content_format: Optional[ChatTemplateContentFormatOption] = None):
        super().__init__(llm_server, config)

        self.response_role = "assistant"
        self.chat_template = chat_template
        self.chat_template_content_format = chat_template_content_format
        self.tool_call_id_type = 'random'
        
        self._init_parsers()
    
    def _init_parsers(self) -> None:
        self.reasoning_parser: Optional[Callable[[AnyTokenizer],
                                            ReasoningParser]] = None
        if self.config.reasoning_parser is not None:
            try:
                self.reasoning_parser = (ReasoningParserManager.get_reasoning_parser(self.config.reasoning_parser))
                assert self.reasoning_parser is not None
                logger.info(f"Reasoning parser '{self.config.reasoning_parser}' initialized successfully")
            except TypeError as e:
                logger.error(f"Failed to initialize reasoning parser: {e}")

        self.tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None
        if self.config.enable_auto_tool_choice:
            try:
                if(self.config.tool_call_parser == 'kimi_k2'):
                    self.tool_call_id_type = 'kimi_k2'
                self.tool_parser = ToolParserManager.get_tool_parser(self.config.tool_call_parser)
                logger.info(f"Tool parser '{self.config.tool_call_parser}' initialized successfully")
            except TypeError as e:
                logger.error(f"❌ [DEBUG] Failed to initialize tool parser: {e}")
    
    @lru_cache(maxsize=128)
    def _format_message(self, role: str, content: str) -> str:
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
        return f"{role_map.get(role, role.capitalize())}: {content}"
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        formatted_messages = [self._format_message(msg.role, msg.content) for msg in messages]
        formatted_messages.append("Assistant:")
        return "\n".join(formatted_messages)
    
    def _convert_to_ksana_request(self, request: ChatCompletionRequest) -> Dict[str, Any]:

        converter = RequestConverter(self.config, self.tokenizer)
        
        request_dict = request.model_dump(by_alias=True)
        
        # Handle Extra body
        if hasattr(request, '__pydantic_extra__') and request.__pydantic_extra__:
            request_dict.update(request.__pydantic_extra__)
        
        ksana_request = converter.convert_to_ksana_format(
            request_dict,
            api_type="chat",
        )
        
        return ksana_request
    
    def _count_tokens(self, text: str) -> int:
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def _count_messages_tokens(self, messages: List[ChatMessage]) -> int:
        formatted_prompt = self._messages_to_prompt(messages)
        lines = formatted_prompt.split('\n')
        if lines and lines[-1].strip() == "Assistant:":
            input_prompt = '\n'.join(lines[:-1])
        else:
            input_prompt = formatted_prompt
        return self._count_tokens(input_prompt)
    
    async def _generate_stream_chunk(
        self,
        request_id: str,
        model_name: str,
        tokenizer: AnyTokenizer,
        content: Optional[str] = None,
        finish_reason: Optional[str] = None,
        role: Optional[str] = None,
        logprobs: Optional[Dict[str, Any]] = None,
        usage: Optional[Any] = None,
        choice_index: int = 0,
        delta_message: Optional[DeltaMessage] = None,
        current_output_token: Optional[List[int]] = None,
        num_output_top_logprobs: Optional[int] = None,
    ) -> str:
        
        if content is None and finish_reason is None and role is None and delta_message is None:
            return "data: [DONE]\n"
        
        if delta_message is not None and logprobs is None:
            # filter out empty strings from delta_message
            delta_dict = delta_message.model_dump(exclude_unset=True, exclude_none=True)
            delta_dict = {k: v for k, v in delta_dict.items() if v != ""}
            
            if not delta_dict and not finish_reason and not usage and not logprobs:
                return ""
            
            chunk_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": choice_index,
                    "delta": delta_dict,
                    "finish_reason": finish_reason,
                    "logprobs": logprobs
                }]
            }
            if usage:
                chunk_data["usage"] = usage
            return f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        
        converter = RequestConverter(self.config, tokenizer)
        return converter.format_chat_completion_stream_chunk(
            request_id=request_id,
            model_name=model_name,
            content=content,
            role=role,
            finish_reason=finish_reason,
            logprobs=logprobs,
            usage=usage,
            choice_index=choice_index,
            delta_message=delta_message,
            current_output_token=current_output_token,
            num_output_top_logprobs=num_output_top_logprobs
        )
    
    async def _chat_completion_stream_generator(
        self,
        ksana_output_iterator: AsyncGenerator,
        request_id: str,
        model_name: str,
        tokenizer: AnyTokenizer,
        request_logprobs: bool = False,
        request: Optional[ChatCompletionRequest] = None,
        conversation: Optional[List[Any]] = None,
        prompt_tokens_count: Optional[List[int]] = None,
    ) -> AsyncGenerator[str, None]:
        
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        
        previous_token_lengths = [0] * num_choices
        unchanged_count = [0] * num_choices

        # Determine if we need to use Named tool parser
        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request))
        
        # Judge if we need to judge accumulated tokens
        previous_token_ids = []
        delta_text = ""  # 初始化 delta_text 变量

        all_previous_token_ids: Optional[list[list[int]]]   
        function_name_returned = [False] * num_choices
        if self.tool_call_id_type == 'kimi_k2':
            # Due to Kimi K2 tool call ID using functions.{func_name}:{idx} for its ID
            # Get History tool call count from Prompt to avoid duplicated tool call IDs
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0
        
        previous_texts = [""] * num_choices
        all_previous_token_ids = [[]] * num_choices
        if tool_choice_auto or self.reasoning_parser:
            # For reasoning parser and tool call all enabled
            added_content_delta_arr = [False] * num_choices
            reasoning_end_arr = [False] * num_choices
        elif request.tool_choice == "required":
            all_previous_token_ids = None

        if self.reasoning_parser:
            reasoning_parser = self.reasoning_parser(tokenizer)
        if tool_choice_auto and self.tool_parser:
            tool_parsers: list[Optional[ToolParser]] = [
                self.tool_parser(tokenizer)
            ] * num_choices
        else:
            tool_parsers = [None] * num_choices

        stream_options = getattr(request, 'stream_options', None)
        include_usage = False
        if stream_options:
            include_usage = getattr(stream_options, 'include_usage', False)

        try:
            async for output_data in ksana_output_iterator:

                if output_data is None:
                    continue
                
                if isinstance(output_data, tuple):
                    output_index, ksana_python_output = output_data
                else:
                    output_index = 0
                    ksana_python_output = output_data

                # First chunk , response with assistant role, get prompt token ids
                if first_iteration:
                    first_iteration = False
                    if prompt_tokens_count and output_index < len(prompt_tokens_count):
                        num_prompt_tokens = prompt_tokens_count[output_index]
                    elif prompt_tokens_count and len(prompt_tokens_count) > 0:
                        num_prompt_tokens = prompt_tokens_count[0]
                    else:
                        num_prompt_tokens = 0
                    role = self.get_chat_request_role(request)

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name)

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: Union[str, list[dict[str, str]]] = ""
                        if conversation and "content" in conversation[-1] \
                                and conversation[-1].get("role") == role:
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        logprobs=None,
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name)

                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                                yield f"data: {data}\n\n"
                    
                is_finished = False
                if hasattr(ksana_python_output, 'finish_status') and ksana_python_output.finish_status:
                    if hasattr(ksana_python_output.finish_status, 'OK') and not ksana_python_output.finish_status.OK():
                        is_finished = True
                
                # Notice : The Ksana Output is Accumulated mode.
                for i, choice_tokens in enumerate(ksana_python_output.output_tokens):
                    try:
                        if finish_reason_sent[i]:
                            continue
                        
                        current_token_ids = list(choice_tokens)
                        current_token_length = len(current_token_ids)
                        
                        # check token length unchanged 
                        if current_token_length == previous_token_lengths[i]:
                            unchanged_count[i] += 1
                            if unchanged_count[i] >= 2:
                                is_finished = True
                        else:
                            unchanged_count[i] = 0  
                            previous_token_lengths[i] = current_token_length
                        
                        full_text = self.llm_server.pre_post_processor.decode(current_token_ids, True)
                        
                        delta_text = ""
                        prev_length = len(previous_texts[i])

                        # update previous

                        previous_text = previous_texts[i]
                        if all_previous_token_ids is not None:
                            previous_token_ids = all_previous_token_ids[i]
                        prev_token_length = len(previous_token_ids)
                        current_text = full_text

                        # calc delta_token_ids
                        output_tokenids = choice_tokens[prev_token_length: ]
                        
                        tool_parser_instance = tool_parsers[i]

                        if full_text and len(full_text) > prev_length:
                            delta_text = full_text[prev_length:]
                            previous_texts[i] = full_text
                                                
                        if not delta_text and not choice_tokens and \
                            not previous_num_tokens[i]:
                            # Chunked prefill case, don't return empty chunks
                            continue
                        
                        delta_message: Optional[DeltaMessage]

                        if all_previous_token_ids is not None:
                            all_previous_token_ids[i] = current_token_ids

                        # handle streaming deltas for tools with named tool_choice
                        if tool_choice_function_name:
                            if (self.reasoning_parser
                                    and not reasoning_parser.is_reasoning_end(
                                        previous_token_ids)):
                                assert reasoning_parser is not None
                                delta_message = (
                                    reasoning_parser.
                                    extract_reasoning_content_streaming(
                                        previous_text,
                                        current_text,
                                        delta_text,
                                        previous_token_ids,
                                        current_token_ids,
                                        output_tokenids,
                                    ))
                                # When encountering think end id in delta_token_ids,
                                # process the `content`. Only keep 'content',
                                # remove 'reasoning_content'
                                if reasoning_parser.is_reasoning_end(
                                        list(output_tokenids)):
                                    if delta_message and delta_message.content:
                                        # This need to be added to next `delta_text`
                                        current_text = delta_message.content
                                        delta_message.content = None
                                    else:
                                        current_text = ""
                            else:
                                # Just to add remaining `content`
                                if self.reasoning_parser:
                                    delta_text = previous_text + delta_text
                                    current_text = ""

                                if function_name_returned[i]:
                                    delta_tool_call = DeltaToolCall(
                                        function=DeltaFunctionCall(
                                            arguments=delta_text),
                                        index=i)
                                else:
                                    delta_tool_call = DeltaToolCall(
                                        id=make_tool_call_id(),
                                        type="function",
                                        function=DeltaFunctionCall(
                                            name=tool_choice_function_name,
                                            arguments=delta_text),
                                        index=i)
                                    function_name_returned[i] = True

                                delta_message = DeltaMessage(tool_calls=[
                                    delta_tool_call,
                                ])
                        elif request and request.tool_choice == "required":
                            fn_name_returned = function_name_returned[i]

                            if self.reasoning_parser:
                                _, content = \
                                reasoning_parser.extract_reasoning_content(
                                    current_text,
                                    request
                                )
                            else:
                                content = current_text

                            delta_message, function_name_returned[i] \
                                = self.extract_tool_call_required_streaming(
                                previous_text=previous_text,
                                current_text=content,
                                delta_text=delta_text,
                                function_name_returned=fn_name_returned,
                                tool_call_idx=history_tool_call_cnt
                            )
                            
                            if delta_message:
                                chunk_data = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model_name,
                                    "choices": [{
                                        "index": i,
                                        "delta": delta_message.model_dump(exclude_unset=True),
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n"
                                continue  # 对于 required 模式，工具调用是唯一输出
                            if (delta_message and delta_message.tool_calls and
                                    delta_message.tool_calls[0].id is not None):
                                history_tool_call_cnt += 1
                                tools_streamed[i] = True
                        # auto tool choice and reasoning parser enabled, extract reasoning content
                        elif tool_choice_auto and self.reasoning_parser:
                            assert reasoning_parser is not None
                            assert tool_parser_instance is not None
                            assert added_content_delta_arr is not None
                            assert reasoning_end_arr is not None

                            if not reasoning_end_arr[i]:
                                delta_message = (
                                    reasoning_parser.
                                    extract_reasoning_content_streaming(
                                        previous_text,
                                        current_text,
                                        delta_text,
                                        previous_token_ids,
                                        current_token_ids,
                                        output_tokenids,
                                    ))
                                # When encountering think end id in prompt_token_ids
                                # i.e {"enable_thinking": False},
                                # set reasoning status to end.
                                # Remove the text and token ids related
                                # to 'reasoning_content'.
                                if hasattr(ksana_python_output, 'input_tokens') and \
                                    ksana_python_output.input_tokens and \
                                        reasoning_parser.is_reasoning_end(
                                        list(ksana_python_output.input_tokens)):
                                    reasoning_end_arr[i] = True
                                    current_token_ids = list(output_tokenids)
                                    if delta_message and delta_message.content:
                                        current_text = delta_message.content
                                        delta_message.content = None
                                    else:
                                        current_text = ""
                                # When encountering think end id in delta_token_ids,
                                # set reasoning status to end.
                                # Remove the text and token ids related
                                # to 'reasoning_content'.
                                if reasoning_parser.is_reasoning_end(
                                        list(output_tokenids)):
                                    reasoning_end_arr[i] = True
                                    current_token_ids =  \
                                        reasoning_parser.extract_content_ids(
                                            list(output_tokenids))
                                    if delta_message and delta_message.content:
                                        current_text = delta_message.content
                                        delta_message.content = None
                                    else:
                                        current_text = ""

                            # handle tool calls only after reasoning is done,
                            else:
                                delta_token_ids = list(output_tokenids)
                                # First time to tool call,
                                # add the remaining text and token ids
                                # to delta from previous
                                if not added_content_delta_arr[i]:
                                    added_content_delta_arr[i] = True
                                    previous_text = ""
                                    previous_token_ids = []
                                    delta_text = current_text
                                    delta_token_ids = current_token_ids

                                # Get tool parser instance for this choice
                                if tool_parser_instance is not None:
                                    delta_message = (
                                        tool_parser_instance.extract_tool_calls_streaming(
                                            previous_text=previous_text,
                                            current_text=current_text,
                                            delta_text=delta_text,
                                            previous_token_ids=previous_token_ids,
                                            current_token_ids=current_token_ids,
                                            delta_token_ids=delta_token_ids,
                                            request=request))
                                    # Check tool parser returned content length is abnormal
                                    if delta_message and delta_message.content:
                                        # Calc recalculated delta text
                                        if output_tokenids:
                                            recalculated_delta_text = self.llm_server.pre_post_processor.decode(
                                                list(output_tokenids), True)
                                            # Notice:
                                            # If tool_parser returns a wrong content length, it maybe:
                                            # 1. Contains longer reasoning content, which is not expected
                                            # it may contain reasoning content, delete
                                            # 2. Contains shorter tool_call content, it may be a tool call
                                            # Use it directly
                                            if len(delta_message.content) > len(recalculated_delta_text):
                                                delta_text = recalculated_delta_text
                                                delta_message.content = recalculated_delta_text
                                    
                                    # Post-process tool calls for streaming (DeltaToolCall)
                                    if delta_message and delta_message.tool_calls:
                                        self.post_process_tool_calls(delta_message.tool_calls, history_tool_call_cnt)
                                    
                                else:
                                    delta_message = DeltaMessage(content=delta_text)

                        # when only tool calls
                        elif tool_choice_auto:
                            assert tool_parser_instance is not None
                            delta_message = (
                                tool_parser_instance.extract_tool_calls_streaming(
                                    previous_text=previous_text,
                                    current_text=current_text,
                                    delta_text=delta_text,
                                    previous_token_ids=previous_token_ids,
                                    current_token_ids=current_token_ids,
                                    delta_token_ids=output_tokenids,
                                    request=request))
                            # Check tool parser returned content length is abnormal
                            if delta_message and delta_message.content:
                                if output_tokenids:
                                    recalculated_delta_text = self.llm_server.pre_post_processor.decode(
                                        list(output_tokenids), True)
                                    if len(delta_message.content) > len(recalculated_delta_text):
                                        delta_text = recalculated_delta_text
                                        delta_message.content = recalculated_delta_text
                            
                            # Post-process tool calls for streaming (DeltaToolCall)
                            if delta_message and delta_message.tool_calls:
                                self.post_process_tool_calls(delta_message.tool_calls, history_tool_call_cnt)
                        elif self.reasoning_parser:
                            delta_message = (reasoning_parser.
                                             extract_reasoning_content_streaming(
                                                previous_text,
                                                current_text,
                                                delta_text,
                                                previous_token_ids,
                                                current_token_ids,
                                                output_tokenids,
                                            ))
                        
                        else:
                            delta_message = DeltaMessage(content = delta_text)
                        
                        previous_num_tokens[i] += len(output_tokenids)

                        if delta_message is None:
                            continue
                        
                        logprobs = getattr(ksana_python_output, 'logprobs', None) if request_logprobs else None

                        # 当有 logprobs 时，直接使用 delta_message，这样可以保留 reasoning_content 和 content 的区分
                        # 当没有 logprobs 时，也使用 delta_message 以保持一致性
                        if delta_message and delta_message != DeltaMessage(content=""):
                            chunk = await self._generate_stream_chunk(
                                request_id, model_name, tokenizer,
                                logprobs=logprobs,
                                delta_message=delta_message,
                                choice_index=i,
                                current_output_token=output_tokenids,
                                num_output_top_logprobs=getattr(request, 'top_logprobs', 0) if request_logprobs else 0,
                            )
                            if chunk:
                                yield chunk
                        
                        # if this choice is finished, send finish_reason
                        if is_finished and not finish_reason_sent[i]:
                            finish_reason = "stop"

                            if tool_parsers and tool_parsers[i] is not None:
                                if hasattr(tool_parser_instance, 'prev_tool_call_arr') \
                                    and len(tool_parser_instance.prev_tool_call_arr) > 0:
                                    finish_reason = "tool_calls"
                            
                            # if it is the last choice and we need usage info, send it
                            usage_info = None
                            if include_usage and i == num_choices - 1:
                                completion_tokens = current_token_length
                                usage_info = {
                                    "prompt_tokens": num_prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": num_prompt_tokens + completion_tokens
                                }
                            
                            # send finish_reason
                            chunk = await self._generate_stream_chunk(
                                request_id, model_name, tokenizer,
                                finish_reason=finish_reason,
                                usage=usage_info,
                                choice_index=i
                            )
                            if chunk:
                                yield chunk
                            finish_reason_sent[i] = True
                        if all_previous_token_ids is not None:
                            previous_token_ids = current_token_ids
                    except (AttributeError, TypeError) as e:
                        logger.error(f"Decoding error: {e}")

            # if there are still finish_reason not sent,
            for i in range(num_choices):
                if not finish_reason_sent[i]:
                    finish_reason = "stop"
                    # check if tool parser is used
                    if tool_parsers and tool_parsers[i] is not None:
                        tool_parser_instance = tool_parsers[i]
                        if hasattr(tool_parser_instance, 'prev_tool_call_arr') \
                            and len(tool_parser_instance.prev_tool_call_arr) > 0:
                            finish_reason = "tool_calls"
                    
                    usage_info = None
                    if include_usage:
                        completion_tokens = previous_token_lengths[i]
                        usage_info = {
                            "prompt_tokens": num_prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": num_prompt_tokens + completion_tokens
                        }

                    chunk = await self._generate_stream_chunk(
                        request_id, model_name, tokenizer,
                        finish_reason=finish_reason,
                        usage=usage_info,
                        choice_index=i
                    )
                    if chunk:
                        yield chunk
                    finish_reason_sent[i] = True
                        
        except asyncio.CancelledError:
            logger.info("Streaming cancelled by client")
            raise
        except (RuntimeError, OSError, MemoryError) as e:
            logger.error(f"Streaming resource error: {e}")
            converter = RequestConverter(self.config)
            error_data = converter.format_streaming_error(str(e), "internal_server_error", 500)
            yield error_data
        except (TypeError, AttributeError) as e:
            logger.error(f"Streaming data processing error: {e}")
            converter = RequestConverter(self.config)
            error_data = converter.format_streaming_error("Data processing failed", "internal_server_error", 500)
            yield error_data
        finally:
            yield "data: [DONE]\n\n"
    
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse, JSONResponse]:

        error_check_ret = await self._check_model(request)
        tokenizer = self.tokenizer
        if error_check_ret is not None:
            return error_check_ret
        
        try:
            model_name = self._get_model_name(request.model)
            tool_parser = self.tool_parser
            
            if (request.tool_choice == "auto" and
                    not (self.config.enable_auto_tool_choice and tool_parser is not None)):
                return self.create_error_response(
                    "\"auto\" tool choice requires "
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )

            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]
            
            # preprocess chat messages, add hf chat template, etc.
            (
                conversation,
                _,
                engine_prompts,
            ) = await self._preprocess_chat(
                request,
                tokenizer,
                request.messages,
                chat_template=getattr(request, 'chat_template', None) or self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                add_generation_prompt=getattr(request, 'add_generation_prompt', True),
                continue_final_message=getattr(request, 'continue_final_message', False),
                tool_dicts=tool_dicts,
                documents=getattr(request, 'documents', None),
                chat_template_kwargs=getattr(request, 'chat_template_kwargs', None),
                tool_parser=tool_parser,
                truncate_prompt_tokens=getattr(request, 'truncate_prompt_tokens', None),
                add_special_tokens=getattr(request, 'add_special_tokens', True),
            )
                        
        except (TypeError, RuntimeError, ValueError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(
                f"Preprocessing failed: {str(e)}",
                ErrorType.VALIDATION_ERROR,
                HTTPStatus.BAD_REQUEST
            )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        req_ctx = self._get_trace_context(raw_request) if raw_request else None

        all_outputs = []
        prompt_tokens_per_request = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_token_ids = engine_prompt.get("prompt_token_ids", [])
                prompt_tokens_per_request.append(len(prompt_token_ids))  # laod prompt tokens nums
                
                ksana_request = self._convert_to_ksana_request(request)
                ksana_request["input_tokens"] = prompt_token_ids
                max_new_tokens = request.max_tokens or request.max_completion_tokens or 8192
                max_new_tokens = max_new_tokens
                ksana_request["sampling_config"]["max_new_tokens"] = max_new_tokens if max_new_tokens > 0 else 0
                #multi_modal_data & mm_processor_kwargs
                if "multi_modal_data" in engine_prompt:
                    ksana_request["multi_modal_data"] = engine_prompt["multi_modal_data"]
                
                if "mm_processor_kwargs" in engine_prompt:
                    ksana_request["mm_processor_kwargs"] = engine_prompt["mm_processor_kwargs"]
                
                # call KsanaLLM generate func, get output
                status, output = await self.llm_server.model.generate(
                    request_dict=ksana_request,
                    req_ctx=req_ctx,
                    streamer=request.stream
                )
                
                if not status.OK():
                    logger.error(f"Engine prompt {i+1} failed: {status.GetMessage()}")
                    continue
                
                all_outputs.append((status, output))
            
            # all output failed
            if not all_outputs:
                return self.create_error_response(
                    "All engine prompts failed to generate output",
                    ErrorType.INTERNAL_SERVER_ERROR,
                    HTTPStatus.INTERNAL_SERVER_ERROR
                )
        
        except ValueError as e:
            logger.exception("Error in request processing")
            return self.create_error_response(
                f"Request processing failed: {str(e)}",
                ErrorType.VALIDATION_ERROR,
                HTTPStatus.BAD_REQUEST
            )

        if request.stream:
            request_logprobs = getattr(request, 'logprobs', False)

            async def stream_generator():
                valid_outputs = [
                    (idx, output)
                    for idx, (status, output) in enumerate(all_outputs)
                    if status.OK()
                ]
                async def merged_outputs():
                    # 记录失败的输出以减少日志调用
                    failed_indices = [
                        idx for idx, (status, _) in enumerate(all_outputs)
                        if not status.OK()
                    ]
                    if failed_indices:
                        logger.warning(f"Outputs failed at indices: {failed_indices}")
                    
                    # 处理有效输出
                    for idx, output in valid_outputs:
                        async for item in output:
                            yield (idx, item)
                    
                    yield None

                async for chunk in self._chat_completion_stream_generator(
                    merged_outputs(), request_id, model_name, tokenizer,
                    request_logprobs, request, conversation, prompt_tokens_per_request
                ):
                    yield chunk
            
            return stream_generator()
        else:
            return await self._chat_completion_full_generator(
                all_outputs, request_id, model_name, request, conversation, tokenizer,
                prompt_tokens_per_request
            )

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        last_message = request.messages[-1]
        if isinstance(last_message, dict):
            return last_message["role"]
        else:
            return last_message.role

    async def _chat_completion_full_generator(
        self,
        all_outputs: List[Tuple[Any, Any]],  # List of (status, output) tuples
        request_id: str,
        model_name: str,
        request: Optional[ChatCompletionRequest] = None,
        conversation: Optional[List[Any]] = None,
        tokenizer: Optional[Any] = None,
        prompt_tokens_per_request: Optional[List[int]] = None
    ) -> ChatCompletionResponse:

        choices = []
        if self.tool_call_id_type == 'kimi_k2':
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        role = self.get_chat_request_role(request) if request else "assistant"
        
        request_logprobs = getattr(request, 'logprobs', False) if request else False
        top_logprobs = getattr(request, 'top_logprobs', 0) if request_logprobs else 0
        try:
            for i, (status, output) in enumerate(all_outputs):
                full_output_text = ""
                output_tokens = []
                output_logprobs = None
                finish_reason = None

                tool_calls = []
                # Check output stauts
                if not status.OK():
                    error_message = f"Error: {status.GetMessage()}"
                    logger.warning(f"Output {i+1} has error status: {error_message}")
                    
                    message = ChatMessage(
                        role=role,
                        content=error_message
                    )
                    
                    choice = ChatCompletionResponseChoice(
                        index=i,
                        message=message,
                        finish_reason="error"
                    )
                    choices.append(choice)
                    continue

                # Dealing Ksana Python output, decode output tokens
                for request_output in output.output_tokens:
                    output_tokens.extend(request_output)
                full_output_text = self.llm_server.pre_post_processor.decode(output_tokens, True)

                finish_reason = "stop"
                
                # 处理 logprobs（如果请求了）
                if output.logprobs:
                    try:
                        converter = RequestConverter(self.config, tokenizer)
                        loop = asyncio.get_running_loop()
                        def _convert_logprobs() -> Optional[ChoiceLogprobs]:
                            return converter._convert_ksana_logprobs_to_openai(
                                output.logprobs,
                                token_ids=output_tokens,
                                num_output_top_logprobs=top_logprobs,
                            )
                        output_logprobs = await loop.run_in_executor(None, _convert_logprobs)
                    except (IndexError, KeyError, TypeError, ValueError) as e:
                        logger.warning(f"Failed to process logprobs: {e}")
                        output_logprobs = None
                    except asyncio.CancelledError:
                        logger.warning("Logprobs conversion was cancelled for req_id=%s index=%d", request_id, i)
                        raise
                else:
                    output_logprobs = None

                if self.reasoning_parser is not None:
                    try:
                        reasoning_parser = self.reasoning_parser(tokenizer)
                    except RuntimeError as e:
                        logger.exception("Error in reasoning parser creation.")
                        continue
                    reasoning_content, content = (
                        reasoning_parser.extract_reasoning_content(
                            full_output_text, request=request))
                else:
                    reasoning_content = None
                    content = full_output_text
                
                #Dealing with tool calls
                if (not self.config.enable_auto_tool_choice or not self.tool_parser) and \
                    (not isinstance (request.tool_choice,
                                     ChatCompletionNamedToolChoiceParam)
                                    and request.tool_choice != "required"):
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=content
                    )

                elif request.tool_choice and type (
                    request.tool_choice) is ChatCompletionNamedToolChoiceParam:
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        tool_calls=[
                            ToolCall(
                                id=make_tool_call_id(id_type=self.tool_call_id_type,
                                                     func_name=request.tool_choice.function.name,
                                                     idx=history_tool_call_cnt),
                                function=FunctionCall(
                                    name=request.tool_choice.function.name,
                                    arguments=content
                                ),
                                type="function"
                            )
                        ])
                    history_tool_call_cnt += 1
                    finish_reason = "tool_calls"

                elif request.tool_choice and request.tool_choice == "required":
                    # temporary tool call class
                    # TODO(ethanyczeng): support Mistral tool call class 
                    assert content is not None
                    tool_calls = TypeAdapter(
                        list[FunctionDefinition]).validate_json(content)
                    tool_call_ids = []
                    for tool_call in tool_calls:
                        tool_call_ids.append(
                            make_tool_call_id(id_type=self.tool_call_id_type,
                                              func_name=tool_call.name,
                                              idx=history_tool_call_cnt))
                        history_tool_call_cnt += 1
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content="",
                        tool_calls=[
                            ToolCall(id=tool_call_ids[i],
                                function=FunctionCall(
                                name=tool_call.name,
                                arguments=json.dumps(tool_call.parameters,
                                                        ensure_ascii=False)))
                            for i, tool_call in enumerate(tool_calls)
                        ])
                    finish_reason = "tool_calls"
                
                elif not request.tool_choice or request.tool_choice == "none":
                    # dont use any tools
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=content
                    )
                
                elif request.tools and (request.tool_choice == "auto" or request.tool_choice is None) \
                     and self.config.enable_auto_tool_choice and self.tool_parser:
                    # auto tool choice enabled
                    try:
                        tool_parser = self.tool_parser(tokenizer)
                    except RuntimeError as e:
                        logger.error(f"Tool parser failed: {e}")
                        message = ChatMessage(
                            role=role,
                            reasoning_content=reasoning_content,
                            content=content
                        )

                    tool_call_info = tool_parser.extract_tool_calls(
                        content if content is not None else "", request=request)
                    
                    if tool_call_info.tools_called:
                        self.post_process_tool_calls(tool_call_info.tool_calls, history_tool_call_cnt)
                        message = ChatMessage(
                            role=role,
                            reasoning_content=reasoning_content,
                            content=tool_call_info.content,
                            tool_calls=tool_call_info.tool_calls
                        )
                        history_tool_call_cnt += len(tool_call_info.tool_calls)
                        finish_reason = "tool_calls"
                    else:
                        message = ChatMessage(
                            role=role,
                            reasoning_content=reasoning_content,
                            content=content
                        )
                
                else:
                    message = ChatMessage(
                        role=role,
                        reasoning_content=reasoning_content,
                        content=content
                    )
                
                choice = ChatCompletionResponseChoice(
                    index=i,
                    message=message,
                    finish_reason=finish_reason or "stop",
                    logprobs=output_logprobs 
                )
                choices.append(choice)
                
                completion_tokens = len(output_tokens)
                total_completion_tokens += completion_tokens
            
            if not choices:
                message = ChatMessage(
                    role=role,
                    content="No valid outputs generated"
                )
                choice = ChatCompletionResponseChoice(
                    index=0,
                    message=message,
                    finish_reason="error"
                )
                choices.append(choice)
                
            if request.echo:
                last_msg_content: Union[str, list[dict[str, str]]] = ""
                if conversation and "content" in conversation[-1] and conversation[
                        -1].get("role") == role:
                    last_msg_content = conversation[-1]["content"] or ""
                if isinstance(last_msg_content, list):
                    last_msg_content = "\n".join(msg['text']
                                                for msg in last_msg_content)

                for choice in choices:
                    full_message = last_msg_content + (choice.message.content
                                                    or "")
                    choice.message.content = full_message

            if prompt_tokens_per_request:
                total_prompt_tokens = sum(prompt_tokens_per_request[:len(all_outputs)])
            else:
                if request and request.messages:
                    prompt_tokens = self._count_messages_tokens(request.messages)
                    total_prompt_tokens = prompt_tokens * len(all_outputs)
                else:
                    total_prompt_tokens = 0
            
            usage = UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens
            )
            
            response = ChatCompletionResponse(
                id=request_id,
                object="chat.completion",
                created=int(time.time()),
                model=model_name,
                choices=choices,
                usage=usage
            )
            
            response_dict = response.model_dump(exclude_none=True)
            return ChatCompletionResponse(**response_dict)
            
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error in full response generation: {e}")
            return self.create_error_response(
                f"Response generation failed: {str(e)}",
                ErrorType.INTERNAL_SERVER_ERROR,
                HTTPStatus.INTERNAL_SERVER_ERROR
            )
    def post_process_tool_calls(self,
        tool_calls: Union[List[ToolCall], List[DeltaToolCall]],
        history_tool_call_cnt: int = 0
    ):
        """
        Post-process tool calls to ensure proper formatting.
        
        This function handles both full-text ToolCall and streaming DeltaToolCall scenarios.
        For kimi_k2 mode, it regenerates tool_call_id based on function name and history count.
        
        Args:
            tool_calls: List of ToolCall (full-text) or DeltaToolCall (streaming)
            history_tool_call_cnt: Historical tool call count for ID generation
        """
        if self.tool_call_id_type != 'kimi_k2':
            return
        
        for i, tool_call in enumerate(tool_calls):
            # Check if this is a DeltaToolCall (streaming) or ToolCall (full-text)
            is_delta = isinstance(tool_call, DeltaToolCall)
            id_type = self.tool_call_id_type
            func_name = tool_call.function.name
            
            if is_delta:
                if tool_call.function and func_name:
                    idx = i + tool_call.index + history_tool_call_cnt
                    tool_call.id = make_tool_call_id(id_type=id_type, func_name=func_name, idx=idx)
            else:
                if hasattr(tool_call.function, 'id'):
                    idx = i + tool_call.function.id + history_tool_call_cnt
                else:
                    idx = i + history_tool_call_cnt
                tool_call.id = make_tool_call_id(id_type=id_type, func_name=func_name, idx=idx)

    def _should_stream_with_auto_tool_parsing(self,
                                              request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (request.tools and self.tool_parser and self.config.enable_auto_tool_choice
                and request.tool_choice in ['auto', None])

    @staticmethod
    def _bracket_level(text: str) -> int:
        """计算文本中的括号层级"""
        level = 0
        for c in text:
            if c == '{':
                level += 1
            elif c == '}':
                level -= 1
        return level
    
    @staticmethod
    def _filter_delta_text(delta_text: str,
                           previous_text: str) -> tuple[str, bool]:
        # remove last '},' of the tool definition stemming from the
        # "name"/"parameters" outer object or closing ']' of the tool list
        # count occurrences of opening and closing curly braces and
        # once level 0 is reached stop outputting text
        # if 0 is reached while parsing the delta_text we know the current
        # tool will finish in this current iteration
        bracket_level = KsanaOpenAIServingChat._bracket_level(previous_text)
        updated_delta, passed_zero = "", False
        for c in delta_text:
            if c == '{':
                bracket_level += 1
                passed_zero = bracket_level == 0
            elif c == '}':
                bracket_level -= 1
                passed_zero = bracket_level == 0

            if bracket_level != 0:
                updated_delta += c
            else:
                # if a comma is reached at level 0 we can stop
                if c == ',':
                    break
        return updated_delta, passed_zero

    def extract_tool_call_required_streaming(
        self,
        previous_text: str,
        current_text: Optional[str],
        delta_text: str,
        function_name_returned: bool,
        tool_call_idx: Optional[int] = None
    ) -> tuple[Optional[DeltaMessage], bool]:
        if current_text is None or current_text == "":
            # if the current text is empty, we cannot parse it
            return None, function_name_returned
        try:
            obj = partial_json_parser.loads(current_text)
        except partial_json_parser.core.exceptions.MalformedJSON:
            logger.debug('not enough tokens to parse into JSON yet')
            obj = None

        # check if the current text is a valid array
        # containing a partial tool calling object
        # if not repeat
        if obj is None or not isinstance(obj, list) or not len(obj) > 0:
            function_name_returned = False
            delta_message = None
        else:
            _, finishes_previous_tool = KsanaOpenAIServingChat._filter_delta_text(
                delta_text, previous_text)
            # take the last tool call from the generated list
            current_tool_call = obj[-1]

            # once parameters have been generated the name is complete as well
            if not finishes_previous_tool and ("name" not in current_tool_call
                                               or "parameters"
                                               not in current_tool_call):
                function_name_returned = False
                delta_message = None
            else:
                if not function_name_returned:
                    # get partly generated arguments from the latest tool call
                    param_match = re.search(r'.*"parameters":\s*(.*)',
                                            current_text)
                    arguments = param_match.group(1) if param_match else ""
                    arguments, _ = KsanaOpenAIServingChat._filter_delta_text(
                        arguments, previous_text)

                    # if this iteration finishes a previous tool call but a
                    # new incomplete tool is already generated, take the
                    # previous from the list
                    if (finishes_previous_tool
                            and "parameters" not in current_tool_call):
                        current_tool_call = obj[-2]

                    function_name_returned = True
                    tool_call_id = make_tool_call_id(
                        id_type=self.tool_call_id_type,
                        func_name=current_tool_call["name"],
                        idx=tool_call_idx)
                    delta_message = DeltaMessage(tool_calls=[
                        DeltaToolCall(id=tool_call_id,
                                      function=DeltaFunctionCall(
                                          name=current_tool_call["name"],
                                          arguments=arguments),
                                      index=len(obj) - 1,
                                      type="function")
                    ])

                else:
                    delta_text, _ = KsanaOpenAIServingChat._filter_delta_text(
                        delta_text, previous_text)

                    if delta_text != "":
                        delta_message = DeltaMessage(tool_calls=[
                            DeltaToolCall(
                                function=DeltaFunctionCall(
                                    # OpenAI API returns None
                                    # instead of name every time
                                    name=None,
                                    arguments=delta_text),
                                index=len(obj) - 1)
                        ])
                    else:
                        delta_message = None

        return delta_message, function_name_returned
