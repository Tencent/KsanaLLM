# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""
OpenAI请求/响应转换器 - 统一处理各种API的格式转换
From OpenAI Request to KsanaLLM Request
"""

import time
from typing import Dict, List, Optional, Union, Any, Callable, Type
from dataclasses import dataclass
from pydantic import BaseModel
import orjson
from openaiapi.openai_protocol import (
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest,
    ChatMessage, ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatCompletionStreamResponse, ChatCompletionResponseStreamChoice, DeltaMessage,
    CompletionResponse, CompletionResponseChoice, CompletionLogProbs,
    CompletionStreamResponse, CompletionResponseStreamChoice,
    EmbeddingResponse, EmbeddingResponseData,
    UsageInfo,
    ChoiceLogprobs, ChatCompletionTokenLogprob, TopLogprob
)

from utilize.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KsanaConfig:
    """Ksana配置"""
    default_model_name: str = "ksana-llm"
    default_temperature: float = 0.7
    default_top_k: int = 1
    default_top_p: float = 1.0
    default_num_beams: int = 1
    default_num_return_sequences: int = 1
    default_length_penalty: float = 1.0
    default_repetition_penalty: float = 1.0
    default_no_repeat_ngram_size: int = 0
    default_encoder_no_repeat_ngram_size: int = 0
    default_decoder_no_repeat_ngram_size: int = 0
    default_logprobs: int = 0
    default_do_sample: bool = False
    default_ignore_eos: bool = False
    token_estimation_factor: float = 4.0


class RequestConverter:
    """
    统一的请求/响应转换器
    将OpenAI格式的请求转换为KsanaLLM内部格式，并将KsanaLLM输出转换为标准OpenAI格式
    """
    
    def __init__(self, config: Optional[KsanaConfig] = None, tokenizer=None):
        self.config = config or KsanaConfig()
        self.tokenizer = tokenizer
    
    def convert_chat_completion(
        self,
        request: ChatCompletionRequest,
        messages_to_prompt_func: Optional[Callable[[List[ChatMessage]], str]] = None
    ) -> Dict[str, Any]:
        return self._convert_base_request(
            request,
            api_type="chat",
            messages_to_prompt_func=messages_to_prompt_func
        )
    
    def convert_completion(
        self,
        request: CompletionRequest
    ) -> Dict[str, Any]:
        return self._convert_base_request(request, api_type="completion")
    
    def convert_embedding(
        self,
        request: EmbeddingRequest
    ) -> Dict[str, Any]:
        return self._convert_base_request(request, api_type="embedding")
    
    def convert_json_schema2str(self, json_schema: Union[dict, str, Type[BaseModel]]) -> str:
        """
        Convert a JSON schema to a string.
        """
        if isinstance(json_schema, dict):
            schema_str = orjson.dumps(json_schema).decode()
        elif isinstance(json_schema, str):
            schema_str = json_schema
        elif isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
            schema_str = orjson.dumps(json_schema.model_json_schema()).decode()
        else:
            raise ValueError(
                f"Cannot parse schema {json_schema}. The schema must be either "
                + "a Pydantic class, a dictionary or a string that contains the JSON "
                + "schema specification"
            )
        return schema_str

    def convert_to_ksana_format(
        self,
        request_dict: Dict[str, Any],
        api_type: str,
        messages_to_prompt_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        通用转换方法 - 兼容旧接口
        
        Args:
            request_dict: 请求字典
            api_type: API类型 ("chat", "completion", "embedding")
            messages_to_prompt_func: 消息转换函数
        """
        # Create Temporary Request Object
        if api_type == "chat":
            request = ChatCompletionRequest(**request_dict)
            return self.convert_chat_completion(request, messages_to_prompt_func)
        elif api_type == "completion":
            request = CompletionRequest(**request_dict)
            return self.convert_completion(request)
        elif api_type == "embedding":
            request = EmbeddingRequest(**request_dict)
            return self.convert_embedding(request)
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
    
    def _convert_base_request(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest, EmbeddingRequest],
        api_type: str,
        messages_to_prompt_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        基础请求转换逻辑
        
        Args:
            request: OpenAI格式的请求对象
            api_type: API类型
            messages_to_prompt_func: 消息转换函数（仅用于chat API）
        """

        ksana_request = {
            "stream": getattr(request, 'stream', False),
            "sampling_config": self._build_sampling_config(request)
        }

        if api_type == "chat":
            self._handle_chat_input(request, ksana_request, messages_to_prompt_func)
        elif api_type == "completion":
            self._handle_completion_input(request, ksana_request)
        elif api_type == "embedding":
            self._handle_embedding_input(request, ksana_request)
        
        # 处理停止条件
        self._handle_stop_conditions(request, ksana_request)
        
        # 处理扩展参数
        self._handle_extended_params(request, ksana_request)
        
        return ksana_request
    
    def _build_sampling_config(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest, EmbeddingRequest]
    ) -> Dict[str, Any]:

        config = {}

        do_sample = getattr(request, 'do_sample', None)
        config["do_sample"] = do_sample if do_sample is not None else self.config.default_do_sample
        config["temperature"] = getattr(request, 'temperature', None) or self.config.default_temperature
        config["topp"] = getattr(request, 'top_p', None) or self.config.default_top_p
        config["topk"] = getattr(request, 'top_k', None) or self.config.default_top_k
        
        config["repetition_penalty"] = getattr(request, 'repetition_penalty', None) \
            or self.config.default_repetition_penalty
        config["length_penalty"] = getattr(request, 'length_penalty', self.config.default_length_penalty)
        config["presence_penalty"] = getattr(request, 'presence_penalty', 0.0)
        config["frequency_penalty"] = getattr(request, 'frequency_penalty', 0.0)
        
        num_beams = getattr(request, 'num_beams', None)
        config["num_beams"] = num_beams if num_beams is not None else self.config.default_num_beams
        
        # OpenAI的n参数对应KsanaLLM的num_return_sequences
        config["num_return_sequences"] = getattr(request, 'n', None)
                
        no_repeat_ngram_size = getattr(request, 'no_repeat_ngram_size', None)
        config["no_repeat_ngram_size"] = no_repeat_ngram_size if no_repeat_ngram_size is not None \
                                                                else self.config.default_no_repeat_ngram_size
        
        encoder_no_repeat_ngram_size = getattr(request, 'encoder_no_repeat_ngram_size', None)
        config["encoder_no_repeat_ngram_size"] = encoder_no_repeat_ngram_size \
            if encoder_no_repeat_ngram_size is not None else self.config.default_encoder_no_repeat_ngram_size
        
        decoder_no_repeat_ngram_size = getattr(request, 'decoder_no_repeat_ngram_size', None)
        config["decoder_no_repeat_ngram_size"] = decoder_no_repeat_ngram_size \
            if decoder_no_repeat_ngram_size is not None else self.config.default_decoder_no_repeat_ngram_size
        
        # 输出控制 - 处理 logprobs 和 top_logprobs 的组合
        logprobs = getattr(request, 'logprobs', None)
        top_logprobs = getattr(request, 'top_logprobs', 0)
        if isinstance(logprobs, bool):
            config["logprobs"] = top_logprobs if logprobs and top_logprobs > 0 else 0
        elif logprobs is not None:
            config["logprobs"] = logprobs
        else:
            config["logprobs"] = self.config.default_logprobs
                
        ignore_eos = getattr(request, 'ignore_eos', None)
        config["ignore_eos"] = ignore_eos if ignore_eos is not None else self.config.default_ignore_eos
        
        config["echo"] = getattr(request, 'echo', False)
        
        config["min_tokens"] = getattr(request, 'min_tokens', 0)
        config["skip_special_tokens"] = getattr(request, 'skip_special_tokens', True)
        config["spaces_between_special_tokens"] = getattr(request, 'spaces_between_special_tokens', True)
        config["include_stop_str_in_output"] = getattr(request, 'include_stop_str_in_output', False)
        
        max_tokens = getattr(request, 'max_completion_tokens', None) or getattr(request, 'max_tokens', None)
        if max_tokens:
            config["max_new_tokens"] = max_tokens
        
        if hasattr(request, 'truncate_prompt_tokens') and request.truncate_prompt_tokens:
            config["truncate_prompt_tokens"] = request.truncate_prompt_tokens
        
        if hasattr(request, 'seed') and request.seed is not None:
            config["seed"] = request.seed
        
        if config["num_return_sequences"] > config["num_beams"]:
            config["num_return_sequences"] = config["num_beams"]

        # handle response_format
        # Notice: If response_format is not Null, enable_structured_output is always True
        if hasattr(request, 'response_format') and request.response_format:
            config["enable_structured_output"] = True
            if(request.response_format.type == "json_schema"):
                schema_ = request.response_format.json_schema.schema_
                assert schema_ is not None, "json_schema should not be None"
                config["json_schema"] = self.convert_json_schema2str(schema_)
            elif(request.response_format.type == "json_object"):
                config["json_schema"] = '{"type": "object"}'
        if hasattr(request, 'enable_structured_output') and request.enable_structured_output:
            config["enable_structured_output"] = request.enable_structured_output
        return config
    
    def _handle_chat_input(
        self,
        request: ChatCompletionRequest,
        ksana_request: Dict[str, Any],
        messages_to_prompt_func: Optional[Callable]
    ):
        # Use input_tokens as the primary input if available
        if hasattr(request, 'input_tokens') and request.input_tokens:
            ksana_request["input_tokens"] = request.input_tokens
        elif messages_to_prompt_func and request.messages:
            prompt = messages_to_prompt_func(request.messages)
            ksana_request["prompt"] = prompt
        else:
            messages_text = []
            for msg in request.messages:
                role_text = f"{msg.role.capitalize()}: {msg.content or ''}"
                messages_text.append(role_text)
            messages_text.append("Assistant:")
            ksana_request["prompt"] = "\n".join(messages_text)
        
        if hasattr(request, 'tools') and request.tools:
            ksana_request["tools"] = [tool.model_dump() for tool in request.tools]
            
            if hasattr(request, 'tool_choice') and request.tool_choice is not None:
                ksana_request["tool_choice"] = request.tool_choice
            elif hasattr(self.config, 'enable_auto_tool_choice') and self.config.enable_auto_tool_choice:
                # 如果启用了auto_tool_choice，默认为"auto"
                ksana_request["tool_choice"] = "auto"
            else:
                # 否则默认为"auto"（当有工具时，应该默认为auto而不是none）
                ksana_request["tool_choice"] = "auto"
        
    def _handle_completion_input(
        self,
        request: CompletionRequest,
        ksana_request: Dict[str, Any]
    ):
        if request.prompt:
            if isinstance(request.prompt, str):
                ksana_request["prompt"] = request.prompt
            elif isinstance(request.prompt, list):
                if request.prompt and isinstance(request.prompt[0], str):
                    ksana_request["prompt"] = request.prompt[0]
                elif request.prompt and isinstance(request.prompt[0], int):
                    ksana_request["input_tokens"] = request.prompt
                else:
                    ksana_request["input_tokens"] = request.prompt[0] if request.prompt else []
        
        if hasattr(request, 'suffix') and request.suffix:
            ksana_request["suffix"] = request.suffix

    def _handle_embedding_input(
        self,
        request: EmbeddingRequest,
        ksana_request: Dict[str, Any]
    ):
        if isinstance(request.input, str):
            ksana_request["input"] = request.input
        elif isinstance(request.input, list):
            # str list
            if request.input and isinstance(request.input[0], str):
                ksana_request["input"] = request.input
            # Token list
            elif request.input and isinstance(request.input[0], int):
                ksana_request["input_tokens"] = request.input
            else:
                ksana_request["input_tokens"] = request.input
        
        if hasattr(request, 'encoding_format'):
            ksana_request["encoding_format"] = request.encoding_format
        if hasattr(request, 'dimensions') and request.dimensions:
            ksana_request["dimensions"] = request.dimensions
    
    def _handle_stop_conditions(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest, EmbeddingRequest],
        ksana_request: Dict[str, Any]
    ):

        if hasattr(request, 'stop') and request.stop:
            stop_words = [request.stop] if isinstance(request.stop, str) else request.stop
            ksana_request["sampling_config"]["stop_words"] = stop_words

        if hasattr(request, 'stop_token_ids') and request.stop_token_ids:
            ksana_request["sampling_config"]["stop_token_ids"] = request.stop_token_ids
    
    def _handle_extended_params(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest, EmbeddingRequest],
        ksana_request: Dict[str, Any]
    ):
        if hasattr(request, 'user') and request.user:
            ksana_request["user"] = request.user
        
        if hasattr(request, 'logit_bias') and request.logit_bias:
            ksana_request["sampling_config"]["logit_bias"] = request.logit_bias
        
        # 处理KsanaLLM特定的扩展参数
        extended_params = [
            'structured_output_regex', 'model_type', 'use_chat_template',
            'encoder_no_repeat_ngram_size', 'decoder_no_repeat_ngram_size',
            'stop_strings', 'allowed_token_ids'
        ]
        
        for param in extended_params:
            if hasattr(request, param):
                value = getattr(request, param)
                if value is not None:
                    if param in ['stop_strings', 'allowed_token_ids']:
                        ksana_request["sampling_config"][param] = value
                    else:
                        ksana_request[param] = value
        
        # handle Extra Fields
        if hasattr(request, '__pydantic_extra__') and request.__pydantic_extra__:
            self._handle_extra_fields(request.__pydantic_extra__, ksana_request)
    
    def _handle_extra_fields(
        self,
        extra_fields: Dict[str, Any],
        ksana_request: Dict[str, Any]
    ):

        ksana_param_mapping = {
            'num_beams': ('sampling_config', 'num_beams'),
            'num_return_sequences': ('sampling_config', 'num_return_sequences'),
            'length_penalty': ('sampling_config', 'length_penalty'),
            'repetition_penalty': ('sampling_config', 'repetition_penalty'),
            
            'no_repeat_ngram_size': ('sampling_config', 'no_repeat_ngram_size'),
            'encoder_no_repeat_ngram_size': ('sampling_config', 'encoder_no_repeat_ngram_size'),
            'decoder_no_repeat_ngram_size': ('sampling_config', 'decoder_no_repeat_ngram_size'),
            
            'stop_token_ids': ('sampling_config', 'stop_token_ids'),
            'stop_strings': ('sampling_config', 'stop_strings'),
            'ignore_eos': ('sampling_config', 'ignore_eos'),
            
            'logprobs': ('sampling_config', 'logprobs'),
            'structured_output_regex': (None, 'structured_output_regex'),
            
            'model_type': (None, 'model_type'),
            'use_chat_template': (None, 'use_chat_template'),
            'input_tokens': (None, 'input_tokens'),
            
            'do_sample': ('sampling_config', 'do_sample'),
            'max_tokens': ('sampling_config', 'max_new_tokens'),
            'top_k': ('sampling_config', 'top_k'),
            'top_p': ('sampling_config', 'top_p'),
            'temperature': ('sampling_config', 'temperature'),
        }
        
        # 处理额外字段
        for field_name, field_value in extra_fields.items():
            if field_name in ksana_param_mapping and field_value is not None:
                config_section, param_name = ksana_param_mapping[field_name]
                
                if config_section is None:
                    ksana_request[param_name] = field_value
                else:
                    if config_section not in ksana_request:
                        ksana_request[config_section] = {}
                    ksana_request[config_section][param_name] = field_value
            else:
                logger.warning(f"Unknown extra parameter: {field_name}")
        
        if 'stop' in extra_fields and extra_fields['stop'] is not None:
            stop_value = extra_fields['stop']
            stop_words = [stop_value] if isinstance(stop_value, str) else stop_value
            ksana_request["sampling_config"]["stop_words"] = stop_words
    
    # ===== 输出格式化方法 =====
    
    def _count_tokens_accurate(self, text: str) -> int:
        """精确计算token数量"""
        if not text:
            return 0
            
        try:
            if self.tokenizer is not None:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Token counting failed, using estimation: {e}")
        
        # 回退到估算方法
        return max(1, len(text) // int(self.config.token_estimation_factor))
    
    def _convert_ksana_logprobs_to_openai(
        self,
        ksana_logprobs: Any,
        token_ids: Optional[List[int]] = None,
        num_output_top_logprobs: int = 0
    ) -> Optional[ChoiceLogprobs]:
        """将KsanaLLM的logprobs转换为OpenAI格式
        
        KsanaLLM格式: std::vector<std::vector<std::vector<std::pair<int, float>>>>
        - 第一层：批次
        - 第二层：序列位置
        - 第三层：每个位置的top-k候选 (token_id, logprob)
        """
        if ksana_logprobs is None:
            return None
        
        try:
            content_list = []
            
            # 处理第一个批次的结果
            if not isinstance(ksana_logprobs, list) or len(ksana_logprobs) == 0:
                return None
                
            batch_logprobs = ksana_logprobs[0]
            if not isinstance(batch_logprobs, list):
                return None
            
            # 如果提供了实际生成的token_ids，使用它们；否则使用第一个候选
            for position_idx, position_logprobs in enumerate(batch_logprobs):
                if not isinstance(position_logprobs, list) or len(position_logprobs) == 0:
                    continue
                
                # 确定当前位置的token和logprob
                if token_ids and position_idx < len(token_ids):
                    # 使用实际生成的token
                    current_token_id = token_ids[position_idx]
                    current_logprob = -9999.0
                    # 在候选中查找对应的logprob
                    for tid, lp in position_logprobs:
                        if tid == current_token_id:
                            current_logprob = lp
                            break
                else:
                    # 使用第一个候选
                    current_token_id, current_logprob = position_logprobs[0]
                
                # 解码token为文本
                token_text = self._decode_token(current_token_id)
                
                # 构建top_logprobs列表
                top_logprobs_list = []
                if num_output_top_logprobs > 0:
                    # 取前N个候选
                    for tid, lp in position_logprobs[:num_output_top_logprobs]:
                        candidate_text = self._decode_token(tid)
                        top_logprobs_list.append(TopLogprob(
                            token=candidate_text,
                            logprob=float(lp),
                            bytes=list(candidate_text.encode("utf-8", errors="replace"))
                        ))
                
                # 添加到content列表
                content_list.append(ChatCompletionTokenLogprob(
                    token=token_text,
                    logprob=float(current_logprob),
                    bytes=list(token_text.encode("utf-8", errors="replace")),
                    top_logprobs=top_logprobs_list
                ))
            
            return ChoiceLogprobs(content=content_list) if content_list else None
            
        except (IndexError, TypeError, ValueError, AttributeError) as e:
            logger.error(f"Failed to convert logprobs: {e}")
            logger.debug(f"Logprobs data: {ksana_logprobs}")
            return None
    
    def _decode_token(self, token_id: int) -> str:
        try:
            if self.tokenizer:
                return self.tokenizer.decode([token_id])
            else:
                return f"<token_{token_id}>"
        except (AttributeError, TypeError, ValueError):
            return f"<token_{token_id}>"
    
    def _determine_finish_reason(
        self,
        ksana_output: Any,
        stop_strings: List[str] = None,
        max_tokens_reached: bool = False
    ) -> str:
        
        # 检查是否达到最大token限制
        if max_tokens_reached:
            return "length"
        
        # 检查是否遇到停止字符串
        if hasattr(ksana_output, 'finish_reason'):
            finish_reason = getattr(ksana_output, 'finish_reason', None)
            if finish_reason:
                # 映射KsanaLLM的finish_reason到OpenAI标准
                reason_mapping = {
                    "stop": "stop",
                    "length": "length",
                    "eos": "stop",
                    "stop_string": "stop",
                    "max_tokens": "length"
                }
                return reason_mapping.get(finish_reason, "stop")
        
        return "stop"
    
    def format_chat_completion_response(
        self,
        request_id: str,
        model_name: str,
        ksana_output: Any,
        prompt_text: str = "",
        prompt_tokens: Optional[int] = None,
        request_logprobs: bool = False,
        max_tokens: Optional[int] = None,
        decode_func: Optional[Callable] = None,
        num_choices: int = 1,
        processed_outputs: Optional[List[Any]] = None,
        extracted_tool_calls: Optional[List[Any]] = None
    ) -> ChatCompletionResponse:
        
        try:
            choices = []
            total_completion_tokens = 0
            
            # 处理多个choices
            outputs_to_process = processed_outputs if processed_outputs else []
            if not outputs_to_process and hasattr(ksana_output, 'output_tokens'):
                if isinstance(ksana_output.output_tokens, list) and num_choices > 1:
                    # 如果是列表且需要多个choice，取前num_choices个
                    outputs_to_process = ksana_output.output_tokens[:num_choices]
                else:
                    # 单个输出或只需要一个choice
                    outputs_to_process = [ksana_output.output_tokens] * num_choices
            
            # 为每个choice生成响应
            for choice_idx in range(num_choices):
                output_text = ""
                completion_tokens = 0
                formatted_logprobs = None
                
                # 获取当前choice的输出
                if choice_idx < len(outputs_to_process):
                    current_output = outputs_to_process[choice_idx]
                    
                    if decode_func:
                        try:
                            # 确保我们传递正确的格式给decode函数
                            if isinstance(current_output, list):
                                # 如果是嵌套列表，取第一个元素
                                if len(current_output) > 0 and isinstance(current_output[0], list):
                                    output_text = decode_func(current_output[0], False)
                                else:
                                    output_text = decode_func(current_output, False)
                            else:
                                output_text = decode_func(current_output, False)
                        except (TypeError, ValueError, IndexError) as e:
                            logger.warning(f"Decode function failed for choice {choice_idx}: {e}")
                            # 回退到字符串转换
                            if hasattr(current_output, 'text'):
                                output_text = current_output.text
                            else:
                                output_text = str(current_output)
                    elif hasattr(current_output, 'text'):
                        output_text = current_output.text
                    else:
                        output_text = str(current_output)
                    
                    completion_tokens = self._count_tokens_accurate(output_text)
                    
                    # 处理logprobs
                    if request_logprobs and hasattr(ksana_output, 'logprobs'):
                        # 获取实际生成的token IDs（如果需要匹配）
                        actual_token_ids = None
                        if isinstance(current_output, list):
                            actual_token_ids = current_output
                        
                        formatted_logprobs = self._convert_ksana_logprobs_to_openai(
                            ksana_output.logprobs,
                            token_ids=actual_token_ids,
                            num_output_top_logprobs=getattr(request_logprobs, 'value', 5) \
                                if hasattr(request_logprobs, 'value') else 0
                        )
                
                # 获取当前choice的工具调用
                current_tool_calls = None
                if extracted_tool_calls and choice_idx < len(extracted_tool_calls):
                    current_tool_calls = extracted_tool_calls[choice_idx]
                
                # 确定finish_reason
                max_tokens_reached = (max_tokens is not None and completion_tokens >= max_tokens)
                if current_tool_calls:
                    finish_reason = "tool_calls"
                else:
                    finish_reason = self._determine_finish_reason(
                        ksana_output,
                        max_tokens_reached=max_tokens_reached
                    )
                
                # 构建choice
                choice = ChatCompletionResponseChoice(
                    index=choice_idx,
                    message=ChatMessage(
                        role="assistant",
                        content=output_text if not current_tool_calls else None,
                        tool_calls=current_tool_calls
                    ),
                    logprobs=formatted_logprobs,
                    finish_reason=finish_reason
                )
                
                choices.append(choice)
                total_completion_tokens += completion_tokens
            
            # 计算prompt tokens
            if prompt_tokens is None:
                prompt_tokens = self._count_tokens_accurate(prompt_text)
            
            # 构建usage信息
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=prompt_tokens + total_completion_tokens
            )
            
            # 构建响应
            response = ChatCompletionResponse(
                id=request_id,
                object="chat.completion",
                created=int(time.time()),
                model=model_name,
                choices=choices,
                usage=usage
            )
            
            return response
            
        except (TypeError, ValueError, AttributeError) as e:
            logger.error(f"Failed to format chat completion response: {e}")
            return ChatCompletionResponse(
                id=request_id,
                object="chat.completion",
                created=int(time.time()),
                model=model_name,
                choices=[ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=""),
                    finish_reason="stop"
                )],
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            )
    
    def format_chat_completion_stream_chunk(
        self,
        request_id: str,
        model_name: str,
        content: Optional[str] = None,
        role: Optional[str] = None,
        finish_reason: Optional[str] = None,
        logprobs: Optional[Any] = None,
        usage: Optional[UsageInfo] = None,
        choice_index: int = 0
    ) -> str:
        """格式化Chat Completion流式响应块"""
        
        try:
            delta = DeltaMessage()
            if role:
                delta.role = role
                delta.content = ""
            elif content is not None:
                delta.content = content
            
            formatted_logprobs = None
            if logprobs is not None:
                formatted_logprobs = self._convert_ksana_logprobs_to_openai(logprobs)
            
            # 构建choice
            choice = ChatCompletionResponseStreamChoice(
                index=choice_index,
                delta=delta,
                logprobs=formatted_logprobs,
                finish_reason=finish_reason
            )
            
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model_name,
                choices=[choice],
                usage=usage
            )
            
            return f"data: {orjson.dumps(chunk.model_dump(exclude_unset=True)).decode()}\n\n"
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Failed to format stream chunk: {e}")
            return "data: {}\n\n"
    
    def format_completion_stream_chunk(
        self,
        request_id: str,
        model_name: str,
        content: Optional[str] = None,
        finish_reason: Optional[str] = None,
        logprobs: Optional[Any] = None,
        usage: Optional[UsageInfo] = None,
        choice_index: int = 0
    ) -> str:
        """格式化Completion流式响应块"""
        
        try:
            formatted_logprobs = None
            if logprobs is not None:
                formatted_logprobs = self._convert_ksana_logprobs_to_openai(logprobs)
            # 构建choice
            choice = CompletionResponseStreamChoice(
                index=choice_index,
                text=content or "",
                logprobs=formatted_logprobs,  # 这里直接使用原始的logprobs，因为Completion API的格式可能不同
                finish_reason=finish_reason
            )
            
            chunk = CompletionStreamResponse(
                id=request_id,
                object="text_completion",
                created=int(time.time()),
                model=model_name,
                choices=[choice],
                usage=usage
            )
            
            return f"data: {orjson.dumps(chunk.model_dump(exclude_unset=True)).decode()}\n\n"
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Failed to format completion stream chunk: {e}")
            return "data: {}\n\n"
    
    def format_completion_response(
        self,
        request_id: str,
        model_name: str,
        ksana_output: Any,
        prompt_text: str = "",
        prompt_tokens: Optional[int] = None,
        echo: bool = False,
        request_logprobs: Optional[int] = None,
        decode_func: Optional[Callable] = None
    ) -> CompletionResponse:
        
        try:
            output_text = ""
            if hasattr(ksana_output, 'output_tokens') and ksana_output.output_tokens:
                if decode_func:
                    output_text = decode_func(ksana_output.output_tokens[0], False)
                else:
                    output_text = str(ksana_output.output_tokens[0])
            
            if echo:
                output_text = prompt_text + output_text
            
            if prompt_tokens is None:
                prompt_tokens = self._count_tokens_accurate(prompt_text)
            completion_tokens = self._count_tokens_accurate(output_text)
            total_tokens = prompt_tokens + completion_tokens
            
            logprobs = None
            if request_logprobs and request_logprobs > 0:
                # TODO: 实现Completion API的logprobs格式转换
                logprobs = CompletionLogProbs(
                    text_offset=[],
                    token_logprobs=[],
                    tokens=[],
                    top_logprobs=[]
                )
            
            choice = CompletionResponseChoice(
                index=0,
                text=output_text,
                logprobs=logprobs,
                finish_reason=self._determine_finish_reason(ksana_output)
            )
            
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            
            response = CompletionResponse(
                id=request_id,
                object="text_completion",
                created=int(time.time()),
                model=model_name,
                choices=[choice],
                usage=usage
            )
            
            return response
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Failed to format completion response: {e}")
            return CompletionResponse(
                id=request_id,
                object="text_completion",
                created=int(time.time()),
                model=model_name,
                choices=[CompletionResponseChoice(
                    index=0,
                    text="",
                    finish_reason="stop"
                )],
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            )
    
    def format_embedding_response(
        self,
        request_id: str,
        model_name: str,
        embeddings: List[List[float]],
        input_texts: List[str],
        encoding_format: str = "float"
    ) -> EmbeddingResponse:
        """格式化Embedding响应"""
        
        try:
            data = []
            total_tokens = 0
            
            for i, (embedding, text) in enumerate(zip(embeddings, input_texts)):
                if encoding_format == "base64":
                    import base64
                    import struct
                    float_bytes = b''.join(struct.pack('f', f) for f in embedding)
                    embedding_data = base64.b64encode(float_bytes).decode('utf-8')
                else:
                    embedding_data = embedding
                
                data.append(EmbeddingResponseData(
                    index=i,
                    object="embedding",
                    embedding=embedding_data
                ))
                
                total_tokens += self._count_tokens_accurate(text)
            
            usage = UsageInfo(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens
            )
            
            response = EmbeddingResponse(
                id=request_id,
                object="list",
                created=int(time.time()),
                model=model_name,
                data=data,
                usage=usage
            )
            
            return response
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Failed to format embedding response: {e}")
            return EmbeddingResponse(
                id=request_id,
                object="list",
                created=int(time.time()),
                model=model_name,
                data=[],
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            )
    
    def format_error_response(
        self,
        error_message: str,
        error_type: str = "internal_server_error",
        error_code: int = 500,
        param: Optional[str] = None
    ) -> Dict[str, Any]:
        """格式化错误响应"""
        return {
            "error": {
                "message": error_message,
                "type": error_type,
                "code": error_code,
                "param": param
            }
        }
    
    def format_streaming_error(
        self,
        error_message: str,
        error_type: str = "internal_server_error",
        error_code: int = 500
    ) -> str:
        """格式化流式错误响应"""
        error_data = self.format_error_response(error_message, error_type, error_code)
        import orjson
        return f"data: {orjson.dumps(error_data).decode()}\n\n"
