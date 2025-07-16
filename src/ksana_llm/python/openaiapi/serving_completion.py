# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ==============================================================================
# Adapted from vLLM project
# [vLLM Project]
# Ref:
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/serving_completion.py
# ==============================================================================
"""
OpenAI Completions API服务实现
"""

import uuid
from typing import Dict, Any, Optional, Union
from http import HTTPStatus

import asyncio
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from openaiapi.openai_adapter import (
    ErrorType, 
    KsanaOpenAIServing, OpenAIConfig
)
from openaiapi.openai_protocol import (
    CompletionRequest
)
from openaiapi.request_converter import RequestConverter

from utilize.logger import get_logger

logger = get_logger(__name__)


class KsanaOpenAIServingCompletion(KsanaOpenAIServing):
    
    def __init__(self, llm_server, config: Optional[OpenAIConfig] = None):
        super().__init__(llm_server, config)
    
    async def create_completion(self, request: Request) -> Union[JSONResponse, Any]:
        """创建 completion 响应"""
        try:
            # 解析请求
            request_dict = await request.json()
            
            # 验证必需参数
            prompt = request_dict.get("prompt")
            if not prompt:
                return self.create_error_response(
                    "Missing required parameter: prompt",
                    ErrorType.BAD_REQUEST_ERROR,
                    HTTPStatus.BAD_REQUEST
                )
            
            try:
                completion_request = CompletionRequest(**request_dict)
            except (TypeError, ValueError, KeyError) as e:
                return self.create_error_response(
                    f"Invalid request format: {str(e)}",
                    ErrorType.VALIDATION_ERROR,
                    HTTPStatus.BAD_REQUEST
                )
            
            # 检查模型
            model_check_result = await self._check_model(completion_request)
            if model_check_result is not None:
                return model_check_result
            
            # 获取 tokenizer
            tokenizer = getattr(self.llm_server, 'tokenizer', None)
            if tokenizer is None:
                return self.create_error_response(
                    "Tokenizer not available",
                    ErrorType.INTERNAL_SERVER_ERROR,
                    HTTPStatus.INTERNAL_SERVER_ERROR
                )
            
            # Tokenize 输入
            try:
                if isinstance(prompt, str):
                    prompt_inputs = [await self._tokenize_prompt_input_async(
                        completion_request, tokenizer, prompt
                    )]
                elif isinstance(prompt, list):
                    if len(prompt) == 0:
                        return self.create_error_response(
                            "Prompt cannot be empty",
                            ErrorType.BAD_REQUEST_ERROR,
                            HTTPStatus.BAD_REQUEST
                        )
                    
                    prompt_inputs = await self._tokenize_prompt_input_or_inputs_async(
                        completion_request, tokenizer, prompt
                    )
                else:
                    return self.create_error_response(
                        "Prompt must be a string or list of strings",
                        ErrorType.BAD_REQUEST_ERROR,
                        HTTPStatus.BAD_REQUEST
                    )
            except ValueError as e:
                return self.create_error_response(
                    str(e),
                    ErrorType.BAD_REQUEST_ERROR,
                    HTTPStatus.BAD_REQUEST
                )
            except (AttributeError, RuntimeError) as e:
                logger.error(f"Tokenization error: {e}")
                return self.create_error_response(
                    "Failed to process input",
                    ErrorType.INTERNAL_SERVER_ERROR,
                    HTTPStatus.INTERNAL_SERVER_ERROR
                )
            
            # 转换为 Ksana 格式
            converter = RequestConverter(self.config, tokenizer=tokenizer)
            
            try:
                ksana_request = converter.convert_to_ksana_format(
                    request_dict,
                    api_type="completion"
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Request conversion error: {e}")
                return self.create_error_response(
                    f"Request conversion failed: {str(e)}",
                    ErrorType.VALIDATION_ERROR,
                    HTTPStatus.BAD_REQUEST
                )
            
            # 获取追踪上下文
            req_ctx = self._get_trace_context(request) if request else None
            
            # 生成请求 ID
            request_id = self._base_request_id(request, f"cmpl-{uuid.uuid4().hex}")
            
            # 记录输入日志
            if len(prompt_inputs) > 0:
                self._log_inputs(request_id, prompt_inputs[0], None)
            
            # 调用 KsanaLLM 生成
            try:
                status, output = await self.llm_server.model.generate(
                    request_dict=ksana_request,
                    streamer=completion_request.stream,
                    req_ctx=req_ctx,
                )
            except (RuntimeError, asyncio.TimeoutError, ConnectionError) as e:
                logger.error(f"Generation error: {e}")
                return self.create_error_response(
                    "Generation failed",
                    ErrorType.INTERNAL_SERVER_ERROR,
                    HTTPStatus.INTERNAL_SERVER_ERROR
                )
            
            if not status.OK():
                return self.create_error_response(
                    status.GetMessage(),
                    ErrorType.INTERNAL_SERVER_ERROR,
                    HTTPStatus.INTERNAL_SERVER_ERROR
                )
            
            # 处理响应
            model_name = self._get_model_name(completion_request.model)
            
            if completion_request.stream:
                # 流式响应
                return self._completion_stream_generator(
                    output, request_id, model_name, completion_request, prompt_inputs[0]
                )
            else:
                # 非流式响应
                response = converter.format_completion_response(
                    request_id=request_id,
                    model_name=model_name,
                    ksana_output=output,
                    prompt_text=prompt_inputs[0]["prompt"] if prompt_inputs else "",
                    echo=completion_request.echo,
                    request_logprobs=completion_request.logprobs,
                    decode_func=self.llm_server.pre_post_processor.decode
                )
                
                return JSONResponse(content=response.model_dump(exclude_none=True))
            
        except HTTPException:
            raise
        except (RuntimeError, SystemError, MemoryError) as e:
            logger.error(f"Unexpected error in create_completion: {e}")
            return self.create_error_response(
                "Internal server error",
                ErrorType.INTERNAL_SERVER_ERROR,
                HTTPStatus.INTERNAL_SERVER_ERROR
            )
    
    async def _completion_stream_generator(
        self,
        ksana_output_iterator,
        request_id: str,
        model_name: str,
        request,
        prompt_input: Dict[str, Any]
    ):
        """生成流式 completion 响应"""
        try:
            converter = RequestConverter(self.config, tokenizer=getattr(self.llm_server, 'tokenizer', None))
            sent_length = 0
            
            async for ksana_python_output in ksana_output_iterator:
                if ksana_python_output is None:
                    # 结束流
                    yield converter.format_completion_stream_chunk(
                        request_id=request_id,
                        model_name=model_name,
                        content=None,
                        finish_reason="stop"
                    )
                    break
                
                for request_output in ksana_python_output.output_tokens:
                    try:
                        full_text = self.llm_server.pre_post_processor.decode(request_output, True)
                        
                        if full_text and len(full_text) > sent_length:
                            delta_text = full_text[sent_length:]
                            sent_length = len(full_text)
                            
                            if delta_text:
                                yield converter.format_completion_stream_chunk(
                                    request_id=request_id,
                                    model_name=model_name,
                                    content=delta_text,
                                    logprobs=getattr(ksana_python_output, 'logprobs', None) \
                                        if request.logprobs else None
                                )
                    except (UnicodeDecodeError, ValueError, AttributeError) as e:
                        logger.error(f"Decoding error in completion stream: {e}")
                        continue
                        
        except asyncio.CancelledError:
            logger.info("Completion streaming cancelled by client")
            raise
        except (RuntimeError, ConnectionError, asyncio.TimeoutError) as e:
            logger.error(f"Completion streaming error: {e}")
            error_data = self.create_streaming_error_response(
                "Streaming failed", "internal_server_error", 500
            )
            yield f"data: {error_data}\n\n"
        finally:
            yield "data: [DONE]\n\n"
