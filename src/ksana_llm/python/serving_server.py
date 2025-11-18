# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import os
import signal
import sys
from typing import Dict

import orjson
import uvicorn
import uvloop
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import status as http_status
from transformers import logging

import ksana_llm
from ksana_llm.arg_utils import EngineArgs
from ksana_llm.ksana_engine import ReasoningConfig

try:
    from openaiapi.openai_adapter import add_openai_routes, OpenAIConfig
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"\033[33m------OpenAI modules not available: {e}--------\033[0m")
    print("OpenAI compatible routes will be disabled.")
    OPENAI_AVAILABLE = False

# 配置常量
CONFIG = {
    "TIMEOUT_KEEP_ALIVE": 200,
    "FIELDS_TO_EXTRACT": ["x-remote-ip", "kv-comm-group-key", "kv-comm-request-id", 'Content-Type'],
    "DEFAULT_LOG_LEVEL": "INFO",
}


class LLMServer:
    def __init__(self, engine_args: EngineArgs):
        self.app = FastAPI(title="Ksana LLM Server", version="API-1.0.0")
        self.model = None
        self.tokenizer = None
        self.pre_post_processor = None
        self.engine_args = engine_args
        self._setup_routes()

    def initialize(self) -> None:
        """初始化模型"""
        self.model = ksana_llm.KsanaLLMEngine.from_engine_args(self.engine_args)
        self.reasoning_parser = getattr(self.engine_args, 'reasoning_parser', None)
        
        # Prepare reasoning config if reasoning_parser is enabled
        reasoning_config = None
        if self.reasoning_parser is not None:
            try:
                from openaiapi.reasoning import ReasoningParserManager
                reasoning_parser_class = ReasoningParserManager.get_reasoning_parser(self.reasoning_parser)
                reasoning_parser_instance = reasoning_parser_class(self.model.tokenizer)
                if reasoning_parser_instance is None:
                    raise ValueError(f"Failed to create reasoning parser instance for: {self.reasoning_parser}")
                reasoning_config = ReasoningConfig(think_end_token_id=reasoning_parser_instance.think_end_token_id)
            except (ImportError, ValueError, AttributeError, TypeError) as e:
                raise RuntimeError(f"Failed to create reasoning config: {e}") from e

        # Initialize the model with reasoning config
        self.model.initialize(reasoning_config=reasoning_config)
        self.tokenizer = self.model.tokenizer
        self.pre_post_processor = self.model.pre_post_processor

        if OPENAI_AVAILABLE:
            try:
                # 创建OpenAI配置，传入解析器参数, 添加路由函数
                openai_config = OpenAIConfig(
                    tool_call_parser=getattr(self.engine_args, 'tool_call_parser', None),
                    reasoning_parser=self.reasoning_parser,
                    enable_auto_tool_choice=getattr(self.engine_args, 'enable_auto_tool_choice', False),
                    tool_parser_plugin=getattr(self.engine_args, 'tool_parser_plugin', None)
                )
                
                chat_template = getattr(self.engine_args, 'chat_template', None)
                chat_template_content_format = getattr(self.engine_args, 'chat_template_content_format', None)
                
                add_openai_routes(
                    self,
                    openai_config,
                    chat_template=chat_template,
                    chat_template_content_format=chat_template_content_format
                )
                print("\033[32m======OpenAI compatible routes added successfully========\033[0m\n")
            except AttributeError as e:
                print(f"\033[33m------OpenAI configuration error: {e}--------\033[0m")
                print("OpenAI compatible routes will be disabled.")
        else:
            print("\033[33m------OpenAI modules not available, skipping OpenAI routes--------\033[0m")

    def _setup_routes(self) -> None:
        """设置路由"""
        self.app.post("/generate")(self.generate)
        self.app.post("/forward")(self.forward)
        self.app.get("/health")(self.health_check)

    @staticmethod
    def get_trace_context(request: Request) -> Dict[str, str]:
        """获取请求的追踪上下文"""
        return {
            field: request.headers.get(field)
            for field in CONFIG["FIELDS_TO_EXTRACT"]
            if request.headers.get(field) is not None
        }


    async def format_streaming_output(self, ksana_python_output_iterator):
        """Perform streaming generation."""

        async def stream_results():
            async for ksana_python_output in ksana_python_output_iterator:
                if ksana_python_output is None:
                    return
                input_token_ids = ksana_python_output.input_tokens
                output_texts = []
                output_token_ids = []
                for request_output in ksana_python_output.output_tokens:
                    # Decode the output tokens using the tokenizer
                    output_token_ids.append(request_output)
                    try:
                        output_text = self.pre_post_processor.decode(request_output, True)
                    except Exception as e:
                        print(
                            "Exception occurred during decoding, invalid token ids:",
                            input_token_ids,
                        )
                        raise ValueError("Invalid token ids!") from e
                    output_texts.append(output_text)
                ret = {
                    "texts": output_texts,
                    "output_token_ids": output_token_ids,  # the output token IDs
                    "logprobs": ksana_python_output.logprobs,
                    "input_token_ids": input_token_ids,  # the input token IDs
                }
                yield orjson.dumps(ret) + b"\0"

        # Return the asynchronous generator
        return stream_results()


    def format_output(self, ksana_python_output):
        # Decode the output tokens into a human-readable text using the tokenizer
        output_text = []
        for tokens in ksana_python_output.output_tokens:
            try:
                output_text.append(self.pre_post_processor.decode(tokens, False))
            except Exception as e:
                print("Exception occurred during decoding, invalid token ids:", tokens)
                raise ValueError("Invalid token ids!") from e

        # Create a response with the generated text and token IDs
        return {
            "texts": output_text,  # the generated text
            "output_token_ids": ksana_python_output.output_tokens,  # the generated token IDs
            "logprobs": ksana_python_output.logprobs,
            "input_token_ids": ksana_python_output.input_tokens,  # the input token IDs
        }


    async def generate(self, request: Request) -> Response:
        """Generate completion for the request."""
        try:
            req_ctx = self.get_trace_context(request)
            request_dict = orjson.loads(await request.body())
            enable_streaming = request_dict.get("stream", True)
            status, output = await self.model.generate(
                request_dict=request_dict,
                streamer=enable_streaming,
                req_ctx=req_ctx,
                )
            if not status.OK():
                raise HTTPException(
                    status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "message": status.GetMessage(),
                        "code": status.GetCode().value,
                    },
                )

            if enable_streaming:
                response_stream = await self.format_streaming_output(output)
                return StreamingResponse(response_stream)
            else:
                response_data = self.format_output(output)
                return JSONResponse(response_data)

        except Exception as e:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


    async def forward(self, request: Request) -> Response:
        """Generate next token for the request.

        The request should be a JSON object packed by msgpack.
        """
        try:
            req_ctx = self.get_trace_context(request)
            request_bytes = await request.body()
            status, response_bytes = await self.model.forward(request_bytes, req_ctx)
            if not status.OK() or response_bytes is None:
                raise HTTPException(
                    status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "message": status.GetMessage(),
                        "code": status.GetCode().value,
                    },
                )

            return Response(content=response_bytes, media_type="application/x-msgpack")

        except Exception as e:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )


    async def health_check(self) -> Dict[str, str]:
        """健康检查接口"""
        return {"status": "healthy"}



def main():
    """主函数"""
    uvloop.install()
    logging.set_verbosity_error()

    engine_args = EngineArgs.from_command_line()
    server = LLMServer(engine_args)
    server.initialize()

    if engine_args.endpoint != "python":
        signal.pause()
        sys.exit(0)

    # Set the log level of uvicorn based on KLLM_LOG_LEVEL.
    log_level = os.getenv("KLLM_LOG_LEVEL", "INFO").upper()
    if log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        log_level = log_level.lower()
    else:
        log_level = "info"
        print(
            f"Uvicorn's logging not support env: KLLM_LOG_LEVEL={log_level}, keep it as defalt(info)."
        )

    # distributed config.
    world_size = int(os.environ.get('WORLD_SIZE', "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))

    # For standalone or distributed master node, listen on server port.
    # For distributed worker node, wait until cluster destroyed.
    if world_size == 1 or node_rank == 0:
        server.app.root_path = engine_args.root_path
        uvicorn.run(
            server.app,
            host=engine_args.host,
            port=engine_args.port,
            log_level=log_level,
            access_log=engine_args.access_log,
            timeout_keep_alive=CONFIG["TIMEOUT_KEEP_ALIVE"],
            ssl_keyfile=engine_args.ssl_keyfile,
            ssl_certfile=engine_args.ssl_certfile,
        )
    else:
        print("Uvicorn running on NONE.")
        import threading
        threading.Event().wait()

if __name__ == "__main__":
    main()
