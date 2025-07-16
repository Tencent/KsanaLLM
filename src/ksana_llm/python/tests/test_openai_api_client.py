# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys
import tempfile
import shutil
import logging
import json
import time
import threading
import pytest
import pytest_asyncio
import uvicorn
from openai import OpenAI, AsyncOpenAI

# 调整系统路径以导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from serving_server import LLMServer
from ksana_llm.arg_utils import EngineArgs
from utils import modify_yaml_field

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


class RemoteKsanaServer:
    """管理远程 Ksana LLM 服务器的上下文管理器"""
    
    def __init__(self, model_dir: str, yaml_path: str, port: int = 8000):
        self.model_dir = model_dir
        self.yaml_path = yaml_path
        self.port = port
        self.server = None
        self.server_thread = None
        self.base_url = f"http://localhost:{self.port}/v1"
        self.uvicorn_server = None
        
    def __enter__(self):
        """启动服务器"""
        args = EngineArgs.from_config_file(self.yaml_path)
        # 设置 OpenAI 相关参数
        args.tool_call_parser = "hermes"
        args.reasoning_parser = "qwen3"
        args.enable_auto_tool_choice = True
        args.tool_parser_plugin = None
        args.chat_template = None
        args.chat_template_content_format = None
        args.host = "127.0.0.1"
        args.port = self.port
        
        self.server = LLMServer(args)
        self.server.initialize()
        
        config = uvicorn.Config(
            app=self.server.app,
            host="127.0.0.1",
            port=self.port,
            log_level="error"
        )
        self.uvicorn_server = uvicorn.Server(config)
        
        def run_server():
            self.uvicorn_server.run()
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        self._wait_for_server()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """停止服务器"""
        if self.uvicorn_server:
            self.uvicorn_server.should_exit = True
        if self.server_thread:
            self.server_thread.join(timeout=5)
            
    def _wait_for_server(self, timeout: int = 90):
        """等待服务器启动"""
        import requests
        start_time = time.time()
        
        # 尝试不同的端点
        endpoints = [
            f"http://localhost:{self.port}/health",
            f"http://localhost:{self.port}/models",
        ]
        
        while time.time() - start_time < timeout:
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code in [200, 404]:  # 404 也表示服务器在运行
                        logger.info(f"服务器已在端口 {self.port} 上启动 (endpoint: {endpoint}, status: {response.status_code})")
                        time.sleep(2)
                        return
                except requests.exceptions.RequestException as e:
                    logger.debug(f"尝试连接 {endpoint} 失败: {e}")
                    pass
            
            time.sleep(1)
            
        # 最后尝试获取更多调试信息
        try:
            response = requests.get(f"http://localhost:{self.port}/", timeout=5)
            logger.error(f"服务器响应根路径: {response.status_code} - {response.text[:200]}")
        except requests.exceptions.RequestException as e:
            logger.error(f"无法连接到服务器: {e}")
            
        raise TimeoutError(f"服务器在 {timeout} 秒内未能启动")
        
    def get_client(self) -> OpenAI:
        """获取同步 OpenAI 客户端"""
        return OpenAI(
            api_key="EMPTY",  # Ksana 不需要 API key
            base_url=self.base_url
        )
        
    def get_async_client(self) -> AsyncOpenAI:
        """获取异步 OpenAI 客户端"""
        return AsyncOpenAI(
            api_key="EMPTY",  # Ksana 不需要 API key
            base_url=self.base_url
        )


class TestOpenAIAPIClient:
    """使用 OpenAI 客户端测试 LLMServer 的 OpenAI API 兼容性"""
    
    @pytest.fixture(scope="class")
    def server(self, default_ksana_yaml_path):
        # 设置默认的模型目录
        model_dir = "/models/qwen3-8B"
        # 使用提供的默认 yaml 路径
        yaml_path = default_ksana_yaml_path
            
        # 创建临时目录和配置文件副本
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"创建临时目录: {temp_dir}")
        
        try:
            # 复制 YAML 到临时目录
            temp_yaml_path = os.path.join(temp_dir, "ksana.yaml")
            shutil.copyfile(yaml_path, temp_yaml_path)
            
            # 修改 YAML 配置 - 使用点分隔的字符串路径
            modify_yaml_field(temp_yaml_path, "model_spec.base_model.model_dir", model_dir)
            
            # 启动服务器
            with RemoteKsanaServer(model_dir, temp_yaml_path, port=8001) as server:
                yield server
                
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"清理临时目录: {temp_dir}")
            
    @pytest_asyncio.fixture
    async def async_client(self, server):
        """获取异步 OpenAI 客户端"""
        client = server.get_async_client()
        yield client
            
    @pytest.fixture
    def client(self, server):
        """获取同步 OpenAI 客户端"""
        return server.get_client()
        
    @pytest.mark.asyncio
    async def test_chat_completions_non_streaming(self, async_client: AsyncOpenAI):
        """测试非流式聊天完成"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        response = await async_client.chat.completions.create(
            model="ksana-llm",
            messages=messages,
            max_tokens=50,
            temperature=0.7,
            stream=False
        )
        
        assert response.id is not None
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        
        choice = response.choices[0]
        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content is not None
        assert len(choice.message.content) > 0
        assert choice.finish_reason in ["stop", "length", "max_tokens", None]
        
        # 检查 usage
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens
        
    @pytest.mark.asyncio
    async def test_chat_completions_streaming(self, async_client: AsyncOpenAI):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count from 1 to 5."}
        ]
        
        stream = await async_client.chat.completions.create(
            model="ksana-llm",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            stream=True
        )
        
        chunks = []
        finish_reason_count = 0
        
        async for chunk in stream:
            chunks.append(chunk)
            assert chunk.id is not None
            assert chunk.object == "chat.completion.chunk"
            assert chunk.model is not None
            
            if chunk.choices:
                choice = chunk.choices[0]
                assert choice.index == 0
                
                if choice.delta.role:
                    assert choice.delta.role == "assistant"
                    
                if choice.finish_reason is not None:
                    finish_reason_count += 1
                    assert choice.finish_reason in ["stop", "length"]
        
        print(f"\n\n\nReceived {len(chunks)} chunks with finish_reason count: {finish_reason_count}\n\n\n")

        # 确保至少收到一些块
        assert len(chunks) > 0
        # 确保只有最后一个块有 finish_reason
        assert finish_reason_count == 1
        
        # 重建完整响应
        full_response = ""
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content
                    print(f"Chunk content: {delta.content}")
                # 检查是否有 reasoning_content 属性（这是 KsanaLLM 的扩展）
                elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    full_response += delta.reasoning_content
                    print(f"Chunk reasoning content: {delta.reasoning_content}")
                
        assert len(full_response) > 0
        
    @pytest.mark.asyncio
    async def test_chat_with_logprobs(self, async_client: AsyncOpenAI):
        """测试带 logprobs 的聊天完成"""
        messages = [
            {"role": "user", "content": "Say 'hello world'"}
        ]
        
        response = await async_client.chat.completions.create(
            model="ksana-llm",
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            logprobs=True,
            top_logprobs=3
        )
        
        choice = response.choices[0]
        assert choice.logprobs is not None
        assert choice.logprobs.content is not None
        
        # 检查每个 token 的 logprobs
        for token_logprobs in choice.logprobs.content:
            assert token_logprobs.token is not None
            assert token_logprobs.logprob is not None
            assert len(token_logprobs.top_logprobs) <= 3
            
            for top_logprob in token_logprobs.top_logprobs:
                assert top_logprob.token is not None
                assert top_logprob.logprob is not None
                
    @pytest.mark.asyncio
    async def test_completions_endpoint(self, async_client: AsyncOpenAI):
        """测试 /v1/completions 端点"""
        response = await async_client.completions.create(
            model="ksana-llm",
            prompt="Once upon a time",
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"Response: {response}")

        assert response.id is not None
        assert response.object == "text_completion"
        # 模型名称可能是实际的模型名
        assert response.model is not None
        assert len(response.choices) == 1
        
        choice = response.choices[0]
        assert choice.text is not None
        assert len(choice.text) > 0
        assert choice.finish_reason in ["stop", "length", "max_tokens", None]
        
    @pytest.mark.asyncio
    async def test_embeddings_endpoint(self, async_client: AsyncOpenAI):
        """测试 /v1/embeddings 端点"""
        response = await async_client.embeddings.create(
            model="ksana-llm",
            input="Hello, world!"
        )
        
        assert response.object == "list"
        assert len(response.data) == 1
        
        embedding = response.data[0]
        assert embedding.object == "embedding"
        assert embedding.index == 0
        assert isinstance(embedding.embedding, list)
        assert len(embedding.embedding) > 0
        assert all(isinstance(x, float) for x in embedding.embedding)
        
    @pytest.mark.asyncio
    async def test_tool_calling(self, async_client: AsyncOpenAI):
        """测试工具调用功能"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        messages = [
            {"role": "user", "content": "What's the weather like in Boston?"}
        ]
        
        response = await async_client.chat.completions.create(
            model="ksana-llm",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        choice = response.choices[0]
        message = choice.message
        print(f"Message content: {message.content}")
        # 检查是否有工具调用
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            assert tool_call.id is not None
            assert tool_call.type == "function"
            assert tool_call.function.name == "get_weather"
            
            # 解析参数
            args = json.loads(tool_call.function.arguments)
            assert "location" in args

            
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, async_client: AsyncOpenAI):
        """测试多轮对话"""
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 5 + 3?"},
        ]
        
        # 第一轮
        response1 = await async_client.chat.completions.create(
            model="ksana-llm",
            messages=messages,
            max_tokens=50
        )
        
        # 添加助手的回复
        messages.append({
            "role": "assistant",
            "content": response1.choices[0].message.content
        })
        
        # 第二轮
        messages.append({
            "role": "user",
            "content": "Now multiply that by 2"
        })
        
        response2 = await async_client.chat.completions.create(
            model="ksana-llm",
            messages=messages,
            max_tokens=50
        )
        
        assert response2.choices[0].message.content is not None
        # 确保对话有上下文连贯性
        
    def test_sync_client(self, client: OpenAI):
        """测试同步客户端"""
        # 测试聊天完成
        response = client.chat.completions.create(
            model="ksana-llm",
            messages=[
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=10
        )
        
        assert response.choices[0].message.content is not None
