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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, '../../../build/lib')
sys.path.insert(0, '.')

from serving_server import LLMServer
from ksana_llm.arg_utils import EngineArgs
from utils import modify_yaml_field

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("pydot").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class RemoteKsanaServer:
    
    def __init__(self, model_dir: str = None, yaml_path: str = None, port: int = 8000):
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
        args.host = "127.0.0.1"
        args.port = self.port
        
        self.server = LLMServer(args)

        self.server.initialize()
        logger.info("LLM 服务器初始化完成")
     
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
        if self.uvicorn_server:
            self.uvicorn_server.should_exit = True
        if self.server_thread:
            self.server_thread.join(timeout=5)
            
    def _wait_for_server(self, timeout: int = 90):
        import requests
        start_time = time.time()
        
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
                        
                        openai_endpoints = [
                            f"http://localhost:{self.port}/models",
                            f"http://localhost:{self.port}/v1/models",
                            f"http://localhost:{self.port}/v1/chat/completions"
                        ]
                        
                        for openai_endpoint in openai_endpoints:
                            try:
                                if "chat/completions" in openai_endpoint:
                                    test_response = requests.post(openai_endpoint,
                                                                json={"model": "test"},
                                                                timeout=3)
                                else:
                                    test_response = requests.get(openai_endpoint, timeout=3)
                                logger.info(f"OpenAI API 端点检查: {openai_endpoint} -> {test_response.status_code}")
                            except requests.exceptions.RequestException as e:
                                logger.warning(f"OpenAI API 端点检查失败 {openai_endpoint}: {e}")
                        
                        time.sleep(2)
                        return
                except requests.exceptions.RequestException as e:
                    logger.debug(f"尝试连接 {endpoint} 失败: {e}")
                    pass
            
            time.sleep(1)
            
        try:
            response = requests.get(f"http://localhost:{self.port}/", timeout=5)
            logger.error(f"服务器响应根路径: {response.status_code} - {response.text[:200]}")
        except requests.exceptions.RequestException as e:
            logger.error(f"无法连接到服务器: {e}")
            
        raise TimeoutError(f"服务器在 {timeout} 秒内未能启动")
        
    def get_client(self) -> OpenAI:
        return OpenAI(
            api_key="EMPTY",
            base_url=self.base_url
        )
        
    def get_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="EMPTY",
            base_url=self.base_url
        )


class TestStructuredOutputAPI:
    
    @pytest.fixture(scope="class")
    def server(self, default_ksana_yaml_path):
        yaml_path = default_ksana_yaml_path
            
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"创建临时目录: {temp_dir}")
        
        try:
            temp_yaml_path = os.path.join(temp_dir, "ksana.yaml")
            shutil.copyfile(yaml_path, temp_yaml_path)
            
            modify_yaml_field(temp_yaml_path, "setting.batch_scheduler.enable_xgrammar", True)
            
            with RemoteKsanaServer(None, temp_yaml_path, port=8002) as server:
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
    async def test_json_object_response_format(self, async_client: AsyncOpenAI):
        response = await async_client.chat.completions.create(
            model="ksana-llm",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. Always respond with valid JSON format. "
                        "For example: {\"answer\": \"your response here\"}"
                    )
                },
                {"role": "user", "content": "What is the capital of Bulgaria? Please respond in JSON format."}
            ],
            max_tokens=30,
            response_format={"type": "json_object"},
            extra_body={"enable_structured_output": True} 
        )
        
        print(f"\n=== 完整响应对象 ===")
        print(f"Response: {response}")
        
        text = response.choices[0].message.content

        if not text or text.strip() == "":
            pytest.fail(f"Response is empty or None. Full response object: {response}")
        
        if "No valid outputs generated" in text:
            pytest.fail(f"Model failed to generate valid output: {text}")
        
        try:
            js_obj = json.loads(text)
            print(f"解析的 JSON 对象: {js_obj}")
        except json.JSONDecodeError as e:
            pytest.fail(f"Response is not valid JSON. Error: {e}. Response: {text}")
        
        assert isinstance(js_obj, dict), f"Response is not a JSON object: {text}"
        
    @pytest.mark.asyncio
    async def test_json_schema_response_format(self, async_client: AsyncOpenAI):
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "A simple answer"
                }
            },
            "required": ["answer"],
            "additionalProperties": False
        }
        
        response = await async_client.chat.completions.create(
            model="ksana-llm",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Always respond with the exact JSON format requested."
                },
                {"role": "user", "content": "What is 1+1? Answer in JSON format with an 'answer' field."}
            ],
            max_tokens=100,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "simple_answer",
                    "schema": schema
                }
            },
            extra_body={"enable_structured_output": True}  # 添加结构化输出启用参数
        )
        
        text = response.choices[0].message.content
        print(f"\n=== JSON Schema Response Debug ===")
        print(f"Raw response: {repr(text)}")
        print("=" * 40)
        
        if not text or text.strip() == "":
            pytest.fail(f"Response is empty or None. Full response object: {response}")
        
        if "No valid outputs generated" in text:
            pytest.fail(f"Model failed to generate valid output: {text}")
        
        try:
            js_obj = json.loads(text)
        except json.JSONDecodeError as e:
            pytest.fail(f"Response is not valid JSON. Error: {e}. Response: {text}")
        
        # 验证它是一个 JSON 对象
        assert isinstance(js_obj, dict), f"Response is not a JSON object: {text}"
        
        # 验证必需字段存在
        assert "answer" in js_obj, f"Required field 'answer' missing from response: {text}"
        
        # 验证字段类型
        assert isinstance(js_obj["answer"], str), f"Field 'answer' should be string: {text}"
            
    def test_json_request_without_structured_output(self, client: OpenAI):
        response = client.chat.completions.create(
            model="ksana-llm",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Always respond with valid JSON format."
                },
                {"role": "user", "content": "请用JSON格式回答：北京是哪个国家的首都？格式：{\"country\": \"答案\"}"}
            ],
            temperature=0.1,
            max_tokens=100,
            response_format={"type": "json_object"}
        )
        
        text = response.choices[0].message.content
        print(f"Response: {text}")
        print("=" * 40)
        
        # 检查响应是否为空或无效
        if not text or text.strip() == "":
            pytest.fail(f"Response is empty: {text}")
        
        if "No valid outputs generated" in text:
            pytest.fail(f"Model failed: {text}")
            
    @pytest.mark.asyncio
    async def test_json_schema_simple_structure(self, async_client: AsyncOpenAI):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[\\w]+$"},
                "population": {"type": "integer"}
            },
            "required": ["name", "population"],
            "additionalProperties": False
        }
        
        response = await async_client.chat.completions.create(
            model="ksana-llm",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "Introduce the Paris. Return in a JSON format."
                }
            ],
            max_tokens=30,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "city_info", "schema": schema}
            },
            extra_body={"enable_structured_output": True}
        )
        
        text = response.choices[0].message.content
        print(f"\n=== Simple JSON Schema Response Debug ===")
        print(f"Raw response: {repr(text)}")
        print(f"Response content: {text}")
        print("=" * 40)
        
        if not text or text.strip() == "":
            pytest.fail(f"Response is empty or None. Full response object: {response}")
        
        if "No valid outputs generated" in text:
            pytest.fail(f"Model failed to generate valid output: {text}")
        
        # 验证响应是有效的 JSON
        try:
            js_obj = json.loads(text)
        except (TypeError, json.JSONDecodeError) as e:
            print("JSONDecodeError", text)
            pytest.fail(f"Response is not valid JSON. Error: {e}. Response: {text}")
        
        # 验证schema约束
        assert isinstance(js_obj["name"], str), f"Field 'name' should be string: {text}"
        assert isinstance(js_obj["population"], int), f"Field 'population' should be integer: {text}"
        
        # 额外验证required字段
        assert "name" in js_obj, f"Required field 'name' missing: {text}"
        assert "population" in js_obj, f"Required field 'population' missing: {text}"
        
    def test_sync_json_object_response_format(self, client: OpenAI):
        """测试同步客户端的 json_object 响应格式"""
        response = client.chat.completions.create(
            model="ksana-llm",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {"role": "user", "content": "Tell me about Python programming language."}
            ],
            temperature=0,
            max_tokens=30,
            response_format={"type": "json_object"},
            extra_body={"enable_structured_output": True}  # 添加结构化输出启用参数
        )
        
        text = response.choices[0].message.content
        print(f"Sync JSON Object Response ({len(text)} characters): {text}")
        
        # 验证响应是有效的 JSON
        try:
            js_obj = json.loads(text)
        except json.JSONDecodeError as e:
            pytest.fail(f"Sync response is not valid JSON. Error: {e}. Response: {text}")
        
        # 验证它是一个 JSON 对象
        assert isinstance(js_obj, dict), f"Sync response is not a JSON object: {text}"
