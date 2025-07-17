# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import sys
import asyncio
import time
from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture
def mock_libtorch():
    mock = MagicMock()
    sys.modules["libtorch_serving"] = mock
    return mock


async def mock_generate(*args, **kwargs):
    await asyncio.sleep(0.5)  
    status_mock = MagicMock()
    status_mock.OK.return_value = True
    # Return a mock response with status_code attribute
    response_mock = MagicMock()
    response_mock.status_code = 200
    return response_mock


@pytest.mark.asyncio
async def test_concurrent_performance(mock_libtorch):
    # Use patch.dict inside the function to avoid decorator interference
    with patch.dict('sys.modules', {
        'ksana_llm': MagicMock(),
        'ksana_llm.arg_utils': MagicMock()
    }):
        from serving_server import LLMServer
        from ksana_llm.arg_utils import EngineArgs

        # Create a mock EngineArgs
        mock_engine_args = MagicMock(spec=EngineArgs)
        mock_engine_args.config_file = "test_config.yaml"
        mock_engine_args.model_dir = "/test/model"
        mock_engine_args.tokenizer_path = "/test/tokenizer"
        mock_engine_args.model_type = "test"
        mock_engine_args.endpoint = "python"
        mock_engine_args.host = "localhost"
        mock_engine_args.port = 8080
        mock_engine_args.access_log = True
        mock_engine_args.plugin_model_enable_trt = True
        mock_engine_args.plugin_thread_pool_size = 1
        mock_engine_args.ssl_keyfile = None
        mock_engine_args.ssl_certfile = None
        mock_engine_args.root_path = None
        mock_engine_args.tokenization = None

        # Initialize server with mock engine_args
        server = LLMServer(mock_engine_args)
        
        # Mock the generate method to return our mock response directly
        server.generate = mock_generate

        num_requests = 10
        start_time = time.time()

        async def make_request():
            request = MagicMock()
            request.body = MagicMock()
            request.body.return_value = b'{"prompt": "test"}'
            return await server.generate(request)

        tasks = [make_request() for _ in range(num_requests)]
        responses = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        assert len(responses) == num_requests
        for response in responses:
            assert response.status_code == 200

        assert 0.4 < total_time < 0.8, (
            f"Unexpected execution time: {total_time}s "
            f"(expected ~0.5s for concurrent execution)"
        )
