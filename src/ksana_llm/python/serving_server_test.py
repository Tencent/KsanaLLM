# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import sys
import asyncio
import time
from unittest.mock import MagicMock
import pytest


@pytest.fixture
def mock_libtorch():
    mock = MagicMock()
    sys.modules["libtorch_serving"] = mock
    return mock


async def mock_generate(*args, **kwargs):
    await asyncio.sleep(0.5)  # 模拟 500ms 延迟
    return MagicMock(OK=lambda: True), {"text": "test response"}


@pytest.mark.asyncio
async def test_concurrent_performance(mock_libtorch):
    from serving_server import LLMServer

    server = LLMServer()
    server.model = MagicMock()
    server.model.generate = mock_generate

    # 创建多个并发请求
    num_requests = 10
    start_time = time.time()

    async def make_request():
        request = MagicMock()
        request.body.return_value = b'{"prompt": "test"}'
        return await server.generate(request)

    # 并发执行请求
    tasks = [make_request() for _ in range(num_requests)]
    responses = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    # 验证响应
    assert len(responses) == num_requests
    for response in responses:
        assert response.status_code == 200

    # 验证总执行时间是否合理(应该接近 500ms)
    assert 0.4 < total_time < 0.6, f"Unexpected execution time: {total_time}"
