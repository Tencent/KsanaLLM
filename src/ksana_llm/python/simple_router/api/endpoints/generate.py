# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import asyncio
import json
import logging
import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from .name_service.name_service import NameServiceRegistry

# Configure logging
logger = logging.getLogger(__name__)

# 在模块初始化时加载 name service 提供者
try:
    NAME_SERVICE_PROVIDER = NameServiceRegistry.get_provider_from_config(
        settings.name_service_provider
    )
    if not NAME_SERVICE_PROVIDER:
        raise RuntimeError(
            f"Failed to load name service provider from '{settings.name_service_provider}'"
        )
    logger.info(f"Loaded name service provider: {NAME_SERVICE_PROVIDER.name}")
except Exception as e:
    logger.error(f"Failed to initialize name service provider: {e}")
    raise

raw_router = APIRouter()


# Constants for stream processing
DELIM = b"\0"  # Delimiter for separating chunks in the stream
EOS = b"[DONE]"  # End of stream marker


async def _drain_stream(resp: httpx.Response, out_q: asyncio.Queue, only_first: bool = False,
                        first_token_ready=None, cons_done_event=None):
    """Process a token stream from a node and place tokens in the output queue"""
    if first_token_ready and not only_first:
        await first_token_ready.wait()

    # Check if cons stream is already done before processing (for prod stream)
    if only_first and cons_done_event and cons_done_event.is_set():
        return

    async for chunk in resp.aiter_bytes():
        # Check if cons stream completed while we were processing (for prod stream)
        if only_first and cons_done_event and cons_done_event.is_set():
            return

        # Split by the delimiter (might be multiple tokens in one chunk)
        parts = chunk.split(DELIM)
        for i, part in enumerate(parts):
            if not part:  # Skip empty parts
                continue

            # Check again before putting data in queue (for prod stream)
            if only_first and cons_done_event and cons_done_event.is_set():
                return

            # Put each part in the output queue
            await out_q.put(part + (DELIM if i < len(parts) - 1 else b""))

            # If we only want the first token and have seen a delimiter, stop
            if only_first:
                if first_token_ready:
                    first_token_ready.set()
                return

    # Mark cons stream as done (for cons stream)
    if not only_first and cons_done_event:
        cons_done_event.set()


async def forward_request(req: Request, endpoint_path: str):
    """
    Generic function to forward requests to both prefill and decode nodes
    and merge the responses into a single stream

    Args:
        req: The incoming request
        endpoint_path: The endpoint path to forward to (e.g., "/generate", "/v1/chat/completions")
    """
    # Get request method and body
    method = req.method
    body = await req.body()

    # Parse body for POST requests
    body_json = {}
    if method in ["POST", "PUT", "PATCH"] and body:
        try:
            body_json = json.loads(body)
        except json.JSONDecodeError:
            body_json = {}

    # Get query parameters for GET requests
    query_params = dict(req.query_params)

    try:
        # 获取选中的节点
        prefill_node, decode_node, prefill_instance, decode_instance = (
            NAME_SERVICE_PROVIDER.get_available_nodes()
        )

        # 构建 URL
        prefill_name, prefill_address = prefill_node
        prefill_url = f"http://{prefill_address}{endpoint_path}"

        decode_name, decode_address = decode_node
        decode_url = f"http://{decode_address}{endpoint_path}"

    except Exception as e:
        logger.error(
            f"Failed to select nodes using provider '{NAME_SERVICE_PROVIDER.name}': {e}"
        )
        raise HTTPException(
            status_code=503, detail=f"No available nodes for processing"
        )

    # 全局自增 communication id
    if not hasattr(forward_request, "_comm_id"):
        forward_request._comm_id = 110
    comm_id = forward_request._comm_id
    headers = {
        "kv-comm-group-key": f"{prefill_name}__{decode_name}",
        "kv-comm-request-id": str(comm_id),
    }
    logger.info(
        f"Routing to prefill: {prefill_url}, decode: {decode_url} and comm_id: {comm_id}"
    )
    forward_request._comm_id += 1

    async def merged_stream():
        q: asyncio.Queue[bytes] = asyncio.Queue()

        async with httpx.AsyncClient(timeout=None) as cli:
            # Prepare request parameters based on method
            request_kwargs = {
                "headers": headers,
            }

            if method in ["POST", "PUT", "PATCH"]:
                request_kwargs["json"] = body_json
            elif method == "GET" and query_params:
                request_kwargs["params"] = query_params

            # ------- 并发建立两个流 -------
            try:
                prod_ctx = cli.stream(method, prefill_url, **request_kwargs)
                cons_ctx = cli.stream(method, decode_url, **request_kwargs)
            except httpx.RequestError as e:
                logger.error(f"HTTP stream request failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to connect to nodes: {e}"
                )
            prod_resp, cons_resp = await asyncio.gather(
                prod_ctx.__aenter__(),
                cons_ctx.__aenter__(),
            )
            NAME_SERVICE_PROVIDER.update_nodes_call_result(
                prefill_instance=prefill_instance,
                decode_instance=decode_instance,
                prefill_success=prod_resp.status_code < 400,
                decode_success=cons_resp.status_code < 400,
            )

            # ------- 并发读取 -------
            first_token_ready = asyncio.Event()
            cons_done_event = asyncio.Event()  # 标记 cons stream 是否已完成

            prod_task = asyncio.create_task(
                _drain_stream(prod_resp, q, only_first=True, first_token_ready=first_token_ready,
                              cons_done_event=cons_done_event)
            )
            cons_task = asyncio.create_task(
                _drain_stream(cons_resp, q, only_first=False, first_token_ready=first_token_ready,
                              cons_done_event=cons_done_event)
            )

            # ◎ 这一步骤可以让 Producer 在发完首 token 后就被关闭
            async def _close_when_done(ctx, task):
                try:
                    await task
                finally:
                    await ctx.__aexit__(None, None, None)

            closer1 = asyncio.create_task(_close_when_done(prod_ctx, prod_task))
            closer2 = asyncio.create_task(_close_when_done(cons_ctx, cons_task))

            # ------- 把队列里的数据写给客户端 -------
            still_running = 2  # 两条后台流
            while still_running:
                try:
                    token = await asyncio.wait_for(q.get(), timeout=0.1)
                    yield token
                except asyncio.TimeoutError:
                    # 查看一下还有没有任务活着
                    still_running = sum(t and not t.done() for t in (closer1, closer2))

            # 后端两条流都结束了，发送真正的结束标识
            logger.info(f"Communication {comm_id} completed, sending EOS")
            yield EOS + DELIM

    return StreamingResponse(merged_stream(), media_type="application/octet-stream")


# Universal proxy routes using path wildcards
@raw_router.api_route(
    "/v1/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"]
)
async def proxy_v1_endpoints(req: Request, path: str):
    """Handle all /v1/* endpoints with any HTTP method"""
    return await forward_request(req, f"/v1/{path}")


@raw_router.post("/generate")
async def generate(req: Request):
    """Handle /generate endpoint (keep for backward compatibility)"""
    return await forward_request(req, "/generate")


# Optional: Add more specific wildcards if needed
@raw_router.api_route(
    "/v2/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"]
)
async def proxy_v2_endpoints(req: Request, path: str):
    """Handle all /v2/* endpoints with any HTTP method"""
    return await forward_request(req, f"/v2/{path}")


@raw_router.api_route(
    "/api/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"]
)
async def proxy_api_endpoints(req: Request, path: str):
    """Handle all /api/* endpoints with any HTTP method"""
    return await forward_request(req, f"/api/{path}")
