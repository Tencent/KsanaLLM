# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import asyncio
import json
import random
import logging
import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from db import db
from .database_nodes import get_available_nodes_from_db

if settings.router_rule == "polaris":
    from .polaris_nodes import get_available_nodes_with_polaris, update_polaris_service_call_result

# Configure logging
logger = logging.getLogger(__name__)

# Configuration for prefill and decode nodes (used as fallback for fixed mode)
PREFILL_NODES = [("prefill_group_1", "localhost:8088")]
DECODE_NODES = [("decode_group_1", "localhost:8089")]

raw_router = APIRouter()


# Helper function to randomly select a node from a list
def pick(l):
    if not l:
        raise HTTPException(status_code=503, detail="No available nodes for processing")
    return random.choice(l)


# Constants for stream processing
DELIM = b"\0"  # Delimiter for separating chunks in the stream
EOS = b"[DONE]"  # End of stream marker


async def _drain_stream(
    resp: httpx.Response, out_q: asyncio.Queue, only_first: bool = False
):
    """Process a token stream from a node and place tokens in the output queue"""
    async for chunk in resp.aiter_bytes():
        # Split by the delimiter (might be multiple tokens in one chunk)
        parts = chunk.split(DELIM)
        for i, part in enumerate(parts):
            if not part:  # Skip empty parts
                continue

            # Put each part in the output queue
            await out_q.put(part + (DELIM if i < len(parts) - 1 else b""))

            # If we only want the first token and have seen a delimiter, stop
            if only_first and i > 0:
                return


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

    # Select prefill and decode nodes based on router_rule
    prefill_url = ""
    decode_url = ""

    def select_nodes_with_fallback(available_nodes, prefill_nodes, decode_nodes):
        if available_nodes["prefill"]:
            prefill = pick(available_nodes["prefill"])
            prefill_name = prefill[0]
            prefill_url = f"http://{prefill[1]}{endpoint_path}"
        else:
            logger.warning("No available prefill nodes found, using fallback")
            prefill = pick(prefill_nodes)
            prefill_name = prefill[0]
            prefill_url = f"http://{prefill[1]}{endpoint_path}"

        if available_nodes["decode"]:
            decode = pick(available_nodes["decode"])
            decode_name = decode[0]
            decode_url = f"http://{decode[1]}{endpoint_path}"
        else:
            logger.warning("No available decode nodes found, using fallback")
            decode = pick(decode_nodes)
            decode_name = decode[0]
            decode_url = f"http://{decode[1]}/generate"
        return prefill, prefill_name, prefill_url, decode, decode_name, decode_url

    prefill_instance = None
    decode_instance = None

    if settings.router_rule == "auto":
        # 先尝试数据库，再尝试 polaris
        available_nodes = get_available_nodes_from_db(db.storage, settings.cluster_name)
        prefill, prefill_name, prefill_url, decode, decode_name, decode_url = select_nodes_with_fallback(
            available_nodes, PREFILL_NODES, DECODE_NODES
        )
    elif settings.router_rule == "polaris":
        available_nodes, prefill_instance, decode_instance = get_available_nodes_with_polaris(
            settings.polaris_namespace, settings.polaris_prefill_service, settings.polaris_decode_service
        )
        prefill, prefill_name, prefill_url, decode, decode_name, decode_url = select_nodes_with_fallback(
            available_nodes, PREFILL_NODES, DECODE_NODES
        )
    else:
        # Use fixed configuration
        prefill = pick(PREFILL_NODES)
        prefill_name = prefill[0]
        prefill_url = f"http://{prefill[1]}{endpoint_path}"
        decode = pick(DECODE_NODES)
        decode_name = decode[0]
        decode_url = f"http://{decode[1]}{endpoint_path}"

    # 全局自增 communication id
    if not hasattr(forward_request, "_comm_id"):
        forward_request._comm_id = 110
    comm_id = forward_request._comm_id
    headers = {
        "kv-comm-group-key": f"{prefill_name}__{decode_name}",
        "kv-comm-request-id": str(comm_id),
    }
    logger.info(f"Routing to prefill: {prefill_url}, decode: {decode_url} and comm_id: {comm_id}")
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
                    status_code=500,
                    detail=f"Failed to connect to nodes: {e}"
                )
            prod_resp, cons_resp = await asyncio.gather(
                prod_ctx.__aenter__(),
                cons_ctx.__aenter__(),
            )
            if settings.router_rule == "polaris":
                # 更新 Polaris 服务调用结果
                update_polaris_service_call_result(
                    settings.polaris_namespace,
                    settings.polaris_prefill_service,
                    settings.polaris_decode_service,
                    prefill_instance,
                    decode_instance,
                    prod_resp.status_code < 400,
                    cons_resp.status_code < 400
                )

            # ------- 并发读取 -------
            prod_task = asyncio.create_task(
                _drain_stream(prod_resp, q, only_first=True)
            )
            cons_task = asyncio.create_task(
                _drain_stream(cons_resp, q, only_first=False)
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
@raw_router.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"])
async def proxy_v1_endpoints(req: Request, path: str):
    """Handle all /v1/* endpoints with any HTTP method"""
    return await forward_request(req, f"/v1/{path}")


@raw_router.post("/generate")
async def generate(req: Request):
    """Handle /generate endpoint (keep for backward compatibility)"""
    return await forward_request(req, "/generate")


# Optional: Add more specific wildcards if needed
@raw_router.api_route("/v2/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"])
async def proxy_v2_endpoints(req: Request, path: str):
    """Handle all /v2/* endpoints with any HTTP method"""
    return await forward_request(req, f"/v2/{path}")


@raw_router.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"])
async def proxy_api_endpoints(req: Request, path: str):
    """Handle all /api/* endpoints with any HTTP method"""
    return await forward_request(req, f"/api/{path}")
