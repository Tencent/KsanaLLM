# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from typing import List, Dict
import asyncio
import json
import random
import logging
import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse


from config import settings
from db import db

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


def get_available_nodes() -> Dict[str, List[str]]:
    """Get available prefill and decode nodes with node_rank=0 from the cluster"""
    nodes = {"prefill": [], "decode": []}

    try:
        cluster = db.storage.get_cluster(settings.cluster_name)

        for group_name, group in cluster.prefill_groups.items():
            if not group.is_ready:
                continue

            for _, node in group.nodes.items():
                if node.is_online and node.node_rank == 0:
                    nodes["prefill"].append((group_name, f"{node.inference_addr}"))

        # Find decode nodes with node_rank=0
        for group_name, group in cluster.decode_groups.items():
            if not group.is_ready:
                continue

            for _, node in group.nodes.items():
                if node.is_online and node.node_rank == 0:
                    nodes["decode"].append((group_name, f"{node.inference_addr}"))

    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Error getting available nodes: {str(e)}")

    return nodes


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


@raw_router.post("/generate")
async def generate(req: Request):
    """
    Generate text by routing the request to both prefill and decode nodes
    and merging the responses into a single stream

    The prefill handles the initial token generation (prefill phase)
    while the decode handles the subsequent tokens (decode phase)
    """
    body = await req.body()
    try:
        body_json = json.loads(body)
    except json.JSONDecodeError:
        body_json = {}

    # Select prefill and decode nodes based on router_rule
    prefill_url = ""
    decode_url = ""

    if settings.router_rule == "auto":
        # Get available nodes
        available_nodes = get_available_nodes()

        # Select random prefill and decode nodes
        if available_nodes["prefill"]:
            prefill = pick(available_nodes["prefill"])
            prefill_name = prefill[0]
            prefill_url = f"http://{prefill[1]}/generate"
        else:
            logger.warning("No available prefill nodes found, using fallback")
            prefill = pick(PREFILL_NODES)
            prefill_name = prefill[0]
            prefill_url = f"http://{prefill[1]}/generate"

        if available_nodes["decode"]:
            decode = pick(available_nodes["decode"])
            decode_name = decode[0]
            decode_url = f"http://{decode[1]}/generate"
        else:
            logger.warning("No available decode nodes found, using fallback")
            decode = pick(DECODE_NODES)
            decode_name = decode[0]
            decode_url = f"http://{decode[1]}/generate"
    else:
        # Use fixed configuration
        prefill = pick(PREFILL_NODES)
        prefill_name = prefill[0]
        prefill_url = f"http://{prefill[1]}/generate"
        decode = pick(DECODE_NODES)
        decode_name = decode[0]
        decode_url = f"http://{decode[1]}/generate"


    # 全局自增 communication id
    if not hasattr(generate, "_comm_id"):
        generate._comm_id = 110
    comm_id = generate._comm_id
    headers = {
        "kv-comm-group-key": f"{prefill_name}__{decode_name}",
        "kv-comm-request-id": str(comm_id),
    }
    logger.info(f"Routing to prefill: {prefill_url}, decode: {decode_url} and comm_id: {comm_id}")
    generate._comm_id += 1

    async def merged_stream():
        q: asyncio.Queue[bytes] = asyncio.Queue()

        async with httpx.AsyncClient(timeout=None) as cli:
            # ------- 并发建立两个流 -------
            prod_ctx = cli.stream("POST", prefill_url, json=body_json, headers=headers)
            cons_ctx = cli.stream("POST", decode_url, json=body_json, headers=headers)

            prod_resp, cons_resp = await asyncio.gather(
                prod_ctx.__aenter__(),
                cons_ctx.__aenter__(),
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
