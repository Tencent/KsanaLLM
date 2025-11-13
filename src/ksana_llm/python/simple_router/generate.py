# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import asyncio
import itertools
import logging
import os
import zlib
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, Response

import config
from name_service.name_service import NameServiceRegistry

logger = logging.getLogger(__name__)

try:
    NAME_SERVICE_PROVIDER = NameServiceRegistry.get_provider_from_config(
        config.get_settings().name_service_provider
    )
    if NAME_SERVICE_PROVIDER is None:
        raise RuntimeError(
            f"Failed to load name service provider '{config.get_settings().name_service_provider}'"
        )
except Exception as exc:  # pylint: disable=broad-except
    logger.error("Failed to initialize name service provider: %s", exc)
    raise

raw_router = APIRouter()

# Constants for stream processing
DELIM = b"\0"  # Delimiter for separating chunks in the stream
EOS = b"[DONE]"  # End of stream marker


async def _drain_stream(
    resp: httpx.Response,
    queue: asyncio.Queue,
    decode_done_event=None,
    only_first: bool = False,
):
    """Process a token stream from a node and place tokens in the output queue"""

    async for chunk in resp.aiter_bytes():
        # Check if decode stream completed while we were processing (for prefill stream)
        if only_first and decode_done_event and decode_done_event.is_set():
            return
        # Split by the delimiter (might be multiple tokens in one chunk)
        parts = chunk.split(DELIM)
        for i, part in enumerate(parts):
            if not part:  # Skip empty parts
                continue
            # Put token in queue first
            await queue.put(part + (DELIM if i < len(parts) - 1 else b""))

            # Decode stream: set event after first token is queued
            if not only_first and decode_done_event and not decode_done_event.is_set():
                decode_done_event.set()

            # If prefill stream, return after first token
            if only_first:
                return


async def forward_request(req: Request, endpoint_path: str):
    """Route a client request to prefill/decode nodes and merge their streams."""

    # Get upstream nodes from name service provider
    try:
        prefill, decode, prefill_instance, decode_instance = (
            NAME_SERVICE_PROVIDER.get_available_nodes()
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to select nodes: %s", exc)
        raise HTTPException(
            status_code=503, detail="No available nodes for processing"
        ) from exc
    prefill_name, prefill_address = prefill
    decode_name, decode_address = decode
    prefill_url = f"http://{prefill_address}{endpoint_path}"
    decode_url = f"http://{decode_address}{endpoint_path}"

    # Update communication ID and set request headers
    if not hasattr(forward_request, "_comm_counter"):
        forward_request._comm_counter = itertools.count(110)  # type: ignore[attr-defined]
    local_seq = next(forward_request._comm_counter)  # type: ignore[attr-defined]
    comm_id = zlib.crc32(f"{os.getpid()}:{local_seq}".encode("ascii")) & 0x7FFFFFFF
    if comm_id == 0:
        comm_id = 1
    headers = {
        "kv-comm-group-key": f"{prefill_name}__{decode_name}",
        "kv-comm-request-id": str(comm_id),
    }
    logger.info(
        "Routing request to prefill=%s decode=%s comm_id=%s",
        prefill_url,
        decode_url,
        comm_id,
    )

    # Create http request to upstream nodes
    client = httpx.AsyncClient(timeout=None)
    request_kwargs: Dict[str, Any] = {"headers": headers, "json": await req.json()}
    try:
        prefill_ctx = client.stream(req.method, prefill_url, **request_kwargs)
        decode_ctx = client.stream(req.method, decode_url, **request_kwargs)
    except httpx.RequestError as exc:
        logger.error("HTTP stream setup failed: %s", exc)
        await client.aclose()
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to nodes: {exc}"
        ) from exc

    # Get both responses and ensure connections are established
    try:
        prefill_resp, decode_resp = await asyncio.gather(
            prefill_ctx.__aenter__(), decode_ctx.__aenter__()
        )
    except Exception as exc:
        await client.aclose()
        raise

    # update node call results
    NAME_SERVICE_PROVIDER.update_nodes_call_result(
        prefill_instance=prefill_instance,
        decode_instance=decode_instance,
        prefill_success=prefill_resp.status_code < 400,
        decode_success=decode_resp.status_code < 400,
    )

    # Check upstream response types
    is_streaming = prefill_resp.headers.get("transfer-encoding") == "chunked"
    if not is_streaming:
        # Drop prefill response while not streaming response and return decode response
        asyncio.create_task(prefill_ctx.__aexit__(None, None, None))
        content = await decode_resp.aread()
        await decode_ctx.__aexit__(None, None, None)
        await client.aclose()
        return Response(
            content=content,
            status_code=decode_resp.status_code,
            headers=dict(decode_resp.headers),
        )

    # Streaming response: merge both streams
    async def stream_generator():
        try:
            decode_done_event = asyncio.Event()

            # If request api is OpenAI then only drain for decode stream
            if req.scope["path"] != "/generate":
                decode_done_event.set()
            queue: asyncio.Queue[bytes] = asyncio.Queue()
            prefill_task = asyncio.create_task(
                _drain_stream(
                    prefill_resp,
                    queue,
                    only_first=True,
                    decode_done_event=decode_done_event,
                )
            )
            decode_task = asyncio.create_task(
                _drain_stream(
                    decode_resp,
                    queue,
                    only_first=False,
                    decode_done_event=decode_done_event,
                )
            )

            async def _close_when_done(ctx, task):
                try:
                    await task
                finally:
                    await ctx.__aexit__(None, None, None)

            prefill_closer = asyncio.create_task(
                _close_when_done(prefill_ctx, prefill_task)
            )
            decode_closer = asyncio.create_task(
                _close_when_done(decode_ctx, decode_task)
            )

            still_running = 2
            while still_running:
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield token
                except asyncio.TimeoutError:
                    still_running = sum(
                        1 for t in (prefill_closer, decode_closer) if not t.done()
                    )

            logger.info(f"Communication {comm_id} completed, sending EOS")
            yield EOS + DELIM
        finally:
            await client.aclose()

    return StreamingResponse(stream_generator(), media_type="application/octet-stream")
