# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
from uuid import uuid4

from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import SETTINGS
from models import CommGroupPair, InferenceGroupStatus, NodeInfo

logger = logging.getLogger(__name__)


class NodeNotFoundError(Exception):
    """Raised when a node cannot be located in the database."""


class InvalidCommGroupError(Exception):
    """Raised when the communication group update request is invalid."""


def register_node(
    session: Session,
    node_id: str | None,
    cluster_name: str,
    inference_addr: str,
    coordinator_addr: str,
    role: str,
    node_rank: int,
    world_size: int,
    device_num: int,
) -> NodeInfo:
    """Persist a newly registered node and update group status."""

    lookup_stmt = select(NodeInfo).where(
        NodeInfo.coordinator_addr == coordinator_addr,
    )

    existing = session.scalar(lookup_stmt)

    if existing is not None:
        _reset_if_prefill_rank0(session, cluster_name, [existing])
        existing.cluster_name = cluster_name
        existing.inference_addr = inference_addr
        existing.role = role
        existing.node_rank = node_rank
        existing.world_size = world_size
        existing.device_num = device_num
        existing.last_heartbeat = datetime.now(timezone.utc)
        existing.is_online = True
        existing.start_time = datetime.now(timezone.utc)
        session.flush()
        logger.info(
            "Node already registered, updated existing node: %s and coordinator address: %s",
            existing.node_id,
            coordinator_addr,
        )
        return existing

    new_node_id = node_id or uuid4().hex
    node = NodeInfo(
        node_id=new_node_id,
        cluster_name=cluster_name,
        inference_addr=inference_addr,
        coordinator_addr=coordinator_addr,
        role=role,
        node_rank=node_rank,
        world_size=world_size,
        device_num=device_num,
        last_heartbeat=datetime.now(timezone.utc),
        is_online=True,
        start_time=datetime.now(timezone.utc),
    )
    logger.info(
        "Registering new node: %s and coordinator address: %s",
        new_node_id,
        coordinator_addr,
    )
    session.add(node)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        existing = session.scalar(lookup_stmt)
        if existing is None:
            raise exc

        _reset_if_prefill_rank0(session, cluster_name, [existing])
        existing.coordinator_addr = coordinator_addr
        existing.world_size = world_size
        existing.device_num = device_num
        existing.last_heartbeat = datetime.now(timezone.utc)
        existing.is_online = True
        existing.start_time = datetime.now(timezone.utc)
        session.flush()
        node = existing
    return node


def _get_stale_nodes(session: Session, cluster_name: str) -> list[NodeInfo]:
    """Return nodes that have missed their heartbeat deadline."""
    timeout_seconds = getattr(SETTINGS, "heartbeat_timeout_seconds", 60)
    timeout_threshold = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
    stale_nodes = session.scalars(
        select(NodeInfo).where(
            NodeInfo.cluster_name == cluster_name,
            NodeInfo.last_heartbeat < timeout_threshold,
        )
    ).all()
    for stale_node in stale_nodes:
        logger.warning(
            "Node %s is stale (last heartbeat: %s)",
            stale_node.node_id,
            stale_node.last_heartbeat,
        )
        stale_node.is_online = False
    session.flush()
    return stale_nodes


def _update_inference_group_status(session: Session, nodes: list[NodeInfo]) -> None:
    """Mark inference group rows as not ready when backing nodes are stale."""

    if not nodes:
        return
    infer_status = session.scalars(
        select(InferenceGroupStatus).where(
            InferenceGroupStatus.cluster_name == nodes[0].cluster_name,
            InferenceGroupStatus.inference_addr.in_(
                [node.inference_addr for node in nodes]
            ),
        )
    ).all()

    for status in infer_status:
        logger.info(
            "Marking inference group %s as not ready due to stale nodes",
            status.inference_addr,
        )
        status.is_ready = False
    session.flush()


def _reset_if_prefill_rank0(
    session: Session, cluster_name: str, nodes: list[NodeInfo]
) -> None:
    """Reset communication group metadata for stale nodes."""
    for node in nodes:
        pairs = session.scalars(
            select(CommGroupPair).where(
                CommGroupPair.cluster_name == cluster_name,
                or_(CommGroupPair.prefill_addr == node.inference_addr),
            )
        ).all()

        for pair in pairs:
            logger.info(
                "Clearing stale metadata for pair %s due to stale node %s",
                pair.comm_key,
                node.node_id,
            )
            if node.role == "prefill" and node.node_rank == 0:
                pair.data_channel_meta = ""
                pair.control_channel_meta = []
                pair.is_active = False

    session.flush()


def _rebuild_comm_group_control_meta(
    session: Session, cluster_name: str, comm_key: str
) -> List[Tuple[int, int, str]]:
    """Rebuild control channel metadata for a given comm_key."""
    try:
        prefill_addr, decode_addr = comm_key.split("__", 1)
    except ValueError:
        logger.error("Invalid comm_key format: %s", comm_key)
        return []

    node_rows = session.scalars(
        select(NodeInfo).where(
            NodeInfo.cluster_name == cluster_name,
            NodeInfo.inference_addr.in_([prefill_addr, decode_addr]),
            NodeInfo.role.in_(["prefill", "decode"]),
        )
    ).all()

    prefill_meta: List[Tuple[int, int, str]] = []
    decode_meta: List[Tuple[int, int, str]] = []

    for row in node_rows:
        device_count = max(row.device_num or 1, 1)
        target = prefill_meta if row.role == "prefill" else decode_meta
        target.extend(
            (row.node_rank, device_id, row.coordinator_addr)
            for device_id in range(device_count)
        )

    prefill_meta.sort(key=lambda item: (item[0], item[1]))
    decode_meta.sort(key=lambda item: (item[0], item[1]))

    return prefill_meta + decode_meta


def update_inference_group_status(session: Session, node_id: str) -> NodeInfo:
    """Refresh the aggregate inference group status."""
    now = datetime.now(timezone.utc)
    node = session.scalar(select(NodeInfo).where(NodeInfo.node_id == node_id))
    if not node:
        raise ValueError(f"Node with id {node_id} not found")

    node.last_heartbeat = now
    node.is_online = True
    session.flush()

    stale_nodes = _get_stale_nodes(session, node.cluster_name)
    _update_inference_group_status(session, stale_nodes)

    online_device_sum, max_world_size, online_node_count = session.execute(
        select(
            func.coalesce(func.sum(NodeInfo.device_num), 0),
            func.coalesce(func.max(NodeInfo.world_size), 0),
            func.count(NodeInfo.id),
        ).where(
            NodeInfo.cluster_name == node.cluster_name,
            NodeInfo.inference_addr == node.inference_addr,
            NodeInfo.role == node.role,
            NodeInfo.is_online.is_(True),
        )
    ).one()

    status = session.scalar(
        select(InferenceGroupStatus).where(
            InferenceGroupStatus.cluster_name == node.cluster_name,
            InferenceGroupStatus.inference_addr == node.inference_addr,
            InferenceGroupStatus.role == node.role,
        )
    )
    if status is None:
        status = InferenceGroupStatus(
            cluster_name=node.cluster_name,
            inference_addr=node.inference_addr,
            role=node.role,
        )
        session.add(status)

    status.world_size_expected = max_world_size
    status.online_device_sum = online_device_sum
    status.online_node_count = online_node_count
    status.is_ready = max_world_size > 0 and online_device_sum == max_world_size
    status.last_computed_at = now

    session.flush()
    return node


def collect_comm_keys(session: Session, node: NodeInfo) -> set[str]:
    """Gather comm_keys that relate the given node to opposite-role peers."""

    opposite_role = "decode" if node.role == "prefill" else "prefill"
    inference_groups = session.scalars(
        select(InferenceGroupStatus).where(
            InferenceGroupStatus.cluster_name == node.cluster_name,
            InferenceGroupStatus.role == opposite_role,
        )
    ).all()
    comm_group_keys = set()
    if inference_groups is None:
        return comm_group_keys
    for opposite_role_group in inference_groups:
        if node.role == "prefill":
            comm_key = node.inference_addr + "__" + opposite_role_group.inference_addr
        else:
            comm_key = opposite_role_group.inference_addr + "__" + node.inference_addr

        comm_group_keys.add(comm_key)

    return comm_group_keys


def _create_empty_comm_pair(session: Session, cluster_name: str, comm_key: str) -> None:
    """Create a new CommGroupPair with empty metadata."""
    try:
        prefill_addr, decode_addr = comm_key.split("__", 1)
    except ValueError:
        logger.error("Invalid comm_key format: %s", comm_key)
        return

    pair = CommGroupPair(
        comm_key=comm_key,
        is_active=False,
        cluster_name=cluster_name,
        prefill_addr=prefill_addr,
        decode_addr=decode_addr,
        control_channel_meta=[],
        data_channel_meta="",
    )
    session.add(pair)
    session.flush()


def get_comm_metadata(
    session: Session, node: NodeInfo, comm_keys: set[str]
) -> Tuple[Dict[str, List[Tuple[int, int, str]]], Dict[str, str]]:
    """Return control/data channel meta keyed by comm_key for the given node."""
    pairs = session.scalars(
        select(CommGroupPair).where(CommGroupPair.cluster_name == node.cluster_name)
    ).all()

    pairs_dict = {pair.comm_key: pair for pair in pairs}
    control_meta: Dict[str, List[Tuple[int, int, str]]] = {}
    data_meta: Dict[str, str] = {}

    addresses_to_expand: set[str] = set()
    legacy_control_meta: Dict[str, List[List[str]]] = {}

    for comm_key in comm_keys:
        pair = pairs_dict.get(comm_key)
        if pair is None:
            control_meta[comm_key] = []
            data_meta[comm_key] = ""
            _create_empty_comm_pair(session, node.cluster_name, comm_key)
            continue

        stored_meta = pair.control_channel_meta or []
        data_meta[comm_key] = pair.data_channel_meta or ""

        if stored_meta and all(
            isinstance(item, (list, tuple)) and len(item) == 3 for item in stored_meta
        ):
            tuples: List[Tuple[int, int, str]] = []
            for entry in stored_meta:
                node_rank, device_id, addr = entry
                tuples.append((int(node_rank), int(device_id), str(addr)))
            control_meta[comm_key] = tuples
        else:
            legacy_control_meta[comm_key] = (
                stored_meta if isinstance(stored_meta, list) else []
            )
            for addr_list in legacy_control_meta[comm_key]:
                for addr in addr_list or []:
                    addresses_to_expand.add(addr)

    if legacy_control_meta:
        addr_to_node = {}
        if addresses_to_expand:
            nodes = session.scalars(
                select(NodeInfo).where(
                    NodeInfo.coordinator_addr.in_(addresses_to_expand)
                )
            ).all()
            addr_to_node = {item.coordinator_addr: item for item in nodes}

        for comm_key, addr_groups in legacy_control_meta.items():
            tuples: List[Tuple[int, int, str]] = []
            for addr_list in addr_groups:
                for addr in addr_list or []:
                    node_info = addr_to_node.get(addr)
                    if node_info is None:
                        tuples.append((0, 0, addr))
                        continue
                    device_count = max(node_info.device_num or 1, 1)
                    tuples.extend(
                        (node_info.node_rank, device_id, addr)
                        for device_id in range(device_count)
                    )

            def _legacy_sort_key(entry: Tuple[int, int, str]) -> Tuple[int, int, int]:
                addr = entry[2]
                node_info = addr_to_node.get(addr)
                role_weight = 0 if node_info and node_info.role == "prefill" else 1
                return role_weight, entry[0], entry[1]

            control_meta[comm_key] = sorted(tuples, key=_legacy_sort_key)

    for comm_key in comm_keys:
        control_meta.setdefault(comm_key, [])
        data_meta.setdefault(comm_key, "")

    all_addresses = {entry[2] for entries in control_meta.values() for entry in entries}
    if all_addresses:
        nodes = session.scalars(
            select(NodeInfo).where(
                NodeInfo.cluster_name == node.cluster_name,
                NodeInfo.coordinator_addr.in_(all_addresses),
            )
        ).all()
        addr_to_role = {item.coordinator_addr: item.role for item in nodes}

        def _sort_key(entry: Tuple[int, int, str]) -> Tuple[int, int, int, str]:
            role_weight = 0 if addr_to_role.get(entry[2]) == "prefill" else 1
            return role_weight, entry[0], entry[1], entry[2]

        for key, entries in control_meta.items():
            control_meta[key] = sorted(entries, key=_sort_key)

    return control_meta, data_meta


def check_and_filter_metadata(
    session: Session,
    node: NodeInfo,
    all_control_meta: Dict[str, List[Tuple[int, int, str]]],
    all_data_meta: Dict[str, str],
) -> Tuple[Dict[str, List[Tuple[int, int, str]]], Dict[str, str]]:
    """Validate metadata and disable comm groups whose peers are not ready."""
    prefill_addrs = []
    decode_addrs = []
    if all_data_meta is None:
        all_data_meta = {}
    # 解析 key，拆分 prefill_addr 和 decode_addr
    for key in all_data_meta.keys():
        if "__" not in key:
            logger.warning(f"Invalid comm_key format in all_data_meta: {key}")
            continue
        prefill_addr, decode_addr = key.split("__", 1)
        prefill_addrs.append(prefill_addr)
        decode_addrs.append(decode_addr)

    all_addrs = list(set(prefill_addrs + decode_addrs))
    if not all_addrs:
        return all_control_meta, all_data_meta

    result = session.execute(
        select(
            InferenceGroupStatus.inference_addr,
            InferenceGroupStatus.role,
            InferenceGroupStatus.is_ready,
        ).where(
            InferenceGroupStatus.cluster_name == node.cluster_name,
            InferenceGroupStatus.inference_addr.in_(all_addrs),
        )
    ).all()

    keys_to_clear = set()

    for addr, role, is_ready in result:
        if is_ready:
            logger.debug("Address %s with role %s is ready", addr, role)
            continue

        if role == "decode":
            keys_to_clear.update(
                key for key in all_control_meta.keys() if key.endswith(f"__{addr}")
            )
        else:
            if node.node_rank != 0:
                keys_to_clear.update(
                    key
                    for key in all_control_meta.keys()
                    if key.startswith(f"{addr}__")
                )

    for key in keys_to_clear:
        all_control_meta[key] = []
        all_data_meta[key] = ""

    if keys_to_clear:
        session.execute(
            update(CommGroupPair)
            .where(
                CommGroupPair.cluster_name == node.cluster_name,
                CommGroupPair.comm_key.in_(keys_to_clear),
            )
            .values(is_active=False)
        )

    active_keys = set(all_control_meta.keys()) - keys_to_clear
    if active_keys:
        session.execute(
            update(CommGroupPair)
            .where(
                CommGroupPair.cluster_name == node.cluster_name,
                CommGroupPair.comm_key.in_(active_keys),
            )
            .values(is_active=True)
        )

    session.flush()

    return all_control_meta, all_data_meta


def register_comm_group(
    session: Session,
    node: NodeInfo,
    comm_key: str,
    comm_id: str,
) -> CommGroupPair:
    """Create or update a communication group registration."""
    if node.role != "prefill" or node.node_rank != 0:
        raise InvalidCommGroupError("Only prefill rank-0 nodes may register comm ids")

    try:
        prefill_addr, decode_addr = comm_key.split("__", 1)
    except ValueError as exc:
        raise InvalidCommGroupError(
            "comm_key must follow 'prefill__decode' format"
        ) from exc

    pair = session.scalar(
        select(CommGroupPair).where(
            CommGroupPair.comm_key == comm_key,
            CommGroupPair.cluster_name == node.cluster_name,
        )
    )
    control_channel_meta = _rebuild_comm_group_control_meta(
        session, node.cluster_name, comm_key
    )
    if pair is None:
        pair = CommGroupPair(
            comm_key=comm_key,
            is_active=True,
            cluster_name=node.cluster_name,
            prefill_addr=prefill_addr,
            decode_addr=decode_addr,
            control_channel_meta=control_channel_meta,
            data_channel_meta=comm_id,
        )
        session.add(pair)
    else:
        pair.control_channel_meta = control_channel_meta
        pair.data_channel_meta = comm_id
    session.flush()
    return pair


def get_node(session: Session, node_id: str) -> NodeInfo:
    """Retrieve a node by its ID."""
    node = session.scalar(select(NodeInfo).where(NodeInfo.node_id == node_id))
    if not node:
        raise NodeNotFoundError(f"Node with id {node_id} not found")
    return node
