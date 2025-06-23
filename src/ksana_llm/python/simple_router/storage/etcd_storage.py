# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""
etcd storage backend implementation, all data is stored in etcd distributed key-value store,
ensuring data persistence and consistency.
"""
from typing import List, Optional, Tuple
from datetime import datetime
import logging
import json
import traceback
from models import ClusterInfo, GroupInfo, NodeInfo, CommGroupPair, DeviceInfoRequest
from storage import StorageBackend
import etcd3
from config import settings


logger = logging.getLogger("comm_coordinator")


class EtcdStorage(StorageBackend):
    """etcd storage backend, all data is saved in etcd distributed key-value storage system"""

    def __init__(self):
        # etcd client
        self.client = None
        self.prefix = settings.etcd_prefix

        # Ensure prefix ends with a slash
        if not self.prefix.endswith("/"):
            self.prefix += "/"

    def init(self) -> bool:
        """Initialize etcd storage backend"""
        try:
            # Connect to etcd
            if (
                hasattr(settings, "etcd_user")
                and hasattr(settings, "etcd_password")
                and settings.etcd_user
                and settings.etcd_password
            ):
                self.client = etcd3.client(
                    host=settings.etcd_host,
                    port=settings.etcd_port,
                    timeout=settings.etcd_timeout,
                    user=settings.etcd_user,
                    password=settings.etcd_password,
                )
            else:
                self.client = etcd3.client(
                    host=settings.etcd_host,
                    port=settings.etcd_port,
                    timeout=settings.etcd_timeout,
                )

            # Test connection
            self.client.get("/test_key")

            logger.info(
                f"etcd storage backend initialized, connected to {settings.etcd_host}:{settings.etcd_port}"
            )
            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to initialize etcd storage backend: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def _load_cache(self):
        """Load cache data from etcd"""
        # 删除本地缓存相关逻辑，不再加载缓存
        pass

    def get_cluster(self, cluster_name: str) -> Optional[ClusterInfo]:
        """Get cluster information from etcd"""
        try:
            # 直接从 etcd 获取
            cluster_key = f"{self.prefix}clusters/{cluster_name}"
            value = self.client.get(cluster_key)[0]

            if not value:
                return None

            # Parse cluster data
            cluster_data = json.loads(value.decode("utf-8"))

            # Restore objects from prefill_groups, decode_groups, comm_groups
            prefill_groups = {}
            decode_groups = {}
            comm_groups = {}

            # Restore prefill groups
            for pg_name, pg_data in cluster_data.get("prefill_groups", {}).items():
                group = self._json_to_group(pg_data)
                if group:
                    prefill_groups[pg_name] = group

            # Restore decode groups
            for cg_name, cg_data in cluster_data.get("decode_groups", {}).items():
                group = self._json_to_group(cg_data)
                if group:
                    decode_groups[cg_name] = group

            # Restore communication group pairs
            for comm_key, comm_data in cluster_data.get("comm_groups", {}).items():
                comm_pair = self._json_to_comm_pair(comm_data)
                if comm_pair:
                    comm_groups[comm_key] = comm_pair

            # Create cluster object
            cluster = ClusterInfo(
                cluster_name=cluster_name,
                prefill_groups=prefill_groups,
                decode_groups=decode_groups,
                comm_groups=comm_groups,
                created_at=datetime.fromisoformat(
                    cluster_data.get("created_at", datetime.now().isoformat())
                ),
                last_updated=datetime.fromisoformat(
                    cluster_data.get("last_updated", datetime.now().isoformat())
                ),
                is_active=cluster_data.get("is_active", True),
                inactive_group_timeout=cluster_data.get("inactive_group_timeout", 300),
            )

            return cluster

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to get cluster from etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def delete_cluster(self, cluster_name: str) -> bool:
        """Delete cluster from etcd"""
        try:
            # Get cluster
            cluster = self.get_cluster(cluster_name)
            if not cluster:
                return False

            # Delete related node mappings
            node_ids = []

            # Collect all node IDs to delete
            for group in list(cluster.prefill_groups.values()) + list(
                cluster.decode_groups.values()
            ):
                node_ids.extend(group.nodes.keys())

            # Delete node mappings from etcd
            for node_id in node_ids:
                node_map_key = f"{self.prefix}node_map/{node_id}"
                self.client.delete(node_map_key)

            # Delete cluster from etcd
            cluster_key = f"{self.prefix}clusters/{cluster_name}"
            self.client.delete(cluster_key)

            return True

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to delete cluster from etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def list_clusters(self) -> List[ClusterInfo]:
        """List all clusters from etcd"""
        # 直接从 etcd 全量获取，不用本地缓存
        clusters = []
        clusters_prefix = f"{self.prefix}clusters/"
        for value, metadata in self.client.get_prefix(clusters_prefix):
            if not value:
                continue
            cluster_data = json.loads(value.decode("utf-8"))

            # Restore objects from prefill_groups, decode_groups, comm_groups
            prefill_groups = {}
            decode_groups = {}
            comm_groups = {}

            # Restore prefill groups
            for pg_name, pg_data in cluster_data.get("prefill_groups", {}).items():
                group = self._json_to_group(pg_data)
                if group:
                    prefill_groups[pg_name] = group

            # Restore decode groups
            for cg_name, cg_data in cluster_data.get("decode_groups", {}).items():
                group = self._json_to_group(cg_data)
                if group:
                    decode_groups[cg_name] = group

            # Restore communication group pairs
            for comm_key, comm_data in cluster_data.get("comm_groups", {}).items():
                comm_pair = self._json_to_comm_pair(comm_data)
                if comm_pair:
                    comm_groups[comm_key] = comm_pair

            # Create cluster object
            cluster = ClusterInfo(
                cluster_name=metadata.key.decode("utf-8").split("/")[-1],
                prefill_groups=prefill_groups,
                decode_groups=decode_groups,
                comm_groups=comm_groups,
                created_at=datetime.fromisoformat(
                    cluster_data.get("created_at", datetime.now().isoformat())
                ),
                last_updated=datetime.fromisoformat(
                    cluster_data.get("last_updated", datetime.now().isoformat())
                ),
                is_active=cluster_data.get("is_active", True),
                inactive_group_timeout=cluster_data.get("inactive_group_timeout", 300),
            )

            clusters.append(cluster)

        return clusters

    def update_node_map(
        self, node_id: str, cluster_name: str, group_name: str, group_role: str
    ) -> bool:
        """Update node mapping to etcd"""
        try:
            # Create node mapping data
            node_map_data = {
                "cluster_name": cluster_name,
                "group_name": group_name,
                "group_role": group_role,
            }

            # Save to etcd
            node_map_key = f"{self.prefix}node_map/{node_id}"
            self.client.put(node_map_key, json.dumps(node_map_data))

            return True

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to update node mapping to etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def delete_node_map(self, node_id: str) -> bool:
        """Delete node mapping from etcd"""
        try:
            # Delete from etcd
            node_map_key = f"{self.prefix}node_map/{node_id}"
            self.client.delete(node_map_key)

            return True

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to delete node mapping from etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def get_node_map(self, node_id: str) -> Optional[Tuple[str, str, str]]:
        """Get node mapping from etcd"""
        try:
            # 直接从 etcd 获取
            node_map_key = f"{self.prefix}node_map/{node_id}"
            value = self.client.get(node_map_key)[0]

            if not value:
                return None

            # Parse node mapping data
            node_map_data = json.loads(value.decode("utf-8"))
            mapping = (
                node_map_data.get("cluster_name", ""),
                node_map_data.get("group_name", ""),
                node_map_data.get("group_role", ""),
            )
            return mapping

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to get node mapping from etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def save_node_map(
        self, node_id: str, cluster_name: str, group_name: str, group_role: str
    ) -> bool:
        """Save node mapping information"""
        try:
            # Create node mapping data
            node_map_data = {
                "cluster_name": cluster_name,
                "group_name": group_name,
                "group_role": group_role,
            }

            # Save to etcd
            node_map_key = f"{self.prefix}node_map/{node_id}"
            self.client.put(node_map_key, json.dumps(node_map_data))

            return True

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to save node mapping to etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def save_cluster(self, cluster: ClusterInfo) -> bool:
        """Save cluster information to etcd"""
        try:
            # comm_groups 必须全是 CommGroupPair，不做 dict 兼容
            for k, v in list(cluster.comm_groups.items()):
                if not isinstance(v, CommGroupPair):
                    raise TypeError(f"comm_groups[{k}] is not CommGroupPair: {type(v)}")

            # Serialize cluster data
            cluster_data = self._serialize_cluster(cluster)

            # Save to etcd
            cluster_key = f"{self.prefix}clusters/{cluster.cluster_name}"
            self.client.put(cluster_key, cluster_data)

            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to save cluster to etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def _serialize_cluster(self, cluster: ClusterInfo) -> str:
        """Serialize ClusterInfo (and all nested objects) to JSON string for etcd storage"""

        def group_to_dict(group: GroupInfo):
            return {
                "group_id": group.group_id,
                "group_name": group.group_name,
                "group_role": group.group_role,
                "cluster_name": group.cluster_name,
                "nodes": {nid: node_to_dict(n) for nid, n in group.nodes.items()},
                "created_at": (
                    group.created_at.isoformat()
                    if isinstance(group.created_at, datetime)
                    else group.created_at
                ),
                "last_updated": (
                    group.last_updated.isoformat()
                    if isinstance(group.last_updated, datetime)
                    else group.last_updated
                ),
                "is_ready": group.is_ready,
                "world_size": group.world_size,
            }

        def node_to_dict(node: NodeInfo):
            return {
                "node_id": node.node_id,
                "hostname": node.hostname,
                "inference_addr": node.inference_addr,
                "coordinator_port": node.coordinator_port,
                "cluster_name": node.cluster_name,
                "group_name": node.group_name,
                "group_role": node.group_role,
                "node_rank": node.node_rank,
                "world_size": node.world_size,
                "devices": [
                    d.dict() if hasattr(d, "dict") else d for d in node.devices
                ],
                "is_online": node.is_online,
                "last_heartbeat": (
                    node.last_heartbeat.isoformat()
                    if isinstance(node.last_heartbeat, datetime)
                    else node.last_heartbeat
                ),
                "comm_id": node.comm_id,
                "job_id": node.job_id,
                "start_time": node.start_time,
            }

        def comm_pair_to_dict(pair: CommGroupPair):
            # Ensure all fields are present, even if None
            return {
                "prefill_group": pair.prefill_group,
                "decode_group": pair.decode_group,
                "comm_id": pair.comm_id if hasattr(pair, "comm_id") else None,
                "created_at": (
                    pair.created_at.isoformat()
                    if hasattr(pair, "created_at")
                    and isinstance(pair.created_at, datetime)
                    else pair.created_at if hasattr(pair, "created_at") else None
                ),
                "last_active": (
                    pair.last_active.isoformat()
                    if hasattr(pair, "last_active")
                    and isinstance(pair.last_active, datetime)
                    else pair.last_active if hasattr(pair, "last_active") else None
                ),
            }

        data = {
            "cluster_name": cluster.cluster_name,
            "prefill_groups": {
                k: group_to_dict(v) for k, v in cluster.prefill_groups.items()
            },
            "decode_groups": {
                k: group_to_dict(v) for k, v in cluster.decode_groups.items()
            },
            "comm_groups": {
                k: comm_pair_to_dict(v) for k, v in cluster.comm_groups.items()
            },
            "created_at": (
                cluster.created_at.isoformat()
                if isinstance(cluster.created_at, datetime)
                else cluster.created_at
            ),
            "last_updated": (
                cluster.last_updated.isoformat()
                if isinstance(cluster.last_updated, datetime)
                else cluster.last_updated
            ),
            "is_active": cluster.is_active,
            "inactive_group_timeout": cluster.inactive_group_timeout,
        }
        return json.dumps(data)

    def _json_to_group(self, data: dict) -> GroupInfo:
        """Convert dict to GroupInfo, including nested NodeInfo objects."""
        nodes = {}
        for node_id, node_data in data.get("nodes", {}).items():
            devices = node_data.get("devices", [])
            if devices and isinstance(devices[0], dict):
                devices = [DeviceInfoRequest(**d) for d in devices]
            node_data["devices"] = devices
            node = NodeInfo(
                node_id=node_data.get("node_id"),
                hostname=node_data.get("hostname"),
                inference_addr=node_data.get("inference_addr"),
                coordinator_port=node_data.get("coordinator_port"),
                cluster_name=node_data.get("cluster_name"),
                group_name=node_data.get("group_name"),
                group_role=node_data.get("group_role"),
                node_rank=node_data.get("node_rank"),
                world_size=node_data.get("world_size"),
                devices=node_data.get("devices"),
                last_heartbeat=(
                    datetime.fromisoformat(node_data["last_heartbeat"])
                    if isinstance(node_data.get("last_heartbeat"), str)
                    else node_data.get("last_heartbeat", datetime.now())
                ),
                is_online=node_data.get("is_online", True),
                comm_id=node_data.get("comm_id"),
                job_id=node_data.get("job_id"),
                start_time=node_data.get("start_time"),
            )
            nodes[node_id] = node
        return GroupInfo(
            group_id=data.get("group_id"),
            group_name=data["group_name"],
            group_role=data["group_role"],
            cluster_name=data["cluster_name"],
            nodes=nodes,
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else data.get("created_at", datetime.now())
            ),
            last_updated=(
                datetime.fromisoformat(data["last_updated"])
                if isinstance(data.get("last_updated"), str)
                else data.get("last_updated", datetime.now())
            ),
            is_ready=data.get("is_ready", False),
            world_size=data.get("world_size"),
        )

    def _json_to_comm_pair(self, data: dict) -> CommGroupPair:
        """Convert dict to CommGroupPair. 兼容老数据，补全缺失字段"""
        # 兼容老数据：如果没有 comm_id/created_at/last_active 字段，补全
        comm_id = data.get("comm_id", "")
        created_at = data.get("created_at", datetime.now().isoformat())
        last_active = data.get("last_active", datetime.now().isoformat())
        return CommGroupPair(
            prefill_group=data["prefill_group"],
            decode_group=data["decode_group"],
            comm_id=comm_id,
            created_at=(
                datetime.fromisoformat(created_at)
                if isinstance(created_at, str)
                else created_at
            ),
            last_active=(
                datetime.fromisoformat(last_active)
                if isinstance(last_active, str)
                else last_active
            ),
        )
