# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""
etcd storage backend implementation, all data is stored in etcd distributed key-value store,
ensuring data persistence and consistency.
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import json
import traceback
from models import ClusterInfo, GroupInfo, NodeInfo, CommGroupPair
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

        # Local cache to improve read performance
        self.cache_clusters: Dict[str, ClusterInfo] = {}
        self.cache_node_map: Dict[str, Tuple[str, str, str]] = {}

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

            # Load cache from etcd
            self._load_cache()

            logger.info(
                f"etcd storage backend initialized, connected to {settings.etcd_host}:{settings.etcd_port}"
            )
            return True
        except Exception as e:    # pylint: disable=broad-except
            logger.error(f"Failed to initialize etcd storage backend: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def _load_cache(self):
        """Load cache data from etcd"""
        try:
            # Load cluster data
            clusters_prefix = f"{self.prefix}clusters/"
            for value, metadata in self.client.get_prefix(clusters_prefix):
                if not value:
                    continue

                # Parse cluster data
                cluster_data = json.loads(value.decode("utf-8"))
                cluster_name = cluster_data.get("cluster_name")

                if not cluster_name:
                    continue

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
                    inactive_group_timeout=cluster_data.get(
                        "inactive_group_timeout", 300
                    ),
                )

                # Add to cache
                self.cache_clusters[cluster_name] = cluster

            # Load node mappings
            node_map_prefix = f"{self.prefix}node_map/"
            for value, metadata in self.client.get_prefix(node_map_prefix):
                if not value:
                    continue

                # Parse node mapping data
                node_map_data = json.loads(value.decode("utf-8"))
                node_id = metadata.key.decode("utf-8").split("/")[-1]

                # Add to cache
                self.cache_node_map[node_id] = (
                    node_map_data.get("cluster_name", ""),
                    node_map_data.get("group_name", ""),
                    node_map_data.get("group_role", ""),
                )

            logger.info(
                f"Loaded {len(self.cache_clusters)} clusters and {len(self.cache_node_map)} node mappings from etcd"
            )

        except Exception as e:    # pylint: disable=broad-except
            logger.error(f"Failed to load cache from etcd: {str(e)}")
            logger.debug(traceback.format_exc())

    def _json_to_group(self, group_data) -> Optional[GroupInfo]:
        """Convert JSON data to GroupInfo object"""
        try:
            # Restore nodes
            nodes = {}
            for node_id, node_data in group_data.get("nodes", {}).items():
                node = NodeInfo(
                    node_id=node_id,
                    hostname=node_data.get("hostname", ""),
                    inference_addr=node_data.get("inference_addr", ""),
                    cluster_name=node_data.get("cluster_name", ""),
                    group_name=node_data.get("group_name", ""),
                    group_role=node_data.get("group_role", ""),
                    node_rank=node_data.get("node_rank", 0),
                    devices=node_data.get("devices", []),
                    world_size=node_data.get("world_size", 0),
                    coordinator_port=node_data.get("coordinator_port", 0),
                    last_heartbeat=datetime.fromisoformat(
                        node_data.get("last_heartbeat", datetime.now().isoformat())
                    ),
                    is_online=node_data.get("is_online", True),
                    comm_id=node_data.get("comm_id"),
                    job_id=node_data.get("job_id"),
                    start_time=node_data.get("start_time"),
                )
                nodes[node_id] = node

            # Create group object
            group = GroupInfo(
                group_id=group_data.get("group_id", ""),
                group_name=group_data.get("group_name", ""),
                group_role=group_data.get("group_role", ""),
                cluster_name=group_data.get("cluster_name", ""),
                nodes=nodes,
                created_at=datetime.fromisoformat(
                    group_data.get("created_at", datetime.now().isoformat())
                ),
                last_updated=datetime.fromisoformat(
                    group_data.get("last_updated", datetime.now().isoformat())
                ),
                is_ready=group_data.get("is_ready", False),
                world_size=group_data.get("world_size"),
            )
            return group
        except Exception as e:    # pylint: disable=broad-except
            logger.error(f"Failed to convert JSON to GroupInfo: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def _json_to_comm_pair(self, comm_data) -> Optional[CommGroupPair]:
        """Convert JSON data to CommGroupPair object"""
        try:
            comm_pair = CommGroupPair(
                prefill_group=comm_data.get("prefill_group", ""),
                decode_group=comm_data.get("decode_group", ""),
                comm_id=comm_data.get("comm_id", ""),
                created_at=datetime.fromisoformat(
                    comm_data.get("created_at", datetime.now().isoformat())
                ),
                last_active=datetime.fromisoformat(
                    comm_data.get("last_active", datetime.now().isoformat())
                ),
            )
            return comm_pair
        except Exception as e:    # pylint: disable=broad-except
            logger.error(f"Failed to convert JSON to CommGroupPair: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def _serialize_cluster(self, cluster: ClusterInfo) -> str:
        """Serialize cluster object to JSON string"""
        # Create cluster data dictionary
        cluster_dict = {
            "cluster_name": cluster.cluster_name,
            "created_at": cluster.created_at.isoformat(),
            "last_updated": cluster.last_updated.isoformat(),
            "is_active": cluster.is_active,
            "inactive_group_timeout": cluster.inactive_group_timeout,
            "prefill_groups": {},
            "decode_groups": {},
            "comm_groups": {},
        }

        # Serialize prefill groups
        for group_name, group in cluster.prefill_groups.items():
            cluster_dict["prefill_groups"][group_name] = self._serialize_group(group)

        # Serialize decode groups
        for group_name, group in cluster.decode_groups.items():
            cluster_dict["decode_groups"][group_name] = self._serialize_group(group)

        # Serialize communication group pairs
        for comm_key, comm_pair in cluster.comm_groups.items():
            cluster_dict["comm_groups"][comm_key] = self._serialize_comm_pair(comm_pair)

        return json.dumps(cluster_dict)

    def _serialize_group(self, group: GroupInfo) -> dict:
        """Serialize group object to dictionary"""
        group_dict = {
            "group_id": group.group_id,
            "group_name": group.group_name,
            "group_role": group.group_role,
            "cluster_name": group.cluster_name,
            "created_at": group.created_at.isoformat(),
            "last_updated": group.last_updated.isoformat(),
            "is_ready": group.is_ready,
            "world_size": group.world_size,
            "nodes": {},
        }

        # Serialize nodes
        for node_id, node in group.nodes.items():
            node_dict = {
                "node_id": node.node_id,
                "hostname": node.hostname,
                "inference_addr": node.inference_addr,
                "cluster_name": node.cluster_name,
                "group_name": node.group_name,
                "group_role": node.group_role,
                "node_rank": node.node_rank,
                "devices": node.devices,
                "last_heartbeat": node.last_heartbeat.isoformat(),
                "is_online": node.is_online,
                "comm_id": node.comm_id,
                "job_id": node.job_id,
                "start_time": node.start_time,
            }
            group_dict["nodes"][node_id] = node_dict

        return group_dict

    def _serialize_comm_pair(self, comm_pair: CommGroupPair) -> dict:
        """Serialize communication group pair object to dictionary"""
        return {
            "prefill_group": comm_pair.prefill_group,
            "decode_group": comm_pair.decode_group,
            "comm_id": comm_pair.comm_id,
            "created_at": comm_pair.created_at.isoformat(),
            "last_active": comm_pair.last_active.isoformat(),
        }

    def save_cluster(self, cluster: ClusterInfo) -> bool:
        """Save cluster information to etcd"""
        try:
            # Serialize cluster data
            cluster_data = self._serialize_cluster(cluster)

            # Save to etcd
            cluster_key = f"{self.prefix}clusters/{cluster.cluster_name}"
            self.client.put(cluster_key, cluster_data)

            # Update local cache
            self.cache_clusters[cluster.cluster_name] = cluster

            return True
        except Exception as e:    # pylint: disable=broad-except
            logger.error(f"Failed to save cluster to etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def get_cluster(self, cluster_name: str) -> Optional[ClusterInfo]:
        """Get cluster information from etcd"""
        # First check cache
        if cluster_name in self.cache_clusters:
            return self.cache_clusters[cluster_name]

        try:
            # Get from etcd
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

            # Update cache
            self.cache_clusters[cluster_name] = cluster

            return cluster

        except Exception as e:    # pylint: disable=broad-except
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

                # Remove from cache
                if node_id in self.cache_node_map:
                    del self.cache_node_map[node_id]

            # Delete cluster from etcd
            cluster_key = f"{self.prefix}clusters/{cluster_name}"
            self.client.delete(cluster_key)

            # Remove from cache
            if cluster_name in self.cache_clusters:
                del self.cache_clusters[cluster_name]

            return True

        except Exception as e:    # pylint: disable=broad-except
            logger.error(f"Failed to delete cluster from etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def list_clusters(self) -> List[ClusterInfo]:
        """List all clusters from etcd"""
        # Return cluster list from cache
        return list(self.cache_clusters.values())

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

            # Update cache
            self.cache_node_map[node_id] = (cluster_name, group_name, group_role)

            return True

        except Exception as e:    # pylint: disable=broad-except
            logger.error(f"Failed to update node mapping to etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def delete_node_map(self, node_id: str) -> bool:
        """Delete node mapping from etcd"""
        try:
            # Delete from etcd
            node_map_key = f"{self.prefix}node_map/{node_id}"
            self.client.delete(node_map_key)

            # Remove from cache
            if node_id in self.cache_node_map:
                del self.cache_node_map[node_id]

            return True

        except Exception as e:    # pylint: disable=broad-except
            logger.error(f"Failed to delete node mapping from etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def get_node_map(self, node_id: str) -> Optional[Tuple[str, str, str]]:
        """Get node mapping from etcd"""
        # First check cache
        if node_id in self.cache_node_map:
            return self.cache_node_map[node_id]

        try:
            # Get from etcd
            node_map_key = f"{self.prefix}node_map/{node_id}"
            value = self.client.get(node_map_key)[0]

            if not value:
                return None

            # Parse node mapping data
            node_map_data = json.loads(value.decode("utf-8"))

            # Update cache
            mapping = (
                node_map_data.get("cluster_name", ""),
                node_map_data.get("group_name", ""),
                node_map_data.get("group_role", ""),
            )
            self.cache_node_map[node_id] = mapping

            return mapping

        except Exception as e:    # pylint: disable=broad-except
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

            # Update cache
            self.cache_node_map[node_id] = (cluster_name, group_name, group_role)

            return True

        except Exception as e:    # pylint: disable=broad-except
            logger.error(f"Failed to save node mapping to etcd: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
