# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""
In-memory storage backend implementation, all data is stored in memory and will be lost when the service restarts.
"""
from typing import Dict, List, Optional, Tuple
import logging
import time
from models import ClusterInfo
from storage import StorageBackend

logger = logging.getLogger("comm_coordinator")


class MemoryStorage(StorageBackend):
    """In-memory storage backend, all data is saved in memory"""

    def __init__(self):
        # Cluster information: cluster_name -> ClusterInfo
        self.clusters: Dict[str, ClusterInfo] = {}

        # Node ID to group mapping: node_id -> (cluster_name, group_name, group_role)
        self.node_map: Dict[str, Tuple[str, str, str]] = {}
        
        # Node status tracking: node_id -> {"last_heartbeat": timestamp, "is_online": bool}
        self.node_status: Dict[str, Dict[str, any]] = {}

    def init(self) -> bool:
        """Initialize storage backend"""
        logger.info("In-memory storage backend initialized")
        return True

    def save_cluster(self, cluster: ClusterInfo) -> bool:
        """Save cluster information"""
        self.clusters[cluster.cluster_name] = cluster
        return True

    def get_cluster(self, cluster_name: str) -> Optional[ClusterInfo]:
        """Get cluster information"""
        return self.clusters.get(cluster_name)

    def delete_cluster(self, cluster_name: str) -> bool:
        """Delete cluster"""
        if cluster_name not in self.clusters:
            return False

        # Delete node mappings related to the cluster
        for node_id, (c_name, _, _) in list(self.node_map.items()):
            if c_name == cluster_name:
                del self.node_map[node_id]

        # Delete the cluster
        del self.clusters[cluster_name]
        return True

    def list_clusters(self) -> List[ClusterInfo]:
        """List all clusters"""
        return list(self.clusters.values())

    def update_node_map(
        self, node_id: str, cluster_name: str, group_name: str, group_role: str
    ) -> bool:
        """Update node mapping"""
        self.node_map[node_id] = (cluster_name, group_name, group_role)
        # Initialize node status if not exists
        if node_id not in self.node_status:
            self.node_status[node_id] = {
                "last_heartbeat": time.time(),
                "is_online": True
            }
        return True

    def delete_node_map(self, node_id: str) -> bool:
        """Delete node mapping"""
        if node_id in self.node_map:
            del self.node_map[node_id]
        if node_id in self.node_status:
            del self.node_status[node_id]
        return True

    def get_node_map(self, node_id: str) -> Optional[Tuple[str, str, str]]:
        """Get node mapping (cluster_name, group_name, group_role)"""
        return self.node_map.get(node_id)

    def save_node_map(
        self, node_id: str, cluster_name: str, group_name: str, group_role: str
    ) -> bool:
        """Save node mapping information"""
        self.node_map[node_id] = (cluster_name, group_name, group_role)
        # Update node status
        if node_id not in self.node_status:
            self.node_status[node_id] = {
                "last_heartbeat": time.time(),
                "is_online": True
            }
        return True

    def cleanup_inactive_nodes(
        self, timeout_seconds: int, delete_physically: bool = False
    ) -> List[str]:
        """Clean up nodes that haven't sent heartbeat for specified timeout.

        Args:
            timeout_seconds: Timeout in seconds to consider a node inactive
            delete_physically: If True, physically delete node data; if False, just mark as offline

        Returns:
            List of node IDs that were cleaned up
        """
        try:
            current_time = time.time()
            inactive_nodes = []

            # Find inactive nodes
            for node_id, status in self.node_status.items():
                time_since_heartbeat = current_time - status.get("last_heartbeat", 0)
                if time_since_heartbeat > timeout_seconds:
                    if status.get("is_online", True):  # Only process if currently online
                        inactive_nodes.append(node_id)

            if not inactive_nodes:
                logger.debug("No inactive nodes found for cleanup")
                return []

            logger.info(f"Found {len(inactive_nodes)} inactive nodes for cleanup")

            if delete_physically:
                # Physically delete node data
                for node_id in inactive_nodes:
                    if node_id in self.node_map:
                        del self.node_map[node_id]
                    if node_id in self.node_status:
                        del self.node_status[node_id]
                logger.info(f"Physically deleted {len(inactive_nodes)} inactive nodes")
            else:
                # Just mark nodes as offline
                for node_id in inactive_nodes:
                    if node_id in self.node_status:
                        self.node_status[node_id]["is_online"] = False
                logger.info(f"Marked {len(inactive_nodes)} nodes as offline")

            return inactive_nodes
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to cleanup inactive nodes: {e}")
            return []

    def cleanup_inactive_groups(
        self, timeout_seconds: int, delete_physically: bool = False
    ) -> Dict[str, List[str]]:
        """Clean up entire groups that have timed-out nodes.

        Args:
            timeout_seconds: Timeout in seconds to consider a node inactive
            delete_physically: If True, physically delete group data; if False, just mark as offline

        Returns:
            Dict with cluster_name as key and list of affected group names as value
        """
        try:
            current_time = time.time()
            affected_clusters = {}

            # Find groups with inactive nodes
            groups_with_inactive_nodes = {}  # (cluster_name, group_name, group_role) -> [node_ids]

            for node_id, status in self.node_status.items():
                time_since_heartbeat = current_time - status.get("last_heartbeat", 0)
                if (time_since_heartbeat > timeout_seconds and 
                        status.get("is_online", True) and 
                        node_id in self.node_map):
                    
                    cluster_name, group_name, group_role = self.node_map[node_id]
                    group_key = (cluster_name, group_name, group_role)
                    
                    if group_key not in groups_with_inactive_nodes:
                        groups_with_inactive_nodes[group_key] = []
                    groups_with_inactive_nodes[group_key].append(node_id)

            # Process affected groups
            for (cluster_name, group_name, group_role), node_ids in groups_with_inactive_nodes.items():
                logger.info(
                    f"Found {len(node_ids)} timed-out nodes in {group_role} "
                    f"group '{group_name}' of cluster '{cluster_name}'"
                )

                if cluster_name not in affected_clusters:
                    affected_clusters[cluster_name] = []

                group_key = f"{group_name}({group_role})"
                affected_clusters[cluster_name].append(group_key)

                if delete_physically:
                    # Find all nodes in this group and delete them
                    nodes_to_delete = [
                        nid for nid, (c, g, r) in self.node_map.items()
                        if c == cluster_name and g == group_name and r == group_role
                    ]
                    
                    for node_id in nodes_to_delete:
                        if node_id in self.node_map:
                            del self.node_map[node_id]
                        if node_id in self.node_status:
                            del self.node_status[node_id]
                    
                    logger.info(
                        f"Physically deleted all data for {group_role} group '{group_name}' in cluster '{cluster_name}'"
                    )
                else:
                    # Just mark nodes in the group as offline
                    for node_id in node_ids:
                        if node_id in self.node_status:
                            self.node_status[node_id]["is_online"] = False
                    
                    logger.info(
                        f"Marked {len(node_ids)} nodes offline in {group_role} "
                        f"group '{group_name}' of cluster '{cluster_name}'"
                    )

            return affected_clusters
        except Exception as e:  # pylint: disable=broad-except
            logger.exception(f"Failed to cleanup inactive groups: {e}")
            return {}
