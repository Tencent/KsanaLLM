# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""
In-memory storage backend implementation, all data is stored in memory and will be lost when the service restarts.
"""
from typing import Dict, List, Optional, Tuple
import logging
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
        return True

    def delete_node_map(self, node_id: str) -> bool:
        """Delete node mapping"""
        if node_id in self.node_map:
            del self.node_map[node_id]
            return True
        return False

    def get_node_map(self, node_id: str) -> Optional[Tuple[str, str, str]]:
        """Get node mapping (cluster_name, group_name, group_role)"""
        return self.node_map.get(node_id)

    def save_node_map(
        self, node_id: str, cluster_name: str, group_name: str, group_role: str
    ) -> bool:
        """Save node mapping information"""
        self.node_map[node_id] = (cluster_name, group_name, group_role)
        return True
