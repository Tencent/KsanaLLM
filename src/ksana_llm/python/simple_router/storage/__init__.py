# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""Storage backend interfaces and factory for the KsanaLLM Router service.

This module provides an abstract interface for storage backends and a factory
for creating concrete storage backend instances. It includes implementations for
memory-based and db-based storage.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type
import logging
from config import settings
from models import ClusterInfo


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend interface."""

    @abstractmethod
    def init(self) -> bool:
        """Initialize storage backend.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        pass

    @abstractmethod
    def save_cluster(self, cluster: ClusterInfo) -> bool:
        """Save cluster information.

        Args:
            cluster: The cluster information to save.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_cluster(self, cluster_name: str) -> Optional[ClusterInfo]:
        """Get cluster information.

        Args:
            cluster_name: The name of the cluster to retrieve.

        Returns:
            Optional[ClusterInfo]: The cluster information if found, None otherwise.
        """
        pass

    @abstractmethod
    def delete_cluster(self, cluster_name: str) -> bool:
        """Delete cluster.

        Args:
            cluster_name: The name of the cluster to delete.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def list_clusters(self) -> List[ClusterInfo]:
        """List all clusters.

        Returns:
            List[ClusterInfo]: A list of all cluster information objects.
        """
        pass

    @abstractmethod
    def update_node_map(
        self, node_id: str, cluster_name: str, group_name: str, group_role: str
    ) -> bool:
        """Update node mapping.

        Args:
            node_id: The ID of the node to update.
            cluster_name: The name of the cluster the node belongs to.
            group_name: The name of the group the node belongs to.
            group_role: The role of the group (prefill or decode).

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def delete_node_map(self, node_id: str) -> bool:
        """Delete node mapping.

        Args:
            node_id: The ID of the node to delete.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_node_map(self, node_id: str) -> Optional[Tuple[str, str, str]]:
        """Get node mapping (cluster_name, group_name, group_role).

        Args:
            node_id: The ID of the node to retrieve.

        Returns:
            Optional[Tuple[str, str, str]]: A tuple containing the cluster name,
                group name, and group role if found, None otherwise.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass


class StorageFactory:
    """Storage backend factory class."""

    _backends: Dict[str, Type[StorageBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type[StorageBackend]):
        """Register storage backend.

        Args:
            name: The name to register the backend under.
            backend_class: The storage backend class to register.
        """
        cls._backends[name] = backend_class

    @classmethod
    def create(cls, mode: str = None) -> StorageBackend:
        """Create storage backend instance.

        Args:
            mode: The storage mode to use. If None, uses the mode from settings.

        Returns:
            StorageBackend: A concrete storage backend instance.
        """
        if mode is None:
            mode = settings.storage_mode

        if mode not in cls._backends:
            logging.warning(
                f"Storage mode '{mode}' not found, using memory storage mode"
            )
            mode = "memory"

        # Create backend instance with configuration
        if mode == "mysql" :
            backend = cls._backends[mode](
                host=settings.mysql_host,
                port=settings.mysql_port,
                user=settings.mysql_user,
                password=settings.mysql_password,
                database=settings.mysql_database,
                charset=settings.mysql_charset,
                autocommit=settings.mysql_autocommit,
            )
        else:
            backend = cls._backends[mode]()

        backend.init()
        return backend


# Import concrete storage backend implementations
from .memory_storage import MemoryStorage

# Try to import MySQL storage if pymysql is available
from .mysql_storage import MySQLStorage

# Register storage backends
StorageFactory.register("memory", MemoryStorage)
StorageFactory.register("mysql", MySQLStorage)

# Create default storage backend instance
default_storage = StorageFactory.create()
