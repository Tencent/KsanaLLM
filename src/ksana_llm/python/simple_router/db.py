# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""Database coordinator for the KsanaLLM Router service.

This module contains the main database coordinator logic for managing clusters,
groups, nodes, and Communication communication in the router service.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Any

from config import settings
from models import ClusterInfo, GroupInfo, NodeInfo, CommGroupPair
from storage import default_storage

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("comm_coordinator")


class CoordinatorDB:
    """Communication Coordinator Database.

    This class manages the coordination between prefill and decode nodes,
    including node registration, heartbeat tracking, and communication group setup.

    Attributes:
        storage: The storage backend interface.
        start_time: The time when the coordinator was started.
        requests_served: The number of requests served.
        heartbeat_timeout: The timeout for node heartbeats in seconds.
        cleanup_interval: The interval for cleanup tasks in seconds.
        cleanup_thread: The thread for periodic cleanup tasks.
        prefill_change_listeners: Listeners for prefill group changes.
        decode_change_listeners: Listeners for decode group changes.
    """

    def __init__(self, heartbeat_timeout=30, cleanup_interval=10):
        """Initialize the coordinator database.

        Args:
            heartbeat_timeout: The timeout for node heartbeats in seconds.
            cleanup_interval: The interval for cleanup tasks in seconds.
        """
        # Use storage interface
        self.storage = default_storage

        # Statistics
        self.start_time = time.time()
        self.requests_served = 0

        # Heartbeat timeout settings (seconds)
        self.heartbeat_timeout = (
            heartbeat_timeout if heartbeat_timeout else settings.node_heartbeat_timeout
        )

        # Cleanup interval (seconds)
        self.cleanup_interval = (
            cleanup_interval if cleanup_interval else settings.cleanup_interval
        )

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_routine, daemon=True
        )
        self.cleanup_thread.start()

        # Register change listeners
        self.prefill_change_listeners = []
        self.decode_change_listeners = []

        logger.info(
            f"Communication coordinator database initialized, using {settings.storage_mode} storage mode"
        )

    def register_cluster(self, cluster_name: str) -> str:
        """Register a new cluster.

        Args:
            cluster_name: The name of the cluster to register.

        Returns:
            The name of the registered cluster.

        Raises:
            ValueError: If a cluster with the same name already exists.
        """
        self.requests_served += 1

        # Check if cluster already exists
        if self.storage.get_cluster(cluster_name):
            raise ValueError(f"Cluster '{cluster_name}' already exists")

        # Create new cluster
        cluster = ClusterInfo(cluster_name=cluster_name)

        # Save cluster
        self.storage.save_cluster(cluster)

        logger.info(f"New cluster registered: {cluster_name}")

        return cluster_name

    def register_group(
        self, cluster_name: str, group_name: str, group_role: str
    ) -> str:
        """Register a new group in a cluster.

        Args:
            cluster_name: The name of the cluster.
            group_name: The name of the group.
            group_role: The role of the group ("prefill" or "decode").

        Returns:
            The ID of the registered group.

        Raises:
            ValueError: If the cluster does not exist or a group with the same name
                       and role already exists.
        """
        self.requests_served += 1

        # Check if cluster exists
        cluster = self.storage.get_cluster(cluster_name)
        if not cluster:
            raise ValueError(f"Cluster '{cluster_name}' does not exist")

        # Check if group already exists
        if group_role == "prefill" and group_name in cluster.prefill_groups:
            raise ValueError(
                f"Prefill group '{group_name}' already exists in cluster '{cluster_name}'"
            )
        if group_role == "decode" and group_name in cluster.decode_groups:
            raise ValueError(
                f"Decode group '{group_name}' already exists in cluster '{cluster_name}'"
            )

        # Create group
        group = GroupInfo(
            group_name=group_name, group_role=group_role, cluster_name=cluster_name
        )

        # Add to cluster
        cluster.add_group(group)

        # Save cluster
        self.storage.save_cluster(cluster)

        # Trigger group change event
        if group_role == "prefill":
            self._notify_prefill_change("create", cluster_name, group_name)
        else:
            self._notify_decode_change("create", cluster_name, group_name)

        logger.info(
            f"Registered {group_role} group '{group_name}' to cluster '{cluster_name}'"
        )

        return group.group_id

    def register_node(self, node: NodeInfo) -> str:
        """Register a new node in the database.

        Args:
            node: The node information to register.

        Returns:
            The node_id of the registered node.

        Raises:
            ValueError: If the cluster could not be created.
        """
        logger.debug(
            f"Starting node registration: {node.hostname}, cluster: {node.cluster_name}, group: {node.group_name}"
        )

        # Create or get cluster
        cluster = self._get_or_create_cluster(node.cluster_name)
        if not cluster:
            raise ValueError(f"Could not create cluster: {node.cluster_name}")

        # Check if group exists, create if not
        group = cluster.get_group(node.group_name, node.group_role)
        if not group:
            logger.debug(
                f"Creating new group: {node.group_name}, type: {node.group_role}"
            )
            group = GroupInfo(
                group_name=node.group_name,
                group_role=node.group_role,
                cluster_name=node.cluster_name,
            )
            # Add the group first
            cluster.add_group(group)

        # Check if there are nodes with the same rank
        existing_node = group.get_node_by_rank(node.node_rank)
        if existing_node and existing_node.node_id != node.node_id:
            if (
                not existing_node.is_online
                or (datetime.now() - existing_node.last_heartbeat).total_seconds() > 60
            ):
                logger.info(f"Removing existing node: {existing_node.node_id}")
                self.delete_node(existing_node.node_id)

        # Add node to group
        if node.node_id not in group.nodes:
            group.add_node(node)

            # If this is the first node and a prefill Master node (rank 0),
            # update all related comm group Communication IDs
            if node.group_role == "prefill" and node.node_rank == 0 and node.comm_id:
                # Use the provided comm_id
                logger.info(
                    f"Getting Communication ID from prefill group '{node.group_name}' master node: {node.comm_id}"
                )
                logger.debug(
                    f"Current decode groups in cluster: {list(cluster.decode_groups.keys())}"
                )

                # Iterate through all decode groups to update or create comm groups
                decode_group_count = len(cluster.decode_groups)
                if decode_group_count == 0:
                    logger.warning(
                        f"No decode groups found when registering prefill master node {node.node_id}."
                        "Communication groups will be created later when decode groups are registered."
                    )

                for decode_group_name, _ in cluster.decode_groups.items():
                    comm_key = f"{node.group_name}__{decode_group_name}"  # 使用双下划线格式保持一致
                    # Check if comm group already exists
                    if comm_key in cluster.comm_groups:
                        # Update existing comm group Communication ID
                        cluster.comm_groups[comm_key].comm_id = node.comm_id
                        logger.info(
                            f"Updated Communication ID for comm group {comm_key}"
                        )
                    else:
                        # Create new comm group
                        cluster.comm_groups[comm_key] = CommGroupPair(
                            prefill_group=node.group_name,
                            decode_group=decode_group_name,
                            comm_id=node.comm_id,
                        )
                        logger.info(
                            f"Created comm group {comm_key} and set Communication ID"
                        )

            # If this is a decode group being created, create comm groups with existing prefill groups
            elif node.group_role == "decode":
                logger.debug(
                    f"Decode node registered, current prefill groups: {list(cluster.prefill_groups.keys())}"
                )

                # Check all existing prefill groups and create comm groups
                for prefill_group_name, prefill_group in cluster.prefill_groups.items():
                    # Find prefill master node (rank 0) to get comm_id
                    master_node = None
                    for prefill_node in prefill_group.nodes.values():
                        if prefill_node.node_rank == 0:
                            master_node = prefill_node
                            break

                    if master_node and master_node.comm_id:
                        comm_key = f"{prefill_group_name}__{node.group_name}"
                        if comm_key not in cluster.comm_groups:
                            # Create new comm group
                            cluster.comm_groups[comm_key] = CommGroupPair(
                                prefill_group=prefill_group_name,
                                decode_group=node.group_name,
                                comm_id=master_node.comm_id,
                            )
                            logger.info(
                                f"Created comm group {comm_key} for existing prefill group"
                                f"with decode group {node.group_name}"
                            )
                        else:
                            logger.debug(f"Comm group {comm_key} already exists")
                    else:
                        logger.debug(
                            f"No master node with comm_id found in prefill group {prefill_group_name}"
                        )
        else:
            # Update existing node
            group.nodes[node.node_id] = node

        # Update group state
        cluster.update_group_ready_state(node.group_name, node.group_role)

        # Save cluster to storage
        self.storage.save_cluster(cluster)

        # Ensure node mapping is saved - explicitly add node mapping
        self.storage.update_node_map(
            node.node_id, node.cluster_name, node.group_name, node.group_role
        )

        logger.info(
            f"Node successfully registered: {node.node_id}, cluster: {node.cluster_name}, group: {node.group_name}"
        )
        return node.node_id

    def update_heartbeat(self, node_id: str) -> bool:
        """Update node heartbeat.

        Args:
            node_id: The ID of the node.

        Returns:
            True if the heartbeat was updated successfully, False otherwise.
        """
        self.requests_served += 1

        # Find node
        node_mapping = self.storage.get_node_map(node_id)
        if not node_mapping:
            return False

        cluster_name, group_name, group_role = node_mapping

        # Get cluster
        cluster = self.storage.get_cluster(cluster_name)
        if not cluster:
            return False

        # Get group
        group = cluster.get_group(group_name, group_role)
        if not group:
            return False

        # Check node
        if node_id not in group.nodes:
            return False

        # Update heartbeat
        node = group.nodes[node_id]
        node.last_heartbeat = datetime.now()
        node.is_online = True

        # Update group state (if it wasn't ready before)
        if not group.is_ready:
            cluster.update_group_ready_state(group_name, group_role)

        # Save cluster
        self.storage.save_cluster(cluster)

        return True

    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Get node information.

        Args:
            node_id: The ID of the node.

        Returns:
            The node information if found, None otherwise.
        """
        self.requests_served += 1

        try:
            # Find node
            node_mapping = self.storage.get_node_map(node_id)
            if not node_mapping:
                logger.warning(f"Node mapping not found: {node_id}")
                return None

            cluster_name, group_name, group_role = node_mapping

            # Get cluster
            cluster = self.storage.get_cluster(cluster_name)
            if not cluster:
                logger.warning(
                    f"Cluster {cluster_name} for node {node_id} does not exist"
                )
                return None

            # Get group
            group = cluster.get_group(group_name, group_role)
            if not group:
                logger.warning(f"Group {group_name} for node {node_id} does not exist")
                return None

            # Return node
            node = group.nodes.get(node_id)
            if not node:
                logger.warning(f"Node {node_id} does not exist in the group")

            return node
        except Exception as e:  # pylint: disable=broad-except
            logger.exception(f"Error getting node: {str(e)}")
            return None

    def get_group(
        self, cluster_name: str, group_name: str, group_role: str
    ) -> Optional[GroupInfo]:
        """Get group information.

        Args:
            cluster_name: The name of the cluster.
            group_name: The name of the group.
            group_role: The role of the group.

        Returns:
            The group information if found, None otherwise.
        """
        self.requests_served += 1

        # Get cluster
        cluster = self.storage.get_cluster(cluster_name)
        if not cluster:
            return None

        # Get group
        return cluster.get_group(group_name, group_role)

    def get_heartbeat_info(self, node_id: str) -> Dict[str, Any]:
        """Get minimal information set for node heartbeat response.

        This method returns only the necessary information for a node heartbeat response,
        ensuring that data modification issues are avoided.

        Args:
            node_id: The ID of the node.

        Returns:
            A dictionary containing the node's heartbeat information.
        """
        self.requests_served += 1

        # Helper function to get online node addresses
        def get_online_node_addresses(group):
            addresses = []
            for node in group.nodes.values():
                if node.is_online:
                    addresses.extend(
                        (
                            node.node_rank,
                            d.device_id,
                            f"{d.device_ip}:{node.coordinator_port}",
                        )
                        for d in node.devices
                    )
            return addresses

        # Find node mapping
        node_mapping = self.storage.get_node_map(node_id)
        if not node_mapping:
            return {}

        cluster_name, group_name, group_role = node_mapping

        # Get cluster, group, and node
        cluster = self.storage.get_cluster(cluster_name)
        if not cluster:
            return {}

        group = cluster.get_group(group_name, group_role)
        if not group:
            return {}

        node = group.nodes.get(node_id)
        if not node:
            return {}

        # Create base response
        minimal_response = {
            "node_id": node_id,
            "node_name": f"{node.group_name}_node_rank_{node.node_rank}",
            "is_online": node.is_online,
            "group_ready": group.is_ready,
            "node_role": node.group_role,
            "node_rank": node.node_rank,
            "timestamp": node.last_heartbeat.isoformat(),
            "coordinator_port": node.coordinator_port,
            "inference_addr": node.inference_addr,
        }

        comm_group_to_address = {}
        comm_group_to_id = {}

        # Get current group addresses
        current_group = cluster.get_group(group_name, group_role)
        if not current_group:
            logger.warning(
                f"Group {group_name} does not exist in cluster {cluster_name}"
            )
            return minimal_response

        current_addrs = get_online_node_addresses(current_group)

        # Determine target groups based on node role
        target_groups = (
            cluster.decode_groups.items()
            if node.group_role == "prefill"
            else cluster.prefill_groups.items()
        )

        for target_group_name, target_group in target_groups:
            if not target_group.is_ready:
                continue

            target_addrs = get_online_node_addresses(target_group)
            if not target_addrs:
                continue

            # Construct communication group key
            comm_key = (
                f"{group_name}__{target_group_name}"
                if node.group_role == "prefill"
                else f"{target_group_name}__{group_name}"
            )

            # Get comm_group id if exists
            comm_group = cluster.comm_groups.get(comm_key)
            comm_group_to_id[comm_key] = comm_group.comm_id if comm_group else None
            if node.group_role == "prefill":
                comm_group_to_address[comm_key] = current_addrs + target_addrs
            else:
                comm_group_to_address[comm_key] = target_addrs + current_addrs

        minimal_response["comm_group_to_address"] = comm_group_to_address
        minimal_response["comm_group_to_id"] = comm_group_to_id

        return minimal_response

    def delete_cluster(self, cluster_name: str) -> bool:
        """Delete cluster.

        Args:
            cluster_name: The name of the cluster to delete.

        Returns:
            True if the cluster was deleted successfully, False otherwise.
        """
        self.requests_served += 1

        # Get cluster
        cluster = self.storage.get_cluster(cluster_name)
        if not cluster:
            logger.warning(f"Attempted to delete non-existent cluster: {cluster_name}")
            return False

        # Delete cluster, storage layer will also clean up related node mappings
        result = self.storage.delete_cluster(cluster_name)

        if result:
            # Trigger change events for all related groups
            for group_name in cluster.prefill_groups:
                self._notify_prefill_change("delete", cluster_name, group_name)

            for group_name in cluster.decode_groups:
                self._notify_decode_change("delete", cluster_name, group_name)

            logger.info(f"Successfully deleted cluster: {cluster_name}")

        return result

    def delete_group(self, cluster_name: str, group_name: str, group_role: str) -> bool:
        """Delete group.

        Args:
            cluster_name: The name of the cluster.
            group_name: The name of the group.
            group_role: The role of the group.

        Returns:
            True if the group was deleted successfully, False otherwise.
        """
        self.requests_served += 1

        # Get cluster
        cluster = self.storage.get_cluster(cluster_name)
        if not cluster:
            logger.warning(
                f"Attempted to delete group from non-existent cluster {cluster_name}"
            )
            return False

        # Get group to delete
        group = cluster.get_group(group_name, group_role)
        if not group:
            logger.warning(
                f"Attempted to delete non-existent {group_role} group: {group_name}"
            )
            return False

        # Delete node mappings
        for node_id in group.nodes.keys():
            self.storage.delete_node_map(node_id)

        # Remove group from cluster
        if group_role == "prefill":
            del cluster.prefill_groups[group_name]
            # Trigger group change event
            self._notify_prefill_change("delete", cluster_name, group_name)
        else:  # decode
            del cluster.decode_groups[group_name]
            # Trigger group change event
            self._notify_decode_change("delete", cluster_name, group_name)

        # Update cluster last modified time and save
        cluster.last_updated = datetime.now()
        self.storage.save_cluster(cluster)

        logger.info(
            f"Successfully deleted {group_role} group: {group_name} (cluster: {cluster_name})"
        )
        return True

    def delete_node(self, node_id: str) -> bool:
        """Delete node.

        Args:
            node_id: The ID of the node to delete.

        Returns:
            True if the node was deleted successfully, False otherwise.
        """
        self.requests_served += 1

        # Find node
        node_mapping = self.storage.get_node_map(node_id)
        if not node_mapping:
            logger.warning(f"Attempted to delete non-existent node: {node_id}")
            return False

        cluster_name, group_name, group_role = node_mapping

        # Get cluster
        cluster = self.storage.get_cluster(cluster_name)
        if not cluster:
            logger.warning(f"Cluster {cluster_name} for node {node_id} does not exist")
            # Clean up node mapping even if cluster doesn't exist
            self.storage.delete_node_map(node_id)
            return False

        # Get group
        group = cluster.get_group(group_name, group_role)
        if not group:
            logger.warning(f"Group {group_name} for node {node_id} does not exist")
            # Clean up node mapping even if group doesn't exist
            self.storage.delete_node_map(node_id)
            return False

        # Delete node
        if node_id in group.nodes:
            # Get node info (for logging)
            node = group.nodes[node_id]
            # Delete node from group
            del group.nodes[node_id]
            # Delete node mapping
            self.storage.delete_node_map(node_id)

            # Update group and cluster last modified time
            group.last_updated = datetime.now()
            cluster.last_updated = datetime.now()

            # Update group ready state
            cluster.update_group_ready_state(group_name, group_role)

            # Save cluster
            self.storage.save_cluster(cluster)

            logger.info(
                f"Successfully deleted node: {node.hostname} (ID={node_id}) from "
                f"{group_role} group '{group_name}' of cluster '{cluster_name}'"
            )
            return True
        else:
            logger.warning(f"Node {node_id} does not exist in group {group_name}")
            # Clean up node mapping even if node doesn't exist in group
            self.storage.delete_node_map(node_id)
            return False

    def _cleanup_routine(self):
        """Background thread for periodically cleaning up timed-out nodes."""
        logger.info(
            f"Starting node timeout cleanup thread (interval: {self.cleanup_interval}s, "
            f"timeout: {self.heartbeat_timeout}s)"
        )

        cleanup_count = 0
        while True:
            try:
                cleanup_count += 1
                logger.debug(
                    f"Running cleanup check #{cleanup_count} (interval: {self.cleanup_interval}s)"
                )
                self._check_all_nodes_timeout()
                logger.info(
                        f"Cleanup thread is running normally (completed {cleanup_count} checks)"
                    )

                time.sleep(self.cleanup_interval)
            except Exception as e:  # pylint: disable=broad-except
                logger.exception(f"Error in cleanup thread: {str(e)}. ")

    def _check_all_nodes_timeout(self):
        """Check heartbeat status of all nodes."""
        total_timeouts = 0

        try:
            clusters = self.storage.list_clusters()
            total_nodes = 0

            logger.debug(f"Checking {len(clusters)} clusters for node timeouts")

            for cluster in clusters:
                # Count total nodes in this cluster
                cluster_nodes = 0
                for group in cluster.prefill_groups.values():
                    cluster_nodes += len(group.nodes)
                for group in cluster.decode_groups.values():
                    cluster_nodes += len(group.nodes)
                total_nodes += cluster_nodes

                # Check node heartbeats
                timeout_nodes = cluster.check_nodes_heartbeat(self.heartbeat_timeout)

                if timeout_nodes:
                    total_timeouts += len(timeout_nodes)
                    logger.warning(
                        f"Cluster '{cluster.cluster_name}' has {len(timeout_nodes)} timed-out "
                        f"nodes (out of {cluster_nodes} total)"
                    )

                # Update state of all affected groups
                for group_name, group in list(cluster.prefill_groups.items()):
                    affected = any(node_id in group.nodes for node_id in timeout_nodes)
                    if affected:
                        old_ready = group.is_ready
                        cluster.update_group_ready_state(group_name, "prefill")
                        if old_ready and not group.is_ready:
                            logger.warning(
                                f"Prefill group '{group_name}' is now not ready due to node timeout"
                            )
                            # Trigger prefill group change
                            self._notify_prefill_change(
                                "update", cluster.cluster_name, group_name
                            )

                for group_name, group in list(cluster.decode_groups.items()):
                    affected = any(node_id in group.nodes for node_id in timeout_nodes)
                    if affected:
                        old_ready = group.is_ready
                        cluster.update_group_ready_state(group_name, "decode")
                        if old_ready and not group.is_ready:
                            logger.warning(
                                f"Decode group '{group_name}' is now not ready due to node timeout"
                            )
                            # Trigger decode group change
                            self._notify_decode_change(
                                "update", cluster.cluster_name, group_name
                            )

                # Save cluster state
                self.storage.save_cluster(cluster)

            # Cleanup inactive nodes in storage backend if there were timeouts
            if total_timeouts > 0:
                # Cleanup individual timed-out nodes (delete physically)
                deleted_nodes = self.storage.cleanup_inactive_nodes(
                    self.heartbeat_timeout,
                    delete_physically=True,  # Always delete physically when timeout occurs
                )
                if deleted_nodes:
                    logger.info(
                        f"Storage backend cleaned up {len(deleted_nodes)} nodes: {', '.join(deleted_nodes)}"
                    )

                logger.info(f"Cleanup complete, total {total_timeouts} nodes timed out")
            else:
                # 定期输出状态，即使没有超时节点
                logger.debug(
                    f"Heartbeat check completed: {total_nodes} total nodes, no timeouts detected"
                )

        except Exception as e:  # pylint: disable=broad-except
            logger.exception(f"Error during timeout check: {str(e)}. ")

    def register_comm_id(self, comm_key: str, comm_id: str) -> bool:
        """Update Communication ID for a comm group.

        Args:
            comm_key: Communication group key, format is "prefill_group_decode_group".
            comm_id: Corresponding Communication ID value.

        Returns:
            Whether update was successful.
        """
        logger.info(f"Updating Communication ID for comm group {comm_key}: {comm_id}")

        # Parse comm group key to extract prefill and decode group
        try:
            parts = comm_key.split("__")
            if len(parts) < 2:
                logger.error(f"Invalid comm group key format: {comm_key}")
                return False

            # Find matching comm group in all clusters
            for cluster in self.storage.list_clusters():
                # Check if key exists in cluster comm groups
                if comm_key in cluster.comm_groups:
                    # Update Communication ID
                    cluster.comm_groups[comm_key].comm_id = comm_id
                    cluster.comm_groups[comm_key].last_active = datetime.now()
                    cluster.last_updated = datetime.now()

                    # Save cluster
                    self.storage.save_cluster(cluster)

                    logger.info(
                        f"Successfully updated Communication ID for comm group {comm_key}"
                    )
                    return True

            logger.warning(f"Comm group not found: {comm_key}")
            return False

        except Exception as e:  # pylint: disable=broad-except
            logger.exception(f"Error updating comm group Communication ID: {str(e)}")
            return False

    def _notify_prefill_change(self, action: str, cluster_name: str, group_name: str):
        """Notify prefill group change.

        Args:
            action: The action that occurred ("create", "update", or "delete").
            cluster_name: The name of the cluster.
            group_name: The name of the group.
        """
        for callback in self.prefill_change_listeners:
            try:
                callback(action, cluster_name, group_name)
            except Exception as e:  # pylint: disable=broad-except
                logger.exception(f"Error calling prefill group change callback: {str(e)}")

    def _notify_decode_change(self, action: str, cluster_name: str, group_name: str):
        """Notify decode group change.

        Args:
            action: The action that occurred ("create", "update", or "delete").
            cluster_name: The name of the cluster.
            group_name: The name of the group.
        """
        for callback in self.decode_change_listeners:
            try:
                callback(action, cluster_name, group_name)
            except Exception as e:  # pylint: disable=broad-except
                logger.exception(f"Error calling decode group change callback: {str(e)}")

    def _get_or_create_cluster(self, cluster_name: str) -> Optional[ClusterInfo]:
        """Get an existing cluster or create a new one if it doesn't exist.

        Args:
            cluster_name: The name of the cluster to get or create.

        Returns:
            The existing cluster or a newly created one, or None if creation fails.
        """
        # Try to get the existing cluster
        cluster = self.storage.get_cluster(cluster_name)
        if cluster:
            return cluster

        # If the cluster doesn't exist, create a new one with default settings
        logger.info(f"Creating new cluster: {cluster_name} (auto-creation)")

        # Create the cluster
        cluster = ClusterInfo(
            cluster_name=cluster_name,
        )

        # Save the cluster
        self.storage.save_cluster(cluster)

        return cluster

    def _save_db(self):
        """Save all database changes - placeholder for compatibility.

        This method is called after node registration.
        In our current implementation, changes are saved directly to storage
        when they happen, so this method is just a placeholder.
        """
        pass


# Global database instance
db = CoordinatorDB(
    heartbeat_timeout=settings.node_heartbeat_timeout,
    cleanup_interval=settings.cleanup_interval,
)
