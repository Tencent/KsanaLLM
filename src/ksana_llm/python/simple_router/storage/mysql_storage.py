# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""
Refactored MySQL storage backend implementation for the KsanaLLM Router service.

This module provides a MySQL-based storage backend that persists cluster, group,
and node information in a MySQL database without using stored procedures.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

import pymysql
from pymysql import Connection, Error as PyMySQLError
from pymysql.cursors import DictCursor

from models import ClusterInfo, GroupInfo, NodeInfo, CommGroupPair, DeviceInfoRequest
from storage import StorageBackend

logger = logging.getLogger("comm_coordinator")


class MySQLStorage(StorageBackend):
    """MySQL storage backend for cluster, group, and node information."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "ksana_llm_router",
        charset: str = "utf8mb4",
        autocommit: bool = True,
        connect_timeout: int = 10,
        read_timeout: int = 10,
        write_timeout: int = 10,
        max_retries: int = 3,
        **kwargs,
    ):
        """Initialize MySQL storage backend.

        Args:
            host: MySQL server hostname
            port: MySQL server port
            user: MySQL username
            password: MySQL password
            database: Database name
            charset: Character set
            autocommit: Whether to enable autocommit
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            write_timeout: Write timeout in seconds
            max_retries: Maximum number of connection retries
            **kwargs: Additional connection parameters
        """
        self.connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": charset,
            "autocommit": autocommit,
            "connect_timeout": connect_timeout,
            "read_timeout": read_timeout,
            "write_timeout": write_timeout,
            "cursorclass": DictCursor,
            **kwargs,
        }
        self.database_name = database
        self.max_retries = max_retries
        self._connection_pool = None

    @contextmanager
    def _get_connection(self):
        """Context manager for getting database connections with automatic cleanup."""
        connection = None
        try:
            connection = self._create_connection()
            yield connection
        except Exception as e:  # pylint: disable=broad-except
            if connection:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()

    def _create_connection(self) -> Connection:
        """Create a new database connection with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                connection = pymysql.connect(**self.connection_params)
                return connection
            except PyMySQLError as e:
                last_exception = e
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    import time

                    time.sleep(1)  # Wait 1 second before retry

        logger.error(f"Failed to connect to MySQL after {self.max_retries} attempts")
        raise last_exception

    def _execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a SELECT query and return results."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                return cursor.fetchall()

    def _execute_update(self, query: str, params: tuple = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                result = cursor.execute(query, params or ())
                conn.commit()
                return result

    def _execute_batch_update(self, query: str, params_list: List[tuple]) -> int:
        """Execute batch INSERT/UPDATE/DELETE queries."""
        if not params_list:
            return 0

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                result = cursor.executemany(query, params_list)
                conn.commit()
                return result

    def init(self) -> bool:
        """Initialize storage backend and create necessary tables."""
        try:
            logger.info(
                f"MySQL storage backend initialized successfully (database: {self.database_name})"
            )
            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to initialize MySQL storage: {e}")
            return False

    def save_cluster(self, cluster: ClusterInfo) -> bool:
        """Save cluster information to database using transactions."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Start transaction
                    conn.begin()

                    # Save cluster
                    self._save_cluster_data(cursor, cluster)

                    # Save groups and nodes
                    self._save_groups_data(cursor, cluster)

                    # Save communication groups
                    self._save_comm_groups_data(cursor, cluster)

                    # Commit transaction
                    conn.commit()

            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to save cluster {cluster.cluster_name}: {e}")
            return False

    def _save_cluster_data(self, cursor, cluster: ClusterInfo):
        """Save cluster basic information."""
        cluster_data = {
            "prefill_groups": {
                k: self._group_to_dict(v) for k, v in cluster.prefill_groups.items()
            },
            "decode_groups": {
                k: self._group_to_dict(v) for k, v in cluster.decode_groups.items()
            },
            "comm_groups": {
                k: self._comm_group_to_dict(v) for k, v in cluster.comm_groups.items()
            },
        }

        query = """
        INSERT INTO `clusters` (cluster_name, created_at, last_updated, is_active, inactive_group_timeout, cluster_data)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        last_updated = VALUES(last_updated),
        is_active = VALUES(is_active),
        inactive_group_timeout = VALUES(inactive_group_timeout),
        cluster_data = VALUES(cluster_data)
        """

        params = (
            cluster.cluster_name,
            cluster.created_at,
            cluster.last_updated,
            cluster.is_active,
            cluster.inactive_group_timeout,
            json.dumps(cluster_data, default=str),
        )

        cursor.execute(query, params)

    def _save_groups_data(self, cursor, cluster: ClusterInfo):
        """Save groups and their nodes to database."""
        all_groups = list(cluster.prefill_groups.values()) + list(
            cluster.decode_groups.values()
        )

        # Prepare batch data for groups
        group_params = []
        node_params = []
        node_map_params = []

        for group in all_groups:
            # Group data
            group_params.append(
                (
                    group.group_id,
                    group.group_name,
                    group.group_role,
                    group.cluster_name,
                    group.created_at,
                    group.last_updated,
                    group.is_ready,
                    group.world_size,
                    json.dumps(
                        {
                            "nodes": {
                                k: self._node_to_dict(v) for k, v in group.nodes.items()
                            }
                        },
                        default=str,
                    ),
                )
            )

            # Node data
            for node in group.nodes.values():
                node_params.append(
                    (
                        node.node_id,
                        node.hostname,
                        node.inference_addr,
                        node.coordinator_port,
                        node.cluster_name,
                        node.group_name,
                        node.group_role,
                        node.node_rank,
                        node.world_size,
                        json.dumps(
                            (
                                [device.dict() for device in node.devices]
                                if node.devices
                                else []
                            ),
                            default=str,
                        ),
                        node.last_heartbeat,
                        node.is_online,
                        node.comm_id,
                        node.job_id,
                        node.start_time,
                        json.dumps(node.to_dict(), default=str),
                    )
                )

                # Node mapping data
                node_map_params.append(
                    (node.node_id, node.cluster_name, node.group_name, node.group_role)
                )

        # Batch insert/update groups
        if group_params:
            group_query = """
            INSERT INTO `groups` (group_id, group_name, group_role, cluster_name, created_at, last_updated, is_ready, world_size, group_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            last_updated = VALUES(last_updated),
            is_ready = VALUES(is_ready),
            world_size = VALUES(world_size),
            group_data = VALUES(group_data)
            """
            cursor.executemany(group_query, group_params)

        # Batch insert/update nodes
        if node_params:
            node_query = """
            INSERT INTO `nodes` (node_id, hostname, inference_addr, coordinator_port, cluster_name, group_name, group_role, 
                              node_rank, world_size, devices, last_heartbeat, is_online, comm_id, job_id, start_time, node_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            hostname = VALUES(hostname),
            inference_addr = VALUES(inference_addr),
            coordinator_port = VALUES(coordinator_port),
            last_heartbeat = VALUES(last_heartbeat),
            is_online = VALUES(is_online),
            comm_id = VALUES(comm_id),
            node_data = VALUES(node_data)
            """
            cursor.executemany(node_query, node_params)

        # Batch insert/update node mappings
        if node_map_params:
            node_map_query = """
            INSERT INTO `node_map` (node_id, cluster_name, group_name, group_role)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            cluster_name = VALUES(cluster_name),
            group_name = VALUES(group_name),
            group_role = VALUES(group_role)
            """
            cursor.executemany(node_map_query, node_map_params)

    def _save_comm_groups_data(self, cursor, cluster: ClusterInfo):
        """Save communication groups to database."""
        if not cluster.comm_groups:
            return

        comm_params = []
        for comm_key, comm_group in cluster.comm_groups.items():
            comm_params.append(
                (
                    comm_key,
                    comm_group.prefill_group,
                    comm_group.decode_group,
                    comm_group.comm_id,
                    cluster.cluster_name,
                    comm_group.created_at,
                    comm_group.last_active,
                    json.dumps(self._comm_group_to_dict(comm_group), default=str),
                )
            )

        comm_query = """
        INSERT INTO `comm_groups` (comm_key, prefill_group, decode_group, comm_id, cluster_name, created_at, last_active, comm_data)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        comm_id = VALUES(comm_id),
        last_active = VALUES(last_active),
        comm_data = VALUES(comm_data)
        """
        cursor.executemany(comm_query, comm_params)

    def get_cluster(self, cluster_name: str) -> Optional[ClusterInfo]:
        """Get cluster information from database."""
        try:
            # Get cluster basic info
            cluster_rows = self._execute_query(
                "SELECT * FROM `clusters` WHERE cluster_name = %s", (cluster_name,)
            )

            if not cluster_rows:
                return None

            cluster_row = cluster_rows[0]

            # Get groups, nodes, and communication groups in parallel queries
            groups_rows = self._execute_query(
                "SELECT * FROM `groups` WHERE cluster_name = %s", (cluster_name,)
            )

            nodes_rows = self._execute_query(
                "SELECT * FROM `nodes` WHERE cluster_name = %s", (cluster_name,)
            )

            comm_rows = self._execute_query(
                "SELECT * FROM `comm_groups` WHERE cluster_name = %s", (cluster_name,)
            )

            # Build cluster object
            cluster = self._build_cluster_from_db_data(
                cluster_row, groups_rows, nodes_rows, comm_rows
            )
            return cluster

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to get cluster {cluster_name}: {e}")
            return None

    def _build_cluster_from_db_data(
        self,
        cluster_row: Dict,
        groups_rows: List[Dict],
        nodes_rows: List[Dict],
        comm_rows: List[Dict],
    ) -> ClusterInfo:
        """Build ClusterInfo object from database data."""

        # Group nodes by group key
        nodes_by_group = {}
        for node_row in nodes_rows:
            group_key = (node_row["group_name"], node_row["group_role"])
            if group_key not in nodes_by_group:
                nodes_by_group[group_key] = []
            nodes_by_group[group_key].append(self._node_from_db_row(node_row))

        # Build groups
        prefill_groups = {}
        decode_groups = {}

        for group_row in groups_rows:
            group = self._group_from_db_row(group_row)
            group_key = (group.group_name, group.group_role)

            # Add nodes to group
            if group_key in nodes_by_group:
                for node in nodes_by_group[group_key]:
                    group.nodes[node.node_id] = node

            if group.group_role == "prefill":
                prefill_groups[group.group_name] = group
            else:
                decode_groups[group.group_name] = group

        # Build communication groups
        comm_groups = {}
        for comm_row in comm_rows:
            comm_group = self._comm_group_from_db_row(comm_row)
            comm_groups[comm_row["comm_key"]] = comm_group

        # Create cluster
        cluster = ClusterInfo(
            cluster_name=cluster_row["cluster_name"],
            prefill_groups=prefill_groups,
            decode_groups=decode_groups,
            comm_groups=comm_groups,
            created_at=cluster_row["created_at"],
            last_updated=cluster_row["last_updated"],
            is_active=cluster_row["is_active"],
            inactive_group_timeout=cluster_row["inactive_group_timeout"],
        )

        return cluster

    def _node_from_db_row(self, node_row: Dict) -> NodeInfo:
        """Create NodeInfo object from database row."""
        devices_data = json.loads(node_row["devices"]) if node_row["devices"] else []
        devices = [DeviceInfoRequest(**device) for device in devices_data]

        return NodeInfo(
            node_id=node_row["node_id"],
            hostname=node_row["hostname"],
            inference_addr=node_row["inference_addr"],
            coordinator_port=node_row["coordinator_port"],
            cluster_name=node_row["cluster_name"],
            group_name=node_row["group_name"],
            group_role=node_row["group_role"],
            node_rank=node_row["node_rank"],
            world_size=node_row["world_size"],
            devices=devices,
            last_heartbeat=node_row["last_heartbeat"],
            is_online=node_row["is_online"],
            comm_id=node_row["comm_id"],
            job_id=node_row["job_id"],
            start_time=node_row["start_time"],
        )

    def _group_from_db_row(self, group_row: Dict) -> GroupInfo:
        """Create GroupInfo object from database row."""
        return GroupInfo(
            group_id=group_row["group_id"],
            group_name=group_row["group_name"],
            group_role=group_row["group_role"],
            cluster_name=group_row["cluster_name"],
            nodes={},  # Will be populated later
            created_at=group_row["created_at"],
            last_updated=group_row["last_updated"],
            is_ready=group_row["is_ready"],
            world_size=group_row["world_size"],
        )

    def _comm_group_from_db_row(self, comm_row: Dict) -> CommGroupPair:
        """Create CommGroupPair object from database row."""
        return CommGroupPair(
            prefill_group=comm_row["prefill_group"],
            decode_group=comm_row["decode_group"],
            comm_id=comm_row["comm_id"],
            created_at=comm_row["created_at"],
            last_active=comm_row["last_active"],
        )

    def delete_cluster(self, cluster_name: str) -> bool:
        """Delete cluster from database."""
        try:
            # Due to foreign key constraints, deleting cluster will cascade delete all related data
            rows_affected = self._execute_update(
                "DELETE FROM `clusters` WHERE cluster_name = %s", (cluster_name,)
            )

            # Also clean up node_map (in case of orphaned records)
            self._execute_update(
                "DELETE FROM `node_map` WHERE cluster_name = %s", (cluster_name,)
            )

            return rows_affected > 0
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to delete cluster {cluster_name}: {e}")
            return False

    def list_clusters(self) -> List[ClusterInfo]:
        """List all active clusters from database."""
        try:
            cluster_rows = self._execute_query(
                "SELECT cluster_name FROM `clusters` WHERE is_active = TRUE ORDER BY cluster_name"
            )

            clusters = []
            for row in cluster_rows:
                cluster = self.get_cluster(row["cluster_name"])
                if cluster:
                    clusters.append(cluster)

            return clusters
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to list clusters: {e}")
            return []

    def update_node_map(
        self, node_id: str, cluster_name: str, group_name: str, group_role: str
    ) -> bool:
        """Update node mapping in database."""
        try:
            query = """
            INSERT INTO `node_map` (node_id, cluster_name, group_name, group_role)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            cluster_name = VALUES(cluster_name),
            group_name = VALUES(group_name),
            group_role = VALUES(group_role)
            """

            self._execute_update(query, (node_id, cluster_name, group_name, group_role))
            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to update node map for {node_id}: {e}")
            return False

    def delete_node_map(self, node_id: str) -> bool:
        """Delete node mapping from database."""
        try:
            rows_affected = self._execute_update(
                "DELETE FROM `node_map` WHERE node_id = %s", (node_id,)
            )
            return rows_affected > 0
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to delete node map for {node_id}: {e}")
            return False

    def get_node_map(self, node_id: str) -> Optional[Tuple[str, str, str]]:
        """Get node mapping from database."""
        try:
            rows = self._execute_query(
                "SELECT cluster_name, group_name, group_role FROM `node_map` WHERE node_id = %s",
                (node_id,),
            )

            if rows:
                row = rows[0]
                return (row["cluster_name"], row["group_name"], row["group_role"])

            return None
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to get node map for {node_id}: {e}")
            return None

    def save_node_map(
        self, node_id: str, cluster_name: str, group_name: str, group_role: str
    ) -> bool:
        """Save node mapping to database."""
        return self.update_node_map(node_id, cluster_name, group_name, group_role)

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
            # Log cleanup start with database statistics
            logger.debug(
                f"Starting cleanup_inactive_nodes (timeout: {timeout_seconds}s, delete_physically: {delete_physically})"
            )
            self._log_database_stats("before cleanup")

            # Find inactive nodes (that have been marked as offline by heartbeat check)
            query = """
            SELECT node_id, cluster_name, group_name, group_role, node_rank, last_heartbeat, is_online
            FROM `nodes` 
            WHERE is_online = FALSE 
            AND last_heartbeat < DATE_SUB(NOW(), INTERVAL %s SECOND)
            """

            inactive_nodes = self._execute_query(query, (timeout_seconds,))
            node_ids = [row["node_id"] for row in inactive_nodes]

            if not node_ids:
                logger.debug("No inactive nodes found for cleanup")
                return []

            # Log detailed information about nodes to be cleaned
            logger.info(f"Found {len(node_ids)} inactive nodes for cleanup:")
            for node_row in inactive_nodes:
                logger.info(
                    f"  - Node {node_row['node_id']}: {node_row['group_role']} group '{node_row['group_name']}' "
                    f"rank {node_row['node_rank']} in cluster '{node_row['cluster_name']}' "
                    f"(last_heartbeat: {node_row['last_heartbeat']}, online: {node_row['is_online']})"
                )

            if node_ids and delete_physically:
                # Check for prefill master nodes (rank 0) that need comm_groups cleanup
                master_prefill_groups = []
                affected_groups = set()  # Track affected groups for cleanup

                for node_row in inactive_nodes:
                    # Track all affected groups
                    affected_groups.add(
                        (
                            node_row["cluster_name"],
                            node_row["group_name"],
                            node_row["group_role"],
                        )
                    )

                    if (
                        node_row["group_role"] == "prefill"
                        and node_row["node_rank"] == 0
                    ):
                        master_prefill_groups.append(
                            {
                                "cluster_name": node_row["cluster_name"],
                                "group_name": node_row["group_name"],
                            }
                        )

                # Before deleting nodes, check current node counts for affected groups
                group_node_counts = {}
                for cluster_name, group_name, group_role in affected_groups:
                    count_query = """
                    SELECT COUNT(*) as total_count
                    FROM `nodes`
                    WHERE cluster_name = %s AND group_name = %s AND group_role = %s
                    """
                    result = self._execute_query(
                        count_query, (cluster_name, group_name, group_role)
                    )
                    total_count = result[0]["total_count"] if result else 0
                    group_node_counts[(cluster_name, group_name, group_role)] = (
                        total_count
                    )

                # Delete comm_groups for prefill master nodes first
                for group_info in master_prefill_groups:
                    self._delete_comm_groups_for_prefill_group(
                        group_info["cluster_name"], group_info["group_name"]
                    )

                # Then physically delete node data
                self._delete_nodes_physically(node_ids)
                logger.info(f"Physically deleted {len(node_ids)} inactive nodes")

                # Now determine which groups became empty and clean them up
                empty_groups = []
                for (
                    cluster_name,
                    group_name,
                    group_role,
                ), total_count in group_node_counts.items():
                    # Count how many nodes from this group were deleted
                    deleted_from_group = sum(
                        1
                        for node_row in inactive_nodes
                        if (
                            node_row["cluster_name"] == cluster_name
                            and node_row["group_name"] == group_name
                            and node_row["group_role"] == group_role
                        )
                    )

                    remaining_nodes = total_count - deleted_from_group

                    if remaining_nodes <= 0:
                        # Group is now empty, clean up group metadata
                        empty_groups.append((cluster_name, group_name, group_role))
                        logger.debug(
                            f"Group {group_role} '{group_name}' in cluster '{cluster_name}' is now "
                            f"empty (had {total_count} nodes, deleted {deleted_from_group})"
                        )

                # Clean up empty groups
                for cluster_name, group_name, group_role in empty_groups:
                    try:
                        # Delete from groups table
                        delete_group_query = """
                        DELETE FROM `groups` 
                        WHERE cluster_name = %s AND group_name = %s AND group_role = %s
                        """
                        deleted_groups = self._execute_update(
                            delete_group_query, (cluster_name, group_name, group_role)
                        )
                        logger.debug(
                            f"Cleaned up empty {group_role} group '{group_name}' in cluster " 
                            f"'{cluster_name}' (deleted {deleted_groups} group records)"
                        )
                    except Exception as e:  # pylint: disable=broad-except
                        logger.warning(
                            f"Failed to clean up empty group {group_name}: {e}"
                        )

                if empty_groups:
                    group_names = [
                        f"{role} group '{name}'" for _, name, role in empty_groups
                    ]
                    logger.info(
                        f"Cleaned up {len(empty_groups)} empty groups: {', '.join(group_names)}"
                    )

                if master_prefill_groups:
                    group_names = [g["group_name"] for g in master_prefill_groups]
                    logger.info(
                        f"Also cleaned up comm_groups for prefill master groups: {', '.join(group_names)}"
                    )

                # Log database statistics after cleanup
                self._log_database_stats("after cleanup")

            elif node_ids and not delete_physically:
                # Just ensure nodes stay marked as offline (they should already be offline)
                logger.info(
                    f"Found {len(node_ids)} inactive nodes (already marked offline)"
                )

            return node_ids
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to cleanup inactive nodes: {e}")
            return []

    def cleanup_inactive_groups(
        self, timeout_seconds: int, delete_physically: bool = False
    ) -> Dict[str, List[str]]:
        """Clean up entire groups that have timed-out nodes.

        This method finds groups with timed-out nodes and optionally deletes all nodes
        and related data for those groups, making it easier for reregistration.

        Args:
            timeout_seconds: Timeout in seconds to consider a node inactive
            delete_physically: If True, physically delete group data; if False, just mark as offline

        Returns:
            Dict with cluster_name as key and list of affected group names as value
        """
        try:
            # Find inactive nodes grouped by their groups
            query = """
            SELECT cluster_name, group_name, group_role, COUNT(*) as timeout_count
            FROM `nodes` 
            WHERE is_online = TRUE 
            AND last_heartbeat < DATE_SUB(NOW(), INTERVAL %s SECOND)
            GROUP BY cluster_name, group_name, group_role
            """

            inactive_groups = self._execute_query(query, (timeout_seconds,))
            affected_clusters = {}

            for group_info in inactive_groups:
                cluster_name = group_info["cluster_name"]
                group_name = group_info["group_name"]
                group_role = group_info["group_role"]
                timeout_count = group_info["timeout_count"]

                logger.info(
                    f"Found {timeout_count} timed-out nodes in {group_role} group '{group_name}' "
                    f"of cluster '{cluster_name}'"
                )

                if cluster_name not in affected_clusters:
                    affected_clusters[cluster_name] = []

                group_key = f"{group_name}({group_role})"
                affected_clusters[cluster_name].append(group_key)

                if delete_physically:
                    # Physically delete entire group data
                    self._delete_group_physically(cluster_name, group_name, group_role)
                    logger.info(
                        f"Physically deleted all data for {group_role} group '{group_name}' in cluster '{cluster_name}'"
                    )
                else:
                    # Just mark all nodes in the group as offline
                    update_query = """
                    UPDATE `nodes` SET is_online = FALSE 
                    WHERE cluster_name = %s AND group_name = %s AND group_role = %s
                    """
                    self._execute_update(
                        update_query, (cluster_name, group_name, group_role)
                    )
                    logger.info(
                        f"Marked all nodes in {group_role} group '{group_name}' as offline"
                    )

            return affected_clusters
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to cleanup inactive groups: {e}")
            return {}

    def get_cluster_stats(self, cluster_name: str) -> Dict:
        """Get cluster statistics."""
        try:
            stats = {}

            # Node counts by role
            node_stats = self._execute_query(
                """
                SELECT group_role, is_online, COUNT(*) as count 
                FROM `nodes` 
                WHERE cluster_name = %s 
                GROUP BY group_role, is_online
            """,
                (cluster_name,),
            )

            stats["nodes"] = {}
            for row in node_stats:
                role = row["group_role"]
                if role not in stats["nodes"]:
                    stats["nodes"][role] = {"online": 0, "offline": 0}

                if row["is_online"]:
                    stats["nodes"][role]["online"] = row["count"]
                else:
                    stats["nodes"][role]["offline"] = row["count"]

            # Group counts
            group_stats = self._execute_query(
                """
                SELECT group_role, is_ready, COUNT(*) as count 
                FROM `groups` 
                WHERE cluster_name = %s 
                GROUP BY group_role, is_ready
            """,
                (cluster_name,),
            )

            stats["groups"] = {}
            for row in group_stats:
                role = row["group_role"]
                if role not in stats["groups"]:
                    stats["groups"][role] = {"ready": 0, "not_ready": 0}

                if row["is_ready"]:
                    stats["groups"][role]["ready"] = row["count"]
                else:
                    stats["groups"][role]["not_ready"] = row["count"]

            return stats
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Failed to get cluster stats for {cluster_name}: {e}")
            return {}

    def _delete_group_physically(
        self, cluster_name: str, group_name: str, group_role: str
    ) -> None:
        """Physically delete all data for a specific group.

        Deletes in correct order to handle foreign key constraints:
        1. node_map (references nodes.node_id)
        2. comm_groups (references cluster and may reference groups)
        3. groups (may reference cluster)
        4. nodes (references cluster)

        Args:
            cluster_name: Name of the cluster
            group_name: Name of the group
            group_role: Role of the group ('prefill' or 'decode')
        """
        try:
            logger.debug(
                f"Physically deleting all data for {group_role} group '{group_name}' in cluster '{cluster_name}'"
            )

            # Query nodes to be deleted for logging
            nodes_query = """
            SELECT node_id, node_rank, hostname 
            FROM `nodes` 
            WHERE cluster_name = %s AND group_name = %s AND group_role = %s
            """
            nodes_to_delete = self._execute_query(
                nodes_query, (cluster_name, group_name, group_role)
            )

            if nodes_to_delete:
                logger.debug(
                    f"Will delete {len(nodes_to_delete)} nodes from group '{group_name}':"
                )
                for node_row in nodes_to_delete:
                    logger.debug(
                        f"  - {node_row['node_id']} (rank {node_row['node_rank']}, hostname: {node_row['hostname']})"
                    )

            # 1. Delete from node_map for all nodes in this group
            delete_node_map_query = """
            DELETE nm FROM `node_map` nm 
            INNER JOIN `nodes` n ON nm.node_id = n.node_id
            WHERE n.cluster_name = %s AND n.group_name = %s AND n.group_role = %s
            """
            deleted_maps = self._execute_update(
                delete_node_map_query, (cluster_name, group_name, group_role)
            )
            logger.debug(
                f"Deleted {deleted_maps} node_map entries for group {group_name}"
            )

            # 2. Delete from comm_groups that involve this group
            if group_role == "prefill":
                # Delete comm_groups where this is the prefill group
                delete_comm_query = "DELETE FROM `comm_groups` WHERE cluster_name = %s AND prefill_group = %s"
            else:
                # Delete comm_groups where this is the decode group
                delete_comm_query = "DELETE FROM `comm_groups` WHERE cluster_name = %s AND decode_group = %s"

            deleted_comms = self._execute_update(
                delete_comm_query, (cluster_name, group_name)
            )
            logger.debug(
                f"Deleted {deleted_comms} comm_group entries for group {group_name}"
            )

            # 3. Delete from groups table
            delete_group_query = """
            DELETE FROM `groups` 
            WHERE cluster_name = %s AND group_name = %s AND group_role = %s
            """
            deleted_groups = self._execute_update(
                delete_group_query, (cluster_name, group_name, group_role)
            )
            logger.debug(
                f"Deleted {deleted_groups} group entries for group {group_name}"
            )

            # 4. Finally delete from nodes table
            delete_nodes_query = """
            DELETE FROM `nodes` 
            WHERE cluster_name = %s AND group_name = %s AND group_role = %s
            """
            deleted_nodes = self._execute_update(
                delete_nodes_query, (cluster_name, group_name, group_role)
            )
            logger.debug(f"Deleted {deleted_nodes} node entries for group {group_name}")

            logger.info(
                f"Successfully deleted all data for {group_role} group '{group_name}' in cluster "
                f"'{cluster_name}' (nodes: {deleted_nodes}, groups: {deleted_groups}, "
                f"comm_groups: {deleted_comms}, node_maps: {deleted_maps})"
            )

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Error during physical group deletion: {e}")
            # Re-raise to let caller handle
            raise

    def _delete_comm_groups_for_prefill_group(
        self, cluster_name: str, prefill_group_name: str
    ) -> None:
        """Delete all comm_groups that involve a specific prefill group.

        This is called when a prefill master node (rank 0) is deleted,
        as it invalidates all communication groups involving this prefill group.

        Args:
            cluster_name: Name of the cluster
            prefill_group_name: Name of the prefill group
        """
        try:
            # First, query which comm_groups will be deleted for logging
            query_comm_groups = """
            SELECT comm_key, decode_group, comm_id
            FROM `comm_groups` 
            WHERE cluster_name = %s AND prefill_group = %s
            """

            comm_groups_to_delete = self._execute_query(
                query_comm_groups, (cluster_name, prefill_group_name)
            )

            if comm_groups_to_delete:
                logger.debug(
                    f"Will delete {len(comm_groups_to_delete)} comm_groups for "
                    f"prefill group '{prefill_group_name}':"
                )
                for comm_row in comm_groups_to_delete:
                    logger.debug(
                        f"  - {comm_row['comm_key']} (decode_group: {comm_row['decode_group']}, "
                        f" comm_id: {comm_row['comm_id']})"
                    )

            # Now delete the comm_groups
            delete_query = """
            DELETE FROM `comm_groups` 
            WHERE cluster_name = %s AND prefill_group = %s
            """

            deleted_count = self._execute_update(
                delete_query, (cluster_name, prefill_group_name)
            )
            logger.debug(
                f"Deleted {deleted_count} comm_group entries for prefill group '{prefill_group_name}'"
            )

        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                f"Error deleting comm_groups for prefill group '{prefill_group_name}': {e}"
            )
            raise

    def _group_to_dict(self, group: GroupInfo) -> Dict:
        """Convert GroupInfo to dictionary for JSON serialization."""
        return {
            "group_id": group.group_id,
            "group_name": group.group_name,
            "group_role": group.group_role,
            "cluster_name": group.cluster_name,
            "created_at": group.created_at.isoformat(),
            "last_updated": group.last_updated.isoformat(),
            "is_ready": group.is_ready,
            "world_size": group.world_size,
            "nodes": {k: self._node_to_dict(v) for k, v in group.nodes.items()},
        }

    def _node_to_dict(self, node: NodeInfo) -> Dict:
        """Convert NodeInfo to dictionary for JSON serialization."""
        return node.to_dict()

    def _comm_group_to_dict(self, comm_group: CommGroupPair) -> Dict:
        """Convert CommGroupPair to dictionary for JSON serialization."""
        return {
            "prefill_group": comm_group.prefill_group,
            "decode_group": comm_group.decode_group,
            "comm_id": comm_group.comm_id,
            "created_at": comm_group.created_at.isoformat(),
            "last_active": comm_group.last_active.isoformat(),
        }

    def close(self):
        """Close any remaining connections."""
        # In this refactored version, we don't maintain persistent connections
        pass

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()

    def _delete_nodes_physically(self, node_ids: List[str]) -> None:
        """Physically delete nodes and all their related data.

        Deletes in correct order to handle foreign key constraints:
        1. node_map (references nodes.node_id)
        2. groups (may reference nodes)
        3. nodes (main table)

        Args:
            node_ids: List of node IDs to delete
        """
        if not node_ids:
            return

        placeholders = ",".join(["%s"] * len(node_ids))
        node_ids_tuple = tuple(node_ids)

        try:
            logger.debug(
                f"Physically deleting {len(node_ids)} nodes: {', '.join(node_ids)}"
            )

            # Delete in order to respect foreign key constraints

            # 1. Delete from node_map first (child table)
            delete_node_map_query = (
                f"DELETE FROM `node_map` WHERE node_id IN ({placeholders})"
            )
            deleted_maps = self._execute_update(delete_node_map_query, node_ids_tuple)
            logger.debug(f"Deleted {deleted_maps} node_map entries")

            # 2. Delete from groups where nodes might be referenced
            # Note: Adjust this based on your actual schema - if groups table has node_id column, this is expected
            delete_groups_query = (
                f"DELETE FROM `groups` WHERE node_id IN ({placeholders})"
            )
            try:
                deleted_groups = self._execute_update(
                    delete_groups_query, node_ids_tuple
                )
                logger.debug(f"Deleted {deleted_groups} group entries")
            except Exception as e:  # pylint: disable=broad-except
                # If groups table doesn't have node_id column, this is expected
                logger.debug(
                    f"Groups table deletion skipped (likely no node_id column): {e}"
                )

            # 3. Finally delete from nodes table (parent table)
            delete_nodes_query = (
                f"DELETE FROM `nodes` WHERE node_id IN ({placeholders})"
            )
            deleted_nodes = self._execute_update(delete_nodes_query, node_ids_tuple)
            logger.debug(f"Deleted {deleted_nodes} node entries")

            logger.info(
                f"Successfully deleted {deleted_nodes} nodes and their related data"
            )

        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Error during physical node deletion: {e}")
            # Re-raise to let caller handle
            raise

    def _log_database_stats(self, stage: str) -> None:
        """Log database statistics for debugging cleanup operations.

        Args:
            stage: Description of when this logging occurs (e.g., "before cleanup", "after cleanup")
        """
        try:
            # Get table counts
            tables_stats = {}

            # Nodes statistics
            nodes_stats = self._execute_query(
                """
                SELECT 
                    COUNT(*) as total_nodes,
                    SUM(CASE WHEN is_online = TRUE THEN 1 ELSE 0 END) as online_nodes,
                    SUM(CASE WHEN is_online = FALSE THEN 1 ELSE 0 END) as offline_nodes,
                    SUM(CASE WHEN group_role = 'prefill' THEN 1 ELSE 0 END) as prefill_nodes,
                    SUM(CASE WHEN group_role = 'decode' THEN 1 ELSE 0 END) as decode_nodes
                FROM `nodes`
            """
            )

            if nodes_stats:
                stats = nodes_stats[0]
                tables_stats["nodes"] = (
                    f"{stats['total_nodes']} total ({stats['online_nodes']} online, "
                    f"{stats['offline_nodes']} offline, {stats['prefill_nodes']} prefill, "
                    f"{stats['decode_nodes']} decode)"
                )

            # Groups statistics
            groups_stats = self._execute_query(
                """
                SELECT 
                    COUNT(*) as total_groups,
                    SUM(CASE WHEN is_ready = TRUE THEN 1 ELSE 0 END) as ready_groups,
                    SUM(CASE WHEN group_role = 'prefill' THEN 1 ELSE 0 END) as prefill_groups,
                    SUM(CASE WHEN group_role = 'decode' THEN 1 ELSE 0 END) as decode_groups
                FROM `groups`
            """
            )

            if groups_stats:
                stats = groups_stats[0]
                tables_stats["groups"] = (
                    f"{stats['total_groups']} total ({stats['ready_groups']} ready, "
                    f"{stats['prefill_groups']} prefill, {stats['decode_groups']} decode)"
                )

            # Comm groups statistics
            comm_stats = self._execute_query(
                "SELECT COUNT(*) as total_comm_groups FROM `comm_groups`"
            )
            if comm_stats:
                tables_stats["comm_groups"] = (
                    f"{comm_stats[0]['total_comm_groups']} total"
                )

            # Node map statistics
            node_map_stats = self._execute_query(
                "SELECT COUNT(*) as total_node_maps FROM `node_map`"
            )
            if node_map_stats:
                tables_stats["node_map"] = (
                    f"{node_map_stats[0]['total_node_maps']} total"
                )

            # Clusters statistics
            cluster_stats = self._execute_query(
                """
                SELECT 
                    COUNT(*) as total_clusters,
                    SUM(CASE WHEN is_active = TRUE THEN 1 ELSE 0 END) as active_clusters
                FROM `clusters`
            """
            )

            if cluster_stats:
                stats = cluster_stats[0]
                tables_stats["clusters"] = (
                    f"{stats['total_clusters']} total ({stats['active_clusters']} active)"
                )

            # Log all statistics
            logger.info(f"Database statistics {stage}:")
            for table_name, stats_info in tables_stats.items():
                logger.info(f"  - {table_name}: {stats_info}")

        except Exception as e:  # pylint: disable=broad-except
            logger.warning(f"Failed to log database statistics {stage}: {e}")
