# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

from typing import Dict, List, Optional, Any, Literal, Tuple
from datetime import datetime
import uuid
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Request and response models
class DeviceInfoRequest(BaseModel):
    """Device information request.

    Attributes:
        local_rank: Local rank of the device.
        device_id: Device ID.
        device_type: Device type (e.g., "NVIDIA L20").
        device_ip: Device IP address.
    """

    device_id: int
    device_type: Optional[str] = None
    device_ip: str = None


class RegisterNodeRequest(BaseModel):
    """Register node request model.

    Attributes:
        inference_addr: Inference address in the format "host:port".
        cluster_name: Name of the cluster (optional).
        group_name: Name of the group.
        group_role: Role of the group ("prefill" or "decode").
        node_rank: Rank of the node within the group.
        hostname: Hostname (optional).
        coordinator_port: Coordinator port (optional).
        devices: List of device information (optional).
        job_id: Job ID (optional).
        start_time: Start time (optional).
        comm_id: Communication ID (optional).
        world_size: Total number of processes across all nodes (optional).
    """

    inference_addr: str
    cluster_name: Optional[str] = None
    group_name: str
    group_role: str  # "prefill" or "decode"
    node_rank: int
    hostname: Optional[str] = None
    coordinator_port: Optional[int] = None
    world_size: int = None
    devices: List[DeviceInfoRequest] = None
    job_id: Optional[str] = None
    start_time: Optional[str] = None
    comm_id: Optional[str] = None
    world_size: Optional[int] = None


class NodeResponse(BaseModel):
    """Node information response model.

    Attributes:
        node_id: ID of the node.
        hostname: Hostname.
        inference_addr: Model Inference address.
        cluster_name: Name of the cluster.
        group_name: Name of the group.
        group_role: Role of the group.
        node_rank: Rank of the node within the group.
        devices: List of device information.
        is_online: Whether the node is online.
        last_heartbeat: Time of the last heartbeat.
        job_id: Job ID (optional).
        start_time: Start time (optional).
        world_size: Total number of processes across all nodes (optional).
    """

    node_id: str
    hostname: str
    inference_addr: str
    cluster_name: str
    group_name: str
    group_role: str
    node_rank: int
    world_size: int = None
    devices: List[DeviceInfoRequest] = None
    is_online: bool
    last_heartbeat: datetime
    job_id: Optional[str] = None
    start_time: Optional[str] = None
    world_size: Optional[int] = None


class SimpleNodeResponse(BaseModel):
    """Simplified node response model.

    Attributes:
        node_id: ID of the node.
        is_online: Whether the node is online.
        last_heartbeat: Time of the last heartbeat.
    """

    node_id: str
    is_online: bool
    last_heartbeat: datetime


class RankAssignment(BaseModel):
    """Rank assignment model.

    Attributes:
        local_rank: Local rank.
        global_rank: Global rank.
    """

    local_rank: int
    global_rank: int


class RankAssignmentData(BaseModel):
    """Complete rank assignment data model.

    Attributes:
        assignments: List of rank assignments.
        world_size: Size of the world (optional).
    """

    assignments: List[RankAssignment]
    world_size: Optional[int] = None


class HeartbeatRequest(BaseModel):
    """Heartbeat request model.

    Attributes:
        node_id: ID of the node.
    """

    node_id: str


class HeartbeatResponse(BaseModel):
    """Heartbeat response model.

    Attributes:
        node_id: ID of the node.
        node_name: Name of the node.
        is_online: Whether the node is online.
        group_ready: Whether the group is ready.
        node_role: Role of the node.
        timestamp: Current timestamp.
        prefill_groups: Information about prefill groups.
        decode_groups: Information about decode groups.
        comm_groups: Information about communication groups.
    """

    node_id: str
    node_name: str
    is_online: bool
    group_ready: bool
    coordinator_port: int
    node_role: str
    timestamp: datetime = datetime.now()
    comm_group_to_address: Dict[str, List[Tuple[int, int, str]]] = {}
    comm_group_to_id: Dict[str, str] = {}


class RankAssignmentResponse(BaseModel):
    """Rank assignment response model.

    Attributes:
        node_id: ID of the node.
        assignments: List of rank assignments.
        world_size: Size of the world (optional).
    """

    node_id: str
    assignments: List[RankAssignment]
    world_size: Optional[int] = None


class RegisterCommIDRequest(BaseModel):
    """Register Communication ID request model.

    Attributes:
        node_id: ID of the node.
        comm_key: Communication key.
        comm_id: Communication ID.
    """

    node_id: str
    comm_key: str
    comm_id: str


class NodeInfo(BaseModel):
    """Physical compute node information"""

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hostname: str
    inference_addr: str
    coordinator_port: int
    cluster_name: str = Field(default="default_cluster")
    group_name: str
    group_role: str
    node_rank: int  # Changed from node_idx to node_rank
    world_size: int
    devices: List[
        DeviceInfoRequest
    ]  # Changed from gpus to device, can be string or list
    last_heartbeat: datetime = Field(default_factory=datetime.now)
    is_online: bool = True
    comm_id: Optional[str] = None
    job_id: Optional[str] = None
    start_time: Optional[str] = None

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "inference_addr": self.inference_addr,
            "coordinator_port": self.coordinator_port,
            "world_size": self.world_size,
            "cluster_name": self.cluster_name,
            "group_name": self.group_name,
            "group_role": self.group_role,
            "node_rank": self.node_rank,
            "devices": self.devices,
            "is_online": self.is_online,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "comm_id": self.comm_id,
            "job_id": self.job_id,
            "start_time": self.start_time,
        }


class CommGroupPair(BaseModel):
    """Communication group pair (prefill-decode)"""

    prefill_group: str
    decode_group: str
    comm_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=lambda: datetime.now())

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class GroupInfo(BaseModel):
    """Group information (prefill/decode group)"""

    group_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    group_name: str
    group_role: Literal["prefill", "decode"]
    cluster_name: str
    nodes: Dict[str, NodeInfo] = Field(default_factory=dict)  # node_id -> NodeInfo
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=lambda: datetime.now())
    is_ready: bool = False
    world_size: Optional[int] = (
        None  # Total process count, calculated from node count and devices per node
    )

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def update_ready_state(self) -> bool:
        """Update group ready state"""
        # Check if there are online nodes
        online_nodes = [n for n in self.nodes.values() if n.is_online]

        if not online_nodes:
            self.is_ready = False
            return False

        # Calculate world_size
        self.world_size = online_nodes[0].world_size

        # Group is ready
        self.is_ready = True
        self.last_updated = datetime.now()

        return True

    def add_node(self, node: NodeInfo) -> bool:
        """Add or update node"""
        if node.group_name != self.group_name:
            return False

        self.nodes[node.node_id] = node
        self.last_updated = datetime.now()
        return True

    def get_node_by_rank(self, node_rank: int) -> Optional[NodeInfo]:
        """Get node by node rank"""
        for node in self.nodes.values():
            if node.node_rank == node_rank:
                return node
        return None


class ClusterInfo(BaseModel):
    """Cluster information"""

    cluster_name: str
    prefill_groups: Dict[str, GroupInfo] = Field(
        default_factory=dict
    )  # group_name -> GroupInfo
    decode_groups: Dict[str, GroupInfo] = Field(
        default_factory=dict
    )  # group_name -> GroupInfo
    comm_groups: Dict[str, CommGroupPair] = Field(
        default_factory=dict
    )  # comm_key -> CommGroupPair
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=lambda: datetime.now())
    is_active: bool = True
    inactive_group_timeout: int = (
        300  # Default 5 minute timeout for detecting inactive groups
    )

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    def get_group(self, group_name: str, group_role: str) -> Optional[GroupInfo]:
        """Get specified group"""
        if group_role == "prefill":
            return self.prefill_groups.get(group_name)
        elif group_role == "decode":
            return self.decode_groups.get(group_name)
        return None

    def add_group(self, group: GroupInfo) -> bool:
        """Add group to cluster"""
        if group.group_role == "prefill":
            self.prefill_groups[group.group_name] = group

            # Create communication pairs for each new prefill group and existing decode group
            for decode_group_name, _ in self.decode_groups.items():
                comm_key = f"{group.group_name}__{decode_group_name}"
                if comm_key not in self.comm_groups:
                    self.comm_groups[comm_key] = CommGroupPair(
                        prefill_group=group.group_name,
                        decode_group=decode_group_name,
                        comm_id="",  # Initially set to empty string
                    )

        elif group.group_role == "decode":
            self.decode_groups[group.group_name] = group

            # Create communication pairs for each existing prefill group and new decode group
            for prefill_group_name, _ in self.prefill_groups.items():
                comm_key = f"{prefill_group_name}__{group.group_name}"
                if comm_key not in self.comm_groups:
                    self.comm_groups[comm_key] = CommGroupPair(
                        prefill_group=prefill_group_name,
                        decode_group=group.group_name,
                        comm_id="",  # Initially set to empty string
                    )

        else:
            return False

        self.last_updated = datetime.now()
        return True

    def update_group_ready_state(self, group_name: str, group_role: str) -> bool:
        """Update ready state of specific group"""
        group = self.get_group(group_name, group_role)
        if not group:
            return False

        ready_state = group.update_ready_state()

        # If the group becomes ready, update the last active time of all related communication groups
        if ready_state:
            self._update_comm_group_activity(group_name, group_role)

        return ready_state

    def _update_comm_group_activity(self, group_name: str, group_role: str) -> None:
        """Update communication group last active time"""
        now = datetime.now()

        if group_role == "prefill":
            # Update all communication groups related to this prefill group
            for _, comm_group in self.comm_groups.items():
                if comm_group.prefill_group == group_name:
                    comm_group.last_active = now
        elif group_role == "decode":
            # Update all communication groups related to this decode group
            for _, comm_group in self.comm_groups.items():
                if comm_group.decode_group == group_name:
                    comm_group.last_active = now

    def check_nodes_heartbeat(self, timeout_seconds: int) -> List[str]:
        """Check all node heartbeats, return list of timed-out node IDs"""
        now = datetime.now()
        timeout_nodes = []

        # Check prefill nodes
        for group in self.prefill_groups.values():
            for node_id, node in group.nodes.items():
                if (
                    node.is_online
                    and (now - node.last_heartbeat).total_seconds() > timeout_seconds
                ):
                    node.is_online = False
                    timeout_nodes.append(node_id)
                    group.last_updated = now

        # Check decode nodes
        for group in self.decode_groups.values():
            for node_id, node in group.nodes.items():
                if (
                    node.is_online
                    and (now - node.last_heartbeat).total_seconds() > timeout_seconds
                ):
                    node.is_online = False
                    timeout_nodes.append(node_id)
                    group.last_updated = now

        if timeout_nodes:
            self.last_updated = now

        return timeout_nodes

    def clean_inactive_comm_groups(self) -> List[str]:
        """Clean communication group pairs that have been inactive for a long time"""
        now = datetime.now()
        inactive_keys = []

        # Check activity status of all communication group pairs
        for comm_key, comm_group in list(self.comm_groups.items()):
            # Get prefill and decode groups
            prefill_group = self.prefill_groups.get(comm_group.prefill_group)
            decode_group = self.decode_groups.get(comm_group.decode_group)

            # Check if groups exist and are active
            prefill_active = prefill_group and prefill_group.is_ready
            decode_active = decode_group and decode_group.is_ready

            # If at least one of the two groups doesn't exist or is inactive, and the communication group has timed out
            if (not prefill_active or not decode_active) and (
                now - comm_group.last_active
            ).total_seconds() > self.inactive_group_timeout:
                del self.comm_groups[comm_key]
                inactive_keys.append(comm_key)
                logger.info(f"Deleted inactive communication group pair: {comm_key}")

        if inactive_keys:
            self.last_updated = now

        return inactive_keys

    def get_comm_groups(self) -> Dict[str, str]:
        """Get NCCL IDs for all communication group pairs"""
        return {k: v.comm_id for k, v in self.comm_groups.items()}
