# KsanaLLM Router API Documentation

This document describes the API interfaces for KsanaLLM Router, including cluster management, node registration, and heartbeat mechanism.

## System Design Documentation

### 1. Architecture Overview

KsanaLLM Router is designed to coordinate distributed LLM inference workloads by managing the communication between multiple compute nodes. The system follows a two-phase inference architecture:

- **Prefill Phase**: Initial prompt processing and generation of the first token
- **Decode Phase**: Generation of subsequent tokens

The router coordinates groups of nodes specialized for each phase and manages the communication between them for efficient inference.

#### Key Components

1. **Cluster Management System**: Organizes compute resources into logical clusters
2. **Group Management**: Manages prefill and decode groups within clusters
3. **Node Registry**: Tracks individual compute nodes, their capabilities, and online status
4. **Storage Backend**: Persists system state with pluggable storage options
5. **Request Router**: Routes inference requests to appropriate nodes
6. **Heartbeat Mechanism**: Monitors node health and availability

### 2. Data Model

The system uses a hierarchical data model:

- **Cluster**: The top-level organizational unit containing multiple groups
  - Contains prefill groups, decode groups, and communication group pairs
  - Each cluster has a unique name and tracks its active state

- **Group**: A collection of nodes with the same role (prefill or decode)
  - Has a unique name within the cluster
  - Has a specific role (prefill/decode)
  - Contains multiple nodes
  - Tracks its ready state based on constituent nodes

- **Node**: An individual compute resource (server)
  - Identified by a UUID
  - Has a rank within its group
  - Contains information about hardware (device type, count)
  - Tracks its online status via heartbeats

- **Communication Group Pair**: Links a prefill group with a decode group
  - Contains a unique Communication ID for inter-group communication
  - Tracks its last active timestamp

#### 2.1 Detailed Data Structure Diagrams

##### Class Diagram

```
┌─────────────────────────┐
│      ClusterInfo        │
├─────────────────────────┤
│ cluster_name: str       │
│ prefill_groups: Dict    │◄────┐
│ decode_groups: Dict     │◄───┐│
│ comm_groups: Dict       │◄─┐ ││
│ created_at: datetime    │  │ ││
│ last_updated: datetime  │  │ ││
│ is_active: bool         │  │ ││
└─────────────────────────┘  │ ││
                             │ ││
┌─────────────────────────┐  │ ││
│     CommGroupPair       │  │ ││
├─────────────────────────┤  │ ││
│ prefill_group: str      │  │ ││
│ decode_group: str       │  │ ││
│ comm_id: str            │◄─┘ ││
│ created_at: datetime    │    ││
│ last_active: datetime   │    ││
└─────────────────────────┘    ││
                               ││
┌─────────────────────────┐    ││
│       GroupInfo         │    ││
├─────────────────────────┤    ││
│ group_id: str           │    ││
│ group_name: str         │    ││
│ group_role: str         │    ││
│ cluster_name: str       │    ││
│ nodes: Dict             │◄─┐ ││
│ created_at: datetime    │  │ ││
│ last_updated: datetime  │  │ ││
│ is_ready: bool          │◄──┘││
│ world_size: int         │    ││
└─────────────────────────┘    ││
                               ││
┌─────────────────────────┐    ││
│       NodeInfo          │    ││
├─────────────────────────┤    ││
│ node_id: str            │    ││
│ hostname: str           │    ││
│ inference_addr: str       │    ││
│ cluster_name: str       │    ││
│ group_name: str         │    ││
│ group_role: str         │    ││
│ node_rank: int          │◄───┘│
│ devices: List[DeviceInfo] │     │
│ last_heartbeat: datetime│     │
│ is_online: bool         │     │
│ comm_id: str            │     │
│ job_id: str             │     │
│ start_time: str         │     │
└─────────────────────────┘     │
                                │
┌─────────────────────────┐     │
│       GroupInfo         │     │
├─────────────────────────┤     │
│ //...existing fields... │◄────┘
└─────────────────────────┘
```

##### Object Relationship Diagram

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   Cluster     │     │    Group      │     │     Node      │
│  (default)    │1   *│  (prefill/    │1   *│  (compute     │
│               │────►│   decode)     │────►│   resource)   │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        │                     │                     │
        │                     │                     │
        │                     │                     │
        │               ┌─────▼─────┐               │
        │               │ GroupPair │               │
        └──────────────►│(prefill+  │◄──────────────┘
                        │ decode)   │
                        └───────────┘
```

### 3. Storage System

The system supports multiple storage backends through an abstraction layer:

- **Memory Storage**: In-memory storage for development and testing
- **Db Storage**: Distributed key-value storage for production environments

The storage layer is responsible for:
- Persisting cluster, group, and node information
- Managing node mappings for efficient lookups
- Supporting cluster operations (create, update, delete)

#### 3.1 Storage Layer Abstraction

```
┌─────────────────────────────┐
│    Storage Interface        │
├─────────────────────────────┤
│ save_cluster()              │
│ get_cluster()               │
│ delete_cluster()            │
│ list_clusters()             │
│ save_node_map()             │
│ get_node_map()              │
│ delete_node_map()           │
└─────────────────────────────┘
          ▲
          │
          │
 ┌────────┴───────┐
 │                │
┌┴────────────┐  ┌┴────────────┐
│MemoryStorage│  │ DBStorage │
└─────────────┘  └─────────────┘
```

### 4. Request Routing

The router implements a split-inference pattern:

1. **Producer-Consumer Model**:
   - Producer nodes (prefill) handle the initial token generation
   - Consumer nodes (decode) handle subsequent tokens
   - The router merges streaming responses from both

2. **Stream Processing**:
   - Supports asynchronous streaming for real-time token delivery
   - Implements proper delimiting and end-of-stream signaling

#### 4.1 Request Flow Diagram

```
┌───────────┐     ┌───────────────────┐     ┌───────────────┐
│  Client   │     │   Router Service  │     │ Prefill Group │
│           │────►│                   │────►│               │
└───────────┘     └───────────────────┘     └───────────────┘
      ▲                     │                       │
      │                     │                       │
      │                     ▼                       ▼
      │             ┌───────────────┐      ┌───────────────┐
      └─────────────┤ Response      │◄─────┤ Decode Group  │
                    │ Aggregator    │      │               │
                    └───────────────┘      └───────────────┘
```

### 5. Node Management

Nodes register with the system by providing:
- Host address
- Group name and role
- Node rank
- Device information
- Job identification

The system maintains node state through:
- Regular heartbeat checks
- Timeout detection for offline nodes
- Group readiness updates based on node availability

#### 5.1 Node Lifecycle Diagram

```
┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
│           │     │           │     │           │     │           │
│Registration│────►│ Active    │────►│ Inactive  │────►│ Removed   │
│           │     │           │     │           │     │           │
└───────────┘     └───────────┘     └───────────┘     └───────────┘
      ▲                 │
      │                 │
      └─────────────────┘
         Re-register
```

### 6. Communication Coordination

The system coordinates inter-group communication by:
- Managing Communication IDs for each prefill-decode group pair
- Distributing communication group information via heartbeat responses
- Supporting registration of communication IDs by prefill master nodes

#### 6.1 Communication Setup Flow

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Prefill       │     │   Router      │     │ Decode        │
│ Master Node   │     │   Service     │     │ Nodes         │
├───────────────┤     ├───────────────┤     ├───────────────┤
│ Generate      │     │               │     │               │
│ Comm ID       │────►│ Store Comm ID │     │               │
│               │     │               │────►│ Get Comm ID   │
│               │     │               │     │ via Heartbeat │
│ Connect to    │◄───────────────────────►│ Connect to    │
│ Decode Nodes  │     │               │     │ Prefill Nodes │
└───────────────┘     └───────────────┘     └───────────────┘
```

### 7. System Operation

The router operates through:
1. **Initialization**: Creates default cluster on startup
2. **Node Registration**: Accepts node registrations with group assignments
3. **Heartbeat Mechanism**: Tracks node availability
4. **Cleanup Process**: Removes timed-out nodes and inactive communication groups
5. **Request Routing**: Directs inference requests to appropriate nodes
6. **Response Streaming**: Merges responses from producer and consumer nodes

#### 7.1 Data Flow Diagram

```
┌────────────┐      ┌────────────┐      ┌────────────┐
│            │      │            │      │            │
│  Client    │ (1)  │  Router    │ (2)  │  Prefill   │
│ Application│─────►│  Service   │─────►│   Group    │
│            │      │            │      │            │
└────────────┘      └────────────┘      └────────────┘
      ▲                   │                   │
      │                   │                   │
      │                   │                   │
      │                   │(3)                │(4)
      │                   ▼                   ▼
      │(6)         ┌────────────┐     ┌────────────┐
      └────────────┤  Response  │◄────┤   Decode   │
                   │ Aggregator │(5)  │   Group    │
                   └────────────┘     └────────────┘

(1) Client sends generation request
(2) Router forwards to prefill group for initial processing
(3) Router sets up streaming connection with response aggregator
(4) Prefill group passes KV cache to decode group
(5) Decode group sends tokens to response aggregator
(6) Response aggregator streams tokens to client
```

## API Endpoints

### Cluster Management

#### List Cluster Information

```bash
# Request
curl -X GET http://localhost:9080/api/v1/cluster-info/

# Response
[
  {
    "cluster_name": "default-cluster",
    "prefill_groups": 2,
    "decode_groups": 4,
    "group_info": [
      {"group_name": "prefill_group_0", "group_role":"prefill", "group_ready":"true"},
      {"group_name": "prefill_group_1", "group_role":"prefill", "group_ready":"true"},
      {"group_name": "decode_group_0", "group_role":"decode", "group_ready":"true"},
      {"group_name": "decode_group_1", "group_role":"decode", "group_ready":"true"},
      {"group_name": "decode_group_2", "group_role":"decode", "group_ready":"true"},
      {"group_name": "decode_group_3", "group_role":"decode", "group_ready":"true"},
    ]
  }
]
```

## Node Management

### Register Node
#### Prefill Node Registration Example, registering a group of pp = 2 Prefill services, note the node_rank values.

```bash
# node1 request
curl -X POST http://localhost:9080/api/v1/nodes/ \
  -H "Content-Type: application/json" \
  -d '{
    "inference_addr": "192.168.0.1:8088"
    "coordinator_port": 13579,
    "group_name": "prefill_group_0",
    "group_role": "prefill",
    "node_rank": 0,
    "devices": [
        {"device_id":0,"device_type": "NVIDIA L20","device_ip":"8.8.8.8"},
        {"device_id":1, "device_type": "NVIDIA L20", "device_ip":"8.8.8.9"}
    ],
    "start_time": "2025-04-06 14:58:58",
    "job_id": "daddecc0-a028-41dd-b0a1-7302b28a9c3b",
  }'

# Response
{
  "node_id": "e39920b3-46a1-43d5-8fef-f458823dc3de",
  "is_online": true,
  "last_heartbeat": "2025-04-06T15:16:47.369453",
}

```

### Get Node Information

```bash
# Request
curl -X GET http://localhost:9080/api/v1/nodes/e39920b3-46a1-43d5-8fef-f458823dc3de

# Response
{
  "node_id": "e39920b3-46a1-43d5-8fef-f458823dc3de",
  "inference_addr": "192.168.0.1:8907",
  "group_name": "prefill_group_0",
  "group_role": "prefill",
  "coordinator_port": 13579,
  "node_rank": 0,
  "devices": [
    {"device_id": 0, "device_type": "NVIDIA L20","device_ip":"8.8.8.8" },
    {"device_id": 1, "device_type": "NVIDIA L20", "device_ip":"8.8.8.9" }
    ],
  "is_online": false,
  "last_heartbeat": "2025-04-06T15:16:47.369453",
  "job_id": "daddecc0-a028-41dd-b0a1-7302b28a9c3b",
  "start_time": "2025-04-06 14:58:58"
}
```

#### Prefill Master Node Register Communication_id

```bash
# Request
curl -X POST http://localhost:9080/api/v1/nodes/registerCommId \
  -H "Content-Type: application/json" \
  -d '{
  "node_id":"node_id",
  "comm_key":"prefill_group_0_decode_group_0",
  "comm_id":"6666666-66666-666666-6666-6666666666"
  }'

# Response
{"status":"OK", "comm_id": "6666666-66666-666666-6666-6666666666"}
```


## Heartbeat Mechanism

Nodes need to send periodic heartbeat requests to maintain online status and get cluster information through heartbeat responses.

### Heartbeat Request Example

```bash
# Request
curl -X POST http://localhost:9080/api/v1/nodes/heartbeat \
  -H "Content-Type: application/json" \
  -d '{
    "node_id": "f3a9c1d2-b6e5-48c7-9d1a-e7f2c9b8d3a5"
  }'
```

#### Prefill Node Heartbeat Response

```json
{
  "node_id":"6f23ea1c-9d87-4f7e-a2c8-45d5f724f2e3",
  "node_name":"prefill_group_0_node_rank_0",
  "group_name":"prefill_group_0",
  "is_online":true,
  "group_ready":true,
  "node_role":"prefill",
  "node_rank":0,
  "timestamp":"2025-04-14T14:56:30.677325",
  "comm_group_to_address":{
    "prefill_group_0__decode_group_0":[
      ("node_rank", "devid","device_ip:coodinator_port"),
      (0,1,"4.4.4.4:14579"),
      (1,0,"1.1.1.1:13579"),
      (1,1,"2.2.2.2:13579")
      ],
    "prefill_group_0__decode_group_1":[
      ("node_rank", "devid","device_ip:coodinator_port"),
      (0,1,"4.4.4.4:14579"),
      (1,0,"1.1.1.1:13579"),
      (1,1,"2.2.2.2:13579")
      ]
  },
  "comm_group_to_id":{
    "prefill_group_0__decode_group_0":"6666666-66666-666666-6666-6666666666","prefill_group_0__decode_group_1":""
  }
}
```


#### Decode Node Heartbeat Response

```json
{
  "node_id": "f3a9c1d2-b6e5-48c7-9d1a-e7f2c9b8d3a5",
  "node_name": "decode_group1_node_rank_0",
  "is_online": true,
  "group_ready": true,
  "group_name": "decode_group_1",
  "node_role": "decode",
  "timestamp": "2025-04-06T15:25:43.504735",
  "prefill_groups": {
    "prefill_group_1": {"node_rank_0": "10.0.0.1:12306", "node_rank_1": "10.0.0.2:12306"},
    "prefill_group_2": {"node_rank_0": "10.0.0.3:12306", "node_rank_1": "10.0.0.3:12307"}
  },
  "decode_groups": {
    "decode_group_1": {"node_rank_0": "10.0.0.6:12306", "node_rank_1": "10.0.0.7:12307"}
  },
  "comm_groups": {
    "prefill_group_1_decode_group_1": "base64(comm_id_1)",
    "prefill_group_2_decode_group_1": "base64(comm_id_3)"
  }
}
```
