# Simple Router Overview

This package exposes a FastAPI application that coordinates prefill/decode inference nodes for KsanaLLM deployments. It tracks node liveness, stores communication metadata, and proxies user traffic to the selected services.

## Architecture Overview

The system consists of three main components:

### 1. Router (this service)
A stateless FastAPI application that:
- Receives and validates requests from clients
- Selects available prefill/decode node pairs from the registry
- Forwards requests to both prefill and decode nodes simultaneously
- Merges streaming responses and returns combined tokens to clients
- Manages node registration and health tracking via database

### 2. Prefill Nodes
Inference servers responsible for:
- Processing input prompts and generating KV cache
- Producing the first output token
- Registering with router via `/RegisterNode` endpoint
- Sending periodic heartbeats to maintain online status
- Rank-0 prefill nodes register communication IDs via `/RegisterCommId`

### 3. Decode Nodes
Inference servers responsible for:
- Receiving KV cache from paired prefill nodes
- Generating subsequent tokens in an autoregressive manner
- Registering with router and maintaining heartbeat status
- Streaming all generated tokens back to router

### Cluster Topology

A typical cluster consists of N prefill nodes and M decode nodes (often N = M). All nodes within a cluster are fully meshed - any prefill node can communicate with any decode node to form a dynamic processing pair. 

The router maintains this flexibility by:
- Tracking all online nodes in the `node_info` table
- Creating `comm_group_pair` entries for each valid `prefill__decode` combination
- Selecting an available pair per request based on node health and availability

**Example cluster with 3 prefill + 3 decode nodes:**
```
Prefill-0, Prefill-1, Prefill-2
   ×           ×           ×
Decode-0,  Decode-1,  Decode-2
```

Each request can use any combination (e.g., Prefill-1 + Decode-2), providing load balancing and fault tolerance across the cluster.

### Communication Flow

```
Client → Router → [Prefill Node + Decode Node] → Router → Client
                     ↓            ↓
                     └─── KV Cache ────┘
```

The router assigns a unique communication ID (`comm_id`) to each request and sends it via custom headers (`kv-comm-group-key`, `kv-comm-request-id`) to both nodes, enabling them to coordinate KV cache transfer.

## Key Components

- **main.py**: Configures logging, initializes the database schema, and instantiates the FastAPI app with router endpoints.
- **config.py**: Loads settings from `config.ini` (cluster name, database backend, name-service provider, logging level, heartbeat timeout). Uses lazy initialization pattern with `get_settings()`.
- **database.py**: Builds the SQLAlchemy engine/session registry and creates tables on startup. SQLite is the default, MySQL is optional.
- **models.py**: SQLAlchemy ORM definitions for `node_info`, `comm_group_pair`, and `inference_group_status` tables.
- **services.py**: Core business logic for node registration, heartbeat processing, comm group management, and readiness calculations.
- **generate.py**: Merges streaming responses from chosen prefill and decode endpoints and exposes generic proxy handlers for `/generate`, `/v1/*`, and `/v2/*`.
- **name_service/**: Pluggable discovery backends. `auto_provider` queries the local database; `polaris_provider` integrates with Tencent Polaris (optional).

## Configuration

Settings are resolved from `config.ini` located beside the package. Key sections:

### `[general]`
- `log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `log_dir`: Directory for log files (default: `./`)
- `cluster_name`: Identifier for this cluster
- `heartbeat_timeout_seconds`: Seconds before marking nodes offline

### `[database]`
- `storage_mode`: Choose `sqlite` or `mysql`
- `sqlite_path`: Path to SQLite database file (supports `~` expansion)
- MySQL connection parameters: `host`, `port`, `user`, `password`, `database`, `charset`, `autocommit`

### `[name_service]`
- `name_service_provider`: Dotted path to the provider module
- `namespace`, `prefill_service`, `decode_service`: Service discovery parameters

**Note on Database Setup:**
- **Development/Testing**: Tables are auto-created via SQLAlchemy's `Base.metadata.create_all()` on startup.
- **Production**: You may use the provided `create_table.sql` if your deployment requires manual schema management or the application account lacks `CREATE TABLE` privileges.

## Database Schema

- **node_info**: Records registered nodes, their heartbeat state, and device counts.
- **comm_group_pair**: Stores control/data channel metadata keyed by the `prefill__decode` pair.
- **inference_group_status**: Aggregates readiness for each inference address.

## Service Endpoints

### Node Management
- `POST /RegisterNode`: Nodes register (or refresh) their presence. Requires cluster name, inference/coordinator addresses, role, rank, world size, and device inventory.
- `POST /Heartbeat`: Updates `last_heartbeat`, recomputes readiness, and returns communication metadata relevant to the caller.
- `POST /RegisterCommId`: Only prefill rank-0 nodes may bind a `comm_key` to a specific `comm_id`. The control metadata is rebuilt automatically.

### Request Proxying
- `/generate`, `/v1/*`, `/v2/*`: Reverse proxies that multiplex traffic to the chosen prefill/decode pair and stream tokens back to the client. Communication IDs are generated per request and sent via `kv-comm-*` headers.

## Name Service Selection

- **auto_provider** (default): Directly inspects active `CommGroupPair` rows and picks online rank-0 nodes from the database.
- **polaris_provider** (optional): Relies on the Tencent Polaris SDK; when configured, it tracks call success/failure to feed health data back to Polaris.

## Running the Service

1. **Install dependencies**:
   ```bash
   pip install fastapi sqlalchemy httpx uvicorn pymysql  # add pymysql for MySQL
   ```

2. **Review configuration**:
   Edit `config.ini` and update the database/name service sections.

3. **Launch with uvicorn**:
   ```bash
   python ./src/ksana_llm/python/simple_router/main.py --host 0.0.0.0 --port 9080 --workers 8 --config /tmp/config.ini
   ```

4. **Node registration workflow**:
   - Nodes call `/RegisterNode` on startup and send periodic `/Heartbeat` requests
   - Prefill rank-0 nodes register communication IDs via `/RegisterCommId` once decode partners are ready

## Operational Notes

- **Heartbeat timeout**: Heartbeats older than `heartbeat_timeout_seconds` mark nodes offline and clear related comm metadata.
- **Stale data cleanup**: Prefill rank-0 registration resets stale communication rows to avoid serving outdated data.
- **Streaming protocol**: Response merges the first prefill token with ongoing decode tokens and appends a `[DONE]\0` marker when both streams finish.
- **Logging**: Logs are written to `{log_dir}/simple_router.log` and console. Configure `log_level` and `log_dir` in `config.ini`.

## Testing

Unit tests can be added under `tests/`. The service logic is structured so that service functions in `services.py` can be exercised with an in-memory SQLite backend using `get_session_factory` from `database.py`.
