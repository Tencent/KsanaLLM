-- KsanaLLM Router MySQL Database Schema
-- This script creates all necessary tables for the MySQL storage backend

-- Create database (optional, can be run separately)
CREATE DATABASE IF NOT EXISTS ksana_llm_router 
    CHARACTER SET utf8mb4 
    COLLATE utf8mb4_unicode_ci;

USE ksana_llm_router;

-- Create clusters table
CREATE TABLE IF NOT EXISTS clusters (
    cluster_name VARCHAR(255) PRIMARY KEY COMMENT 'Unique cluster identifier',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    is_active BOOLEAN DEFAULT TRUE COMMENT 'Whether cluster is active',
    inactive_group_timeout INT DEFAULT 300 COMMENT 'Timeout for inactive groups (seconds)',
    cluster_data JSON COMMENT 'Serialized cluster data'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Stores cluster information and metadata';

-- Create groups table
CREATE TABLE IF NOT EXISTS `groups` (
    group_id VARCHAR(255) PRIMARY KEY COMMENT 'Unique group identifier',
    group_name VARCHAR(255) NOT NULL COMMENT 'Group name',
    group_role ENUM('prefill', 'decode') NOT NULL COMMENT 'Group role',
    cluster_name VARCHAR(255) NOT NULL COMMENT 'Parent cluster',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    is_ready BOOLEAN DEFAULT FALSE COMMENT 'Whether group is ready',
    world_size INT COMMENT 'Total process count',
    group_data JSON COMMENT 'Serialized group data',
    
    INDEX idx_cluster_group (cluster_name, group_name, group_role),
    INDEX idx_ready (is_ready),
    INDEX idx_role (group_role),
    
    FOREIGN KEY (cluster_name) REFERENCES clusters(cluster_name) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Stores group information (prefill/decode groups)';

-- Create nodes table
CREATE TABLE IF NOT EXISTS nodes (
    node_id VARCHAR(255) PRIMARY KEY COMMENT 'Unique node identifier',
    hostname VARCHAR(255) NOT NULL COMMENT 'Node hostname',
    inference_addr VARCHAR(255) NOT NULL COMMENT 'Inference address',
    coordinator_port INT COMMENT 'Coordinator port',
    cluster_name VARCHAR(255) NOT NULL COMMENT 'Parent cluster',
    group_name VARCHAR(255) NOT NULL COMMENT 'Parent group',
    group_role ENUM('prefill', 'decode') NOT NULL COMMENT 'Group role',
    node_rank INT NOT NULL COMMENT 'Node rank within group',
    world_size INT COMMENT 'Total process count',
    devices JSON COMMENT 'Device information',
    last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Last heartbeat time',
    is_online BOOLEAN DEFAULT TRUE COMMENT 'Online status',
    comm_id VARCHAR(255) COMMENT 'Communication ID',
    job_id VARCHAR(255) COMMENT 'Job identifier',
    start_time VARCHAR(255) COMMENT 'Start time',
    node_data JSON COMMENT 'Serialized node data',
    
    INDEX idx_cluster_group_rank (cluster_name, group_name, group_role, node_rank),
    INDEX idx_heartbeat (last_heartbeat),
    INDEX idx_online (is_online),
    INDEX idx_hostname (hostname),
    UNIQUE INDEX idx_unique_rank_per_group (cluster_name, group_name, group_role, node_rank),
    
    FOREIGN KEY (cluster_name) REFERENCES clusters(cluster_name) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Stores individual node information';

-- Create communication groups table
CREATE TABLE IF NOT EXISTS comm_groups (
    comm_key VARCHAR(255) PRIMARY KEY COMMENT 'Communication key (prefill_group__decode_group)',
    prefill_group VARCHAR(255) NOT NULL COMMENT 'Prefill group name',
    decode_group VARCHAR(255) NOT NULL COMMENT 'Decode group name',
    comm_id VARCHAR(255) NOT NULL COMMENT 'Communication ID (UUID)',
    cluster_name VARCHAR(255) NOT NULL COMMENT 'Parent cluster',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last activity timestamp',
    comm_data JSON COMMENT 'Serialized communication data',
    
    INDEX idx_cluster (cluster_name),
    INDEX idx_prefill_group (prefill_group),
    INDEX idx_decode_group (decode_group),
    INDEX idx_last_active (last_active),
    INDEX idx_comm_id (comm_id),
    
    FOREIGN KEY (cluster_name) REFERENCES clusters(cluster_name) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Stores communication group pairs';

-- Create node mapping table for fast lookups
CREATE TABLE IF NOT EXISTS node_map (
    node_id VARCHAR(255) PRIMARY KEY COMMENT 'Node identifier',
    cluster_name VARCHAR(255) NOT NULL COMMENT 'Cluster name',
    group_name VARCHAR(255) NOT NULL COMMENT 'Group name',
    group_role ENUM('prefill', 'decode') NOT NULL COMMENT 'Group role',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    
    INDEX idx_cluster_group (cluster_name, group_name, group_role),
    INDEX idx_role (group_role),
    
    FOREIGN KEY (node_id) REFERENCES nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY (cluster_name) REFERENCES clusters(cluster_name) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Stores node to group mapping for fast lookups';

-- Create views for common queries (optional)

-- View: Active clusters with group counts
CREATE OR REPLACE VIEW v_cluster_summary AS
SELECT 
    c.cluster_name,
    c.is_active,
    c.created_at,
    c.last_updated,
    COUNT(DISTINCT CASE WHEN g.group_role = 'prefill' THEN g.group_id END) as prefill_groups_count,
    COUNT(DISTINCT CASE WHEN g.group_role = 'decode' THEN g.group_id END) as decode_groups_count,
    COUNT(DISTINCT n.node_id) as total_nodes,
    COUNT(DISTINCT CASE WHEN n.is_online = TRUE THEN n.node_id END) as online_nodes
FROM clusters c
LEFT JOIN `groups` g ON c.cluster_name = g.cluster_name
LEFT JOIN nodes n ON c.cluster_name = n.cluster_name
GROUP BY c.cluster_name, c.is_active, c.created_at, c.last_updated;

-- View: Node status overview
CREATE OR REPLACE VIEW v_node_status AS
SELECT 
    n.node_id,
    n.hostname,
    n.inference_addr,
    n.cluster_name,
    n.group_name,
    n.group_role,
    n.node_rank,
    n.is_online,
    n.last_heartbeat,
    g.is_ready as group_ready,
    g.world_size as group_world_size,
    TIMESTAMPDIFF(SECOND, n.last_heartbeat, NOW()) as seconds_since_heartbeat
FROM nodes n
JOIN `groups` g ON n.cluster_name = g.cluster_name 
    AND n.group_name = g.group_name 
    AND n.group_role = g.group_role;

-- View: Communication groups status
CREATE OR REPLACE VIEW v_comm_groups_status AS
SELECT 
    cg.comm_key,
    cg.prefill_group,
    cg.decode_group,
    cg.comm_id,
    cg.cluster_name,
    cg.created_at,
    cg.last_active,
    TIMESTAMPDIFF(SECOND, cg.last_active, NOW()) as seconds_since_active,
    pg.is_ready as prefill_ready,
    dg.is_ready as decode_ready,
    (pg.is_ready AND dg.is_ready) as both_groups_ready
FROM comm_groups cg
LEFT JOIN `groups` pg ON cg.cluster_name = pg.cluster_name 
    AND cg.prefill_group = pg.group_name 
    AND pg.group_role = 'prefill'
LEFT JOIN `groups` dg ON cg.cluster_name = dg.cluster_name 
    AND cg.decode_group = dg.group_name 
    AND dg.group_role = 'decode';


-- Show table structure
SHOW TABLES;

-- Show table sizes
SELECT 
    TABLE_NAME as 'Table',
    TABLE_ROWS as 'Rows',
    ROUND(((DATA_LENGTH + INDEX_LENGTH) / 1024 / 1024), 2) as 'Size (MB)'
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'ksana_llm_router'
ORDER BY (DATA_LENGTH + INDEX_LENGTH) DESC;
