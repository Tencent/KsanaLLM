# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""Configuration settings for the KsanaLLM Router service.

This module defines the configuration settings for the KsanaLLM Router service,
including heartbeat timeouts, log levels, API endpoints, and storage options.
"""


import configparser
import os


class Settings:
    """Service Configuration for KsanaLLM Router (from config.ini)."""
    def __init__(self, ini_path=None):
        if ini_path is None:
            ini_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        config = configparser.ConfigParser()
        config.read(ini_path, encoding='utf-8')

        g = config["general"]
        m = config["mysql"]

        self.node_heartbeat_timeout = int(g.get("node_heartbeat_timeout", 30))
        self.cleanup_interval = int(g.get("cleanup_interval", 60))
        self.log_level = g.get("log_level", "INFO")
        self.api_prefix = g.get("api_prefix", "/api/v1")
        self.cluster_name = g.get("cluster_name", "default-cluster")
        self.storage_mode = g.get("storage_mode", "memory")
        self.router_rule = g.get("router_rule", "auto")

        self.mysql_host = m.get("host", "localhost")
        self.mysql_port = int(m.get("port", 3306))
        self.mysql_user = m.get("user", "root")
        self.mysql_password = m.get("password", "")
        self.mysql_database = m.get("database", "ksana_llm_router")
        self.mysql_charset = m.get("charset", "utf8mb4")
        self.mysql_autocommit = m.getboolean("autocommit", True)

settings = Settings()
