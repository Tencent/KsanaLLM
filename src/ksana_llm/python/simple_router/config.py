# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""Configuration settings for the KsanaLLM Router service.

This module defines the configuration settings for the KsanaLLM Router service,
including heartbeat timeouts, log levels, API endpoints, and storage options.
"""

from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service Configuration for KsanaLLM Router.

    Attributes:
        node_heartbeat_timeout: Time in seconds after which a node is considered offline.
        cleanup_interval: Time in seconds between cleanup tasks.
        log_level: Logging level for the application.
        api_prefix: Prefix for all API endpoints.
        cluster_name: Default cluster name to use.
        storage_mode: Storage backend mode ('memory' or 'etcd').
        etcd_host: Hostname for etcd connection.
        etcd_port: Port for etcd connection.
        etcd_prefix: Key prefix for etcd storage.
        etcd_timeout: Connection timeout in seconds for etcd.
        etcd_user: Username for etcd authentication.
        etcd_password: Password for etcd authentication.
        router_rule: The rule to use when routing requests ('fixed' or 'auto').
    """

    # Heartbeat timeout (seconds)
    node_heartbeat_timeout: int = 30

    # Cleanup task interval (seconds)
    cleanup_interval: int = 60

    # Log level
    log_level: str = "INFO"

    # API prefix
    api_prefix: str = "/api/v1"

    # Default cluster name
    cluster_name: str = "default-cluster"

    # Storage mode: memory or etcd
    storage_mode: Literal["memory", "etcd"] = "memory"

    # ETCD configuration
    etcd_host: str = "localhost"
    etcd_port: int = 2379
    etcd_prefix: str = "/ksana_llm/router/"
    etcd_timeout: int = 5  # Connection timeout (seconds)
    etcd_user: str = ""  # If etcd requires authentication
    etcd_password: str = ""  # If etcd requires authentication

    # Router rule: fixed or auto
    router_rule: Literal["fixed", "auto"] = "auto"

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False


settings = Settings()
