# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from __future__ import annotations

import configparser
import os
from typing import Optional


class PrefillDecodeSettings:
    """Settings backed by `config.ini` under the package directory."""

    def __init__(self, ini_path: Optional[str] = None):
        # Resolve configuration path precedence:
        # Default: config.ini alongside this file
        if ini_path is None:
            ini_path = os.path.join(os.path.dirname(__file__), "config.ini")
        ini_path = os.path.abspath(os.path.expanduser(ini_path))
        config_dir = os.path.dirname(ini_path)

        config = configparser.ConfigParser()
        read_ok = config.read(ini_path, encoding="utf-8")
        if not read_ok:
            raise FileNotFoundError(f"Failed to load configuration file: {ini_path}")

        db_section = config["database"]
        general_section = config["general"] if config.has_section("general") else {}
        name_service_section = (
            config["name_service"] if config.has_section("name_service") else {}
        )

        self.log_level = general_section.get("log_level", "INFO").upper()
        self.log_dir = general_section.get("log_dir", "./")
        self.cluster_name = general_section.get("cluster_name", "default_cluster")
        self.heartbeat_timeout_seconds = general_section.getint(
            "heartbeat_timeout_seconds", 60
        )
        self.storage_mode = db_section.get("storage_mode", "sqlite").lower()
        if self.storage_mode == "sqlite":
            sqlite_path_raw = db_section.get("sqlite_path", "./simple_router.db")
            sqlite_path_expanded = os.path.expanduser(sqlite_path_raw)
            if sqlite_path_expanded == ":memory:":
                raise ValueError("In-memory SQLite database is not supported.")
            if not os.path.isabs(sqlite_path_expanded):
                sqlite_path_expanded = os.path.normpath(
                    os.path.join(config_dir, sqlite_path_expanded)
                )
            self.sqlite_path = sqlite_path_expanded

            # MySQL settings not used in SQLite mode
            self.mysql_host = None
            self.mysql_port = None
            self.mysql_user = None
            self.mysql_password = None
            self.mysql_database = None
            self.mysql_charset = None
            self.mysql_autocommit = None
        elif self.storage_mode == "mysql":
            self.mysql_host = db_section.get("host", "localhost")
            self.mysql_port = db_section.getint("port", 3306)
            self.mysql_user = db_section.get("user", "root")
            self.mysql_password = db_section.get("password", "")
            self.mysql_database = db_section.get("database", "ksana_llm_router")
            self.mysql_charset = db_section.get("charset", "utf8mb4")
            self.mysql_autocommit = db_section.getboolean("autocommit", True)

            # SQLite settings not used in MySQL mode
            self.sqlite_path = None
        else:
            raise ValueError(
                f"Invalid storage_mode: {self.storage_mode}. "
                "Must be either 'sqlite' or 'mysql'."
            )

        default_provider = "simple_router.name_service.auto_provider"
        self.name_service_provider = name_service_section.get(
            "name_service_provider", default_provider
        )
        self.namespace = name_service_section.get("namespace", "Production")
        self.prefill_service = name_service_section.get(
            "prefill_service", "prefill-service"
        )
        self.decode_service = name_service_section.get(
            "decode_service", "decode-service"
        )


SETTINGS: Optional[PrefillDecodeSettings] = None


def get_settings() -> PrefillDecodeSettings:
    """Get or initialize the global settings instance."""
    global SETTINGS
    if SETTINGS is None:
        SETTINGS = PrefillDecodeSettings()
    return SETTINGS


def reload_settings(ini_path: Optional[str] = None) -> PrefillDecodeSettings:
    """Reload global settings from a given path."""
    global SETTINGS
    SETTINGS = PrefillDecodeSettings(ini_path)
    return SETTINGS
