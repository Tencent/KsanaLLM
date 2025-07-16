# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# [vLLM Project]
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/logger.py
import datetime
import json
import logging
import os
import sys
from collections.abc import Hashable
from functools import lru_cache, partial
from logging import Logger
from logging.config import dictConfig
from os import path
from types import MethodType
from typing import Any, Optional, cast


# Configuration variables
KSANA_CONFIGURE_LOGGING = True
KSANA_LOGGING_CONFIG_PATH = ""
KSANA_LOGGING_LEVEL =  "INFO"
KSANA_LOGGING_PREFIX = "[KSANA] "
KSANA_LOG_FILE = "ksana.log"

_FORMAT = (f"{KSANA_LOGGING_PREFIX}%(levelname)s %(asctime)s "
           "[%(filename)s:%(lineno)d] %(message)s")
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Formatter that ensures each log record starts on a new line."""
    
    def format(self, record):
        msg = super().format(record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\n" + parts[0])
        return msg


DEFAULT_LOGGING_CONFIG = {
    "formatters": {
        "ksana": {
            "class": "logging.Formatter",
            "datefmt": _DATE_FORMAT,
            "format": _FORMAT,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "ksana",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "ksana",
            "level": "DEBUG",
            "filename": KSANA_LOG_FILE,
            "mode": "a",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "ksana": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "multimodal": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "openaiapi": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "version": 1,
    "disable_existing_loggers": False
}


@lru_cache
def _print_info_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """Print info message only once."""
    # Set the stacklevel to 2 to print the original caller's line info
    logger.info(msg, *args, stacklevel=2)


@lru_cache
def _print_warning_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """Print warning message only once."""
    # Set the stacklevel to 2 to print the original caller's line info
    logger.warning(msg, *args, stacklevel=2)


@lru_cache
def _print_error_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """Print error message only once."""
    # Set the stacklevel to 2 to print the original caller's line info
    logger.error(msg, *args, stacklevel=2)


class _KsanaLogger(Logger):
    """
    Enhanced logger with additional methods.
    
    Note:
        This class is just to provide type information.
        We actually patch the methods directly on the [`logging.Logger`][]
        instance to avoid conflicting with other libraries.
    """

    def info_once(self, msg: str, *args: Hashable) -> None:
        """
        As [`info`][logging.Logger.info], but subsequent calls with
        the same message are silently dropped.
        """
        _print_info_once(self, msg, *args)

    def warning_once(self, msg: str, *args: Hashable) -> None:
        """
        As [`warning`][logging.Logger.warning], but subsequent calls with
        the same message are silently dropped.
        """
        _print_warning_once(self, msg, *args)
    
    def error_once(self, msg: str, *args: Hashable) -> None:
        """
        As [`error`][logging.Logger.error], but subsequent calls with
        the same message are silently dropped.
        """
        _print_error_once(self, msg, *args)


def _configure_ksana_root_logger() -> None:
    """Configure the root logger for KsanaLLM."""
    logging_config = dict[str, Any]()

    if not KSANA_CONFIGURE_LOGGING and KSANA_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "KSANA_CONFIGURE_LOGGING evaluated to false, but "
            "KSANA_LOGGING_CONFIG_PATH was given. KSANA_LOGGING_CONFIG_PATH "
            "implies KSANA_CONFIGURE_LOGGING. Please enable "
            "KSANA_CONFIGURE_LOGGING or unset KSANA_LOGGING_CONFIG_PATH.")

    if KSANA_CONFIGURE_LOGGING:
        logging_config = DEFAULT_LOGGING_CONFIG

    if KSANA_LOGGING_CONFIG_PATH:
        if not path.exists(KSANA_LOGGING_CONFIG_PATH):
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s",
                KSANA_LOGGING_CONFIG_PATH)
        with open(KSANA_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            raise ValueError("Invalid logging config. Expected dict, got %s.",
                             type(custom_config).__name__)
        logging_config = custom_config

    if logging_config:
        dictConfig(logging_config)


def init_logger(name: str) -> _KsanaLogger:
    """
    Initialize a logger with enhanced functionality.
    
    The main purpose of this function is to ensure that loggers are
    retrieved in such a way that we can be sure the root ksana logger has
    already been configured.
    
    Args:
        name: The name of the logger (typically __name__)
        
    Returns:
        An enhanced logger instance
    """
    logger = logging.getLogger(name)

    # Patch additional methods onto the logger instance
    methods_to_patch = {
        "info_once": _print_info_once,
        "warning_once": _print_warning_once,
        "error_once": _print_error_once,
    }

    for method_name, method in methods_to_patch.items():
        setattr(logger, method_name, MethodType(method, logger))

    return cast(_KsanaLogger, logger)


def _trace_calls(log_path, root_dir, frame, event, arg=None):
    """Trace function calls for debugging."""
    if event in ['call', 'return']:
        # Extract the filename, line number, function name, and the code object
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func_name = frame.f_code.co_name
        if not filename.startswith(root_dir):
            # only log the functions in the specified root_dir
            return
        # Log every function call or return
        try:
            last_frame = frame.f_back
            if last_frame is not None:
                last_filename = last_frame.f_code.co_filename
                last_lineno = last_frame.f_lineno
                last_func_name = last_frame.f_code.co_name
            else:
                # initial frame
                last_filename = ""
                last_lineno = 0
                last_func_name = ""
            with open(log_path, 'a') as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                if event == 'call':
                    f.write(f"{ts} Call to"
                            f" {func_name} in {filename}:{lineno}"
                            f" from {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
                else:
                    f.write(f"{ts} Return from"
                            f" {func_name} in {filename}:{lineno}"
                            f" to {last_func_name} in {last_filename}:"
                            f"{last_lineno}\n")
        except NameError:
            # modules are deleted during shutdown
            pass
    return partial(_trace_calls, log_path, root_dir)


def enable_trace_function_call(log_file_path: str,
                               root_dir: Optional[str] = None):
    """
    Enable tracing of every function call in code under `root_dir`.
    This is useful for debugging hangs or crashes.
    
    Args:
        log_file_path: The path to the log file.
        root_dir: The root directory of the code to trace. If None, it is the
                  ksana_llm root directory.

    Note:
        This call is thread-level, any threads calling this function
        will have the trace enabled. Other threads will not be affected.
    """
    logger.warning(
        "KSANA_TRACE_FUNCTION is enabled. It will record every"
        " function executed by Python. This will slow down the code. It "
        "is suggested to be used for debugging hang or crashes only.")
    logger.info("Trace frame log is saved to %s", log_file_path)
    if root_dir is None:
        # by default, this is the ksana_llm root directory
        root_dir = os.path.dirname(os.path.dirname(__file__))
    sys.settrace(partial(_trace_calls, log_file_path, root_dir))


# The root logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_configure_ksana_root_logger()

# Create a logger for this module
logger = init_logger(__name__)


# 便捷函数，用于快速获取logger
def get_logger(name: str) -> _KsanaLogger:
    """
    Get a logger instance.
    
    This is a convenience function that wraps init_logger.
    
    Args:
        name: The name of the logger (typically __name__)
        
    Returns:
        An enhanced logger instance
    """
    return init_logger(name)
