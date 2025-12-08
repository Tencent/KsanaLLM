# tests/conftest.py

import os
import sys
import atexit
from fnmatch import fnmatch
import pytest
from utils import read_from_csv


@pytest.fixture(scope="session")
def default_ksana_yaml_path():
    """
    Fixture to provide the default ksana_yaml file path.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "../../../../examples/ksana_llm.yaml")


@pytest.fixture(scope="session")
def benchmark_inputs():
    """
    Fixture to provide benchmark inputs from CSV.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        current_dir, "../../../../benchmarks/benchmark_input.csv"
    )
    return read_from_csv(csv_path)


def pytest_configure(config):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)


def pytest_ignore_collect(path):
    """
    只要测试文件 basename 匹配任意一个 --ignore-glob 模式就跳过。
    支持 *、?、[] 通配符，可写多次。
    """
    # 1. 取用户传来的模式列表
    patterns = getattr(pytest_ignore_collect, "_patterns", [])
    if not patterns:
        patterns = []
        for arg in sys.argv:
            if arg.startswith("--ignore-glob="):
                patterns.append(arg.split("=", 1)[1])
        pytest_ignore_collect._patterns = patterns

    # 2. 用 basename 做模糊匹配
    basename = os.path.basename(str(path))
    return any(fnmatch(basename, pat) for pat in patterns)


def on_exit():
    os._exit(0)


atexit.register(on_exit)
