#!/usr/bin/env python3
# Copyright 2025 Tencent Inc. All rights reserved.
# 
# 加载YAML配置文件的示例代码

import os
import sys
import yaml
from typing import Dict, Any, Optional


def load_yaml_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    加载YAML配置文件
    
    Args:
        config_path: YAML配置文件路径
        
    Returns:
        配置字典，如果加载失败则返回None
    """
    if not os.path.exists(config_path):
        print(f"错误：配置文件 '{config_path}' 不存在", file=sys.stderr)
        return None
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config
    except yaml.YAMLError as e:
        print(f"错误：解析YAML文件时出错: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"错误：加载配置文件时出错: {e}", file=sys.stderr)
        return None


def update_yaml_config(config_path: str, new_config: Dict[str, Any]) -> bool:
    """
    更新YAML配置文件
    
    Args:
        config_path: YAML配置文件路径
        new_config: 新的配置字典
        
    Returns:
        更新是否成功
    """
    try:
        # 如果文件存在，先读取原有配置
        existing_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = yaml.safe_load(f) or {}
        
        # 合并配置
        merged_config = {**existing_config, **new_config}
        
        # 写入文件
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(merged_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        print(f"错误：更新配置文件时出错: {e}", file=sys.stderr)
        return False


def main():
    """示例用法"""
    # 示例配置文件路径
    config_path = "gemm_algo_map.yaml"
    
    # 加载配置
    config = load_yaml_config(config_path)
    if config:
        print("成功加载配置:")
        print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        
        # 示例：更新配置
        new_config = {
            "deep_gemm_kernel": {
                "fp8_fp8_bf16": {
                    "m": 128,
                    "n": 128,
                    "k": 1024,
                    "block_m": 64,
                    "block_n": 64,
                    "num_stages": 4,
                    "num_tma_multicast": 2,
                    "performance": "10.5ms"
                }
            }
        }
        
        if update_yaml_config(config_path + ".new", new_config):
            print(f"配置已更新并保存到 {config_path}.new")
    else:
        print("加载配置失败")


if __name__ == "__main__":
    main()