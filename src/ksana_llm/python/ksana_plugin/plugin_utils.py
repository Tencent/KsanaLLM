# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import os
import torch


def get_weight_map(model_path, filter_value=None):
    import json
    from glob import glob
    # read weight map
    weight_map_json = glob(os.path.join(model_path, "*index.json"))
    if len(weight_map_json) == 1:
        with open(weight_map_json[0]) as file:
            weight_map_files = json.load(file)
        weight_map_files = weight_map_files["weight_map"]
        if filter_value is not None:
            filtered_values = {value for key, value in weight_map_files.items() if filter_value in key}
            weight_map_files = list(filtered_values)
    else:
        weight_map_files = glob(os.path.join(model_path, "*.safetensors"))
    return weight_map_files


def free_cache():
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def load_safetensors(file_path: str):
    from safetensors.torch import load
    with open(file_path, "rb") as f:
        data = f.read()
    loaded = load(data)
    return loaded


def check_file_dir(file_path: str):
    file_dir = os.path.dirname(file_path)
    if file_dir != '' and not os.path.exists(file_dir):
        os.makedirs(file_dir)


def get_module(module_name, py_path, class_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def adjust_device_memory_ratio(config_file: str, reserved_device_memory_ratio: float):
    """Adjust the memory ratio for multi-modal models
    """
    import re
    with open(config_file, "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml_file.read()
    # Overwrite the yaml config file
    # Use regular expressions to preserve the original order and comments
    match = re.search(r'(\s*reserved_device_memory_ratio:)(\s*\d+\.\d+|\d+)(.*)',
                      yaml_data)
    # Only adjust if the current configuration value is too low
    if match and float(match.group(2)) < reserved_device_memory_ratio:
        yaml_data = re.sub(r'(\s*reserved_device_memory_ratio:)(\s*\d+\.\d+|\d+)(.*)',
                           f'{match.group(1)} {reserved_device_memory_ratio}{match.group(3)}',
                           yaml_data)
    with open(config_file, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(yaml_data)


def build_trt_process(plugin_dir, model_path):
    from filelock import FileLock, Timeout
    import subprocess

    print(f"[I] Start converting Model!")

    # If there are multiple devices, only one process is needed to convert
    lock_file = "build_model.lock"
    try:
        with FileLock(lock_file, timeout=1):
            print("[W] Lock acquired, running ksana_plugin_model.py")
            script = os.path.join(plugin_dir, "ksana_plugin_model.py")
            py_command = f"python {script} {model_path}"

            # Out of memory: can be converted locally, serving load
            result = subprocess.run(py_command, shell=True, stdout=subprocess.PIPE)
            log = result.stdout.decode('utf-8')
            print(log)

            if result.returncode != 0:
                raise Exception(f"[E] ksana_plugin_model.py failed with error: {result.stderr}")

            print("[E] ksana_plugin_model.py finished successfully")
    except Timeout:
        print("Another instance of ksana_plugin_model.py is running, waiting...")
        with FileLock(lock_file):
            print("Lock acquired after waiting, continuing execution")


def build_trt(model, model_path):
    from trt_engine import Engine

    onnx_path = model.get_onnx_path(model_path)
    trt_path = model.get_trt_path(model_path)
    trt_engine = Engine(trt_path)

    if not os.path.exists(onnx_path):
        # Start converting ONNX!
        precision = torch.float16
        vit = model.get_model(model_path, precision).eval()

        sample_input = model.get_sample_input()
        input_list = model.get_input_names()
        output_list = model.get_output_names()
        dynamic_list = model.get_dynamic_axes()

        trt_engine.export_onnx(vit,
                                sample_input, onnx_path,
                                input_list, output_list, dynamic_list)

        del vit
        free_cache()

    # Start converting TRT engine
    input_profile = model.get_trt_profile()
    trt_engine.build_trt(onnx_path, enable_fp16=True, enable_refit=False, input_profile=input_profile)
