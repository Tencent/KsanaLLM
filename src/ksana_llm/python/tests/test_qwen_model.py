# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import pytest
from test_ksana_engine import run_test

# Test Parameters
test_models = [
    ("/model/qwen1.5-hf/0.5B-Chat", "qwen", 2.0),
]

test_nums = [1, 10]

test_configs = [(1, 0, 0.3, 4096, False), (1, 0, 0.3, 4096, True)]

# Parametrized Test


@pytest.mark.parametrize("model_dir, prompt_affix_type, max_ppl", test_models)
@pytest.mark.parametrize("test_num", test_nums)
@pytest.mark.parametrize(
    "tensor_para_size, block_host_memory_factor, reserved_device_memory_ratio,"
    " max_batch_size, enable_auto_prefix_cache",
    test_configs,
)
def test_qwen(
    model_dir,
    prompt_affix_type,
    max_ppl,
    test_num,
    tensor_para_size,
    block_host_memory_factor,
    reserved_device_memory_ratio,
    max_batch_size,
    enable_auto_prefix_cache,
    default_ksana_yaml_path,
    benchmark_inputs,
):
    """
    Parametrized test for the Qwen model.

    This test modifies the YAML configuration, runs the model generation,
    computes perplexity, and asserts that the average perplexity is below a
    threshold.

    Args:
        model_dir (str): Directory of the model.
        prompt_affix_type (str): Affix to prepend to prompts.
        max_ppl (float): Maximum permitted perplexity.
        test_num (int): Number of tests to run.
        tensor_para_size (int): Tensor parallel size.
        block_host_memory_factor (float): Block host memory factor.
        reserved_device_memory_ratio (float): Reserved device memory ratio.
        max_batch_size (int): Maximum batch size.
        enable_auto_prefix_cache (bool): Enable auto prefix cache.
        default_ksana_yaml_path (str): Path to the default ksana YAML config.
        benchmark_inputs (List[str]): List of input prompts (fixture).
    """
    run_test(
        model_dir,
        prompt_affix_type,
        max_ppl,
        test_num,
        tensor_para_size,
        block_host_memory_factor,
        reserved_device_memory_ratio,
        max_batch_size,
        enable_auto_prefix_cache,
        default_ksana_yaml_path,
        benchmark_inputs,
    )
