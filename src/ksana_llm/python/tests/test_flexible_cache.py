# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import asyncio
import sys
import tempfile
import shutil
import logging
import time
import pytest
from transformers import AutoTokenizer
from utils import modify_yaml_field

# Adjust the system path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ksana_llm  # noqa: E402
from ksana_llm.arg_utils import EngineArgs

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_test(model_dir, default_ksana_yaml_path):
    """
    Execute the model test within a temporary directory.

    Args:
        model_dir (str): Directory of the model.
        default_ksana_yaml_path (str): Path to the default ksana YAML config.
    """
    temp_dir = tempfile.mkdtemp()
    logger.debug(f"Created temporary directory: {temp_dir}")

    try:
        # Copy the default YAML to the temporary directory
        ksana_yaml_path = os.path.join(temp_dir, "ksana.yaml")
        shutil.copyfile(default_ksana_yaml_path, ksana_yaml_path)
        assert os.path.exists(ksana_yaml_path), "Failed to copy ksana.yaml"

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
            use_fast=True,
        )

        # Modify YAML configuration
        yaml_modifications = {
            "setting.global.tensor_para_size": 1,
            "setting.batch_scheduler.max_token_len": 32768,
            "setting.block_manager.block_host_memory_factor": 0.0,
            "setting.block_manager.reserved_device_memory_ratio": 0.3,
            "setting.batch_scheduler.max_batch_size": 1,
            "setting.batch_scheduler.enable_auto_prefix_cache": True,
            "setting.batch_scheduler.min_flexible_cache_num": 16,
            "setting.batch_scheduler.split_fuse_token_num": 0,
            "model_spec.base_model.model_dir": model_dir,
            "setting.global.is_version_report": False,
            "setting.profiler.stat_interval_second": 0,
        }

        for field_path, value in yaml_modifications.items():
            modify_yaml_field(ksana_yaml_path, field_path, value)

        # Initialize the engine
        engine_args = EngineArgs.from_config_file(ksana_yaml_path)
        engine = ksana_llm.KsanaLLMEngine.from_engine_args(engine_args)
        engine.initialize()
        tokenizer = engine.tokenizer

        logger.debug("Initialized ksana_llm engine.")

        async def generate_for_prompt(prompt, request_dict):
            formatted_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
            ).replace("%s", prompt)
            input_tokens = tokenizer.encode(formatted_prompt)
            request_dict['input_tokens'] = input_tokens

            _, output = await engine.generate(
                model_name="",  # Specify the model name if needed
                request_dict=request_dict,
                streamer=None,
            )
            return input_tokens, output.output_tokens[0]

        request_dict = {}
        numbers_text = " ".join(str(i) for i in range(1, 5000))
        text1 = (
            "你是一个乐于助人的小助手，现在我将提问一个问题，请你认真的回答这个问题。"
            "这是一个数学相关的问题，请问仔细观察%s，这些数字中有偶数有奇数，请你仔细考虑"
            "偶数奇数的定义，仔细思考后告诉我其中100是奇数还是偶数？先给出结果再解释原因。"
        ) % numbers_text
        text2 = (
            "你是一个乐于助人的小助手，我将提问一个问题，请你认真的回答这个问题。"
            "这是一个数学相关的问题，请问仔细观察%s，这些数字中有偶数有奇数，请你仔细考虑"
            "偶数奇数的定义，仔细思考后告诉我其中100是奇数还是偶数？先给出结果再解释原因。"
        ) % numbers_text
        request_dict["max_new_tokens"] = 1
        await generate_for_prompt(text1, request_dict)
        start_time = time.time()
        output = await generate_for_prompt(text2, request_dict)
        end_time = time.time()
        flexible_cache_execution_time = end_time - start_time
        request_dict["max_new_tokens"] = 100
        output = await generate_for_prompt(text2, request_dict)
        flexible_cache_ans = tokenizer.decode(
            output[1], skip_special_tokens=True)

        modify_yaml_field(
            ksana_yaml_path,
            "setting.batch_scheduler.min_flexible_cache_num",
            0,
        )
        # Initialize the engine
        engine_args = EngineArgs.from_config_file(ksana_yaml_path)
        engine = ksana_llm.KsanaLLMEngine.from_engine_args(engine_args)
        engine.initialize()
        tokenizer = engine.tokenizer

        logger.debug("Initialized ksana_llm engine.")
        request_dict["max_new_tokens"] = 1
        await generate_for_prompt(text1, request_dict)
        start_time = time.time()
        output = await generate_for_prompt(text2, request_dict)
        end_time = time.time()
        base_execution_time = end_time - start_time
        request_dict["max_new_tokens"] = 100
        output = await generate_for_prompt(text2, request_dict)
        base_ans = tokenizer.decode(output[1], skip_special_tokens=True)
        print(f"flexible_cache_execution_time: {flexible_cache_execution_time}")
        print(f"base_execution_time          : {base_execution_time}")
        print(f"flexible_cache_ans: {flexible_cache_ans}")
        print(f"base_ans          : {base_ans}")
        assert (
            flexible_cache_execution_time / base_execution_time < 0.75
        ), (
            f"Flexible cache execution time ({flexible_cache_execution_time}) "
            f"exceeds 75% of the base execution time ({base_execution_time})."
        )

        assert (
            flexible_cache_ans == base_ans == "100是偶数。"
        ), (
            f"Answers do not match: flexible cache answer"
            f" is '{flexible_cache_ans}', expected answer is '{base_ans}'."
        )
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        logger.debug(f"Deleted temporary directory: {temp_dir}")


@pytest.mark.parametrize("model_dir", ["/model/qwen1.5-hf/0.5B-Chat"])
def test_flexible_cache(model_dir, default_ksana_yaml_path):
    asyncio.run(run_test(model_dir, default_ksana_yaml_path))
