# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import time

import pytest
from utils import modify_yaml_field

# Adjust the system path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ksana_llm.arg_utils import EngineArgs

import ksana_llm  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_test(model_dir, enable_prefix_cache, default_ksana_yaml_path):
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

        # Modify YAML configuration
        yaml_modifications = {
            "setting.global.tensor_para_size": 1,
            "setting.batch_scheduler.max_token_len": 2048,
            "setting.block_manager.block_host_memory_factor": 0.1,
            "setting.block_manager.reserved_device_memory_ratio": 0.3,
            "setting.batch_scheduler.max_batch_size": 1,
            "setting.batch_scheduler.enable_auto_prefix_cache": enable_prefix_cache,
            "setting.batch_scheduler.min_flexible_cache_num": 0,
            "setting.batch_scheduler.split_fuse_token_num": 0,
            "model_spec.base_model.model_dir": model_dir,
            "setting.global.is_version_report": False,
            "setting.profiler.stat_interval_second": 0,
        }

        for field_path, value in yaml_modifications.items():
            modify_yaml_field(ksana_yaml_path, field_path, value)

        # Initialize the model
        engine_args = EngineArgs.from_config_file(ksana_yaml_path)
        model = ksana_llm.KsanaLLMEngine.from_engine_args(engine_args)
        model.initialize()

        tokenizer = model.tokenizer

        logger.debug("Initialized ksana_llm model.")

        async def generate_for_prompt(prompt, sampling_config, structured_output_regex):
            formatted_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
            ).replace("%s", prompt)
            input_tokens = tokenizer.encode(formatted_prompt)
            _, output = await model.generate(
                model_name="",  # Specify the model name if needed
                request_dict={
                    "input_tokens": input_tokens,
                    "sampling_config": sampling_config,
                    "structured_output_regex": structured_output_regex
                },
                streamer=None,
            )
            output = tokenizer.decode(output.output_tokens[0], skip_special_tokens=True)
            return output

        async def test_case(prompt, structured_output_regex):
            # Req 0: The LLM large model generates with the prompt for the first time.
            start_time = time.time()
            req_0_output = await generate_for_prompt(prompt, sampling_config, "")
            end_time = time.time()
            req_0_time = end_time - start_time
            logger.debug(f"The Request 0(Base): cost time = {req_0_time}, "
                          f"output = \n{req_0_output}")

            # Req 1: The LLM large model can be optimized using Prefix Cache.
            start_time = time.time()
            req_1_output = await generate_for_prompt(prompt, sampling_config, "")
            end_time = time.time()
            req_1_time = end_time - start_time
            logger.debug(f"The Request 1(Prefix Cache): cost time = {req_1_time}, "
                          f"output = \n{req_1_output}")

            # Req 2: The LLM large model can be optimized using Structured Output
            start_time = time.time()
            req_2_output = await generate_for_prompt(prompt, sampling_config,
                                               structured_output_regex)
            end_time = time.time()
            req_2_time = end_time - start_time
            logger.debug(f"The Request 2(Structured Output): cost time = {req_2_time}, "
                          f"output = \n{req_2_output}")

            assert (
                req_2_time / req_1_time < 0.68
            ), (
                f"Structured execution time {req_2_time} "
                f"exceeds 68% of the base execution time {req_1_time}"
            )
            assert (req_0_output == req_1_output
            ), (
                f"Prefix Caching result {req_1_output} "
                f"does not the same with the first execution result {req_0_output}"
            )
            # Different compilation options may lead to variations in the results at
            # this point. You can adjust the test cases accordingly.
            assert (req_0_output == req_2_output
            ), (
                f"Structured output execution result {req_2_output} "
                f"does not the same with the first execution result {req_0_output}"
            )
        sampling_config = {}
        sampling_config["max_new_tokens"] = 128
        text1 = "赫敏格兰杰是一名哈利波特系列小说中的角色。请将她的个人信息填到" \
                "下面的 json 表格中{'姓名': xxx, '年龄': xxx, '职业': xxx, '爱好'" \
                ": xxx, '学院': xxx}"
        regex1 = """```json
{
  "姓名": "[*]",
  "年龄": [*],
  "职业": "[*]",
  "爱好": "[*]",
  "学院": "[*]"
}
```"""
        await test_case(text1, regex1)

        text2 = "请挑选几个哈利波特系列小说中的角色,但不超过3个,将他们的个人信息填到" \
                "下面的 json 表格中 [{'角色1': '角色1的名字'}, {'角色2': '角色2的名字'}, ...]"
        regex2 = """```json
{
  "角色[*]": "[*]"(?:,
  "角色[*]": "[*]")*
}
```"""
        await test_case(text2, regex2)
    finally:
        # Clean up the temporary directory
        del model
        shutil.rmtree(temp_dir)
        logger.debug(f"Deleted temporary directory: {temp_dir}")


@pytest.mark.parametrize("model_dir", ["/model/qwen1.5-hf/0.5B-Chat"])
@pytest.mark.parametrize("enable_prefix_cache", [True])
def test_structured_output(model_dir, enable_prefix_cache, default_ksana_yaml_path):
    logger.debug(f"Running test_structured_output with model_dir: {model_dir}")
    asyncio.run(run_test(model_dir, enable_prefix_cache, default_ksana_yaml_path))
