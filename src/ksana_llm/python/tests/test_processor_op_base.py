# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys
import tempfile
import shutil
import logging
import pytest


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


def run_test(model_dir, default_ksana_yaml_path):
    """
    Execute the model test within a temporary directory.

    Args:
        model_dir (str): Directory of the model.
        default_ksana_yaml_path (str): Path to the default ksana YAML config.
    """
    temp_dir = tempfile.mkdtemp()
    logger.debug(f"Created temporary directory: {temp_dir}")

    try:
        print(f'start to run processor op base test')
        # Copy the default YAML to the temporary directory
        ksana_yaml_path = os.path.join(temp_dir, "ksana.yaml")
        shutil.copyfile(default_ksana_yaml_path, ksana_yaml_path)
        assert os.path.exists(ksana_yaml_path), "Failed to copy ksana.yaml"

        engine_args = EngineArgs.from_config_file(ksana_yaml_path)
        pre_post_processor = ksana_llm.TokenizerProcessorOpBase(
            engine_args.model_dir,
            engine_args.tokenizer_path,
            engine_args.tokenization
            )
        
        # Test encode
        test_prompt = "hello world"        
        encode_rst = pre_post_processor.encode(test_prompt)
        encode_base = [14990, 1879]
        print(f'encode_rst is: {encode_rst} and encode_base is: {encode_base}')
        assert (
            encode_rst == encode_base
        )
        
        # Test get stop tokens ids
        stop_token_ids = pre_post_processor.get_stop_token_ids()
        stop_token_ids_base = [151645, 151643]
        print(f'stop_token_ids: {stop_token_ids} and stop_token_ids_base: {stop_token_ids_base}')
        assert (
            stop_token_ids == stop_token_ids_base
        )
        
        # Test decode token ids
        test_input_tokens = [14990, 1879]
        is_stream_generate = True
        decoded_str = pre_post_processor.decode(test_input_tokens, is_stream_generate)
        decoded_str_base = "hello world"
        print(f'decoded_str: {decoded_str} and decoded_str_base is: {decoded_str_base}')
        assert (
            decoded_str == decoded_str_base
        )
        
        # Test support model type
        fake_model_type = 'test'
        verified_model_type = pre_post_processor.check_is_supported_model_type(fake_model_type)
        assert (
            verified_model_type == "empty"
        )
        real_model_type = 'qwen'
        verified_real_model_type = pre_post_processor.check_is_supported_model_type(real_model_type)
        assert (
            verified_real_model_type == "qwen"
        )
        print(f'end to run processor op base test')
        
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        logger.debug(f"Delete temporary directory: {temp_dir}")


@pytest.mark.parametrize("model_dir", ["/model/qwen1.5-hf/0.5B-Chat"])
def test_processor_op_base(model_dir, default_ksana_yaml_path):
    run_test(model_dir, default_ksana_yaml_path)
